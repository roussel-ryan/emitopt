import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union

import torch
from pydantic import Field

from scipy.optimize import minimize
from torch import Tensor
from xopt.generators.bayesian.bax.algorithms import Algorithm

from .sampling import (
    draw_linear_product_kernel_post_paths,
    draw_product_kernel_post_paths,
)
from botorch.models.model import Model, ModelList
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from gpytorch.kernels import ProductKernel, MaternKernel

from emitopt.analysis import compute_emit_bmag
from emitopt.algorithms import ScipyMinimizeEmittanceXY

def unif_random_sample_domain(n_samples, domain):
    ndim = len(domain)

    # uniform sample, rescaled, and shifted to cover the domain
    x_samples = torch.rand(n_samples, ndim) * torch.tensor(
        [bounds[1] - bounds[0] for bounds in domain]
    ) + torch.tensor([bounds[0] for bounds in domain])

    return x_samples


class GridMinimizeEmitBmag(ScipyMinimizeEmittanceXY):
    name = "GridMinimizeEmitBmag"
    x_key: str = Field(None,
        description="key designating the beamsize squared output in x from evaluate function")
    y_key: str = Field(None,
        description="key designating the beamsize squared output in y from evaluate function")
    scale_factor: float = Field(1.0,
        description="factor by which to multiply the quad inputs to get focusing strengths")
    q_len: float = Field(
        description="the longitudinal thickness of the measurement quadrupole"
    )
    rmat_x: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for x dimension"
    )
    rmat_y: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for y dimension"
    )
    twiss0_x: Tensor = Field(None,
        description="1d tensor length 2 containing design x-twiss: [beta0_x, alpha0_x] (for bmag)"
    )
    twiss0_y: Tensor = Field(None,
        description="1d tensor length 2 containing design y-twiss: [beta0_y, alpha0_y] (for bmag)"
    )
    meas_dim: int = Field(
        description="index identifying the measurement quad dimension in the model"
    )
    n_steps_measurement_param: int = Field(
        11, description="number of steps to use in the virtual measurement scans"
    )
    thick_quad: bool = Field(True,
        description="Whether to use thick-quad (or thin, if False) transport for emittance calc")
    n_grid_points: int = Field(10,
        description="Number of points in each grid dimension. Only used if method='Grid'.")

    @property
    def observable_names_ordered(self) -> list:  
        # get observable model names in the order they appear in the model (ModelList)
        return [key for key in [self.x_key, self.y_key] if key]

    def get_execution_paths(self, model: ModelList, bounds: Tensor, tkwargs=None, verbose=False):
        if not (self.x_key or self.y_key):
            raise ValueError("must provide a key for x, y, or both.")
        if (self.x_key and self.rmat_x is None) or (self.y_key and self.rmat_y is None):
            raise ValueError("must provide rmat for each transverse dimension (x/y) being modeled.")
    
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        temp_id = self.meas_dim + 1
        tuning_domain = torch.cat((bounds.T[: self.meas_dim], bounds.T[temp_id:]))

        tuning_bounds = tuning_domain.T
        assert isinstance(tuning_bounds, Tensor)
        # create mesh
        if len(tuning_bounds) != 2:
            raise ValueError("tuning_bounds must have the shape [2, ndim]")

        dim = len(tuning_bounds[0])
        # add in a machine eps
        eps = 1e-5
        linspace_list = [
            torch.linspace(
                tuning_bounds.T[i][0] + eps, tuning_bounds.T[i][1] - eps, self.n_grid_points, **tkwargs
            )
            for i in range(dim)
        ]

        xx = torch.meshgrid(*linspace_list, indexing="ij")
        mesh_pts = torch.stack(xx).flatten(start_dim=1).T
        # print(mesh_pts.shape)
        # evaluate the function on grid points
        f_values, emit, bmag, is_valid, validity_rate, bss = self.evaluate_objective(model, mesh_pts, bounds,
                                                            tkwargs=tkwargs, n_samples=self.n_samples)
        f_values = torch.nan_to_num(f_values, float('inf'))
        best_id = torch.argmin(f_values, dim=1)
        best_x = torch.index_select(mesh_pts, dim=0, index=best_id).reshape(self.n_samples, 1, -1)
        xs_exe = self.get_meas_scan_inputs(best_x, bounds, tkwargs)

        ys_exe = torch.tensor([], **tkwargs)
        emit_best = torch.tensor([], **tkwargs)
        
        # is there a way to avoid this for loop? probably
        for sample_id in range(self.n_samples):
            ys_exe = torch.cat((ys_exe, torch.index_select(bss[sample_id], dim=0, index=best_id[sample_id])), dim=0)
            emit_best = torch.cat((emit_best, torch.index_select(f_values[sample_id], dim=0, index=best_id[sample_id])), dim=0)

        emit_best = emit_best.reshape(self.n_samples, 1)

        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "x_tuning_best": best_x,
            "emit_best": emit_best,
        }
    


        return xs_exe, ys_exe, results_dict

    def get_meas_scan_inputs(self, x_tuning: Tensor, bounds: Tensor, tkwargs: dict=None):
        """
        A function that generates the inputs for virtual emittance measurement scans at the tuning
        configurations specified by x_tuning.

        Parameters:
            x_tuning: a tensor of shape n_points x n_tuning_dims, where each row specifies a tuning
                        configuration where we want to do an emittance scan.
                        >>batchshape x n_tuning_configs x n_tuning_dims (ex: batchshape = n_samples x n_tuning_configs)
        Returns:
            xs: tensor, shape (n_tuning_configs*n_steps_meas_scan) x d,
                where n_tuning_configs = x_tuning.shape[0],
                n_steps_meas_scan = len(x_meas),
                and d = x_tuning.shape[1] -- the number of tuning parameters
                >>batchshape x n_tuning_configs*n_steps x ndim
        """
        # each row of x_tuning defines a location in the tuning parameter space
        # along which to perform a quad scan and evaluate emit

        # expand the x tensor to represent quad measurement scans
        # at the locations in tuning parameter space specified by X
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        x_meas = torch.linspace(
            *bounds.T[self.meas_dim], self.n_steps_measurement_param, **tkwargs
        )
        
        # prepare column of measurement scans coordinates
        x_meas_expanded = x_meas.reshape(-1,1).repeat(*x_tuning.shape[:-1],1)
        
        # repeat tuning configs as necessary and concat with column from the line above
        # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
        # where d is the full dimension of the model/posterior space (tuning & meas)
        x_tuning_expanded = torch.repeat_interleave(x_tuning, 
                                                    self.n_steps_measurement_param, 
                                                    dim=-2)


        x = torch.cat(
            (x_tuning_expanded[..., :self.meas_dim], x_meas_expanded, x_tuning_expanded[..., self.meas_dim:]), 
            dim=-1
        )

        return x
            

    def evaluate_objective(self, model, x_tuning, bounds, tkwargs:dict=None, n_samples=10000, use_bmag=True):
        emit, bmag, is_valid, validity_rate, bss = self.evaluate_posterior_emittance_samples(model, 
                                                                                             x_tuning, 
                                                                                             bounds, 
                                                                                             tkwargs, 
                                                                                             n_samples)
        if self.x_key and self.y_key:
            res = (emit[...,0] * emit[...,1]).sqrt()
            if use_bmag:
                res = (bmag[...,0] * bmag[...,1]).sqrt() * res
        else:
            res = emit
            if use_bmag:
                res = (bmag * res).squeeze(-1)
        return res, emit, bmag, is_valid, validity_rate, bss

    def evaluate_posterior_emittance_samples(self, model, x_tuning, bounds, tkwargs:dict=None, n_samples=10000):
        # x_tuning must be shape n_tuning_configs x n_tuning_dims
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
        x = self.get_meas_scan_inputs(x_tuning, bounds, tkwargs) # result shape n_tuning_configs*n_steps x ndim
        
        if isinstance(model, ModelList):
            assert len(x_tuning.shape)==2
            p = model.posterior(x) 
            bss = p.sample(torch.Size([n_samples])) # result shape n_samples x n_tuning_configs*n_steps x num_outputs (1 or 2)

            x = x.reshape(x_tuning.shape[0], self.n_steps_measurement_param, -1) # result n_tuning_configs x n_steps x ndim
            x = x.repeat(n_samples,1,1,1) 
            # result shape n_samples x n_tuning_configs x n_steps x ndim
            bss = bss.reshape(n_samples, x_tuning.shape[0], self.n_steps_measurement_param, -1)
            # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
        else:
            # assert x_tuning.shape[0]==self.n_samples
            assert x_tuning.shape[0]==1
            beamsize_squared_list = [sample_funcs(x).reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param)
                                     for sample_funcs in model]
            # each tensor in beamsize_squared (list) will be shape n_samples x n_tuning_configs x n_steps

            x = x.reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param, -1)
            # n_samples x n_tuning_configs x n_steps x ndim
            bss = torch.stack(beamsize_squared_list, dim=-1) 
            # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
            
        if self.x_key and not self.y_key:
            k = x[..., self.meas_dim] * self.scale_factor # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            beta0 = self.twiss0_x[0].repeat(*bss.shape[:2], 1)
            alpha0 = self.twiss0_x[1].repeat(*bss.shape[:2], 1)
        elif self.y_key and not self.x_key:
            k = x[..., self.meas_dim] * (-1. * self.scale_factor) # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            beta0 = self.twiss0_y[0].repeat(*bss.shape[:2], 1)
            alpha0 = self.twiss0_y[1].repeat(*bss.shape[:2], 1)
        else:
            k_x = (x[..., self.meas_dim] * self.scale_factor) # n_samples x n_tuning x n_steps
            k_y = k_x * -1. # n_samples x n_tuning x n_steps
            k = torch.cat((k_x, k_y)) # shape (2*n_samples x n_tuning x n_steps)
            
            beamsize_squared = torch.cat((bss[...,0], bss[...,1])) 
            # shape (2*n_samples x n_tuning x n_steps)

            rmat_x = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat_y = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat = torch.cat((rmat_x, rmat_y)) # shape (2*n_samples x n_tuning x 2 x 2)
            
            beta0_x = self.twiss0_x[0].repeat(*bss.shape[:2], 1)
            beta0_y = self.twiss0_y[0].repeat(*bss.shape[:2], 1)
            beta0 = torch.cat((beta0_x, beta0_y))

            alpha0_x = self.twiss0_x[1].repeat(*bss.shape[:2], 1)
            alpha0_y = self.twiss0_y[1].repeat(*bss.shape[:2], 1)
            alpha0 = torch.cat((alpha0_x, alpha0_y))

        emit, bmag, sig, is_valid = compute_emit_bmag(k, 
                                          beamsize_squared, 
                                          self.q_len, 
                                          rmat, 
                                          beta0,
                                          alpha0,
                                          thick=self.thick_quad)
        # result shapes: (n_samples x n_tuning), (n_samples x n_tuning), (n_samples x n_tuning x 3 x 1), (n_samples x n_tuning) 
        # or (2*n_samples x n_tuning), (2*n_samples x n_tuning), (2*n_samples x n_tuning x 3 x 1), (2*n_samples x n_tuning) 

        if self.x_key and self.y_key:
            emit = torch.cat((emit[:bss.shape[0]].unsqueeze(-1), emit[bss.shape[0]:].unsqueeze(-1)), dim=-1)
            bmag = torch.cat((bmag[:bss.shape[0]].unsqueeze(-1), bmag[bss.shape[0]:].unsqueeze(-1)), dim=-1)
            is_valid = torch.logical_and(is_valid[:bss.shape[0]], is_valid[bss.shape[0]:])
        else:
            emit = emit.unsqueeze(-1)
            bmag = bmag.unsqueeze(-1)
        #final shapes: n_samples x n_tuning_configs (?? NEED TO CHECK THIS, don't think it's correct)
        
        validity_rate = torch.sum(is_valid, dim=0)/is_valid.shape[0]
        #shape n_tuning_configs
        
        return emit, bmag, is_valid, validity_rate, bss

import matplotlib.pyplot as plt
def plot_virtual_emittance(optimizer, reference_point, dim='xy', ci=0.95, n_points = 50, n_samples=1000, y_max=1., use_bmag=False):
    """
    Plots the emittance cross-sections corresponding to the GP posterior beam size model. 
    This function uses n_samples to produce a confidence interval.
    It DOES NOT use the pathwise sample functions, but rather draws new samples using BoTorch's 
    built-in posterior sampling.
    """
    tkwargs = optimizer.generator._tkwargs
    x_origin = []
    for name in optimizer.generator.vocs.variable_names:
        if name in reference_point.keys():
            x_origin += [reference_point[name]]
    x_origin = torch.tensor(x_origin, **tkwargs).reshape(1,-1)
    #extract GP models
    model = optimizer.generator.train_model()
    if len(optimizer.generator.algorithm.observable_names_ordered) == 2:
        if dim == 'x':
            algorithm = copy.deepcopy(optimizer.generator.algorithm)
            algorithm.y_key = None
            bax_model_ids = [optimizer.generator.vocs.output_names.index(algorithm.x_key)]
        elif dim == 'y':
            algorithm = copy.deepcopy(optimizer.generator.algorithm)
            algorithm.x_key = None
            bax_model_ids = [optimizer.generator.vocs.output_names.index(algorithm.y_key)]
        else:
            algorithm = copy.deepcopy(optimizer.generator.algorithm)
            bax_model_ids = [optimizer.generator.vocs.output_names.index(name)
                                    for name in optimizer.generator.algorithm.observable_names_ordered]
    bax_model = model.subset_output(bax_model_ids)
    meas_dim = algorithm.meas_dim
    
    bounds = optimizer.generator._get_optimization_bounds()
    tuning_domain = torch.cat((bounds.T[: meas_dim], bounds.T[meas_dim + 1:]))
    
    tuning_param_names = optimizer.vocs.variable_names
    del tuning_param_names[meas_dim]
        
    n_tuning_dims = x_origin.shape[1]
    
    fig, axs = plt.subplots(4, n_tuning_dims, sharex='col', sharey='row')
    fig.set_size_inches(3*n_tuning_dims, 9)
        
    for i in range(n_tuning_dims):
        # do a scan of the posterior emittance (via valid sampling)
        x_scan = torch.linspace(*tuning_domain[i], n_points, **tkwargs)
        x_tuning = x_origin.repeat(n_points, 1)
        x_tuning[:,i] = x_scan
        objective, emit, bmag, is_valid, validity_rate, bss = algorithm.evaluate_objective(bax_model, 
                                                                                   x_tuning, 
                                                                                   bounds,
                                                                                   tkwargs,
                                                                                   n_samples,
                                                                                   use_bmag=use_bmag)

        if algorithm.x_key and algorithm.y_key:
            emit = (emit[...,0] * emit[...,1]).sqrt()
            bmag = (bmag[...,0] * bmag[...,1]).sqrt()
        else:
            emit = emit.squeeze(-1)
            bmag = bmag.squeeze(-1)
        print(bmag) 
        all_quants = []
        for result in [objective, emit, bmag]:
            quants = torch.tensor([], **tkwargs)

            for j in range(len(x_scan)):
                cut_ids = torch.tensor(range(len(result[:,j])), device=tkwargs['device'])[is_valid[:,j]]
                result_valid = torch.index_select(result[:,j], dim=0, index=cut_ids)
                q = torch.tensor([(1.-ci)/2., 0.5, (1.+ci)/2.], **tkwargs)
                if len(cut_ids)>=10:
                    quant = torch.quantile(result_valid, q=q, dim=0).reshape(1,-1)
                else:
                    quant = torch.tensor([[float('nan'), float('nan'), float('nan')]], **tkwargs)
                quants = torch.cat((quants, quant))
            all_quants += [quants]

        if n_tuning_dims==1:
            ax = axs[0]
        else:
            ax = axs[0,i]
        ax.fill_between(x_scan, all_quants[0][:,0], all_quants[0][:,2], alpha=0.3)
        ax.plot(x_scan, all_quants[0][:,1])
        ax.axvline(x_origin[0,i], c='r')
        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.set_ylabel('Objective')
            ax.set_ylim(top=y_max)

        if n_tuning_dims==1:
            ax = axs[1]
        else:
            ax = axs[1,i]
        ax.fill_between(x_scan, all_quants[1][:,0], all_quants[1][:,2], alpha=0.3)
        ax.plot(x_scan, all_quants[1][:,1])
        ax.axvline(x_origin[0,i], c='r')
        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.set_ylabel('Emittance')
            ax.set_ylim(top=y_max)
            
        if n_tuning_dims==1:
            ax = axs[2]
        else:
            ax = axs[2,i]
        ax.fill_between(x_scan, all_quants[2][:,0], all_quants[2][:,2], alpha=0.3)
        ax.plot(x_scan, all_quants[2][:,1])
        ax.axvline(x_origin[0,i], c='r')
        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.set_ylabel('Bmag')
            ax.set_ylim(bottom=0.95)
            
        if n_tuning_dims==1:
            ax = axs[3]
        else:
            ax = axs[3,i]
        ax.plot(x_scan, validity_rate, c='m')
        ax.axvline(x_origin[0,i], c='r')
        ax.set_ylim(0,1.05)
        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.set_ylabel('Sample Validity Rate')
            
    return fig, axs