import copy
from abc import ABC
from typing import Dict, Optional, Tuple, Union

import torch
from botorch.models.model import Model
from pydantic import Field

from scipy.optimize import minimize
from torch import Tensor
from xopt.generators.bayesian.bax.algorithms import Algorithm

from .sampling import (
    draw_linear_product_kernel_post_paths,
    draw_product_kernel_post_paths,
)

from .utils import (
    get_meas_scan_inputs_from_tuning_configs,
    get_valid_emittance_samples,
    post_mean_emit_squared,
    post_path_emit_squared,
    post_path_misalignment,
    sum_samplewise_emittance_flat_x,
    sum_samplewise_misalignment_flat_x,
)


def unif_random_sample_domain(n_samples, domain):
    ndim = len(domain)

    # uniform sample, rescaled, and shifted to cover the domain
    x_samples = torch.rand(n_samples, ndim) * torch.tensor(
        [bounds[1] - bounds[0] for bounds in domain]
    ) + torch.tensor([bounds[0] for bounds in domain])

    return x_samples


class ScipyMinimizeEmittance(Algorithm, ABC):
    name = "ScipyMinimizeEmittance"
    scale_factor: float = Field(1.0,
        description="factor by which to multiply the quad inputs to get focusing strengths")
    q_len: float = Field(
        description="the longitudinal thickness of the measurement quadrupole"
    )
    distance: float = Field(
        description="the distance (drift length) from measurement quad to observation screen"
    )
    meas_dim: int = Field(
        description="index identifying the measurement quad dimension in the model"
    )
    n_steps_measurement_param: int = Field(
        3, description="number of steps to use in the virtual measurement scans"
    )
    n_steps_exe_paths: int = Field(
        11, description="number of points to retain as execution path subsequences"
    )

    def get_execution_paths(self, model: Model, bounds: Tensor):
        (
            x_stars_all,
            emit_stars_all,
            is_valid,
            post_paths_cpu,
        ) = self.get_sample_optimal_tuning_configs(model, bounds, cpu=True)

        device = torch.tensor(1).device
        torch.set_default_tensor_type("torch.DoubleTensor")

        # prepare column of measurement scans coordinates
        x_meas_dense = torch.linspace(*bounds.T[self.meas_dim], self.n_steps_exe_paths)

        # expand the X tensor to represent quad measurement scans
        # at the locations in tuning parameter space specified by X
        xs = get_meas_scan_inputs_from_tuning_configs(
            self.meas_dim, x_stars_all, x_meas_dense
        )

        xs_exe = xs.reshape(self.n_samples, self.n_steps_exe_paths, -1)

        # evaluate posterior samples at input locations
        ys_exe = post_paths_cpu(xs_exe).reshape(
            self.n_samples, self.n_steps_exe_paths, 1
        )

        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")

        if sum(is_valid) < 3:
            print("Scipy failed to find at least 3 physically valid solutions.")
            # no cut
            cut_ids = torch.tensor(range(self.n_samples))
        else:
            # only keep the physically valid solutions
            cut_ids = torch.tensor(range(self.n_samples))[is_valid]

        xs_exe = torch.index_select(xs_exe.to(device), dim=0, index=cut_ids)
        ys_exe = torch.index_select(ys_exe.to(device), dim=0, index=cut_ids)
        x_stars = torch.index_select(x_stars_all.to(device), dim=0, index=cut_ids)
        emit_stars = torch.index_select(emit_stars_all.to(device), dim=0, index=cut_ids)

        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "x_stars": x_stars,
            "emit_stars": emit_stars,
            "x_stars_all": x_stars_all,
            "emit_stars_all": emit_stars_all,
            "is_valid": is_valid,
            "post_paths_cpu": post_paths_cpu,
        }

        return xs_exe, ys_exe, results_dict

    def get_sample_optimal_tuning_configs(
        self, model: Model, bounds: Tensor, verbose=False, cpu=False
    ):
        temp_id = self.meas_dim + 1
        tuning_domain = torch.cat((bounds.T[: self.meas_dim], bounds.T[temp_id:]))
        device = torch.tensor(1).device
        torch.set_default_tensor_type("torch.DoubleTensor")

        x_meas = torch.linspace(
            *bounds.T[self.meas_dim], self.n_steps_measurement_param
        )
        cpu_model = copy.deepcopy(model).cpu()

        post_paths_cpu = draw_product_kernel_post_paths(
            cpu_model, n_samples=self.n_samples
        )

        xs_tuning_init = unif_random_sample_domain(
            self.n_samples, tuning_domain
        ).double()

        x_tuning_init = xs_tuning_init.flatten()

        # minimize
        def target_func_for_scipy(x_tuning_flat):
            return (
                sum_samplewise_emittance_flat_x(
                    post_paths_cpu,
                    self.scale_factor,
                    self.q_len,
                    self.distance,
                    torch.tensor(x_tuning_flat),
                    self.meas_dim,
                    x_meas.cpu(),
                )
                .detach()
                .cpu()
                .numpy()
            )

        def target_func_for_torch(x_tuning_flat):
            return sum_samplewise_emittance_flat_x(
                post_paths_cpu,
                self.scale_factor,
                self.q_len,
                self.distance,
                x_tuning_flat,
                self.meas_dim,
                x_meas.cpu(),
            )

        def target_jac(x):
            return (
                torch.autograd.functional.jacobian(
                    target_func_for_torch, torch.tensor(x)
                )
                .detach()
                .cpu()
                .numpy()
            )

        res = minimize(
            target_func_for_scipy,
            x_tuning_init.detach().cpu().numpy(),
            jac=target_jac,
            bounds=tuning_domain.repeat(self.n_samples, 1).detach().cpu().numpy(),
            options={"eps": 1e-03},
        )
        if verbose:
            print(
                "ScipyMinimizeEmittance evaluated",
                self.n_samples,
                "(pathwise) posterior samples",
                res.nfev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyMinimizeEmittance evaluated",
                self.n_samples,
                "(pathwise) posterior sample jacobians",
                res.njev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyMinimizeEmittance took",
                res.nit,
                "steps in get_sample_optimal_tuning_configs().",
            )

        x_stars_flat = torch.tensor(res.x)

        x_stars_all = x_stars_flat.reshape(
            self.n_samples, -1
        )  # each row represents its respective sample's optimal tuning config

        emit_stars_all, is_valid = post_path_emit_squared(
            post_paths_cpu,
            self.scale_factor,
            self.q_len,
            self.distance,
            x_stars_all,
            self.meas_dim,
            x_meas.cpu(),
            samplewise=True,
        )

        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")

        if cpu:
            return (
                x_stars_all,
                emit_stars_all,
                is_valid,
                post_paths_cpu,
            )  # X_stars should still be on cpu
        else:
            return (
                x_stars_all.to(device),
                emit_stars_all.to(device),
                is_valid.to(device),
                post_paths_cpu,
            )

    def mean_output(self, model: Model, bounds: Tensor, num_restarts=1):
        x_meas = torch.linspace(
            *bounds.T[self.meas_dim], self.n_steps_measurement_param
        )

        temp_id = self.meas_dim + 1
        tuning_domain = torch.cat((bounds.T[: self.meas_dim], bounds.T[temp_id:]))

        def target_func_for_scipy(x_tuning_flat):
            return (
                post_mean_emit_squared(
                    model,
                    self.scale_factor,
                    self.q_len,
                    self.distance,
                    torch.tensor(x_tuning_flat).reshape(num_restarts, -1),
                    self.meas_dim,
                    x_meas,
                )[0]
                .flatten()
                .sum()
                .detach()
                .cpu()
                .numpy()
            )

        def target_func_for_torch(x_tuning_flat):
            return (
                post_mean_emit_squared(
                    model,
                    self.scale_factor,
                    self.q_len,
                    self.distance,
                    x_tuning_flat.reshape(num_restarts, -1),
                    self.meas_dim,
                    x_meas,
                )[0]
                .flatten()
                .sum()
            )

        def target_jac(x):
            return (
                torch.autograd.functional.jacobian(
                    target_func_for_torch, torch.tensor(x)
                )
                .detach()
                .cpu()
                .numpy()
            )

        x_tuning_init = (
            unif_random_sample_domain(num_restarts, tuning_domain).double().flatten()
        )

        res = minimize(
            target_func_for_scipy,
            x_tuning_init.detach().cpu().numpy(),
            jac=target_jac,
            bounds=tuning_domain.repeat(num_restarts, 1).detach().cpu().numpy(),
            #                        tol=1e-5,
            options={"eps": 1e-03},
        )

        x_tuned_candidates = torch.tensor(res.x).reshape(num_restarts, -1)
        min_emit_sq_candidates = post_mean_emit_squared(
            model,
            self.scale_factor,
            self.q_len,
            self.distance,
            x_tuned_candidates,
            self.meas_dim,
            x_meas,
        )[0].squeeze()

        min_emit_sq_id = torch.argmin(min_emit_sq_candidates)

        x_tuned = x_tuned_candidates[min_emit_sq_id].reshape(1, -1)

        (
            emits_at_target_valid,
            sample_validity_rate,
        ) = get_valid_emittance_samples(
            model,
            self.scale_factor,
            self.q_len,
            self.distance,
            x_tuned,
            bounds.T,
            self.meas_dim,
            n_samples=10000,
            n_steps_quad_scan=10,
        )

        return (
            x_tuned,
            emits_at_target_valid,
            sample_validity_rate,
        )


class ScipyBeamAlignment(Algorithm, ABC):
    name = "ScipyBeamAlignment"
    meas_dims: Union[int, list[int]] = Field(
        description="list of indeces identifying the measurement quad dimensions in the model"
    )

    def get_execution_paths(
        self, model: Model, bounds: Tensor
    ) -> Tuple[Tensor, Tensor, Dict]:
        """get execution paths that minimize the objective function"""

        x_stars_all, xs, ys, post_paths_cpu = self.get_sample_optimal_tuning_configs(
            model, bounds, cpu=False
        )

        xs_exe = xs
        ys_exe = ys.reshape(*ys.shape, 1)

        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "X_stars": x_stars_all,
            "post_paths_cpu": post_paths_cpu,
        }

        return xs_exe, ys_exe, results_dict

    def get_sample_optimal_tuning_configs(
        self, model: Model, bounds: Tensor, verbose=False, cpu=False
    ):
        meas_scans = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        )
        ndim = bounds.shape[1]
        tuning_dims = [i for i in range(ndim) if i not in self.meas_dims]
        tuning_domain = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(tuning_dims)
        )

        device = torch.tensor(1).device
        torch.set_default_tensor_type("torch.DoubleTensor")

        cpu_model = copy.deepcopy(model).cpu()

        post_paths_cpu = draw_linear_product_kernel_post_paths(
            cpu_model, n_samples=self.n_samples
        )

        xs_tuning_init = unif_random_sample_domain(
            self.n_samples, tuning_domain
        ).double()

        x_tuning_init = xs_tuning_init.flatten()

        # minimize
        def target_func_for_scipy(x_tuning_flat):
            return (
                sum_samplewise_misalignment_flat_x(
                    post_paths_cpu,
                    torch.tensor(x_tuning_flat),
                    self.meas_dims,
                    meas_scans.cpu(),
                )
                .detach()
                .cpu()
                .numpy()
            )

        def target_func_for_torch(x_tuning_flat):
            return sum_samplewise_misalignment_flat_x(
                post_paths_cpu, x_tuning_flat, self.meas_dims, meas_scans.cpu()
            )

        def target_jac(x):
            return (
                torch.autograd.functional.jacobian(
                    target_func_for_torch, torch.tensor(x)
                )
                .detach()
                .cpu()
                .numpy()
            )

        res = minimize(
            target_func_for_scipy,
            x_tuning_init.detach().cpu().numpy(),
            jac=target_jac,
            bounds=tuning_domain.repeat(self.n_samples, 1).detach().cpu().numpy(),
            options={"eps": 1e-03},
        )
        if verbose:
            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior samples",
                res.nfev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior sample jacobians",
                res.njev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment took",
                res.nit,
                "steps in get_sample_optimal_tuning_configs().",
            )

        x_stars_flat = torch.tensor(res.x)

        x_stars_all = x_stars_flat.reshape(
            self.n_samples, -1
        )  # each row represents its respective sample's optimal tuning config

        misalignment, xs, ys = post_path_misalignment(
            post_paths_cpu,
            x_stars_all,  # n x d tensor
            self.meas_dims,  # list of integers
            meas_scans.cpu(),  # tensor of measurement device(s) scan inputs, shape: len(meas_dims) x 2
            samplewise=True,
        )

        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")

        if cpu:
            return x_stars_all, xs, ys, post_paths_cpu  # x_stars should still be on cpu
        else:
            return x_stars_all.to(device), xs.to(device), ys.to(device), post_paths_cpu
