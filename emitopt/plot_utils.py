import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from emitopt.beam_dynamics import build_quad_rmat, propagate_beam_quad_scan


def plot_valid_thick_quad_fits(k, beamsize, q_len, rmat, emit, bmag, sig, ci=0.95, tkwargs=None):
    """
    Plots the fit results produced by emitopt.analysis.compute_emit_bayesian().

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        beamsize: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        sig: tensor, shape (n_scans x 3 x 1), containing the computed sig11, sig12, sig22
                corresponding to each measurement scan
                
        emit: shape (n_scans x 1) containing the geometric emittance fit results for each scan

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
        
        rmat: the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.
                
        ci: "Confidence interval" for plotting upper/lower quantiles.

        tkwargs: dict containing the tensor device and dtype
    """
    
    tkwargs = twkargs if tkwargs else {"dtype": torch.double, "device": "cpu"}


    
    fig, axs = plt.subplots(3)
    fig.set_size_inches(5,9)
    
    ax=axs[0]
    
    k_fit = torch.linspace(k.min(), k.max(), 10, **tkwargs)
    quad_rmats = build_quad_rmat(k_fit, q_len) # result shape (len(k_fit) x 2 x 2)
    total_rmats = rmat.reshape(1,2,2).double() @ quad_rmats.double() # result shape (len(k_fit) x 2 x 2)
    sig_final = propagate_beam_quad_scan(sig, emit, total_rmats)[0] # result shape len(sig) x len(k_fit) x 3 x 1
    bss_fit = sig_final[:,:,0,0]

    upper_quant = torch.quantile(bss_fit.sqrt(), q=0.5 + ci / 2.0, dim=0)
    lower_quant = torch.quantile(bss_fit.sqrt(), q=0.5 - ci / 2.0, dim=0)

    fit = ax.fill_between(
        k_fit.detach().numpy(),
        lower_quant*1.e6,
        upper_quant*1.e6,
        alpha=0.3,
        label='Model Fit',
        zorder=1,
    )
    
    obs = ax.scatter(
        k, beamsize*1.e6, marker="x", s=120, c="orange", label="Measurements", zorder=2
    )
    ax.set_title("Beam Size at Screen")
    ax.set_xlabel(r"Measurement Quad Geometric Focusing Strength ($[k]=m^{-2}$)")
    ax.set_ylabel(r"r.m.s. Beam Size")# ($[\sigma]=\mu m$)")
    ax.legend(handles=[obs, fit])
    
    ax=axs[1]
    ax.hist(emit.flatten(), density=True)
    ax.set_title('Geometric Emittance Distribution')
    ax.set_xlabel(r'Geometric Emittance')# ($[\epsilon]=m*rad$)')
    ax.set_ylabel('Probability Density')
    
    ax=axs[2]
    ax.hist(bmag.flatten(), range=(1,5), bins=20, density=True)
    ax.set_title(r'$\beta_{mag}$ Distribution')
    ax.set_xlabel(r'$\beta_{mag}$ at Screen')
    ax.set_ylabel('Probability Density')
    
    plt.tight_layout()


def plot_sample_optima_convergence_inputs(results, tuning_parameter_names=None, show_valid_only=True):
    """
    Plots the distribution of optimal input values from the virtual minimization results as a function 
    of iteration number (one plot for each input variable).
    
    Parameters:
        results: dict containing {iteration_number:bax_generator.algorithm_results} as key/value pairs.
    """
    ndim = results[1]["x_tuning_best"].shape[-1]
    niter = max(results.keys())
    nsamples = results[1]["x_tuning_best"].shape[0]

    if tuning_parameter_names is None:
        tuning_parameter_names = ['tp_' + str(i) for i in range(ndim)]
    
    fig, axs = plt.subplots(ndim, 1)
    fig.set_size_inches(8,3*ndim)

    for i in range(ndim):

        if ndim > 1:
            ax = axs[i]
        else:
            ax = axs

        ax.set_ylabel(tuning_parameter_names[i])
        ax.set_xlabel("iteration")
        
        if i == 0:
            ax.set_title("Sample Optima Distribution: Tuning Parameters")
            
        for key in results.keys():
            if show_valid_only:
                ax.scatter(torch.tensor([key]).repeat(len(results[key]["x_tuning_best_retained"][...,i])), 
                           results[key]["x_tuning_best_retained"][...,i].flatten(),
                           c='C0')
            else:
                ax.scatter(torch.tensor([key]).repeat(len(results[key]["x_tuning_best"][...,i])), 
                           results[key]["x_tuning_best"][...,i].flatten(),
                           c='C0')
    plt.tight_layout()
    
    return fig, axs


def plot_sample_optima_convergence_emits(results):
    """
    Plots the distribution of optimal emittance values from the virtual minimization
    results as a function of iteration number.
    
    Parameters:
        results: dict containing {iteration_number:bax_generator.algorithm_results} as key/value pairs.
    """
    niter = max(results.keys())
    nsamples = results[1]["emit_best"].shape[0]
    
    fig, ax = plt.subplots(1)

    ax.set_ylabel("$\epsilon$")
    ax.set_xlabel("iteration")
    ax.set_title("Sample Optima Distribution: Emittance")
    for key in results.keys():
        ax.scatter(torch.tensor([key]).repeat(len(results[key]["emit_best"].flatten())), 
                   results[key]["emit_best"].flatten().detach(), 
                   c='C0')
    plt.tight_layout()
    
    return fig, ax


def plot_model_cross_section(model, vocs, scan_dict, nx=50, ny=50):
    scan_var_model_ids = {}
    scan_inputs_template = []
    for i, var_name in enumerate(vocs.variable_names):
        if isinstance(scan_dict[var_name], list):
            if len(scan_dict[var_name])!=2:
                raise ValueError("Entries must be either float or 2-element list of floats.")
            scan_var_model_ids[var_name] = i
            scan_inputs_template += [scan_dict[var_name][0]]
        else:
            if not isinstance(scan_dict[var_name],float):
                raise ValueError("Entries must be either float or 2-element list of floats.")
            scan_inputs_template += [scan_dict[var_name]]
    
    if len(scan_var_model_ids)!=2:
        raise ValueError("Exactly 2 keys must have entries that are 2-element lists.")
    
    scan_inputs = torch.tensor(scan_inputs_template).repeat(nx*ny, 1)

    scan_varx = list(scan_var_model_ids.keys())[0]
    scan_vary = list(scan_var_model_ids.keys())[1]

    lsx = torch.linspace(*scan_dict[scan_varx], nx)
    lsy = torch.linspace(*scan_dict[scan_vary], ny)
    meshx, meshy = torch.meshgrid((lsx, lsy), indexing='xy')
    mesh_points_serial = torch.cat((meshx.reshape(-1,1), meshy.reshape(-1,1)), dim=1)
    
    model_idx = scan_var_model_ids[scan_varx]
    scan_inputs[:,model_idx] = mesh_points_serial[:,0]
    
    model_idy = scan_var_model_ids[scan_vary]
    scan_inputs[:,model_idy] = mesh_points_serial[:,1]

    with torch.no_grad():
        posterior = model.posterior(scan_inputs)
        mean = posterior.mean
        var = posterior.variance

    mean = mean.reshape(ny, nx)
    var = var.reshape(ny, nx)
    
    fig, axs = plt.subplots(2)

    ax = axs[0]
    m = ax.pcolormesh(meshx, meshy, mean)
    ax.set_xlabel(scan_varx)
    ax.set_ylabel(scan_vary)
    ax.set_title('Posterior Mean')
    cbar_m = fig.colorbar(m, ax=ax)
    
    ax = axs[1]
    v = ax.pcolormesh(meshx, meshy, var)
    ax.set_xlabel(scan_varx)
    ax.set_ylabel(scan_vary)
    ax.set_title('Posterior Variance')
    cbar_v = fig.colorbar(v, ax=ax)

    plt.tight_layout()


def plot_pathwise_surface_samples_2d(optimizer): # paper figure
    """
    Plots a few GP model posterior sample surfaces for a 2d input space. Also plots the virtual emittance
    prediction as a function of the single tuning parameter.
    """
    if ndim==2:

        device = torch.tensor(1).device
        torch.set_default_tensor_type('torch.DoubleTensor')

        fig, axs = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
        fig.set_size_inches(15,10)

        ax = axs[0]

        for s in range(3):

            # plot first 3 beam size surface samples
            xlin, ylin = torch.arange(-3,1,0.05), torch.arange(-40,40, 1.)
            X, Y = torch.meshgrid(xlin, ylin)
            XY = torch.cat((X.reshape(-1,1), Y.reshape(-1,1)), dim=1)
            print(XY.shape)
            Z = optimizer.generator.algorithm_results['sample_funcs_list'][0](XY)[s].reshape(X.shape).detach()
            cmap='viridis'
            surf = ax.plot_surface(Y, X, Z, cmap=cmap,
                                   linewidth=0, antialiased=True, alpha=0.3, rasterized=True)

            # add orange parabolic highlights
            ax.plot(Y[0,:].numpy(), Z[0,:].numpy(), zs=X[0,0].item(), zdir='y', c='C1', lw=2, zorder=10)
            ax.plot(Y[int(len(Z[0,:])/2),:].numpy(), Z[int(len(Z[0,:])/2),:].numpy(), zs=X[int(len(Z[0,:])/2),0].item(), zdir='y', c='C1', lw=2)
            ax.plot(Y[-1,:].numpy(), Z[-1,:].numpy(), zs=X[-1,0].item(), zdir='y', c='C1', lw=2)




        # plot initial observations
        x0 = torch.tensor(optimizer.data['x0'].values)[:n_obs_init]
        x1 = torch.tensor(optimizer.data['x1'].values)[:n_obs_init]
        y = torch.tensor([item.item() for item in optimizer.data['y'].values])[:n_obs_init]
        ax.scatter(x1.flatten(), x0.flatten(), y.flatten(), marker='o', c='C0', alpha=1, s=80, label='Random (Initial) Observations', zorder=15)

        # plot bax observations
        x0 = torch.tensor(optimizer.data['x0'].values)[n_obs_init:]
        x1 = torch.tensor(optimizer.data['x1'].values)[n_obs_init:]
        y = torch.tensor([item.item() for item in optimizer.data['y'].values])[n_obs_init:]
        ax.scatter(x1.flatten(), x0.flatten(), y.flatten(), marker='o', c='C1', alpha=1, s=80, label='BAX Observations', zorder=15)

        ax.set_title('Beam Size Surface Samples')
        ax.set_ylabel('Tuning Parameter')
        ax.set_xlabel('Measurement Parameter')
        ax.set_zlabel('Beam Size Squared')

        ax.set_ylim(-3, 1)
        ax.set_zlim(0)

        # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        ax.legend()
        ax.dist = 12



        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")



        # do a scan (along the tuning dimension) of our emittance predictions
        emit_lowers = torch.tensor([])
        emit_uppers = torch.tensor([])
        emit_meds = torch.tensor([])
        for tuning_param in xlin:
            x_tuning = tuning_param.reshape(1,-1).to(device)
            emits, emit_x, emit_y, sample_validity_rate = get_valid_geo_mean_emittance_samples_thick_quad(bax_model, 
                                                     scale_factor, 
                                                     0.108, 
                                                     2.26, 
                                                     x_tuning, 
                                                     vocs.bounds.T, 
                                                     meas_dim, 
                                                     n_samples=100000, 
                                                     n_steps_quad_scan=10)
            emit_lower = torch.quantile(emits, q=0.025, dim=0)
            emit_upper = torch.quantile(emits, q=0.975, dim=0)
            emit_med = torch.quantile(emits, q=0.5, dim=0)

            emit_lowers = torch.cat((emit_lowers, emit_lower))
            emit_uppers = torch.cat((emit_uppers, emit_upper))
            emit_meds = torch.cat((emit_meds, emit_med))

        #get a few batches of n_samples pathwise sample optima
        x_stars_all = torch.tensor([])
        emit_stars_all = torch.tensor([])
        for i in range(5):
            algo = optimizer.generator.algorithm
            results_dict = algo.get_execution_paths(beam_size_model, torch.tensor(vocs.bounds))[-1]
            x_stars = results_dict['x_stars']
            emit_stars = results_dict['emit_stars'].detach()
            x_stars_all = torch.cat((x_stars_all, x_stars), dim=0)
            emit_stars_all = torch.cat((emit_stars_all, emit_stars), dim=0)

        ax = axs[1]

        # plot median emittance curve
        medline, = ax.plot(emit_meds.cpu().numpy(), xlin.numpy(), zs=0, zdir='z', c='g', label='Median')

        opt_cross = ax.scatter(emit_stars_all.flatten().cpu(), x_stars_all.flatten().cpu(), zs=0, zdir='z', marker='x', s=40, c='m', alpha=0.5, label='Sample Optima')

        # plot emittance 95% confidence interval as a Poly3DCollection (ordering of vertices matters)
        verts = (
            [(emit_lowers[i].item(), xlin[i].item(), 0) for i in range(len(xlin))] + 
            [(emit_uppers[i].item(), xlin[i].item(), 0) for i in range(len(xlin))][::-1]
        )
        ax.add_collection3d(Poly3DCollection([verts],color='g', edgecolor='None', alpha=0.5)) # Add a polygon instead of fill_between


        ax.set_xlabel('Emittance')
        ax.set_ylabel('Tuning Parameter')
        ax.set_title('Emittance Measurement Samples')

        ax.set_xlim(0,25)
        ax.set_ylim(-3,1)
        ax.set_zlim(0,1)

        # remove vertical tick marks
        ax.set_zticks([])

        # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        orange_patch = mpatches.Patch(color='g', alpha=0.5, label='95% C.I.')
        ax.legend(handles=[medline, orange_patch, opt_cross])
        ax.dist = 12



        ax = axs[2]
        bins = 10
        freq, edges = torch.histogram(x_stars_all.flatten().cpu(), bins=bins, density=True)
        for i in range(bins):
            uverts = []
            lverts = []
            uverts += [(freq[i].item(), edges[i].item(), 0), (freq[i].item(), edges[i+1].item(), 0)]
            lverts += [(0, edges[i+1].item(), 0), (0, edges[i].item(), 0)]
            verts = uverts + lverts
            ax.add_collection3d(Poly3DCollection([verts],color='m', edgecolor='k')) # Add a polygon instead of fill_between

        ax.set_title('Distribution of Sample Optimal Tuning Parameters')
        ax.set_ylabel('Tuning Parameter')
        ax.set_xlabel('Frequency')

        ax.set_xlim(0,2)
        ax.set_ylim(-3,1)
        ax.set_zlim(0,1)

        # remove vertical tick marks
        ax.set_zticks([])

        # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        ax.dist = 12

        plt.tight_layout()
        plt.savefig('beamsize-surfaces-with-emittance-1.svg', format='svg')
        plt.show()


def plot_pathwise_emittance_vs_tuning(optimizer, x_origin, sample_ids=None, tkwargs:dict=None, transform_target=False):
    """
    Plots the emittance cross-sections corresponding to the posterior beam size pathwise sample functions. 
    The cross-sections are evaluated about the point in input space given by x_origin.
    """
    tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
    
    if sample_ids is None:
        sample_ids = list(range(optimizer.generator.algorithm.n_samples))
        
    n_tuning_dims = x_origin.shape[1]
    fig, axs = plt.subplots(1, n_tuning_dims, sharey='row')
    if n_tuning_dims == 1: axs = [axs]

    fig.set_size_inches(3*(n_tuning_dims), 3)

    meas_dim = optimizer.generator.algorithm.meas_dim
    tuning_dims = list(range(n_tuning_dims + 1))
    tuning_dims.remove(meas_dim)
    for i, scan_dim in enumerate(tuning_dims):
                    
        X_tuning_scan = x_origin.repeat(100,1)
        ls = torch.linspace(*optimizer.vocs.bounds.T[scan_dim],100)
        X_tuning_scan[:,i] = ls
        X_tuning_scan = X_tuning_scan.repeat(optimizer.generator.algorithm.n_samples, 1, 1)
        sample_funcs_list = optimizer.generator.algorithm_results['sample_funcs_list']
        emit, is_valid, validity_rate = optimizer.generator.algorithm.evaluate_posterior_emittance_samples(sample_funcs_list, 
                                                                                     X_tuning_scan, 
                                                                                     optimizer.vocs.bounds,
                                                                                     transform_target=transform_target,
                                                                                    )

        ax = axs[i]
        
        for j in sample_ids:
            sample_label = 'Samples' if j==0 else None
            cutoff_label = 'Physical limit' if j==0 else None
            ax.plot(ls.cpu(), emit[j].detach().cpu(), label=sample_label, c='C0', alpha=0.5)
            
        ax.axhline(0, c='k', ls='--', label=cutoff_label)
        ax.axvline(x_origin[0,i], c='r', label='Scan origin')
        ax.set_xlabel('tuning param ' + str(i))

        if i == 0:
            ax.set_ylabel('$\epsilon}$')
            ax.legend()

    plt.tight_layout()
    return fig, axs


def plot_virtual_emittance(optimizer, x_origin, ci=0.95, tkwargs:dict=None, n_points = 100, n_samples=10000):
    """
    Plots the emittance cross-sections corresponding to the GP posterior beam size model. 
    This function uses n_samples to produce a confidence interval.
    It DOES NOT use the pathwise sample functions, but rather draws new samples using BoTorch's 
    built-in posterior sampling.
    """
    tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
    
    #extract GP models
    model = optimizer.generator.train_model()
    bax_model_ids = [optimizer.generator.vocs.output_names.index(name)
                            for name in optimizer.generator.algorithm.observable_names_ordered]
    bax_model = model.subset_output(bax_model_ids)
    meas_dim = optimizer.generator.algorithm.meas_dim
    
    bounds = optimizer.generator._get_optimization_bounds()
    tuning_domain = torch.cat((bounds.T[: meas_dim], bounds.T[meas_dim + 1:]))
    
    tuning_param_names = optimizer.vocs.variable_names
    del tuning_param_names[meas_dim]
    
    algorithm = optimizer.generator.algorithm
    
    n_tuning_dims = x_origin.shape[1]
    
    fig, axs = plt.subplots(2, n_tuning_dims, sharex='col', sharey='row')
    fig.set_size_inches(3*n_tuning_dims, 6)
        
    for i in range(n_tuning_dims):
        # do a scan of the posterior emittance (via valid sampling)
        x_scan = torch.linspace(*tuning_domain[i], n_points, **tkwargs)
        x_tuning = x_origin.repeat(n_points, 1)
        x_tuning[:,i] = x_scan
        emit, is_valid, validity_rate = algorithm.evaluate_posterior_emittance_samples(bax_model, 
                                                                                   x_tuning, 
                                                                                   bounds,
                                                                                   tkwargs,
                                                                                   n_samples,
                                                                                   transform_target=False)
        quants = torch.tensor([])
        
        for j in range(len(x_scan)):
            cut_ids = torch.tensor(range(len(emit[:,j])), device=tkwargs['device'])[is_valid[:,j]]
            emit_valid = torch.index_select(emit[:,j], dim=0, index=cut_ids)
            q = torch.tensor([(1.-ci)/2., 0.5, (1.+ci)/2.], **tkwargs)
            if len(cut_ids)>=10:
                quant = torch.quantile(emit_valid, q=q, dim=0).reshape(1,-1)
            else:
                quant = torch.tensor([[float('nan'), float('nan'), float('nan')]], **tkwargs)
            quants = torch.cat((quants, quant))

        if n_tuning_dims==1:
            ax = axs[0]
        else:
            ax = axs[0,i]
        ax.fill_between(x_scan, quants[:,0], quants[:,2], alpha=0.3)
        ax.plot(x_scan, quants[:,1])
        ax.axvline(x_origin[0,i], c='r')
        
        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.legend()
            ax.set_ylabel('Emittance')
            
        if n_tuning_dims==1:
            ax = axs[1]
        else:
            ax = axs[1,i]
        ax.plot(x_scan, validity_rate, c='m')
        ax.axvline(x_origin[0,i], c='r')
        ax.set_ylim(0,1)

        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.legend()
            ax.set_ylabel('Sample Validity Rate')
            
    return fig, axs


def plot_posterior_mean_modeled_emittance(optimizer, x_tuning, ground_truth_emittance_fn=None):
    """
    Plots the emittance cross-sections corresponding to the GP posterior mean of the beam size.
    """
    
    # get the beam size squared models in x and y
    model = optimizer.generator.train_model()
    bax_model_ids = [optimizer.generator.vocs.output_names.index(name)
                            for name in optimizer.generator.algorithm.observable_names_ordered]
    bax_model = model.subset_output(bax_model_ids)
    
    n_tuning_dims = x_tuning.shape[1]
    fig, axs = plt.subplots(1, n_tuning_dims)
    if n_tuning_dims == 1: axs = [axs]

    fig.set_size_inches(3*(n_tuning_dims), 3)

    meas_dim = optimizer.generator.algorithm.meas_dim
    tuning_dims = list(range(n_tuning_dims + 1))
    tuning_dims.remove(meas_dim)
    for i, scan_dim in enumerate(tuning_dims):
        X_tuning_scan = x_tuning.repeat(100,1)
        ls = torch.linspace(*optimizer.vocs.bounds.T[scan_dim],100)
        X_tuning_scan[:,i] = ls
        X_meas = torch.linspace(*optimizer.vocs.bounds.T[meas_dim],11)

        
        emit_sq_xy = []
        for bss_model, sign in zip(bax_model.models, [1,-1]):
            emit_sq = post_mean_emit_squared_thick_quad(
                model=bss_model,
                scale_factor=sign*optimizer.generator.algorithm.scale_factor,
                q_len=optimizer.generator.algorithm.q_len,
                distance=optimizer.generator.algorithm.distance,
                x_tuning=X_tuning_scan.cpu(),
                meas_dim=meas_dim,
                x_meas=X_meas.cpu(),
            )[0]
            emit_sq_xy += [emit_sq]
            
        geo_mean_emit = torch.sqrt(emit_sq_xy[0].abs().sqrt() * emit_sq_xy[1].abs().sqrt())
        ax = axs[i]

        if ground_truth_emittance_fn is not None:
            gt_emits, gt_emit_xy = ground_truth_emittance_fn(x_tuning=X_tuning_scan)
            ax.plot(ls, gt_emits, c='k', label='ground truth')
            
        ax.plot(ls.cpu(), geo_mean_emit.detach().cpu()*1.e-6, label='GP mean')
        ax.axhline(0, c='k', ls='--', label='physical cutoff')

        ax.set_xlabel('tuning param ' + str(i))

        if i == 0:
            ax.set_ylabel('$\sqrt{\epsilon_x\epsilon_y}$')
            ax.legend()

    plt.tight_layout()
    plt.show()

# +
import time
from botorch.optim.optimize import optimize_acqf

def plot_acq_func_opt_results(optimizer):
    start = time.time()
    acq = optimizer.generator.get_acquisition(optimizer.generator.model)
    end = time.time()
    print('get_acquisition took', end-start, 'seconds.')
    
    start = time.time()
    for i in range(1):
        res = optimize_acqf(acq_function=acq,
                            bounds=torch.tensor(optimizer.vocs.bounds),
                            q=1,
                            num_restarts=10,
                            raw_samples=20,
                            options={'maxiter':50}
                           )
    end = time.time()
    print('optimize_acqf took', end-start, 'seconds.')
    
    last_acq = res[0]
    
    ndim = optimizer.vocs.bounds.shape[1]
    fig, axs = plt.subplots(1, ndim)

    fig.set_size_inches(3*(ndim), 3)

    for scan_dim in range(ndim):
        X_scan = last_acq.repeat(100,1)
        ls = torch.linspace(last_acq[0,scan_dim]-1,last_acq[0,scan_dim]+1,100)

        X_scan[:,scan_dim] = ls

        acq_scan = torch.tensor([acq(X.reshape(1,-1)) for X in X_scan]).reshape(-1)

        ax = axs[scan_dim]

        ax.plot(ls.cpu(), acq_scan.detach().cpu())
        ax.axvline(last_acq[0,scan_dim].cpu(), c='r', label='Acquisition Result')


        ax.set_xlabel('Input ' + str(scan_dim))

        if scan_dim == 0:
            ax.set_ylabel('Acquisition Function')
            ax.legend()

    plt.tight_layout()
    plt.show()


# -

#TO-DO: Need to handle case for just x or y, CHECK x_meas/k_meas units
def plot_virtual_measurement_scan(optimizer, x_tuning, tkwargs=None):
    assert x_tuning.shape[0]==1
    meas_dim = optimizer.generator.algorithm.meas_dim
    n_steps_measurement_param = optimizer.generator.algorithm.n_steps_measurement_param
    x_meas = torch.linspace(*optimizer.vocs.bounds.T[meas_dim], n_steps_measurement_param)
    x_meas_scan = optimizer.generator.algorithm.get_meas_scan_inputs(x_tuning=x_tuning, 
                                                                     bounds=optimizer.generator._get_optimization_bounds(), 
                                                                     tkwargs=tkwargs)

    # get the beam size squared models in x and y
    model = optimizer.generator.train_model()
    bax_model_ids = [optimizer.generator.vocs.output_names.index(name)
                            for name in optimizer.generator.algorithm.observable_names_ordered]
    bax_model = model.subset_output(bax_model_ids)
    
    if len(bax_model_ids)==2:
        labels = ['$\sigma_{x,rms}^2$', '$\sigma_{y,rms}^2$']
    else:
        labels = ['$\sigma_{rms}^2$']
    
    fig, ax = plt.subplots(1)
    
    for bss_model, label in zip(bax_model.models, labels):
        bss_posterior = bss_model.posterior(x_meas_scan)
        bss_mean = bss_posterior.mean.flatten().detach()
        bss_var = bss_posterior.variance.flatten().detach()
        ax.plot(x_meas, bss_mean.detach(), label=label)
        ax.fill_between(x_meas, (bss_mean-2*bss_var.sqrt()), (bss_mean+2*bss_var.sqrt()), alpha=0.3)
    
    ax.set_title('Mean-Square Beam Size GP Model Output')
#     plt.xlabel('Measurement Quad Focusing Strength ($[k]=m^{-2}$)')
    ax.set_xlabel('Measurement Quad Setting (Machine Units)')
    ax.set_ylabel('Mean-Square Beam Size (mm)')
    ax.legend()
    
    return fig, ax
