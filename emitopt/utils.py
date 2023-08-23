import torch


def sum_samplewise_misalignment_flat_x(
    post_paths, x_tuning_flat, meas_dims, meas_scans
):
    """
    A wrapper function that computes the sum of the samplewise misalignments for more convenient
    minimization with scipy.

    arguments:
        Same as post_path_misalignment() EXCEPT:

        x_tuning_flat: a FLATTENED tensor formerly of shape (n_samples x ndim) where the nth
                        row defines a point in tuning-parameter space at which to evaluate the
                        misalignment of the nth posterior pathwise sample given by post_paths

        NOTE: x_tuning_flat must be 1d (flattened) so the output of this function can be minimized
                with scipy minimization routines (that expect a 1d vector of inputs)
        NOTE: samplewise is set to True to avoid unncessary computation during simultaneous minimization
                of the pathwise misalignments.
    """

    x_tuning = x_tuning_flat.double().reshape(post_paths.n_samples, -1)

    return torch.sum(
        post_path_misalignment(
            post_paths, x_tuning, meas_dims, meas_scans, samplewise=True
        )[0]
    )


def post_path_misalignment(
    post_paths,
    x_tuning,  # n x d tensor
    meas_dims,  # list of integers
    meas_scans,  # tensor of measurement device(s) scan inputs, shape: len(meas_dims) x 2
    samplewise=False,
):
    """
    A function that computes the beam misalignment(s) through a set of measurement quadrupoles
    from a set of pathwise samples taken from a SingleTaskGP model of the beam centroid position with
    respect to some tuning devices and some measurement quadrupoles.

    arguments:
        post_paths: a pathwise posterior sample from a SingleTaskGP model of the beam centroid
                    position (assumes Linear ProductKernel)
        x_tuning: a tensor of shape (n_samples x n_tuning_dims) where the nth row defines a point in
                    tuning-parameter space at which to evaluate the misalignment of the nth
                    posterior pathwise sample given by post_paths
        meas_dims: the dimension indeces of our model that describe the quadrupole measurement devices
        meas_scans: a tensor of measurement scan inputs, shape len(meas_dims) x 2, where the nth row
                    contains two input scan values for the nth measurement quadrupole
        samplewise: boolean. Set to False if you want to evaluate the misalignment for every point on
                    every sample. If set to True, the misalignment for the nth sample (given by post_paths)
                    will only be evaluated at the nth point (given by x_tuning). If samplewise is set to
                    True, x_tuning must be shape n_samples x n_tuning_dims

     returns:
         misalignment: the sum of the squared slopes of the beam centroid model output with respect to the
                         measurement quads
         xs: the virtual scan inputs
         ys: the virtual scan outputs (beam centroid positions)

    NOTE: meas scans only needs to have 2 values for each device because it is expected that post_paths
            are produced from a SingleTaskGP with Linear ProductKernel (i.e. post_paths should have
            linear output for each dimension).
    """
    n_steps_meas_scan = 1 + len(meas_dims)
    n_tuning_configs = x_tuning.shape[0]

    # construct measurement scan inputs
    xs = torch.repeat_interleave(x_tuning, n_steps_meas_scan, dim=0)

    for i in range(len(meas_dims)):
        meas_dim = meas_dims[i]
        meas_scan = meas_scans[i]
        full_scan_column = meas_scan[0].repeat(n_steps_meas_scan, 1)
        full_scan_column[i + 1, 0] = meas_scan[1]
        full_scan_column_repeated = full_scan_column.repeat(n_tuning_configs, 1)

        xs = torch.cat(
            (xs[:, :meas_dim], full_scan_column_repeated, xs[:, meas_dim:]), dim=1
        )

    if samplewise:
        xs = xs.reshape(n_tuning_configs, n_steps_meas_scan, -1)

    ys = post_paths(xs)
    ys = ys.reshape(-1, n_steps_meas_scan)

    rise = ys[:, 1:] - ys[:, 0].reshape(-1, 1)
    run = (meas_scans[:, 1] - meas_scans[:, 0]).T.repeat(ys.shape[0], 1)
    slope = rise / run

    misalignment = slope.pow(2).sum(dim=1)

    if not samplewise:
        ys = ys.reshape(-1, n_tuning_configs, n_steps_meas_scan)
        misalignment = misalignment.reshape(-1, n_tuning_configs)

    return misalignment, xs, ys


def post_path_emit_squared_thick_quad(
    post_paths,
    scale_factor,
    q_len,
    distance,
    x_tuning,
    meas_dim,
    x_meas,
    samplewise=False,
):
    """
    A function that computes the emittance squared at locations in tuning-parameter space defined by x_tuning,
    from a set of pathwise posterior samples produced by a SingleTaskGP model of the beamsize squared
    with respect to some tuning devices and a measurement quadrupole.

    arguments:
        post_paths: a pathwise posterior sample from a SingleTaskGP model of the beam size
        scale_factor: (float) factor by which to multiply model measurement quadrupole inputs to get
                        geometric focusing strengths in [m^-2]
        q_len: (float) the longitudinal "thickness", or length, of the measurement quadrupole in [m]
        distance: (float) the distance (drift length) from the end of the measurement quadrupole
                    to the observation screen in [m]
        x_tuning: tensor of shape (n_points x n_tuning_dims) where each row defines a point
                    in tuning-parameter space at which to evaluate the emittance
        meas_dim: the index giving the input dimension of the measurement quadrupole in our GP model
        x_meas: a 1d tensor giving the measurement device inputs for the virtual measurement scans
        samplewise: boolean. Set to False if you want to evaluate the emittance for every point on
                        every sample. If set to True, the emittance for the nth sample (given by post_paths)
                        will only be evaluated at the nth point (given by x_tuning). If samplewise is set to
                        True, x_tuning must be shape n_samples x n_tuning_dims

    returns:
        emits_squared: a tensor containing the emittance squared results (which can be negative/invalid)
        is_valid: a tensor of booleans, of the same shape as emits_squared, designating whether or not
                    the corresponding entry of the emits_squared tensor is physically valid.
    """

    # get the number of points in the scan uniformly spaced along measurement domain
    n_steps_quad_scan = len(x_meas)

    # get the number of points in the tuning parameter space specified by x_tuning
    n_tuning_configs = x_tuning.shape[0]

    # get the complete tensor of inputs for a set of virtual quad measurement scans to be
    # performed at the locations in tuning parameter space specified by x_tuning
    xs = get_meas_scan_inputs_from_tuning_configs(meas_dim, x_tuning, x_meas)

    k_meas = x_meas * scale_factor

    if samplewise:
        # add assert n_tuning_configs == post_paths.n_samples
        xs = xs.reshape(n_tuning_configs, n_steps_quad_scan, -1)
        ys = post_paths(xs)  # ys will be nsamples x n_steps_quad_scan

        (
            sig, 
            is_valid
        ) = compute_emit_bmag_thick_quad(
            k_meas, ys, q_len, distance
        )[-2:]

        emits_squared = (sig[:,0,0]*sig[:,2,0] - sig[:,1,0]**2).reshape(-1,1)
    else:
        # ys will be shape n_samples x (n_tuning_configs*n_steps_quad_scan)
        ys = post_paths(xs)
        n_samples = ys.shape[0]

        # reshape into batchshape x n_steps_quad_scan
        ys = ys.reshape(n_samples * n_tuning_configs, n_steps_quad_scan)

        (
            sig, 
            is_valid
        ) = compute_emit_bmag_thick_quad(
            k_meas, ys, q_len, distance
        )[-2:]

        emits_squared = (sig[:,0,0]*sig[:,2,0] - sig[:,1,0]**2).reshape(-1,1)
        emits_squared = emits_squared.reshape(n_samples, -1)
        is_valid = is_valid.reshape(n_samples, -1)

        # emits_squared will be a tensor of
        # shape nsamples x n_tuning_configs, where n_tuning_configs
        # is the number of rows in the input tensor X.
        # The nth column of the mth row represents the emittance of the mth sample,
        # evaluated at the nth tuning config specified by the input tensor X.

    return emits_squared, is_valid


def sum_samplewise_emittance_xy_flat_input(
    post_paths_xy, # list length 2 with the pathwise sample functions from each of the x and y beam size models (in that order)
    scale_factor, 
    q_len, 
    distance, 
    x_tuning_flat, 
    meas_dim, 
    x_meas, 
):    
    x_tuning = x_tuning_flat.double().reshape(post_paths_xy[0].n_samples, -1)
    scale_factors = [scale_factor, -scale_factor]
    emit_sq_x, emit_sq_y = [post_path_emit_squared_thick_quad(post_paths,
                                                        scale_factor=sf,
                                                        q_len=q_len,
                                                        distance=distance,
                                                        x_tuning=x_tuning,
                                                        meas_dim=meas_dim,
                                                        x_meas=x_meas,
                                                        samplewise=True
                                                      )[0] for sf, post_paths in zip(scale_factors, post_paths_xy)]
    
    return torch.sum(emit_sq_x.abs().sqrt() * emit_sq_y.abs().sqrt())


def post_mean_emit_squared_thick_quad(
    model,
    scale_factor,
    q_len,
    distance,
    x_tuning,
    meas_dim,
    x_meas,
):
    """
    A function that computes the emittance squared at locations in tuning-parameter space defined by x_tuning,
    using the posterior mean of model.

    arguments:
        model: a SingleTaskGP model of the beamsize squared with respect to some tuning devices
                and a measurement quadrupole.
        scale_factor: (float) factor by which to multiply model measurement quadrupole inputs to get
                        geometric focusing strengths in [m^-2]
        q_len: (float) the longitudinal "thickness", or length, of the measurement quadrupole in [m]
        distance: (float) the distance (drift length) from the end of the measurement quadrupole
                    to the observation screen in [m]
        x_tuning: tensor of shape (n_points x n_tuning_dims) where each row defines a point
                    in tuning-parameter space at which to evaluate the emittance
        meas_dim: the index giving the input dimension of the measurement quadrupole in our GP model
        x_meas: a 1d tensor giving the measurement device inputs for the virtual measurement scans
        samplewise: boolean. Set to False if you want to evaluate the emittance for every point on
                        every sample. If set to True, the emittance for the nth sample (given by post_paths)
                        will only be evaluated at the nth point (given by x_tuning). If samplewise is set to
                        True, x_tuning must be shape n_samples x n_tuning_dims

    returns:
        emits_squared: a tensor containing the emittance squared results (which can be negative/invalid)
        is_valid: a tensor of booleans, of the same shape as emits_squared, designating whether or not
                    the corresponding entry of the emits_squared tensor is physically valid.
    """

    xs = get_meas_scan_inputs_from_tuning_configs(meas_dim, x_tuning, x_meas)
    ys = model.posterior(xs).mean

    ys_batch = ys.reshape(x_tuning.shape[0], -1)

    k_meas = x_meas * scale_factor

    (
        emit,
        bmag_min,
        sig,
        is_valid,
    ) = compute_emit_bmag_thick_quad(k_meas, ys_batch, q_len, distance)

    emit_squared = (sig[:,0,0]*sig[:,2,0] - sig[:,1,0]**2).reshape(-1,1)
    
    return emit_squared, is_valid


def get_meas_scan_inputs_from_tuning_configs(meas_dim, x_tuning, x_meas):
    """
    A function that generates the inputs for virtual emittance measurement scans at the tuning
    configurations specified by x_tuning.

    Parameters:
        meas_dim: int. the dimension index at which to insert the measurement device input scan values.
        x_tuning: a tensor of shape n_points x n_tuning_dims, where each row specifies a tuning
                    configuration where we want to do an emittance scan.
        x_meas: 1d tensor respresenting the measurement quad inputs for the virtual emittance scans.

    Returns:
        xs: tensor, shape (n_tuning_configs*n_steps_meas_scan) x d,
            where n_tuning_configs = x_tuning.shape[0],
            n_steps_meas_scan = len(x_meas),
            and d = x_tuning.shape[1] -- the number of tuning parameters

    """
    # each row of x_tuning defines a location in the tuning parameter space
    # along which to perform a quad scan and evaluate emit

    # x_meas must be shape (n,) and represent a 1d scan along the measurement domain

    # expand the x tensor to represent quad measurement scans
    # at the locations in tuning parameter space specified by X
    n_steps_meas_scan = len(x_meas)

    # get the number of points in the tuning parameter space specified by X
    n_tuning_configs = x_tuning.shape[0]

    # prepare column of measurement scans coordinates
    x_meas_repeated = x_meas.repeat(n_tuning_configs).reshape(
        n_steps_meas_scan * n_tuning_configs, 1
    )

    # repeat tuning configs as necessary and concat with column from the line above
    # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
    # where d is the full dimension of the model/posterior space (tuning & meas)
    xs_tuning = torch.repeat_interleave(x_tuning, n_steps_meas_scan, dim=0)
    xs = torch.cat(
        (xs_tuning[:, :meas_dim], x_meas_repeated, xs_tuning[:, meas_dim:]), dim=1
    )

    return xs


def compute_emit_bmag_thick_quad(k, y_batch, q_len, rmat_quad_to_screen, beta0=1., alpha0=0.):
    """
    A function that computes the emittance(s) corresponding to a set of quadrupole measurement scans
    using a thick quad model.

    Parameters:
        k: 1d torch tensor of shape (n_steps_quad_scan,)
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in the emittance scan

        y_batch: 2d torch tensor of shape (n_scans x n_steps_quad_scan),
                where each row represents the mean-square beamsize outputs in [m^2] of an emittance scan
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
        
        rmat_quad_to_screen: tensor containing the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.

        beta0: the design beta twiss parameter at the screen
        
        alpha0: the design alpha twiss parameter at the screen
        
    Returns:
        emit: shape (n_scans x 1) containing the geometric emittance fit results for each scan
        bmag_min: (n_scans x 1) containing the bmag corresponding to the optimal point for each scan
        sig: shape (n_scans x 3 x 1) containing column vectors of [sig11, sig12, sig22]
        is_valid: 1d tensor identifying physical validity of the emittance fit results
        
    SOURCE PAPER: http://www-library.desy.de/preparch/desy/thesis/desy-thesis-05-014.pdf
    """
    
    # construct the A matrix from eq. (3.2) & (3.3) of source paper
    quad_rmats = build_quad_rmat(k, q_len) # result shape (len(k) x 2 x 2)
    total_rmats = rmat_quad_to_screen.reshape(1,2,2) @ quad_rmats # result shape (len(k) x 2 x 2)
    
    amat = torch.tensor([]) # prepare the A matrix
    for rmat in total_rmats:
        r11, r12 = rmat[0,0], rmat[0,1]
        amat = torch.cat((amat, torch.tensor([[r11**2, 2.*r11*r12, r12**2]])), dim=0)
    # amat result shape (len(k) x 3)
    
    # get sigma matrix elements just before measurement quad from pseudo-inverse
    sig = amat.pinverse().unsqueeze(0) @ y_batch.unsqueeze(-1) # shapes (1 x 3 x len(k)) @ (n_scans x len(k) x 1)
    # result shape (n_scans x 3 x 1) containing column vectors of [sig11, sig12, sig22]
    
    # compute emit
    emit = torch.sqrt(sig[:,0,0]*sig[:,2,0] - sig[:,1,0]**2).reshape(-1,1) # result shape (n_scans x 1)

    # check sigma matrix and emit for physical validity
    is_valid = torch.logical_and(sig[:,0,0] > 0, sig[:,2,0] > 0) # result 1d tensor
    is_valid = torch.logical_and(is_valid, ~torch.isnan(emit.flatten())) # result 1d tensor
    
    # propagate beam parameters to screen
    twiss_at_screen = propagate_sig(sig, emit, total_rmats)[1]
    # result shape (n_scans x len(k) x 3 x 1)
    
    # get design gamma0 from design beta0, alpha0
    gamma0 = (1 + alpha0**2) / beta0
    
    # compute bmag
    bmag = 0.5 * (twiss_at_screen[:,:,0,0] * gamma0
                - 2 * twiss_at_screen[:,:,1,0] * alpha0
                + twiss_at_screen[:,:,2,0] * beta0
               )
    # result shape (n_scans, n_steps_quad_scan)
    
    # select minimum bmag from quad scan
    bmag_min, bmag_min_id = torch.min(bmag, dim=1, keepdim=True) # result shape (n_scans, 1) 
    
    return emit, bmag_min, sig, is_valid


def propagate_sig(sig_init, emit, rmat):
    temp = torch.tensor([[[1., 0., 0.],
                           [0., -1., 0.],
                           [0., 0., 1.]]]).double()
    twiss_init = (temp @ sig_init)/emit.unsqueeze(-1) # result shape (len(sig_init) x 3 x 1)
    
    twiss_transport = twiss_transport_mat_from_rmat(rmat) # result shape (len(rmat) x 3 x 3)

    twiss_final = twiss_transport.unsqueeze(0) @ twiss_init.unsqueeze(1)
    # result shape (len(sig_init) x len(rmat) x 3 x 1)

    sig_final = (temp.unsqueeze(0) @ twiss_final) * emit.reshape(-1,1,1,1) 
    # result shape (len(sig_init) x len(rmat) x 3 x 1)
    
    return sig_final, twiss_final


def twiss_transport_mat_from_rmat(rmat):
    rmat = rmat.reshape(-1,2,2)
    twiss_transport = torch.tensor([])
    for mat in rmat:
        c, s, cp, sp = mat[0,0], mat[0,1], mat[1,0], mat[1,1]

        twiss_transport = torch.cat((twiss_transport, torch.tensor([[[c**2, -2*c*s, s**2],
                                                                   [-c*cp, c*sp + cp*s, -s*sp],
                                                                   [cp**2, -2*cp*sp, sp**2]]]
                                                                ).double()
                                    ))
    return twiss_transport


def build_quad_rmat(k, q_len):
    # construct/collect quad R matrices
    rmat = torch.tensor([])
    for j in k:
        if j > 0:
            c, s, cp, sp = (
                            torch.cos(j.sqrt()*q_len), 
                            1./j.sqrt() * torch.sin(j.sqrt()*q_len),
                            -j.sqrt() * torch.sin(j.sqrt()*q_len), 
                            torch.cos(j.sqrt()*q_len)
                           )
#             c, s, cp, sp = (1., 0., -j*q_len, 1.)
        elif j < 0:
            c, s, cp, sp = (
                            torch.cosh(j.abs().sqrt()*q_len), 
                            1./j.abs().sqrt() * torch.sinh(j.abs().sqrt()*q_len),
                            j.abs().sqrt() * torch.sinh(j.abs().sqrt()*q_len), 
                            torch.cosh(j.abs().sqrt()*q_len)
                           )
#             c, s, cp, sp = (1., 0., j*q_len, 1.)
        elif j == 0:
            c, s, cp, sp = (1., q_len, 0., 1.)

        rmat = torch.cat((rmat, torch.tensor([[[c, s],
                                              [cp, sp]]]
                                            ).double()
                         ))
        
    return rmat


# +
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel
from gpytorch.priors import GammaPrior

def fit_gp_quad_scan(
    k,
    y,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    noise_prior=None,
    tkwargs=None,
):
    """
    A function that fits a GP model to an emittance beam size measurement quad scan
    and returns a set of "virtual scans" (functions sampled from the GP model posterior).
    The GP is fit to the BEAM SIZE SQUARED, and the virtual quad scans are NOT CHECKED 
    for physical validity. 
    
    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k
            
        covar_module: the covariance module to be used in fitting of the SingleTaskGP 
                    (modeling the function y**2 vs. k)
                    If None, uses ScaleKernel(MaternKernel()).

        tkwargs: dict containing the tensor device and dtype    

        n_samples: the number of virtual measurement scan samples to evaluate for our "Bayesian" estimate

        n_steps_quad_scan: the number of steps in our virtual measurement scans


    Returns:
        k_virtual: a 1d tensor representing the inputs for the virtual measurement scans.
                    All virtual scans are evaluated at the same set of input locations.

        bss: a tensor of shape (n_samples x n_steps_quad_scan) where each row repesents 
        the beam size squared results of a virtual quad scan evaluated at the points k_virtual.
    """
    
    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}
        
    k = torch.tensor(k, **tkwargs)
    y = torch.tensor(y, **tkwargs)

    if covar_module is None:
        covar_module = ScaleKernel(
            MaternKernel(), outputscale_prior=GammaPrior(2.0, 0.15)
        )
    if noise_prior is None:
        noise_prior = GammaPrior(10, 1)
    likelihood = GaussianLikelihood(noise_prior=noise_prior)
        
    model = SingleTaskGP(
        k.reshape(-1, 1),
        y.pow(2).reshape(-1, 1),
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=Normalize(1),
        outcome_transform=Standardize(1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    k_virtual = torch.linspace(k.min(), k.max(), n_steps_quad_scan, **tkwargs)

    p = model.posterior(k_virtual.reshape(-1, 1))
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)
    
    return k_virtual, bss


# -

def get_valid_emit_bmag_samples_from_quad_scan(
    k,
    y,
    q_len,
    rmat_quad_to_screen,
    beta0=1.,
    alpha0=0.,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    noise_prior=None,
    visualize=False,
    tkwargs=None,
):
    """
    A function that produces a distribution of possible (physically valid) emittance values corresponding
    to a single quadrupole measurement scan. Data is first modeled by a SingleTaskGP, virtual measurement
    scan samples are then drawn from the model posterior, the samples are modeled by thick-quad transport
    to obtain fits to the beam parameters, and physically invalid results are discarded.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
                    
        rmat_quad_to_screen: tensor containing the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.
                
        beta0: the design beta twiss parameter at the screen
        
        alpha0: the design alpha twiss parameter at the screen
        
        n_samples: the number of virtual measurement scan samples to evaluate for our "Bayesian" estimate

        n_steps_quad_scan: the number of steps in our virtual measurement scans

        covar_module: the covariance module to be used in fitting of the SingleTaskGP 
                    (modeling the function y**2 vs. k)
                    If None, uses ScaleKernel(MaternKernel()).

        visualize: boolean. Set to True to plot the parabolic fitting results.

        tkwargs: dict containing the tensor device and dtype

    Returns:
        emits_valid: a tensor of physically valid emittance results from sampled measurement scans.

        bmag_valid: (n_valid_scans x 1) containing the bmag corresponding to the optimal point 
                        from each physically valid fit.

        sig_valid: tensor, shape (n_valid_scans x 3 x 1), containing the computed 
                        sig11, sig12, sig22 corresponding to each physically valid
                        fit.

        sample_validity_rate: a float between 0 and 1 that describes the rate at which the samples
                        were physically valid/retained.
    """
    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k = torch.tensor(k, **tkwargs)
    y = torch.tensor(y, **tkwargs)

    k_virtual, bss = fit_gp_quad_scan(
        k=k,
        y=y,
        n_samples=n_samples,
        n_steps_quad_scan=n_steps_quad_scan,
        covar_module=covar_module,
        noise_prior=noise_prior,
        tkwargs=tkwargs
    )
    
    (emit, bmag, sig, is_valid) = compute_emit_bmag_thick_quad(k=k_virtual, 
                                                              y_batch=bss, 
                                                              q_len=q_len, 
                                                              rmat_quad_to_screen=rmat_quad_to_screen, 
                                                              beta0=beta0, 
                                                              alpha0=alpha0)

    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    # filter on physical validity
    cut_ids = torch.tensor(range(emit.shape[0]))[is_valid]
    emit_valid = torch.index_select(emit, dim=0, index=cut_ids)
    bmag_valid = torch.index_select(bmag, dim=0, index=cut_ids)
    sig_valid = torch.index_select(sig, dim=0, index=cut_ids)

    if visualize:
        plot_valid_thick_quad_fits(k=k, 
                                   y=y, 
                                   q_len=q_len, 
                                   distance=distance,
                                   emit=emit_valid, 
                                   bmag=bmag_valid,
                                   sig=sig_valid, 
                                   k_virtual=k_virtual,
                                   bss_virtual=bss,
                                  )
    return emit_valid, bmag_valid, sig_valid, sample_validity_rate


def plot_valid_thick_quad_fits(k, y, q_len, distance, emit, bmag, sig, ci=0.95, tkwargs=None, k_virtual=None, bss_virtual=None):
    """
    A function to plot the physically valid fit results
    produced by get_valid_emit_bmag_samples_from_quad_scan().

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        sig: tensor, shape (n_scans x 3 x 1), containing the computed sig11, sig12, sig22
                corresponding to each measurement scan
                
        emit: shape (n_scans x 1) containing the geometric emittance fit results for each scan

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        distance: the longitudinal distance (drift length) in [m] from the measurement
                    quadrupole to the observation screen
        
        (NOT IN USE)
        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.
                
        ci: "Confidence interval" for plotting upper/lower quantiles.

        tkwargs: dict containing the tensor device and dtype
    """
    from matplotlib import pyplot as plt

    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k_fit = torch.linspace(k.min(), k.max(), 10, **tkwargs)
    quad_rmats = build_quad_rmat(k_fit, q_len) # result shape (len(k_fit) x 2 x 2)
    rmat_quad_to_screen = build_quad_rmat(k=torch.tensor([0.]), q_len=distance)
    total_rmats = rmat_quad_to_screen.reshape(1,2,2) @ quad_rmats # result shape (len(k_fit) x 2 x 2)
    sig_final = propagate_sig(sig, emit, total_rmats)[0] # result shape len(sig) x len(k_fit) x 3 x 1
    bss_fit = sig_final[:,:,0,0]

    upper_quant = torch.quantile(bss_fit.sqrt(), q=0.5 + ci / 2.0, dim=0)
    lower_quant = torch.quantile(bss_fit.sqrt(), q=0.5 - ci / 2.0, dim=0)
    
    fig, axs = plt.subplots(3)
    fig.set_size_inches(5,9)
    
    ax=axs[0]
    fit = ax.fill_between(
        k_fit.detach().numpy(),
        lower_quant*1.e6,
        upper_quant*1.e6,
        alpha=0.3,
        label='"Bayesian" Thick-Quad Fits',
        zorder=1,
    )
    
    bss_upper = torch.quantile(bss_virtual, q=0.5 + ci / 2.0, dim=0).sqrt()
    bss_lower = torch.quantile(bss_virtual, q=0.5 - ci / 2.0, dim=0).sqrt()
    virtual_meas = ax.scatter(k_virtual.repeat(bss_virtual.shape[0],1).detach(),
                                        bss_virtual.sqrt()*1.e6,
                                        alpha=0.3,
                                        color='r',
                                        label='Virtual Measurements',
                                        zorder=0) 
    
    obs = ax.scatter(
        k, y*1.e6, marker="x", s=120, c="orange", label="Measurements", zorder=2
    )
    ax.set_title("Beam Size at Screen")
    ax.set_xlabel(r"Measurement Quad Geometric Focusing Strength ($[k]=m^{-2}$)")
    ax.set_ylabel(r"r.m.s. Beam Size")# ($[\sigma]=\mu m$)")
    ax.legend(handles=[obs, fit, virtual_meas])
    
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
    plt.show()
    plt.close()


def get_valid_geo_mean_emittance_samples_thick_quad(
    model,
    scale_factor,
    q_len,
    distance,
    x_tuning=None,
    domain=None,
    meas_dim=None,
    n_samples=10000,
    n_steps_quad_scan=10,
    visualize=False,
):
    """
    A function that takes a model of the beam size squared and produces a distribution of possible
    (physically valid) emittance values corresponding to a particular tuning configuration. Virtual
    measurement scan samples are drawn from the model posterior at the specified location in
    tuning-parameter space, unconstrained parabolic fits are performed on the virtual scan results,
    and physically invalid results are discarded.

    Parameters:
        model = ModelListGP trained on rms beam size squared [m^2] in both x and y

        scale_factor: (float) factor by which to multiply model measurement quadrupole inputs to get
                        geometric focusing strengths in [m^-2]

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        distance: the longitudinal distance (drift length) in [m] from the measurement
                    quadrupole to the observation screen

        x_tuning: a tensor of shape (1 x n_tuning_dims) that defines a point in
                    tuning-parameter space at which to perform virtual measurement scans and evaluate
                    the corresponding emittances

        domain: a tensor of shape (ndim x 2) containing the upper and lower bounds for the input devices

        meas_dim: integer that identifies the index of the measurement quadrupole dimension in the model

        n_samples: integer number of virtual measurement scans to perform at each tuning configuration
                    (Physically invalid results will be discarded.)

        n_steps_quad_scan: integer number of steps to use in the virtual measurement scans
                            (the virtual scans will span the entire measurement device domain)

    Returns:
         emits_at_target_valid: tensor containing the valid emittance results

         sample_validity_rate: a float between 0 and 1 that describes the rate at which the samples
                                were physically valid/retained
    """
    x_meas = torch.linspace(*domain[meas_dim], n_steps_quad_scan)
    xs_1d_scan = get_meas_scan_inputs_from_tuning_configs(meas_dim, x_tuning, x_meas)
    
    bss_model_x, bss_model_y = model.models
    
    # get geometric emittance in x
    p = bss_model_x.posterior(xs_1d_scan)
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)
    ks = x_meas * scale_factor
    (emit_x, bmag_min_x, sig_x, is_valid_x) = compute_emit_bmag_thick_quad(
        ks, bss, q_len, distance
    )
    
    # get geometric emittance in y
    p = bss_model_y.posterior(xs_1d_scan)
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)
    ks = x_meas * (-1.*scale_factor)
    (emit_y, bmag_min_y, sig_y, is_valid_y) = compute_emit_bmag_thick_quad(
        ks, bss, q_len, distance
    )
    
    geo_mean_emit = (emit_x * emit_y).sqrt()
    is_valid = torch.logical_and(is_valid_x, is_valid_y)
        
    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    cut_ids = torch.tensor(range(geo_mean_emit.shape[0]))[is_valid]
    geo_mean_emit_valid = torch.index_select(
        geo_mean_emit, dim=0, index=cut_ids
    )
    emit_x_valid = torch.index_select(emit_x, dim=0, index=cut_ids)
    emit_y_valid = torch.index_select(emit_y, dim=0, index=cut_ids)

    return geo_mean_emit_valid, emit_x_valid, emit_y_valid, sample_validity_rate


def get_quad_strength_conversion_factor(E=0.135, q_len=0.108):
    """
    Computes multiplicative factor to convert from LCLS quad PV values (model input space)
    in [kG] to the geometric focusing strengths in [m^-2].

    Parameters:
        E: Beam energy in [MeV]
        q_len: quadrupole length or "thickness" (longitudinal) in [m]

    Returns:
        conversion_factor: scale factor by which to multiply the LCLS quad PV values [kG] to get
                            the geometric focusing strengths [m^-2]
    Example:
    xs_quad = field integrals in [kG]
    E = beam energy in [GeV]
    q_len = quad thickness in [m]
    conversion_factor = get_quad_strength_conversion_factor(E, q_len)
    ks_quad = conversion_factor * xs_quad # results in the quadrupole geometric focusing strength
    """
    gamma = E / (0.511e-3)  # beam energy (GeV) divided by electron rest energy (GeV)
    beta = 1.0 - 1.0 / (2 * gamma**2)
    conversion_factor = 0.299 / (10.0 * q_len * beta * E)

    return conversion_factor


def normalize_emittance(emit, energy):
    gamma = energy / (0.511e-3)  # beam energy (GeV) divided by electron rest energy (GeV)
    beta = 1.0 - 1.0 / (2 * gamma**2)
    emit_n = gamma * beta * emit
    return emit_n


def plot_sample_optima_convergence_inputs(results, tuning_parameter_names=None, show_valid_only=True):
    ndim = results[1]["x_stars_all"].shape[1]
    niter = max(results.keys())
    nsamples = results[1]["x_stars_all"].shape[0]

    if tuning_parameter_names is None:
        tuning_parameter_names = ['tp_' + str(i) for i in range(ndim)]

    from matplotlib import pyplot as plt
    
    fig, axs = plt.subplots(ndim, 1)

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
                is_valid = results[key]["is_valid"]
                cut_ids = torch.tensor(range(nsamples))[is_valid]
            else:
                cut_ids = torch.tensor(range(nsamples))
            ax.scatter(torch.tensor([key]).repeat(len(cut_ids)), 
                       torch.index_select(results[key]["x_stars_all"], dim=0, index=cut_ids)[:,i],
                       c='C0')
    plt.tight_layout()


def plot_sample_optima_convergence_emits(results):
    niter = max(results.keys())
    nsamples = results[1]["emit_stars_all"].shape[0]
    
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1)

    ax.set_ylabel("$\epsilon$")
    ax.set_xlabel("iteration")
    ax.set_title("Sample Optima Distribution: Emittance")
    for key in results.keys():
        is_valid = results[key]["is_valid"]
        valid_ids = torch.tensor(range(nsamples))[is_valid]
        ax.scatter(torch.tensor([key]).repeat(torch.sum(is_valid)), 
                   torch.index_select(results[key]["emit_stars_all"], dim=0, index=valid_ids)[:,0].detach(), 
                   c='C0')
    plt.tight_layout()


def plot_valid_emit_prediction_at_x_tuning(model, 
                                           x_tuning, 
                                           scale_factor, 
                                           q_len, 
                                           distance, 
                                           bounds, 
                                           meas_dim, 
                                           n_samples, 
                                           n_steps_quad_scan):
    (
    emits_at_target_valid,
    sample_validity_rate,
    ) = get_valid_emittance_samples(
            model,
            scale_factor,
            q_len,
            distance,
            x_tuning,
            bounds.T,
            meas_dim,
            n_samples=n_samples,
            n_steps_quad_scan=n_steps_quad_scan,
        )
    
    from matplotlib import pyplot as plt

    plt.hist(emits_at_target_valid.flatten().cpu(), density=True)
    plt.xlabel('Predicted Optimal Emittance')
    plt.ylabel('Probability Density')
    plt.tight_layout()
    plt.show()
    print('sample validity rate:', sample_validity_rate)


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
    
    from matplotlib import pyplot as plt

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
