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


def sum_samplewise_emittance_flat_x(
    post_paths, scale_factor, q_len, distance, x_tuning_flat, meas_dim, x_meas, positivity_constraint=True
):
    """
    A wrapper function that computes the sum of the samplewise emittances for more convenient
    minimization with scipy.

    arguments:
        Same as post_path_emit_squared() EXCEPT:

        x_tuning_flat: a FLATTENED tensor formerly of shape (n_samples x n_tuning_dims) where the nth
                        row defines a point in tuning-parameter space at which to evaluate the
                        emittance of the nth posterior pathwise sample given by post_paths


    NOTE: x_tuning_flat must be 1d (flattened) so the output of this function can be minimized
            with scipy minimization routines (that expect a 1d vector of inputs)
    NOTE: samplewise is set to True to avoid unncessary computation during simultaneous minimization
            of the pathwise emittance.
    NOTE: the absolute value of the emittance squared is taken to reduce the chance of the scipy
            minimizer finding a non-physical solution.
    """
    x_tuning = x_tuning_flat.double().reshape(post_paths.n_samples, -1)

    emits_squared = post_path_emit_squared(
                post_paths,
                scale_factor,
                q_len,
                distance,
                x_tuning,
                meas_dim,
                x_meas,
                samplewise=True,
            )[0]

    if positivity_constraint:
        return torch.sum(emits_squared.abs())
    else:
        return torch.sum(emits_squared)


def post_path_emit_squared(
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
            emits_squared,
            is_valid,
        ) = compute_emits(
            k_meas, ys, q_len, distance
        )[:2]

    else:
        # ys will be shape n_samples x (n_tuning_configs*n_steps_quad_scan)
        ys = post_paths(xs)

        n_samples = ys.shape[0]

        # reshape into batchshape x n_steps_quad_scan
        ys = ys.reshape(n_samples * n_tuning_configs, n_steps_quad_scan)

        (
            emits_squared,
            is_valid,
        ) = compute_emits(
            k_meas, ys, q_len, distance
        )[:2]

        emits_squared = emits_squared.reshape(n_samples, -1)
        is_valid = is_valid.reshape(n_samples, -1)

        # emits_squared will be a tensor of
        # shape nsamples x n_tuning_configs, where n_tuning_configs
        # is the number of rows in the input tensor X.
        # The nth column of the mth row represents the emittance of the mth sample,
        # evaluated at the nth tuning config specified by the input tensor X.

    return emits_squared, is_valid


def post_mean_emit_squared(
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
        emit_squared,
        is_valid,
        abc_k_space,
        sig,
    ) = compute_emits(k_meas, ys_batch, q_len, distance)

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


def compute_emits(k, y_batch, q_len, distance, ):
    """
    A function that computes the emittance(s) corresponding to a set of quadrupole measurement scans.

    Parameters:
        k: 1d torch tensor of shape (n_steps_quad_scan,)
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in the emittance scan

        y_batch: 2d torch tensor of shape (n_scans x n_steps_quad_scan),
                where each row represents the beamsize squared outputs in [m^2] of an emittance scan
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        distance: the longitudinal distance (drift length) in [m] from the measurement
                    quadrupole to the observation screen

    Returns:
        emits_squared: a tensor containing the emittance squared results (which can be negative/invalid)

        is_valid: a tensor of booleans, of the same shape as emits_squared, designating whether or not
                    the corresponding entry of the emits_squared tensor is physically valid.

        abc_k_space: tensor, shape (n_scans x 3), containing parabola fit coefficients
                        in k-space (geometric focusing strength)

        sig: tensor, shape (n_scans x 3 x 1), containing the computed sig11, sig12, sig22
                corresponding to each measurement scan

    NOTE: every measurement scan is assumed to have been evaluated
            at the single set of measurement param inputs described by k

    NOTE: geometric configuration for LCLS OTR2 emittance/quad measurement scan
            q_len = 0.108  # measurement quad thickness
            distance = 2.26  # drift length from measurement quad to observation screen
    """

    device = k.device

    k = k.reshape(-1, 1)

    # transform inputs to make fitting more convenient
    x_fit = k * distance * q_len

    # pseudo inverse method to calculate parabola coefficients
    a_block = torch.cat(
        (
            x_fit**2,
            x_fit,
            torch.tensor([1], device=device).repeat(len(x_fit)).reshape(x_fit.shape),
        ),
        dim=1,
    )
    b = y_batch.double()

    # compute the parabola coefficients a, b, c (i.e. ax^2 +bx + c)
    # note that these coefficients are for the transformed fit variable x_fit
    abc = a_block.pinverse().repeat(*y_batch.shape[:-1], 1, 1).double() @ b.reshape(
        *b.shape, 1
    )
    abc = abc.reshape(*abc.shape[:-1])

    # check for physical validity of parabolas (concave up and positive vertex)
    is_valid = torch.logical_and(
        abc[:, 0] > 0, (abc[:, 2] > abc[:, 1] ** 2 / (4.0 * abc[:, 0]))
    )

    # analytically calculate the sigma (beam) matrices from parabola coefficients
    # (non-physical results are possible)
    m = torch.tensor(
        [
            [1, 0, 0],
            [-1 / distance, 1 / (2 * distance), 0],
            [1 / (distance**2), -1 / (distance**2), 1 / (distance**2)],
        ],
        device=device,
    )

    # get a tensor of column vectors of sig11, sig12, sig22
    sig = torch.matmul(
        m.repeat(*abc.shape[:-1], 1, 1).double(),
        abc.reshape(*abc.shape[:-1], 3, 1).double(),
    )

    # transform into 2x2 sigma/covar beam matrices
    sigma = (
        sig.reshape(-1, 3)
        .repeat_interleave(torch.tensor([1, 2, 1], device=device), dim=1)
        .reshape(*sig.shape[:-2], 2, 2)
    )

    # compute emittances (squared) from sigma (beam) matrices
    emit_squared = torch.linalg.det(sigma)

    # reshape results
    emit_squared = emit_squared.reshape(y_batch.shape[0], -1)

    # transform the parabola fit coefficients into k-space
    abc_k_space = torch.cat(
        (
            abc[:, 0].reshape(-1, 1) * (distance * q_len) ** 2,
            abc[:, 1].reshape(-1, 1) * (distance * q_len),
            abc[:, 2].reshape(-1, 1),
        ),
        dim=1,
    )

    return emit_squared, is_valid, abc_k_space, sig


def bmag_from_emittance_fit(k, q_len, d, sig, beta0=1., alpha0=0.):
    '''
    Parameters:

        k: 1d tensor of shape (n_steps_quad_scan,)
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in the emittance scan
            
        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        d: the longitudinal distance (drift length) in [m] from the measurement
                    quadrupole to the observation screen
                    
        sig: tensor, shape (n_scans x 3 x 1), containing the computed sig11, sig12, sig22
                corresponding to each measurement scan
                
        beta0: the design beta twiss parameter at the screen
        
        alpha0: the design alpha twiss parameter at the screen
        
    Returns: 
        
        best_bmag_val: 1d tensor shape (n_scans,)
    
    '''
    # get twiss (before quad) from sig (also before quad)
    emits = torch.sqrt(sig[:,0,0]*sig[:,2,0] - sig[:,1,0]**2).reshape(-1,1,1)
    temp = torch.tensor([[[1., 0., 0.],
                       [0., -1., 0.],
                       [0., 0., 1.]]]).double()
    twiss_before_quad = (temp @ sig)/emits # will be shape (n_scans x 3 x 1)
    
    # construct drift space transport matrix
    drift_transport = torch.tensor([[[1., -2*d, d**2],
                                    [0., 1., -d],
                                    [0., 0., 1.]]]).double()
    
    # construct/collect quad transport matrices
    quad_transport = torch.tensor([])
    for j in k:
        if j > 0:
            c, s, cp, sp = (
                            torch.cos(j.sqrt()*q_len), 
                            1./j.sqrt() * torch.sin(j.sqrt()*q_len),
                            -j.sqrt() * torch.sin(j.sqrt()*q_len), 
                            torch.cos(j.sqrt()*q_len)
                           )
        elif j < 0:
            c, s, cp, sp = (
                            torch.cosh(j.abs().sqrt()*q_len), 
                            1./j.abs().sqrt() * torch.sinh(j.abs().sqrt()*q_len),
                            -(j.abs().sqrt()) * torch.sinh(j.abs().sqrt()*q_len), 
                            torch.cosh(j.abs().sqrt()*q_len)
                           )
        elif j == 0:
            c, s, cp, sp = (1., q_len, 0., 1.)

        quad_transport = torch.cat((quad_transport, torch.tensor([[[c**2, -2*c*s, s**2],
                                                                   [-c*cp, c*sp + cp*s, -s*sp],
                                                                   [cp**2, -2*cp*sp, sp**2]]]
                                                                ).double()
                                ))
    quad_transport = quad_transport.unsqueeze(0)
    
    # transport twiss through quad
    twiss_after_quad = quad_transport @ twiss_before_quad.reshape(-1,1,3,1) # shapes (1, n_steps_quad_scan, 3, 3) // (n_scans, 1, 3, 1)
    # result shape (n_scans, n_steps_quad_scan, 3, 1)
    
    # transport twiss to screen
    twiss_at_screen = drift_transport @ twiss_after_quad # shapes (1, 3, 3) // (n_scans, n_steps_quad_scan, 3, 1)
    # result shape (n_scans, n_steps_quad_scan, 3, 1)
    
    # get design gamma0 from design beta0, alpha0
    gamma0 = (1 + alpha0**2) / beta0
    
    # calculate bmag
    bmag = 0.5 * (twiss_at_screen[:,:,0,0] * gamma0
                    - 2 * twiss_at_screen[:,:,1,0] * alpha0
                    + twiss_at_screen[:,:,2,0] * beta0
                   )
    # result shape (n_scans, n_steps_quad_scan)
    
    # select minimum bmag from quad scan
    bmag_min = torch.min(bmag, dim=1, keepdim=True) # result shape (n_scans, 1) 
    
    best_bmag_val = bmag_min[0]
    
    return best_bmag_val


def compute_emit_from_single_beamsize_scan_numpy(
    k, y, q_len, distance, visualize=False, tkwargs=None
):
    """
    A function that computes the emittance corresponding to a single quadrupole measurement scan
    by performing an unconstrained parabolic fit to the data.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the beam size measurements (NOT SQUARED) in [m] of an emittance scan
            with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        distance: the longitudinal distance (drift length) in [m] from the measurement
        quadrupole to the observation screen

        visualize: boolean. Set to True to plot the parabolic fitting results.

        tkwargs: dict containing the tensor device and dtype

    Returns:

        emit: the computed emittance from a simple parabolic fit to each measurement scan
        (can be NaN if the parabolic fit is not physical)

        sig: tensor, shape (n_scans x 3 x 1), containing the computed sig11, sig12, sig22
                corresponding to each measurement scan

    NOTE: every measurement scan is assumed to have been evaluated
    at the single set of measurement param inputs described by k

    NOTE: geometric configuration for LCLS OTR2 emittance/quad measurement scan
        q_len = 0.108  # measurement quad thickness
        distance = 2.26  # drift length from measurement quad to observation screen
    """
    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k = torch.tensor(k, **tkwargs)
    y = torch.tensor(y, **tkwargs)

    (emit_squared, is_valid, abc, sig) = compute_emits(
        k, y.pow(2).reshape(1, -1), q_len, distance
    )
    emit = emit_squared.sqrt()

    if visualize:
        plot_parabolic_fits(k, y, abc, tkwargs=tkwargs)

    return (
        emit.detach().numpy(),
        is_valid.detach().numpy(),
        abc.detach().numpy(),
        sig.detach().numpy(),
    )


from botorch import fit_gpytorch_mll


def fit_gp_quad_scan(
    k,
    y,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
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
            representing the beam size measurements (NOT SQUARED) in [m] of an emittance scan
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

    model = SingleTaskGP(
        k.reshape(-1, 1),
        y.pow(2).reshape(-1, 1),
        covar_module=covar_module,
        input_transform=Normalize(1),
        outcome_transform=Standardize(1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    k_virtual = torch.linspace(k.min(), k.max(), n_steps_quad_scan, **tkwargs)

    p = model.posterior(k_virtual.reshape(-1, 1))
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)
    
    return k_virtual, bss


# +
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel
from gpytorch.priors import GammaPrior


def get_valid_emit_samples_from_quad_scan(
    k,
    y,
    q_len,
    distance,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    visualize=False,
    tkwargs=None,
):
    """
    A function that produces a distribution of possible (physically valid) emittance values corresponding
    to a single quadrupole measurement scan. Data is first modeled by a SingleTaskGP, virtual measurement
    scan samples are then drawn from the model posterior, unconstrained parabolic fits are performed on
    the virtual scan results, and physically invalid results are discarded.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the beam size measurements (NOT SQUARED) in [m] of an emittance scan
            with inputs given by k
            
        covar_module: the covariance module to be used in fitting of the SingleTaskGP 
                    (modeling the function y**2 vs. k)
                    If None, uses ScaleKernel(MaternKernel()).

        visualize: boolean. Set to True to plot the parabolic fitting results.

        tkwargs: dict containing the tensor device and dtype

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        distance: the longitudinal distance (drift length) in [m] from the measurement
                    quadrupole to the observation screen

        n_samples: the number of virtual measurement scan samples to evaluate for our "Bayesian" estimate

        n_steps_quad_scan: the number of steps in our virtual measurement scans


    Returns:
        emits_valid: a tensor of physically valid emittance results from sampled measurement scans.

        sig_valid: tensor, shape (n_valid_scans x 3 x 1), containing the computed 
                        sig11, sig12, sig22 corresponding to each physically valid
                        measurement scan
                        
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
        tkwargs=tkwargs
    )
    
    (emits_sq, is_valid, abc_k_space, sig) = compute_emits(
        k_virtual, bss, q_len, distance
    )
    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    cut_ids = torch.tensor(range(emits_sq.shape[0]))[is_valid]
    emits_sq_valid = torch.index_select(emits_sq, dim=0, index=cut_ids)
    emits_valid = emits_sq_valid.sqrt()
    abc_valid = torch.index_select(abc_k_space, dim=0, index=cut_ids)
    sig_valid = torch.index_select(sig, dim=0, index=cut_ids)

    if visualize:
        plot_parabolic_fits(k, y, abc_valid, ci=0.95, tkwargs=tkwargs)

    return emits_valid, sig_valid, sample_validity_rate


# -


def plot_parabolic_fits(k, y, abc, ci=0.95, tkwargs=None):
    """
    A function to plot the parabolic fits produced as a necessary step in the compute_emits() function.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the beam size measurements (NOT SQUARED) in [m] of an emittance scan
            with inputs given by k

        abc: tensor, shape (n x 3), containing n sets of parabola fit coefficients

        ci: "Confidence interval" for plotting upper/lower quantiles.

        tkwargs: dict containing the tensor device and dtype
    """
    from matplotlib import pyplot as plt

    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k_fit = torch.linspace(k.min(), k.max(), 10, **tkwargs)
    bss_fit = abc @ torch.cat(
        (
            k_fit.pow(2).reshape(1, -1),
            k_fit.pow(1).reshape(1, -1),
            torch.ones_like(k_fit).reshape(1, -1),
        ),
        dim=0,
    )

    if abc.shape[0] > 1:
        title = "Validated Parabolic Fits"
        upper_quant = torch.quantile(bss_fit, q=0.5 + ci / 2.0, dim=0)
        lower_quant = torch.quantile(bss_fit, q=0.5 - ci / 2.0, dim=0)
        fit = plt.fill_between(
            k_fit.detach().numpy(),
            lower_quant,
            upper_quant,
            alpha=0.3,
            label='"Bayesian" Fit',
            zorder=1,
        )
    else:
        title = "Unconstrained Parabolic Fit"
        (fit,) = plt.plot(
            k_fit.detach().numpy(), bss_fit.flatten(), label="Fit", zorder=1
        )
    obs = plt.scatter(
        k, y.pow(2), marker="x", s=120, c="orange", label="Measurements", zorder=2
    )
    plt.title(title)
    plt.xlabel("Measurement Quad Geometric Focusing Strength (k)")
    plt.ylabel("Beam Size Squared")
    plt.legend(handles=[obs, fit])
    plt.show()
    plt.close()


def get_valid_emittance_samples(
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
        model = SingleTaskGP trained on rms beam size squared [m^2]

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

    p = model.posterior(xs_1d_scan)
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)

    ks = x_meas * scale_factor
    (emits_sq_at_target, is_valid, abc_k_space, sig) = compute_emits(
        ks, bss, q_len, distance
    )
    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    cut_ids = torch.tensor(range(emits_sq_at_target.shape[0]))[is_valid]
    emits_sq_at_target_valid = torch.index_select(
        emits_sq_at_target, dim=0, index=cut_ids
    )
    emits_at_target_valid = emits_sq_at_target_valid.sqrt()

    return emits_at_target_valid, sample_validity_rate


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
