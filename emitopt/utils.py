import torch


def sum_samplewise_misalignment_flat_X(
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


def sum_samplewise_emittance_flat_X(
    post_paths, beam_energy, q_len, distance, x_tuning_flat, meas_dim, X_meas
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

    return torch.sum(
        (
            post_path_emit_squared(
                post_paths,
                beam_energy,
                q_len,
                distance,
                x_tuning,
                meas_dim,
                X_meas,
                samplewise=True,
            )[0]
        ).abs()
    )


def post_path_emit_squared(
    post_paths,
    beam_energy,
    q_len,
    distance,
    x_tuning,
    meas_dim,
    X_meas,
    samplewise=False,
    convert_quad_xs=True,
):
    """
    A function that computes the emittance squared at locations in tuning-parameter space defined by x_tuning,
    from a set of pathwise posterior samples produced by a SingleTaskGP model of the beamsize squared
    with respect to some tuning devices and a measurement quadrupole.

    arguments:
        post_paths: a pathwise posterior sample from a SingleTaskGP model of the beam size
        beam_energy: the beam energy in MeV
        q_len: the longitudinal "thickness", or length, of the measurement quadrupole
        distance: the distance (drift length) from the end of the measurement quadrupole
                    to the observation screen
        x_tuning: tensor of shape (n_points x n_tuning_dims) where each row defines a point
                    in tuning-parameter space at which to evaluate the emittance
        meas_dim: the index giving the input dimension of the measurement quadrupole in our GP model
        x_meas: a 1d tensor giving the measurement device inputs for the virtual measurement scans
        samplewise: boolean. Set to False if you want to evaluate the emittance for every point on
                        every sample. If set to True, the emittance for the nth sample (given by post_paths)
                        will only be evaluated at the nth point (given by x_tuning). If samplewise is set to
                        True, x_tuning must be shape n_samples x n_tuning_dims
        convert_quad_xs: boolean. Set to True if the model inputs along the measurement quadrupole dimension
                            are the LCLS field integrals given in kG. Set to False if the model inputs along
                            the measurement quad dimension are the geometric focusing strengths.

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

    if convert_quad_xs:
        k_meas = x_meas * get_quad_strength_conversion_factor(beam_energy, q_len)
    else:
        k_meas = x_meas

    if samplewise:
        # add assert n_tuning_configs == post_paths.n_samples
        xs = xs.reshape(n_tuning_configs, n_steps_quad_scan, -1)
        ys = post_paths(xs)  # ys will be nsamples x n_steps_quad_scan

        (
            emits,
            emits_squared,
            is_valid,
        ) = compute_emits(
            k_meas, ys, q_len, distance
        )[:3]

    else:
        # ys will be shape n_samples x (n_tuning_configs*n_steps_quad_scan)
        ys = post_paths(xs)

        n_samples = ys.shape[0]

        # reshape into batchshape x n_steps_quad_scan
        ys = ys.reshape(n_samples * n_tuning_configs, n_steps_quad_scan)

        (
            emits,
            emits_squared,
            is_valid,
        ) = compute_emits(
            k_meas, ys, q_len, distance
        )[:3]

        emits_squared = emits_squared.reshape(n_samples, -1)
        is_valid = is_valid.reshape(n_samples, -1)

        # emits_squared will be a tensor of
        # shape nsamples x n_tuning_configs, where n_tuning_configs
        # is the number of rows in the input tensor X.
        # The nth column of the mth row represents the emittance of the mth sample,
        # evaluated at the nth tuning config specified by the input tensor X.

    return emits_squared, is_valid


def post_mean_emit(
    model,
    beam_energy,
    q_len,
    distance,
    X_tuning,
    meas_dim,
    X_meas,
    squared=True,
    convert_quad_xs=True,
):
    xs = get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, X_meas)
    ys = model.posterior(xs).mean

    ys_batch = ys.reshape(X_tuning.shape[0], -1)

    if convert_quad_xs:
        k_meas = X_meas * get_quad_strength_conversion_factor(beam_energy, q_len)
    else:
        k_meas = X_meas

    (
        emits,
        emits_squared,
        is_valid,
    ) = compute_emits(
        k_meas, ys_batch, q_len, distance
    )[:3]

    if squared:
        out = emits_squared
    else:
        out = emits

    return out, is_valid


def get_meas_scan_inputs_from_tuning_configs(meas_dim, x_tuning, x_meas):
    """
    A function that generates the inputs for virtual emittance measurement scans at the tuning
    configurations specified by x_tuning.

    Parameters:
        meas_dim: int. the dimension index at which to insert the measurement device input scan values.
        x_tuning: a tensor of shape n_points x n_tuning_dims, where each row specifies a tuning
                    configuration where we want to do an emittance scan.
        x_meas: 1d tensor respresenting the measurement quad inputs for our emittance scans.
    """
    # each row of x_tuning defines a location in the tuning parameter space
    # along which to perform a quad scan and evaluate emit

    # x_meas must be shape (n,) and represent a 1d scan along the measurement domain

    # expand the x tensor to represent quad measurement scans
    # at the locations in tuning parameter space specified by X
    n_steps_meas_scan = len(x_meas)
    n_tuning_configs = x_tuning.shape[
        0
    ]  # the number of points in the tuning parameter space specified by X

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


def compute_emits(k, y_batch, q_len, distance):
    """
    k: 1d torch tensor of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

    y_batch: 2d torch tensor of shape (n_scans x n_steps_quad_scan),
            where each row represents the beamsize squared outputs in [m^2] of an emittance scan
            with inputs given by k

    q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

    distance: the longitudinal distance (drift length) in [m] from the measurement
                quadrupole to the observation screen

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

    # analytically calculate the Sigma (beam) matrices from parabola coefficients
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

    # compute emittances from sigma (beam) matrices
    emit_squared = torch.linalg.det(sigma)
    emit = torch.sqrt(emit_squared)

    # reshape results
    emit_squared = emit_squared.reshape(y_batch.shape[0], -1)
    emit = emit.reshape(y_batch.shape[0], -1)

    # transform the parabola fit coefficients into k-space
    abc_k_space = torch.cat(
        (
            abc[:, 0].reshape(-1, 1) * (distance * q_len) ** 2,
            abc[:, 1].reshape(-1, 1) * (distance * q_len),
            abc[:, 2].reshape(-1, 1),
        ),
        dim=1,
    )

    return emit, emit_squared, is_valid, abc_k_space, sigma


def compute_emit_from_single_beamsize_scan_numpy(
    k, y, q_len, distance, visualize=False, tkwargs=None
):
    """
    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y_batch: 1d numpy array of shape (n_steps_quad_scan, )
        representing the beamsize outputs in [m] of an emittance scan
        with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        distance: the longitudinal distance (drift length) in [m] from the measurement
        quadrupole to the observation screen

        visualize: boolean. Set to True to plot the parabolic fitting results.

        tkwargs: dict containing the tensor device and dtype

    Returns:

        emit: the computed emittance from a simple parabolic fit to each measurement scan
        (can be NaN if the parabolic fit is not physical)

        emit_squared: the computed emittance squared from parabolic fitting (can be negative
        if the fit is not physical)


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

    (emit, emit_squared, is_valid, abc, sigma) = compute_emits(
        k, y.pow(2).reshape(1, -1), q_len, distance
    )

    if visualize:
        plot_parabolic_fits(k, y, abc, tkwargs=tkwargs)

    return (
        emit.detach().numpy(),
        emit_squared.detach().numpy(),
        is_valid.detach().numpy(),
        abc.detach().numpy(),
        sigma.detach().numpy(),
    )


from botorch import fit_gpytorch_mll

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
    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the beamsize outputs in [m] of an emittance scan
            with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        distance: the longitudinal distance (drift length) in [m] from the measurement
                    quadrupole to the observation screen

        n_samples: the number of virtual measurement scan samples to evaluate for our "Bayesian" estimate

        n_steps_quad_scan: the number of steps in our virtual measurement scans

        covar_module: the covariance module to be used in fitting of the SingleTaskGP (modeling the function y vs. k)
                        If None, uses ScaleKernel(MaternKernel()).

        visualize: boolean. Set to True to plot the parabolic fitting results.

        tkwargs: dict containing the tensor device and dtype
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

    (emits, emits_sq, is_valid, abc_k_space, sigmas_all) = compute_emits(
        k_virtual, bss, q_len, distance
    )
    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    cut_ids = torch.tensor(range(emits_sq.shape[0]))[is_valid]
    emits_sq_valid = torch.index_select(emits_sq, dim=0, index=cut_ids)
    emits_valid = emits_sq_valid.sqrt()
    abc_valid = torch.index_select(abc_k_space, dim=0, index=cut_ids)

    if visualize:
        plot_parabolic_fits(k, y, abc_valid, ci=0.95, tkwargs=tkwargs)

    return emits_valid, emits_sq, is_valid, sample_validity_rate, sigmas_all


# -


def plot_parabolic_fits(k, y, abc, ci=0.95, tkwargs=None):
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
    beam_energy,
    q_len,
    distance,
    X_tuning=None,
    domain=None,
    meas_dim=None,
    n_samples=10000,
    n_steps_quad_scan=10,
    visualize=False,
):
    """
    model = SingleTaskGP trained on rms beam size squared [m^2]
    beam_energy [GeV]
    q_len [m]
    distance [m]
    """
    if X_tuning is None and model.train_inputs[0].shape[1] == 1:
        if model._has_transformed_inputs:
            low = model._original_train_inputs.min()
            hi = model._original_train_inputs.max()
        else:
            low = model.train_inputs[0].min()
            hi = model.train_inputs[0].max()
        x_meas = torch.linspace(low, hi, n_steps_quad_scan)
        xs_1d_scan = x_meas.reshape(-1, 1)
    else:
        x_meas = torch.linspace(*domain[meas_dim], n_steps_quad_scan)
        xs_1d_scan = get_meas_scan_inputs_from_tuning_configs(
            meas_dim, X_tuning, x_meas
        )

    p = model.posterior(xs_1d_scan)
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)

    conversion_factor = get_quad_strength_conversion_factor(beam_energy, q_len)
    ks = x_meas * conversion_factor
    (emits_at_target, emits_sq_at_target, is_valid, abc_k_space) = compute_emits(
        ks, bss, q_len, distance
    )[:4]
    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    cut_ids = torch.tensor(range(emits_sq_at_target.shape[0]))[is_valid]
    emits_sq_at_target_valid = torch.index_select(
        emits_sq_at_target, dim=0, index=cut_ids
    )
    emits_at_target_valid = emits_sq_at_target_valid.sqrt()

    if visualize:
        # only designed for beam size squared models with 1d input
        abc_input_space = torch.cat(
            (
                abc_k_space[:, 0].reshape(-1, 1) * (conversion_factor) ** 2,
                abc_k_space[:, 1].reshape(-1, 1) * (conversion_factor),
                abc_k_space[:, 2].reshape(-1, 1),
            ),
            dim=1,
        )
        abc_valid = torch.index_select(abc_input_space, dim=0, index=cut_ids)
        bss_valid = torch.index_select(bss, dim=0, index=cut_ids)
        import os

        from matplotlib import pyplot as plt

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        for y in bss_valid:
            (samples,) = plt.plot(
                x_meas, y, c="r", alpha=0.3, label="Posterior Scan Samples"
            )
        for abc in abc_valid:
            (fits,) = plt.plot(
                x_meas,
                abc[0] * x_meas**2 + abc[1] * x_meas + abc[2],
                c="C0",
                alpha=0.3,
                label="Parabolic Fits",
            )
        plt.scatter(
            model._original_train_inputs.flatten(),
            model.outcome_transform.untransform(model.train_targets)[0].flatten(),
        )
        plt.title("Emittance Measurement Scan Fits")
        plt.xlabel("Measurement PV values")
        plt.ylabel("Beam Size Squared")
        plt.legend(handles=[samples, fits])
        plt.show()
        plt.close()

    return emits_at_target_valid, emits_sq_at_target, is_valid, sample_validity_rate


def get_quad_strength_conversion_factor(E=0.135, q_len=0.108):
    """
    computes multiplicative factor to convert from quad PV values (model input space) to focusing strength
    Ex:
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
