import torch


def sum_samplewise_misalignment_flat_X(post_paths, X_tuning_flat, meas_dims, meas_scans):
        
    X_tuning = X_tuning_flat.double().reshape(post_paths.n_samples, -1)
        
    return torch.sum(
            post_path_misalignment(
                post_paths, X_tuning, meas_dims, meas_scans, samplewise=True
            )[0]
    )


def post_path_misalignment(post_paths,
                       X_tuning, #n x d tensor 
                       meas_dims, #list of integers
                       meas_scans, # tensor of measurement device(s) scan inputs, shape: len(meas_dims) x 2
                       samplewise=False
                       ):
    
    n_steps_meas_scan = 1 + len(meas_dims)
    n_tuning_configs = X_tuning.shape[0]

    #construct measurement scan inputs
    xs = torch.repeat_interleave(X_tuning, n_steps_meas_scan, dim=0)
        
    for i in range(len(meas_dims)):
        meas_dim = meas_dims[i]
        meas_scan = meas_scans[i]
        full_scan_column = meas_scan[0].repeat(n_steps_meas_scan,1)
        full_scan_column[i+1, 0] = meas_scan[1]
        full_scan_column_repeated = full_scan_column.repeat(n_tuning_configs,1)


        xs = torch.cat(
            (xs[:, :meas_dim], full_scan_column_repeated, xs[:, meas_dim:]), dim=1
        )
        
    if samplewise:
        xs = xs.reshape(n_tuning_configs, n_steps_meas_scan, -1)

    ys = post_paths(xs)
    ys = ys.reshape(-1, n_steps_meas_scan)
    
    rise = (ys[:,1:] - ys[:,0].reshape(-1,1))
    run = (meas_scans[:,1] - meas_scans[:,0]).T.repeat(ys.shape[0], 1)
    slope = rise/run
    
    misalignment = slope.pow(2).sum(dim=1)
    
    if not samplewise:
        ys = ys.reshape(-1, n_tuning_configs, n_steps_meas_scan)
        misalignment = misalignment.reshape(-1, n_tuning_configs)
        
    return misalignment, xs, ys


def sum_samplewise_emittance_flat_X(post_paths, beam_energy, q_len, distance, X_tuning_flat, meas_dim, X_meas):
    
    X_tuning = X_tuning_flat.double().reshape(post_paths.n_samples, -1)

    return torch.sum(
        (
            post_path_emit(
                post_paths, beam_energy, q_len, distance, X_tuning, meas_dim, X_meas, samplewise=True, squared=True
            )[0]
        ).abs()
    )


def post_path_emit(
    post_paths, 
    beam_energy, 
    q_len, 
    distance,
    X_tuning,
    meas_dim,
    X_meas,
    samplewise=False,
    squared=True,
    convert_quad_xs=True,
):
    # each row of X_tuning defines a location in the tuning parameter space,
    # along which to perform a quad scan and evaluate emit

    # X_meas must be shape (n,) and represent a 1d scan along the measurement domain

    # if samplewise=False, X should be shape: n_tuning_configs x (ndim-1)
    # the emittance for every point specified by X will be evaluated
    # for every posterior sample path (broadcasting).

    # if samplewise=True, X must be shape: nsamples x (ndim-1)
    # the emittance of the nth sample will be computed for the nth input given by X

    # expand the X tensor to represent quad measurement scans
    # at the locations in tuning parameter space specified by X
    n_steps_quad_scan = len(
        X_meas
    )  # number of points in the scan uniformly spaced along measurement domain
    n_tuning_configs = X_tuning.shape[
        0
    ]  # the number of points in the tuning parameter space specified by X

    xs = get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, X_meas)

    if convert_quad_xs:
        k_meas = X_meas * get_quad_strength_conversion_factor(beam_energy, q_len)
    else:
        k_meas = X_meas

    if samplewise:
        # add assert n_tuning_configs == post_paths.n_samples
        xs = xs.reshape(n_tuning_configs, n_steps_quad_scan, -1)
        ys = post_paths(xs)  # ys will be nsamples x n_steps_quad_scan

        (
            emits,
            emits_squared,
            is_valid,
        ) = compute_emits_from_batched_beamsize_scans(k_meas, ys, q_len, distance)[:3]

    else:
        ys = post_paths(
            xs
        )  # ys will be shape n_samples x (n_tuning_configs*n_steps_quad_scan)

        n_samples = ys.shape[0]

        ys = ys.reshape(
            n_samples * n_tuning_configs, n_steps_quad_scan
        )  # reshape into batchshape x n_steps_quad_scan

        (
            emits,
            emits_squared,
            is_valid,
        ) = compute_emits_from_batched_beamsize_scans(k_meas, ys, q_len, distance)[:3]

        emits = emits.reshape(n_samples, -1)
        emits_squared = emits_squared.reshape(n_samples, -1)
        is_valid = is_valid.reshape(n_samples, -1)

        # emits_flat, emits_squared_raw_flat will be tensors of
        # shape nsamples x n_tuning_configs, where n_tuning_configs
        # is the number of rows in the input tensor X.
        # The nth column of the mth row represents the emittance of the mth sample,
        # evaluated at the nth tuning config specified by the input tensor X.

    if squared:
        out = emits_squared
    else:
        out = emits

    return out, is_valid


def post_mean_emit(
    model, beam_energy, q_len, distance, X_tuning, meas_dim, X_meas, squared=True, convert_quad_xs=True
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
    ) = compute_emits_from_batched_beamsize_scans(
        k_meas, ys_batch, q_len, distance
    )[:3]

    if squared:
        out = emits_squared
    else:
        out = emits

    return out, is_valid


def get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, X_meas):
    # each row of X_tuning defines a location in the tuning parameter space
    # along which to perform a quad scan and evaluate emit

    # X_meas must be shape (n,) and represent a 1d scan along the measurement domain

    # expand the X tensor to represent quad measurement scans
    # at the locations in tuning parameter space specified by X
    n_steps_meas_scan = len(X_meas)
    n_tuning_configs = X_tuning.shape[
        0
    ]  # the number of points in the tuning parameter space specified by X

    # prepare column of measurement scans coordinates
    X_meas_repeated = X_meas.repeat(n_tuning_configs).reshape(
        n_steps_meas_scan * n_tuning_configs, 1
    )

    # repeat tuning configs as necessary and concat with column from the line above
    # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
    # where d is the full dimension of the model/posterior space (tuning & meas)
    xs_tuning = torch.repeat_interleave(X_tuning, n_steps_meas_scan, dim=0)
    xs = torch.cat(
        (xs_tuning[:, :meas_dim], X_meas_repeated, xs_tuning[:, meas_dim:]), dim=1
    )

    return xs


def compute_emits_from_batched_beamsize_scans(ks_meas, ys_batch, q_len, distance):
    '''    
    xs_meas is assumed to be a 1d tensor of shape (n_steps_quad_scan,)
    representing the measurement parameter inputs of the emittance scan

    ys_batch is assumed to be shape n_scans x n_steps_quad_scan,
    where each row represents the beamsize outputs of an emittance scan
    with input given by xs_meas

    note that every measurement scan is assumed to have been evaluated
    at the single set of measurement param inputs described by xs_meas

    geometric configuration for LCLS OTR2 emittance/quad measurement scan
    q_len = 0.108  # measurement quad thickness
    distance = 2.26  # drift length from measurement quad to observation screen
    '''


    device = ks_meas.device

    ks_meas = ks_meas.reshape(-1, 1)
    xs_meas = ks_meas * distance * q_len

    # least squares method to calculate parabola coefficients
    A_block = torch.cat(
        (
            xs_meas**2,
            xs_meas,
            torch.tensor([1], device=device)
            .repeat(len(xs_meas))
            .reshape(xs_meas.shape),
        ),
        dim=1,
    )
    B = ys_batch.double()

    abc = A_block.pinverse().repeat(*ys_batch.shape[:-1], 1, 1).double() @ B.reshape(
        *B.shape, 1
    )
    abc = abc.reshape(*abc.shape[:-1])
    is_valid = torch.logical_and(
        abc[:, 0] > 0, (abc[:, 2] > abc[:, 1] ** 2 / (4.0 * abc[:, 0]))
    )

    # analytically calculate the Sigma (beam) matrices from parabola coefficients
    # (non-physical results are possible)
    M = torch.tensor(
        [
            [1, 0, 0],
            [-1 / distance, 1 / (2 * distance), 0],
            [1 / (distance**2), -1 / (distance**2), 1 / (distance**2)],
        ],
        device=device,
    )

    sigs = torch.matmul(
        M.repeat(*abc.shape[:-1], 1, 1).double(),
        abc.reshape(*abc.shape[:-1], 3, 1).double(),
    )  # column vectors of sig11, sig12, sig22

    Sigmas = (
        sigs.reshape(-1, 3)
        .repeat_interleave(torch.tensor([1, 2, 1], device=device), dim=1)
        .reshape(*sigs.shape[:-2], 2, 2)
    )  # 2x2 Sigma/covar beam matrix

    # compute emittances from Sigma (beam) matrices
    emits_squared_raw = torch.linalg.det(Sigmas)

    emits = torch.sqrt(
        emits_squared_raw
    )  # these are the emittances for every tuning parameter combination.

    emits_squared_raw = emits_squared_raw.reshape(ys_batch.shape[0], -1)
    emits = emits.reshape(ys_batch.shape[0], -1)

    abc_k_space = torch.cat((abc[:,0].reshape(-1,1)*(distance*q_len)**2, 
                             abc[:,1].reshape(-1,1)*(distance*q_len), 
                             abc[:,2].reshape(-1,1)), 
                            dim=1)
    
    return emits, emits_squared_raw, is_valid, abc_k_space


def get_valid_emittance_samples(
    model, beam_energy, q_len, distance, X_tuning=None, domain=None, meas_dim=None, n_samples=10000, n_steps_quad_scan=10, visualize=False
):
    '''
    model = SingleTaskGP trained on rms beam size squared [m^2] 
    beam_energy [GeV]
    q_len [m]
    distance [m]
    '''
    if X_tuning is None and model.train_inputs[0].shape[1] == 1:
        if model._has_transformed_inputs:
            low = model._original_train_inputs.min()
            hi = model._original_train_inputs.max()
        else:
            low = model.train_inputs[0].min()
            hi = model.train_inputs[0].max()
        x_meas = torch.linspace(low, hi, n_steps_quad_scan)
        xs_1d_scan = x_meas.reshape(-1,1)
    else:
        x_meas = torch.linspace(*domain[meas_dim], n_steps_quad_scan)
        xs_1d_scan = get_meas_scan_inputs_from_tuning_configs(meas_dim, X_tuning, x_meas)

    p = model.posterior(xs_1d_scan)
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)

    conversion_factor = get_quad_strength_conversion_factor(beam_energy, q_len)
    ks = x_meas * conversion_factor
    (
        emits_at_target,
        emits_sq_at_target,
        is_valid,
        abc_k_space
    ) = compute_emits_from_batched_beamsize_scans(ks, bss, q_len, distance)
    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    cut_ids = torch.tensor(range(emits_sq_at_target.shape[0]))[is_valid]
    emits_sq_at_target_valid = torch.index_select(
        emits_sq_at_target, dim=0, index=cut_ids
    )
    emits_at_target_valid = emits_sq_at_target_valid.sqrt()
    

    if visualize:
        #only designed for beam size squared models with 1d input
        abc_input_space = torch.cat((abc_k_space[:,0].reshape(-1,1)*(conversion_factor)**2, 
                                     abc_k_space[:,1].reshape(-1,1)*(conversion_factor), 
                                     abc_k_space[:,2].reshape(-1,1)), 
                                    dim=1)
        abc_valid = torch.index_select(
            abc_input_space, dim=0, index=cut_ids
        )
        bss_valid = torch.index_select(
            bss, dim=0, index=cut_ids
        )
        from matplotlib import pyplot as plt
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        for y in bss_valid:
            samples, = plt.plot(x_meas, y, c='r', alpha=0.3, label='Posterior Scan Samples')
        for abc in abc_valid:
            fits, = plt.plot(x_meas, abc[0]*x_meas**2 + abc[1]*x_meas + abc[2], c='C0', alpha=0.3, label='Parabolic Fits')
        plt.scatter(model._original_train_inputs.flatten(),
                    model.outcome_transform.untransform(model.train_targets)[0].flatten())
        plt.title('Emittance Measurement Scan Fits')
        plt.xlabel('Measurement PV values')
        plt.ylabel('Beam Size Squared')
        plt.legend(handles=[samples, fits])
        plt.show()
        plt.close()
        
    return emits_at_target_valid, emits_sq_at_target, is_valid, sample_validity_rate


def get_quad_strength_conversion_factor(E=0.135, q_len=0.108):
    '''
    computes multiplicative factor to convert from quad PV values (model input space) to focusing strength
    Ex:
    xs_quad = field integrals in [kG]
    E = beam energy in [GeV]
    q_len = quad thickness in [m]
    conversion_factor = get_quad_strength_conversion_factor(E, q_len)
    ks_quad = conversion_factor * xs_quad # results in the quadrupole geometric focusing strength
    '''
    gamma = E / (0.511e-3)  # beam energy (GeV) divided by electron rest energy (GeV)
    beta = 1.0 - 1.0 / (2 * gamma**2)
    conversion_factor = 0.299 / (10. * q_len * beta * E)

    return conversion_factor
