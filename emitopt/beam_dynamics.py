import torch
from matplotlib import pyplot as plt


def compute_emit_bmag(k, beamsize_squared, q_len, rmat, beta0=1., alpha0=0., get_bmag=True, thick=True):
    """
    A function that computes the emittance(s) corresponding to a set of quadrupole measurement scans
    using a thick quad model.

    Parameters:
        k: torch tensor of shape (n_steps_quad_scan,) or (batchshape x n_steps_quad_scan)
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in the emittance scan(s)

        beamsize_squared: torch tensor of shape (batchshape x n_steps_quad_scan),
                representing the mean-square beamsize outputs in [m^2] of the emittance scan(s)
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
        
        rmat: tensor shape (2x2) or (batchshape x 2 x 2)
                containing the 2x2 R matrices describing the transport from the end of the 
                measurement quad to the observation screen.

        beta0: float or tensor shape (batchshape x 1) designating the design beta twiss parameter at the screen
        
        alpha0: float or tensor shape (batchshape x 1) designating the design alpha twiss parameter at the screen
        
        get_bmag: boolean, whether or not to compute the bmag along with the emittance
                    (Set to False for faster computation of only the emittance)
    Returns:
        emit: tensor shape (batchshape) containing the geometric emittance fit results for each scan
        bmag_min: tensor shape (batchshape) containing the bmag corresponding to the optimal point for each scan
        sig: shape tensor shape (batchshape x 3 x 1) containing column vectors of [sig11, sig12, sig22]
        is_valid: tensor shape (batchshape) identifying physical validity of the emittance fit results
        
    SOURCE PAPER: http://www-library.desy.de/preparch/desy/thesis/desy-thesis-05-014.pdf
    """
    # get initial sigma 
    sig, total_rmats = beam_matrix_from_quad_scan(k, beamsize_squared, q_len, rmat, thick=thick)
    
    emit = torch.sqrt(sig[...,0,0]*sig[...,2,0] - sig[...,1,0]**2) # result shape (batchshape)
    
    # check sigma matrix and emit for physical validity
    is_valid = torch.logical_and(sig[...,0,0] > 0, sig[...,2,0] > 0) # result batchshape
    is_valid = torch.logical_and(is_valid, ~torch.isnan(emit)) # result batchshape
    
    if get_bmag:
        bmag = compute_bmag(sig, emit, total_rmats, beta0, alpha0) # result batchshape
    else:
        bmag = None

    return emit, bmag, sig, is_valid


def beam_matrix_from_quad_scan(k, beamsize_squared, q_len, rmat, thick=True):
    """
    Reconstructs the beam matrices corresponding to a set of quadrupole measurement scans
    using a thick quad model and the pseudoinverse method.

    Parameters:
        k: torch tensor of shape (n_steps_quad_scan,) or (batchshape x n_steps_quad_scan),
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in a batch of emittance scans

        beamsize_squared: torch tensor of shape (batchshape x n_steps_quad_scan),
                where each row represents the mean-square beamsize outputs in [m^2] of an emittance scan
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
        
        rmat: tensor shape (2x2) or (batchshape x 2 x 2)
                containing the 2x2 R matrices describing the transport from the end of the 
                measurement quad to the observation screen.
                
    Outputs:
        
    """
    
    # construct the A matrix from eq. (3.2) & (3.3) of source paper
    quad_rmats = build_quad_rmat(k, q_len, thick=thick) # result shape (batchshape x nsteps x 2 x 2)
    total_rmats = rmat.unsqueeze(-3).double() @ quad_rmats.double() 
    # result shape (batchshape x nsteps x 2 x 2)
    
    # prepare the A matrix
    r11, r12 = total_rmats[...,0,0], total_rmats[...,0,1]
    amat = torch.stack((r11**2, 2.*r11*r12, r12**2), dim=-1)
    # amat result (batchshape x nsteps x 3)

    # get sigma matrix elements just before measurement quad from pseudo-inverse
    sig = amat.pinverse() @ beamsize_squared.unsqueeze(-1).double()
    # shapes (batchshape x 3 x nsteps) @ (batchshape x nsteps x 1)
    # result shape (batchshape x 3 x 1) containing column vectors of [sig11, sig12, sig22]
    
    return sig, total_rmats


def compute_bmag(sig, emit, total_rmats, beta0, alpha0):
    """
    parameters:
        sig: tensor shape batchshape x 3 x 1 giving the initial beam matrix before the measurement quad
        
        emit: tensor shape batchshape giving the emittance for each initial beam matrix
        
        total_rmats: tensor shape batchshape x nsteps x 2 x 2 giving the rmats that describe transport
                    through the meas quad and to the screen for each step in the measurement scan(s)
        
        beta0: float or tensor shape (batchshape x 1) designating the design beta (twiss) parameter
                at the screen
        
        alpha0: float or tensor shape (batchshape x 1) designating the design alpha (twiss) parameter
                at the screen
    returns:
        bmag_min: tensor shape batchshape containing the minimum (best) bmag from each measurement scan
    """
    twiss_at_screen = propagate_beam_quad_scan(sig, emit, total_rmats)[1]
    # result shape (batchshape x nsteps x 3 x 1)

    # get design gamma0 from design beta0, alpha0
    gamma0 = (1 + alpha0**2) / beta0

    # compute bmag
    bmag = 0.5 * (twiss_at_screen[...,0,0] * gamma0
                - 2 * twiss_at_screen[...,1,0] * alpha0
                + twiss_at_screen[...,2,0] * beta0
               )
    # result shape (batchshape x nsteps)

    # select minimum bmag from quad scan
    bmag_min, bmag_min_id = torch.min(bmag, dim=-1) # result shape (batchshape)

    return bmag_min


def propagate_beam_quad_scan(sig_init, emit, rmat):
    """
    parameters:
        sig_init: shape batchshape x 3 x 1
        emit: shape batchshape
        rmat: shape batchshape x nsteps x 2 x 2
    returns:
        sig_final: shape batchshape x nsteps x 3 x 1
        twiss_final: shape batchshape x nsteps x 3 x 1
    """
    temp = torch.tensor([[[1., 0., 0.],
                           [0., -1., 0.],
                           [0., 0., 1.]]], device=sig_init.device).double()
    twiss_init = (temp @ sig_init) @ (1/emit.reshape(*emit.shape,1,1)) # result shape (batchshape x 3 x 1)
    
    twiss_transport = twiss_transport_mat_from_rmat(rmat) # result shape (batchshape x 3 x 3)

    twiss_final = twiss_transport @ twiss_init.unsqueeze(-3)
    # result shape (batchshape x nsteps x 3 x 1)

    sig_final = (temp @ twiss_final) @ emit.reshape(*emit.shape,1,1,1) 
    # result shape (batchshape x nsteps x 3 x 1)
    
    return sig_final, twiss_final


def twiss_transport_mat_from_rmat(rmat):
    c, s, cp, sp = rmat[...,0,0], rmat[...,0,1], rmat[...,1,0], rmat[...,1,1]
    result = torch.stack((
        torch.stack((c**2, -2*c*s, s**2), dim=-1), 
        torch.stack((-c*cp, c*sp + cp*s, -s*sp), dim=-1),
        torch.stack((cp**2, -2*cp*sp, sp**2), dim=-1)), 
        dim=-2
    )
    return result


def build_quad_rmat(k, q_len, thick=True):
    if thick:
        eps = 2.220446049250313e-16  # machine epsilon to double precision
        sqrt_k = k.abs().sqrt() + eps
        c, s, cp, sp = (
                        torch.cos(sqrt_k*q_len)*(k >= 0) + torch.cosh(sqrt_k*q_len)*(k < 0), 
                        1./sqrt_k * torch.sin(sqrt_k*q_len)*(k >= 0) + 1./sqrt_k * torch.sinh(sqrt_k*q_len)*(k < 0),
                        -sqrt_k * torch.sin(sqrt_k*q_len)*(k >= 0) + sqrt_k * torch.sinh(sqrt_k*q_len)*(k < 0), 
                        torch.cos(sqrt_k*q_len)*(k >= 0) + torch.cosh(sqrt_k*q_len)*(k < 0)
                       )
    else:
        c, s, cp, sp = (torch.ones_like(k), torch.zeros_like(k), -k.abs()*q_len, torch.ones_like(k))
        
    result = torch.stack((
        torch.stack((c, s), dim=-1), 
        torch.stack((cp, sp), dim=-1),), 
        dim=-2
    )
     
    return result


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

        msbs: a tensor of shape (n_samples x n_steps_quad_scan) where each row repesents 
        the mean-square beamsize results of a virtual quad scan evaluated at the points k_virtual.
    """
    
    tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
        
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
    bss_virtual = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)
    
    return k_virtual, bss_virtual


# -

def compute_emit_bayesian(
    k,
    beamsize,
    q_len,
    rmat,
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

        beamsize: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
                    
        rmat: tensor containing the (fixed) 2x2 R matrix describing the transport from the end of the 
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
    tkwargs = twkargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

    k = torch.tensor(k, **tkwargs)
    beamsize = torch.tensor(beamsize, **tkwargs)

    k_virtual, bss_virtual = fit_gp_quad_scan(
        k=k,
        y=beamsize,
        n_samples=n_samples,
        n_steps_quad_scan=n_steps_quad_scan,
        covar_module=covar_module,
        noise_prior=noise_prior,
        tkwargs=tkwargs
    )
    
    (emit, bmag, sig, is_valid) = compute_emit_bmag(k=k_virtual, 
                                                              beamsize_squared=bss_virtual, 
                                                              q_len=q_len, 
                                                              rmat=rmat, 
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
                                   beamsize=beamsize, 
                                   q_len=q_len, 
                                   rmat=rmat,
                                   emit=emit_valid, 
                                   bmag=bmag_valid,
                                   sig=sig_valid, 
                                   k_virtual=k_virtual,
                                   bss_virtual=bss_virtual,
                                  )
    return emit_valid, bmag_valid, sig_valid, sample_validity_rate


def plot_valid_thick_quad_fits(k, beamsize, q_len, rmat, emit, bmag, sig, ci=0.95, tkwargs=None, k_virtual=None, bss_virtual=None):
    """
    A function to plot the physically valid fit results
    produced by get_valid_emit_bmag_samples_from_quad_scan().

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

        distance: the longitudinal distance (drift length) in [m] from the measurement
                    quadrupole to the observation screen
        
        (NOT IN USE)
        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.
                
        ci: "Confidence interval" for plotting upper/lower quantiles.

        tkwargs: dict containing the tensor device and dtype
    """
    
    tkwargs = twkargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

    k_fit = torch.linspace(k.min(), k.max(), 10, **tkwargs)
    quad_rmats = build_quad_rmat(k_fit, q_len) # result shape (len(k_fit) x 2 x 2)
    total_rmats = rmat.reshape(1,2,2).double() @ quad_rmats.double() # result shape (len(k_fit) x 2 x 2)
    sig_final = propagate_beam_quad_scan(sig, emit, total_rmats)[0] # result shape len(sig) x len(k_fit) x 3 x 1
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
        k, beamsize*1.e6, marker="x", s=120, c="orange", label="Measurements", zorder=2
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


def get_quad_scale_factor(E=0.135, q_len=0.108):
    """
    Computes multiplicative scale factor to convert from LCLS quad PV values (model input space)
    in [kG] to the geometric focusing strengths in [m^-2].

    Parameters:
        E: Beam energy in [GeV]
        q_len: quadrupole length or "thickness" (longitudinal) in [m]

    Returns:
        conversion_factor: scale factor by which to multiply the LCLS quad PV values [kG] to get
                            the geometric focusing strengths [m^-2]
    Example:
    xs_quad = field integrals in [kG]
    E = beam energy in [GeV]
    q_len = quad thickness in [m]
    scale_factor = get_quad_scale_factor(E, q_len)
    ks_quad = scale_factor * xs_quad # results in the quadrupole geometric focusing strength
    """
    gamma = E / (0.511e-3)  # beam energy (GeV) divided by electron rest energy (GeV)
    beta = 1.0 - 1.0 / (2 * gamma**2)
    scale_factor = 0.299 / (10.0 * q_len * beta * E)

    return scale_factor


def normalize_emittance(emit, energy):
    gamma = energy / (0.511e-3)  # beam energy (GeV) divided by electron rest energy (GeV)
    beta = 1.0 - 1.0 / (2 * gamma**2)
    emit_n = gamma * beta * emit
    return emit_n
