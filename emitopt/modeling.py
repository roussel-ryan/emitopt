import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel
from gpytorch.priors import GammaPrior


def fit_gp_meas_scan(
    k,
    y,
    covar_module=None,
    noise_prior=None,
    tkwargs=None,
):
    """
    Fits a GP model to an emittance beam size measurement quad scan.
    The GP is fit to the BEAM SIZE SQUARED.
    
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

    Returns:
        model: a fitted SingleTaskGP model.
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
    
    return model


def get_virtual_meas_scans(
    k,
    y,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    noise_prior=None,
    tkwargs=None,
):
    """
    Fits a GP model to an emittance beam size measurement quad scan and draws virtual measurement scan
    samples of the beam size SQUARED.
    NOTE: The GP is fit to the BEAM SIZE SQUARED.
    
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

        bss_virtual: a tensor of shape (n_samples x n_steps_quad_scan) where each row repesents 
        the beam size squared results of a virtual quad scan evaluated at the points k_virtual.
    """

    model = fit_gp_meas_scan(k, y, covar_module, noise_prior, tkwargs)
    
    k_virtual = torch.linspace(k.min(), k.max(), n_steps_quad_scan, **tkwargs)

    p = model.posterior(k_virtual.reshape(-1, 1))
    bss_virtual = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)
    
    return k_virtual, bss_virtual
