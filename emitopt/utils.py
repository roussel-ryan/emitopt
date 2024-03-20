import torch
def get_quad_scale_factor(E, q_len):
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

import pandas as pd
def get_measurement_scan_inputs(x_tuning: dict, x_meas: dict) -> pd.DataFrame:
    """
    A function that generates the inputs for emittance measurement scans at the tuning
    configurations specified by x_tuning.

    Parameters:
        x_tuning: a dictionary with key/value pairs giving the tuning parameter name and a list
                    of settings at which we want to perform measurement scans. All tuning parameters 
                    must have the same number of settings specified.
        x_meas: a dictionary with a single key/value pair giving the measurement quadrupole name, 
                            and a list of quadrupole settings for the measurement scan.
                    
        
    Returns:
        df_meas_scans: a pandas dataframe containing the inputs for measurement scans performed
                        at the locations in tuning parameter space specified by x_tuning
                        
    """
    df_tuning = pd.DataFrame(x_tuning)
    df_meas_scans = pd.DataFrame()

    assert len(x_meas.keys()) == 1
    for meas_device, meas_vals in x_meas.items():
        for meas_val in meas_vals:
            df = copy(df_tuning)
            df[meas_device] = meas_val
            df_meas_scans = pd.concat([df_meas_scans, df], ignore_index=True)
    return df_meas_scans

def x_tuning_to_dict(generator, x_tuning):
    """
    Converts a single set of tuning parameters to a dictionary for input to Xopt
    
    x_tuning = tensor of shape (1, n_tuning_dims)
    """
    
    tuning_parameter_names = [name for i, name in enumerate(generator.vocs.variable_names)
                              if i != generator.algorithm.meas_dim]
    x_tuning_dict = {}
    for i in range(x_tuning.shape[1]):
        x_tuning_dict[tuning_parameter_names[i]] = x_tuning[0,i]
    return x_tuning_dict

def x_tuning_to_tensor(generator, x_tuning_dict):
    """
    Converts a dict of tuning parameter settings to a tensor of shape (n_points, n_tuning_dims)
    Each variable in x_tuning_dict must have the same number of values for concatenation.
    """
    x_tuning = torch.cat([x_tuning_dict[name].reshape(-1,1) 
                          for i, name in enumerate(generator.vocs.variable_names)
                          if i != generator.algorithm.meas_dim],
                         dim=1)
    return x_tuning

def get_bax_optimum(generator):
    x_tuning_best = generator.algorithm_results['x_tuning_best'].mean(dim=0)
    x_tuning_dict = x_tuning_to_dict(generator, x_tuning_best)
    return x_tuning_dict