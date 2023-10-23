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
