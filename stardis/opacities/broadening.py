import numpy as np
from astropy import constants as const
from scipy.special import voigt_profile


def calc_gamma_linear_stark():
    """
    Calculates broadening parameter for linear Stark broadening.???
    
    Parameters
    ----------
    
    Returns
    -------
    gamma_linear_stark : float
        Broadening parameter for linear Stark broadening.
    """
    gamma_linear_stark = 0
    
    return gamma_linear_stark


def calc_gamma_resonance():
    """
    Calculates broadening parameter for resonance broadening.???
    
    Parameters
    ----------
    
    Returns
    -------
    gamma_resonance : float
        Broadening parameter for resonance broadening.
    """
    gamma_resonance = 0
    
    return gamma_resonance


def calc_gamma_quadratic_stark():
    """
    Calculates broadening parameter for quadratic Stark broadening.???
    
    Parameters
    ----------
    
    Returns
    -------
    gamma_quadratic_stark : float
        Broadening parameter for quadratic Stark broadening.
    """
    gamma_quadratic_stark = 0
    
    return gamma_quadratic_stark


def calc_gamma_van_der_waals():
    """
    Calculates broadening parameter for van der Waals broadening.???
    
    Parameters
    ----------
    
    Returns
    -------
    gamma_van_der_waals : float
        Broadening parameter for van der Waals broadening.
    """
    gamma_van_der_waals = 0
    
    return gamma_van_der_waals


def calc_gamma_collision():
    """
    Calculates total collision broadening parameter by adding up all
    contributing broadening parameters.
    
    Parameters
    ----------
    
    Returns
    -------
    gamma_collision : float
        Total collision broadening parameter.
    """
    gamma_linear_stark = calc_gamma_linear_stark()
    gamma_resonance = calc_gamma_resonance()
    gamma_quadratic_stark = calc_gamma_quadratic_stark()
    gamma_van_der_waals = calc_gamma_van_der_waals()
    
    gamma_collision = gamma_linear_stark + gamma_resonance + gamma_quadratic_stark + gamma_van_der_waals
    
    return gamma_collision


def assemble_phis(splasma, marcs_model_fv, nu, lines_considered):
    """
    Puts together several line profiles at a single frequency for all shells.

    Parameters
    ----------
    splasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    marcs_model_fv : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    nu : float
        Frequency at which line profiles are being evaluated.
    lines_considered : pandas.core.frame.DataFrame
        DataFrame of all lines considered.

    Returns
    -------
    phis : numpy.ndarray
        Array of shape (no_of_lines_considered, no_of_shells). Line profiles
        of each line in each shell evaluated at the specified frequency.
    """
    delta_nus = nu - lines_considered.nu
    phis = np.zeros((len(delta_nus), len(marcs_model_fv)))
    
    for j in range(len(marcs_model_fv)):
        T = marcs_model_fv.t[j]
        for i in range(len(delta_nus)):
            line = lines_considered.loc[i]
            nu_line = line.nu
            atomic_number = line.atomic_number
            atomic_mass = splasma.atomic_mass[atomic_number]
            sigma_doppler = (nu_line / const.c.cgs.value) * np.sqrt(const.k_B.cgs.value * T / atomic_mass)
            gamma_collision = calc_gamma_collision()
            gamma = line.A_ul + gamma_collision
            lorentz_hwhm = gamma / (4 * np.pi)
            phis[i, j] = voigt_profile(delta_nus[i], sigma_doppler, lorentz_hwhm)
    
    return phis
