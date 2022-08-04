import numpy as np


def calc_phi(delta_nu):
    """
    Calculates line profile of a single line in a single shell at a single
    frequency.
    
    Parameters
    ----------
    delta_nu : float
        Difference between the frequency that the profile is evaluated at
        and the resonance frequency of the line.
    """
    gauss_prefactor = 1 / (3.5e10 * np.sqrt(2 * np.pi))
    phi = gauss_prefactor * np.exp(-0.5 * (delta_nu / 3.5e10) ** 2)
    return phi


def assemble_phis(splasma, marcs_model_fv, nu, line_id_start, line_id_end):
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
    line_id_start : int
        Line id for first line considered.
    line_id_end : int
        Line id for line after last line considered.
        
    Returns
    -------
    phis : numpy.ndarray
        Array of shape (no_of_lines_considered, no_of_shells). Line profiles
        of each line in each shell evaluated at the specified frequency.
    """
    lines_nu = splasma.lines.nu.values[::-1]
    delta_nus = nu - lines_nu[line_id_start:line_id_end]
    phis = np.zeros((len(delta_nus), len(marcs_model_fv)))
    for j in range(len(marcs_model_fv)):
        for i in range(len(delta_nus)):
            phis[i,j] = calc_phi(delta_nus[i])
    return phis
