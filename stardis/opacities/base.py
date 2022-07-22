import pandas as pd
import numpy as np
from astropy import units as u, constants as const


THERMAL_DE_BROGLIE_CONST = const.h**2 / (2 * np.pi * const.m_e * const.k_B)
H_MINUS_CHI = 0.754195 * u.eV  # see https://en.wikipedia.org/wiki/Hydrogen_anion
SAHA_CONST = const.h**2 / (2 * np.pi * const.m_e * const.k_B)

# H minus opacity
def read_wbr_cross_section(wbr_fpath):
    """
    Reads H minus cross sections by wavelength from Wishart (1979) and
    Broad and Reinhardt (1976).

    Parameters
    ----------
    wbr_fpath : str
        Filepath to read H minus cross sections.

    Returns
    -------
    wbr_cross_section : pandas.core.frame.DataFrame
        H minus cross sections by wavelength.
    """

    wbr_cross_section = pd.read_csv(
        wbr_fpath,
        names=["wavelength", "cross_section"],
        comment="#",
    )
    wbr_cross_section.wavelength *= 10  ## nm to AA
    wbr_cross_section.cross_section *= 1e-18  ## to cm^2

    return wbr_cross_section


def calc_tau_h_minus(
    splasma,
    marcs_model_fv,
    tracing_nus,
    wbr_fpath,
):
    """
    Calculates H minus optical depth.

    Parameters
    ----------
    splasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    marcs_model_fv : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    wbr_fpath : str
        Filepath to read H minus cross sections.

    Returns
    -------
    tau_h_minus : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). H minus optical
        depth in each shell for each frequency in tracing_nus.
    """

    wbr_cross_section = read_wbr_cross_section(wbr_fpath)

    tracing_lambdas = tracing_nus.to(u.AA, u.spectral()).value

    # sigma
    h_minus_sigma_nu = np.interp(
        tracing_lambdas,
        wbr_cross_section.wavelength,
        wbr_cross_section.cross_section,
    )

    # tau = sigma * n * l; shape: (num cells, num tracing nus) - tau for each frequency in each cell
    tau_h_minus = (
        h_minus_sigma_nu
        * (np.array(splasma.h_minus_density) * marcs_model_fv.cell_length.values)[
            None
        ].T
    )
    return tau_h_minus


# electron opacity
def calc_tau_e(
    splasma,
    marcs_model_fv,
    tracing_nus,
):
    """
    Calculates electron scattering optical depth.

    Parameters
    ----------
    splasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    marcs_model_fv : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.

    Returns
    -------
    tau_e : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Electron scattering
        optical depth in each shell for each frequency in tracing_nus.
    """

    new_electron_density = splasma.electron_densities.values - splasma.h_minus_density

    tau_e_by_shell = (
        const.sigma_T.cgs.value
        * splasma.electron_densities.values
        * marcs_model_fv.cell_length.values
    )

    tau_e = np.zeros([len(marcs_model_fv), len(tracing_nus)])
    for j in range(len(marcs_model_fv)):
        tau_e[j] = tau_e_by_shell[j]
    return tau_e


# photoionization opacity
def calc_tau_photo(
    splasma,
    marcs_model_fv,
    tracing_nus,
    level,
    strength,
    cutoff_frequency,
):
    """
    Calculates photoionization optical depth.

    Parameters
    ----------
    splasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    marcs_model_fv : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    level : tuple
        Species being ionized. Expressed as
        (atomic_number, ion_number, level_number).
    strength : float
        Coefficient to inverse cube term in equation for photoionization
        optical depth, expressed in cm^2.
    cutoff_frequency : float
        Minimum frequency of light to ionize atom.

    Returns
    -------
    tau_photo : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Photoiosnization
        optical depth in each shell for each frequency in tracing_nus.
    """

    nl = splasma.level_number_density.loc[level] * marcs_model_fv.cell_length.values

    tau_photo = np.zeros([len(marcs_model_fv), len(tracing_nus)])

    for i in range(len(tracing_nus)):
        nu = tracing_nus[i]
        if nu.value >= cutoff_frequency:
            for j in range(len(marcs_model_fv)):
                tau_photo[j, i] = strength * nl[j] * (cutoff_frequency / nu.value) ** 3

    return tau_photo


# line opacity
def calc_tau_line(splasma, marcs_model_fv, tracing_nus):
    """
    Calculates line interaction optical depth.

    Parameters
    ----------
    splasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    marcs_model_fv : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.

    Returns
    -------
    tau_line : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Line interaction
        optical depth in each shell for each frequency in tracing_nus.
    """

    tau_line = np.zeros([len(marcs_model_fv), len(tracing_nus)])

    gauss_prefactor = 1 / (0.35e10 * np.sqrt(2 * np.pi))

    alpha_line = splasma.alpha_line.reset_index(drop=True).values[::-1]
    delta_tau_lines = alpha_line * marcs_model_fv.cell_length.values

    # transition doesn't happen at a specific nu due to several factors (changing temperatires, doppler shifts, relativity, etc.)
    # so we take a window 2e11 Hz wide - if nu falls within that, we consider it
    lines_nu = splasma.lines.nu.values[::-1]  # reverse to bring them to ascending order
    # search_sorted finds the index before which a (tracing_nu +- 1e11) can be inserted
    # in lines_nu array to maintain its sort order
    line_id_starts = lines_nu.searchsorted(tracing_nus.value - 1e11)
    line_id_ends = lines_nu.searchsorted(tracing_nus.value + 1e11)

    for i in range(len(tracing_nus)):  # iterating over nus (columns)
        nu, line_id_start, line_id_end = (
            tracing_nus[i],
            line_id_starts[i],
            line_id_ends[i],
        )

        if line_id_start != line_id_end:
            delta_tau = delta_tau_lines[line_id_start:line_id_end]
            delta_nu = nu.value - lines_nu[line_id_start:line_id_end]
            phi = gauss_prefactor * np.exp(-0.5 * (delta_nu / 0.35e10) ** 2)
            tau_line[:, i] = (delta_tau * phi[None].T).sum(axis=0)
        else:
            tau_line[:, i] = np.zeros(len(marcs_model_fv))

    return tau_line
