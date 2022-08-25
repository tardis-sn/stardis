import pandas as pd
import numpy as np
from numba.core import types
from numba.typed import Dict
from astropy import units as u, constants as const

from stardis.opacities.broadening import assemble_phis


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

    alpha_line = splasma.alpha_line.reset_index(drop=True).values[::-1]
    delta_tau_lines = alpha_line * marcs_model_fv.cell_length.values

    lines = splasma.lines[::-1].reset_index()  # bring lines in ascending order of nu

    # add ionization energy to lines
    ionization_data = splasma.ionization_data.reset_index()
    ionization_data["ion_number"] -= 1
    lines = pd.merge(
        lines, ionization_data, how="left", on=["atomic_number", "ion_number"]
    )

    # add level energy (lower and upper) to lines
    levels_energy = splasma.atomic_data.levels.energy
    levels_g = splasma.atomic_data.levels.g
    lines = pd.merge(
        lines,
        levels_energy,
        how="left",
        left_on=["atomic_number", "ion_number", "level_number_lower"],
        right_on=["atomic_number", "ion_number", "level_number"],
    ).rename(columns={"energy": "level_energy_lower"})
    lines = pd.merge(
        lines,
        levels_energy,
        how="left",
        left_on=["atomic_number", "ion_number", "level_number_upper"],
        right_on=["atomic_number", "ion_number", "level_number"],
    ).rename(columns={"energy": "level_energy_upper"})

    # transition doesn't happen at a specific nu due to several factors (changing temperatures, doppler shifts, relativity, etc.)
    # so we take a window 1e12 Hz wide - if nu falls within that, we consider it
    # search_sorted finds the index before which a (tracing_nu +- 1e11) can be inserted
    # in lines_nu array to maintain its sort order
    line_id_starts = lines.nu.values.searchsorted(tracing_nus.value - 1e12) + 1
    line_id_ends = lines.nu.values.searchsorted(tracing_nus.value + 1e12) + 1

    line_cols = map_items_to_indices(lines.columns.to_list())
    lines_array = lines.to_numpy()

    for i in range(len(tracing_nus)):  # iterating over nus (columns)
        # starting and ending indices of all lines considered at a particular frequency, `nu`
        line_id_start, line_id_end = (line_id_starts[i], line_id_ends[i])

        if line_id_start != line_id_end:
            # optical depth of each considered line for each shell at `nu`
            delta_taus = delta_tau_lines[line_id_start:line_id_end]

            # line profiles of each considered line for each shell at `nu`
            phis = assemble_phis(
                atomic_masses=splasma.atomic_mass.values,
                temperatures=marcs_model_fv.t.values,
                electron_densities=splasma.electron_densities.values,
                h_densities=np.array(splasma.ion_number_density.loc[1, 0]),
                he_abundances=np.array(splasma.abundance.loc[2]),
                nu=tracing_nus[i].value,
                lines_considered=lines_array[line_id_start:line_id_end],
                line_cols=line_cols,
            )

            # apply spectral broadening to optical depths (by multiplying line profiles)
            # and take sum of these broadened optical depths along "considered lines" axis
            # to obtain line-interaction optical depths for each shell at `nu` (1D array)
            tau_line[:, i] = (delta_taus * phis).sum(axis=0)

        else:
            tau_line[:, i] = np.zeros(len(marcs_model_fv))

    return tau_line


def map_items_to_indices(items):
    """
    Creates dictionary matching quantities in lines dataframe to their indices.

    Parameters
    ----------
    items : list
        List of column names.

    Returns
    -------
    items_dict : dict
    """
    items_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )

    for i, item in enumerate(items):
        items_dict[item] = i

    return items_dict
