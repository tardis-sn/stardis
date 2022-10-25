import pandas as pd
import numpy as np
from numba.core import types
from numba.typed import Dict
from astropy import units as u, constants as const

from stardis.opacities.broadening import assemble_phis


THERMAL_DE_BROGLIE_CONST = const.h ** 2 / (2 * np.pi * const.m_e * const.k_B)
H_MINUS_CHI = 0.754195 * u.eV  # see https://en.wikipedia.org/wiki/Hydrogen_anion
SAHA_CONST = const.h ** 2 / (2 * np.pi * const.m_e * const.k_B)

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
        wbr_fpath, names=["wavelength", "cross_section"], comment="#",
    )
    wbr_cross_section.wavelength *= 10  ## nm to AA
    wbr_cross_section.cross_section *= 1e-18  ## to cm^2

    return wbr_cross_section


def calc_alpha_h_minus(
    stellar_plasma, stellar_model, tracing_nus, wbr_fpath,
):
    """
    Calculates H minus optical depth.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    fv_geometry : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    wbr_fpath : str
        Filepath to read H minus cross sections.

    Returns
    -------
    alpha_h_minus : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). H minus optical
        depth in each shell for each frequency in tracing_nus.
    """

    fv_geometry = stellar_model.fv_geometry

    wbr_cross_section = read_wbr_cross_section(wbr_fpath)

    tracing_lambdas = tracing_nus.to(u.AA, u.spectral()).value

    # sigma
    h_minus_sigma_nu = np.interp(
        tracing_lambdas, wbr_cross_section.wavelength, wbr_cross_section.cross_section,
    )

    # alpha = sigma * n * l; shape: (num cells, num tracing nus) - alpha for each frequency in each cell
    alpha_h_minus = h_minus_sigma_nu * np.array(stellar_plasma.h_minus_density)[None].T
    return alpha_h_minus


# electron opacity
def calc_alpha_e(
    stellar_plasma, stellar_model, tracing_nus,
):
    """
    Calculates electron scattering optical depth.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    fv_geometry : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.

    Returns
    -------
    alpha_e : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Electron scattering
        optical depth in each shell for each frequency in tracing_nus.
    """

    fv_geometry = stellar_model.fv_geometry

    new_electron_density = (
        stellar_plasma.electron_densities.values - stellar_plasma.h_minus_density
    )

    alpha_e_by_shell = (
        const.sigma_T.cgs.value * stellar_plasma.electron_densities.values
    )

    alpha_e = np.zeros([len(fv_geometry), len(tracing_nus)])
    for j in range(len(fv_geometry)):
        alpha_e[j] = alpha_e_by_shell[j]
    return alpha_e


# photoionization opacity
def calc_alpha_h_photo(
    stellar_plasma, stellar_model, tracing_nus, levels=[1, 2, 3], strength=7.91e-18,
):
    """
    Calculates photoionization optical depth.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    fv_geometry : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    level : tuple
        Species being ionized. Expressed as
        (atomic_number, ion_number, level_number).
    strength : float
        Coefficient to inverse cube term in equation for photoionization
        optical depth, expressed in cm^2.

    Returns
    -------
    alpha_photo : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Photoiosnization
        optical depth in each shell for each frequency in tracing_nus.
    """

    fv_geometry = stellar_model.fv_geometry

    ionization_energy = stellar_plasma.ionization_data.loc[(1, 1)]

    alpha_h_photo = np.zeros([len(fv_geometry), len(tracing_nus)])

    for level in levels:
        alpha_h_photo_level = np.zeros([len(fv_geometry), len(tracing_nus)])
        level_tuple = (1, 0, level)
        cutoff_frequency = (
            ionization_energy - stellar_plasma.excitation_energy.loc[level_tuple]
        ) / const.h.cgs.value
        n = stellar_plasma.level_number_density.loc[level_tuple]
        for i in range(len(tracing_nus)):
            nu = tracing_nus[i]
            if nu.value >= cutoff_frequency:
                for j in range(len(fv_geometry)):
                    alpha_h_photo_level[j, i] = (
                        strength * n[j] * (cutoff_frequency / nu.value) ** 3
                    )
        alpha_h_photo += alpha_h_photo_level

    return alpha_h_photo


# line opacity
def calc_alpha_line_at_nu(stellar_plasma, stellar_model, tracing_nus):
    """
    Calculates line interaction optical depth.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    fv_geometry : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.

    Returns
    -------
    alpha_line : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Line interaction
        optical depth in each shell for each frequency in tracing_nus.
    """

    fv_geometry = stellar_model.fv_geometry

    alpha_line_at_nu = np.zeros([len(fv_geometry), len(tracing_nus)])

    alpha_lines = stellar_plasma.alpha_line.reset_index(drop=True).values[::-1]

    lines = stellar_plasma.lines[
        ::-1
    ].reset_index()  # bring lines in ascending order of nu

    # add ionization energy to lines
    ionization_data = stellar_plasma.ionization_data.reset_index()
    ionization_data["ion_number"] -= 1
    lines = pd.merge(
        lines, ionization_data, how="left", on=["atomic_number", "ion_number"]
    )

    # add level energy (lower and upper) to lines
    levels_energy = stellar_plasma.atomic_data.levels.energy
    levels_g = stellar_plasma.atomic_data.levels.g
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

    h_densities = stellar_plasma.ion_number_density.loc[1, 0].to_numpy()
    he_abundances = stellar_plasma.abundance.loc[2].to_numpy()

    for i in range(len(tracing_nus)):  # iterating over nus (columns)
        # starting and ending indices of all lines considered at a particular frequency, `nu`
        line_id_start, line_id_end = (line_id_starts[i], line_id_ends[i])

        if line_id_start != line_id_end:
            # optical depth of each considered line for each shell at `nu`
            alphas = alpha_lines[line_id_start:line_id_end]

            # line profiles of each considered line for each shell at `nu`
            phis = assemble_phis(
                atomic_masses=stellar_plasma.atomic_mass.values,
                temperatures=fv_geometry.t.values,
                electron_densities=stellar_plasma.electron_densities.values,
                h_densities=h_densities,
                he_abundances=he_abundances,
                nu=tracing_nus[i].value,
                lines_considered=lines_array[line_id_start:line_id_end],
                line_cols=line_cols,
            )

            # apply spectral broadening to optical depths (by multiplying line profiles)
            # and take sum of these broadened optical depths along "considered lines" axis
            # to obtain line-interaction optical depths for each shell at `nu` (1D array)
            alpha_line_at_nu[:, i] = (alphas * phis).sum(axis=0)

        else:
            alpha_line_at_nu[:, i] = np.zeros(len(fv_geometry))

    return alpha_line_at_nu


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
    items_dict = Dict.empty(key_type=types.unicode_type, value_type=types.int64,)

    for i, item in enumerate(items):
        items_dict[item] = i

    return items_dict


def calc_alphas(
    stellar_plasma,
    stellar_model,
    tracing_nus,
    wbr_fpath,
    h_photo_levels=[1, 2, 3],
    h_photo_strength=7.91e-18,
):
    """
    TODO: allow for selecting certain opacities to calculate
    """
    alpha_h_minus = calc_alpha_h_minus(
        stellar_plasma, stellar_model, tracing_nus, wbr_fpath
    )
    alpha_e = calc_alpha_e(stellar_plasma, stellar_model, tracing_nus)
    alpha_h_photo = calc_alpha_h_photo(
        stellar_plasma,
        stellar_model,
        tracing_nus,
        levels=h_photo_levels,
        strength=h_photo_strength,
    )
    alpha_line_at_nu = calc_alpha_line_at_nu(stellar_plasma, stellar_model, tracing_nus)

    return alpha_h_minus + alpha_e + alpha_h_photo + alpha_line_at_nu
