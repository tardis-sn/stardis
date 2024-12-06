import pandas as pd
import numpy as np
from pathlib import Path
import numba
import logging

from astropy import units as u, constants as const

from stardis.radiation_field.opacities.opacities_solvers.broadening import (
    calculate_broadening,
    calculate_molecule_broadening,
)
from stardis.radiation_field.opacities.opacities_solvers.voigt import voigt_profile
from stardis.radiation_field.opacities.opacities_solvers.util import (
    sigma_file,
    get_number_density,
)

# constants
VACUUM_ELECTRIC_PERMITTIVITY = 1 / (4 * np.pi)
BF_CONSTANT = (
    64
    * np.pi**4
    * const.e.esu**10
    * const.m_e.cgs
    / (3 * np.sqrt(3) * const.c.cgs * const.h.cgs**6)
).value
FF_CONSTANT = (
    4
    / (3 * const.h.cgs * const.c.cgs)
    * const.e.esu**6
    * np.sqrt(2 * np.pi / (3 * const.m_e.cgs**3 * const.k_B.cgs))
).value
RYDBERG_FREQUENCY = (const.c.cgs * const.Ryd.cgs).value

logger = logging.getLogger(__name__)


# Calculate opacity from any table specified by the user
def calc_alpha_file(stellar_plasma, stellar_model, tracing_nus, opacity_source, fpath):
    """
    Calculates opacities when a cross-section file is provided.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    opacity_source : str
        String representing the opacity source. Used to obtain the appropriate number density of the corresponding species.
    fpath : str
        Path to the cross-section file.

    Returns
    -------
    alpha_file : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). File opacity in
        each depth point for each frequency in tracing_nus.
    """

    tracing_lambdas = tracing_nus.to(u.AA, u.spectral()).value

    sigmas = sigma_file(
        tracing_lambdas, stellar_model.temperatures.value, Path(fpath), opacity_source
    )
    number_density, atomic_number, ion_number = get_number_density(
        stellar_plasma, opacity_source
    )  ###TODO: Should revisit this function to make it more general and less hacky.
    return sigmas * number_density.to_numpy()[:, np.newaxis]


# rayleigh opacity
def calc_alpha_rayleigh(stellar_plasma, stellar_model, tracing_nus, species):
    """
    Calculates Rayleigh scattering opacity.
    https://iopscience.iop.org/article/10.3847/0004-637X/817/2/116
    https://ui.adsabs.harvard.edu/abs/1962ApJ...136..690D/

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    species : list of str
        List of species for which Rayleigh scattering is to be considered.
        Currently only "H", "He", and "H2" are supported.

    Returns
    -------
    alpha_rayleigh : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Rayleigh scattering
        opacity at each depth point for each frequency in tracing_nus.
    """

    nu_H = const.c.cgs * const.Ryd.cgs
    upper_bound = 2.3e15 * u.Hz
    tracing_nus[tracing_nus > upper_bound] = 0
    relative_nus = tracing_nus.value / (2 * nu_H.value)

    nu4 = relative_nus**4
    nu6 = relative_nus**6
    nu8 = relative_nus**8

    # This seems super hacky. We initialize arrays in many places using the shape of temperatures or geometry. Revisit later to standardize.
    coefficient4 = np.zeros(stellar_model.no_of_depth_points)
    coefficient6 = np.zeros(stellar_model.no_of_depth_points)
    coefficient8 = np.zeros(stellar_model.no_of_depth_points)

    if "H" in species:
        density = np.array(stellar_plasma.ion_number_density.loc[1, 0])
        coefficient4 += 20.24 * density
        coefficient6 += 239.2 * density
        coefficient8 += 2256 * density
    if "He" in species:
        density = np.array(stellar_plasma.ion_number_density.loc[2, 0])
        coefficient4 += 1.913 * density
        coefficient6 += 4.52 * density
        coefficient8 += 7.90 * density
    if "H2" in species:
        density = np.array(stellar_plasma.h2_density)
        coefficient4 += 28.39 * density
        coefficient6 += 215.0 * density
        coefficient8 += 1303 * density

    alpha_rayleigh = (
        coefficient4[None].T * nu4
        + coefficient6[None].T * nu6
        + coefficient8[None].T * nu8
    )

    alpha_rayleigh *= const.sigma_T.cgs.value

    return alpha_rayleigh


# electron opacity
def calc_alpha_electron(
    stellar_plasma,
    stellar_model,
    tracing_nus,
    disable_electron_scattering=False,
):
    """
    Calculates electron scattering opacity.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    disable_electron_scattering : bool, optional
        Forces function to return 0. By default False.

    Returns
    -------
    alpha_electron : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Electron scattering
        opacity at each depth point for each frequency in tracing_nus.
    """

    if disable_electron_scattering:
        return 0
    alpha_electron_by_depth_point = (
        const.sigma_T.cgs.value * stellar_plasma.electron_densities.values
    )

    alpha_electron = np.repeat(
        alpha_electron_by_depth_point[:, np.newaxis], len(tracing_nus), axis=1
    )

    return alpha_electron


# hydrogenic bound-free and free-free opacity
def calc_alpha_bf(stellar_plasma, stellar_model, tracing_nus, species):
    """
    Calculates bound-free opacity.
    https://ui.adsabs.harvard.edu/abs/2014tsa..book.....H/ chapter 7

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    species : tardis.io.configuration.config_reader.Configuration
        Dictionary (in the form of a Configuration object) containing all
        species for which bound-free opacity is to be calculated.

    Returns
    -------
    alpha_bf : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Bound-free opacity at
        each depth_point for each frequency in tracing_nus.
    """
    # This implementation will only work with 1D.

    inv_nu3 = tracing_nus.value ** (-3)
    alpha_bf = np.zeros((stellar_model.no_of_depth_points, len(tracing_nus)))

    for spec, dct in species.items():
        # just for reading atomic number and ion number
        ion_number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec + "_bf"
        )

        ionization_energy = stellar_plasma.ionization_data.loc[
            (atomic_number, ion_number + 1)
        ]

        alpha_spec = np.zeros((stellar_model.no_of_depth_points, len(tracing_nus)))

        levels = [
            (i, j, k)
            for i, j, k in stellar_plasma.levels
            if (i == atomic_number and j == ion_number)
        ]
        for level in levels:
            alpha_level = np.zeros((stellar_model.no_of_depth_points, len(tracing_nus)))
            cutoff_frequency = (
                ionization_energy - stellar_plasma.excitation_energy.loc[level]
            ) / const.h.cgs.value
            number_density = np.array(stellar_plasma.level_number_density.loc[level])
            for i in range(len(tracing_nus)):
                nu = tracing_nus[i].value
                alpha_level[:, i] = calc_contribution_bf(
                    nu, cutoff_frequency, number_density, ion_number
                )

            alpha_spec += alpha_level

        alpha_bf += alpha_spec

    alpha_bf *= inv_nu3

    return alpha_bf


@numba.njit
def calc_contribution_bf(nu, cutoff_frequency, number_density, ion_number):
    """
    Calculates the contribution of a single level to the bound-free opacity
    coefficient of a single frequency at each depth point.

    Parameters
    ----------
    nu : float
        Frequency in Hz.
    cutoff_frequency : float
        Lowest frequency in Hz that can ionize an electron in the level being
        considered.
    number_density : numpy.ndarray
        Number density of level.
    ion_number : int
        Ion number of the ion being considered.

    Returns
    -------
    numpy.ndarray
        Bound-free opacity coefficient contribution.
    """

    if nu >= cutoff_frequency:
        n5 = ((ion_number + 1) * np.sqrt(RYDBERG_FREQUENCY / cutoff_frequency)) ** 5
        return BF_CONSTANT * (ion_number + 1) ** 4 * number_density / n5

    else:
        return 0 * number_density


def calc_alpha_ff(stellar_plasma, stellar_model, tracing_nus, species):
    """
    Calculates free-free opacity.
    https://ui.adsabs.harvard.edu/abs/2014tsa..book.....H/ chapter 7

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    species : tardis.io.configuration.config_reader.Configuration
        Dictionary (in the form of a Configuration object) containing all
        species for which free-free opacity is to be calculated.

    Returns
    -------
    alpha_ff : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Free-free opacity at
        each depth point for each frequency in tracing_nus.
    """

    temperatures = stellar_model.temperatures.value

    inv_nu3 = tracing_nus.value ** (-3)
    alpha_ff = np.zeros((stellar_model.no_of_depth_points, len(tracing_nus)))

    for spec, dct in species.items():
        alpha_spec = np.zeros((stellar_model.no_of_depth_points, len(tracing_nus)))

        number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec + "_ff"
        )

        ###TODO: optimize this loop
        for j in range(len(number_density)):
            alpha_spec[j] = number_density[j] / np.sqrt(temperatures[j])

        alpha_spec *= FF_CONSTANT * ion_number**2
        alpha_ff += alpha_spec

    alpha_ff *= inv_nu3

    return alpha_ff


def gaunt_times_departure(tracing_nus, temperatures, gaunt_fpath, departure_fpath):
    """
    To be implemented.
    """
    pass


# line opacity
def calc_alpha_line_at_nu(
    stellar_plasma,
    stellar_model,
    tracing_nus,
    line_opacity_config,
):
    """
    Calculates line opacity.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    line_opacity_config : tardis.io.configuration.config_reader.Configuration
        Line opacity section of the STARDIS configuration.

    Returns
    -------
    alpha_line_at_nu : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Line opacity at
        each depth_point for each frequency in tracing_nus.
    gammas : numpy.ndarray
        Array of shape (no_of_lines, no_of_depth_points). Collisional broadening
        parameter of each line at each depth_point.
    doppler_widths : numpy.ndarray
        Array of shape (no_of_lines, no_of_depth_points). Doppler width of each
        line at each depth_point.
    """

    if line_opacity_config.disable:
        return 0, 0, 0

    use_vald = line_opacity_config.vald_linelist.use_linelist
    if use_vald:
        lines = stellar_plasma.lines_from_linelist
    else:
        lines = stellar_plasma.lines.reset_index()

        # add ionization energy to lines
        ionization_data = stellar_plasma.ionization_data.reset_index()
        ionization_data["ion_number"] -= 1
        lines = pd.merge(
            lines, ionization_data, how="left", on=["atomic_number", "ion_number"]
        )

        # add level energy (lower and upper) to lines
        levels_energy = stellar_plasma.atomic_data.levels.energy
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

    lines_sorted = lines.sort_values("nu")
    lines_sorted_in_range = lines_sorted[
        lines_sorted.nu.between(tracing_nus.min(), tracing_nus.max())
    ]
    line_nus = lines_sorted_in_range.nu.to_numpy()

    if use_vald:
        alphas_and_nu = stellar_plasma.alpha_line_from_linelist.sort_values("nu")
    else:
        alphas_and_nu = stellar_plasma.alpha_line.sort_values("nu")

    alphas_array = (
        alphas_and_nu[alphas_and_nu.nu.between(tracing_nus.min(), tracing_nus.max())]
        .drop(labels="nu", axis=1)
        .to_numpy()
    )

    if not line_opacity_config.vald_linelist.use_vald_broadening:
        autoionization_lines = (
            lines_sorted_in_range.level_energy_upper
            > lines_sorted_in_range.ionization_energy
        ).values

        lines_sorted_in_range = lines_sorted_in_range[~autoionization_lines].copy()
        alphas_array = alphas_array[~autoionization_lines].copy()
        line_nus = line_nus[~autoionization_lines].copy()

    lines_sorted_in_range = lines_sorted_in_range.apply(
        pd.to_numeric
    )  # weird bug cropped up with ion_number being an object instead of an int

    gammas, doppler_widths = calculate_broadening(
        lines_sorted_in_range,
        stellar_model,
        stellar_plasma,
        line_opacity_config.broadening,
        use_vald_broadening=line_opacity_config.vald_linelist.use_vald_broadening
        and line_opacity_config.vald_linelist.use_linelist,  # don't try to use vald broadening if you don't use vald linelists at all
    )
    logger.info("Calculating line opacities at spectral points.")
    alpha_line_at_nu = calc_alan_entries(
        stellar_model.no_of_depth_points,
        tracing_nus.value,
        line_nus,
        doppler_widths,
        gammas,
        alphas_array,
    )

    return alpha_line_at_nu, gammas, doppler_widths


def calc_molecular_alpha_line_at_nu(
    stellar_plasma,
    stellar_model,
    tracing_nus,
    line_opacity_config,
):
    if line_opacity_config.disable:
        return 0, 0, 0

    lines = stellar_plasma.molecule_lines_from_linelist
    lines_sorted = lines.sort_values("nu")
    lines_sorted_in_range = lines_sorted[
        lines_sorted.nu.between(tracing_nus.min(), tracing_nus.max())
    ]

    line_nus = lines_sorted_in_range.nu.to_numpy()

    alphas_and_nu = stellar_plasma.molecule_alpha_line_from_linelist.sort_values("nu")

    alphas_array = (
        alphas_and_nu[alphas_and_nu.nu.between(tracing_nus.min(), tracing_nus.max())]
        .drop(labels="nu", axis=1)
        .to_numpy()
    )

    gammas, doppler_widths = calculate_molecule_broadening(
        lines_sorted_in_range,
        stellar_model,
        stellar_plasma,
        line_opacity_config.broadening,
    )
    alpha_line_at_nu = calc_alan_entries(
        stellar_model.no_of_depth_points,
        tracing_nus.value,
        line_nus,
        doppler_widths,
        gammas,
        alphas_array,
    )

    return alpha_line_at_nu, gammas, doppler_widths


@numba.njit(parallel=True)
def calc_alan_entries(
    no_of_depth_points,
    tracing_nus_values,
    line_nus,
    doppler_widths,
    gammas,
    alphas_array,
):
    """
    This is a helper function to prepare appropriate calling of the voigt profile calculation and allow for structure that
    can be parallelized.

    Parameters
    ----------
    no_of_depth_points : int
        The number of depth points.
    tracing_nus_values : array
        The frequencies at which to calculate the Alan entries.
    lines_nus : array
        The frequencies of the lines.
    doppler_widths : array
        The Doppler widths for each frequency.
    gammas : array
        The damping constants for each frequency.
    alphas_array : array
        The array of alpha values.

    Returns
    -------
    alpha_line_at_nu : array
        The calculated Alan entries for each frequency in `tracing_nus_values`.

    """
    tracing_nus_reversed = tracing_nus_values[::-1]
    alpha_line_at_nu = np.zeros((no_of_depth_points, len(tracing_nus_values)))
    d_nu = -np.diff(
        tracing_nus_values
    ).max()  # This is the smallest step size of the tracing_nus_values

    intermediate_arrays = np.zeros(
        (
            numba.config.NUMBA_DEFAULT_NUM_THREADS,
            no_of_depth_points,
            len(tracing_nus_values),
        )
    )

    for line_index in numba.prange(len(line_nus)):
        line_nu = line_nus[line_index]
        thread_id = numba.get_thread_id()
        for depth_point_index in range(no_of_depth_points):
            # If gamma is not for each depth point, we need to index it differently
            line_gamma = (
                gammas[line_index, depth_point_index]
                if gammas.shape[1] > 1
                else gammas[line_index, 0]
            )
            alpha = alphas_array[line_index, depth_point_index]
            doppler_width = doppler_widths[line_index, depth_point_index]

            # Now we need to find the closest frequency in the tracing_nus_values, which is in descending order
            closest_frequency_index = len(tracing_nus_values) - np.searchsorted(
                tracing_nus_reversed, line_nu
            )

            # We want to consider grid points within a certain range of the line_nu
            line_broadening = (
                ((line_gamma + doppler_width) * alpha) / d_nu * 20
            )  # Scale by alpha of the line
            line_broadening_range = max(10.0, line_broadening)  # Force a minimum range

            lower_freq_index = max(
                closest_frequency_index - int(line_broadening_range), 0
            )
            upper_freq_index = min(
                closest_frequency_index + int(line_broadening_range),
                len(tracing_nus_values),
            )

            delta_nus = tracing_nus_values[lower_freq_index:upper_freq_index] - line_nu

            intermediate_arrays[
                thread_id, depth_point_index, lower_freq_index:upper_freq_index
            ] += _calc_alan_entries(
                delta_nus,
                doppler_width,
                line_gamma,
                alpha,
            )

    # Combine the results from the intermediate arrays into the final array
    for thread_id in range(numba.config.NUMBA_DEFAULT_NUM_THREADS):
        alpha_line_at_nu += intermediate_arrays[thread_id]

    return alpha_line_at_nu


@numba.njit
def _calc_alan_entries(
    delta_nus,
    doppler_widths_at_depth_point,
    gammas_at_depth_point,
    alphas_at_depth_point,
):
    """
    Calculates the line opacity at a single frequency at a single depth point.

    Parameters
    ----------
    delta_nus : numpy.ndarray
        Difference between the frequency considered and the frequency of each
        line considered.
    doppler_widths_at_depth_point : numpy.ndarray
        The doppler width of each line considered in the at_depth_point considered.
    gammas_at_depth_point : numpy.ndarray
        The broadening parameter of each line considered at the depth point
        considered.
    alphas_at_depth_point : numpy.ndarray
        The total opacity of each line considered at the depth point considered.

    Returns
    -------
    float
        Line opacity.
    """
    phis = voigt_profile(
        delta_nus, doppler_widths_at_depth_point, gammas_at_depth_point
    )

    return phis * alphas_at_depth_point


def calc_alphas(
    stellar_plasma,
    stellar_model,
    stellar_radiation_field,
    opacity_config,
):
    """
    Calculates each opacity and adds it to the opacity dictionary contained in the radiation field.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    stellar_radiation_field stardis.radiation_field.base.RadiationField
        Contains the frequencies at which opacities are calculated. Also holds the resulting opacity information.
    opacity_config : tardis.io.configuration.config_reader.Configuration
        Opacity section of the STARDIS configuration.

    Returns
    -------
    alphas : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Total opacity at
        each depth point for each frequency in tracing_nus.
    """

    for (
        opacity_source,
        fpath,
    ) in opacity_config.file.items():  # Iterate through requested file opacities
        alpha_file = calc_alpha_file(
            stellar_plasma,
            stellar_model,
            stellar_radiation_field.frequencies,
            opacity_source,
            fpath,
        )
        stellar_radiation_field.opacities.opacities_dict[
            f"alpha_file_{opacity_source}"
        ] = alpha_file

    alpha_bf = calc_alpha_bf(
        stellar_plasma,
        stellar_model,
        stellar_radiation_field.frequencies,
        opacity_config.bf,
    )
    stellar_radiation_field.opacities.opacities_dict["alpha_bf"] = alpha_bf

    alpha_ff = calc_alpha_ff(
        stellar_plasma,
        stellar_model,
        stellar_radiation_field.frequencies,
        opacity_config.ff,
    )
    stellar_radiation_field.opacities.opacities_dict["alpha_ff"] = alpha_ff

    alpha_rayleigh = calc_alpha_rayleigh(
        stellar_plasma,
        stellar_model,
        stellar_radiation_field.frequencies,
        opacity_config.rayleigh,
    )
    stellar_radiation_field.opacities.opacities_dict["alpha_rayleigh"] = alpha_rayleigh

    alpha_electron = calc_alpha_electron(
        stellar_plasma,
        stellar_model,
        stellar_radiation_field.frequencies,
        opacity_config.disable_electron_scattering,
    )
    stellar_radiation_field.opacities.opacities_dict["alpha_electron"] = alpha_electron

    alpha_line_at_nu, gammas, doppler_widths = calc_alpha_line_at_nu(
        stellar_plasma,
        stellar_model,
        stellar_radiation_field.frequencies,
        opacity_config.line,
    )
    stellar_radiation_field.opacities.opacities_dict[
        "alpha_line_at_nu"
    ] = alpha_line_at_nu
    stellar_radiation_field.opacities.opacities_dict["alpha_line_at_nu_gammas"] = gammas
    stellar_radiation_field.opacities.opacities_dict[
        "alpha_line_at_nu_doppler_widths"
    ] = doppler_widths

    if opacity_config.line.include_molecules:
        (
            molecule_alpha_line_at_nu,
            molecule_gammas,
            molecule_doppler_widths,
        ) = calc_molecular_alpha_line_at_nu(
            stellar_plasma,
            stellar_model,
            stellar_radiation_field.frequencies,
            opacity_config.line,
        )

        stellar_radiation_field.opacities.opacities_dict[
            "molecule_alpha_line_at_nu"
        ] = molecule_alpha_line_at_nu
        stellar_radiation_field.opacities.opacities_dict[
            "molecule_alpha_line_at_nu_gammas"
        ] = molecule_gammas
        stellar_radiation_field.opacities.opacities_dict[
            "molecule_alpha_line_at_nu_doppler_widths"
        ] = molecule_doppler_widths

    alphas = stellar_radiation_field.opacities.calc_total_alphas()

    return alphas
