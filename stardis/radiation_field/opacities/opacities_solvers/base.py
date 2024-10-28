import pandas as pd
import numpy as np
from pathlib import Path
import numba

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
    n_threads=1,
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

    line_range = line_opacity_config.broadening_range

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

    gammas, doppler_widths = calculate_broadening(
        lines_sorted_in_range,
        stellar_model,
        stellar_plasma,
        line_opacity_config.broadening,
        use_vald_broadening=line_opacity_config.vald_linelist.use_vald_broadening,
    )

    delta_nus = tracing_nus.value - line_nus[:, np.newaxis]

    # If no broadening range, compute the contribution of every line at every frequency.
    h_lines_indices = None
    line_range_value = None

    # If there is a broadening range, first make sure the range is in frequency units, and then iterate through each frequency to calculate the contribution of each line within the broadening range.
    if (
        line_range is not None
    ):  # This if statement block appropriately handles if the broadening range is in frequency or wavelength units.
        h_lines_indices = (
            lines_sorted_in_range.atomic_number == 1
        ).to_numpy()  # Hydrogen lines are much broader than other lines, so they need special treatment to ignore the broadening range.
        if line_range.unit.physical_type == "length":
            lambdas = tracing_nus.to(u.AA, equivalencies=u.spectral())
            lambdas_plus_broadening_range = lambdas + line_range.to(u.AA)
            nus_plus_broadening_range = lambdas_plus_broadening_range.to(
                u.Hz, equivalencies=u.spectral()
            )
            line_range_value = (tracing_nus - nus_plus_broadening_range).value
        elif line_range.unit.physical_type == "frequency":
            line_range_value = line_range.to(u.Hz).value
        else:
            raise ValueError(
                "Broadening range must be in units of length or frequency."
            )

    if n_threads == 1:  # Single threaded
        alpha_line_at_nu = calc_alan_entries(
            stellar_model.no_of_depth_points,
            tracing_nus.value,
            delta_nus,
            doppler_widths,
            gammas,
            alphas_array,
            line_range_value,
            h_lines_indices,
        )

    else:  # Parallel threaded
        alpha_line_at_nu = calc_alan_entries_parallel(
            stellar_model.no_of_depth_points,
            tracing_nus.value,
            delta_nus,
            doppler_widths,
            gammas,
            alphas_array,
            line_range_value,
            h_lines_indices,
        )

    return alpha_line_at_nu, gammas, doppler_widths


def calc_molecular_alpha_line_at_nu(
    stellar_plasma,
    stellar_model,
    tracing_nus,
    line_opacity_config,
    n_threads=1,
):
    if line_opacity_config.disable:
        return 0, 0, 0

    line_range = line_opacity_config.broadening_range

    lines = stellar_plasma.molecule_lines_from_linelist
    lines_sorted = lines.sort_values("nu")
    lines_sorted_in_range = lines_sorted[
        lines_sorted.nu.between(tracing_nus.min(), tracing_nus.max())
    ]
    line_nus = lines_sorted_in_range.nu.to_numpy()

    alphas_and_nu = stellar_plasma.molecule_alpha_line_from_linelist

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

    delta_nus = tracing_nus.value - line_nus[:, np.newaxis]

    line_range_value = None

    # If there is a broadening range, first make sure the range is in frequency units, and then iterate through each frequency to calculate the contribution of each line within the broadening range.
    if (
        line_range is not None
    ):  # This if statement block appropriately handles if the broadening range is in frequency or wavelength units.
        h_lines_indices = np.full(
            len(lines_sorted_in_range), False
        )  # This is wonky but necessary for the calc_alan_entries function
        if line_range.unit.physical_type == "length":
            lambdas = tracing_nus.to(u.AA, equivalencies=u.spectral())
            lambdas_plus_broadening_range = lambdas + line_range.to(u.AA)
            nus_plus_broadening_range = lambdas_plus_broadening_range.to(
                u.Hz, equivalencies=u.spectral()
            )
            line_range_value = (tracing_nus - nus_plus_broadening_range).value
        elif line_range.unit.physical_type == "frequency":
            line_range_value = line_range.to(u.Hz).value
        else:
            raise ValueError(
                "Broadening range must be in units of length or frequency."
            )

    if n_threads == 1:  # Single threaded
        alpha_line_at_nu = calc_alan_entries(
            stellar_model.no_of_depth_points,
            tracing_nus.value,
            delta_nus,
            doppler_widths,
            gammas,
            alphas_array,
            line_range_value,
            h_lines_indices,
        )

    else:  # Parallel threaded
        alpha_line_at_nu = calc_alan_entries_parallel(
            stellar_model.no_of_depth_points,
            tracing_nus.value,
            delta_nus,
            doppler_widths,
            gammas,
            alphas_array,
            line_range_value,
            h_lines_indices,
        )

    return alpha_line_at_nu, gammas, doppler_widths


@numba.njit(parallel=True)
def calc_alan_entries_parallel(
    no_of_depth_points,
    tracing_nus_values,
    delta_nus,
    doppler_widths,
    gammas,
    alphas_array,
    broadening_range=None,
    h_lines_indices=None,
):
    """
    This is a helper function to appropriately parallelize the alpha line at nu calculation.
    It is analagous to the calc_alan_entries function, but with the addition of the numba.prange decorator.

    Parameters
    ----------
    no_of_depth_points : int
        The number of depth points.
    tracing_nus_values : array
        The frequencies at which to calculate the Alan entries.
    delta_nus : array
        The differences between the frequencies and the line frequencies.
    doppler_widths : array
        The Doppler widths for each frequency.
    gammas : array
        The damping constants for each frequency.
    alphas_array : array
        The array of alpha values.
    broadening_range : array, optional
        The broadening range for each frequency. If provided, only frequencies within
        this range will be considered. Default is None.
    h_lines_indices : array, optional
        The indices of the hydrogen lines. If provided, these lines will always be
        considered, regardless of the broadening range. Default is None.

    Returns
    -------
    alpha_line_at_nu : array
        The calculated Alan entries for each frequency in `tracing_nus_values`.

    """

    alpha_line_at_nu = np.zeros((no_of_depth_points, len(tracing_nus_values)))

    if broadening_range is None:
        for frequency_index in numba.prange(len(tracing_nus_values)):
            alpha_line_at_nu[:, frequency_index] = _calc_alan_entries(
                delta_nus[:, frequency_index, np.newaxis],
                doppler_widths,
                gammas,
                alphas_array,
            )

    else:
        for frequency_index in numba.prange(len(tracing_nus_values)):
            broadening_mask = (
                np.abs(delta_nus[:, frequency_index])
                < broadening_range[frequency_index]
            )
            broadening_mask = np.logical_or(broadening_mask, h_lines_indices)

            alpha_line_at_nu[:, frequency_index] = _calc_alan_entries(
                delta_nus[:, frequency_index, np.newaxis][broadening_mask],
                doppler_widths[broadening_mask],
                gammas[broadening_mask],
                alphas_array[broadening_mask],
            )

    return alpha_line_at_nu


@numba.njit
def calc_alan_entries(
    no_of_depth_points,
    tracing_nus_values,
    delta_nus,
    doppler_widths,
    gammas,
    alphas_array,
    broadening_range=None,
    h_lines_indices=None,
):
    """
    This is a helper function to prepare appropriate calling of the voigt profile calculation and allow for structure that
    can be parallelized. In the no broadening case it simply calls the voigt profile calculator with the appropriate structure.
    In the broadening case it first creates a mask to only consider lines within the broadening range, and then calls the function
    only on those lines. The variable line would make the input matrix not square, and prohibits easy access with numba, so an
    explicit for loop must be called.

    Parameters
    ----------
    no_of_depth_points : int
        The number of depth points.
    tracing_nus_values : array
        The frequencies at which to calculate the Alan entries.
    delta_nus : array
        The differences between the frequencies and the line frequencies.
    doppler_widths : array
        The Doppler widths for each frequency.
    gammas : array
        The damping constants for each frequency.
    alphas_array : array
        The array of alpha values.
    broadening_range : array, optional
        The broadening range for each frequency. If provided, only frequencies within
        this range will be considered. Default is None.
    h_lines_indices : array, optional
        The indices of the hydrogen lines. If provided, these lines will always be
        considered, regardless of the broadening range. Default is None.

    Returns
    -------
    alpha_line_at_nu : array
        The calculated Alan entries for each frequency in `tracing_nus_values`.

    """

    alpha_line_at_nu = np.zeros((no_of_depth_points, len(tracing_nus_values)))

    if broadening_range is None:
        for frequency_index in range(len(tracing_nus_values)):
            alpha_line_at_nu[:, frequency_index] = _calc_alan_entries(
                delta_nus[:, frequency_index, np.newaxis],
                doppler_widths,
                gammas,
                alphas_array,
            )

    else:
        for frequency_index in range(len(tracing_nus_values)):
            broadening_mask = (
                np.abs(delta_nus[:, frequency_index])
                < broadening_range[frequency_index]
            )
            broadening_mask = np.logical_or(broadening_mask, h_lines_indices)

            alpha_line_at_nu[:, frequency_index] = _calc_alan_entries(
                delta_nus[:, frequency_index, np.newaxis][broadening_mask],
                doppler_widths[broadening_mask],
                gammas[broadening_mask],
                alphas_array[broadening_mask],
            )

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

    return np.sum(phis * alphas_at_depth_point, axis=0)


def calc_alphas(
    stellar_plasma,
    stellar_model,
    stellar_radiation_field,
    opacity_config,
    n_threads=1,
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
        n_threads,
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
            n_threads,
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
