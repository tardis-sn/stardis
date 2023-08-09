import pandas as pd
import numpy as np

import numba

from tardis.io.config_reader import ConfigurationNameSpace

from astropy import units as u, constants as const

from stardis.opacities.broadening import calculate_broadening
from stardis.opacities.voigt import voigt_profile
from stardis.opacities.util import sigma_file, map_items_to_indices, get_number_density


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


# H minus opacity
def calc_alpha_file(stellar_plasma, stellar_model, tracing_nus, species):
    """
    Calculates opacities when a cross-section file is provided.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    species : tardis.io.config_reader.Configuration
        Dictionary (in the form of a Configuration object) containing all
        species and the cross-section files for which opacity is to be
        calculated.

    Returns
    -------
    alpha_file : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). File opacity in
        each shell for each frequency in tracing_nus.
    """

    tracing_lambdas = tracing_nus.to(u.AA, u.spectral()).value
    temperatures = stellar_model.temperatures.value
    alpha_file = np.zeros((len(temperatures), len(tracing_lambdas)))

    for spec, fpath in species.items():
        sigmas = sigma_file(tracing_lambdas, temperatures, fpath)

        number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec
        )

        alpha_spec = sigmas * np.array(number_density)[None].T

        alpha_file += alpha_spec

    return alpha_file


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
        Array of shape (no_of_shells, no_of_frequencies). Rayleigh scattering
        opacity in each shell for each frequency in tracing_nus.
    """

    temperatures = stellar_model.temperatures.value
    nu_H = const.c.cgs * const.Ryd.cgs
    upper_bound = 2.3e15 * u.Hz
    tracing_nus[tracing_nus > upper_bound] = 0
    relative_nus = tracing_nus.value / (2 * nu_H.value)

    nu4 = relative_nus**4
    nu6 = relative_nus**6
    nu8 = relative_nus**8

    # This seems super hacky. We initialize arrays in many places using the shape of temperatures or geometry. Revisit later to standardize.
    coefficient4 = np.zeros(len(temperatures))
    coefficient6 = np.zeros(len(temperatures))
    coefficient8 = np.zeros(len(temperatures))

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
        Array of shape (no_of_shells, no_of_frequencies). Electron scattering
        opacity in each shell for each frequency in tracing_nus.
    """

    if disable_electron_scattering:
        return 0
    geometry = (
        stellar_model.geometry.r
    )  # Eventually change when considering other coordinate systems than radial1d.

    alpha_electron_by_shell = (
        const.sigma_T.cgs.value * stellar_plasma.electron_densities.values
    )

    alpha_electron = np.zeros([len(geometry), len(tracing_nus)])
    for j in range(len(geometry)):
        alpha_electron[j] = alpha_electron_by_shell[j]

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
    species : tardis.io.config_reader.Configuration
        Dictionary (in the form of a Configuration object) containing all
        species for which bound-free opacity is to be calculated.

    Returns
    -------
    alpha_bf : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Bound-free opacity in
        each shell for each frequency in tracing_nus.
    """
    # This implementation will only work with 1D.
    geometry = (
        stellar_model.geometry.r
    )  # This is just used to initialize the opacitiy array. There might be a better place to grab the shape from?

    inv_nu3 = tracing_nus.value ** (-3)
    alpha_bf = np.zeros((len(geometry), len(tracing_nus)))

    for spec, dct in species.items():
        # just for reading atomic number and ion number
        ion_number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec + "_bf"
        )

        ionization_energy = stellar_plasma.ionization_data.loc[
            (atomic_number, ion_number + 1)
        ]

        alpha_spec = np.zeros((len(geometry), len(tracing_nus)))

        levels = [
            (i, j, k)
            for i, j, k in stellar_plasma.levels
            if (i == atomic_number and j == ion_number)
        ]
        for level in levels:
            alpha_level = np.zeros((len(geometry), len(tracing_nus)))
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
    coefficient of a single frequency in each shell.

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
    species : tardis.io.config_reader.Configuration
        Dictionary (in the form of a Configuration object) containing all
        species for which free-free opacity is to be calculated.

    Returns
    -------
    alpha_ff : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Free-free opacity in
        each shell for each frequency in tracing_nus.
    """

    temperatures = stellar_model.temperatures.value
    geometry = (
        stellar_model.geometry.r
    )  # Will need to change when considering anything other than radial1d.

    inv_nu3 = tracing_nus.value ** (-3)
    alpha_ff = np.zeros([len(geometry), len(tracing_nus)])

    for spec, dct in species.items():
        alpha_spec = np.zeros([len(geometry), len(tracing_nus)])

        number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec + "_ff"
        )

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
    line_opacity_config : tardis.io.config_reader.Configuration
        Line opacity section of the STARDIS configuration.

    Returns
    -------
    alpha_line_at_nu : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Line opacity in
        each shell for each frequency in tracing_nus.
    gammas : numpy.ndarray
        Array of shape (no_of_lines, no_of_shells). Collisional broadening
        parameter of each line in each shell.
    doppler_widths : numpy.ndarray
        Array of shape (no_of_lines, no_of_shells). Doppler width of each
        line in each shell.
    """

    if line_opacity_config.disable:
        return 0, 0, 0

    broadening_methods = line_opacity_config.broadening
    _nu_min = line_opacity_config.min.to(u.Hz, u.spectral())
    _nu_max = line_opacity_config.max.to(u.Hz, u.spectral())
    line_nu_min = min(_nu_min, _nu_max)
    line_nu_max = max(_nu_min, _nu_max)
    line_range = line_opacity_config.broadening_range

    linear_stark = "linear_stark" in broadening_methods
    quadratic_stark = "quadratic_stark" in broadening_methods
    van_der_waals = "van_der_waals" in broadening_methods
    radiation = "radiation" in broadening_methods

    lines = stellar_plasma.lines.reset_index()  # bring lines in ascending order of nu

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

    line_cols = map_items_to_indices(lines.columns.to_list())

    lines_sorted = lines.sort_values("nu").reset_index(drop=True)
    lines_sorted_in_range = lines_sorted[
        lines_sorted.nu.between(line_nu_min, line_nu_max)
    ]
    lines_array = lines_sorted_in_range.to_numpy()

    atomic_masses = stellar_plasma.atomic_mass.values
    temperatures = stellar_model.temperatures.value
    electron_densities = stellar_plasma.electron_densities.values

    h_densities = stellar_plasma.ion_number_density.loc[1, 0].to_numpy()

    alphas_and_nu = stellar_plasma.alpha_line.sort_values("nu").reset_index(drop=True)
    alphas_and_nu_in_range = alphas_and_nu[
        alphas_and_nu.nu.between(line_nu_min, line_nu_max)
    ]
    alphas = alphas_and_nu_in_range.drop(labels="nu", axis=1)
    alphas_array = alphas.to_numpy()

    no_shells = len(stellar_model.geometry.r)

    line_nus, gammas, doppler_widths = calculate_broadening(
        lines_array,
        line_cols,
        no_shells,
        atomic_masses,
        electron_densities,
        temperatures,
        h_densities,
        linear_stark=linear_stark,
        quadratic_stark=quadratic_stark,
        van_der_waals=van_der_waals,
        radiation=radiation,
    )

    alpha_line_at_nu = np.zeros((no_shells, len(tracing_nus)))

    for i in range(len(tracing_nus)):
        nu = tracing_nus[i].value
        delta_nus = nu - line_nus

        for j in range(no_shells):
            gammas_in_shell = gammas[:, j]
            doppler_widths_in_shell = doppler_widths[:, j]
            alphas_in_shell = alphas_array[:, j]

            if line_range is None:
                alpha_line_at_nu[j, i] = calc_alan_entries(
                    delta_nus,
                    doppler_widths_in_shell,
                    gammas_in_shell,
                    alphas_in_shell,
                )

            else:
                line_range_value = line_range.to(u.Hz).value
                line_start = line_nus.searchsorted(nu - line_range_value) + 1
                line_end = line_nus.searchsorted(nu + line_range_value) + 1
                delta_nus_considered = delta_nus[line_start:line_end]
                gammas_considered = gammas_in_shell[line_start:line_end]
                doppler_widths_considered = doppler_widths_in_shell[line_start:line_end]
                alphas_considered = alphas_in_shell[line_start:line_end]
                alpha_line_at_nu[j, i] = calc_alan_entries(
                    delta_nus_considered,
                    doppler_widths_considered,
                    gammas_considered,
                    alphas_considered,
                )

    return alpha_line_at_nu, gammas, doppler_widths


@numba.njit
def calc_alan_entries(
    delta_nus,
    doppler_widths_in_shell,
    gammas_in_shell,
    alphas_in_shell,
):
    """
    Calculates the line opacity at a single frequency in a single shell.

    Parameters
    ----------
    delta_nus : numpy.ndarray
        Difference between the frequency considered and the frequency of each
        line considered.
    doppler_widths_in_shell : numpy.ndarray
        The doppler width of each line considered in the shell considered.
    gammas_in_shell : numpy.ndarray
        The broadening parameter of each line considered in the shell
        considered.
    alphas_in_shell : numpy.ndarray
        The total opacity of each line considered in the shell considered.

    Returns
    -------
    float
        Line opacity.
    """

    phis = np.zeros(len(delta_nus))

    for k in range(len(delta_nus)):
        delta_nu = np.abs(delta_nus[k])
        doppler_width = doppler_widths_in_shell[k]
        gamma = gammas_in_shell[k]

        phis[k] = voigt_profile(delta_nu, doppler_width, gamma)

    return np.sum(phis * alphas_in_shell)


def calc_alphas(
    stellar_plasma,
    stellar_model,
    tracing_nus,
    opacity_config,
):
    """
    Calculates total opacity.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    opacity_config : tardis.io.config_reader.Configuration
        Opacity section of the STARDIS configuration.

    Returns
    -------
    alphas : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Total opacity in
        each shell for each frequency in tracing_nus.
    gammas : numpy.ndarray
        Array of shape (no_of_lines, no_of_shells). Collisional broadening
        parameter of each line in each shell.
    doppler_widths : numpy.ndarray
        Array of shape (no_of_lines, no_of_shells). Doppler width of each
        line in each shell.
    """

    alpha_file = calc_alpha_file(
        stellar_plasma,
        stellar_model,
        tracing_nus,
        opacity_config.file,
    )
    alpha_bf = calc_alpha_bf(
        stellar_plasma,
        stellar_model,
        tracing_nus,
        opacity_config.bf,
    )
    alpha_ff = calc_alpha_ff(
        stellar_plasma,
        stellar_model,
        tracing_nus,
        opacity_config.ff,
    )
    alpha_rayleigh = calc_alpha_rayleigh(
        stellar_plasma,
        stellar_model,
        tracing_nus,
        opacity_config.rayleigh,
    )
    alpha_electron = calc_alpha_electron(
        stellar_plasma,
        stellar_model,
        tracing_nus,
        opacity_config.disable_electron_scattering,
    )
    alpha_line_at_nu, gammas, doppler_widths = calc_alpha_line_at_nu(
        stellar_plasma,
        stellar_model,
        tracing_nus,
        opacity_config.line,
    )

    alphas = (
        alpha_file
        + alpha_bf
        + alpha_ff
        + alpha_rayleigh
        + alpha_electron
        + alpha_line_at_nu
    )

    ### TODO create opacity_dict to return
    return alphas, gammas, doppler_widths
