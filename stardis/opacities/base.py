import pandas as pd
import numpy as np

import numba

from astropy import units as u, constants as const

from stardis.opacities.broadening import calculate_broadening
from stardis.opacities.voigt import voigt_profile
from stardis.opacities.util import sigma_file, map_items_to_indices, get_number_density

VACUUM_ELECTRIC_PERMITTIVITY = 1 / (4 * np.pi)
BF_CONSTANT = (
    4
    * const.e.esu**2
    / (
        3
        * np.pi
        * np.sqrt(3)
        * VACUUM_ELECTRIC_PERMITTIVITY
        * const.m_e.cgs
        * const.c.cgs**2
        * const.Ryd.cgs
    )
)
FF_CONSTANT = (
    4
    / (3 * const.h.cgs * const.c.cgs)
    * (const.e.esu**2 / (4 * np.pi * VACUUM_ELECTRIC_PERMITTIVITY)) ** 3
    * np.sqrt(2 * np.pi / (3 * const.m_e.cgs**3 * const.k_B.cgs))
)


# H minus opacity
def calc_alpha_file(stellar_plasma, stellar_model, tracing_nus, species):

    tracing_lambdas = tracing_nus.to(u.AA, u.spectral()).value
    fv_geometry = stellar_model.fv_geometry
    temperatures = fv_geometry.temperatures
    alphas = np.zeros((len(temperatures), len(tracing_lambdas)))

    for spec, ff, fpath in species:

        sigmas = sigma_file(tracing_lambdas, temperatures, fpath)

        number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec
        )

        if number_density is None:
            #### WARN ####
            continue

        alpha_spec = sigmas * number_density[None].T

        alpha_file += alpha_spec

    return alpha_file


# rayleigh opacity
def calc_alpha_rayleigh(stellar_plasma, stellar_model, tracing_nus, species):

    fv_geometry = stellar_model.fv_geometry
    temperatures = fv_geometry.t.values
    EH = const.h.cgs * const.c.cgs * const.Ryd.cgs
    upper_bound = 2.3e15 * u.Hz
    tracing_nus[tracing_nus > upper_bound] = 0
    relative_nus = tracing_nus / (2 * EH)

    nu4 = relative_nus**4
    nu6 = relative_nus**6
    nu8 = relative_nus**8
    nu10 = relative_nus**10

    coefficient4 = np.zeros(len(temperatures))
    coefficient6 = np.zeros(len(temperatures))
    coefficient8 = np.zeros(len(temperatures))
    coefficient10 = np.zeros(len(temperatures))

    if "H" in species:  ##################???
        density = stellar_plasma.ion_number_density.loc[1, 0, 0]
        coefficent4 += 2 * density
        coefficent6 += 4 * density
        coefficent8 += 6 * density
        coefficent10 += 8 * density
    if "He" in species:  ##################???
        density = stellar_plasma.ion_number_density.loc[2, 0, 0]
        coefficent4 += 1 * density
        coefficent6 += 1 * density
        coefficent8 += 1 * density
        coefficent10 += 1 * density
    if "H2" in species:  ##################???
        density = stellar_plasma.h2_density
        coefficent4 += 1 * density
        coefficent6 += 1 * density
        coefficent8 += 1 * density
        coefficent10 += 1 * density

    alpha_rayleigh = (
        coefficient4[None].T * nu4
        + coefficient6[None].T * nu6
        + coefficient8[None].T * nu8
        + coefficient10[None].T * nu10
    )

    return alpha_rayleigh


# electron opacity
def calc_alpha_e(
    stellar_plasma,
    stellar_model,
    tracing_nus,
):
    """
    Calculates electron scattering opacity.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    stellar_model : stardis.io.base.StellarModel
        Stellar model.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.

    Returns
    -------
    alpha_e : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Electron scattering
        opacity in each shell for each frequency in tracing_nus.
    """

    fv_geometry = stellar_model.fv_geometry

    alpha_e_by_shell = (
        const.sigma_T.cgs.value * stellar_plasma.electron_densities.values
    )

    alpha_e = np.zeros([len(fv_geometry), len(tracing_nus)])
    for j in range(len(fv_geometry)):
        alpha_e[j] = alpha_e_by_shell[j]
    return alpha_e


# hydrogenic bound-free and free-free opacity
def calc_alpha_bf(stellar_plasma, stellar_model, tracing_nus, species):

    fv_geometry = stellar_model.fv_geometry

    inv_nu3 = tracing_nus ** (-3)
    alpha_bf = np.zeros([len(fv_geometry), len(tracing_nus)])

    for spec, gaunt_fpath, departure_fpath in species:

        # just for reading atomic number and ion number
        ion_number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec + "_bf"
        )

        if (
            (ion_number_density is None)
            or (atomic_number is none)
            or (ion_number is None)
        ):
            #### WARN ####
            continue

        ionization_energy = stellar_plasma.ionization_data.loc[
            (atomic_number, ion_number)
        ]

        alpha_spec = np.zeros([len(fv_geometry), len(tracing_nus)])

        levels = [
            (i, j, k)
            for i, j, k in stellar_plasma.levels
            if (i == atomic_number and j == ion_number)
        ]
        for level in levels:
            alpha_level = np.zeros([len(fv_geometry), len(tracing_nus)])
            cutoff_frequency = (
                ionization_energy - stellar_plasma.excitation_energy.loc[level_tuple]
            ) / const.h.cgs.value
            number_density = stellar_plasma.level_number_density.loc[level]
            for i in range(len(tracing_nus)):
                nu = tracing_nus[i]
                alpha_level[:, i] = calc_contribution_bf(
                    nu, cutoff_frequency, number_density, ion_number
                )

            alpha_spec += alpha_h_photo_level

        alpha_bf += alpha_spec

    alpha_bf *= inv_nu3

    return alpha_bf


@numba.njit
def calc_contribution_bf(nu, cutoff_frequency, number_density, ion_number):

    if nu >= cutoff_frequency:
        return (
            BF_CONSTANT * (ion_number + 1) ** 4 * number_density * cutoff_frequency**3
        )

    else:
        return 0 * number_density


def calc_alpha_ff(stellar_plasma, stellar_model, tracing_nus, species):

    fv_geometry = stellar_model.fv_geometry
    temperatures = fv_geometry.t.values

    inv_nu3 = tracing_nus ** (-3)
    alpha_bf = np.zeros([len(fv_geometry), len(tracing_nus)])

    for spec, gaunt_fpath, departure_fpath in species:

        alpha_spec = np.zeros([len(fv_geometry), len(tracing_nus)])

        number_density, atomic_number, ion_number = get_number_density(
            stellar_plasma, spec + "_ff"
        )

        if (
            (ion_number_density is None)
            or (atomic_number is none)
            or (ion_number is None)
        ):
            #### WARN ####
            continue

        for j in range(len(number_density)):
            alpha_spec[j] = number_density[j] / np.sqrt(temperatures)

        alpha_spec *= FF_CONSTANT * ion_number**2
        alpha_bf += alpha_spec

    alpha_bf *= inv_nu3

    return alpha_bf


def gaunt_times_departure(tracing_nus, temperatures, gaunt_fpath, departure_fpath):
    pass


# line opacity
def calc_alpha_line_at_nu(
    stellar_plasma,
    stellar_model,
    tracing_nus,
    broadening_methods=[
        "linear_stark",
        "quadratic_stark",
        "van_der_waals",
        "radiation",
    ],
    line_nu_min=0,
    line_nu_max=np.inf,
    line_range=None,
):

    linear_stark = "linear_stark" in broadening_methods
    quadratic_stark = "quadratic_stark" in broadening_methods
    van_der_waals = "van_der_waals" in broadening_methods
    radiation = "radiation" in broadening_methods

    fv_geometry = stellar_model.fv_geometry

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
    temperatures = fv_geometry.t.values
    electron_densities = stellar_plasma.electron_densities.values

    h_densities = stellar_plasma.ion_number_density.loc[1, 0].to_numpy()
    h_mass = atomic_masses[0]

    alphas_and_nu = stellar_plasma.alpha_line.sort_values("nu").reset_index(drop=True)
    alphas_and_nu_in_range = alphas_and_nu[
        alphas_and_nu.nu.between(line_nu_min, line_nu_max)
    ]
    alphas = alphas_and_nu_in_range.drop(labels="nu", axis=1)
    alphas_array = alphas.to_numpy()

    no_shells = len(fv_geometry)

    line_nus, gammas, doppler_widths = calculate_broadening(
        lines_array,
        line_cols,
        no_shells,
        atomic_masses,
        h_mass,
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
                line_start = line_nus.searchsorted(nu - line_range) + 1
                line_end = line_nus.searchsorted(nu + line_range) + 1
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

    return alpha_line_at_nu


@numba.njit
def calc_alan_entries(
    delta_nus,
    doppler_widths_in_shell,
    gammas_in_shell,
    alphas_in_shell,
):

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
    alpha_sources=["h_minus", "e", "h_photo", "line"],
    wbr_fpath=None,
    h_photo_levels=[1, 2, 3],
    h_photo_strength=7.91e-18,
    broadening_methods=[
        "doppler",
        "linear_stark",
        "quadratic_stark",
        "van_der_waals",
        "radiation",
    ],
    line_nu_min=0,
    line_nu_max=np.inf,
    line_range=None,
):
    """
    Calculates total opacity.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
        Stellar plasma.
    stellar_model : stardis.io.base.StellarModel
        Stellar model.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    alpha_sources: list, optional
        List of sources of opacity to be considered. Options are "h_minus",
        "e", "h_photo", and "line". By default all are included.
    wbr_fpath: str, optional
        Filepath to read H minus cross sections. By default None. Must be
        provided if H minus opacities are calculated.
    h_photo_levels: list, optional
        Level numbers considered for hydrogen photoionization. By default
        [1,2,3] which corresponds to the n=2 level of hydrogen with fine
        splitting.
    h_photo_strength : float, optional
        Coefficient to inverse cube term in equation for photoionization
        opacity, expressed in cm^2. By default 7.91e-18.
    broadening_methods : list, optional
        List of broadening mechanisms to be considered. Options are "doppler",
        "linear_stark", "quadratic_stark", "van_der_waals", and "radiation".
        By default all are included.

    Returns
    -------
    numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Total opacity in
        each shell for each frequency in tracing_nus.
    """

    if "h_minus" in alpha_sources:
        alpha_h_minus = calc_alpha_h_minus(
            stellar_plasma, stellar_model, tracing_nus, wbr_fpath
        )
    else:
        alpha_h_minus = 0

    if "e" in alpha_sources:
        alpha_e = calc_alpha_e(stellar_plasma, stellar_model, tracing_nus)
    else:
        alpha_e = 0

    if "h_photo" in alpha_sources:
        alpha_h_photo = calc_alpha_h_photo(
            stellar_plasma,
            stellar_model,
            tracing_nus,
            levels=h_photo_levels,
            strength=h_photo_strength,
        )
    else:
        h_photo = 0

    if "line" in alpha_sources:
        alpha_line_at_nu = calc_alpha_line_at_nu(
            stellar_plasma,
            stellar_model,
            tracing_nus,
            broadening_methods,
            line_nu_min,
            line_nu_max,
            line_range=line_range,
        )
    else:
        alpha_line_at_nu = 0

    return alpha_h_minus + alpha_e + alpha_h_photo + alpha_line_at_nu
