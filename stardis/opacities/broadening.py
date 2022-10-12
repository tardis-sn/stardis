import numpy as np
from astropy import constants as const
import numba

from stardis.opacities.voigt import voigt_profile


SPEED_OF_LIGHT = const.c.cgs.value
BOLTZMANN_CONSTANT = const.k_B.cgs.value
PLANCK_CONSTANT = const.h.cgs.value
RYDBERG_ENERGY = (const.h.cgs * const.c.cgs * const.Ryd.cgs).value
ELEMENTARY_CHARGE = const.e.esu.value
BOHR_RADIUS = const.a0.cgs.value
VACUUM_ELECTRIC_PERMITTIVITY = 1 / (4 * np.pi)


@numba.njit
def calc_doppler_width(nu_line, temperature, atomic_mass):
    """
    Calculates doppler width.

    Parameters
    ----------
    nu_line : float
        Frequency of line being considered.
    temperature : float
        Temperature of shell being considered.
    atomic_mass : float
        Atomic mass of element being considered in grams.

    Returns
    -------
    float
    """
    return (
        nu_line
        / SPEED_OF_LIGHT
        * np.sqrt(2 * BOLTZMANN_CONSTANT * temperature / atomic_mass)
    )


@numba.njit
def calc_n_effective(atomic_number, ionization_energy, level_energy):
    """
    Calculates the effective principal quantum number of an energy level.

    Parameters
    ----------
    atomic_number : int
        Atomic number of the atom being considered.
    ionization_energy : float
        Energy in ergs needed to ionize the atom in the ground state.
    level_energy : float
        Energy of the level in ergs, where the ground state is set to zero energy.

    Returns
    -------
    float
    """
    return np.sqrt(RYDBERG_ENERGY / (ionization_energy - level_energy)) * atomic_number


@numba.njit
def calc_gamma_linear_stark(n_eff_upper, n_eff_lower, electron_density):
    """
    Calculates broadening parameter for linear Stark broadening.

    Parameters
    ----------
    n_eff_upper : float
        Effective principal quantum number of upper level of transition.
    n_eff_lower : float
        Effective principal quantum number of lower level of transition.
    electron_density : float
        Electron density in shell being considered.

    Returns
    -------
    gamma_linear_stark : float
        Broadening parameter for linear Stark broadening.
    """

    if n_eff_upper - n_eff_lower < 1.5:
        a1 = 0.642
    else:
        a1 = 1

    gamma_linear_stark = (
        0.51
        * a1
        * (n_eff_upper**2 - n_eff_lower**2)
        * (electron_density ** (2 / 3))
    )

    return gamma_linear_stark


@numba.njit
def calc_gamma_quadratic_stark(
    atomic_number, n_eff_upper, n_eff_lower, electron_density, temperature
):
    """
    Calculates broadening parameter for quadratic Stark broadening.

    Parameters
    ----------
    atomic_number : int
        Atomic number of the atom being considered.
    n_eff_upper : float
        Effective principal quantum number of upper level of transition.
    n_eff_lower : float
        Effective principal quantum number of lower level of transition.
    electron_density : float
        Electron density in shell being considered.
    temperature : float
        Temperature of shell being considered.

    Returns
    -------
    gamma_quadratic_stark : float
        Broadening parameter for quadratic Stark broadening.
    """
    c4_prefactor = (ELEMENTARY_CHARGE**2 * BOHR_RADIUS**3) / (
        36 * PLANCK_CONSTANT * VACUUM_ELECTRIC_PERMITTIVITY * atomic_number**4
    )
    c4 = c4_prefactor * (
        (n_eff_upper * ((5 * n_eff_upper**2) + 1)) ** 2
        - (n_eff_lower * ((5 * n_eff_lower**2) + 1)) ** 2
    )

    gamma_quadratic_stark = (
        10**19
        * BOLTZMANN_CONSTANT
        * electron_density
        * c4 ** (2 / 3)
        * temperature ** (1 / 6)
    )

    return gamma_quadratic_stark


@numba.njit
def calc_gamma_van_der_waals(
    atomic_number,
    atomic_mass,
    n_eff_upper,
    n_eff_lower,
    temperature,
    h_density,
    he_abundance,
    h_mass,
    he_mass,
):
    """
    Calculates broadening parameter for van der Waals broadening.

    Parameters
    ----------
    atomic_number : int
        Atomic number of the atom being considered.
    n_eff_upper : float
        Effective principal quantum number of upper level of transition.
    n_eff_lower : float
        Effective principal quantum number of lower level of transition.
    temperature : float
        Temperature of shell being considered.
    h_density : float
        Number density of Hydrogen in shell being considered.
    he_abundance : float
        Fractional abundance (by mass) of Helium in shell being considered.
    h_mass : float
        Atomic mass of Hydrogen in grams.
    he_mass : float
        Atomic mass of Helium in grams.

    Returns
    -------
    gamma_van_der_waals : float
        Broadening parameter for van der Waals broadening.
    """
    c6_prefactor = (
        5.625
        * ELEMENTARY_CHARGE**2
        * BOHR_RADIUS**5
        / (PLANCK_CONSTANT * VACUUM_ELECTRIC_PERMITTIVITY)
    )

    c6 = c6_prefactor * atomic_number**6 / (n_eff_upper**4 - n_eff_lower**4)

    gamma_van_der_waals = (
        8.08
        * (
            (1 + atomic_mass / h_mass) ** 0.3
            + he_abundance * (1 + atomic_mass / he_mass) ** 2
        )
        * (8 * BOLTZMANN_CONSTANT / (np.pi * atomic_mass)) ** 0.3
        * c6**0.4
        * temperature**0.3
        * h_density
    )

    return gamma_van_der_waals


@numba.njit
def calc_gamma_collision(
    atomic_number,
    atomic_mass,
    ion_number,
    ionization_energy,
    upper_level_energy,
    lower_level_energy,
    electron_density,
    temperature,
    h_density,
    he_abundance,
    h_mass,
    he_mass,
):
    """
    Calculates total collision broadening parameter by adding up all
    contributing broadening parameters.

    Parameters
    ----------
    atomic_number : int
        Atomic number of element being considered.
    atomic_mass : float
        Atomic mass of element being considered in grams.
    ion_number : int
        Ion number of ion being considered.
    ionization_energy : float
        Ionization energy in ergs of ion being considered.
    upper_level_energy : float
        Energy in ergs of upper level of transition being considered.
    lower_level_energy : float
        Energy in ergs of upper level of transition being considered.
    electron_density : float
        Electron density in shell being considered.
    temperature : float
        Temperature of shell being considered.
    h_density : float
        Number density of Hydrogen in shell being considered.
    he_abundance : float
        Fractional abundance (by mass) of Helium in shell being considered.
    h_mass : float
        Atomic mass of Hydrogen in grams.
    he_mass : float
        Atomic mass of Helium in grams.

    Returns
    -------
    gamma_collision : float
        Total collision broadening parameter.
    """
    n_eff_upper = calc_n_effective(atomic_number, ionization_energy, upper_level_energy)
    n_eff_lower = calc_n_effective(atomic_number, ionization_energy, lower_level_energy)

    if atomic_number - ion_number == 1:  # species is hydrogenic
        gamma_linear_stark = calc_gamma_linear_stark(
            n_eff_upper, n_eff_lower, electron_density
        )
    else:
        gamma_linear_stark = 0

    gamma_quadratic_stark = calc_gamma_quadratic_stark(
        atomic_number, n_eff_upper, n_eff_lower, electron_density, temperature
    )

    gamma_van_der_waals = calc_gamma_van_der_waals(
        atomic_number,
        atomic_mass,
        n_eff_upper,
        n_eff_lower,
        temperature,
        h_density,
        he_abundance,
        h_mass,
        he_mass,
    )

    gamma_collision = gamma_linear_stark + gamma_quadratic_stark + gamma_van_der_waals

    return gamma_collision


@numba.njit
def assemble_phis(
    atomic_masses,
    temperatures,
    electron_densities,
    h_densities,
    he_abundances,
    nu,
    lines_considered,
    line_cols,
):
    """
    Puts together several line profiles at a single frequency for all shells.

    Parameters
    ----------
    atomic_masses : numpy.ndarray
        Atomic masses of all atoms considered in the model in grams.
    temperatures : numpy.ndarray
        Temperatures of all shells.
    electron_densities : numpy.ndarray
        Electron Densities of all shells.
    h_densities : numpy.ndarray
        Neutral hydrogen number density in all shells.
    he_abundances : numpy.ndarray
        Fractional abundance (by mass) of Helium in all shells.
    nu : float
        Frequency at which line profiles are being evaluated.
    lines_considered : numpy.ndarray
        Attributes of all considered lines as 2D array of shape
        (no_of_lines_considered, no_of_line_cols).
    line_cols : numba.typed.Dict
        Column names of lines_considered mapped to indices.

    Returns
    -------
    phis : numpy.ndarray
        Array of shape (no_of_lines_considered, no_of_shells). Line profiles
        of each line in each shell evaluated at the specified frequency.
    """

    phis = np.zeros((len(lines_considered), len(temperatures)))

    for j in range(len(temperatures)):  # iterate over shells (columns)
        for i in range(len(lines_considered)):  # iterate over lines considered (rows)
            delta_nu = nu - lines_considered[i, line_cols["nu"]]

            atomic_number = int(lines_considered[i, line_cols["atomic_number"]])
            atomic_mass = atomic_masses[atomic_number - 1]
            doppler_width = calc_doppler_width(
                lines_considered[i, line_cols["nu"]], temperatures[j], atomic_mass
            )

            gamma_collision = calc_gamma_collision(
                atomic_number=atomic_number,
                atomic_mass=atomic_mass,
                ion_number=int(lines_considered[i, line_cols["ion_number"]]),
                ionization_energy=lines_considered[i, line_cols["ionization_energy"]],
                upper_level_energy=lines_considered[i, line_cols["level_energy_upper"]],
                lower_level_energy=lines_considered[i, line_cols["level_energy_lower"]],
                electron_density=electron_densities[j],
                temperature=temperatures[j],
                h_density=h_densities[j],
                he_abundance=he_abundances[j],
                h_mass=atomic_masses[0],
                he_mass=atomic_masses[1],
            )
            gamma = (
                lines_considered[i, line_cols["A_ul"]] + gamma_collision
            )  # includes radiation broadening

            phis[i, j] = voigt_profile(delta_nu, doppler_width, gamma)

    return phis
