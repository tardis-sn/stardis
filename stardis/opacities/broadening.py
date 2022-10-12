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
# FIXME: const.eps0.cgs.value doesn't work, is this ok?
VACUUM_ELECTRIC_PERMITTIVITY = 1


@numba.njit
def calc_doppler_width(nu_line, temperature, atomic_mass):
    return (
        nu_line
        / SPEED_OF_LIGHT
        * np.sqrt(2 * BOLTZMANN_CONSTANT * temperature / atomic_mass)
    )


@numba.njit
def calc_n_effective(atomic_number, ionization_energy, level_energy):
    return np.sqrt(RYDBERG_ENERGY / (ionization_energy - level_energy)) * atomic_number


@numba.njit
def calc_gamma_linear_stark(n_eff_upper, n_eff_lower, electron_density):
    """
    Calculates broadening parameter for linear Stark broadening.???

    Parameters
    ----------

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


@numba.njit
def calc_gamma_quadratic_stark(
    atomic_number, n_eff_upper, n_eff_lower, electron_density, temperature
):
    """
    Calculates broadening parameter for quadratic Stark broadening.???

    Parameters
    ----------

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


@numba.njit
def calc_gamma_collision(
    atomic_number,
    ion_number,
    ionization_energy,
    upper_level_energy,
    lower_level_energy,
    electron_density,
    temperature,
):
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
    n_eff_upper = calc_n_effective(atomic_number, ionization_energy, upper_level_energy)
    n_eff_lower = calc_n_effective(atomic_number, ionization_energy, lower_level_energy)

    if atomic_number - ion_number == 1:  # species is hydrogenic
        gamma_linear_stark = calc_gamma_linear_stark(
            n_eff_upper, n_eff_lower, electron_density
        )
    else:
        gamma_linear_stark = 0

    gamma_resonance = calc_gamma_resonance()

    gamma_quadratic_stark = calc_gamma_quadratic_stark(
        atomic_number, n_eff_upper, n_eff_lower, electron_density, temperature
    )

    gamma_van_der_waals = calc_gamma_van_der_waals()

    gamma_collision = (
        gamma_linear_stark
        + gamma_resonance
        + gamma_quadratic_stark
        + gamma_van_der_waals
    )

    return gamma_collision


@numba.njit
def assemble_phis(
    atomic_masses,
    temperatures,
    electron_densities,
    nu,
    lines_considered,
    line_cols,
):
    """
    Puts together several line profiles at a single frequency for all shells.

    Parameters
    ----------
    atomic_masses : numpy.ndarray
        Atomic masses present in the stellar plasma.
    temperatures : numpy.ndarray
        Temperatures of all shells.
    electron_densities : numpy.ndarray
        Electron Densities of all shells.
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
                ion_number=int(lines_considered[i, line_cols["ion_number"]]),
                ionization_energy=lines_considered[i, line_cols["ionization_energy"]],
                upper_level_energy=lines_considered[i, line_cols["level_energy_upper"]],
                lower_level_energy=lines_considered[i, line_cols["level_energy_lower"]],
                electron_density=electron_densities[j],
                temperature=temperatures[j],
            )
            gamma = lines_considered[i, line_cols["A_ul"]] + gamma_collision

            phis[i, j] = voigt_profile(delta_nu, doppler_width, gamma)

    return phis
