import numpy as np
from astropy import constants as const
import numba

from stardis.opacities.voigt import voigt_profile


SPEED_OF_LIGHT = const.c.cgs.value
BOLTZMANN_CONSTANT = const.k_B.cgs.value


@numba.jit
def calc_doppler_width(nu_line, temperature, atomic_mass):
    return (
        nu_line
        / SPEED_OF_LIGHT
        * np.sqrt(2 * BOLTZMANN_CONSTANT * temperature / atomic_mass)
    )


@numba.jit
def calc_gamma_linear_stark():
    """
    Calculates broadening parameter for linear Stark broadening.???

    Parameters
    ----------

    Returns
    -------
    gamma_linear_stark : float
        Broadening parameter for linear Stark broadening.
    """
    gamma_linear_stark = 0

    return gamma_linear_stark


@numba.jit
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


@numba.jit
def calc_gamma_quadratic_stark():
    """
    Calculates broadening parameter for quadratic Stark broadening.???

    Parameters
    ----------

    Returns
    -------
    gamma_quadratic_stark : float
        Broadening parameter for quadratic Stark broadening.
    """
    gamma_quadratic_stark = 0

    return gamma_quadratic_stark


@numba.jit
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


@numba.jit
def calc_gamma_collision():
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
    gamma_linear_stark = calc_gamma_linear_stark()
    gamma_resonance = calc_gamma_resonance()
    gamma_quadratic_stark = calc_gamma_quadratic_stark()
    gamma_van_der_waals = calc_gamma_van_der_waals()

    gamma_collision = (
        gamma_linear_stark
        + gamma_resonance
        + gamma_quadratic_stark
        + gamma_van_der_waals
    )

    return gamma_collision


@numba.jit
def assemble_phis(
    atomic_masses,
    temperatures,
    nu,
    lines_considered_nu,
    lines_considered_atomic_num,
    lines_considered_A_ul,
):
    """
    Puts together several line profiles at a single frequency for all shells.

    Parameters
    ----------
    atomic_masses : numpy.ndarray
        Atomic masses present in the stellar plasma.
    temperatures : numpy.ndarray
        Temperatures of all shells.
    nu : float
        Frequency at which line profiles are being evaluated.
    lines_considered_nu : numpy.ndarray
        Frequencies of all lines considered.
    lines_considered_atomic_num : numpy.ndarray
        Atomic numbers of all lines considered.
    lines_considered_A_ul : numpy.ndarray
        A_ul of all lines considered.

    Returns
    -------
    phis : numpy.ndarray
        Array of shape (no_of_lines_considered, no_of_shells). Line profiles
        of each line in each shell evaluated at the specified frequency.
    """
    phis = np.zeros((len(lines_considered_nu), len(temperatures)))

    for j in range(len(temperatures)):
        for i in range(len(lines_considered_nu)):
            delta_nu = nu - lines_considered_nu[i]

            atomic_mass = atomic_masses[lines_considered_atomic_num[i]]
            doppler_width = calc_doppler_width(
                lines_considered_nu[i], temperatures[j], atomic_mass
            )

            gamma_collision = calc_gamma_collision()
            gamma = lines_considered_A_ul[i] + gamma_collision

            phis[i, j] = voigt_profile(delta_nu, doppler_width, gamma)

    return phis
