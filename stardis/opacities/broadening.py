import numpy as np
from astropy import constants as const
import numba


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
def calc_n_effective(ion_number, ionization_energy, level_energy):
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
    return np.sqrt(RYDBERG_ENERGY / (ionization_energy - level_energy)) * ion_number


@numba.njit
def calc_gamma_linear_stark(n_eff_upper, n_eff_lower, electron_density):
    """
    Calculates broadening parameter for linear Stark broadening.
    https://ui.adsabs.harvard.edu/abs/1978JQSRT..20..333S%2F/abstract

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
    ion_number, n_eff_upper, n_eff_lower, electron_density, temperature
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
        36 * PLANCK_CONSTANT * VACUUM_ELECTRIC_PERMITTIVITY * ion_number**4
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
    ion_number,
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
    https://doi.org/10.1093/mnras/136.4.381

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
    c6 = 6.46e-34 * (n_eff_upper**2 * (5 * n_eff_upper**2 + 1) - n_eff_lower**2 * (5 * n_eff_lower**2 + 1)) / (2 * ion_number**2 )

    gamma_van_der_waals = (
        17
        * (8 * BOLTZMANN_CONSTANT * temperature / (np.pi * h_mass)) ** 0.3
        * c6**0.4
        * h_density
    )

    return gamma_van_der_waals


@numba.njit
def calc_gamma(
    atomic_number,
    atomic_mass,
    ion_number,
    ionization_energy,
    upper_level_energy,
    lower_level_energy,
    A_ul,
    electron_density,
    temperature,
    h_density,
    he_abundance,
    h_mass,
    he_mass,
    linear_stark=True,
    quadratic_stark=True,
    van_der_waals=True,
    radiation=True
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
    linear_stark : bool, optional
        True if linear Stark broadening is to be considered, otherwise False.
        By default True.
    quadratic_stark : bool, optional
        True if quadratic Stark broadening is to be considered, otherwise
        False. By default True.
    van_der_waals : bool, optional
        True if Van Der Waals broadening is to be considered, otherwise False.
        By default True.

    Returns
    -------
    gamma_collision : float
        Total collision broadening parameter.
    """
    n_eff_upper = calc_n_effective(ion_number, ionization_energy, upper_level_energy)
    n_eff_lower = calc_n_effective(ion_number, ionization_energy, lower_level_energy)

    if (atomic_number == 1) and linear_stark:  # only for hydrogen
        gamma_linear_stark = calc_gamma_linear_stark(
            n_eff_upper, n_eff_lower, electron_density
        )
    else:
        gamma_linear_stark = 0

    if quadratic_stark:
        gamma_quadratic_stark = calc_gamma_quadratic_stark(
            ion_number, n_eff_upper, n_eff_lower, electron_density, temperature
        )
    else:
        gamma_quadratic_stark = 0

    if van_der_waals:
        gamma_van_der_waals = calc_gamma_van_der_waals(
            ion_number,
            atomic_mass,
            n_eff_upper,
            n_eff_lower,
            temperature,
            h_density,
            he_abundance,
            h_mass,
            he_mass,
        )
    else:
        gamma_van_der_waals = 0
        
    if radiation:
        gamma_radiation = A_ul
    else:
        gamma_radiation = 0

    gamma = gamma_linear_stark + gamma_quadratic_stark + gamma_van_der_waals + gamma_radiation

    return gamma


def calculate_broadening(
    lines_array,
    line_cols,
    no_shells,
    atomic_masses,
    h_mass,
    electron_densities,
    temperatures,
    h_densities,
    linear_stark=True,
    quadratic_stark=True,
    van_der_waals=True,
    radiation=True,
):
    
    line_nus = np.zeros(len(lines_array))
    gammas = np.zeros((len(lines_array), no_shells))
    doppler_widths = np.zeros((len(lines_array), no_shells))
    
    for i in range(len(lines_array)):
        
        atomic_number = int(lines_array[i, line_cols["atomic_number"]])
        atomic_mass = atomic_masses[atomic_number - 1]
        ion_number = int(lines_array[i, line_cols["ion_number"]])+1
        ionization_energy = lines_array[i, line_cols["ionization_energy"]]
        upper_level_energy = lines_array[i, line_cols["level_energy_upper"]]
        lower_level_energy = lines_array[i, line_cols["level_energy_lower"]]
        A_ul = lines_array[i, line_cols["A_ul"]]
        line_nu = lines_array[i, line_cols["nu"]]
        
        line_nus[i] = line_nu
        
        for j in range(no_shells):
            
            electron_density=electron_densities[j]
            temperature=temperatures[j]
            h_density=h_densities[j]
            
            gammas[i,j] = calc_gamma(
                atomic_number=atomic_number,
                atomic_mass=atomic_mass,
                ion_number=ion_number,
                ionization_energy=ionization_energy,
                upper_level_energy=upper_level_energy,
                lower_level_energy=lower_level_energy,
                A_ul=A_ul,
                electron_density=electron_density,
                temperature=temperature,
                h_density=h_density,
                he_abundance=0,
                h_mass=h_mass,
                he_mass=0,
                linear_stark=linear_stark,
                quadratic_stark=quadratic_stark,
                van_der_waals=van_der_waals,
                radiation=radiation,
            )
            
            doppler_widths[i,j] = doppler_width = calc_doppler_width(
                line_nu, temperature, atomic_mass
            )
        
    return line_nus, gammas, doppler_widths
