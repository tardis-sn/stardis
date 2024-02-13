import numpy as np
from astropy import constants as const
import math
import numba
from numba import cuda

GPUs_available = cuda.is_available()

# Commenting out until cupy is hooked up
# if GPUs_available:
#     import cupy as cp

PI = float(np.pi)
SPEED_OF_LIGHT = float(const.c.cgs.value)
BOLTZMANN_CONSTANT = float(const.k_B.cgs.value)
PLANCK_CONSTANT = float(const.h.cgs.value)
RYDBERG_ENERGY = float((const.h.cgs * const.c.cgs * const.Ryd.cgs).value)
ELEMENTARY_CHARGE = float(const.e.esu.value)
BOHR_RADIUS = float(const.a0.cgs.value)
VACUUM_ELECTRIC_PERMITTIVITY = 1.0 / (4.0 * PI)


@numba.njit
def _calc_doppler_width(nu_line, temperature, atomic_mass):
    """
    Calculates doppler width.
    https://ui.adsabs.harvard.edu/abs/2003rtsa.book.....R/

    Parameters
    ----------
    nu_line : float
        Frequency of line being considered.
    temperature : float
        Temperature of depth points being considered.
    atomic_mass : float
        Atomic mass of element being considered in grams.

    Returns
    -------
    float
    """
    nu_line, temperature, atomic_mass = (
        float(nu_line),
        float(temperature),
        float(atomic_mass),
    )

    return (
        nu_line
        / SPEED_OF_LIGHT
        * math.sqrt(2.0 * BOLTZMANN_CONSTANT * temperature / atomic_mass)
    )


@numba.vectorize(nopython=True)
def calc_doppler_width(nu_line, temperature, atomic_mass):
    return _calc_doppler_width(nu_line, temperature, atomic_mass)


@cuda.jit
def _calc_doppler_width_cuda(res, nu_line, temperature, atomic_mass):
    tid = cuda.grid(1)
    size = len(res)

    if tid < size:
        res[tid] = _calc_doppler_width(nu_line[tid], temperature[tid], atomic_mass[tid])


def calc_doppler_width_cuda(
    nu_line,
    temperature,
    atomic_mass,
    nthreads=256,
    ret_np_ndarray=False,
    dtype=float,
):
    arg_list = (
        nu_line,
        temperature,
        atomic_mass,
    )

    shortest_arg_idx = np.argmin(map(len, arg_list))
    size = len(arg_list[shortest_arg_idx])

    nblocks = 1 + (size // nthreads)

    arg_list = tuple(map(lambda v: cp.array(v, dtype=dtype), arg_list))

    res = cp.empty_like(arg_list[shortest_arg_idx], dtype=dtype)

    _calc_doppler_width_cuda[nblocks, nthreads](
        res,
        *arg_list,
    )

    return cp.asnumpy(res) if ret_np_ndarray else res


@numba.njit
def _calc_n_effective(ion_number, ionization_energy, level_energy):
    """
    Calculates the effective principal quantum number of an energy level.

    Parameters
    ----------
    ion_number : int
        Ion number of the atom being considered.
    ionization_energy : float
        Energy in ergs needed to ionize the atom in the ground state.
    level_energy : float
        Energy of the level in ergs, where the ground state is set to zero energy.

    Returns
    -------
    float
    """
    ion_number, ionization_energy, level_energy = (
        int(ion_number),
        float(ionization_energy),
        float(level_energy),
    )
    return math.sqrt(RYDBERG_ENERGY / (ionization_energy - level_energy)) * ion_number


@numba.vectorize(nopython=True)
def calc_n_effective(ion_number, ionization_energy, level_energy):
    return _calc_n_effective(
        ion_number,
        ionization_energy,
        level_energy,
    )


@cuda.jit
def _calc_n_effective_cuda(res, ion_number, ionization_energy, level_energy):
    tid = cuda.grid(1)
    size = len(res)

    if tid < size:
        res[tid] = _calc_n_effective(
            ion_number[tid],
            ionization_energy[tid],
            level_energy[tid],
        )


def calc_n_effective_cuda(
    ion_number,
    ionization_energy,
    level_energy,
    nthreads=256,
    ret_np_ndarray=False,
    dtype=float,
):
    arg_list = (
        ion_number,
        ionization_energy,
        level_energy,
    )

    shortest_arg_idx = np.argmin(map(len, arg_list))
    size = len(arg_list[shortest_arg_idx])

    nblocks = 1 + (size // nthreads)

    arg_list = tuple(map(lambda v: cp.array(v, dtype=dtype), arg_list))

    res = cp.empty_like(arg_list[shortest_arg_idx], dtype=dtype)

    _calc_n_effective_cuda[nblocks, nthreads](
        res,
        *arg_list,
    )

    return cp.asnumpy(res) if ret_np_ndarray else res


@numba.njit
def _calc_gamma_linear_stark(n_eff_upper, n_eff_lower, electron_density):
    """
    Calculates broadening parameter for linear Stark broadening.
    https://ui.adsabs.harvard.edu/abs/1978JQSRT..20..333S/

    Parameters
    ----------
    n_eff_upper : float
        Effective principal quantum number of upper level of transition.
    n_eff_lower : float
        Effective principal quantum number of lower level of transition.
    electron_density : float
        Electron density at depth point being considered.

    Returns
    -------
    gamma_linear_stark : float
        Broadening parameter for linear Stark broadening.
    """

    n_eff_upper, n_eff_lower, electron_density = (
        float(n_eff_upper),
        float(n_eff_lower),
        float(electron_density),
    )

    a1 = 0.642 if (n_eff_upper - n_eff_lower < 1.5) else 1.0

    gamma_linear_stark = (
        0.60
        * a1
        * (n_eff_upper**2 - n_eff_lower**2)
        * (electron_density ** (2.0 / 3.0))
    )

    return gamma_linear_stark


@numba.vectorize(nopython=True)
def calc_gamma_linear_stark(n_eff_upper, n_eff_lower, electron_density):
    return _calc_gamma_linear_stark(n_eff_upper, n_eff_lower, electron_density)


@cuda.jit
def _calc_gamma_linear_stark_cuda(res, n_eff_upper, n_eff_lower, electron_density):
    tid = cuda.grid(1)
    size = len(res)

    if tid < size:
        res[tid] = _calc_gamma_linear_stark(
            n_eff_upper[tid],
            n_eff_lower[tid],
            electron_density[tid],
        )


def calc_gamma_linear_stark_cuda(
    n_eff_upper,
    n_eff_lower,
    electron_density,
    nthreads=256,
    ret_np_ndarray=False,
    dtype=float,
):
    arg_list = (
        n_eff_upper,
        n_eff_lower,
        electron_density,
    )

    shortest_arg_idx = np.argmin(map(len, arg_list))
    size = len(arg_list[shortest_arg_idx])

    nblocks = 1 + (size // nthreads)

    arg_list = tuple(map(lambda v: cp.array(v, dtype=dtype), arg_list))

    res = cp.empty_like(arg_list[shortest_arg_idx], dtype=dtype)

    _calc_gamma_linear_stark_cuda[nblocks, nthreads](
        res,
        *arg_list,
    )

    return cp.asnumpy(res) if ret_np_ndarray else res


@numba.njit
def _calc_gamma_quadratic_stark(
    ion_number,
    n_eff_upper,
    n_eff_lower,
    electron_density,
    temperature,
):
    """
    Calculates broadening parameter for quadratic Stark broadening.
    Code adapted from https://github.com/tiagopereira/Transparency.jl/blob/5c8ee69/src/broadening.jl#L117
    Source (cited by Transparency.jl) https://doi.org/10.1017/CBO9781316036570
    Also cites Traving (1960) "Uber die Theorie der Druckverbreiterung von Spektrallinien" (not in stardis.bib)

    Parameters
    ----------
    ion_number : int
        Ion number of the ion being considered.
    n_eff_upper : float
        Effective principal quantum number of upper level of transition.
    n_eff_lower : float
        Effective principal quantum number of lower level of transition.
    electron_density : float
        Electron density at depth point being considered.
    temperature : float
        Temperature at depth point being considered.

    Returns
    -------
    gamma_quadratic_stark : float
        Broadening parameter for quadratic Stark broadening.
    """
    ion_number, n_eff_upper, n_eff_lower, electron_density, temperature = (
        int(ion_number),
        float(n_eff_upper),
        float(n_eff_lower),
        float(electron_density),
        float(temperature),
    )
    c4_prefactor = (
        ELEMENTARY_CHARGE * ELEMENTARY_CHARGE * BOHR_RADIUS * BOHR_RADIUS * BOHR_RADIUS
    ) / (
        36.0
        * PLANCK_CONSTANT
        * VACUUM_ELECTRIC_PERMITTIVITY
        * ion_number
        * ion_number
        * ion_number
        * ion_number
    )
    c4_term_1 = n_eff_upper * ((5.0 * n_eff_upper * n_eff_upper) + 1)
    c4_term_2 = n_eff_lower * ((5.0 * n_eff_lower * n_eff_lower) + 1)
    c4 = c4_prefactor * (c4_term_1 * c4_term_1 - c4_term_2 * c4_term_2)

    gamma_quadratic_stark = (
        1e19
        * BOLTZMANN_CONSTANT
        * electron_density
        * c4 ** (2.0 / 3.0)
        * temperature ** (1.0 / 6.0)
    )

    return gamma_quadratic_stark


@numba.vectorize(nopython=True)
def calc_gamma_quadratic_stark(
    ion_number,
    n_eff_upper,
    n_eff_lower,
    electron_density,
    temperature,
):
    return _calc_gamma_quadratic_stark(
        ion_number,
        n_eff_upper,
        n_eff_lower,
        electron_density,
        temperature,
    )


@cuda.jit
def _calc_gamma_quadratic_stark_cuda(
    res,
    ion_number,
    n_eff_upper,
    n_eff_lower,
    electron_density,
    temperature,
):
    tid = cuda.grid(1)
    size = len(res)

    if tid < size:
        res[tid] = _calc_gamma_quadratic_stark(
            ion_number[tid],
            n_eff_upper[tid],
            n_eff_lower[tid],
            electron_density[tid],
            temperature[tid],
        )


def calc_gamma_quadratic_stark_cuda(
    ion_number,
    n_eff_upper,
    n_eff_lower,
    electron_density,
    temperature,
    nthreads=256,
    ret_np_ndarray=False,
    dtype=float,
):
    arg_list = (
        ion_number,
        n_eff_upper,
        n_eff_lower,
        electron_density,
        temperature,
    )

    shortest_arg_idx = np.argmin(map(len, arg_list))
    size = len(arg_list[shortest_arg_idx])

    nblocks = 1 + (size // nthreads)

    arg_list = tuple(map(lambda v: cp.array(v, dtype=dtype), arg_list))

    res = cp.empty_like(arg_list[shortest_arg_idx], dtype=dtype)

    _calc_gamma_quadratic_stark_cuda[nblocks, nthreads](
        res,
        *arg_list,
    )

    return cp.asnumpy(res) if ret_np_ndarray else res


@numba.njit
def _calc_gamma_van_der_waals(
    ion_number,
    n_eff_upper,
    n_eff_lower,
    temperature,
    h_density,
    h_mass,
):
    """
    Calculates broadening parameter for van der Waals broadening.
    https://doi.org/10.1093/mnras/136.4.381

    Parameters
    ----------
    ion_number : int
        Ion number of the atom being considered.
    n_eff_upper : float
        Effective principal quantum number of upper level of transition.
    n_eff_lower : float
        Effective principal quantum number of lower level of transition.
    temperature : float
        Temperature of depth points being considered.
    h_density : float
        Number density of Hydrogen at depth point being considered.
    h_mass : float
        Atomic mass of Hydrogen in grams.

    Returns
    -------
    gamma_van_der_waals : float
        Broadening parameter for van der Waals broadening.
    """
    ion_number, n_eff_upper, n_eff_lower, temperature, h_density, h_mass = (
        int(ion_number),
        float(n_eff_upper),
        float(n_eff_lower),
        float(temperature),
        float(h_density),
        float(h_mass),
    )
    c6 = (
        6.46e-34
        * (
            (5 * n_eff_upper**4 + n_eff_upper**2)
            - (5 * n_eff_lower**4 + n_eff_lower**2)
        )
        / (2 * ion_number * ion_number)
    )

    gamma_van_der_waals = (
        17
        * (8 * BOLTZMANN_CONSTANT * temperature / (PI * h_mass)) ** 0.3
        * c6**0.4
        * h_density
    )

    return gamma_van_der_waals


@numba.vectorize(nopython=True)
def calc_gamma_van_der_waals(
    ion_number,
    n_eff_upper,
    n_eff_lower,
    temperature,
    h_density,
    h_mass,
):
    return _calc_gamma_van_der_waals(
        ion_number,
        n_eff_upper,
        n_eff_lower,
        temperature,
        h_density,
        h_mass,
    )


@cuda.jit
def _calc_gamma_van_der_waals_cuda(
    res,
    ion_number,
    n_eff_upper,
    n_eff_lower,
    temperature,
    h_density,
    h_mass,
):
    tid = cuda.grid(1)
    size = len(res)

    if tid < size:
        res[tid] = _calc_gamma_van_der_waals(
            ion_number[tid],
            n_eff_upper[tid],
            n_eff_lower[tid],
            temperature[tid],
            h_density[tid],
            h_mass[tid],
        )


def calc_gamma_van_der_waals_cuda(
    ion_number,
    n_eff_upper,
    n_eff_lower,
    temperature,
    h_density,
    h_mass,
    nthreads=256,
    ret_np_ndarray=False,
    dtype=float,
):
    arg_list = (
        ion_number,
        n_eff_upper,
        n_eff_lower,
        temperature,
        h_density,
        h_mass,
    )

    shortest_arg_idx = np.argmin(map(len, arg_list))
    size = len(arg_list[shortest_arg_idx])

    nblocks = 1 + (size // nthreads)

    arg_list = tuple(map(lambda v: cp.array(v, dtype=dtype), arg_list))

    res = cp.empty_like(arg_list[shortest_arg_idx], dtype=dtype)

    _calc_gamma_van_der_waals_cuda[nblocks, nthreads](
        res,
        *arg_list,
    )

    return cp.asnumpy(res) if ret_np_ndarray else res


@numba.njit
def calc_gamma(
    atomic_number,
    ion_number,
    ionization_energy,
    upper_level_energy,
    lower_level_energy,
    A_ul,
    electron_density,
    temperature,
    h_density,
    h_mass,
    linear_stark=True,
    quadratic_stark=True,
    van_der_waals=True,
    radiation=True,
):
    """
    Calculates total collision broadening parameter for a specific line
    and depth points.

    Parameters
    ----------
    atomic_number : int
        Atomic number of element being considered.
    ion_number : int
        Ion number of ion being considered.
    ionization_energy : float
        Ionization energy in ergs of ion being considered.
    upper_level_energy : float
        Energy in ergs of upper level of transition being considered.
    lower_level_energy : float
        Energy in ergs of upper level of transition being considered.
    A_ul : float
        Einstein A coefficient for the line being considered.
    electron_density : float
        Electron density at depth point being considered.
    temperature : float
        Temperature of depth points being considered.
    h_density : float
        Number density of Hydrogen at depth point being considered.
    h_mass : float
        Atomic mass of Hydrogen in grams.
    linear_stark : bool, optional
        True if linear Stark broadening is to be considered, otherwise False.
        By default True.
    quadratic_stark : bool, optional
        True if quadratic Stark broadening is to be considered, otherwise
        False. By default True.
    van_der_waals : bool, optional
        True if Van Der Waals broadening is to be considered, otherwise False.
        By default True.
    radiation : bool, optional
        True if radiation broadening is to be considered, otherwise False.
        By default True.

    Returns
    -------
    gamma_collision : float
        Total collision broadening parameter.
    """
    n_eff_upper = calc_n_effective(ion_number, ionization_energy, upper_level_energy)
    n_eff_lower = calc_n_effective(ion_number, ionization_energy, lower_level_energy)

    if (
        atomic_number == 1
    ) and linear_stark:  # only for hydrogen # why not all hydrogenic?
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
            n_eff_upper,
            n_eff_lower,
            temperature,
            h_density,
            h_mass,
        )
    else:
        gamma_van_der_waals = 0

    if radiation:
        gamma_radiation = A_ul
    else:
        gamma_radiation = 0

    gamma = (
        gamma_linear_stark
        + gamma_quadratic_stark
        + gamma_van_der_waals
        + gamma_radiation
    )

    return gamma


def calculate_broadening(
    lines,
    stellar_model,
    stellar_plasma,
    broadening_line_opacity_config,
):
    """
    Calculates broadening information for each line at each depth point.

    Parameters
    ----------
    lines_array :
        Array containing each line and properties of the line.
    line_cols : dict
        Matches the name of a quantity to its column index in lines_array.
    no_depth_points : int
        Number of depth pointss.
    atomic_masses : numpy.ndarray
        Atomic mass of all elements included in the simulation.
    electron_densities : numpy.ndarray
        Electron density at each depth point.
    temperatures : numpy.ndarray
        Temperature at each depth point.
    h_densities : numpy.ndarray
        Number density of hydrogen at each depth point.
    linear_stark : bool, optional
        True if linear Stark broadening is to be considered, otherwise False.
        By default True.
    quadratic_stark : bool, optional
        True if quadratic Stark broadening is to be considered, otherwise
        False. By default True.
    van_der_waals : bool, optional
        True if Van Der Waals broadening is to be considered, otherwise False.
        By default True.
    radiation : bool, optional
        True if radiation broadening is to be considered, otherwise False.
        By default True.

    Returns
    -------
    line_nus : numpy.ndarray
        Frequency of each line.
    gammas : numpy.ndarray
        Array of shape (no_of_lines, no_depth_points). Collisional broadening
        parameter of each line at each depth point.
    doppler_widths : numpy.ndarray
        Array of shape (no_of_lines, no_depth_points). Doppler width of each
        line at each depth point.
    """

    gammas = np.zeros((len(lines), stellar_model.no_of_depth_points))
    doppler_widths = np.zeros((len(lines), stellar_model.no_of_depth_points))

    linear_stark = "linear_stark" in broadening_line_opacity_config
    quadratic_stark = "quadratic_stark" in broadening_line_opacity_config
    van_der_waals = "van_der_waals" in broadening_line_opacity_config
    radiation = "radiation" in broadening_line_opacity_config

    h_mass = stellar_plasma.atomic_mass.loc[1]
    temperatures = stellar_model.temperatures.value
    h_densities = stellar_plasma.ion_number_density.loc[1, 0]
    electron_densities = stellar_plasma.electron_densities

    for i in range(len(lines)):
        for j in range(stellar_model.no_of_depth_points):

            gammas[i, j] = calc_gamma(
                atomic_number=lines.atomic_number.iloc[i],
                ion_number=lines.ion_number.iloc[i] + 1,
                ionization_energy=lines.ionization_energy.iloc[i],
                upper_level_energy=lines.level_energy_upper.iloc[i],
                lower_level_energy=lines.level_energy_lower.iloc[i],
                A_ul=lines.A_ul.iloc[i],
                electron_density=electron_densities.loc[j],
                temperature=temperatures[j],
                h_density=h_densities[j],
                h_mass=h_mass,
                linear_stark=linear_stark,
                quadratic_stark=quadratic_stark,
                van_der_waals=van_der_waals,
                radiation=radiation,
            )

    doppler_widths = calc_doppler_width(
        lines.nu.values[:, np.newaxis],
        temperatures,
        stellar_plasma.atomic_mass.values[
            lines.atomic_number.values - 1,
            np.newaxis,
        ],
    )

    return gammas, doppler_widths
