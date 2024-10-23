import pytest
import numpy as np
from astropy import constants as const, units as u

from stardis.radiation_field.opacities.opacities_solvers.broadening import (
    calc_doppler_width,
    _calc_doppler_width_cuda,
    calc_doppler_width_cuda,
    calc_n_effective,
    _calc_n_effective_cuda,
    calc_n_effective_cuda,
    calc_gamma_linear_stark,
    _calc_gamma_linear_stark_cuda,
    calc_gamma_linear_stark_cuda,
    calc_gamma_quadratic_stark,
    _calc_gamma_quadratic_stark_cuda,
    calc_gamma_quadratic_stark_cuda,
    calc_gamma_van_der_waals,
    _calc_gamma_van_der_waals_cuda,
    calc_gamma_van_der_waals_cuda,
    rotation_broadening,
)

GPUs_available = False  # cuda.is_available()

if GPUs_available:
    import cupy as cp


PI = np.pi
SPEED_OF_LIGHT = const.c.cgs.value
BOLTZMANN_CONSTANT = const.k_B.cgs.value
PLANCK_CONSTANT = const.h.cgs.value
RYDBERG_ENERGY = (const.h.cgs * const.c.cgs * const.Ryd.cgs).value
ELEMENTARY_CHARGE = const.e.esu.value
BOHR_RADIUS = const.a0.cgs.value
VACUUM_ELECTRIC_PERMITTIVITY = 1 / (4 * PI)


@pytest.mark.parametrize(
    "calc_doppler_width_sample_values_input_nu_line, calc_doppler_width_sample_values_input_temperature, calc_doppler_width_sample_values_input_atomic_mass, calc_doppler_width_sample_values_expected_result",
    [
        (
            SPEED_OF_LIGHT,
            0.5,
            BOLTZMANN_CONSTANT,
            1.0,
        ),
        (
            np.array(2 * [SPEED_OF_LIGHT]),
            np.array(2 * [0.5]),
            np.array(2 * [BOLTZMANN_CONSTANT]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_doppler_width_sample_values(
    calc_doppler_width_sample_values_input_nu_line,
    calc_doppler_width_sample_values_input_temperature,
    calc_doppler_width_sample_values_input_atomic_mass,
    calc_doppler_width_sample_values_expected_result,
):
    assert np.allclose(
        calc_doppler_width(
            calc_doppler_width_sample_values_input_nu_line,
            calc_doppler_width_sample_values_input_temperature,
            calc_doppler_width_sample_values_input_atomic_mass,
            0.0,
        ),
        calc_doppler_width_sample_values_expected_result,
    )  # No microturbulence for legacy reasons - 0.0


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_doppler_width_cuda_unwrapped_sample_values_input_nu_line, calc_doppler_width_cuda_unwrapped_sample_values_input_temperature, calc_doppler_width_cuda_unwrapped_sample_values_input_atomic_mass, calc_doppler_width_cuda_unwrapped_sample_values_expected_result",
    [
        (
            np.array(2 * [SPEED_OF_LIGHT]),
            np.array(2 * [0.5]),
            np.array(2 * [BOLTZMANN_CONSTANT]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_doppler_width_cuda_unwrapped_sample_values(
    calc_doppler_width_cuda_unwrapped_sample_values_input_nu_line,
    calc_doppler_width_cuda_unwrapped_sample_values_input_temperature,
    calc_doppler_width_cuda_unwrapped_sample_values_input_atomic_mass,
    calc_doppler_width_cuda_unwrapped_sample_values_expected_result,
):
    arg_list = (
        calc_doppler_width_cuda_unwrapped_sample_values_input_nu_line,
        calc_doppler_width_cuda_unwrapped_sample_values_input_temperature,
        calc_doppler_width_cuda_unwrapped_sample_values_input_atomic_mass,
    )

    arg_list = tuple(map(cp.array, arg_list))
    result_values = cp.empty_like(arg_list[0])

    nthreads = 256
    length = len(calc_doppler_width_cuda_unwrapped_sample_values_expected_result)
    nblocks = 1 + (length // nthreads)

    _calc_doppler_width_cuda[nblocks, nthreads](result_values, *arg_list)

    assert np.allclose(
        cp.asnumpy(result_values),
        calc_doppler_width_cuda_unwrapped_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_doppler_width_cuda_sample_values_input_nu_line, calc_doppler_width_cuda_sample_values_input_temperature, calc_doppler_width_cuda_sample_values_input_atomic_mass, calc_doppler_width_cuda_wrapped_sample_cuda_values_expected_result",
    [
        (
            np.array(2 * [SPEED_OF_LIGHT]),
            np.array(2 * [0.5]),
            np.array(2 * [BOLTZMANN_CONSTANT]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_doppler_width_cuda_wrapped_sample_cuda_values(
    calc_doppler_width_cuda_sample_values_input_nu_line,
    calc_doppler_width_cuda_sample_values_input_temperature,
    calc_doppler_width_cuda_sample_values_input_atomic_mass,
    calc_doppler_width_cuda_wrapped_sample_cuda_values_expected_result,
):
    arg_list = (
        calc_doppler_width_cuda_sample_values_input_nu_line,
        calc_doppler_width_cuda_sample_values_input_temperature,
        calc_doppler_width_cuda_sample_values_input_atomic_mass,
    )
    assert np.allclose(
        calc_doppler_width_cuda(*map(cp.asarray, arg_list)),
        calc_doppler_width_cuda_wrapped_sample_cuda_values_expected_result,
    )


@pytest.mark.parametrize(
    "calc_n_effective_sample_values_input_ion_number, calc_n_effective_sample_values_input_ionization_energy, calc_n_effective_sample_values_input_level_energy, calc_n_effective_sample_values_expected_result",
    [
        (
            1.0,
            RYDBERG_ENERGY,
            0,
            1.0,
        ),
        (
            np.array(2 * [1]),
            np.array(2 * [RYDBERG_ENERGY]),
            np.array(2 * [0]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_n_effective_sample_values(
    calc_n_effective_sample_values_input_ion_number,
    calc_n_effective_sample_values_input_ionization_energy,
    calc_n_effective_sample_values_input_level_energy,
    calc_n_effective_sample_values_expected_result,
):
    assert np.allclose(
        calc_n_effective(
            calc_n_effective_sample_values_input_ion_number,
            calc_n_effective_sample_values_input_ionization_energy,
            calc_n_effective_sample_values_input_level_energy,
        ),
        calc_n_effective_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_n_effective_cuda_unwrapped_sample_values_input_ion_number, calc_n_effective_cuda_unwrapped_sample_values_input_ionization_energy, calc_n_effective_cuda_unwrapped_sample_values_input_level_energy, calc_n_effective_cuda_unwrapped_sample_values_expected_result",
    [
        (
            np.array(2 * [1]),
            np.array(2 * [RYDBERG_ENERGY]),
            np.array(2 * [0]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_n_effective_cuda_unwrapped_sample_values(
    calc_n_effective_cuda_unwrapped_sample_values_input_ion_number,
    calc_n_effective_cuda_unwrapped_sample_values_input_ionization_energy,
    calc_n_effective_cuda_unwrapped_sample_values_input_level_energy,
    calc_n_effective_cuda_unwrapped_sample_values_expected_result,
):
    arg_list = (
        calc_n_effective_cuda_unwrapped_sample_values_input_ion_number,
        calc_n_effective_cuda_unwrapped_sample_values_input_ionization_energy,
        calc_n_effective_cuda_unwrapped_sample_values_input_level_energy,
    )

    arg_list = tuple(map(cp.array, arg_list))
    result_values = cp.empty_like(arg_list[0])

    nthreads = 256
    length = len(calc_n_effective_cuda_unwrapped_sample_values_expected_result)
    nblocks = 1 + (length // nthreads)

    _calc_n_effective_cuda[nblocks, nthreads](result_values, *arg_list)

    assert np.allclose(
        cp.asnumpy(result_values),
        calc_n_effective_cuda_unwrapped_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_n_effective_cuda_sample_values_input_ion_number, calc_n_effective_cuda_sample_values_input_ionization_energy, calc_n_effective_cuda_sample_values_input_level_energy, calc_n_effective_cuda_wrapped_sample_cuda_values_expected_result",
    [
        (
            np.array(2 * [1]),
            np.array(2 * [RYDBERG_ENERGY]),
            np.array(2 * [0]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_n_effective_cuda_wrapped_sample_cuda_values(
    calc_n_effective_cuda_sample_values_input_ion_number,
    calc_n_effective_cuda_sample_values_input_ionization_energy,
    calc_n_effective_cuda_sample_values_input_level_energy,
    calc_n_effective_cuda_wrapped_sample_cuda_values_expected_result,
):
    arg_list = (
        calc_n_effective_cuda_sample_values_input_ion_number,
        calc_n_effective_cuda_sample_values_input_ionization_energy,
        calc_n_effective_cuda_sample_values_input_level_energy,
    )
    assert np.allclose(
        calc_n_effective_cuda(*map(cp.asarray, arg_list)),
        calc_n_effective_cuda_wrapped_sample_cuda_values_expected_result,
    )


@pytest.mark.parametrize(
    "calc_gamma_linear_stark_sample_values_input_n_eff_upper, calc_gamma_linear_stark_sample_values_input_n_eff_lower, calc_gamma_linear_stark_sample_values_input_electron_density, calc_gamma_linear_stark_sample_values_expected_result",
    [
        (
            1,
            0,
            (0.60 * 0.642) ** (-3 / 2),
            1.0,
        ),
        (
            np.array(2 * [1]),
            np.array(2 * [0]),
            np.array(2 * [(0.60 * 0.642) ** (-3 / 2)]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_gamma_linear_stark_sample_values(
    calc_gamma_linear_stark_sample_values_input_n_eff_upper,
    calc_gamma_linear_stark_sample_values_input_n_eff_lower,
    calc_gamma_linear_stark_sample_values_input_electron_density,
    calc_gamma_linear_stark_sample_values_expected_result,
):
    assert np.allclose(
        calc_gamma_linear_stark(
            calc_gamma_linear_stark_sample_values_input_n_eff_upper,
            calc_gamma_linear_stark_sample_values_input_n_eff_lower,
            calc_gamma_linear_stark_sample_values_input_electron_density,
        ),
        calc_gamma_linear_stark_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_upper, calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_lower, calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_electron_density, calc_gamma_linear_stark_cuda_unwrapped_sample_values_expected_result",
    [
        (
            np.array(2 * [1]),
            np.array(2 * [0]),
            np.array(2 * [(0.60 * 0.642) ** (-3 / 2)]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_gamma_linear_stark_cuda_unwrapped_sample_values(
    calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_upper,
    calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_lower,
    calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_electron_density,
    calc_gamma_linear_stark_cuda_unwrapped_sample_values_expected_result,
):
    arg_list = (
        calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_upper,
        calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_lower,
        calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_electron_density,
    )

    arg_list = tuple(map(cp.array, arg_list))
    result_values = cp.empty_like(arg_list[0], dtype=float)

    nthreads = 256
    length = len(calc_gamma_linear_stark_cuda_unwrapped_sample_values_expected_result)
    nblocks = 1 + (length // nthreads)

    _calc_gamma_linear_stark_cuda[nblocks, nthreads](result_values, *arg_list)
    print(result_values)
    assert np.allclose(
        cp.asnumpy(result_values),
        calc_gamma_linear_stark_cuda_unwrapped_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_upper, calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_lower, calc_gamma_linear_stark_cuda_wrapped_sample_values_input_electron_density, calc_gamma_linear_stark_cuda_wrapped_sample_values_expected_result",
    [
        (
            np.array(2 * [1]),
            np.array(2 * [0]),
            np.array(2 * [(0.60 * 0.642) ** (-3 / 2)]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_doppler_width_cuda_wrapped_sample_cuda_values(
    calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_upper,
    calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_lower,
    calc_gamma_linear_stark_cuda_wrapped_sample_values_input_electron_density,
    calc_gamma_linear_stark_cuda_wrapped_sample_values_expected_result,
):
    arg_list = (
        calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_upper,
        calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_lower,
        calc_gamma_linear_stark_cuda_wrapped_sample_values_input_electron_density,
    )
    assert np.allclose(
        calc_gamma_linear_stark_cuda(*map(cp.asarray, arg_list)),
        calc_gamma_linear_stark_cuda_wrapped_sample_values_expected_result,
    )


c4_prefactor = (ELEMENTARY_CHARGE**2 * BOHR_RADIUS**3) / (
    36.0 * PLANCK_CONSTANT * VACUUM_ELECTRIC_PERMITTIVITY
)


@pytest.mark.parametrize(
    "calc_gamma_quadratic_stark_sample_values_input_ion_number, calc_gamma_quadratic_stark_sample_values_input_n_eff_upper, calc_gamma_quadratic_stark_sample_values_input_n_eff_lower, calc_gamma_quadratic_stark_sample_values_input_electron_density,  calc_gamma_quadratic_stark_sample_values_input_temperature,calc_gamma_quadratic_stark_sample_values_expected_result",
    [
        (
            1,  # ion_number
            1.0,  # n_eff_upper
            0.0,  # n_eff_lower
            1.0e-19
            / BOLTZMANN_CONSTANT
            * (36 * c4_prefactor) ** (-2.0 / 3.0),  # electron_density
            1.0,  # temperature
            1,  # Expected output
        ),
        (
            np.array(2 * [1], dtype=int),
            np.array(2 * [1.0]),
            np.array(2 * [0.0]),
            np.array(
                2 * [1.0e-19 / BOLTZMANN_CONSTANT * (36 * c4_prefactor) ** (-2.0 / 3.0)]
            ),
            np.array(2 * [1.0]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_gamma_quadratic_stark_sample_values(
    calc_gamma_quadratic_stark_sample_values_input_ion_number,
    calc_gamma_quadratic_stark_sample_values_input_n_eff_upper,
    calc_gamma_quadratic_stark_sample_values_input_n_eff_lower,
    calc_gamma_quadratic_stark_sample_values_input_electron_density,
    calc_gamma_quadratic_stark_sample_values_input_temperature,
    calc_gamma_quadratic_stark_sample_values_expected_result,
):
    assert np.allclose(
        calc_gamma_quadratic_stark(
            calc_gamma_quadratic_stark_sample_values_input_ion_number,
            calc_gamma_quadratic_stark_sample_values_input_n_eff_upper,
            calc_gamma_quadratic_stark_sample_values_input_n_eff_lower,
            calc_gamma_quadratic_stark_sample_values_input_electron_density,
            calc_gamma_quadratic_stark_sample_values_input_temperature,
        ),
        calc_gamma_quadratic_stark_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_gamma_quadratic_stark_sample_values_input_ion_number, calc_gamma_quadratic_stark_sample_values_input_n_eff_upper, calc_gamma_quadratic_stark_sample_values_input_n_eff_lower, calc_gamma_quadratic_stark_sample_values_input_electron_density,  calc_gamma_quadratic_stark_sample_values_input_temperature,calc_gamma_quadratic_stark_sample_values_expected_result",
    [
        (
            np.array(2 * [1], dtype=int),
            np.array(2 * [1.0]),
            np.array(2 * [0.0]),
            np.array(
                2 * [1.0e-19 / BOLTZMANN_CONSTANT * (36 * c4_prefactor) ** (-2.0 / 3.0)]
            ),
            np.array(2 * [1.0]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_gamma_quadratic_stark_cuda_unwrapped_sample_values(
    calc_gamma_quadratic_stark_sample_values_input_ion_number,
    calc_gamma_quadratic_stark_sample_values_input_n_eff_upper,
    calc_gamma_quadratic_stark_sample_values_input_n_eff_lower,
    calc_gamma_quadratic_stark_sample_values_input_electron_density,
    calc_gamma_quadratic_stark_sample_values_input_temperature,
    calc_gamma_quadratic_stark_sample_values_expected_result,
):
    arg_list = (
        calc_gamma_quadratic_stark_sample_values_input_ion_number,
        calc_gamma_quadratic_stark_sample_values_input_n_eff_upper,
        calc_gamma_quadratic_stark_sample_values_input_n_eff_lower,
        calc_gamma_quadratic_stark_sample_values_input_electron_density,
        calc_gamma_quadratic_stark_sample_values_input_temperature,
    )

    arg_list = tuple(map(cp.array, arg_list))
    result_values = cp.empty_like(arg_list[0])

    nthreads = 256
    length = len(calc_gamma_quadratic_stark_sample_values_expected_result)
    nblocks = 1 + (length // nthreads)

    _calc_gamma_quadratic_stark_cuda[nblocks, nthreads](result_values, *arg_list)

    assert np.allclose(
        cp.asnumpy(result_values),
        calc_gamma_quadratic_stark_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_gamma_quadratic_stark_sample_values_input_ion_number, calc_gamma_quadratic_stark_sample_values_input_n_eff_upper, calc_gamma_quadratic_stark_sample_values_input_n_eff_lower, calc_gamma_quadratic_stark_sample_values_input_electron_density,  calc_gamma_quadratic_stark_sample_values_input_temperature,calc_gamma_quadratic_stark_sample_values_expected_result",
    [
        (
            np.array(2 * [1], dtype=int),
            np.array(2 * [1.0]),
            np.array(2 * [0.0]),
            np.array(
                2 * [1.0e-19 / BOLTZMANN_CONSTANT * (36 * c4_prefactor) ** (-2.0 / 3.0)]
            ),
            np.array(2 * [1.0]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_gamma_quadratic_stark_cuda_wrapped_sample_cuda_values(
    calc_gamma_quadratic_stark_sample_values_input_ion_number,
    calc_gamma_quadratic_stark_sample_values_input_n_eff_upper,
    calc_gamma_quadratic_stark_sample_values_input_n_eff_lower,
    calc_gamma_quadratic_stark_sample_values_input_electron_density,
    calc_gamma_quadratic_stark_sample_values_input_temperature,
    calc_gamma_quadratic_stark_sample_values_expected_result,
):
    arg_list = (
        calc_gamma_quadratic_stark_sample_values_input_ion_number,
        calc_gamma_quadratic_stark_sample_values_input_n_eff_upper,
        calc_gamma_quadratic_stark_sample_values_input_n_eff_lower,
        calc_gamma_quadratic_stark_sample_values_input_electron_density,
        calc_gamma_quadratic_stark_sample_values_input_temperature,
    )
    assert np.allclose(
        calc_gamma_quadratic_stark_cuda(*map(cp.asarray, arg_list)),
        calc_gamma_quadratic_stark_sample_values_expected_result,
    )


@pytest.mark.parametrize(
    "calc_gamma_van_der_waals_sample_values_input_ion_number,calc_gamma_van_der_waals_sample_values_input_n_eff_upper,calc_gamma_van_der_waals_sample_values_input_n_eff_lower, calc_gamma_van_der_waals_sample_values_input_temperature, calc_gamma_van_der_waals_sample_values_input_h_density,calc_gamma_van_der_waals_sample_values_expected_result",
    [
        (
            1,  # ion_number
            1.0,  # n_eff_upper
            0.0,  # n_eff_lower
            np.pi / 8 / BOLTZMANN_CONSTANT / 17 ** (1.0 / 0.3),  # temperature
            (3.0 * 6.46e-34) ** (-0.4),  # h_density
            13582529.79905836,  # Expected output
        ),
        (
            np.array(2 * [1], dtype=int),
            np.array(2 * [1.0]),
            np.array(2 * [0.0]),
            np.array(2 * [np.pi / 8 / BOLTZMANN_CONSTANT / 17 ** (1.0 / 0.3)]),
            np.array(2 * [(3.0 * 6.46e-34) ** (-0.4)]),
            np.array(2 * [13582529.79905836]),
        ),
    ],
)
def test_calc_gamma_van_der_waals_sample_values(
    calc_gamma_van_der_waals_sample_values_input_ion_number,
    calc_gamma_van_der_waals_sample_values_input_n_eff_upper,
    calc_gamma_van_der_waals_sample_values_input_n_eff_lower,
    calc_gamma_van_der_waals_sample_values_input_temperature,
    calc_gamma_van_der_waals_sample_values_input_h_density,
    calc_gamma_van_der_waals_sample_values_expected_result,
):
    assert np.allclose(
        calc_gamma_van_der_waals(
            calc_gamma_van_der_waals_sample_values_input_ion_number,
            calc_gamma_van_der_waals_sample_values_input_n_eff_upper,
            calc_gamma_van_der_waals_sample_values_input_n_eff_lower,
            calc_gamma_van_der_waals_sample_values_input_temperature,
            calc_gamma_van_der_waals_sample_values_input_h_density,
        ),
        calc_gamma_van_der_waals_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_gamma_van_der_waals_sample_values_input_ion_number, calc_gamma_van_der_waals_sample_values_input_n_eff_upper, calc_gamma_van_der_waals_sample_values_input_n_eff_lower, calc_gamma_van_der_waals_sample_values_input_temperature,  calc_gamma_van_der_waals_sample_values_input_h_density,calc_gamma_van_der_waals_sample_values_expected_result",
    [
        (
            np.array(2 * [1], dtype=int),
            np.array(2 * [1.0]),
            np.array(2 * [0.0]),
            np.array(2 * [np.pi / 8 / BOLTZMANN_CONSTANT / 17 ** (1.0 / 0.3)]),
            np.array(2 * [(3.0 * 6.46e-34) ** (-0.4)]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_gamma_van_der_waals_cuda_unwrapped_sample_values(
    calc_gamma_van_der_waals_sample_values_input_ion_number,
    calc_gamma_van_der_waals_sample_values_input_n_eff_upper,
    calc_gamma_van_der_waals_sample_values_input_n_eff_lower,
    calc_gamma_van_der_waals_sample_values_input_temperature,
    calc_gamma_van_der_waals_sample_values_input_h_density,
    calc_gamma_van_der_waals_sample_values_expected_result,
):
    arg_list = (
        calc_gamma_van_der_waals_sample_values_input_ion_number,
        calc_gamma_van_der_waals_sample_values_input_n_eff_upper,
        calc_gamma_van_der_waals_sample_values_input_n_eff_lower,
        calc_gamma_van_der_waals_sample_values_input_temperature,
        calc_gamma_van_der_waals_sample_values_input_h_density,
    )

    arg_list = tuple(map(cp.array, arg_list))
    result_values = cp.empty_like(arg_list[0])

    nthreads = 256
    length = len(calc_gamma_van_der_waals_sample_values_expected_result)
    nblocks = 1 + (length // nthreads)

    _calc_gamma_van_der_waals_cuda[nblocks, nthreads](result_values, *arg_list)

    assert np.allclose(
        cp.asnumpy(result_values),
        calc_gamma_van_der_waals_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_gamma_van_der_waals_sample_values_input_ion_number, calc_gamma_van_der_waals_sample_values_input_n_eff_upper, calc_gamma_van_der_waals_sample_values_input_n_eff_lower, calc_gamma_van_der_waals_sample_values_input_temperature,  calc_gamma_van_der_waals_sample_values_input_h_density,calc_gamma_van_der_waals_sample_values_expected_result",
    [
        (
            np.array(2 * [1], dtype=int),
            np.array(2 * [1.0]),
            np.array(2 * [0.0]),
            np.array(2 * [np.pi / 8 / BOLTZMANN_CONSTANT / 17 ** (1.0 / 0.3)]),
            np.array(2 * [(3.0 * 6.46e-34) ** (-0.4)]),
            np.array(2 * [1.0]),
        ),
    ],
)
def test_calc_gamma_van_der_waals_cuda_wrapped_sample_cuda_values(
    calc_gamma_van_der_waals_sample_values_input_ion_number,
    calc_gamma_van_der_waals_sample_values_input_n_eff_upper,
    calc_gamma_van_der_waals_sample_values_input_n_eff_lower,
    calc_gamma_van_der_waals_sample_values_input_temperature,
    calc_gamma_van_der_waals_sample_values_input_h_density,
    calc_gamma_van_der_waals_sample_values_expected_result,
):
    arg_list = (
        calc_gamma_van_der_waals_sample_values_input_ion_number,
        calc_gamma_van_der_waals_sample_values_input_n_eff_upper,
        calc_gamma_van_der_waals_sample_values_input_n_eff_lower,
        calc_gamma_van_der_waals_sample_values_input_temperature,
        calc_gamma_van_der_waals_sample_values_input_h_density,
    )
    assert np.allclose(
        calc_gamma_van_der_waals_cuda(*map(cp.asarray, arg_list)),
        calc_gamma_van_der_waals_sample_values_expected_result,
    )


# Test this with actual regression data
def test_rotational_broadening(example_stardis_output):
    actual_wavelengths, actual_fluxes_no_broadening = rotation_broadening(
        20 * u.km / u.s,
        example_stardis_output.lambdas,
        example_stardis_output.spectrum_lambda,
        v_rot=0 * u.km / u.s,
    )

    expected_broadening_fluxes = [
        21851984.04113946,
        21851937.30115837,
        21851843.93664505,
        21851704.17866379,
        21851518.37423182,
        21851286.98683553,
    ]
    actual_wavelengths, actual_fluxes = rotation_broadening(
        20 * u.km / u.s,
        example_stardis_output.lambdas,
        example_stardis_output.spectrum_lambda,
        v_rot=500 * u.km / u.s,
    )
    assert np.allclose(actual_wavelengths, example_stardis_output.lambdas)
    assert np.allclose(
        actual_fluxes_no_broadening, example_stardis_output.spectrum_lambda
    )
    assert np.allclose(actual_fluxes[:6].value, expected_broadening_fluxes)
