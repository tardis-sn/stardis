import pytest
import numpy as np
from astropy import constants as const
from numba import cuda

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
)

GPUs_available = cuda.is_available()

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
    "calc_doppler_width_sample_values_input_nu_line,calc_doppler_width_sample_values_input_temperature,calc_doppler_width_sample_values_input_atomic_mass, calc_doppler_width_sample_values_expected_result",
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
        ),
        calc_doppler_width_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_doppler_width_cuda_unwrapped_sample_values_input_nu_line,calc_doppler_width_cuda_unwrapped_sample_values_input_temperature,calc_doppler_width_cuda_unwrapped_sample_values_input_atomic_mass,calc_doppler_width_cuda_unwrapped_sample_values_expected_result",
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
    "calc_n_effective_sample_values_input_ion_number,calc_n_effective_sample_values_input_ionization_energy,calc_n_effective_sample_values_input_level_energy, calc_n_effective_sample_values_expected_result",
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
    "calc_n_effective_cuda_unwrapped_sample_values_input_ion_number,calc_n_effective_cuda_unwrapped_sample_values_input_ionization_energy,calc_n_effective_cuda_unwrapped_sample_values_input_level_energy,calc_n_effective_cuda_unwrapped_sample_values_expected_result",
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
    "calc_gamma_linear_stark_sample_values_input_n_eff_upper,calc_gamma_linear_stark_sample_values_input_n_eff_lower,calc_gamma_linear_stark_sample_values_input_electron_density, calc_gamma_linear_stark_sample_values_expected_result",
    [
        (
            1,
            0,
            (0.51 * 0.642) ** (-3 / 2),
            1.0,
        ),
        (
            np.array(2 * [1]),
            np.array(2 * [0]),
            np.array(2 * [(0.51 * 0.642) ** (-3 / 2)]),
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
    "calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_upper,calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_n_eff_lower,calc_gamma_linear_stark_cuda_unwrapped_sample_values_input_electron_density,calc_gamma_linear_stark_cuda_unwrapped_sample_values_expected_result",
    [
        (
            np.array(2 * [1]),
            np.array(2 * [0]),
            np.array(2 * [(0.51 * 0.642) ** (-3 / 2)]),
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

    assert np.allclose(
        cp.asnumpy(result_values),
        calc_gamma_linear_stark_cuda_unwrapped_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_upper,calc_gamma_linear_stark_cuda_wrapped_sample_values_input_n_eff_lower,calc_gamma_linear_stark_cuda_wrapped_sample_values_input_electron_density,calc_gamma_linear_stark_cuda_wrapped_sample_values_expected_result",
    [
        (
            np.array(2 * [1]),
            np.array(2 * [0]),
            np.array(2 * [(0.51 * 0.642) ** (-3 / 2)]),
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
