import pytest
import numpy as np
from math import sqrt
from numba import cuda
from stardis.opacities.voigt import (
    faddeeva,
    _faddeeva_cuda,
    faddeeva_cuda,
    voigt_profile,
)

# Test cases must also take into account use of a GPU to run. If there is no GPU then the test cases will fail.
GPUs_available = cuda.is_available()


@pytest.mark.parametrize(
    "faddeeva_sample_values_input, faddeeva_sample_values_expected_result",
    [
        (0, 1 + 0j),
        (0.0, 1.0 + 0.0j),
        (np.array([0.0]), np.array([1.0 + 0.0j])),
        (np.array([0, 0]), np.array([1 + 0j, 1 + 0j])),
    ],
)
def test_faddeeva_sample_values(
    faddeeva_sample_values_input, faddeeva_sample_values_expected_result
):
    assert np.allclose(
        faddeeva(faddeeva_sample_values_input),
        faddeeva_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "faddeeva_cuda_unwrapped_sample_values_input, faddeeva_cuda_unwrapped_sample_values_expected_result",
    [
        # (0, 1 + 0j),
        # (0.0, 1.0 + 0.0j),
        (np.array([0.0], dtype=complex), np.array([1.0 + 0.0j])),
        (np.array([0, 0], dtype=complex), np.array([1 + 0j, 1 + 0j])),
    ],
)
def test_faddeeva_cuda_unwrapped_sample_values(
    faddeeva_cuda_unwrapped_sample_values_input,
    faddeeva_cuda_unwrapped_sample_values_expected_result,
):
    test_values = cuda.to_device(faddeeva_cuda_unwrapped_sample_values_input)
    result_values = cuda.device_array_like(test_values)

    length = len(faddeeva_cuda_unwrapped_sample_values_input)

    _faddeeva_cuda.forall(length)(result_values, test_values)

    assert np.allclose(
        result_values.copy_to_host(),
        faddeeva_cuda_unwrapped_sample_values_expected_result,
    )


@pytest.mark.skipif(
    not GPUs_available, reason="No GPU is available to test CUDA function"
)
@pytest.mark.parametrize(
    "faddeeva_cuda_wrapped_sample_values_input, faddeeva_cuda_wrapped_sample_values_expected_result",
    [
        # (0, 1 + 0j),
        # (0.0, 1.0 + 0.0j),
        (np.array([0.0]), np.array([1.0 + 0.0j])),
        (np.array([0, 0]), np.array([1 + 0j, 1 + 0j])),
    ],
)
def test_faddeeva_cuda_wrapped_sample_values(
    faddeeva_cuda_wrapped_sample_values_input,
    faddeeva_cuda_wrapped_sample_values_expected_result,
):
    assert np.allclose(
        faddeeva_cuda(faddeeva_cuda_wrapped_sample_values_input),
        faddeeva_cuda_wrapped_sample_values_expected_result,
    )


test_voigt_profile_division_by_zero_test_values = [
    -100,
    -5,
    -1,
    0,
    0.0,
    1j,
    1.2,
    3,
    100,
]


@pytest.mark.parametrize(
    "voigt_profile_division_by_zero_input_delta_nu",
    test_voigt_profile_division_by_zero_test_values,
)
@pytest.mark.parametrize(
    "voigt_profile_division_by_zero_input_gamma",
    test_voigt_profile_division_by_zero_test_values,
)
def test_voigt_profile_division_by_zero(
    voigt_profile_division_by_zero_input_delta_nu,
    voigt_profile_division_by_zero_input_gamma,
):
    with pytest.raises(ZeroDivisionError):
        _ = voigt_profile(
            voigt_profile_division_by_zero_input_delta_nu,
            0,
            voigt_profile_division_by_zero_input_gamma,
        )


@pytest.mark.parametrize(
    "voigt_profile_sample_values_input_delta_nu, voigt_profile_sample_values_input_doppler_width, voigt_profile_sample_values_input_gamma, voigt_profile_sample_values_expected_result",
    [
        (0, 1, 0, 1 / sqrt(np.pi)),
        (0, 2, 0, 1 / (sqrt(np.pi) * 2)),
    ],
)
def test_voigt_profile_sample_values_sample_values(
    voigt_profile_sample_values_input_delta_nu,
    voigt_profile_sample_values_input_doppler_width,
    voigt_profile_sample_values_input_gamma,
    voigt_profile_sample_values_expected_result,
):
    assert np.allclose(
        voigt_profile(
            voigt_profile_sample_values_input_delta_nu,
            voigt_profile_sample_values_input_doppler_width,
            voigt_profile_sample_values_input_gamma,
        ),
        voigt_profile_sample_values_expected_result,
    )
