import pytest
from numpy import allclose, pi as PI
from math import sqrt

from stardis.radiation_field.opacities.opacities_solvers.voigt import (
    faddeeva,
    voigt_profile,
)


@pytest.mark.parametrize(
    "faddeeva_sample_values_input, faddeeva_sample_values_expected_result",
    [
        (0, 1 + 0j),
        (0.0, 1.0 + 0.0j),
    ],
)
def test_faddeeva_sample_values(
    faddeeva_sample_values_input, faddeeva_sample_values_expected_result
):
    assert allclose(
        faddeeva(faddeeva_sample_values_input),
        faddeeva_sample_values_expected_result,
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
        (0, 1, 0, 1 / sqrt(PI)),
        (0, 2, 0, 1 / (sqrt(PI) * 2)),
    ],
)
def test_voigt_profile_sample_values_sample_values(
    voigt_profile_sample_values_input_delta_nu,
    voigt_profile_sample_values_input_doppler_width,
    voigt_profile_sample_values_input_gamma,
    voigt_profile_sample_values_expected_result,
):
    assert allclose(
        voigt_profile(
            voigt_profile_sample_values_input_delta_nu,
            voigt_profile_sample_values_input_doppler_width,
            voigt_profile_sample_values_input_gamma,
        ),
        voigt_profile_sample_values_expected_result,
    )
