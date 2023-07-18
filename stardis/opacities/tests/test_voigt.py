import pytest
from numpy import allclose, pi as PI
from math import sqrt
from stardis.opacities import voigt


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
        voigt.faddeeva(faddeeva_sample_values_input),
        faddeeva_sample_values_expected_result,
    )


@pytest.mark.parametrize(
    "voigt_profile_input_delta_nu, voigt_profile_input_doppler_width, voigt_profile_input_gamma",
    [
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
    ],
)
def test_voigt_profile_division_by_zero(
    voigt_profile_input_delta_nu,
    voigt_profile_input_doppler_width,
    voigt_profile_input_gamma,
):
    with pytest.raises(ZeroDivisionError):
        _ = voigt.voigt_profile(
            voigt_profile_input_delta_nu,
            voigt_profile_input_doppler_width,
            voigt_profile_input_gamma,
        )


@pytest.mark.parametrize(
    "voigt_profile_input_delta_nu, voigt_profile_input_doppler_width, voigt_profile_input_gamma, voigt_profile_expected_result",
    [
        (0, 1, 0, 1 / sqrt(PI)),
        (0, 2, 0, 1 / (sqrt(PI) * 2)),
    ],
)
def test_voigt_profile_sample_values(
    voigt_profile_input_delta_nu,
    voigt_profile_input_doppler_width,
    voigt_profile_input_gamma,
    voigt_profile_expected_result,
):
    assert allclose(
        voigt.voigt_profile(
            voigt_profile_input_delta_nu,
            voigt_profile_input_doppler_width,
            voigt_profile_input_gamma,
        ),
        voigt_profile_expected_result,
    )
