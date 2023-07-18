import pytest
from numpy import allclose, pi as PI
from math import sqrt


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
    from stardis.opacities.voigt import faddeeva

    assert allclose(
        faddeeva(faddeeva_sample_values_input),
        faddeeva_sample_values_expected_result,
    )


@pytest.mark.parametrize("voigt_profile_division_by_zero_input_delta_nu", range(11))
@pytest.mark.parametrize("voigt_profile_division_by_zero_input_gamma", range(11))
def test_voigt_profile_division_by_zero(
    voigt_profile_division_by_zero_input_delta_nu,
    voigt_profile_division_by_zero_input_gamma,
):
    from stardis.opacities.voigt import voigt_profile

    with pytest.raises(ZeroDivisionError):
        _ = voigt_profile(
            voigt_profile_division_by_zero_input_delta_nu,
            0,
            voigt_profile_division_by_zero_input_gamma,
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
    from stardis.opacities.voigt import voigt_profile

    assert allclose(
        voigt_profile(
            voigt_profile_input_delta_nu,
            voigt_profile_input_doppler_width,
            voigt_profile_input_gamma,
        ),
        voigt_profile_expected_result,
    )
