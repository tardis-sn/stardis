from stardis.radiation_field.opacities.opacities_solvers import calc_alpha_line_at_nu


def test_calc_alpha_line_at_nu_with_broadening(
    example_stellar_plasma,
    example_stellar_model,
    example_tracing_nus,
    example_config_broadening,
):
    alpha_line_at_nu, gammas, doppler_widths = calc_alpha_line_at_nu(
        example_stellar_plasma,
        example_stellar_model,
        example_tracing_nus,
        example_config_broadening.opacity.line,
    )
    assert len(gammas) == len(doppler_widths)
    assert len(alpha_line_at_nu) == example_stellar_model.no_of_depth_points
