import pytest
from numpy import allclose, pi as PI
from astropy import constants as const

from stardis.opacities.broadening import calc_doppler_width


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
        (SPEED_OF_LIGHT, 0.5, BOLTZMANN_CONSTANT, 1.0),
        # (0.0, 1.0 + 0.0j),
    ],
)
def test_calc_doppler_width_sample_values(
    calc_doppler_width_sample_values_input_nu_line,
    calc_doppler_width_sample_values_input_temperature,
    calc_doppler_width_sample_values_input_atomic_mass,
    calc_doppler_width_sample_values_expected_result,
):
    assert allclose(
        calc_doppler_width(
            calc_doppler_width_sample_values_input_nu_line,
            calc_doppler_width_sample_values_input_temperature,
            calc_doppler_width_sample_values_input_atomic_mass,
        ),
        calc_doppler_width_sample_values_expected_result,
    )
