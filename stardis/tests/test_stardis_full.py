import numpy as np
from numpy.testing import assert_allclose


def test_stardis_output(example_stardis_output, example_tracing_nus):
    assert len(example_stardis_output.spectrum_nu) == len(example_tracing_nus)


def test_stardis_broadening_output(
    example_stardis_output_broadening, example_tracing_nus
):
    assert len(example_stardis_output_broadening.spectrum_nu) == len(
        example_tracing_nus
    )


def test_stardis_parallel_output(
    example_stardis_output_parallel,
    example_tracing_nus,
    example_stardis_output_broadening,
):
    assert len(example_stardis_output_parallel.spectrum_nu) == len(example_tracing_nus)
    assert ~np.all(np.isnan(example_stardis_output_parallel.spectrum_nu))
    assert_allclose(
        example_stardis_output_parallel.spectrum_nu,
        example_stardis_output_broadening.spectrum_nu,
    )
