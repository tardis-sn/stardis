import pytest

import numpy as np

# from astropy import units as u

from importlib_resources import files

from pathlib import Path

from stardis.io.model.marcs import read_marcs_model


@pytest.fixture
def marcs_test_model():
    fname = "marcs_test.mod.gz"
    fpath = Path("stardis.io.model.tests.data")
    test_data_filepath = fpath / fname

    model = read_marcs_model(test_data_filepath)
    return model


def test_read_marcs_model_scaled_log_number_fraction(marcs_test_model):
    """
    Test reading a MARCS model file
    """

    assert np.allclose(
        marcs_test_model.data.scaled_log_number_fraction_1,
        12.0,
    )


def test_read_marcs_model_metadata_surface_grav(marcs_test_model):
    assert np.allclose(
        marcs_test_model.metadata["surface_grav"].value,
        10000,
    )


def test_read_marcs_model_metadata_x(marcs_test_model):
    assert np.allclose(
        marcs_test_model.metadata["x"],
        0.73826,
    )


def test_read_marcs_model_data_depth(marcs_test_model):
    assert np.allclose(
        marcs_test_model.data.depth.iloc[-1],
        44610000.0,
    )


def test_read_marcs_model_data_lgtaur(marcs_test_model):
    assert np.allclose(
        marcs_test_model.data.lgtaur.iloc[0],
        -5.0,
    )
