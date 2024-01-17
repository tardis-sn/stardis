import pytest

import numpy as np

# from astropy import units as u

from importlib_resources import files

from stardis.io.model.marcs import read_marcs_model

@pytest.fixture
def create_model():
    fname = "marcs_test.mod.gz"
    fpath = files("stardis.io.model.tests.data")
    test_data_filepath= fpath.joinpath(fname)
    model = read_marcs_model(test_data_filepath)
    return  model


def test_read_marcs_model_scaled_log_number_fraction(create_model):
    """
    Test reading a MARCS model file
    """

    assert np.allclose(
        create_model.data.scaled_log_number_fraction_1,
        12.0,
    )


def test_read_marcs_model_metadata_surface_grav(create_model):

    assert np.allclose(
        create_model.metadata["surface_grav"].value,
        10000,
    )


def test_read_marcs_model_metadata_x(create_model):
    
    assert np.allclose(
        create_model.metadata["x"],
        0.73826,
    )


def test_read_marcs_model_data_depth(create_model):
    
    assert np.allclose(
        create_model.data.depth.iloc[-1],
        44610000.0,
    )


def test_read_marcs_model_data_lgtaur(create_model):
    
    assert np.allclose(
        create_model.data.lgtaur.iloc[0],
        -5.0,
    )
