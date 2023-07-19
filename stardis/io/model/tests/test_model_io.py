import pytest

import numpy as np

# from astropy import units as u

from importlib_resources import files


@pytest.fixture
def marcs_model_test_data_file_path():
    fname = "marcs_test.mod.gz"
    fpath = files("stardis.io.model.tests.data")
    return fpath.joinpath(fname)


def test_read_marcs_model_scaled_log_number_fraction(marcs_model_test_data_file_path):
    """
    Test reading a MARCS model file
    """
    from stardis.io.model.marcs import read_marcs_model

    model = read_marcs_model(marcs_model_test_data_file_path)

    assert np.allclose(
        model.data.scaled_log_number_fraction_1,
        12.0,
    )


def test_read_marcs_model_metadata_surface_grav(marcs_model_test_data_file_path):
    from stardis.io.model.marcs import read_marcs_model

    model = read_marcs_model(marcs_model_test_data_file_path)

    assert np.allclose(
        model.metadata["surface_grav"].value,
        10000,
    )
    #
    # np.testing.assert_almost_equal(model.metadata["x"], 0.73826)
    # np.testing.assert_almost_equal(model.data.depth.iloc[-1], 44610000.0)
    # np.testing.assert_almost_equal(model.data.lgtaur.iloc[0], -5.0)
