import numpy as np

from astropy import units as u

from pathlib import Path

from stardis.io.model.marcs import read_marcs_model

MODEL_DATA_PATH = Path(__file__).parent / "data"


def test_read_marcs_model():
    """
    Test reading a MARCS model file
    """
    fname = MODEL_DATA_PATH / "marcs_test.mod.gz"
    model = read_marcs_model(fname)

    np.testing.assert_almost_equal(model.metadata["surface_grav"].value, 10000)
    np.testing.assert_almost_equal(model.metadata["x"], 0.73826)
    np.testing.assert_almost_equal(model.data.depth.iloc[-1], 44610000.0)
    np.testing.assert_almost_equal(model.data.lgtaur.iloc[0], -5.0)

    assert (model.data.scaled_log_number_fraction_1 == 12.0).all()
