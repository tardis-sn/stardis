import pytest

import numpy as np

# from astropy import units as u

from importlib_resources import files

from stardis.io.model.marcs import read_marcs_model
from pathlib import Path
from stardis.io.model.mesa import read_mesa_model

MARCS_TEST_FPATH = Path(__file__).parent / "data" / "marcs_test.mod.gz"
MESA_TEST_FPATH = Path(__file__).parent / "data" / "end_core_h_burn.mod"


@pytest.fixture
def marcs_model_test_data_file_path():
    fname = "marcs_test.mod.gz"
    fpath = files("stardis.io.model.tests.data")
    return fpath.joinpath(fname)


@pytest.fixture(scope="session")
def marcs_model_test_data():
    return read_marcs_model(marcs_model_test_data_file_path)


@pytest.fixture(scope="session")
def mesa_model_test_data_file_path():
    fname = "end_core_h_burn.mod"
    fpath = files("stardis.io.model.tests.data")
    return fpath.joinpath(fname)


@pytest.fixture(scope="session")
def mesa_model():
    return read_mesa_model(MESA_TEST_FPATH)


@pytest.fixture(scope="session")
def mesa_stellar_model(mesa_model, example_kurucz_atomic_data):
    return mesa_model.to_stellar_model(atom_data=example_kurucz_atomic_data)


def test_mesa_model_ingestion(mesa_model):
    assert mesa_model.metadata["Number of Shells"] == 832
    assert mesa_model.metadata["Model Number"] == 295
    assert len(mesa_model.data) == 832
    assert mesa_model.data.lnT.iloc[0] == 8.660037236737706


def test_mesa_stellar_model(mesa_stellar_model):
    assert np.all(mesa_stellar_model.geometry.r.diff() > 0)


def test_read_marcs_model_scaled_log_number_fraction(marcs_model_test_data_file_path):
    """
    Test reading a MARCS model file
    """
    model = read_marcs_model(marcs_model_test_data_file_path)

    assert np.allclose(
        model.data.scaled_log_number_fraction_1,
        12.0,
    )


def test_read_marcs_model_metadata_surface_grav(marcs_model_test_data_file_path):
    model = read_marcs_model(marcs_model_test_data_file_path)

    assert np.allclose(
        model.metadata["surface_grav"].value,
        10000,
    )


def test_read_marcs_model_metadata_x(marcs_model_test_data_file_path):
    model = read_marcs_model(marcs_model_test_data_file_path)

    assert np.allclose(
        model.metadata["x"],
        0.73826,
    )


def test_read_marcs_model_data_depth(marcs_model_test_data_file_path):
    model = read_marcs_model(marcs_model_test_data_file_path)

    assert np.allclose(
        model.data.depth.iloc[-1],
        44610000.0,
    )


def test_read_marcs_model_data_lgtaur(marcs_model_test_data_file_path):
    model = read_marcs_model(marcs_model_test_data_file_path)

    assert np.allclose(
        model.data.lgtaur.iloc[0],
        -5.0,
    )
