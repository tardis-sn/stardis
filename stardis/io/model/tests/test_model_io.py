import pytest
import numpy as np
from pathlib import Path

from stardis.io.model.marcs import read_marcs_model
from stardis.io.model.mesa import read_mesa_model
from stardis.io.model.util import rescale_nuclide_mass_fractions

MARCS_TEST_FPATH = Path(__file__).parent / "data" / "marcs_test.mod.gz"
MESA_TEST_FPATH = Path(__file__).parent / "data" / "end_core_h_burn.mod"


@pytest.fixture(scope="session")
def marcs_model():
    return read_marcs_model(MARCS_TEST_FPATH)


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


def test_mesa_truncation(mesa_model):
    len_before_truncation = len(mesa_model.data)
    mesa_model.truncate_model(len_before_truncation - 1)
    assert len(mesa_model.data) == len_before_truncation - 1


def test_marcs_model(marcs_model):

    assert np.allclose(
        marcs_model.data.scaled_log_number_fraction_1,
        12.0,
    )

    assert np.allclose(
        marcs_model.metadata["surface_grav"].value,
        10000,
    )

    assert np.allclose(
        marcs_model.metadata["x"],
        0.73826,
    )

    assert np.allclose(
        marcs_model.data.depth.iloc[-1],
        44610000.0,
    )

    assert np.allclose(
        marcs_model.data.lgtaur.iloc[0],
        -5.0,
    )


def test_rescale_nuclide_mass_fraction(example_stellar_model):
    rescaled = rescale_nuclide_mass_fractions(
        example_stellar_model.composition.nuclide_mass_fraction, [4, 5], [1.1, 0.8]
    )
    assert np.allclose(
        rescaled.loc[5].values,
        example_stellar_model.composition.nuclide_mass_fraction.loc[5].values * 0.8,
    )

    assert np.allclose(
        rescaled.loc[5].values,
        example_stellar_model.composition.nuclide_mass_fraction.loc[4].values * 1.1,
    )
