import pytest
from pathlib import Path

from tardis.io.atom_data.base import AtomData
from tardis.io.atom_data.util import download_atom_data


@pytest.fixture(scope="session")
def example_atomic_data():
    download_atom_data("kurucz_cd23_chianti_H_He")
    return AtomData.from_hdf("kurucz_cd23_chianti_H_He.h5")
