import numpy as np
from numpy.testing import assert_allclose

def test_stardis_output_model(example_stardis_output, regression_data):
    expected = regression_data.sync_hdf_store(example_stardis_output.stellar_model)

