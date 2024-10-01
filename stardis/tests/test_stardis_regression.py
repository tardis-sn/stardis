import numpy as np
import pandas as pd

def test_stardis_output_model(example_stardis_output, regression_data):
    actual = example_stardis_output.stellar_model
    expected = regression_data.sync_hdf_store(actual)

    np.testing.assert_allclose(
        actual.temperatures.value,
        expected['/stellar_model/temperatures']
    )
    np.testing.assert_allclose(
        actual.geometry.r.value,
        expected['/stellar_model/geometry/r']
    )
    
    
