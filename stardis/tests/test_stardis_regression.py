import numpy as np
import pandas as pd

plasma_properties = [
    'alpha_line',
    'beta_rad',
    'electron_densities',
    'g',
    'g_electron',
    'general_level_boltzmann_factor',
    'ion_number_density',
    'ionization_data',
    'level_boltzmann_factor',
    'level_number_density',
    'levels',
    'lines',
    'lines_lower_level_index',
    'lines_upper_level_index',
    'metastability',
    'nlte_excitation_species',
    'nlte_ionization_species',
    'nu',
    'number_density',
    'partition_function',
    'phi',
    'selected_atoms',
    'stimulated_emission_factor',
    't_electrons',
    't_rad',
    'wavelength_cm'
]

plasma_properties_complex = [

]

def test_stardis_stellar_model(example_stardis_output, regression_data):
    actual = example_stardis_output.stellar_model
    expected = regression_data.sync_hdf_store(actual)

    np.testing.assert_allclose(
        actual.temperatures.value,
        expected['/stellar_model/temperatures']
    )

def test_stardis_stellar_model_geometry(example_stardis_output, regression_data):
    # test Radial1DGeometry
    actual = example_stardis_output.stellar_model.geometry.r.value
    expected = regression_data.sync_ndarray(actual)
    np.testing.assert_allclose(actual, expected)

def test_stardis_plasma(example_stardis_output, regression_data):
    # test BasePlasma
    expected = regression_data.sync_hdf_store(example_stardis_output.stellar_plasma)
    actual_plasma = example_stardis_output.stellar_plasma
    for item in plasma_properties:
        actual_item = getattr(actual_plasma, item)
        expected_item = expected[f'plasma/{item}']
        if isinstance(actual_item, list):
            actual_item = np.ndarray(actual_item)
        
        if any(isinstance(actual_item, object_type) for object_type in [pd.MultiIndex, pd.Index]):
            expected_item = expected[f'plasma/{item}'].values.flatten()
            np.testing.assert_array_equal(actual_item.values, expected_item)

        elif isinstance(actual_item, np.ndarray):
            np.testing.assert_allclose(
                actual=actual_item,
                desired=expected_item,
                err_msg=f"Error encountered which comparing regression data for property- {item}"
            )
        elif isinstance(actual_item, pd.DataFrame):
            pd.testing.assert_frame_equal(actual_item, expected_item)
        elif isinstance(actual_item, pd.Series):
            pd.testing.assert_series_equal(actual_item, expected_item)
        else:
            plasma_properties_complex.append(item)
    if plasma_properties_complex: 
        raise AssertionError(plasma_properties_complex)

def test_stardis_radiation_field(example_stardis_output, regression_data):
    expected = regression_data.sync_hdf_store(example_stardis_output.stellar_radiation_field)
    actual = example_stardis_output.stellar_radiation_field

    np.testing.assert_allclose(
        expected["/radiation_field/frequencies"].values, 
        actual.frequencies.value
    )

    np.testing.assert_allclose(
        expected["/radiation_field/F_nu"].values, 
        actual.F_nu
    )
