import numpy as np
import pandas as pd

from scipy.interpolate import interp1d, interp2d
from radioactivedecay.utils import elem_to_Z
from numba.core import types
from numba.typed import Dict

roman_numerals = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}


def sigma_file(tracing_lambdas, temperatures, fpath):

    df = pd.read_csv(fpath, header=None, comment="#")
    if np.isnan(df.loc[0, 0]):
        file_wavelengths = np.array(df.loc[1:, 0])
        file_temperatures = np.array(df.loc[0, 1:])
        file_cross_sections = np.array(df.loc[1:, 1:]).T
        fn = interp2d(file_wavelengths, file_temperatures, file_cross_sections)
        sigmas = fn(tracing_lambdas, temperatures)
    else:
        file_wavelengths = np.array(df.loc[:, 0])
        file_temperatures = np.array([0])
        file_cross_sections = np.array(df.loc[:, 1])
        fn = interp1d(
            file_wavelengths,
            file_cross_sections,
            bounds_error=False,
            fill_value=(file_cross_sections[0], file_cross_sections[-1]),
        )
        sigmas = np.zeros((len(temperatures), len(tracing_lambdas)))
        sigmas[:] = fn(tracing_lambdas)

    return sigmas


def map_items_to_indices(items):
    """
    Creates dictionary matching quantities in lines dataframe to their indices.

    Parameters
    ----------
    items : list
        List of column names.

    Returns
    -------
    items_dict : dict
    """
    items_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )

    for i, item in enumerate(items):
        items_dict[item] = i

    return items_dict


def get_number_density(stellar_plasma, spec):

    if spec == "Hminus_bf":
        return stellar_plasma.h_minus_density, None, None
    elif spec == "Hminus_ff":
        return (
            stellar_plasma.ion_number_density.loc[1, 0]
            * stellar_plasma.electron_densities,
            None,
            None,
        )
    elif spec == "Heminus_ff":
        return (
            stellar_plasma.ion_number_density.loc[2, 0]
            * stellar_plasma.electron_densities,
            None,
            None,
        )
    elif spec == "H2minus_ff":
        return stellar_plasma.h2_density * stellar_plasma.electron_densities, None, None
    elif spec == "H2plus_ff":
        return (
            stellar_plasma.ion_number_density.loc[1, 0]
            * stellar_plasma.ion_number_density.loc[1, 1],
            None,
            None,
        )
    elif spec == "H2plus_bf":
        return None, None, None  # Maybe implement?

    ion = spec[: len(spec) - 3]

    try:
        atomic_number = elem_to_Z(ion[0])
        ion_number = roman_numerals[ion[1:]] - 1
    except KeyError:
        atomic_number = elem_to_Z(ion[0:2])
        ion_number = roman_numerals[ion[2:]] - 1

    number_density = 1

    if spec[len(spec) - 2 :] == "ff":
        ion_number += 1
        number_density *= stellar_plasma.electron_densities

    number_density *= stellar_plasma.ion_number_density.loc[atomic_number, ion_number]

    return number_density, atomic_number, ion_number
