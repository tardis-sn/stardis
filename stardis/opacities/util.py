import pandas as pd

from numba.core import types
from numba.typed import Dict


def read_wbr_cross_section(wbr_fpath):
    """
    Reads H minus cross sections by wavelength from Wishart (1979) and
    Broad and Reinhardt (1976).

    Parameters
    ----------
    wbr_fpath : str
        Filepath to read H minus cross sections.

    Returns
    -------
    wbr_cross_section : pandas.core.frame.DataFrame
        H minus cross sections by wavelength.
    """

    wbr_cross_section = pd.read_csv(
        wbr_fpath,
        names=["wavelength", "cross_section"],
        comment="#",
    )
    wbr_cross_section.wavelength *= 10  ## nm to AA
    wbr_cross_section.cross_section *= 1e-18  ## to cm^2

    return wbr_cross_section


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

