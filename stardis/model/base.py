import pandas as pd
import numpy as np


class StellarModel:
    """
    Class containing information about the stellar model.

    Parameters
    ----------
    fv_geometry : pandas.core.frame.DataFrame
    abundances : pandas.core.frame.DataFrame
    boundary_temps : numpy.ndarray
    geometry : stardis.model.geometry
    composition : stardis.model.composition.base.Composition

    Attributes
    ----------
    fv_geometry : pandas.core.frame.DataFrame
        Finite volume model DataFrame.
    abundances : pandas.core.frame.DataFrame
        Abundance DataFrame with all included elements and mass abundances.
    temperatures : numpy.ndarray
        Temperatures in K of all depth points. Note that array is transposed.
    geometry : stardis.model.geometry
        Geometry of the model.
    composition : stardis.model.composition.base.Composition
        Composition of the model. Includes density and atomic mass fractions.
    """

    def __init__(self, fv_geometry, abundances, temperatures, geometry, composition):
        self.fv_geometry = fv_geometry
        self.abundances = abundances
        self.temperatures = temperatures
        self.geometry = geometry
        self.composition = composition


def read_marcs_to_fv(fpath, atom_data, final_atomic_number=30):
    """
    Reads MARCS model and produces a finite volume model.

    Parameters
    ----------
    fpath : str
        The filepath to the MARCS model.
    atom_data : tardis.io.atom_data.base.AtomData
    final_atomic_number : int, optional
        Atomic number for the final element included in the model. Default
        is 30.

    Returns
    -------
    stardis.model.base.StellarModel
    """

    # This is a hacky workaround to avoid circular imports. Should be fixed when we remove the read_marcs_to_fv function.
    from stardis.io.model.marcs import read_marcs_model

    marcs_raw_model = read_marcs_model(fpath, gzipped=False)
    geometry = marcs_raw_model.to_geometry()
    composition = marcs_raw_model.to_composition(
        atom_data=atom_data, final_atomic_number=final_atomic_number
    )

    marcs_model1 = pd.read_csv(
        fpath, skiprows=24, nrows=56, delim_whitespace=True, index_col="k"
    )
    marcs_model2 = pd.read_csv(
        fpath, skiprows=81, nrows=56, delim_whitespace=True, index_col="k"
    )
    del marcs_model2["lgTauR"]
    marcs_model = marcs_model1.join(marcs_model2)
    marcs_model.columns = [item.lower() for item in marcs_model.columns]
    marcs_model["lgtaur"] = 10 ** marcs_model["lgtaur"]
    marcs_model["lgtau5"] = 10 ** marcs_model["lgtau5"]

    marcs_model = marcs_model.rename(columns={"lgtaur": "tau_ref", "lgtau5": "tau_500"})
    with open(fpath) as fh:
        marcs_lines = fh.readlines()
    marcs_abundance_scale_str = " ".join([item.strip() for item in marcs_lines[12:22]])
    marcs_abundances = pd.DataFrame(
        data=map(np.float64, marcs_abundance_scale_str.split()),
        columns=["abundance_scale"],
    )
    marcs_abundances.replace({-99: np.nan}, inplace=True)
    marcs_abundances = marcs_abundances.set_index(
        np.arange(1, len(marcs_abundances) + 1)
    )
    marcs_abundances.index.name = "atomic_number"
    marcs_abundances["mass_density"] = (
        10 ** marcs_abundances["abundance_scale"]
    ) * atom_data.atom_data.mass
    marcs_abundances["mass_abundance"] = (
        marcs_abundances["mass_density"] / marcs_abundances["mass_density"].sum()
    )
    marcs_model = marcs_model[::-1]

    boundary_temps = marcs_model.t.values[None].T

    marcs_model_fv = pd.DataFrame(
        data=0.5 * (marcs_model.iloc[:-1].values + marcs_model.iloc[1:].values),
        columns=marcs_model.columns,
    )
    marcs_model_fv["r_inner"] = marcs_model["depth"].iloc[:-1].values
    marcs_model_fv["r_outer"] = marcs_model["depth"].iloc[1:].values
    marcs_model_fv["cell_length"] = -(
        marcs_model_fv["r_outer"].values - marcs_model_fv["r_inner"].values
    )
    marcs_model_fv["t_inner"] = marcs_model.t[:-1].values
    marcs_model_fv["t_outer"] = marcs_model.t[1:].values

    marcs_abundances_all = pd.DataFrame(
        columns=marcs_model_fv.index.values, index=marcs_abundances.index
    )
    for i in range(len(marcs_abundances_all.columns)):
        marcs_abundances_all[i] = marcs_abundances["mass_abundance"]
    marcs_abundances_all = marcs_abundances_all[:final_atomic_number]

    return StellarModel(
        marcs_model_fv, marcs_abundances_all, boundary_temps, geometry, composition
    )
