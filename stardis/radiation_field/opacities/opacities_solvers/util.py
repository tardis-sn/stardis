import numpy as np
import pandas as pd

from astropy import units as u, constants as const
from tardis.util.base import species_string_to_tuple

from scipy.interpolate import LinearNDInterpolator


def sigma_file(tracing_lambdas, temperatures, fpath, opacity_source=None):
    """
    Reads and interpolates a cross-section file.

    Parameters
    ----------
    tracing_lambdas : numpy.ndarray
        Wavelengths to compute the cross-section for.
    temperatures : numpy.ndarray
        Temperatures to compute the cross-section for.
    fpath : str
        Filepath to cross-section file.
    opacity_source: str
        Opacity source. Used to identify the table to be read appropriately.

    Returns
    -------
    sigmas : numpy.ndarray
        Array of shape (no_of_temperatures, no_of_wavelengths). Cross-section
        for each wavelength and temperature combination.
    """
    if (
        opacity_source == "H2plus_bf"
    ):  # This section specifically ingests the Stancil 1994 h2_plus_bf_S1994.dat table found in data.
        h2_plus_bf_table = pd.read_csv(
            fpath, delimiter=r"\s+", index_col=0, comment="#"
        )
        h2_plus_bf_table.replace({"-": "e-"}, regex=True, inplace=True)
        h2_plus_bf_table = h2_plus_bf_table.astype(float)
        file_wavelengths = (h2_plus_bf_table.index.values * u.nm).to(u.AA).value
        file_temperatures = h2_plus_bf_table.columns.values.astype(int)
        file_cross_sections = h2_plus_bf_table.to_numpy()

        file_waves_mesh, file_temps_mesh = np.meshgrid(
            file_wavelengths, file_temperatures, indexing="ij"
        )
        linear_interp_h2plus_bf = LinearNDInterpolator(
            np.vstack([file_waves_mesh.ravel(), file_temps_mesh.ravel()]).T,
            file_cross_sections.flatten(),
        )
        lambdas, temps = np.meshgrid(tracing_lambdas, temperatures)
        sigmas = (
            linear_interp_h2plus_bf(lambdas, temps) * 1e-18
        )  # Scaling from Stancil 1994 table

    elif (
        opacity_source == "Hminus_ff"
    ):  # This section specifically ingests the Bell and Berrington 1987 h_minus_ff_B1987.dat table found in data.
        h_minus_ff_table = pd.read_csv(fpath, delimiter=r"\s+", comment="#")
        h_minus_ff_table.columns = h_minus_ff_table.columns.str.strip(",")

        file_wavelengths = h_minus_ff_table[h_minus_ff_table.columns[0]].values
        file_thetas = h_minus_ff_table.columns[1:].astype(float)
        file_values = h_minus_ff_table.to_numpy()[:, 1:]

        file_waves_mesh, file_thetas_mesh = np.meshgrid(
            file_wavelengths, file_thetas, indexing="ij"
        )
        linear_interp_hminus_ff = LinearNDInterpolator(
            np.vstack([file_waves_mesh.ravel(), file_thetas_mesh.ravel()]).T,
            file_values.flatten(),
        )
        lambdas, thetas = np.meshgrid(tracing_lambdas, 5040 / temperatures)
        sigmas = (
            linear_interp_hminus_ff(lambdas, thetas)
            * 1e-26  # Scaling value from Bell 1987 stable
            * const.k_B.cgs.value
            * temperatures[:, np.newaxis]
        )

    elif (
        opacity_source == "Hminus_bf"
    ):  # This is currently used to read h_minus_bf_W1979.dat, Wishart 1979
        h_minus_bf_table = pd.read_csv(
            fpath, header=None, comment="#", names=["wavelength", "cross_section"]
        )
        sigmas = np.interp(
            tracing_lambdas,
            h_minus_bf_table.wavelength.values,
            h_minus_bf_table.cross_section.values,
        )

    else:
        raise ValueError(f"Unknown opacity_source: {opacity_source}")

    return sigmas


def get_number_density(stellar_plasma, opacity_source):
    """
    Computes number density, atomic number, and ion number for an opacity
    source provided as a string.

    Parameters
    ----------

    Returns
    -------
    number_density : numpy.ndarray or pandas.Series
    atomic_number : int
    ion_number : int
    """

    if opacity_source == "Hminus_bf":
        return stellar_plasma.h_minus_density, None, None
    elif opacity_source == "Hminus_ff":
        return (
            stellar_plasma.ion_number_density.loc[1, 0]
            * stellar_plasma.electron_densities,
            None,
            None,
        )
    elif opacity_source == "Heminus_ff":
        return (
            stellar_plasma.ion_number_density.loc[2, 0]
            * stellar_plasma.electron_densities,
            None,
            None,
        )
    elif opacity_source == "H2minus_ff":
        return stellar_plasma.h2_density * stellar_plasma.electron_densities, None, None
    elif opacity_source == "H2plus_ff":
        return (
            stellar_plasma.ion_number_density.loc[1, 0]
            * stellar_plasma.ion_number_density.loc[1, 1],
            None,
            None,
        )
    elif opacity_source == "H2plus_bf":
        return stellar_plasma.h2_plus_density, None, None

    ion = opacity_source[: len(opacity_source) - 3]

    atomic_number, ion_number = species_string_to_tuple(ion.replace("_", " "))

    number_density = 1

    if opacity_source[len(opacity_source) - 2 :] == "ff":
        ion_number += 1
        number_density *= stellar_plasma.electron_densities

    number_density *= stellar_plasma.ion_number_density.loc[atomic_number, ion_number]

    return number_density, atomic_number, ion_number
