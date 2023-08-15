import os

import numpy as np

from tardis.io.atom_data import AtomData
from tardis.io.config_validator import validate_yaml, validate_dict
from tardis.io.config_reader import Configuration

from astropy import units as u

from stardis.plasma import create_stellar_plasma
from stardis.opacities import calc_alphas
from stardis.transport import raytrace
from stardis.radiation_field import RadiationField


base_dir = os.path.abspath(os.path.dirname(__file__))
schema = os.path.join(base_dir, "config_schema.yml")


def run_stardis(config_fname, tracing_lambdas_or_nus):
    """
    Runs a STARDIS simulation.

    Parameters
    ----------
    config_fname : str
        Filepath to the STARDIS configuration. Must be a YAML file.
    tracing_lambdas_or_nus : astropy.units.Quantity
        Numpy array of the frequencies or wavelengths to calculate the
        spectrum for. Must have units attached to it, with dimensions
        of either length or inverse time.

    Returns
    -------
    stardis.base.STARDISOutput
        Contains all the key outputs of the STARDIS simulation.
    """

    tracing_nus = tracing_lambdas_or_nus.to(u.Hz, u.spectral())

    config_dict = validate_yaml(config_fname, schemapath=schema)
    config = Configuration(config_dict)

    adata = AtomData.from_hdf(config.atom_data)

    # model
    if config.model.type == "marcs":
        from stardis.io.model.marcs import read_marcs_model

        raw_marcs_model = read_marcs_model(
            config.model.fname, gzipped=config.model.gzipped
        )
        stellar_model = raw_marcs_model.to_stellar_model(
            adata, final_atomic_number=config.model.final_atomic_number
        )

    # Handle case of when there are fewer elements requested vs. elements in the atomic mass fraction table.
    adata.prepare_atom_data(
        np.arange(
            1,
            np.min(
                [
                    len(
                        stellar_model.composition.atomic_mass_fraction.columns.tolist()
                    ),
                    config.model.final_atomic_number,
                ]
            )
            + 1,
        )
    )
    # plasma
    stellar_plasma = create_stellar_plasma(stellar_model, adata)

    if True:  # change to checking source function from config
        from stardis.radiation_field.source_functions.base import (
            black_body_source_function,
        )

        radiation_field = RadiationField(tracing_nus, black_body_source_function)

    # Below becomes radiation field
    alphas, gammas, doppler_widths = calc_alphas(
        stellar_plasma=stellar_plasma,
        stellar_model=stellar_model,
        tracing_nus=tracing_nus,
        opacity_config=config.opacity,
    )

    F_nu = raytrace(
        stellar_model, alphas, tracing_nus, no_of_thetas=config.no_of_thetas
    )

    return STARDISOutput(
        stellar_plasma, stellar_model, alphas, gammas, doppler_widths, F_nu, tracing_nus
    )


class STARDISOutput:
    """
    Class containing all the key outputs of a STARDIS simulation.

    Parameters
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    alphas : numpy.ndarray
    gammas : numpy.ndarray
    doppler_widths : numpy.ndarray
    F_nu : numpy.ndarray
    nus: astropy.units.Quantity

    Attributes
    ----------
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    alphas : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Total opacity at
        each depth point for each frequency in tracing_nus.
    gammas : numpy.ndarray
        Array of shape (no_of_lines, no_of_depth_points). Collisional broadening
        parameter of each line at each depth point.
    doppler_widths : numpy.ndarray
        Array of shape (no_of_lines, no_of_depth_points). Doppler width of each
        line at each depth point.
    F_nu : astropy.units.Quantity
        Array of shape (no_of_depth points, no_of_frequencies). Output flux with
        respect to frequency at each depth point for each frequency. Units of erg/s/cm^2/Hz.
    F_lambda : astropy.units.Quantity
        Array of shape (no_of_depth_points, no_of_frequencies). Output flux with
        respect to wavelength at each depth point for each wavelength. Units of erg/s/cm^2/Angstrom.
    spectrum_nu : astropy.units.Quantity
        Output flux with respect to frequency at the outermost depth point for each
        frequency. Units of erg/s/cm^2/Hz.
    spectrum_lambda : astropy.units.Quantity
        Output flux with respect to wavelength at the outermost depth point for each
        wavelength. Units of erg/s/cm^2/Angstrom.
    nus : astropy.units.Quantity
        Numpy array of frequencies used for spectrum with units of Hz.
    lambdas : astropy.units.Quantity
        Numpy array of wavelengths used for spectrum with units of Angstroms.
    """

    def __init__(
        self, stellar_plasma, stellar_model, alphas, gammas, doppler_widths, F_nu, nus
    ):
        self.stellar_plasma = stellar_plasma
        self.stellar_model = stellar_model
        self.alphas = alphas
        self.gammas = gammas
        self.doppler_widths = doppler_widths

        self.nus = nus
        self.lambdas = nus.to(u.AA, u.spectral())

        self.F_nu = F_nu * u.erg / u.s / u.cm**2 / u.Hz
        self.F_lambda = (self.F_nu * nus / self.lambdas).to(
            u.erg / u.s / u.cm**2 / u.AA
        )

        self.spectrum_nu = self.F_nu[-1]
        self.spectrum_lambda = self.F_lambda[-1]
