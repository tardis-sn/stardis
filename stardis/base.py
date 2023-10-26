import os

import numpy as np

from tardis.io.atom_data import AtomData
from tardis.io.configuration.config_validator import validate_yaml
from tardis.io.configuration.config_reader import Configuration

from astropy import units as u

from stardis.plasma import create_stellar_plasma
from stardis.radiation_field.opacities.opacities_solvers import calc_alphas
from stardis.radiation_field.radiation_field_solvers import raytrace
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
    # This does not yet truncate vald linelists. TODO - also truncate vald
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
    stellar_plasma = create_stellar_plasma(stellar_model, adata, config)

    if True:  ###TODO change to checking source function from config
        from stardis.radiation_field.source_functions.blackbody import (
            blackbody_flux_at_nu,
        )

    stellar_radiation_field = RadiationField(
        tracing_nus, blackbody_flux_at_nu, stellar_model
    )

    calc_alphas(
        stellar_plasma=stellar_plasma,
        stellar_model=stellar_model,
        stellar_radiation_field=stellar_radiation_field,
        opacity_config=config.opacity,
    )

    raytrace(stellar_model, stellar_radiation_field, no_of_thetas=config.no_of_thetas)

    return STARDISOutput(
        config.result_options, stellar_model, stellar_plasma, stellar_radiation_field
    )


class STARDISOutput:
    """
    Class containing all the key outputs of a STARDIS simulation.

    Parameters
    ----------
    result_options : dict
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_model : stardis.model.base.StellarModel
    stellar_radiation_field : stardis.radiation_field.radiation_field.RadiationField

    Attributes
    ----------
    stellar_model : stardis.model.base.StellarModel
    stellar_plasma : tardis.plasma.base.BasePlasma
    stellar_radiation_field : stardis.radiation_field.radiation_field.RadiationField
        contains the following attributes:
            frequencies : astropy.units.Quantity
                Frequencies of the radiation field.
            source_function : stardis.radiation_field.source_function
            opacities : stardis.radiation_field.opacities
                Stardis opacities object. Contains the opacities contributed by the stellar atmosphere.
            F_nu : numpy.ndarray
                Radiation field fluxes at each frequency at each depth point.
    nus : astropy.units.Quantity
        Numpy array of frequencies used for spectrum with units of Hz.
    lambdas : astropy.units.Quantity
        Numpy array of wavelengths used for spectrum with units of Angstroms.
    spectrum_nu : astropy.units.Quantity
        Output flux with respect to frequency at the outermost depth point for each
        frequency. Units of erg/s/cm^2/Hz.
    spectrum_lambda : astropy.units.Quantity
        Output flux with respect to wavelength at the outermost depth point for each
        wavelength. Units of erg/s/cm^2/Angstrom.
    """

    def __init__(
        self,
        result_options,
        stellar_model,
        stellar_plasma,
        stellar_radiation_field,
    ):
        if result_options.return_model:
            self.stellar_model = stellar_model
        if result_options.return_plasma:
            self.stellar_plasma = stellar_plasma
        if result_options.return_radiation_field:
            self.stellar_radiation_field = stellar_radiation_field

        self.nus = stellar_radiation_field.frequencies
        self.lambdas = self.nus.to(u.AA, u.spectral())

        F_nu = stellar_radiation_field.F_nu * u.erg / u.s / u.cm**2 / u.Hz
        F_lambda = (F_nu * self.nus / self.lambdas).to(u.erg / u.s / u.cm**2 / u.AA)

        self.spectrum_nu = F_nu[-1]
        self.spectrum_lambda = F_lambda[-1]
