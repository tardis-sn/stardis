import numpy as np

from tardis.io.atom_data import AtomData
from tardis.io.configuration.config_validator import validate_yaml
from tardis.io.configuration.config_reader import Configuration

from astropy import units as u
from pathlib import Path

from stardis.plasma import create_stellar_plasma
from stardis.radiation_field.opacities.opacities_solvers import calc_alphas
from stardis.radiation_field.radiation_field_solvers import raytrace
from stardis.radiation_field import RadiationField
from stardis.io.model.marcs import read_marcs_model
from stardis.io.model.mesa import read_mesa_model
from stardis.radiation_field.source_functions.blackbody import blackbody_flux_at_nu
import logging


BASE_DIR = Path(__file__).parent
SCHEMA_PATH = BASE_DIR / "config_schema.yml"


###TODO: Make a function that parses the config and model files and outputs python objects to be passed into run stardis so they can be individually modified in python
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

    config_dict = validate_yaml(config_fname, schemapath=SCHEMA_PATH)
    config = Configuration(config_dict)

    adata = AtomData.from_hdf(config.atom_data)

    # model
    logging.info("Reading model")
    if config.model.type == "marcs":
        raw_marcs_model = read_marcs_model(
            Path(config.model.fname), gzipped=config.model.gzipped
        )
        stellar_model = raw_marcs_model.to_stellar_model(
            adata, final_atomic_number=config.model.final_atomic_number
        )

    elif config.model.type == "mesa":
        raw_mesa_model = read_mesa_model(Path(config.model.fname))
        if config.model.truncate_to_shell != -99:
            raw_mesa_model.truncate_model(config.model.truncate_to_shell)
        elif config.model.truncate_to_shell < 0:
            raise ValueError(
                f"{config.model.truncate_to_shell} shells were requested for mesa model truncation. -99 is default for no truncation."
            )

        stellar_model = raw_mesa_model.to_stellar_model(
            adata, final_atomic_number=config.model.final_atomic_number
        )

    else:
        raise ValueError("Model type not recognized. Must be either 'marcs' or 'mesa'")

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
        ),
        line_interaction_type="macroatom",
        nlte_species=[],
        continuum_interaction_species=[],
    )
    # plasma
    logging.info("Creating plasma")
    stellar_plasma = create_stellar_plasma(stellar_model, adata, config)

    stellar_radiation_field = RadiationField(
        tracing_nus, blackbody_flux_at_nu, stellar_model
    )
    logging.info("Calculating alphas")
    calc_alphas(
        stellar_plasma=stellar_plasma,
        stellar_model=stellar_model,
        stellar_radiation_field=stellar_radiation_field,
        opacity_config=config.opacity,
        parallel_config=config.parallel,
    )
    logging.info("Raytracing")
    raytrace(
        stellar_model,
        stellar_radiation_field,
        no_of_thetas=config.no_of_thetas,
        parallel_config=config.parallel,
    )

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
