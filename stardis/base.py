import numba

from stardis.io.base import parse_config_to_model
from stardis.plasma import create_stellar_plasma
from stardis.radiation_field.opacities.opacities_solvers import calc_alphas
from stardis.radiation_field.radiation_field_solvers import raytrace
from stardis.radiation_field import RadiationField
from stardis.radiation_field.source_functions.blackbody import blackbody_flux_at_nu
from astropy import units as u

import logging


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

    config, adata, stellar_model = parse_config_to_model(config_fname)
    set_num_threads(config.n_threads)

    stellar_plasma = create_stellar_plasma(stellar_model, adata, config)
    stellar_radiation_field = create_stellar_radiation_field(
        tracing_nus, stellar_model, stellar_plasma, config
    )

    return STARDISOutput(
        config.result_options, stellar_model, stellar_plasma, stellar_radiation_field
    )


def set_num_threads(n_threads):
    # Set multithreading as specified by the config
    if n_threads == 1:
        logging.info("Running in serial mode")
    elif n_threads == -99:
        logging.info("Running with max threads")
    elif n_threads > 1:
        logging.info(f"Running with {n_threads} threads")
        numba.set_num_threads(n_threads)
    else:
        raise ValueError(
            "n_threads must be a positive integer less than the number of available threads, or -99 to run with max threads."
        )


def create_stellar_radiation_field(tracing_nus, stellar_model, stellar_plasma, config):
    stellar_radiation_field = RadiationField(
        tracing_nus, blackbody_flux_at_nu, stellar_model
    )
    logging.info("Calculating alphas")
    calc_alphas(
        stellar_plasma=stellar_plasma,
        stellar_model=stellar_model,
        stellar_radiation_field=stellar_radiation_field,
        opacity_config=config.opacity,
        n_threads=config.n_threads,
    )
    logging.info("Raytracing")
    raytrace(
        stellar_model,
        stellar_radiation_field,
        no_of_thetas=config.no_of_thetas,
        n_threads=config.n_threads,
    )

    return stellar_radiation_field


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
