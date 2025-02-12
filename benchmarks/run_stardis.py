# Import necessary code
import os
from pathlib import Path
import numpy as np
from stardis.base import run_stardis
from astropy import units as u

from tardis.io.atom_data import AtomData
from tardis.io.configuration.config_validator import validate_yaml
from tardis.io.configuration.config_reader import Configuration

from stardis.io.model.marcs import read_marcs_model
from stardis.plasma import create_stellar_plasma
from stardis.radiation_field.opacities.opacities_solvers import (
    calc_alphas,
    calc_alpha_line_at_nu,
    calc_alpha_file,
    calc_alpha_rayleigh,
    calc_alpha_electron,
)
from stardis.radiation_field.radiation_field_solvers import raytrace
from stardis.radiation_field import RadiationField
from stardis.radiation_field.source_functions.blackbody import blackbody_flux_at_nu

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR.parent / "stardis" / "io" / "schemas" / "config_schema.yml"
CONFIG_PATH = BASE_DIR / "benchmark_config.yml"


class Sim10AA:
    """
    Class to benchmark Stardis functions.
    """

    timeout = 1800  # Worst case timeout of 30 mins

    def setup(self):

        tracing_lambdas = np.arange(6560, 6570, 0.01) * u.Angstrom
        os.chdir(BASE_DIR)

        tracing_nus = tracing_lambdas.to(u.Hz, u.spectral())
        config_dict = validate_yaml(CONFIG_PATH, schemapath=SCHEMA_PATH)
        config = Configuration(config_dict)

        adata = AtomData.from_hdf(config.atom_data)

        raw_marcs_model = read_marcs_model(
            Path(config.model.fname), gzipped=config.model.gzipped
        )
        stellar_model = raw_marcs_model.to_stellar_model(
            adata, final_atomic_number=config.model.final_atomic_number
        )

        adata.prepare_atom_data(
            np.arange(
                1,
                np.min(
                    [
                        len(
                            stellar_model.composition.elemental_mass_fraction.columns.tolist()
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
        self.adata = adata

        stellar_plasma = create_stellar_plasma(stellar_model, adata, config)

        stellar_radiation_field = RadiationField(
            tracing_nus, blackbody_flux_at_nu, stellar_model
        )

        calc_alphas(
            stellar_plasma=stellar_plasma,
            stellar_model=stellar_model,
            stellar_radiation_field=stellar_radiation_field,
            opacity_config=config.opacity,
        )

        self.tracing_lambdas = tracing_lambdas

        self.stellar_model = stellar_model
        self.stellar_plasma = stellar_plasma
        self.stellar_radiation_field = stellar_radiation_field
        self.config = config

    def time_run_stardis(self):
        run_stardis(CONFIG_PATH, self.tracing_lambdas)

    def time_raytrace(self):
        raytrace(
            self.stellar_model,
            self.stellar_radiation_field,
        )

    def time_calc_alpha_line_at_nu(self):
        calc_alpha_line_at_nu(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            self.config.opacity.line,
        )

    def time_calc_alpha_file(self):
        calc_alpha_file(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            list(self.config.opacity.file.keys())[0],
            list(self.config.opacity.file.values())[0],
        )

    def calc_alpha_rayleigh(self):
        calc_alpha_rayleigh(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            self.config.opacity.rayleigh,
        )

    def calc_alpha_electron(self):
        calc_alpha_electron(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            self.config.opacity.electron,
        )

    def time_create_plasma(self):
        create_stellar_plasma(self.stellar_model, self.adata, self.config)


class Sim100AA:
    """
    Class to benchmark Stardis functions.
    """

    timeout = 1800  # Worst case timeout of 30 mins

    def setup(self):

        tracing_lambdas = np.arange(6500, 6600, 0.01) * u.Angstrom
        os.chdir(BASE_DIR)

        tracing_nus = tracing_lambdas.to(u.Hz, u.spectral())
        config_dict = validate_yaml(CONFIG_PATH, schemapath=SCHEMA_PATH)
        config = Configuration(config_dict)

        adata = AtomData.from_hdf(config.atom_data)

        raw_marcs_model = read_marcs_model(
            Path(config.model.fname), gzipped=config.model.gzipped
        )
        stellar_model = raw_marcs_model.to_stellar_model(
            adata, final_atomic_number=config.model.final_atomic_number
        )

        adata.prepare_atom_data(
            np.arange(
                1,
                np.min(
                    [
                        len(
                            stellar_model.composition.elemental_mass_fraction.columns.tolist()
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
        self.adata = adata

        stellar_plasma = create_stellar_plasma(stellar_model, adata, config)

        stellar_radiation_field = RadiationField(
            tracing_nus, blackbody_flux_at_nu, stellar_model
        )

        calc_alphas(
            stellar_plasma=stellar_plasma,
            stellar_model=stellar_model,
            stellar_radiation_field=stellar_radiation_field,
            opacity_config=config.opacity,
        )

        self.tracing_lambdas = tracing_lambdas

        self.stellar_model = stellar_model
        self.stellar_plasma = stellar_plasma
        self.stellar_radiation_field = stellar_radiation_field
        self.config = config

    def time_run_stardis(self):
        run_stardis(CONFIG_PATH, self.tracing_lambdas)

    def time_raytrace(self):
        raytrace(
            self.stellar_model,
            self.stellar_radiation_field,
        )

    def time_calc_alpha_line_at_nu(self):
        calc_alpha_line_at_nu(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            self.config.opacity.line,
        )

    def time_calc_alpha_file(self):
        calc_alpha_file(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            list(self.config.opacity.file.keys())[0],
            list(self.config.opacity.file.values())[0],
        )

    def calc_alpha_rayleigh(self):
        calc_alpha_rayleigh(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            self.config.opacity.rayleigh,
        )

    def calc_alpha_electron(self):
        calc_alpha_electron(
            self.stellar_plasma,
            self.stellar_model,
            self.stellar_radiation_field.frequencies,
            self.config.opacity.electron,
        )
