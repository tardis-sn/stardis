# Import necessary code

import os
import numpy as np
from stardis.base import run_stardis
from astropy import units as u

from tardis.io.atom_data import AtomData
from tardis.io.config_validator import validate_yaml
from tardis.io.config_reader import Configuration

from stardis.io.model.marcs import read_marcs_model
from stardis.plasma import create_stellar_plasma
from stardis.opacities import calc_alphas
from stardis.transport import raytrace
from stardis.opacities import calc_alpha_line_at_nu


class BenchmarkStardis:
    """
    Class to benchmark Stardis functions.
    """

    timeout = 1800  # Worst case timeout of 30 mins

    def setup(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        schema = os.path.join(base_dir, "config_schema.yml")
        config = os.path.join(base_dir, "benchmark_config.yml")
        tracing_lambdas = np.arange(6540, 6590, 0.01) * u.Angstrom
        os.chdir(base_dir)

        tracing_nus = tracing_lambdas.to(u.Hz, u.spectral())

        config_dict = validate_yaml(config, schemapath=schema)
        config = Configuration(config_dict)

        adata = AtomData.from_hdf(config.atom_data)

        if config.model.type == "marcs":
            raw_marcs_model = read_marcs_model(config.model.fname, gzipped=False)
            stellar_model = raw_marcs_model.to_stellar_model(
                adata, final_atomic_number=config.model.final_atomic_number
            )

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

        stellar_plasma = create_stellar_plasma(stellar_model, adata)

        alphas, gammas, doppler_widths = calc_alphas(
            stellar_plasma=stellar_plasma,
            stellar_model=stellar_model,
            tracing_nus=tracing_nus,
            opacity_config=config.opacity,
        )

        self.stellar_plasma = stellar_plasma
        self.stellar_model = stellar_model
        self.alphas = alphas
        self.tracing_nus = tracing_nus
        self.tracing_lambdas = tracing_lambdas
        self.config = config

    def time_run_stardis(self):
        run_stardis(self.config, self.tracing_lambdas)

    def time_raytrace(self):
        raytrace(
            self.stellar_model,
            self.alphas,
            self.tracing_nus,
            no_of_thetas=self.config.no_of_thetas,
        )

    def time_calc_alpha(self):
        calc_alpha_line_at_nu(
            self.stellar_plasma,
            self.stellar_model,
            self.tracing_nus,
            self.config.opacity.line,
        )
