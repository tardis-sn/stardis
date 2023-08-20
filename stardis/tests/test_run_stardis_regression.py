from stardis import run_stardis
import numpy as np
from astropy import units as u
import os


def test_run_stardis_regression(ndarrays_regression):
    tracing_lambdas = np.arange(6540, 6590, 0.01) * u.Angstrom
    config_fname = "benchmarks/benchmark_config.yml"
    base_dir = os.path.abspath(os.path.dirname(config_fname))
    config_fname = os.path.basename(config_fname)
    os.chdir(base_dir)
    sim_result = run_stardis(config_fname, tracing_lambdas)
    ndarrays_regression.check({"alphas": sim_result.alphas})
