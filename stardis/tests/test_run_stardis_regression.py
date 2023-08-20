from stardis import run_stardis
import numpy as np
from astropy import units as u


def test_run_stardis_regression(data_regression):
    tracing_lambdas = np.arange(6540, 6590, 0.01) * u.Angstrom
    sim_result = run_stardis("stardis_example.yml", tracing_lambdas)
    data_regression.check(sim_result)
