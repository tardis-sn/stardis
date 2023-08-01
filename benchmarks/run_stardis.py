# Import necessary code

import os
import numpy as np
from stardis.base import run_stardis
from astropy import units as u


class BenchmarkRunStardis:
    """
    Class to benchmark run_stardis function.
    """

    timeout = 1800  # Worst case timeout of 30 mins

    def setup(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.config = os.path.join(base_dir, "benchmark_config.yml")
        self.tracing_lambdas = np.arange(6540, 6590, 0.01) * u.Angstrom
        os.chdir(base_dir)

    def time_run_stardis(self):
        run_stardis(self.config, self.tracing_lambdas)
