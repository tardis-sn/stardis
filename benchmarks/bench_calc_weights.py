import numpy as np
from stardis.radiation_field.radiation_field_solvers.base import (
    calc_weights,
    calc_weights_parallel,
)


class BenchCalcWeightsSmall:
    """
    Benchmark calc_weights and calc_weights_parallel on a small
    grid of 50 depth gaps and 1000 frequencies.
    """

    timeout = 1800  # Worst case timeout of 30 mins

    def setup(self):
        rng = np.random.default_rng(42)
        self.delta_tau = rng.uniform(0, 100, size=(50, 1000))
        calc_weights_parallel(rng.uniform(0, 100, size=(5, 10)))

    def time_calc_weights(self):
        calc_weights(self.delta_tau)

    def time_calc_weights_parallel(self):
        calc_weights_parallel(self.delta_tau)


class BenchCalcWeightsLarge:
    """
    Benchmark calc_weights and calc_weights_parallel on a large
    grid of 200 depth gaps and 5000 frequencies.
    """

    timeout = 1800  # Worst case timeout of 30 mins

    def setup(self):
        rng = np.random.default_rng(42)
        self.delta_tau = rng.uniform(0, 100, size=(200, 5000))
        calc_weights_parallel(rng.uniform(0, 100, size=(5, 10)))

    def time_calc_weights(self):
        calc_weights(self.delta_tau)

    def time_calc_weights_parallel(self):
        calc_weights_parallel(self.delta_tau)
