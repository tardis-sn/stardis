import numba
import numpy as np

@numba.njit
def calc_weights(delta_tau):
    if delta_tau < 5e-4:
        w0 = delta_tau * (1 - delta_tau / 2)
        w1 = delta_tau ** 2 * (0.5 - delta_tau / 3)
    elif delta_tau > 50:
        w0 = 1.0
        w1 = 1.0
    else:
        exp_delta_tau = np.exp(-delta_tau)
        w0 = 1 - exp_delta_tau
        w1 = w0 - delta_tau * exp_delta_tau
    return w0, w1