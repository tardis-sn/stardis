import numba
import numpy as np
from astropy import units as u, constants as const

def bb_nu(tracing_nus, boundary_temps):
    bb_prefactor = (2 * const.h.cgs * tracing_nus ** 3) / const.c.cgs ** 2
    bb = bb_prefactor / (
        np.exp(
            (
                (const.h.cgs * tracing_nus)
                / (const.k_B.cgs * boundary_temps * u.K)
            ).value
        )
        - 1
    )
    return bb


def bb_lambda(tracing_lambdas, boundary_temps):
    bbw_prefactor = (2 * const.h.cgs * const.c.cgs ** 2) / (tracing_lambda) ** 5
    bbw = bbw_prefactor / (
        np.exp(
            (
                (const.h.cgs * const.c.cgs)
                / (const.k_B.cgs * tracing_lambda * boundary_temps * u.K)
            )
        )
        - 1
    )
    return bbw


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


#def raytrace()