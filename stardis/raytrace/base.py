import numba
import numpy as np
from astropy import units as u, constants as const


def bb_nu(tracing_nus, boundary_temps):
    bb_prefactor = (2 * const.h.cgs * tracing_nus**3) / const.c.cgs**2
    bb = bb_prefactor / (
        np.exp(
            ((const.h.cgs * tracing_nus) / (const.k_B.cgs * boundary_temps * u.K)).value
        )
        - 1
    )
    return bb


def bb_lambda(tracing_lambdas, boundary_temps):
    bbw_prefactor = (2 * const.h.cgs * const.c.cgs**2) / (tracing_lambda) ** 5
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
        w1 = delta_tau**2 * (0.5 - delta_tau / 3)
    elif delta_tau > 50:
        w0 = 1.0
        w1 = 1.0
    else:
        exp_delta_tau = np.exp(-delta_tau)
        w0 = 1 - exp_delta_tau
        w1 = w0 - delta_tau * exp_delta_tau
    return w0, w1


def raytrace(bb, all_taus, tracing_nus, no_of_shells):
    source = bb[1:].value
    delta_source = bb.diff(axis=0).value  # for cells, not boundary
    I_nu = np.ones((no_of_shells + 1, len(tracing_nus))) * -99
    I_nu[0] = bb[0]  # the innermost boundary is photosphere

    for i in range(len(tracing_nus)):  # iterating over nus (columns)

        for j in range(no_of_shells):  # iterating over cells/shells (rows)

            curr_tau = 0

            for tau in all_taus:
                curr_tau += tau[j, i]

            w0, w1 = calc_weights(curr_tau)

            if curr_tau == 0:
                second_term = 0
            else:
                second_term = w1 * delta_source[j, i] / curr_tau

            I_nu[j + 1, i] = (
                (1 - w0) * I_nu[j, i] + w0 * source[j, i] + second_term
            )  # van Noort 2001 eq 14

    return I_nu
