import numpy as np
from astropy import units as u, constants as const
import numba

H_CGS = const.h.cgs.value
C_CGS = const.c.cgs.value
K_B_CGS = const.k_B.cgs.value


@numba.njit
def bb_nu(tracing_nus, temps):
    """
    Planck blackbody intensity distribution w.r.t. frequency.

    Parameters
    ----------
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    temps : numpy.ndarray
        Temperatures in K of all depth points. Note that array must
        be transposed. Must not have astropy units when passed into the function.

    Returns
    -------
    bb : astropy.unit.quantity.Quantity
        Numpy array of shape (no_of_depth_points, no_of_frequencies) with units
        of erg/(s cm^2 Hz). Blackbody specific intensity at each depth point
        for each frequency in tracing_nus.
    """

    bb_prefactor = (2 * H_CGS * tracing_nus**3) / C_CGS**2
    bb_flux_nu = bb_prefactor / (
        np.exp(((H_CGS * tracing_nus) / (K_B_CGS * temps))) - 1
    )
    return bb_flux_nu
