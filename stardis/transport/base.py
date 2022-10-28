import numba
import numpy as np
from astropy import units as u, constants as const


def bb_nu(tracing_nus, boundary_temps):
    """
    Planck blackbody intensity distribution w.r.t. frequency.

    Parameters
    ----------
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    boundary_temps : numpy.ndarray
        Temperatures in K of all shell boundaries. Note that array must
        be transposed.

    Returns
    -------
    bb : astropy.unit.quantity.Quantity
        Numpy array of shape (no_of_shells + 1, no_of_frequencies) with units
        of erg/(s cm^2 Hz). Blackbody specific intensity at each shell
        boundary for each frequency in tracing_nus.
    """

    bb_prefactor = (2 * const.h.cgs * tracing_nus**3) / const.c.cgs**2
    bb = bb_prefactor / (
        np.exp(
            ((const.h.cgs * tracing_nus) / (const.k_B.cgs * boundary_temps * u.K)).value
        )
        - 1
    )
    return bb


@numba.njit
def calc_weights(delta_tau):
    """
    Calculates w0 and w1 coefficients in van Noort 2001 eq 14.

    Parameters
    ----------
    delta_tau : float
        Total optical depth.

    Returns
    -------
    w0 : float
    w1 : float
    """

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


def single_theta_trace(stellar_model, alphas, tracing_nus, theta):
    """
    Performs ray tracing following van Noort 2001 eq 14.

    Parameters
    ----------
    bb : astropy.unit.quantity.Quantity
        Numpy array of shape (no_of_shells + 1, no_of_frequencies) with units
        of erg/(s cm^2 Hz). Blackbody specific intensity at each shell
        boundary for each frequency in tracing_nus.
    all_taus : iterable
        Contains all optical depths used. Each entry must be an array of shape
        (no_of_shells, no_of_frequencies).
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    no_of_shells : int
        Number of shells in the model.

    Returns
    -------
    I_nu_theta : numpy.ndarray
        Array of shape (no_of_shells + 1, no_of_frequencies). Output specific
        intensity at each shell boundary for each frequency in tracing_nus.
    """
    fv_geometry = stellar_model.fv_geometry
    taus = alphas.T * fv_geometry.cell_length.to_numpy() / np.cos(theta)
    no_of_shells = len(fv_geometry)

    bb = bb_nu(tracing_nus, stellar_model.boundary_temps)
    source = bb[1:].value
    delta_source = bb.diff(axis=0).value  # for cells, not boundary
    I_nu_theta = np.ones((no_of_shells + 1, len(tracing_nus))) * -99
    I_nu_theta[0] = bb[0]  # the innermost boundary is photosphere

    for i in range(len(tracing_nus)):  # iterating over nus (columns)

        for j in range(no_of_shells):  # iterating over cells/shells (rows)

            curr_tau = taus[i, j]

            w0, w1 = calc_weights(curr_tau)

            if curr_tau == 0:
                second_term = 0
            else:
                second_term = w1 * delta_source[j, i] / curr_tau

            I_nu_theta[j + 1, i] = (
                (1 - w0) * I_nu_theta[j, i] + w0 * source[j, i] + second_term
            )  # van Noort 2001 eq 14

    return I_nu_theta


<<<<<<< HEAD
def raytrace(stellar_model, alphas, tracing_nus, no_of_thetas=10):
    dtheta = (np.pi / 2) / no_of_thetas
    start_theta = dtheta / 2
    end_theta = (np.pi / 2) - (dtheta / 2)
    thetas = np.linspace(start_theta, end_theta, no_of_thetas)
    F_nu = np.zeros((len(stellar_model.fv_geometry) + 1, len(tracing_nus)))

    for theta in thetas:
        weight = 2 * np.pi * dtheta * np.sin(theta) * np.cos(theta)
        F_nu += weight * single_theta_trace(stellar_model, alphas, tracing_nus, theta)
=======
def raytrace(stellar_model, alphas, tracing_nus, no_of_thetas=20):

    thetas = np.linspace(0, np.pi / 2, no_of_thetas + 1, endpoint=False)[1:]
    F_nu = np.zeros((len(stellar_model.fv_geometry) + 1, len(tracing_nus)))

    for theta in thetas:
        factor = np.sin(theta) * np.cos(theta) / 2
        F_nu += factor * single_theta_trace(stellar_model, alphas, tracing_nus, theta)
>>>>>>> multi-angle raytracing

    return F_nu
