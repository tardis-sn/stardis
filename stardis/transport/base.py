import numba
import numpy as np
from astropy import units as u, constants as const

H_CGS = const.h.cgs.value
C_CGS = const.c.cgs.value
K_B_CGS = const.k_B.cgs.value


@numba.njit
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

    bb_prefactor = (2 * H_CGS * tracing_nus**3) / C_CGS**2
    bb = bb_prefactor / (
        np.exp(((H_CGS * tracing_nus) / (K_B_CGS * boundary_temps))) - 1
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


@numba.njit()
def single_theta_trace(
    geometry_dist_to_next_depth_point,
    boundary_temps,
    alphas,
    tracing_nus,
    theta,
):
    """
    Performs ray tracing at an angle following van Noort 2001 eq 14.

    Parameters
    ----------
    geometry_dist_to_next_depth_point : numpy.ndarray
        Distance to next depth point in geometry column as a numpy array from the finite volume model.
    boundary_temps : numpy.ndarray
        Temperatures in K of all shell boundaries. Note that array must
        be transposed.
    alphas : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Total opacity in
        each shell for each frequency in tracing_nus.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    theta : float
        Angle that the ray makes with the normal/radial direction.

    Returns
    -------
    I_nu_theta : numpy.ndarray
        Array of shape (no_of_shells + 1, no_of_frequencies). Output specific
        intensity at each shell boundary for each frequency in tracing_nus.
    """
    # Messing with this. There's one fewer tau than alpha because you can't calculate optical depth for the first point because no distance to it.
    # Need to calculate a mean opacity for the traversal between points. Linearly interpolate?
    mean_alphas = (alphas[1:] + alphas[:-1]) * 0.5
    taus = mean_alphas.T * geometry_dist_to_next_depth_point / np.cos(theta)
    no_of_depth_gaps = len(geometry_dist_to_next_depth_point)

    bb = bb_nu(tracing_nus, boundary_temps)
    source = bb
    delta_source = bb[1:] - bb[:-1]
    I_nu_theta = np.ones((no_of_depth_gaps + 1, len(tracing_nus))) * np.nan
    I_nu_theta[0] = bb[0]  # the innermost boundary is photosphere

    for i in range(len(tracing_nus)):  # iterating over nus (columns)
        for j in range(no_of_depth_gaps):  # iterating over depth_gaps (rows)
            curr_tau = taus[i, j]

            w0, w1 = calc_weights(curr_tau)

            if curr_tau == 0:
                second_term = 0
            else:
                second_term = w1 * delta_source[j, i] / curr_tau

            I_nu_theta[j + 1, i] = (
                (1 - w0) * I_nu_theta[j, i]
                + w0
                * source[
                    j + 1, i
                ]  # Changed to j + 1 b/c van Noort 2001 mentions using Source of future point to update, not current.
                + second_term
            )  # van Noort 2001 eq 14

    return I_nu_theta


def raytrace(stellar_model, alphas, tracing_nus, no_of_thetas=20):
    """
    Raytraces over many angles and integrates to get flux using the midpoint
    rule.

    Parameters
    ----------
    stellar_model : stardis.model.base.StellarModel
    alphas : numpy.ndarray
        Array of shape (no_of_shells, no_of_frequencies). Total opacity in
        each shell for each frequency in tracing_nus.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    no_of_thetas : int, optional
        Number of angles to sample for ray tracing, by default 20.

    Returns
    -------
    F_nu : numpy.ndarray
        Array of shape (no_of_shells + 1, no_of_frequencies). Output flux at
        each shell boundary for each frequency in tracing_nus.
    """

    dtheta = (np.pi / 2) / no_of_thetas
    start_theta = dtheta / 2
    end_theta = (np.pi / 2) - (dtheta / 2)
    thetas = np.linspace(start_theta, end_theta, no_of_thetas)
    F_nu = np.zeros((len(stellar_model.geometry.r), len(tracing_nus)))

    for theta in thetas:
        weight = 2 * np.pi * dtheta * np.sin(theta) * np.cos(theta)
        F_nu += weight * single_theta_trace(
            stellar_model.geometry.dist_to_next_depth_point,
            stellar_model.temperatures.value.reshape(-1, 1),
            alphas,
            tracing_nus,
            theta,
        )

    return F_nu
