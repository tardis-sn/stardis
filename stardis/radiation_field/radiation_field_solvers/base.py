import numba
import numpy as np


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
    w2: float
    """

    if delta_tau < 5e-4:
        w0 = delta_tau * (1 - delta_tau / 2)
        w1 = delta_tau**2 * (0.5 - delta_tau / 3)
        w2 = delta_tau**3 * (1 / 3 - delta_tau / 4)
    elif delta_tau > 50:
        w0 = 1.0
        w1 = 1.0
        w2 = 2.0
    else:
        exp_delta_tau = np.exp(-delta_tau)
        w0 = 1 - exp_delta_tau
        w1 = w0 - delta_tau * exp_delta_tau
        w2 = 2 * w1 - delta_tau * delta_tau * exp_delta_tau
    return w0, w1, w2


@numba.njit()
def single_theta_trace(
    geometry_dist_to_next_depth_point,
    temps,
    alphas,
    tracing_nus,
    theta,
    source_function,
):
    """
    Performs ray tracing at an angle following van Noort 2001 eq 14.

    Parameters
    ----------
    geometry_dist_to_next_depth_point : numpy.ndarray
        Distance to next depth point in geometry column as a numpy array from the finite volume model.
    temps : numpy.ndarray
        Temperatures in K of all depth point. Note that array must
        be transposed.
    alphas : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Total opacity at
        each depth point for each frequency in tracing_nus.
    tracing_nus : astropy.unit.quantity.Quantity
        Numpy array of frequencies used for ray tracing with units of Hz.
    theta : float
        Angle that the ray makes with the normal/radial direction.

    Returns
    -------
    I_nu_theta : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Output specific
        intensity at each depth point for each frequency in tracing_nus.
    """
    # Need to calculate a mean opacity for the traversal between points. Linearly interporlating, but could have a choice for interpolation scheme here.
    mean_alphas = (alphas[1:] + alphas[:-1]) * 0.5
    taus = mean_alphas.T * geometry_dist_to_next_depth_point / np.cos(theta)
    no_of_depth_gaps = len(geometry_dist_to_next_depth_point)

    ###TODO: Generalize this for source functions other than blackbody that may require args other than frequency and temperature
    source = source_function(tracing_nus, temps)
    delta_source = source[1:] - source[:-1]
    I_nu_theta = np.ones((no_of_depth_gaps + 1, len(tracing_nus))) * np.nan
    I_nu_theta[0] = source[0]  # the innermost depth point is the photosphere

    for i in range(len(tracing_nus)):  # iterating over nus (columns)
        for j in range(no_of_depth_gaps):  # iterating over depth_gaps (rows)
            curr_tau = taus[i, j]

            w0, w1, w2 = calc_weights(curr_tau)

            if curr_tau == 0:
                second_term = 0
            else:
                second_term = w1 * delta_source[j, i] / curr_tau
            if j < no_of_depth_gaps - 1:
                next_tau = taus[i, j + 1]
                third_term = w2 * (
                    (
                        (delta_source[j + 1, i] / next_tau)
                        + (-delta_source[j, i] / curr_tau)
                    )
                    / (curr_tau + next_tau)
                )

            else:
                third_term = 0
            I_nu_theta[j + 1, i] = (
                (1 - w0) * I_nu_theta[j, i]
                + w0
                * source[
                    j + 1, i
                ]  # Changed to j + 1 b/c van Noort 2001 mentions using Source of future point to update, not current.
                + second_term
                + third_term
            )

    return I_nu_theta


def raytrace(stellar_model, stellar_radiation_field, no_of_thetas=20):
    """
    Raytraces over many angles and integrates to get flux using the midpoint
    rule.

    Parameters
    ----------
    stellar_model : stardis.model.base.StellarModel
    stellar_radiation_field : stardis.radiation_field.base.StellarRadiationField
        Contains temperatures, frequencies, and opacities needed to calculate F_nu.
    no_of_thetas : int, optional
        Number of angles to sample for ray tracing, by default 20.

    Returns
    -------
    F_nu : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Output flux at
        each depth_point for each frequency in tracing_nus.
    """

    dtheta = (np.pi / 2) / no_of_thetas
    start_theta = dtheta / 2
    end_theta = (np.pi / 2) - (dtheta / 2)
    thetas = np.linspace(start_theta, end_theta, no_of_thetas)

    ###TODO: Thetas should probably be held by the model? Then can be passed in from there.
    for theta in thetas:
        weight = 2 * np.pi * dtheta * np.sin(theta) * np.cos(theta)
        stellar_radiation_field.F_nu += weight * single_theta_trace(
            stellar_model.geometry.dist_to_next_depth_point,
            stellar_model.temperatures.value.reshape(-1, 1),
            stellar_radiation_field.opacities.total_alphas,
            stellar_radiation_field.frequencies,
            theta,
            stellar_radiation_field.source_function,
        )

    return stellar_radiation_field.F_nu
