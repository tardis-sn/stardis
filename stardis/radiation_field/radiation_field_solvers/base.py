import numba
import numpy as np


@numba.njit()
def calc_weights_parallel(delta_tau):
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
    w0 = np.ones_like(delta_tau)
    w1 = np.ones_like(delta_tau)
    w2 = np.ones_like(delta_tau) * 2.0

    for i in range(delta_tau.shape[0]):
        if delta_tau[i] < 5e-4:
            w0[i] = delta_tau[i] * (1 - delta_tau[i] / 2)
            w1[i] = delta_tau[i] ** 2 * (0.5 - delta_tau[i] / 3)
            w2[i] = delta_tau[i] ** 3 * (1 / 3 - delta_tau[i] / 4)
        elif delta_tau[i] < 50:
            exp_delta_tau = np.exp(-delta_tau[i])
            w0[i] = 1 - exp_delta_tau
            w1[i] = w0[i] - delta_tau[i] * exp_delta_tau
            w2[i] = 2 * w1[i] - delta_tau[i] * delta_tau[i] * exp_delta_tau

    return w0, w1, w2


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
    w0 = np.ones_like(delta_tau)
    w1 = np.ones_like(delta_tau)
    w2 = np.ones_like(delta_tau) * 2.0

    mask1 = delta_tau < 5e-4
    mask2 = delta_tau > 50
    mask3 = ~np.logical_or(mask1, mask2)

    w0[mask1] = delta_tau[mask1] * (1 - delta_tau[mask1] / 2)
    w1[mask1] = delta_tau[mask1] ** 2 * (0.5 - delta_tau[mask1] / 3)
    w2[mask1] = delta_tau[mask1] ** 3 * (1 / 3 - delta_tau[mask1] / 4)

    exp_delta_tau = np.exp(-delta_tau[mask3])
    w0[mask3] = 1 - exp_delta_tau
    w1[mask3] = w0[mask3] - delta_tau[mask3] * exp_delta_tau
    w2[mask3] = 2 * w1[mask3] - delta_tau[mask3] * delta_tau[mask3] * exp_delta_tau

    return w0, w1, w2


@numba.njit(parallel=True)
def single_theta_trace_parallel(
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
    taus = (
        mean_alphas * geometry_dist_to_next_depth_point.reshape(-1, 1) / np.cos(theta)
    )
    no_of_depth_gaps = len(geometry_dist_to_next_depth_point)

    ###TODO: Generalize this for source functions other than blackbody that may require args other than frequency and temperature
    source = source_function(tracing_nus, temps)
    I_nu_theta = np.zeros((no_of_depth_gaps + 1, len(tracing_nus)))
    I_nu_theta[0] = source[0]  # the innermost depth point is the photosphere

    for gap_index in numba.prange(
        no_of_depth_gaps - 1
    ):  # iterating over depth_gaps (rows)

        w0, w1, w2 = calc_weights_parallel(taus[gap_index, :])

        second_term = (
            w1
            * (
                (source[gap_index + 1] - source[gap_index + 2])
                * (taus[gap_index, :] / taus[gap_index + 1, :])
                - (source[gap_index + 1] - source[gap_index])
                * (taus[gap_index + 1, :] / taus[gap_index, :])
            )
            / (taus[gap_index, :] + taus[gap_index + 1, :])
        )
        third_term = w2 * (
            (
                (
                    (source[gap_index + 2] - source[gap_index + 1])
                    / taus[gap_index + 1, :]
                )
                + ((source[gap_index] - source[gap_index + 1]) / taus[gap_index, :])
            )
            / (taus[gap_index, :] + taus[gap_index + 1, :])
        )
        I_nu_theta[gap_index + 1] = (
            (1 - w0) * I_nu_theta[gap_index]
            + w0 * source[gap_index + 1]
            + second_term
            + third_term
        )

    w0, w1, w2 = calc_weights_parallel(taus[-1, :])
    third_term = w2 * (source[-2] - source[-1]) / taus[-1, :] ** 2
    I_nu_theta[-1] = (1 - w0) * I_nu_theta[-2] + w0 * source[-1] + third_term

    return I_nu_theta


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
    taus = (
        mean_alphas * geometry_dist_to_next_depth_point.reshape(-1, 1) / np.cos(theta)
    )
    no_of_depth_gaps = len(geometry_dist_to_next_depth_point)

    ###TODO: Generalize this for source functions other than blackbody that may require args other than frequency and temperature
    source = source_function(tracing_nus, temps)
    I_nu_theta = np.ones((no_of_depth_gaps + 1, len(tracing_nus))) * np.nan
    I_nu_theta[0] = source[0]  # the innermost depth point is the photosphere

    for gap_index in range(no_of_depth_gaps):  # iterating over depth_gaps (rows)

        w0, w1, w2 = calc_weights(taus[gap_index, :])

        if gap_index < no_of_depth_gaps - 1:
            second_term = (
                w1
                * (
                    (source[gap_index + 1] - source[gap_index + 2])
                    * (taus[gap_index, :] / taus[gap_index + 1, :])
                    - (source[gap_index + 1] - source[gap_index])
                    * (taus[gap_index + 1, :] / taus[gap_index, :])
                )
                / (taus[gap_index, :] + taus[gap_index + 1, :])
            )
            third_term = w2 * (
                (
                    (
                        (source[gap_index + 2] - source[gap_index + 1])
                        / taus[gap_index + 1, :]
                    )
                    + ((source[gap_index] - source[gap_index + 1]) / taus[gap_index, :])
                )
                / (taus[gap_index, :] + taus[gap_index + 1, :])
            )

        else:  # handle the last depth point, assuming the same source as the preceeding value and tau as 0
            second_term = np.zeros_like(w0)
            third_term = (
                w2
                * (source[gap_index] - source[gap_index + 1])
                / taus[gap_index, :] ** 2
            )

        I_nu_theta[gap_index + 1] = (
            (1 - w0) * I_nu_theta[gap_index]
            + w0 * source[gap_index + 1]
            + second_term
            + third_term
        )

    return I_nu_theta


def raytrace(
    stellar_model, stellar_radiation_field, no_of_thetas=20, parallel_config=None
):
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
    if parallel_config is False:
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

    elif parallel_config:
        for theta in thetas:
            weight = 2 * np.pi * dtheta * np.sin(theta) * np.cos(theta)
            stellar_radiation_field.F_nu += weight * single_theta_trace_parallel(
                stellar_model.geometry.dist_to_next_depth_point,
                stellar_model.temperatures.value.reshape(-1, 1),
                stellar_radiation_field.opacities.total_alphas,
                stellar_radiation_field.frequencies,
                theta,
                stellar_radiation_field.source_function,
            )

    return stellar_radiation_field.F_nu
