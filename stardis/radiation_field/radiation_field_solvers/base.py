import numba
import numpy as np
from tqdm.notebook import tqdm


@numba.njit(parallel=True)
def calc_weights_parallel(delta_tau):
    """
    Calculates w0 and w1 coefficients in van Noort 2001 eq 14.

    Parameters
    ----------
    delta_tau : float
        Total optical depth at a single depth point and frequency

    Returns
    -------
    w0 : float
    w1 : float
    w2 : float
    """
    w0 = np.ones_like(delta_tau)
    w1 = np.ones_like(delta_tau)
    w2 = np.ones_like(delta_tau) * 2.0

    for gap_index in numba.prange(delta_tau.shape[0]):
        for nu_index in range(delta_tau.shape[1]):
            if delta_tau[gap_index, nu_index] < 5e-4:
                w0[gap_index, nu_index] = delta_tau[gap_index, nu_index] * (
                    1 - delta_tau[gap_index, nu_index] / 2
                )
                w1[gap_index, nu_index] = delta_tau[gap_index, nu_index] ** 2 * (
                    0.5 - delta_tau[gap_index, nu_index] / 3
                )
                w2[gap_index, nu_index] = delta_tau[gap_index, nu_index] ** 3 * (
                    1 / 3 - delta_tau[gap_index, nu_index] / 4
                )
            elif delta_tau[gap_index, nu_index] < 50:
                w0[gap_index, nu_index] = 1 - np.exp(-delta_tau[gap_index, nu_index])
                w1[gap_index, nu_index] = w0[gap_index, nu_index] - delta_tau[
                    gap_index, nu_index
                ] * np.exp(-delta_tau[gap_index, nu_index])
                w2[gap_index, nu_index] = 2 * w1[gap_index, nu_index] - delta_tau[
                    gap_index, nu_index
                ] ** 2 * np.exp(-delta_tau[gap_index, nu_index])

    return w0, w1, w2


def calc_weights(delta_tau):
    """
    Calculates w0 and w1 coefficients in van Noort 2001 eq 14.

    Parameters
    ----------
    delta_tau : array of shape (no_of_depth_gaps, no_of_frequencies)
        Total optical depth.

    Returns
    -------
    w0 : array of shape (no_of_depth_gaps, no_of_frequencies)
    w1 : array of shape (no_of_depth_gaps, no_of_frequencies)
    w2 : array of shape (no_of_depth_gaps, no_of_frequencies)
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

    w0, w1, w2 = calc_weights_parallel(taus)

    # return I_nu_theta
    for nu_index in numba.prange(len(tracing_nus)):
        for gap_index in range(no_of_depth_gaps - 1):
            # Start by solving all the weights and prefactors except the last jump which would go out of bounds
            second_term = (
                w1[gap_index, nu_index]
                * (
                    (source[gap_index + 1, nu_index] - source[gap_index + 2, nu_index])
                    * (taus[gap_index, nu_index] / taus[gap_index + 1, nu_index])
                    - (source[gap_index + 1, nu_index] - source[gap_index, nu_index])
                    * (taus[gap_index + 1, nu_index] / taus[gap_index, nu_index])
                )
                / (taus[gap_index, nu_index] + taus[gap_index + 1, nu_index])
            )
            third_term = w2[gap_index, nu_index] * (
                (
                    (
                        (
                            source[gap_index + 2, nu_index]
                            - source[gap_index + 1, nu_index]
                        )
                        / taus[gap_index + 1, nu_index]
                    )
                    + (
                        (source[gap_index, nu_index] - source[gap_index + 1, nu_index])
                        / taus[gap_index, nu_index]
                    )
                )
                / (taus[gap_index, nu_index] + taus[gap_index + 1, nu_index])
            )
            # Solve the raytracing equation for all points other than the final jump
            I_nu_theta[gap_index + 1, nu_index] = (
                (1 - w0[gap_index, nu_index]) * I_nu_theta[gap_index, nu_index]
                + w0[gap_index, nu_index] * source[gap_index + 1, nu_index]
                + second_term
                + third_term
            )

        third_term = (
            w2[-1, nu_index]
            * (source[-2, nu_index] - source[-1, nu_index])
            / taus[-1, nu_index] ** 2
        )
        # Solve the raytracing equation for the final jump assuming source does not change and tau is 0
        I_nu_theta[-1, nu_index] = (
            (1 - w0[-1, nu_index]) * I_nu_theta[-2, nu_index]
            + w0[-1, nu_index] * source[-1, nu_index]
            + third_term
        )

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

    # Solve for all the weights and prefactors except the last jump which would go out of bounds
    w0, w1, w2 = calc_weights(taus)
    gap_indices = np.arange(no_of_depth_gaps - 1)
    second_term = (
        w1[gap_indices]
        * (
            (source[gap_indices + 1] - source[gap_indices + 2])
            * (taus[gap_indices] / taus[gap_indices + 1])
            - (source[gap_indices + 1] - source[gap_indices])
            * (taus[gap_indices + 1] / taus[gap_indices])
        )
        / (taus[gap_indices] + taus[gap_indices + 1])
    )
    third_term = w2[gap_indices] * (
        (
            (
                (source[gap_indices + 2] - source[gap_indices + 1])
                / taus[gap_indices + 1]
            )
            + ((source[gap_indices] - source[gap_indices + 1]) / taus[gap_indices])
        )
        / (taus[gap_indices] + taus[gap_indices + 1])
    )

    for gap_index in range(
        no_of_depth_gaps - 1
    ):  # solve the ray tracing equation out to the surface of the star, not including the last jump
        I_nu_theta[gap_index + 1] = (
            (1 - w0[gap_index]) * I_nu_theta[gap_index]
            + w0[gap_index] * source[gap_index + 1]
            + second_term[gap_index]
            + third_term[gap_index]
        )

    # Solve for the last jump and the final output intensity assuming source does not change and tau is 0
    third_term = w2[-1] * (source[-2] - source[-1]) / taus[-1] ** 2
    I_nu_theta[-1] = (1 - w0[-1]) * I_nu_theta[-2] + w0[-1] * source[-1] + third_term

    return I_nu_theta


def raytrace(
    stellar_model, stellar_radiation_field, no_of_thetas=20, parallel_config=False
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
