import numba
import numpy as np
from astropy import units as u


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

    for nu_index in numba.prange(delta_tau.shape[1]):
        for gap_index in range(delta_tau.shape[0]):
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
    ray_dist_to_next_depth_point,
    temps,
    alphas,
    tracing_nus,
    source_function,
    inward_rays=False,
):
    """
    Performs ray tracing at an angle following van Noort 2001 eq 14, parallelized over frequencies.

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
    inward_rays : bool, optional
        If True, rays are traced from the outermost layer to the innermost layer before the standard tracing
        from the innermost layer to the outermost layer, by default False. Useful in spherical geometries.

    Returns
    -------
    I_nu_theta : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Output specific
        intensity at each depth point for each frequency in tracing_nus.
    """
    # Need to calculate a mean opacity for the traversal between points. Linearly interporlating. Van Noort paper suggests interpolating
    # alphas in log space. We could have a choice for interpolation scheme here.
    mean_alphas = np.exp((np.log(alphas[1:]) + np.log(alphas[:-1])) * 0.5)

    taus = np.zeros_like(mean_alphas, dtype=np.float64)
    for nu_index in numba.prange(taus.shape[1]):
        for gap_index in range(taus.shape[0]):
            taus[gap_index, nu_index] = (
                mean_alphas[gap_index, nu_index]
                * ray_dist_to_next_depth_point[gap_index]
            )

    no_of_depth_gaps = len(ray_dist_to_next_depth_point)

    source = source_function(tracing_nus, temps)
    I_nu_theta = np.zeros((no_of_depth_gaps + 1, len(tracing_nus)))

    # the innermost depth point is approximated as a blackbody - changed to 0 for spherical geometry case. This should be fine if models are optically thick.

    w0, w1, w2 = calc_weights_parallel(taus)

    # In spherical geometry, for outside rays, you need to handle the ray that goes through the star and comes out the other side.
    if inward_rays:
        for nu_index in numba.prange(len(tracing_nus)):
            for gap_index in np.arange(0, no_of_depth_gaps)[::-1]:
                # Start by solving all the weights and prefactors except the last jump which would go out of bounds
                if taus[gap_index, nu_index] == 0 or taus[gap_index - 1, nu_index] == 0:
                    I_nu_theta[gap_index, nu_index] = I_nu_theta[
                        gap_index + 1, nu_index
                    ]  # If no optical depth, no change in intensity
                else:
                    second_term = (
                        w1[gap_index, nu_index]
                        * (
                            (
                                source[gap_index, nu_index]
                                - source[gap_index - 1, nu_index]
                            )
                            * (
                                taus[gap_index, nu_index]
                                / taus[gap_index - 1, nu_index]
                            )
                            - (
                                source[gap_index, nu_index]
                                - source[gap_index + 1, nu_index]
                            )
                            * (
                                taus[gap_index - 1, nu_index]
                                / taus[gap_index, nu_index]
                            )
                        )
                        / (taus[gap_index, nu_index] + taus[gap_index - 1, nu_index])
                    )
                    third_term = w2[gap_index, nu_index] * (
                        (
                            (
                                (
                                    source[gap_index - 1, nu_index]
                                    - source[gap_index, nu_index]
                                )
                                / taus[gap_index - 1, nu_index]
                            )
                            + (
                                (
                                    source[gap_index + 1, nu_index]
                                    - source[gap_index, nu_index]
                                )
                                / taus[gap_index, nu_index]
                            )
                        )
                        / (taus[gap_index, nu_index] + taus[gap_index - 1, nu_index])
                    )
                    # Solve the raytracing equation for all points other than the final jump
                    I_nu_theta[gap_index, nu_index] = (
                        (1 - w0[gap_index, nu_index])
                        * I_nu_theta[gap_index + 1, nu_index]
                        + w0[gap_index, nu_index] * source[gap_index, nu_index]
                        + second_term
                        + third_term
                    )

    for nu_index in numba.prange(len(tracing_nus)):
        for gap_index in range(no_of_depth_gaps - 1):
            # Start by solving all the weights and prefactors except the last jump which would go out of bounds
            if taus[gap_index, nu_index] == 0:
                I_nu_theta[gap_index + 1, nu_index] = I_nu_theta[
                    gap_index, nu_index
                ]  # If no optical depth, no change in intensity
            else:
                second_term = (
                    w1[gap_index, nu_index]
                    * (
                        (
                            source[gap_index + 1, nu_index]
                            - source[gap_index + 2, nu_index]
                        )
                        * (taus[gap_index, nu_index] / taus[gap_index + 1, nu_index])
                        - (
                            source[gap_index + 1, nu_index]
                            - source[gap_index, nu_index]
                        )
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
                            (
                                source[gap_index, nu_index]
                                - source[gap_index + 1, nu_index]
                            )
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

        # Below is the final jump, assuming source does not change and tau is 0 beyond the last depth point

        if taus[-1, nu_index] == 0:
            I_nu_theta[-1, nu_index] = I_nu_theta[-2, nu_index]
        else:
            third_term = (
                w2[-1, nu_index]
                * (source[-2, nu_index] - source[-1, nu_index])
                / taus[-1, nu_index] ** 2
            )
            # Solve the raytracing equation for the final jump
            I_nu_theta[-1, nu_index] = (
                (1 - w0[-1, nu_index]) * I_nu_theta[-2, nu_index]
                + w0[-1, nu_index] * source[-1, nu_index]
                + third_term
            )

    return I_nu_theta


def all_thetas_trace(
    ray_dist_to_next_depth_point,
    temps,
    alphas,
    tracing_nus,
    num_of_thetas,
    source_function,
    inward_rays=False,
):
    """
    Performs ray tracing at an angle following van Noort 2001 eq 14.

    Currently not using this method in favor of the parallelized njit version because it does not work correctly in spherical geometry.
    This method is kept for reference.

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
    num_of_thetas : int
    inward_rays : bool, optional
        If True, rays are traced from the outermost layer to the innermost layer before the standard tracing
        from the innermost layer to the outermost layer, by default False. Useful in spherical geomet

    Returns
    -------
    I_nu_theta : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Output specific
        intensity at each depth point for each frequency in tracing_nus.
    """
    # Need to calculate a mean opacity for the traversal between points. Linearly interporlating. Van Noort paper suggests interpolating
    # alphas in log space. We could have a choice for interpolation scheme here.
    mean_alphas = np.exp((np.log(alphas[1:]) + np.log(alphas[:-1])) * 0.5)

    taus = (
        mean_alphas[:, :, np.newaxis] * ray_dist_to_next_depth_point[:, np.newaxis, :]
    )
    no_of_depth_gaps = len(ray_dist_to_next_depth_point)

    source = source_function(tracing_nus, temps)[:, :, np.newaxis]
    I_nu_theta = np.zeros((no_of_depth_gaps + 1, len(tracing_nus), num_of_thetas))
    # the innermost depth point is approximated as a blackbody - changed to 0 for spherical geometry case.

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

    if inward_rays:
        pass

    for gap_index in range(
        no_of_depth_gaps - 1
    ):  # solve the ray tracing equation out to the surface of the star, not including the last jump
        tau_0_mask = taus[gap_index] == 0
        I_nu_theta[gap_index + 1][tau_0_mask] = I_nu_theta[
            gap_index, tau_0_mask
        ]  # If no optical depth, no change in intensity
        I_nu_theta[gap_index + 1, ~tau_0_mask] = (
            (1 - w0[gap_index, ~tau_0_mask]) * I_nu_theta[gap_index, ~tau_0_mask]
            + w0[gap_index, ~tau_0_mask]
            * np.repeat(source[gap_index + 1], num_of_thetas, axis=1)[~tau_0_mask]
            + second_term[gap_index, ~tau_0_mask]
            + third_term[gap_index, ~tau_0_mask]
        )

    # Solve for the last jump and the final output intensity assuming source does not change and tau forward beyond boundary is 0
    third_term = w2[-1] * (source[-2] - source[-1]) / taus[-1] ** 2
    I_nu_theta[-1] = (1 - w0[-1]) * I_nu_theta[-2] + w0[-1] * source[-1] + third_term

    return I_nu_theta


def raytrace(
    stellar_model,
    stellar_radiation_field,
    n_threads=1,
):
    """
    Raytraces over many angles and integrates to get flux using the midpoint
    rule.

    Parameters
    ----------
    stellar_model : stardis.model.base.StellarModel
    stellar_radiation_field : stardis.radiation_field.base.StellarRadiationField
        Contains temperatures, frequencies, and opacities needed to calculate F_nu.
    spherical : bool, optional
        If True, rays are traced from the outermost layer to the innermost layer before the standard tracing
        from the innermost layer to the outermost layer, by default False. Distance of rays through
        each the star is also calculated differently.

    Returns
    -------
    F_nu : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Output flux at
        each depth_point for each frequency in tracing_nus.
    """

    if stellar_model.spherical:
        ray_distances = calculate_spherical_ray(
            stellar_radiation_field.thetas, stellar_model.geometry.r
        )
        inward_rays = True
    else:
        ray_distances = stellar_model.geometry.dist_to_next_depth_point.reshape(
            -1, 1
        ) / np.cos(stellar_radiation_field.thetas)
        inward_rays = False
    if (
        False
    ):  # Commenting out serial threaded block for now - currently doesn't work with spherical geometry and not sure it's worth maintaining
        # if n_threads == 1:  # Single threaded
        stellar_radiation_field.I_nus = all_thetas_trace(
            ray_distances,
            stellar_model.temperatures.value.reshape(-1, 1),
            stellar_radiation_field.opacities.total_alphas,
            stellar_radiation_field.frequencies,
            len(stellar_radiation_field.thetas),
            stellar_radiation_field.source_function,
            inward_rays,
        )
        stellar_radiation_field.F_nu = np.sum(
            stellar_radiation_field.I_nus_weights * stellar_radiation_field.I_nus,
            axis=2,
        )
    else:  # Parallel threaded
        for theta_index, theta in enumerate(stellar_radiation_field.thetas):
            I_nu = single_theta_trace_parallel(
                ray_distances[:, theta_index],
                stellar_model.temperatures.value.reshape(-1, 1),
                stellar_radiation_field.opacities.total_alphas,
                stellar_radiation_field.frequencies,
                stellar_radiation_field.source_function,
                inward_rays=inward_rays,
            )
            if stellar_radiation_field.track_individual_intensities:
                stellar_radiation_field.I_nus[:, :, theta_index] = I_nu

            stellar_radiation_field.F_nu += (
                I_nu * stellar_radiation_field.I_nus_weights[theta_index]
            )

    if stellar_model.spherical:
        photospheric_correction = (
            stellar_model.geometry.r[-1] / stellar_model.geometry.reference_r
        ) ** 2
        stellar_radiation_field.F_nu *= photospheric_correction  # Outermost radius is larger than the photosphere so need to downscale the flux

    return stellar_radiation_field.F_nu


def calculate_spherical_ray(thetas, depth_points_radii):
    """
    Calculate the distance a ray travels between the depth points of the star in spherical geometry.

    Parameters
    ----------
    thetas : numpy.ndarray
        Array of angles in radians.
    depth_points_radii : numpy.ndarray
        Array of radii of each depth point in the star.

    Returns
    -------
    ray_distance_through_layer_by_impact_parameter : numpy.ndarray
        Array of shape (no_of_depth_points - 1, no_of_thetas). Distance a ray travels through each layer of the star
        for each angle.
    """
    ray_distance_through_layer_by_impact_parameter = np.zeros(
        (len(depth_points_radii) - 1, len(thetas))
    )

    for theta_index, theta in enumerate(thetas):
        b = depth_points_radii[-1] * np.sin(theta)  # impact parameter of the ray
        ray_z_coordinate_grid = np.sqrt(
            depth_points_radii**2 - b**2
        )  # Rays that don't go deeper than a layer will have a nan here

        ray_distance = np.diff(ray_z_coordinate_grid)
        ray_distance_through_layer_by_impact_parameter[
            ~np.isnan(ray_distance), theta_index
        ] = ray_distance[~np.isnan(ray_distance)]

    return ray_distance_through_layer_by_impact_parameter
