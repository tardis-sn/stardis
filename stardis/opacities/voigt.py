import numpy as np
import numba
from numba import cuda


@numba.njit
def faddeeva(z):
    """
    The Faddeeva function. Code adapted from
    https://github.com/tiagopereira/Transparency.jl/blob/966fb46c21/src/voigt.jl#L13.

    Parameters
    ----------
    z : complex

    Returns
    -------
    w : complex
    """
    s = abs(np.real(z)) + np.imag(z)

    if s > 15.0:
        # region I
        w = 1j * 1 / np.sqrt(np.pi) * z / (z**2 - 0.5)

    elif s > 5.5:
        # region II
        w = (
            1j
            * (z * (z**2 * 1 / np.sqrt(np.pi) - 1.4104739589))
            / (0.75 + z**2 * (z**2 - 3.0))
        )

    else:
        x = np.real(z)
        y = np.imag(z)
        t = y - 1j * x

        if y >= 0.195 * abs(x) - 0.176:
            # region III
            w = (
                16.4955
                + t * (20.20933 + t * (11.96482 + t * (3.778987 + 0.5642236 * t)))
            ) / (
                16.4955
                + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
            )

        else:
            # region IV
            u = t * t
            numerator = t * (
                36183.31
                - u
                * (
                    3321.99
                    - u
                    * (
                        1540.787
                        - u * (219.031 - u * (35.7668 - u * (1.320522 - u * 0.56419)))
                    )
                )
            )
            denominantor = 32066.6 - u * (
                24322.8
                - u
                * (
                    9022.23
                    - u * (2186.18 - u * (364.219 - u * (61.5704 - u * (1.84144 - u))))
                )
            )
            w = np.exp(u) - numerator / denominantor

    return w


@numba.njit
def voigt_profile(delta_nu, doppler_width, gamma):
    """
    Calculates the Voigt profile, the convolution of a Lorentz profile
    and a Gaussian profile.

    Parameters
    ----------
    delta_nu : float
        Difference between the frequency that the profile is being evaluated at
        and the line's resonance frequency.
    doppler_width : float
        Doppler width for Gaussian profile.
    gamma : float
        Broadening parameter for Lorentz profile.

    Returns
    -------
    phi : float
        Value of Voigt profile.
    """
    z = (delta_nu + (gamma / (4 * np.pi)) * 1j) / doppler_width
    phi = np.real(faddeeva(z)) / (np.sqrt(np.pi) * doppler_width)
    return phi
