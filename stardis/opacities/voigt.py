import numpy as np
import numba
from numba import cuda
import cmath

GPUs_available = cuda.is_available()

if GPUs_available:
    import cupy as cp

SQRT_PI = np.sqrt(np.pi, dtype=float)
PI = float(np.pi)


@numba.njit()
def _faddeeva(z):
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
    z = complex(z)
    x = float(z.real)
    y = float(z.imag)
    t = y - 1j * x
    s = abs(x) + y
    w = complex(0.0)
    u = t * t

    IN_REGION_I = s > 15.0
    IN_REGION_II = (not IN_REGION_I) and (s > 5.5)
    IN_REGION_III = (
        (not IN_REGION_I) and (not IN_REGION_II) and (y >= 0.195 * abs(x) - 0.176)
    )
    IN_REGION_IV = (not IN_REGION_I) and (not IN_REGION_II) and (not IN_REGION_III)

    # If in Region I
    w = 1j * 1 / SQRT_PI * z / (z**2 - 0.5) if IN_REGION_I else w

    # If in Region II
    w = (
        1j
        * (z * (z**2 * 1 / SQRT_PI - 1.4104739589))
        / (0.75 + z**2 * (z**2 - 3.0))
        if IN_REGION_II
        else w
    )

    # If in Region III
    w = (
        (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + 0.5642236 * t))))
        / (
            16.4955
            + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
        )
        if IN_REGION_III
        else w
    )

    # If in Region IV
    numerator = t * (
        36183.31
        - u
        * (
            3321.99
            - u
            * (1540.787 - u * (219.031 - u * (35.7668 - u * (1.320522 - u * 0.56419))))
        )
    )
    denominator = 32066.6 - u * (
        24322.8
        - u
        * (9022.23 - u * (2186.18 - u * (364.219 - u * (61.5704 - u * (1.84144 - u)))))
    )
    w = (cmath.exp(u) - numerator / denominator) if IN_REGION_IV else w

    return w


@numba.vectorize(nopython=True)
def faddeeva(z):
    return _faddeeva(z)


@cuda.jit
def _faddeeva_cuda(res, z):
    tid = cuda.grid(1)
    size = len(res)

    if tid < size:
        res[tid] = _faddeeva(z[tid])


def faddeeva_cuda(z, nthreads=256, ret_np_ndarray=False):
    size = len(z)
    nblocks = 1 + (size // nthreads)
    z = cp.asarray(z, dtype=complex)
    res = cp.empty_like(z)

    _faddeeva_cuda[nblocks, nthreads](res, z)
    return cp.asnumpy(res) if ret_np_ndarray else res


@numba.njit
def _voigt_profile(delta_nu, doppler_width, gamma):
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
    z = (delta_nu + (gamma / (4 * PI)) * 1j) / doppler_width
    phi = faddeeva(z).real / (SQRT_PI * doppler_width)
    return phi


@numba.vectorize(nopython=True)
def voigt_profile(delta_nu, doppler_width, gamma):
    return _voigt_profile(delta_nu, doppler_width, gamma)
