import numpy as np

from tardis.io.atom_data import AtomData

from astropy import units as u

from stardis.io import read_marcs_to_fv
from stardis.plasma import create_stellar_plasma
from stardis.opacities import calc_alphas
from stardis.transport import raytrace


def run_stardis(
    adata_fpath,
    marcs_model_fpath,
    tracing_lambdas,
    final_atomic_number=30,
    alpha_sources=["h_minus", "e", "h_photo", "line"],
    wbr_fpath=None,
    h_photo_levels=[1, 2, 3],
    h_photo_strength=7.91e-18,
    broadening_methods=[
        "doppler",
        "linear_stark",
        "quadratic_stark",
        "van_der_waals",
        "radiation",
    ],
    line_nu_min=0,
    line_nu_max=np.inf,
    line_range=None,
    no_of_thetas=20,
):
    """
    Runs STARDIS based on given parameters.
 
    Parameters
    ----------
    adata_fpath: str
        Filepath to the atomic data.
    marcs_model_fpath: str
        Filepath to the MARCS model.
    tracing_lambdas : astropy.units.Quantity
        Numpy array of wavelengths used for ray tracing with units of Angstroms.
    final_atomic_number : int, optional
        Atomic number for the final element included in the model. Default
        is 30.
    alpha_sources: list, optional
        List of sources of opacity to be considered. Options are "h_minus",
        "e", "h_photo", and "line". By default all are included.
    wbr_fpath: str, optional
        Filepath to read H minus cross sections. By default None. Must be
        provided if H minus opacities are calculated.
    h_photo_levels: list, optional
        Level numbers considered for hydrogen photoionization. By default
        [1,2,3] which corresponds to the n=2 level of hydrogen with fine
        splitting.
    h_photo_strength : float, optional
        Coefficient to inverse cube term in equation for photoionization
        opacity, expressed in cm^2. By default 7.91e-18.
    broadening_methods : list, optional
        List of broadening mechanisms to be considered. Options are "doppler",
        "linear_stark", "quadratic_stark", "van_der_waals", and "radiation".
        By default all are included.
    no_of_thetas : int, optional
        Number of angles to sample for ray tracing, by default 20.
   
    Returns
    -------
    stardis.base.StellarSpectrum
        The stellar spectrum.
    """

    # TODO: allow inputing tracing_nus, allow inputing without units
    adata = AtomData.from_hdf(adata_fpath)
    stellar_model = read_marcs_to_fv(
        marcs_model_fpath, adata, final_atomic_number=final_atomic_number
    )
    adata.prepare_atom_data(stellar_model.abundances.index.tolist())
    tracing_nus = tracing_lambdas.to(u.Hz, u.spectral())
    stellar_plasma = create_stellar_plasma(stellar_model, adata)
    alphas = calc_alphas(
        stellar_plasma=stellar_plasma,
        stellar_model=stellar_model,
        tracing_nus=tracing_nus,
        alpha_sources=alpha_sources,
        wbr_fpath=wbr_fpath,
        h_photo_levels=h_photo_levels,
        h_photo_strength=h_photo_strength,
        broadening_methods=broadening_methods,
        line_nu_min=line_nu_min,
        line_nu_max=line_nu_max,
        line_range=line_range,
    )
    F_nu = raytrace(stellar_model, alphas, tracing_nus, no_of_thetas=no_of_thetas)

    return StellarSpectrum(F_nu, tracing_nus)


class StellarSpectrum:
    """
    Class containing information about the synthetic stellar spectrum.

    Parameters
    ----------
    F_nu : numpy.ndarray
        Array of shape (no_of_shells + 1, no_of_frequencies). Output flux with
        respect to frequency at each shell boundary for each frequency.
    nus: astropy.units.Quantity
        Numpy array of frequencies used for spectrum with units of Hz.

    Attributes
    ----------
    F_nu : numpy.ndarray
        Array of shape (no_of_shells + 1, no_of_frequencies). Output flux with
        respect to frequency at each shell boundary for each frequency.
    F_lambda : numpy.ndarray
        Array of shape (no_of_shells + 1, no_of_frequencies). Output flux with
        respect to wavelength at each shell boundary for each wavelength.
    spectrum_nu : numpy.ndarray
        Output flux with respect to frequency at the outer boundary for each
        frequency.
    spectrum_lambda : numpy.ndarray
        Output flux with respect to wavelength at the outer boundary for each
        wavelength.
    nus: astropy.units.Quantity
        Numpy array of frequencies used for spectrum with units of Hz.
    lambdas: astropy.units.Quantity
        Numpy array of wavelengths used for spectrum with units of Angstroms.
    """

    def __init__(self, F_nu, nus):        
        length = len(F_nu)
        lambdas = nus.to(u.AA, u.spectral())
        F_lambda = F_nu * nus / lambdas
        # TODO: Units
        self.F_nu = F_nu
        self.F_lambda = F_lambda.value
        self.spectrum_nu = F_nu[length - 1]
        self.spectrum_lambda = F_lambda[length - 1]
        self.nus = nus
        self.lambdas = lambdas
