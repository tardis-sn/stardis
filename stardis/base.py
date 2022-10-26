from astropy import units as u

from tardis.io.atom_data import AtomData

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
    broadening_methods=["doppler", "linear_stark", "quadratic_stark", "van_der_waals", "radiation"]
):
    
    # TODO: allow inputing tracing_nus, allow inputing without units
    adata = AtomData.from_hdf('kurucz_cd23_chianti_H_He.h5')
    stellar_model = read_marcs_to_fv('data/marcs/sun.mod', adata, final_atomic_number=final_atomic_number)
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
    )
    I_nu = raytrace(stellar_model, alphas, tracing_nus)
    
    return StellarSpectrum(I_nu, tracing_nus, tracing_lambdas)
    

class StellarSpectrum:
    def __init__(self, I_nu, tracing_nus, tracing_lambdas):
        length = len(I_nu)
        I_lambda = I_nu * tracing_nus/tracing_lambdas
        # TODO: Units
        self.I_nu = I_nu
        self.I_lambda = I_lambda
        self.spectrum_nu = I_nu[length-1]
        self.spectrum_lambda = I_lambda[length-1]
        self.nus = tracing_nus
        self.lambdas = tracing_lambdas
