import pandas as pd
import numpy as np
from astropy import units as u, constants as const


THERMAL_DE_BROGLIE_CONST = const.h ** 2 / (2 * np.pi * const.m_e * const.k_B)
H_MINUS_CHI = (
    0.754195 * u.eV
)  # see https://en.wikipedia.org/wiki/Hydrogen_anion
SAHA_CONST = const.h ** 2 / (2 * np.pi * const.m_e * const.k_B)

# H minus opacity
def read_wbr_cross_section(wbr_fpath):
    wbr_cross_section = pd.read_csv(
        wbr_fpath, names=["wavelength", "cross_section"], comment="#",
    )
    wbr_cross_section.wavelength *= 10  ## nm to AA
    wbr_cross_section.cross_section *= 1e-18  ## to cm^2

    return wbr_cross_section


def calc_hminus_density(h_neutral_density, temperature, electron_density):
    thermal_de_broglie = (
        (THERMAL_DE_BROGLIE_CONST / temperature) ** (3 / 2)
    ).to(u.cm ** 3)
    phi = (thermal_de_broglie / 4) * np.exp(
        H_MINUS_CHI / (const.k_B * temperature)
    )
    return h_neutral_density * electron_density * phi.value


def calc_tau_h_minus(
    h_neutral_density,
    temperature,
    electron_density,
    wbr_fpath,
    tracing_wavelength,
    cell_length,
):
    # n or number density
    h_minus_density = calc_hminus_density(
        h_neutral_density, temperature, electron_density
    )

    wbr_cross_section = read_wbr_cross_section(wbr_fpath)

    # sigma
    h_minus_sigma_nu = np.interp(
        tracing_wavelength,
        wbr_cross_section.wavelength,
        wbr_cross_section.cross_section,
    )

    # tau = sigma * n * l; shape: (num cells, num tracing nus) - tau for each frequency in each cell
    tau_h_minus = h_minus_sigma_nu * (h_minus_density * cell_length)[None].T
    return tau_h_minus


# electron opacity
def calc_tau_e(
    splasma,
    marcs_model_fv,
    tracing_nus,
):
    
    h_minus_density = calc_hminus_density(
        h_neutral_density=splasma.ion_number_density.loc[(1, 0)].values,
        temperature=marcs_model_fv.t.values * u.K,
        electron_density=splasma.electron_densities.values,
    )
    
    new_electron_density = splasma.electron_densities.values - h_minus_density
    
    tau_e = const.sigma_T.cgs.value * splasma.electron_densities.values * marcs_model_fv.cell_length.values
    
    tau_e_final = np.zeros([len(marcs_model_fv),len(tracing_nus)])
    for j in range(len(marcs_model_fv)):
        tau_e_final[j]=tau_e[j]
    return tau_e_final


# line opacity
def calc_tau_nus(delta_tau, delta_nu):
    gauss_prefactor = 1 / (0.35e10 * np.sqrt(2 * np.pi))

    phi = gauss_prefactor * np.exp(-0.5 * (delta_nu / 0.35e10) ** 2)
    return (delta_tau * phi[None].T).sum(axis=0)
