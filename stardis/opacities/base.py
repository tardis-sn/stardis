import pandas as pd
import numpy as np
from astropy import units as u, constants as const


THERMAL_DE_BROGLIE_CONST = const.h**2 / (2 * np.pi * const.m_e * const.k_B)
H_MINUS_CHI = 0.754195 * u.eV  # see https://en.wikipedia.org/wiki/Hydrogen_anion
SAHA_CONST = const.h**2 / (2 * np.pi * const.m_e * const.k_B)

# H minus opacity
def read_wbr_cross_section(wbr_fpath):
    wbr_cross_section = pd.read_csv(
        wbr_fpath,
        names=["wavelength", "cross_section"],
        comment="#",
    )
    wbr_cross_section.wavelength *= 10  ## nm to AA
    wbr_cross_section.cross_section *= 1e-18  ## to cm^2

    return wbr_cross_section


def calc_hminus_density(h_neutral_density, temperature, electron_density):
    thermal_de_broglie = ((THERMAL_DE_BROGLIE_CONST / temperature) ** (3 / 2)).to(
        u.cm**3
    )
    phi = (thermal_de_broglie / 4) * np.exp(H_MINUS_CHI / (const.k_B * temperature))
    return h_neutral_density * electron_density * phi.value


def calc_tau_h_minus(
    splasma,
    marcs_model_fv,
    tracing_nus,
    wbr_fpath,
):
    # n or number density
    h_minus_density = calc_hminus_density(
        h_neutral_density=splasma.ion_number_density.loc[(1, 0)].values,
        temperature=marcs_model_fv.t.values * u.K,
        electron_density=splasma.electron_densities.values,
    )

    wbr_cross_section = read_wbr_cross_section(wbr_fpath)

    tracing_lambda = tracing_nus.to(u.AA, u.spectral()).value

    # sigma
    h_minus_sigma_nu = np.interp(
        tracing_lambda,
        wbr_cross_section.wavelength,
        wbr_cross_section.cross_section,
    )

    # tau = sigma * n * l; shape: (num cells, num tracing nus) - tau for each frequency in each cell
    tau_h_minus = (
        h_minus_sigma_nu * (h_minus_density * marcs_model_fv.cell_length.values)[None].T
    )
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

    tau_e = (
        const.sigma_T.cgs.value
        * splasma.electron_densities.values
        * marcs_model_fv.cell_length.values
    )

    tau_e_final = np.zeros([len(marcs_model_fv), len(tracing_nus)])
    for j in range(len(marcs_model_fv)):
        tau_e_final[j] = tau_e[j]
    return tau_e_final


# photoionization opacity
def calc_tau_photo(
    splasma,
    marcs_model_fv,
    tracing_nus,
    level,
    strength,
    cutoff_frequency,
):
    """
    level is ordered pair (atomic_number,ion_number,level_number)
    """

    nl = splasma.level_number_density.loc[level] * marcs_model_fv.cell_length.values

    tau_photo = np.zeros([len(marcs_model_fv), len(tracing_nus)])

    for i in range(len(tracing_nus)):
        nu = tracing_nus[i]
        if nu.value >= cutoff_frequency:
            for j in range(len(marcs_model_fv)):
                tau_photo[j, i] = strength * nl[j] * (cutoff_frequency / nu.value) ** 3

    return tau_photo


# line opacity
def calc_tau_nus(splasma, marcs_model_fv, tracing_nus):

    tau_nus = np.zeros([len(marcs_model_fv), len(tracing_nus)])

    gauss_prefactor = 1 / (0.35e10 * np.sqrt(2 * np.pi))

    alpha_line = splasma.alpha_line.reset_index(drop=True).values[::-1]
    delta_tau_lines = alpha_line * marcs_model_fv.cell_length.values

    # transition doesn't happen at a specific nu due to several factors (changing temperatires, doppler shifts, relativity, etc.)
    # so we take a window 2e11 Hz wide - if nu falls within that, we consider it
    lines_nu = splasma.lines.nu.values[::-1]  # reverse to bring them to ascending order
    # search_sorted finds the index before which a (tracing_nu +- 1e11) can be inserted
    # in lines_nu array to maintain its sort order
    line_id_starts = lines_nu.searchsorted(tracing_nus.value - 1e11)
    line_id_ends = lines_nu.searchsorted(tracing_nus.value + 1e11)

    for i in range(len(tracing_nus)):  # iterating over nus (columns)
        nu, line_id_start, line_id_end = (
            tracing_nus[i],
            line_id_starts[i],
            line_id_ends[i],
        )

        if line_id_start != line_id_end:
            delta_tau = delta_tau_lines[line_id_start:line_id_end]
            delta_nu = nu.value - lines_nu[line_id_start:line_id_end]
            phi = gauss_prefactor * np.exp(-0.5 * (delta_nu / 0.35e10) ** 2)
            tau_nus[:, i] = (delta_tau * phi[None].T).sum(axis=0)
        else:
            tau_nus[:, i] = np.zeros(len(marcs_model_fv))

    return tau_nus
