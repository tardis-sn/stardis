import numpy as np
import pandas as pd
import logging

from astropy import constants as const, units as u
from tardis.util.base import element_symbol2atomic_number

from tardis.plasma.properties.base import ProcessingPlasmaProperty

ALPHA_COEFFICIENT = (np.pi * const.e.gauss**2) / (const.m_e.cgs * const.c.cgs)

logger = logging.getLogger(__name__)


class MoleculeNumberDensities(ProcessingPlasmaProperty):

    # Need to think about negative ions - ignoring for now
    # applicable for equilibrium constants given by Barklem and Collet 2016, which are given in SI units

    outputs = ("molecule_number_densities",)

    def calculate(self, ion_number_density, t_electrons, atomic_data):
        # This first implementation takes ~half a second. Much slower than is reasonable I think.

        number_densities_arr = np.zeros(
            (len(atomic_data.molecule_data.equilibrium_constants), len(t_electrons))
        )

        ###TODO - KEEP TRACK OF THE IONS SO WE CAN GET THEIR MASSES LATER FOR DOPPLER BROADENING PURPOSES
        ions_arr = np.zeros((len(atomic_data.molecule_data.equilibrium_constants), 2))

        equilibrium_const_temps = (
            atomic_data.molecule_data.equilibrium_constants.columns.values
        )
        included_elements = ion_number_density.index.get_level_values(0).unique()

        row_tracker = 0
        for row in atomic_data.molecule_data.dissociation_energies.iterrows():
            ionization_state_1 = 0
            ionization_state_2 = 0
            try:
                ion1_arr = row[1].Ion1.split("+")
                ion2_arr = row[1].Ion2.split("+")
                if len(ion1_arr) == 2:
                    ionization_state_1 = 1
                if len(ion2_arr) == 2:
                    ionization_state_2 = 1

                ion1 = element_symbol2atomic_number(ion1_arr[0])
                ion2 = element_symbol2atomic_number(ion2_arr[0])
                ions_arr[row_tracker] = [ion1, ion2]
                row_tracker += 1
                if ion1 not in included_elements:
                    logger.warning(
                        f"{row[1].Ion1} not in included elements. Assuming no {row[0]}."
                    )
                    continue
                elif ion2 not in included_elements:
                    logger.warning(
                        f"{row[1].Ion2} not in included elements. Assuming no {row[0]}."
                    )
                    continue
            except:
                row_tracker += 1
                continue  # This will currently skip over negative ions
            ion1_number_density = ion_number_density.loc[ion1, ionization_state_1]
            ion2_number_density = ion_number_density.loc[ion2, ionization_state_2]

            pressure_equilibirium_const_at_depth_point = np.interp(
                t_electrons,
                equilibrium_const_temps,
                atomic_data.molecule_data.equilibrium_constants.loc[row[0]].values,
            )
            equilibirium_const_at_depth_point = (
                (
                    10 ** (pressure_equilibirium_const_at_depth_point)
                    * (u.N / u.m**2)
                    / (const.k_B * t_electrons * u.K)
                )
                .to(u.cm**-3)
                .value
            )

            molecule_number_density = (
                ion1_number_density * ion2_number_density
            ) / equilibirium_const_at_depth_point

            number_densities_arr[
                atomic_data.molecule_data.equilibrium_constants.index.get_loc(row[0])
            ] = molecule_number_density

        densities_df = pd.DataFrame(
            number_densities_arr,
            index=atomic_data.molecule_data.equilibrium_constants.index,
            columns=ion_number_density.columns,
        )
        densities_df["ion1"] = ions_arr[:, 0].astype(int)
        densities_df["ion2"] = ions_arr[:, 1].astype(int)
        return densities_df


class AlphaLineValdMolecules(ProcessingPlasmaProperty):
    """
    Attributes
    ----------
    alpha_line_from_linelist : DataFrame
            A pandas DataFrame with dtype float. This represents the alpha calculation
            for each line from Vald at each depth point. Refer to Rybicki and Lightman
            equation 1.80. Voigt profiles are calculated later, and B_12 is substituted
            appropriately out for f_lu. This assumes LTE for lower level population.
    lines_from_linelist: DataFrame
            A pandas dataframe containing the lines and information about the lines in
            the same form and shape as alpha_line_from_linelist.
    """

    outputs = ("molecule_alpha_line_from_linelist", "molecule_lines_from_linelist")
    latex_name = (r"\alpha_{\textrm{line, vald}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2} n_{lower} f_{lu}}{m_{e} c}\
        \Big(1-exp(-h \nu / k T) \phi(\nu)\Big)",
    )

    def calculate(
        self,
        atomic_data,
        molecule_number_densities,
        t_electrons,
    ):
        # solve n_lower : n_i = N * g_i / U * e ^ (-E_i/kT)
        # get f_lu : loggf -> use g = 2j+1
        # emission_correction = (1-e^(-h*nu / kT))
        # alphas = ALPHA_COEFFICIENT * n_lower * f_lu * emission_correction

        ###TODO: handle other broadening parameters
        points = len(t_electrons)

        linelist = atomic_data.linelist_molecules[
            [
                "molecule",
                "wavelength",
                "log_gf",
                "e_low",
                "e_up",
                "j_lo",
                "j_up",
                "rad",
                "stark",
                "waals",
            ]
        ]

        # Calculate degeneracies
        linelist["g_lo"] = linelist.j_lo * 2 + 1
        linelist["g_up"] = linelist.j_up * 2 + 1

        exponent_by_point = np.exp(
            np.outer(
                -linelist.e_low.values * u.eV, 1 / (t_electrons * u.K * const.k_B)
            ).to(1)
        )

        prepared_molecule_number_densities = molecule_number_densities.copy()
        prepared_molecule_number_densities.index.name = "molecule"

        # grab densities for n_lower - need to use linelist as the index and normalize by dividing by the partition function
        linelist_with_densities = linelist.merge(
            prepared_molecule_number_densities,
            how="left",
            on=["molecule"],
        )

        n_lower = (
            (
                exponent_by_point * linelist_with_densities[np.arange(points)]
            ).values.T  # arange mask of the dataframe returns the set of densities of the appropriate ion for the line at each point
            * linelist_with_densities.g_lo.values
        )

        linelist["f_lu"] = (
            10**linelist.log_gf / linelist.g_lo
        )  # vald log gf is "oscillator strength f times the statistical weight g of the parent level"  see 1995A&AS..112..525P, section 2. Structure of VALD

        line_nus = (linelist.wavelength.values * u.AA).to(
            u.Hz, equivalencies=u.spectral()
        )

        emission_correction = 1 - np.exp(
            (
                -const.h
                / const.k_B
                * np.outer(
                    line_nus,
                    1 / (t_electrons * u.K),
                )
            ).to(1)
        )

        alphas = pd.DataFrame(
            (
                ALPHA_COEFFICIENT
                * n_lower
                * linelist.f_lu.values
                * emission_correction.T
            ).T
        )

        if np.any(np.isnan(alphas)) or np.any(np.isinf(np.abs(alphas))):
            raise ValueError(
                "Some alpha_line from vald are nan, inf, -inf " " Something went wrong!"
            )

        alphas["nu"] = line_nus.value
        linelist["nu"] = line_nus.value

        # Linelist preparation below is taken from opacities_solvers/base/calc_alpha_line_at_nu
        linelist["level_energy_lower"] = ((linelist["e_low"].values * u.eV).cgs).value
        linelist["level_energy_upper"] = ((linelist["e_up"].values * u.eV).cgs).value

        # Radiation broadening parameter is approximated as the einstein A coefficient. Vald parameters are in log scale.
        linelist["A_ul"] = 10 ** (
            linelist["rad"]
        )  # see 1995A&AS..112..525P for appropriate units - may be off by a factor of 4pi

        return alphas, linelist
