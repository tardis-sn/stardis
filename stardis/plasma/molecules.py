import numpy as np
import pandas as pd
import logging

from astropy import constants as const, units as u
from scipy.interpolate import CubicSpline
from tardis.util.base import element_symbol2atomic_number

from tardis.plasma.properties.base import ProcessingPlasmaProperty

ALPHA_COEFFICIENT = (np.pi * const.e.gauss**2) / (const.m_e.cgs * const.c.cgs)

logger = logging.getLogger(__name__)


class MoleculeIonNumberDensity(ProcessingPlasmaProperty):
    """

    Calculates the number density of each molecule at each depth point using Barklem and Collet 2016 equilibrium constants.
    Multiplies the number density of each constituent ion and divides by the number density constant. Negative ions are ignored.
    Attributes
    ----------
    molecule_number_density : DataFrame
            A pandas DataFrame with dtype float. This represents the number density of each molecule at each depth point.
    molecule_ion_map : DataFrame
            A pandas DataFrame with the constituent ions of each molecule. Useful for calculating molecular masses later for doppler broadening.
    """

    # Need to think about negative ions - ignoring for now
    # applicable for equilibrium constants given by Barklem and Collet 2016, which are given in SI units

    outputs = ("molecule_number_density", "molecule_ion_map")

    def calculate(self, ion_number_density, t_electrons, atomic_data):
        # Preprocessing - split ions into symbol, charge, and number
        try:
            molecules_df = atomic_data.molecule_data.dissociation_energies.copy()
        except:
            raise ValueError(
                "No molecular dissociation energies found in atomic data. Use Carsus to generate atomic data with the Barklem and Collet 2016 data."
            )

        for ion in [1, 2]:
            molecules_df = self.preprocess_ion(molecules_df, ion)

        number_densities_arr = np.zeros(
            (len(atomic_data.molecule_data.equilibrium_constants), len(t_electrons))
        )
        # Get the temperatures at which the equilibrium constants are given for interpolation
        equilibrium_const_temps = (
            atomic_data.molecule_data.equilibrium_constants.columns.values
        )
        included_elements = ion_number_density.index.get_level_values(0).unique()

        for (
            molecule,
            molecule_row,
        ) in (
            molecules_df.iterrows()
        ):  # Loop over all molecules, calculate number densities using Barklem and Collet 2016 equilibrium constants - if a component ion does not exist in the plasma or is negative, assume no molecule
            if (molecule_row.Ion1_charge == -1) or (molecule_row.Ion2_charge == -1):
                logger.warning(
                    f"Negative ionic molecules not currently supported. Assuming no {molecule}."
                )
                continue

            elif molecule_row.Ion1 not in included_elements:
                logger.warning(
                    f"{molecule_row.Ion1} not in included elements. Assuming no {molecule}."
                )
                continue
            elif molecule_row.Ion2 not in included_elements:
                logger.warning(
                    f"{molecule_row.Ion2} not in included elements. Assuming no {molecule}."
                )
                continue

            ion1_number_density = ion_number_density.loc[
                molecule_row.Ion1, molecule_row.Ion1_charge
            ]
            ion2_number_density = ion_number_density.loc[
                molecule_row.Ion2, molecule_row.Ion2_charge
            ]

            pressure_equil_spline = CubicSpline(
                equilibrium_const_temps,
                atomic_data.molecule_data.equilibrium_constants.loc[molecule].values,
                extrapolate=True,
            )

            pressure_equilibirium_const_at_depth_point = pressure_equil_spline(
                t_electrons
            )

            # Convert from pressure constants to number density constants using ideal gas law
            # k is the equilibrium concentration constant, pressure is Pa
            k = (
                (
                    (10**pressure_equilibirium_const_at_depth_point)
                    * (u.Pa)
                    / (const.k_B * t_electrons * u.K)
                ).to(u.cm**-3)
            ).value

            # Different formulae for homonuclear heteronuclear diatomic molecules
            if (molecule_row.Ion1 == molecule_row.Ion2) and (
                molecule_row.Ion1_charge == molecule_row.Ion2_charge
            ):
                molecule_number_density = (1 / 8) * (
                    (-((k * (k + 8 * ion1_number_density)) ** 0.5))
                    + k
                    + 4 * ion1_number_density
                )

            else:
                molecule_number_density = 0.5 * (
                    -np.sqrt(
                        k**2
                        + 2 * k * (ion1_number_density + ion2_number_density)
                        + (ion1_number_density - ion2_number_density) ** 2
                    )
                    + k
                    + ion1_number_density
                    + ion2_number_density
                )

            np.maximum(molecule_number_density, 0, out=molecule_number_density)

            number_densities_arr[
                atomic_data.molecule_data.equilibrium_constants.index.get_loc(molecule)
            ] = molecule_number_density

        molecule_densities_df = pd.DataFrame(
            number_densities_arr,
            index=atomic_data.molecule_data.equilibrium_constants.index,
            columns=ion_number_density.columns,
        )
        # Keep track of the individual ions - useful to calculate molecular masses later for doppler broadening
        molecule_ion_map = pd.DataFrame(
            molecules_df[["Ion1", "Ion2"]],
        )

        return molecule_densities_df, molecule_ion_map

    def preprocess_ion(self, molecules_df, ion):
        """
        Preprocesses a component ion in the molecule data to add the ion's atomic number and charge to the df.
        """
        molecules_df[
            [f"Ion{ion}_symbol", f"Ion{ion}_positive", f"Ion{ion}_negative"]
        ] = molecules_df[f"Ion{ion}"].str.extract(r"([A-Z][a-z]?+)(\+*)(\-*)")
        molecules_df[f"Ion{ion}"] = molecules_df[f"Ion{ion}_symbol"].apply(
            element_symbol2atomic_number
        )
        molecules_df[f"Ion{ion}_charge"] = molecules_df[f"Ion{ion}_positive"].apply(
            len
        ) - molecules_df[f"Ion{ion}_negative"].apply(len)
        return molecules_df


class MoleculePartitionFunction(ProcessingPlasmaProperty):
    """
    Processes the partition function for each molecule at each depth point by interpolating the partition function data.
    From Barklem and Collet 2016.
    Attributes
    ----------
    molecule_partition_function : DataFrame
            A pandas DataFrame with dtype float. This represents the partition function
            for each molecule at each depth point.
    """

    outputs = ("molecule_partition_function",)

    def calculate(self, t_electrons, atomic_data):
        partition_functions = pd.DataFrame(
            np.zeros(
                (len(atomic_data.molecule_data.partition_functions), len(t_electrons))
            ),
            index=atomic_data.molecule_data.partition_functions.index,
        )

        for molecule in atomic_data.molecule_data.partition_functions.index:
            partition_functions.loc[molecule] = np.interp(
                t_electrons,
                atomic_data.molecule_data.partition_functions.columns.values,
                atomic_data.molecule_data.partition_functions.loc[molecule].values,
            )

        return partition_functions


class AlphaLineValdMolecule(ProcessingPlasmaProperty):
    """
    Calculates the alpha (line integrated absorption coefficient in cm^-1) values for each molecular line from Vald at each depth point. This is adapted from the AlphaLineVald calculation.
    Attributes
    ----------
    molecule_alpha_line_from_linelist : DataFrame
            A pandas DataFrame with dtype float. This represents the alpha calculation
            for each line from Vald at each depth point. Refer to Rybicki and Lightman
            equation 1.80. Voigt profiles are calculated later, and B_12 is substituted
            appropriately out for f_lu. This assumes LTE for lower level population.
    molecule_lines_from_linelist: DataFrame
            A pandas dataframe containing the lines and information about the lines in
            the same form and shape as alpha_line_from_linelist.
    """

    outputs = ("molecule_alpha_line_from_linelist", "molecule_lines_from_linelist")
    latex_name = (r"\alpha_{\textrm{moleculeline, vald}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2} n_{lower} f_{lu}}{m_{e} c}\
        \Big(1-exp(-h \nu / k T) \phi(\nu)\Big)",
    )

    def calculate(
        self,
        atomic_data,
        molecule_number_density,
        t_electrons,
        molecule_partition_function,
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
        ].copy()

        # Calculate degeneracies
        linelist["g_lo"] = linelist.j_lo * 2 + 1
        linelist["g_up"] = linelist.j_up * 2 + 1

        exponent_by_point = np.exp(
            np.outer(
                -linelist.e_low.values * u.eV, 1 / (t_electrons * u.K * const.k_B)
            ).to(1)
        )

        molecule_densities_div_partition_function = molecule_number_density.copy().div(
            molecule_partition_function
        )
        molecule_densities_div_partition_function.index.name = "molecule"

        # grab densities for n_lower - need to use linelist as the index and normalize by dividing by the partition function
        linelist_with_density_div_partition_function = linelist.merge(
            molecule_densities_div_partition_function,
            how="left",
            on=["molecule"],
        )

        n_lower = (
            (
                exponent_by_point
                * linelist_with_density_div_partition_function[np.arange(points)]
            ).values.T  # arange mask of the dataframe returns the set of densities of the appropriate ion for the line at each point
            * linelist_with_density_div_partition_function.g_lo.values
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
        )  # see 1995A&AS..112..525P for appropriate units

        return alphas, linelist


class AlphaLineShortlistValdMolecule(ProcessingPlasmaProperty):
    """
    Calculates the alpha (line integrated absorption coefficient in cm^-1) values for each molecular line from Vald at each depth point. This is adapted from the AlphaLineShortlistVald calculation.

    Attributes
    ----------
    alpha_line_from_linelist : DataFrame
            A pandas DataFrame with dtype float. This represents the alpha calculation
            for each line from Vald at each depth point. This is adapted from the AlphaLineVald calculation
            because shortlists do not contain js or an upper energy level. This works because the degeneracies
            cancel between the n_lower recalculation and in calculating f_lu from g*f given by the linelist.
    lines_from_linelist: DataFrame
            A pandas dataframe containing the lines and information about the lines in
            the same form and shape as alpha_line_from_linelist.
    """

    outputs = ("molecule_alpha_line_from_linelist", "molecule_lines_from_linelist")
    latex_name = (r"\alpha_{\textrm{moleculeline, vald}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2} n_{lower} f_{lu}}{m_{e} c}\
        \Big(1-exp(-h \nu / k T) \phi(\nu)\Big)",
    )

    def calculate(
        self,
        atomic_data,
        molecule_number_density,
        t_electrons,
        molecule_partition_function,
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
                "rad",
                "stark",
                "waals",
            ]
        ].copy()

        linelist["e_up"] = (
            (
                linelist.e_low.values * u.eV
                + (const.h * const.c / (linelist.wavelength.values * u.AA))
            )
            .to(u.eV)
            .value
        )

        exponent_by_point = np.exp(
            np.outer(
                -linelist.e_low.values * u.eV, 1 / (t_electrons * u.K * const.k_B)
            ).to(1)
        )

        molecule_densities_div_partition_function = molecule_number_density.copy().div(
            molecule_partition_function
        )
        molecule_densities_div_partition_function.index.name = "molecule"

        # grab densities for n_lower - need to use linelist as the index and normalize by dividing by the partition function
        linelist_with_density_div_partition_function = linelist.merge(
            molecule_densities_div_partition_function,
            how="left",
            on=["molecule"],
        )

        prefactor = (
            exponent_by_point
            * linelist_with_density_div_partition_function[np.arange(points)]
        ).values.T  # arange mask of the dataframe returns the set of densities of the appropriate ion for the line at each point

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
                * prefactor
                * 10**linelist.log_gf.values
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
        )  # see 1995A&AS..112..525P for appropriate units

        return alphas, linelist
