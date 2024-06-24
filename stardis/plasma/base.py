import numpy as np
import pandas as pd
import logging

from astropy import constants as const, units as u
from scipy.interpolate import interp1d

from tardis.plasma.base import BasePlasma
from tardis.plasma.properties.base import (
    DataFrameInput,
    ProcessingPlasmaProperty,
)

from tardis.plasma.properties.property_collections import (
    basic_inputs,
    basic_properties,
    lte_excitation_properties,
    lte_ionization_properties,
    macro_atom_properties,
    dilute_lte_excitation_properties,
    nebular_ionization_properties,
    non_nlte_properties,
    nlte_properties,
    helium_nlte_properties,
    helium_numerical_nlte_properties,
    helium_lte_properties,
    detailed_j_blues_properties,
    detailed_j_blues_inputs,
    continuum_interaction_properties,
    continuum_interaction_inputs,
    adiabatic_cooling_properties,
    two_photon_properties,
)

import tardis.plasma
from tardis.opacities.tau_sobolev import TauSobolev


THERMAL_DE_BROGLIE_CONST = const.h**2 / (2 * np.pi * const.k_B)
H_MINUS_CHI = 0.754195 * u.eV  # see https://en.wikipedia.org/wiki/Hydrogen_anion
H2_DISSOCIATION_ENERGY = 4.476 * u.eV
ALPHA_COEFFICIENT = (np.pi * const.e.gauss**2) / (const.m_e.cgs * const.c.cgs)
H2_IONIZATION_ENERGY = 15.422 * u.eV
H2_PLUS_K_EQUILIBRIUM_CONSTANT = [
    0.9600,
    9.7683,
    29.997,
    59.599,
    265.32,
    845.01,
    1685.3,
    4289.5,
]  # from Stancil 1994 https://articles.adsabs.harvard.edu/pdf/1994ApJ...430..360S table 1
H2_PLUS_K_SAMPLE_TEMPS = [
    3150,
    4200,
    5040,
    6300,
    8400,
    12600,
    18600,
    25200,
]  # see directly above


class HMinusDensity(ProcessingPlasmaProperty):
    """
    Attributes
    ----------
    h_minus_density : Pandas DataFrame, dtype float
        Density of H-, indexed by depth point.
    """

    outputs = ("h_minus_density",)

    def calculate(self, ion_number_density, t_rad, electron_densities):
        t_rad = t_rad * u.K
        h_neutral_density = ion_number_density.loc[1, 0]
        thermal_de_broglie = (
            (THERMAL_DE_BROGLIE_CONST / (const.m_e * t_rad)) ** (3 / 2)
        ).to(u.cm**3)
        phi = (thermal_de_broglie / 4) * np.exp(H_MINUS_CHI / (const.k_B * t_rad))
        return h_neutral_density * electron_densities * phi.value


class H2Density(ProcessingPlasmaProperty):
    """
    Used Kittel and Kroemer "Thermal Physics".
    Currently unused.

    Attributes
    ----------
    h2_density : Pandas DataFrame, dtype float
        Density of H2, indexed by depth point.
    """

    outputs = ("h2_density",)

    def calculate(self, ion_number_density, t_rad):
        t_rad = t_rad * u.K
        h_neutral_density = ion_number_density.loc[1, 0].values * u.cm**-3
        thermal_de_broglie = (
            (2 * THERMAL_DE_BROGLIE_CONST / (const.m_p * t_rad)) ** (3 / 2)
        ).to(u.cm**3)
        phi = thermal_de_broglie * np.exp(H2_DISSOCIATION_ENERGY / (const.k_B * t_rad))
        return ((h_neutral_density**2 * phi).to(u.cm**-3)).value


class H2PlusDensity(ProcessingPlasmaProperty):
    """
    Post Saha equation calculation of H2+ density, following Stancil 1994, https://articles.adsabs.harvard.edu/pdf/1994ApJ...430..360S.
    Should be valid for low H2 densities.
    Equation from Kittel and Kroemer "Thermal Physics", chapter 9 equation 35.

    Attributes
    ----------
    h2_plus_density : Pandas DataFrame, dtype float
        Density of H2+, indexed by depth point.
    """

    outputs = ("h2_plus_density",)

    def calculate(self, ion_number_density, t_rad):
        interp_Ks = interp1d(H2_PLUS_K_SAMPLE_TEMPS, H2_PLUS_K_EQUILIBRIUM_CONSTANT)
        h_neutral_density = ion_number_density.loc[1, 0]
        h_plus_density = ion_number_density.loc[1, 1]
        return (
            h_neutral_density * h_plus_density / interp_Ks(t_rad) * 1e-19
        )  # scale factor from Stancil 1994 table 1


class AlphaLine(ProcessingPlasmaProperty):
    """
    Attributes
    ----------
    alpha_line : Pandas DataFrame, dtype float
        Sobolev optical depth for each line. Indexed by line.
        Columns as zones.
    """

    outputs = ("alpha_line",)
    latex_name = (r"\alpha_{\textrm{line}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2}}{m_{e} c}f_{lu}\
        n_{lower} \Big(1-\dfrac{g_{lower}n_{upper}}{g_{upper}n_{lower}}\Big)",
    )

    def calculate(
        self,
        lines,
        level_number_density,
        lines_lower_level_index,
        stimulated_emission_factor,
        f_lu,
    ):
        f_lu = f_lu.values[np.newaxis].T
        # wavelength = wavelength_cm.values[np.newaxis].T

        n_lower = level_number_density.values.take(
            lines_lower_level_index, axis=0, mode="raise"
        )
        alpha = ALPHA_COEFFICIENT * n_lower * stimulated_emission_factor * f_lu

        if np.any(np.isnan(alpha)) or np.any(np.isinf(np.abs(alpha))):
            raise ValueError(
                "Some alpha_line are nan, inf, -inf " " Something went wrong!"
            )

        df = pd.DataFrame(
            alpha,
            index=lines.index,
            columns=np.array(level_number_density.columns),
        )

        df["nu"] = lines.nu

        return df


class AlphaLineVald(ProcessingPlasmaProperty):
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

    outputs = ("alpha_line_from_linelist", "lines_from_linelist")
    latex_name = (r"\alpha_{\textrm{line, vald}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2} n_{lower} f_{lu}}{m_{e} c}\
        \Big(1-exp(-h \nu / k T) \phi(\nu)\Big)",
    )

    def calculate(
        self,
        atomic_data,
        ion_number_density,
        t_electrons,
        ionization_data,
        partition_function,
    ):
        # solve n_lower : n_i = N * g_i / U * e ^ (-E_i/kT)
        # get f_lu : loggf -> use g = 2j+1
        # emission_correction = (1-e^(-h*nu / kT))
        # alphas = ALPHA_COEFFICIENT * n_lower * f_lu * emission_correction

        ###TODO: handle other broadening parameters

        points = len(t_electrons)

        linelist = atomic_data.linelist.rename(columns={"ion_charge": "ion_number"})[
            [
                "atomic_number",
                "ion_number",
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

        # Truncate to final atomic number
        linelist = linelist[
            linelist.atomic_number <= (atomic_data.selected_atomic_numbers.max())
        ]

        # Calculate degeneracies
        linelist["g_lo"] = linelist.j_lo * 2 + 1
        linelist["g_up"] = linelist.j_up * 2 + 1

        exponent_by_point = np.exp(
            np.outer(
                -linelist.e_low.values * u.eV, 1 / (t_electrons * u.K * const.k_B)
            ).to(1)
        )

        # grab densities for n_lower - need to use linelist as the index and normalize by dividing by the partition function
        linelist_with_densities = linelist.merge(
            ion_number_density / partition_function,
            how="left",
            on=["atomic_number", "ion_number"],
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
        # Necessary for correct handling of ion numbers using charge instead of astronomy convention (i.e., 0 is neutral, 1 is singly ionized, etc.)
        ionization_energies = ionization_data.reset_index()
        ionization_energies["ion_number"] -= 1
        linelist = pd.merge(
            linelist,
            ionization_energies,
            how="left",
            on=["atomic_number", "ion_number"],
        )

        linelist["level_energy_lower"] = ((linelist["e_low"].values * u.eV).cgs).value
        linelist["level_energy_upper"] = ((linelist["e_up"].values * u.eV).cgs).value

        # Radiation broadening parameter is approximated as the einstein A coefficient. Vald parameters are in log scale.
        linelist["A_ul"] = 10 ** (
            linelist["rad"]
        )  # see 1995A&AS..112..525P for appropriate units - may be off by a factor of 4pi

        # Need to remove autoionization lines - can't handle with current broadening treatment because can't calculate effective principal quantum number
        valid_indices = linelist.level_energy_upper < linelist.ionization_energy

        return alphas[valid_indices], linelist[valid_indices]


class AlphaLineShortlistVald(ProcessingPlasmaProperty):
    """
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

    outputs = ("alpha_line_from_linelist", "lines_from_linelist")
    latex_name = (r"\alpha_{\textrm{line, vald}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2} n_{lower} f_{lu}}{m_{e} c}\
        \Big(1-exp(-h \nu / k T) \phi(\nu)\Big)",
    )

    def calculate(
        self,
        atomic_data,
        ion_number_density,
        t_electrons,
        ionization_data,
        partition_function,
    ):
        ###TODO: handle other broadening parameters
        points = len(t_electrons)

        linelist = atomic_data.linelist.rename(columns={"ion_charge": "ion_number"})[
            [
                "atomic_number",
                "ion_number",
                "wavelength",
                "log_gf",
                "e_low",
                "rad",
                "stark",
                "waals",
            ]
        ]

        # Truncate to final atomic number
        linelist = linelist[
            linelist.atomic_number <= (atomic_data.selected_atomic_numbers.max())
        ]

        # Calculate energy of upper level
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

        linelist_with_densities = linelist.merge(
            ion_number_density / partition_function,
            how="left",
            on=["atomic_number", "ion_number"],
        )

        prefactor = (
            exponent_by_point * linelist_with_densities[np.arange(points)]
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
        # Necessary for correct handling of ion numbers using charge instead of astronomy convention (i.e., 0 is neutral, 1 is singly ionized, etc.)
        ionization_energies = ionization_data.reset_index()
        ionization_energies["ion_number"] -= 1
        linelist = pd.merge(
            linelist,
            ionization_energies,
            how="left",
            on=["atomic_number", "ion_number"],
        )

        linelist["level_energy_lower"] = ((linelist["e_low"].values * u.eV).cgs).value
        linelist["level_energy_upper"] = ((linelist["e_up"].values * u.eV).cgs).value

        # Radiation broadening parameter is approximated as the einstein A coefficient. Vald parameters are in log scale.
        linelist["A_ul"] = 10 ** (linelist["rad"]) / (
            4 * np.pi
        )  # see 1995A&AS..112..525P for appropriate units - may be off by a factor of 4pi

        # Need to remove autoionization lines - can't handle with current broadening treatment because can't calculate effective principal quantum number
        valid_indices = linelist.level_energy_upper < linelist.ionization_energy

        return alphas[valid_indices], linelist[valid_indices]


# Properties that haven't been used in creating stellar plasma yet,
# might be useful in future ----------------------------------------------------


class InputNumberDensity(DataFrameInput):
    """
    Attributes
    ----------
    number_density : Pandas DataFrame, dtype float
        Indexed by atomic number, columns corresponding to zones.
    """

    outputs = ("number_density",)
    latex_name = ("N_{i}",)


class SelectedAtoms(ProcessingPlasmaProperty):
    """
    Attributes
    ----------
    selected_atoms : Pandas Int64Index, dtype int
        Atomic numbers of elements required for particular simulation.
    """

    outputs = ("selected_atoms",)

    def calculate(self, number_density):
        return number_density.index


# Creating stellar plasma ------------------------------------------------------


def create_stellar_plasma(
    stellar_model, atom_data, config
):  ###TODO: Clean up this function. Very messy.
    """
    Creates stellar plasma.

    Parameters
    ----------
    stellar_model : stardis.model.base.StellarModel
    atom_data : tardis.io.atom_data.base.AtomData
    config : stardis.config_reader.Configuration

    Returns
    -------
    tardis.plasma.base.BasePlasma
    """

    logging.info("Creating plasma")

    # basic_properties.remove(tardis.plasma.properties.general.NumberDensity)
    plasma_modules = []
    plasma_modules += basic_inputs
    plasma_modules += basic_properties
    plasma_modules += lte_ionization_properties
    plasma_modules += lte_excitation_properties
    plasma_modules += non_nlte_properties

    plasma_modules.append(
        tardis.plasma.properties.partition_function.LevelBoltzmannFactorNoNLTE
    )
    plasma_modules.remove(TauSobolev)
    plasma_modules.remove(tardis.plasma.properties.plasma_input.TimeExplosion)
    plasma_modules.remove(tardis.plasma.properties.plasma_input.DilutionFactor)
    plasma_modules.remove(tardis.plasma.properties.plasma_input.HeliumTreatment)
    plasma_modules.remove(
        tardis.plasma.properties.plasma_input.ContinuumInteractionSpecies
    )
    plasma_modules += helium_lte_properties

    if hasattr(config.opacity.file, "Hminus_bf"):
        plasma_modules.append(HMinusDensity)
    if hasattr(config.opacity.file, "H2minus_bf"):
        plasma_modules.append(H2Density)
    if hasattr(config.opacity.file, "H2plus_bf"):
        plasma_modules.append(H2PlusDensity)

    if config.opacity.line.vald_linelist.use_linelist:
        if config.opacity.line.vald_linelist.shortlist:
            plasma_modules.append(AlphaLineShortlistVald)
        else:
            plasma_modules.append(AlphaLineVald)
    else:
        plasma_modules.append(AlphaLine)

    # plasma_modules.remove(tardis.plasma.properties.radiative_properties.StimulatedEmissionFactor)
    # plasma_modules.remove(tardis.plasma.properties.general.SelectedAtoms)
    # plasma_modules.remove(tardis.plasma.properties.plasma_input.Density)

    return BasePlasma(
        plasma_properties=plasma_modules,
        t_rad=stellar_model.temperatures.value,
        abundance=stellar_model.composition.elemental_mass_fraction,
        atomic_data=atom_data,
        number_density=stellar_model.composition.elemental_number_density,
        link_t_rad_t_electron=1.0,
        nlte_ionization_species=[],
        nlte_excitation_species=[],
    )
