import numpy as np
import pandas as pd

from astropy import constants as const, units as u

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


THERMAL_DE_BROGLIE_CONST = const.h**2 / (2 * np.pi * const.k_B)
H_MINUS_CHI = 0.754195 * u.eV  # see https://en.wikipedia.org/wiki/Hydrogen_anion
H2_DISSOCIATION_ENERGY = 4.476 * u.eV
ALPHA_COEFFICIENT = (np.pi * const.e.gauss**2) / (const.m_e.cgs * const.c.cgs)


class HMinusDensity(ProcessingPlasmaProperty):
    """
    Attributes
    ----------
    h_minus_density : Pandas DataFrame, dtype float
        Density of H-, indexed by shell.
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

    Attributes
    ----------
    h2_density : Pandas DataFrame, dtype float
        Density of H2, indexed by shell.
    """

    outputs = ("h2_density",)

    def calculate(self, ion_number_density, t_rad):
        t_rad = t_rad * u.K
        h_neutral_density = ion_number_density.loc[1, 0]
        thermal_de_broglie = (
            (2 * THERMAL_DE_BROGLIE_CONST / (const.m_p * t_rad)) ** (3 / 2)
        ).to(u.cm**3)
        phi = thermal_de_broglie * np.exp(H2_DISSOCIATION_ENERGY / (const.k_B * t_rad))
        return h_neutral_density**2 * phi.value


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


def create_stellar_plasma(stellar_model, atom_data):
    """
    Creates stellar plasma.

    Parameters
    ----------
    stellar_model : stardis.model.base.StellarModel
    atom_data : tardis.io.atom_data.base.AtomData

    Returns
    -------
    tardis.plasma.base.BasePlasma
    """

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
    plasma_modules.remove(tardis.plasma.properties.radiative_properties.TauSobolev)
    plasma_modules.remove(tardis.plasma.properties.plasma_input.TimeExplosion)
    plasma_modules.remove(tardis.plasma.properties.plasma_input.DilutionFactor)
    plasma_modules.remove(tardis.plasma.properties.plasma_input.HeliumTreatment)
    plasma_modules.remove(
        tardis.plasma.properties.plasma_input.ContinuumInteractionSpecies
    )
    plasma_modules += helium_lte_properties

    plasma_modules.append(AlphaLine)

    plasma_modules.append(HMinusDensity)
    plasma_modules.append(H2Density)

    # plasma_modules.remove(tardis.plasma.properties.radiative_properties.StimulatedEmissionFactor)
    # plasma_modules.remove(tardis.plasma.properties.general.SelectedAtoms)
    # plasma_modules.remove(tardis.plasma.properties.plasma_input.Density)

    return BasePlasma(
        plasma_properties=plasma_modules,
        t_rad=stellar_model.temperatures.value,
        abundance=stellar_model.composition.atomic_mass_fraction,
        atomic_data=atom_data,
        density=stellar_model.composition.density.value,
        link_t_rad_t_electron=1.0,
        nlte_ionization_species=[],
        nlte_excitation_species=[],
    )
