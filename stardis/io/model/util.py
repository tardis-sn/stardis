from pathlib import Path
import pandas as pd
import numpy as np
import logging
from tardis.util.base import element_symbol2atomic_number, atomic_number2element_symbol


PATH_TO_ASPLUND_2009 = Path(__file__).parent / "data" / "asplund_2009_processed.csv"
PATH_TO_ASPLUND_2020 = Path(__file__).parent / "data" / "asplund_2020_processed.csv"

ASPLUND_2009_HE_MASS_FRAC_Y = (
    0.2492280  # The Asplund 2009 mass fraction of measured He
)
ASPLUND_2009_HEAVY_MASS_FRAC_Z = (
    0.01337  # The Asplund 2009 mass fraction of measured heavy metals
)

ASPLUND_2020_HE_MASS_FRAC_Y = 0.2423
ASPLUND_2020_HEAVY_MASS_FRAC_Z = 0.0139

def create_scaled_solar_profile(
    atom_data,
    helium_mass_frac_Y=ASPLUND_2020_HE_MASS_FRAC_Y,
    heavy_metal_mass_frac_Z=ASPLUND_2020_HEAVY_MASS_FRAC_Z,
    final_atomic_number=None,
    composition_source="asplund_2020"
):
    """
    Scales the solar mass fractions based on the given atom data, helium_mass_frac_Y, and heavy_metal_mass_frac_Z, using the photospheric composition from Asplund 2009.
    Default helium_mass_frac_Y and heavy_metal_mass_frac_Z are calculated using Asplund 2009 and NIST atomic weights, i.e., if you use their default values you will get
    back the solar composition measured by Asplund 2009.

    Args:
        atom_data: The atom data used to scale the solar mass fractions.
        helium_mass_frac_Y: The helium abundance. Default is 0.2492280.
        heavy_metal_mass_frac_Z: The metallicity. Default is 0.01337.

    Returns:
        pandas.DataFrame: The scaled mass fractions.

    """
    if composition_source == "asplund_2020":
        solar_values = pd.read_csv(PATH_TO_ASPLUND_2020, index_col=0)
        he_y_tot = ASPLUND_2020_HE_MASS_FRAC_Y
        he_z_tot = ASPLUND_2020_HEAVY_MASS_FRAC_Z
        if helium_mass_frac_Y == -99:
            helium_mass_frac_Y = he_y_tot
        if heavy_metal_mass_frac_Z == -99:
            heavy_metal_mass_frac_Z = he_z_tot
            
    elif composition_source == "asplund_2009":
        solar_values = pd.read_csv(PATH_TO_ASPLUND_2009, index_col=0)
        he_y_tot = ASPLUND_2009_HE_MASS_FRAC_Y
        he_z_tot = ASPLUND_2009_HEAVY_MASS_FRAC_Z
        if helium_mass_frac_Y == -99:
            helium_mass_frac_Y = he_y_tot
        if heavy_metal_mass_frac_Z == -99:
            heavy_metal_mass_frac_Z = he_z_tot
        
    else:
        raise ValueError(
            f"Unknown composition source: {composition_source}. Use 'asplund_2009' or 'asplund_2020'."
        )
    if final_atomic_number is not None:
        solar_values = solar_values[solar_values.index <= final_atomic_number]

    solar_values["mass_fractions"] = (
        atom_data.atom_data.mass.loc[solar_values.index.values]
        * 10**solar_values.Value.values
    ).values
    solar_values.drop(columns=["Element", "Value"], inplace=True)
    full_index = np.arange(solar_values.index.min(), solar_values.index.max() + 1)
    solar_values = solar_values.reindex(full_index, fill_value=0)

    # Scale Helium
    solar_values.loc[2] = (
        solar_values.loc[2]
        * helium_mass_frac_Y
        / he_y_tot
    )
    # Scale Metals
    solar_values.loc[3:] = (
        solar_values.loc[3:]
        * heavy_metal_mass_frac_Z
        / he_z_tot
    )

    # Return scaled mass fractions by dividing by total mass. Implicitly lowers the hydrogen abundance so that the total mass fraction is 1.
    return solar_values.div(solar_values.sum(axis=0))


def rescale_nuclide_mass_fractions(nuclide_mass_fraction, nuclides, scale_factors):
    """
    Renormalizes the nuclide_mass_fraction after multiplying the specified nuclides by a list of scale factors.

    Args:
    nuclide_mass_fraction: The mass_fraction object to rescale.
        nuclides: The nuclides to rescale by specified scale factors.
        scale_factors: How much to rescale the specified nuclides by before renormalizing.

    returns: The rescaled mass fractions.
    """

    new_mass_fractions = nuclide_mass_fraction.copy().T

    for nuclide, scale_factor in zip(nuclides, scale_factors):
        if not isinstance(nuclide, int):
            nuclide = element_symbol2atomic_number(nuclide)
        logging.info(
            f"Rescaling {atomic_number2element_symbol(nuclide)} by {scale_factor}"
        )
        if nuclide not in new_mass_fractions.columns:
            raise ValueError(f"{nuclide} not available in the simulation")

        new_mass_fractions[nuclide] = new_mass_fractions[nuclide] * scale_factor

    return new_mass_fractions.T.div(
        new_mass_fractions.T.sum(axis=0)
    )  # renormalize the composition after scaling
