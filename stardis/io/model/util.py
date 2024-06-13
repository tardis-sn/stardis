from pathlib import Path
import pandas as pd
import logging
from tardis.util.base import element_symbol2atomic_number, atomic_number2element_symbol


PATH_TO_ASPLUND_2009 = Path(__file__).parent / "data" / "asplund_2009_processed.csv"
ASPLUND_DEFAULT_HELIUM_MASS_FRACTION_Y = (
    0.2492280  # The Asplund 2009 mass fraction of measured He
)
ASPLUND_DEFAULT_HEAVY_ELEMENTS_MASS_FRACTION_Z = (
    0.01337  # The Asplund 2009 mass fraction of measured heavy metals
)


def create_scaled_solar_profile(
    atom_data,
    helium_mass_frac_Y=ASPLUND_DEFAULT_HELIUM_MASS_FRACTION_Y,
    heavy_metal_mass_frac_Z=ASPLUND_DEFAULT_HEAVY_ELEMENTS_MASS_FRACTION_Z,
    final_atomic_number=None,
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
    solar_values = pd.read_csv(PATH_TO_ASPLUND_2009, index_col=0)
    if final_atomic_number is not None:
        solar_values = solar_values[solar_values.index <= final_atomic_number]

    solar_values["mass_fractions"] = (
        atom_data.atom_data.mass.loc[solar_values.index.values]
        * 10**solar_values.Value.values
    ).values
    solar_values.drop(columns=["Element", "Value"], inplace=True)

    # Scale Helium
    solar_values.loc[2] = (
        solar_values.loc[2]
        * helium_mass_frac_Y
        / ASPLUND_DEFAULT_HELIUM_MASS_FRACTION_Y
    )
    # Scale Metals
    solar_values.loc[3:] = (
        solar_values.loc[3:]
        * heavy_metal_mass_frac_Z
        / ASPLUND_DEFAULT_HEAVY_ELEMENTS_MASS_FRACTION_Z
    )

    # Return scaled mass fractions by dividing by total mass. Implicitly lowers the hydrogen abundance so that the total mass fraction is 1.
    return solar_values.div(solar_values.sum(axis=0))


def rescale_chemical_mass_fractions(composition, elements, scale_factors):
    """
    Renormalizes the composition after multiplying the specified elements by a list of scale factors.

    Args:
    composition: The composition object to rescale.
        nuclides: The nuclides to rescale by specified scale factors.
        scale_factors: How much to rescale the specified nuclides by before renormalizing.
    """

    new_mass_fractions = composition.nuclide_mass_fraction.copy().T

    for element, scale_factor in zip(elements, scale_factors):
        if not isinstance(element, int):
            element = element_symbol2atomic_number(element)
        logging.info(
            f"Rescaling {atomic_number2element_symbol(element)} by {scale_factor}"
        )

        new_mass_fractions[element] = new_mass_fractions[element] * scale_factor

    composition.nuclide_mass_fraction = new_mass_fractions.T.div(
        new_mass_fractions.T.sum(axis=0)
    )  # renormalize the composition after scaling
