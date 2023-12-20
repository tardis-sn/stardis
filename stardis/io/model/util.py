from pathlib import Path
import pandas as pd


PATH_TO_ASPLUND_2009 = Path(__file__).parent / "data" / "asplund_2009_processed.csv"
ASPLUND_DEFAULT_HELIUM_MASS_FRACTION_Y = (
    0.2492280  # The Asplund 2009 mass fraction of measured He
)
ASPLUND_DEFAULT_HEAVY_ELEMENTS_MASS_FRACTION_Z = (
    0.01337  # The Asplund 2009 mass fraction of measured heavy metals
)


def create_scaled_solar_profile(
    atom_data,
    Y=ASPLUND_DEFAULT_HELIUM_MASS_FRACTION_Y,
    Z=ASPLUND_DEFAULT_HEAVY_ELEMENTS_MASS_FRACTION_Z,
    final_atomic_number=None,
):
    """
    Scales the solar mass fractions based on the given atom data, Y, and Z, using the photospheric composition from Asplund 2009.
    Default Y and Z are calculated using Asplund 2009 and NIST atomic weights, i.e., if you use their default values you will get
    back the solar composition measured by Asplund 2009.

    Args:
        atom_data: The atom data used to scale the solar mass fractions.
        Y: The helium abundance. Default is 0.2492280.
        Z: The metallicity. Default is 0.01337.

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
        solar_values.loc[2] * Y / ASPLUND_DEFAULT_HELIUM_MASS_FRACTION_Y
    )
    # Scale Metals
    solar_values.loc[3:] = (
        solar_values.loc[3:] * Z / ASPLUND_DEFAULT_HEAVY_ELEMENTS_MASS_FRACTION_Z
    )

    # Return scaled mass fractions by dividing by total mass. Implicitly lowers the hydrogen abundance so that the total mass fraction is 1.
    return solar_values.div(solar_values.sum(axis=0))
