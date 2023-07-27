class Composition:
    """
    Holds composition information

    Parameters
    ----------
    density : astropy.units.quantity.Quantity
        Density of the plasma
    atomic_mass_fraction : pandas.core.frame.DataFrame
        Mass fraction of each element



    """

    def __init__(self, density, atomic_mass_fraction):
        self.density = density
        self.atomic_mass_fraction = atomic_mass_fraction
