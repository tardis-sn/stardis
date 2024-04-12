class Composition:
    """
    Holds composition information

    Parameters
    ----------
    density : astropy.units.quantity.Quantity
        Density of the plasma at each depth point
    atomic_mass_fraction : pandas.core.frame.DataFrame
        Mass fraction of each element at each depth point



    """

    def __init__(self, density, atomic_mass_fraction):
        self.density = density
        self.atomic_mass_fraction = atomic_mass_fraction

    def rescale_from_element_fraction(self, element, scale_factor):
        """
        Renormalizes the composition after multiplying the specified element by the scale factor.

        Args:
            element: The element to multiply.
            scale_factor: How much to rescale the element by.

        Returns:
            Composition: The rescaled composition object.
        """
        new_atomic_mass_fraction = self.atomic_mass_fraction.copy()
        new_atomic_mass_fraction.loc[element] = (
            new_atomic_mass_fraction.loc[element] * scale_factor
        )
        self.atomic_mass_fraction = new_atomic_mass_fraction.div(
            new_atomic_mass_fraction.sum(axis=0)
        )
