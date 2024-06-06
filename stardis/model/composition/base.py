from tardis.util.base import element_symbol2atomic_number


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

    def rescale_elements(self, elements, scale_factors):
        """
        Renormalizes the composition after multiplying the specified element by the scale factor.

        Args:
            element: The elements to multiply.
            scale_factor: How much to rescale the elements by.

        Returns:
            Composition: The rescaled composition object.
        """

        # If elements and scale_factors are not lists, make them lists
        if not isinstance(elements, list):
            elements = [elements]
        if not isinstance(scale_factors, list):
            scale_factors = [scale_factors]

        new_atomic_mass_fraction = self.atomic_mass_fraction.copy()

        for element, scale_factor in zip(elements, scale_factors):
            if not isinstance(element, int):
                element = element_symbol2atomic_number(element)
            new_atomic_mass_fraction.loc[element] = (
                new_atomic_mass_fraction.loc[element] * scale_factor
            )

        self.atomic_mass_fraction = new_atomic_mass_fraction.div(
            new_atomic_mass_fraction.sum(axis=0)
        )
