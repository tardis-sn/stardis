from stardis.radiation_field.opacities import Opacities


class RadiationField:
    """
    Class containing information about the radiation field.
    ###TODO Radiation field temperature should be a separate attribute, for the case of differing gas and radiation.

    Parameters
    ----------
    frequencies : astronopy.units.Quantity
    source_function : stardis.radiation_field.source_function
    opacities : stardis.radiation_field.opacities

    Attributes
    ----------
    frequencies : astropy.units.Quantity
        Frequencies of the radiation field.
    source_function : stardis.radiation_field.source_function
        Source function of the radiation field.
    opacities : star.dis.radiation_field.opacities
        Composition of the model. Includes density and atomic mass fractions.
    """

    def __init__(self, frequencies, source_function):
        self.frequencies = frequencies
        self.source_function = source_function
        self.opacities = Opacities(frequencies)
