from stardis.radiation_field.opacities import Opacities
import numpy as np


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
    opacities : stardis.radiation_field.opacities
        Opacities object. Contains a dictionary of opacities contributed from different sources and the calc_total_alphas() method to
        calculate the total opacity at each frequency at each depth point.
    F_nu : numpy.ndarray
        Radiation field fluxes at each frequency at each depth point. Initialized as zeros and calculated by a solver.
    """

    def __init__(self, frequencies, source_function, stellar_model):
        self.frequencies = frequencies
        self.source_function = source_function
        self.opacities = Opacities(frequencies, stellar_model)
        self.F_nu = np.zeros((stellar_model.no_of_depth_points, len(frequencies)))
