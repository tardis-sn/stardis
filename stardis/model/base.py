from tardis.io.util import HDFWriterMixin


class StellarModel(HDFWriterMixin):
    """
    Class containing information about the stellar model.

    Parameters
    ----------
    temperatures : numpy.ndarray
    geometry : stardis.model.geometry
    composition : stardis.model.composition.base.Composition

    Attributes
    ----------
    temperatures : numpy.ndarray
        Temperatures in K of all depth points. Note that array is transposed.
    geometry : stardis.model.geometry
        Geometry of the model.
    composition : stardis.model.composition.base.Composition
        Composition of the model. Includes density and atomic mass fractions.
    no_of_depth_points : int
        Class attribute to be easily accessible for initializing arrays that need to match the shape of the model.
    spherical : bool
        Flag for spherical geometry.
    microturbulence : float
        Microturbulence in km/s.
    """

    hdf_properties = ["temperatures", "geometry", "composition"]

    def __init__(
        self, temperatures, geometry, composition, spherical=False, microturbulence=0.0
    ):
        self.temperatures = temperatures
        self.geometry = geometry
        self.composition = composition
        self.spherical = spherical
        self.microturbulence = microturbulence

    @property
    def no_of_depth_points(self):
        return self.temperatures.shape[0]
