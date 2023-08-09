class Radial1DGeometry:

    """
    Holds information about model geometry (distribution of depth points) for radial 1D models.

    Parameters
    ----------
    r : astropy.units.quantity.Quantity
        The spacial coordinate of the depth point
    ----------

    Attributes
    ----------
    dist_to_next_depth_point : astropy.units.quantity.Quantity
        distance to the next depth point
    """

    def __init__(self, r):
        self.r = r

    @property
    def dist_to_next_depth_point(self):
        # Can't be a quantity for njitting purposes
        return (self.r[1:] - self.r[:-1]).value
