class Radial1DGeometry:

    """
    Holds information about model geometry for radial 1D models.

    Parameters
    ----------
    r : astropy.units.quantity.Quantity
    ----------

    Attributes
    ----------
    cell_length : astropy.units.quantity.Quantity
        Length in each shell
    """

    def __init__(self, r):
        self.r = r

    @property
    def cell_length(self):
        # NEED TO CHANGE THE NAME!!
        return self.r[1:] - self.r[:-1]
