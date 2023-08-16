class Opacities:
    """
    Holds opacity information.

    ###TODO: Change to dict that holds each of the sources of opacity separately. Then have a method that combines them all to return the total opacity.
    Paramaters
    ----------
    opacities : dict
        Python dictionary to contain each of the sources of opacity by name as well as their contribution at each frequency specified in the radiation field.
    total_alphas : numpy.ndarray
        Array of the total opacity at each frequency specified in the radiation field at each depth points.
        Added as an attribute when calc_total_alphas() is called.
    """

    def __init__(self):
        self.opacities = {}

    ###TODO: Better implementation for this
    def calc_total_alphas(self):
        for i, item in enumerate(self.opacities.items()):
            if "gammas" not in item[0] and "doppler" not in item[0]:
                if i == 0:
                    self.total_alphas = item[1]
                else:
                    self.total_alphas += item[1]
        return self.total_alphas
