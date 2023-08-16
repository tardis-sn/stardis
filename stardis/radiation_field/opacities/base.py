class Opacities:
    """
    Holds opacity information.

    ###TODO: Change to dict that holds each of the sources of opacity separately. Then have a method that combines them all to return the total opacity.
    Paramaters
    ----------
    alphas : numpy.ndarray
        Array of shape (no_of_depth_points, no_of_frequencies). Total opacity at
        each depth point for each frequency in tracing_nus.

    """

    def __init__(self):
        self.opacities = {}
        self.total_alphas = None

    ###TODO: Better implementation for this
    def calc_total_alphas(self):
        for i, item in enumerate(self.opacities.items()):
            if "gammas" not in item[0] and "doppler" not in item[0]:
                if i == 0:
                    self.total_alphas = item[1]
                else:
                    self.total_alphas += item[1]
        return self.total_alphas
