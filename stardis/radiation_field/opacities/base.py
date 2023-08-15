import numpy as np


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

    def __init__(self, frequencies):
        self.total_alphas = np.zeros_like(frequencies)
        self.opacities = {}

    def get_total_opacities(self):
        pass
