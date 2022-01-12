import numpy as np
import pandas as pd

from astropy import constants as const

from tardis.plasma.properties.base import DataFrameInput, ProcessingPlasmaProperty

class AlphaLine(ProcessingPlasmaProperty):
    """
    Attributes
    ----------
    alpha_line : Pandas DataFrame, dtype float
          Sobolev optical depth for each line. Indexed by line.
          Columns as zones.
    """

    outputs = ("alpha_line",)
    latex_name = (r"\tau_{\textrm{line}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2}}{m_{e} c}f_{lu}\
        n_{lower} \Big(1-\dfrac{g_{lower}n_{upper}}{g_{upper}n_{lower}}\Big)",
    )

    def __init__(self, plasma_parent):
        super(AlphaLine, self).__init__(plasma_parent)
        self.alpha_coefficient  = (np.pi * const.e.gauss**2) / (const.m_e.cgs * const.c.cgs)

    def calculate(
        self,
        lines,
        level_number_density,
        lines_lower_level_index,
        stimulated_emission_factor,
        f_lu,
    ):
        f_lu = f_lu.values[np.newaxis].T
        #wavelength = wavelength_cm.values[np.newaxis].T
        
        n_lower = level_number_density.values.take(
            lines_lower_level_index, axis=0, mode="raise"
        )
        alpha =  self.alpha_coefficient * n_lower * stimulated_emission_factor

        if np.any(np.isnan(alpha)) or np.any(
            np.isinf(np.abs(alpha))
        ):
            raise ValueError(
                "Some alpha_line are nan, inf, -inf "
                " Something went wrong!"
            )

        return pd.DataFrame(
            alpha,
            index=lines.index,
            columns=np.array(level_number_density.columns),
        )