import numpy as np
import pandas as pd

from astropy import constants as const, units as u

from tardis.plasma.properties.base import DataFrameInput, ProcessingPlasmaProperty, ArrayInput


ALPHA_COEFFICIENT  = (np.pi * const.e.gauss**2) / (const.m_e.cgs * const.c.cgs)
THERMAL_DE_BROGLIE_CONST = const.h**2 / (2 * np.pi * const.m_e * const.k_B)
H_MINUS_CHI = 0.754195 * u.eV # see https://en.wikipedia.org/wiki/Hydrogen_anion
class AlphaLine(ProcessingPlasmaProperty):
    """
    Attributes
    ----------
    alpha_line : Pandas DataFrame, dtype float
          Sobolev optical depth for each line. Indexed by line.
          Columns as zones.
    """

    outputs = ("alpha_line",)
    latex_name = (r"\alpha_{\textrm{line}}",)
    latex_formula = (
        r"\dfrac{\pi e^{2}}{m_{e} c}f_{lu}\
        n_{lower} \Big(1-\dfrac{g_{lower}n_{upper}}{g_{upper}n_{lower}}\Big)",
    )

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
        alpha =  ALPHA_COEFFICIENT * n_lower * stimulated_emission_factor

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

class CellLength(ArrayInput):
    outputs = ('cell_length',)

class HMinusOpacityWBR(ProcessingPlasmaProperty):
    outputs = ('tau_h_minus')
    def __init__(self, plasma_parent, wbr_opacity_df):
        super(HMinusOpacityWBR, self).__init__(plasma_parent)
        self.wbr_opacity_df = wbr_opacity_df

    def calculate(self, h_minus_density, tracing_nus, cell_length):
        tracing_wavelength = (tracing_nus / const.c)
        h_minus_sigma_nu = np.interp(tracing_wavelength, 
                        self.wbr_opacity_df.wavelength,
                        self.wbr_opacity_df.cross_section)
        tau_h_minus = h_minus_sigma_nu * (h_minus_density * cell_length)[None].T
        return tau_h_minus


class HMinusDensity(ProcessingPlasmaProperty):
    outputs = ('h_minus_density',)
    def calculate(self, ion_number_density, t_rad, electron_densities):
        t_rad = t_rad * u.K
        h_neutral_density = ion_number_density.loc[1,0]
        thermal_de_broglie = ((THERMAL_DE_BROGLIE_CONST / t_rad) ** (3/2)).to(u.cm**3)
        phi = (thermal_de_broglie / 4) * np.exp(H_MINUS_CHI / (const.k_B * t_rad))
        return h_neutral_density * electron_densities * phi.value

class TracingNus(ArrayInput):
    outputs = ('tracing_nus',)
    pass
