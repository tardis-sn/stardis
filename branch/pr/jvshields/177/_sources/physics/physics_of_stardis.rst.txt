******************
Physics of STARDIS
******************

================
Model and Plasma
================

STARDIS breaks down the stellar atmosphere into spherical shells as shown below and approximates that the plasma state is uniform throughout each shell. We rely on the MARCS code, a code that generates models of stellar atmospheres, to determine the temperatures, elemental abundances, and densities in each shell. The existing TARDIS plasma infrastructure determines the rest of the plasma state, namely the excitation and ionization properties and transition rates.

.. image:: media/model_and_plasma-1.png
   :width: 500 px
   :alt: Diagram showing inner and outer boundaries of the stellar atmosphere

=========
Opacities
=========

To determine an output spectrum, we need to understand how photons of light move through the atmosphere and what interactions they experience. Opacity is a measure of how likely it is that light will be scattered or absorbed by some material, like the stellar plasma, per unit distance it travels. This is contributed to by several mechanisms, which are described :ref:`below<Opacity Sources>`.

---------------
Opacity Sources
---------------

The first four of these interactions are called continuum interactions, as they affect light at a large range of frequencies. Line interactions, on the other hand, only occur with light around specific frequencies, corresponding to the electron’s jump in energy. These are called *resonant frequencies*.

^^^^^^^^^^^^^^^^^^^^^
Bound-Free Absorbtion
^^^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha = \frac{64 \pi^4 e^{10} m_e Z^4}{3 \sqrt 3 h^6 c n_{\text{eff}}^5} n

.. image:: media/bound_free_absorbtion-1.png
   :width: 500 px
   :alt: Diagram of bound-free absorbtion
	   

^^^^^^^^^^^^^^^^^^^^
Free-Free Absorbtion
^^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha = \frac{4 e^6 Z^2 n}{3 h c v^3} \sqrt{\frac{2 \pi}{3 m_e^3 k_B T}}

.. image:: media/free_free_absorbtion-1.png
   :width: 500 px
   :alt: Diagram of free-free absorbtion


^^^^^^^^^^^^^^^^^^^
Rayleigh Scattering
^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha = \sigma_T n \left ( c_4 \left ( \frac{v}{2 v_H} \right )^4 + c_6 \left ( \frac{v}{2 v_H} \right )^6 + c_8 \left ( \frac{v}{2 v_H} \right )^8 \right )

.. image:: media/rayleigh_scattering-1.png
   :width: 500 px
   :alt: Diagram of Rayleigh scattering


^^^^^^^^^^^^^^^^^^^
Electron Scattering
^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha = \sigma_T n_E

.. image:: media/electron_scattering-1.png
   :width: 500 px
   :alt: Diagram of electron scattering


^^^^^^^^^^^^^^^^
Line Interaction
^^^^^^^^^^^^^^^^

.. math::
   \alpha = \frac{\pi e^2}{m_e c} n_l f_{lu} \left (1 - \frac{g_l n_u}{g_u n_l} \right ) \phi(v)

.. image:: media/line_interaction-1.png
   :width: 500 px
   :alt: Diagram of line interaction
   
----------
Broadening
----------

Line interaction opacity does not occur only at the exact resonant frequencies; lines are broadened to reach other nearby frequencies. Thus, the line interaction opacity is the total line opacity

.. math::
   \alpha_{lu} = \frac{\pi e^2}{m_e c} n_l f_{lu} \left ( 1 - \frac{g_l n_u}{g_u n_l} \right )

times the *line profile* :math:`\phi(v)` which describes the broadening.

..
   The below was taken from https://stackoverflow.com/a/42522042

|wide_line_profile| vs. |narrow_line_profile|

.. |wide_line_profile| image:: media/wide_line_profile.png
   :width: 45 %
   :alt: Broad line profile

.. |narrow_line_profile| image:: media/narrow_line_profile.png
   :width: 45 %
   :alt: Less broad line profile

Above are examples of line profiles, the left being very broadened and the right being less broadened.
	 
The line profile uses the following parameters for determining how much the line is broadened:

- The Einstein coefficient :math:`A_{ul}`, describing the line’s natural acceptance of non-resonant frequencies.
- The doppler width :math:`\Delta v_D`, the range of frequencies that are doppler shifted to be the resonant frequency due to the movement of ions in the plasma.
- The collisional broadening parameter :math:`\gamma_{\text{col}}`, describing the effects of forces between ions or ions and electrons which shift the resonant frequency.

The line profile centered at the resonant frequency :math:`v_{lu}` is then:

.. math::
   \phi(v) = \frac{\gamma_{\text{col}}}{4 \pi^{\frac{5}{2}} \Delta v_D} \int_{-\infty}^{\infty} \frac{\exp \left ( -\frac{y^2}{\Delta v_D^2} \right )}{\left (v - v_{lu} - y \right )^2 + \left ( \frac{\gamma_{\text{col}} + A_{ul}}{4 \pi} \right )^2} \, d y

=========
Transport
=========

Finally, we use the opacity information to trace beams of light coming from the photosphere at different angles and frequencies to find the final intensity. We use the equation

.. math::
   I_{N + 1}(v, \theta) = I_N(v, \theta) e^{-\tau} + (1 - e^{- \tau}) B_{N + 1} (v) + (1 - e^{-\tau} - \tau e^{-\tau}) \frac{\Delta B_{N + 1}(v)}{\tau}

where :math:`\tau = \frac{\alpha l}{\cos \theta}` is the *optical depth*, :math:`l` is the depth of each shell, and :math:`B(v)` is the blackbody distribution.

.. image:: media/transport.png
   :width: 500 px
   :alt: A diagram of how the opacity is a function of the angle and frequency of a location in the photosphere


The flux density (the desired spectrum) is then:

.. math::
   F(v) = 2 \pi \int_0^{\frac{\pi}{2}} I(v, \theta) \sin \theta \cos \theta \, d \theta.

