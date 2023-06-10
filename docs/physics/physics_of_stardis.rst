******************
Physics of STARDIS
******************

================
Model and Plasma
================

STARDIS breaks down the stellar atmosphere into spherical shells as shown below and approximates that the plasma state is uniform throughout each shell. We rely on the MARCS code, a code that generates models of stellar atmospheres, to determine the temperatures, elemental abundances, and densities in each shell. The existing TARDIS plasma infrastructure determines the rest of the plasma state, namely the excitation and ionization properties and transition rates.

.. image:: media/model_and_plasma-1.png
   :width: 100 %
   :alt: Diagram showing inner and outer boundaries of the stellar atmosphere

=========
Opacities
=========

To determine an output spectrum, we need to understand how photons of light move through the atmosphere and what interactions they experience. Opacity is a measure of how likely it is that light will be scattered or absorbed by some material, like the stellar plasma, per unit distance it travels. This is contributed to by several mechanisms, which are described below.

---------------
Opacity Sources
---------------

The first four of these interactions are called continuum interactions, as they affect light at a large range of frequencies. Line interactions, on the other hand, only occur with light around specific frequencies, corresponding to the electron’s jump in energy. These are called resonant frequencies.

^^^^^^^^^^^^^^^^^^^^^
Bound-Free Absorbtion
^^^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha = \frac{64 \pi^4 e^{10} m_e Z^4}{3 \sqrt 3 h^6 c n_{\text{eff}}^5} n

^^^^^^^^^^^^^^^^^^^^
Free-Free Absorbtion
^^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha = \frac{4 e^6 Z^2 n}{3 h c v^3} \sqrt{\frac{2 \pi}{3 m_e^3 k_B T}}
   
^^^^^^^^^^^^^^^^^^^
Rayleigh Scattering
^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha = \sigma_T n \left ( c_4 \left ( \frac{v}{2 v_H} \right )^4 + c_6 \left ( \frac{v}{2 v_H} \right )^6 + c_8 \left ( \frac{v}{2 v_H} \right )^8 \right )


^^^^^^^^^^^^^^^^^^^
Electron Scattering
^^^^^^^^^^^^^^^^^^^

.. math::
   \alpha \sigma_T n_E
   

^^^^^^^^^^^^^^^^
Line Interaction
^^^^^^^^^^^^^^^^

.. math::
   \alpha = \frac{\pi e^2}{m_e c} n_l f_{lu} \left (1 - \frac{g_l n_u}{g_u n_l} \right ) \phi(v)

   
