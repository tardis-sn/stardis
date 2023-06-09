******************
Physics of STARDIS
******************

================
Model and Plasma
================

STARDIS breaks down the stellar atmosphere into spherical shells as shown below and approximates that the plasma state is uniform throughout each shell. We rely on the MARCS code, a code that generates models of stellar atmospheres, to determine the temperatures, elemental abundances, and densities in each shell. The existing TARDIS plasma infrastructure determines the rest of the plasma state, namely the excitation and ionization properties and transition rates.

-- figure:: media/model_and_plasma-1.png
   :scale: 100 %
   :alt: Diagram showing inner and outer boundaries of the stellar atmosphere
