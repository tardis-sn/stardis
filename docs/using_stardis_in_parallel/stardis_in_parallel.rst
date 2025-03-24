Using STARDIS in Parallel
=========================

When you run a STARDIS simulation, the yaml file determines how many
threads are used. By default, it will only use 1 thread. The more
threads you set a simulation to use, the more computing power the
simulation will take advantage of for calculating line opacity and ray
tracing steps (as both of these are parallelized in STARDIS). **Setting
n_threads to 0 will make the simulation use all available threads.**

Here is a what a YAML configuration file that has the STARDIS simulation
use 3 threads could look like

.. code:: yaml

   stardis_config_version: 1.0
   n_threads: 3 # <-----------  add your 'n_threads: <integer>' here
   atom_data: kurucz_cd23_chianti_H_He.h5
   input_model:
       ...

One reason you may want to consider increasing threads used is when
computing spectra with large numbers (i.e. 10,000 or more) of wavelength
points.
