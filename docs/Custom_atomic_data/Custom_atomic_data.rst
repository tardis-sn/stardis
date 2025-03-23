Custom Atomic Data
==================

Using Your Own Atomic Data
--------------------------

To run a simulation, STARDIS requires an atomic data file that has
information on the properties of atoms and molecules needed by STARDIS.
While
`“kurucz_cd23_chianti_H_He.h5” <https://github.com/tardis-sn/tardis-regression-data/raw/main/atom_data/kurucz_cd23_chianti_H_He.h5>`__
is used in the STARDIS quickstart documentation, you can create and use
other atomic data files by using
`CARSUS <https://tardis-sn.github.io/carsus/>`__. For more information
on using CARSUS you can access the `CARSUS
documentation <https://tardis-sn.github.io/carsus/index.html>`__.

Relevant CARSUS Data to Collect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While CARSUS accumulates all sorts of data when compiling the atomic
data files, not everything it can include is necessary or relevant for
running STARDIS. Here is a list of what data/readers you should make
sure to include when running CARSUS:

-  `atomic weights & ionization energy from
   NIST <https://tardis-sn.github.io/carsus/io/nist.html>`__
-  `Robert Kurucz’s Atomic Linelist
   (GFALL) <https://tardis-sn.github.io/carsus/io/gfall.html>`__
-  `atomic and molecular transitions from
   VALD <https://tardis-sn.github.io/carsus/io/vald.html>`__
-  `Molecular formation from Barklem & Collet
   2016 <https://tardis-sn.github.io/carsus/io/barklem2016.html>`__

Using an Atomic Data File
~~~~~~~~~~~~~~~~~~~~~~~~~

Your atomic data file should have the file extension ``.h5`` . To use
your new atomic data file in a simulation, add/edit the line in your
YAML file ``atom_data: <path/to/atom_data>``. You can reference the
`STARDIS Configuration YAML structure
here <../quickstart/quickstart.ipynb#the-stardis-configuration>`__, but
you should set the atomic data file on the second line of your YAML
file, as shown below.

.. code:: yaml

   stardis_config_version: 1.0
   atom_data: <path/to/atomic_data>
   input_model:
   ...

Using VALD linelists
--------------------

Possibly one of the most important reasons you would want to use a
custom atomic data file would be to take advantage of a tailored VALD
linelist. You can refer to `VALD’s
documentation <https://www.astro.uu.se/valdwiki>`__ for information on
these lists if you are unfamiliar. To create a detailed and accurate
stellar spectrum, we highly recommend looking into using a VALD linelist
bundled into your atomic data.

If you have included a VALD linelist in your atomic data file, then you
must make the following change to your YAML file for the linelist to be
used:

.. code:: yaml

   ...
   line:
           disable: False
           broadening: [radiation, linear_stark, quadratic_stark, van_der_waals]
           vald_linelist:  #<-----
               use_linelist: True #<----- will default to false, so must set to True
               shortlist: <boolean> #<----- VALD can output lists in 'long' form or 'short' form; set accordingly
   no_of_thetas: 20
   ...
