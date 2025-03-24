Custom Star Model
=================

In the quickstart notebook you saw a STARDIS simulation using the file
``sun.mod`` to create spectra based on the sun. These ``.mod`` files are
how you supply a star model to STARDIS, and here we will show how to
obtain and use such file for a star of your chosen parameters.

Using MARCS to get .mod files
-----------------------------

STARDIS uses files from the `MARCS site <https://marcs.oreme.org/>`__,
which “contains about 52,000 stellar atmospheric models of spectral
types F, G and K”. Here we will only be discussing the aspects of MARCS
needed for STARDIS. To get started, go to the ‘Search Marcs’ tab of the
MARCS site `here! <https://marcs.oreme.org/data/>`__

.. container::

*picture of the Search Marcs page*

Getting an atmosphere model is as easy as entering a range of parameters
of your choosing. For more information on what each of these qualities
are, you can click the question marks next to each option or refer to
`the MARCS documentation <https://marcs.oreme.org/documents/>`__. In the
below example, we have entered some arbitrary values into MARCS, and on
the right are now all files that qualify. Note how the names of the
files show the exact parameters of each file and how there are several
pages of files you can look through.

.. container::

MARCS uses a ‘shopping cart’ style system for selecting models, called
your ‘basket’. Once you have selected your chosen models, proceed to the
‘Basket’ tab to download them. **Make sure you only have ‘.mod’ files
selected, as that is the format used by STARDIS.**

Below is an example of a yaml file using the ‘sun.mod’ file. Note that
you can provide either the relative or absolute path to your new model
data.

.. code:: yaml

   stardis_config_version: 1.0
   atom_data: kurucz_cd23_chianti_H_He.h5
   input_model:
       type: marcs
       fname: sun.mod # <----- this is where the .mod file is specified
       final_atomic_number: 30
   opacity:
   ...

gzipped files
-------------

when you download a mod file from MARCS, it will be gzipped (shown by
the file ending with ‘.mod.gz’). By default STARDIS will assume you have
extracted the file, however you can add a line to your YAML file
``gzipped: True`` as shown below and STARDIS will take care of this for
you.

.. code:: yaml

   ...
   input_model:
       type: marcs
       fname: sun.mod
       gzipped: True # <----- use this if the file is gzipped
       final_atomic_number: 30
   ...
