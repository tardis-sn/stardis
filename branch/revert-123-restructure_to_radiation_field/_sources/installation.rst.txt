..
   This file was converted from MarkDown using pandoc 2.19.2, Compiled with pandoc-types 1.22.2.1, texmath 0.12.5.2, skylighting 0.13, citeproc 0.8.0.1, ipynb 0.2, hslua 2.2.1, Scripting engine: Lua 5.4
   The command was `$ pandoc -t rst installation.md -o installation.rst`__

Downloading and Installation
============================

Setting Up the Environment
--------------------------

.. note::
   * STARDIS is only supported on macOS and GNU/Linux. Windows users can run STARDIS on a virtual machine.

   * STARDIS packages and dependencies are distributed only through the `conda <https://docs.conda.io/en/latest/>`__ package management system, therefore installation requires a conda distribution to be installed on your system. STARDIS uses `Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`__ or `Mamba <https://mamba.readthedocs.io/en/latest/installation.html>`__ by default. Other distributions are untested.

STARDIS uses exclusively the packages in the TARDIS enviroment, as well
as using the TARDIS code itself. However, since STARDIS can be sensitive
to changes in TARDIS, we strongly suggest that users create a separate
environment for STARDIS that pins the TARDIS version. To do this, run
the following in the terminal (replacing ``{platform}`` with
``linux-64``, ``linux-64-cuda``, or ``osx-64`` as applicable). 

.. code-block:: bash

   $ cd <path-to-stardis-directory>
   $ conda create --name stardis --file conda-{platform}.lock
   $ conda activate stardis
   $ pip install git+https://github.com/tardis-sn/tardis.git@release-2023.04.16

The third command (``conda activate stardis``) activates the
environment, which is necessary to correctly install STARDIS using the directions below.

If you are using Mamba, the steps are similar:

.. code-block:: bash

   $ cd <path-to-stardis-directory>
   $ mamba create --name stardis --file conda-{platform}.lock
   $ mamba activate stardis
   $ pip install git+https://github.com/tardis-sn/tardis.git@release-2023.04.16
   
Downloading and Installing STARDIS
----------------------------------

.. note::
   Both of the below instructions should be run with your ``stardis`` virtual environment activated. That is, you should run ``$ conda activate stardis`` or ``$ mamba activate stardis`` if you are using conda or mamba, respectively, in the terminal before you continue with the directions below. 

For Non-Developers
^^^^^^^^^^^^^^^^^^

STARDIS can be downloaded for **non-developers** by running

::
   
   $ pip install git+https://github.com/tardis-sn/stardis.git@main

in the terminal.

For Developers
^^^^^^^^^^^^^^

**Developers** should fork and clone the STARDIS repository.
First, `fork the
repository <https://github.com/tardis-sn/stardis/fork>`__ and `configure
GitHub to work with SSH
keys <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`__,
and then run the following in the terminal:

::

   $ git clone git@github.com:<username>/stardis.git
   $ cd stardis
   $ git remote add upstream git@github.com:tardis-sn/stardis.git
   $ git fetch upstream
   $ git checkout upstream/main
   $ pip install -e .[test,docs]
