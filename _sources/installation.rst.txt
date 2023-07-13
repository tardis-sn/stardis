..
   This file was converted from MarkDown using pandoc 2.19.2, Compiled with pandoc-types 1.22.2.1, texmath 0.12.5.2, skylighting 0.13, citeproc 0.8.0.1, ipynb 0.2, hslua 2.2.1, Scripting engine: Lua 5.4
   The command was `$ pandoc -t rst installation.md -o installation.rst`__

Downloading and Installation
============================

Setting Up the Environment
--------------------------

.. note::
   * STARDIS is only supported on macOS and GNU/Linux. Windows users can run STARDIS on a virtual machine.

   * STARDIS packages and dependencies are distributed only through the `conda <https://docs.conda.io/en/latest/>`__ package management system, therefore installation requires a conda distribution to be installed on your system. STARDIS uses `Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`__ by default. Other distributions are untested.

STARDIS uses exclusively the packages in the TARDIS enviroment, as well
as using the TARDIS code itself. However, since STARDIS can be sensitive
to changes in TARDIS, we strongly suggest that users create a separate
environment for STARDIS that pins the TARDIS version. To do this, run
the following in the terminal (replacing ``{platform}`` with either
``linux`` or ``osx`` as applicable):

::

   $ wget -q https://github.com/tardis-sn/tardis/releases/latest/download/conda-{platform}-64.lock
   $ conda create --name stardis --file conda-{platform}-64.lock
   $ conda activate stardis
   $ pip install git+https://github.com/tardis-sn/tardis.git@release-2023.04.16

The third command (``conda activate stardis``) activates the
environment, which is necessary to install the pinned version of TARDIS
to your STARDIS environment.

Downloading STARDIS
-------------------

STARDIS can be downloaded by **non-developers** by running
``$ pip install git+https://github.com/tardis-sn/stardis.git@main`` in
the terminal while in your STARDIS environment.

**Developers** must instead clone and fork the STARDIS repository.
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
   $ python setup.py develop
