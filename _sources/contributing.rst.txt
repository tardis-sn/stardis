Contributing
============

Generating Conda Lockfiles
--------------------------

In the interest of reproducibility, STARDIS uses `conda-lock <https://conda.github.io/conda-lock/>`__ files to keep the virtual environment consistent across similar machines and installations. High level specifications of the environments are given in ``stardis_env3.yml`` and ``stardis_env3_cuda.yml``, and then ``conda-lock`` is used to create the corresponding lockfile. The steps are as follow.

.. note::
   These steps are only tested using `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__.

.. note::
   Both sets of steps below assume that all commands are being run from your local STARDIS git repository, which can be obtained using the developer installation instructions.

Without CUDA Support
^^^^^^^^^^^^^^^^^^^^

#. Activate the Conda virtual environment:

   .. code-block:: bash
		   
      $ conda activate stardis

#. `Create a lockfile <https://conda.github.io/conda-lock/cli/gen/#conda-lock-lock>`__ using conda-lock's `unified lockfile format <https://conda.github.io/conda-lock/output/#unified-lockfile>`__:

   .. code-block:: bash
   
      $ conda-lock -f stardis_env3.yml --conda $(which conda) --strip-auth
   
#. `Render <https://conda.github.io/conda-lock/cli/gen/#conda-lock-render>`__ platform-specific lockfiles:

   .. code-block:: bash
      
      $ conda-lock render conda-lock.yml

With CUDA Support
^^^^^^^^^^^^^^^^^^^^

#. Activate the Conda virtual environment:

   .. code-block:: bash
		   
      $ conda activate stardis-cuda

#. `Create a lockfile <https://conda.github.io/conda-lock/cli/gen/#conda-lock-lock>`__ using conda-lock's `unified lockfile format <https://conda.github.io/conda-lock/output/#unified-lockfile>`__:

   .. code-block:: bash
   
      $ conda-lock -f stardis_env3_cuda.yml --conda $(which conda) --strip-auth --lockfile conda-lock-cuda.yml
   
#. `Render <https://conda.github.io/conda-lock/cli/gen/#conda-lock-render>`__ platform-specific lockfiles:

   .. code-block:: bash
      
      $ conda-lock render --filename-template conda-{platform}-cuda.lock conda-lock-cuda.yml
