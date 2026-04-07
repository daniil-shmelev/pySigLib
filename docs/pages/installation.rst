Installation
========================

.. tab-set::

   .. tab-item:: Windows

      pySigLib requires an installation of the MSVC compiler in order to compile the package.
      Please ensure this exists, then run:

      .. code-block:: console

          pip install pysiglib

      pySigLib will automatically detect CUDA, provided the ``CUDA_PATH`` environment variable is set correctly.
      To manually disable CUDA and build pySigLib for CPU only, set the ``CUSIG`` environment variable to ``0``:

      .. code-block:: console

          set CUSIG=0
          pip install pysiglib

   .. tab-item:: Linux

      pySigLib requires an installation of the GCC compiler in order to compile the package.
      Please ensure this exists, then run:

      .. code-block:: console

          pip install pysiglib

      pySigLib will automatically detect CUDA, provided the ``CUDA_PATH`` environment variable is set correctly.
      On most systems, this path will be ``/usr/lib/nvidia-cuda-toolkit`` and one can set it manually by running:

      .. code-block:: bash

          export CUDA_PATH=/usr/lib/nvidia-cuda-toolkit

      To manually disable CUDA and build pySigLib for CPU only, set the ``CUSIG`` environment variable to ``0``:

      .. code-block:: bash

          export CUSIG=0
          pip install pysiglib

   .. tab-item:: macOS

      pySigLib requires the Xcode Command Line Tools in order to compile the package.
      Please ensure these are installed (``xcode-select --install``), then run:

      .. code-block:: console

          pip install pysiglib

      pySigLib does not support CUDA on macOS, and will build without it when installed.

JAX Support
------------------------

pySigLib provides an optional JAX API with full support for ``jax.jit``, ``jax.grad``, and ``jax.vmap``.
To use it, install JAX before installing pySigLib:

.. code-block:: console

    pip install jax
    pip install pysiglib

pySigLib will automatically detect JAX and build the XLA FFI bindings. If JAX is not installed,
this is skipped and the rest of pySigLib works normally. To verify:

.. code-block:: python

    import pysiglib
    print(pysiglib.BUILT_WITH_JAX_FFI)  # True if JAX FFI was built

For GPU support with JAX, install the CUDA variant of JAX:

.. code-block:: console

    pip install jax[cuda12]

Build Options
------------------------

The following environment variables can be used to control the build:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Variable
     - Default
     - Description
   * - ``CUSIG``
     - ``ON``
     - Set to ``0`` to disable CUDA and build for CPU only.
   * - ``PYSIGLIB_JAX_FFI``
     - ``ON``
     - Set to ``0`` to disable JAX FFI support. When ``ON`` (default), JAX FFI
       is built automatically if JAX is installed, and skipped otherwise.
   * - ``SIGLIB_VEC``
     - ``ON``
     - Set to ``0`` to disable AVX vectorization.
   * - ``CUDA_ARCH``
     - ``native``
     - CUDA architectures to compile for. Accepts ``native`` (local GPU only),
       ``all`` (all architectures), ``all-major``, or a semicolon-separated list
       (e.g. ``"80;89;90"``). Use ``all`` when building portable wheels.

Editable Installs
------------------------

pySigLib supports editable installs for development:

.. code-block:: console

    pip install -e .
