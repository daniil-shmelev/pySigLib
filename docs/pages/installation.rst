Installation
========================

Install from PyPI
------------------------

The recommended way to install pySigLib is from PyPI using pre-built wheels.
No compiler toolchain is required — wheels are published for Windows, Linux,
and macOS (arm64).

.. code-block:: console

    pip install pysiglib              # CPU only
    pip install pysiglib[cuda]        # with CUDA GPU support

The ``[cuda]`` extra installs the companion ``pysiglib-cuda`` plugin, which
ships the CUDA binaries (``cusig``).

JAX support
++++++++++++++++++++++++++++++

The XLA FFI bindings for JAX are already built into every pySigLib wheel, so
there is no pySigLib-side extra to install. Just install JAX separately:

.. code-block:: console

    pip install jax

For GPU-accelerated JAX, use its CUDA variant:

.. code-block:: console

    pip install pysiglib[cuda] jax[cuda12]

To verify the installation:

.. code-block:: python

    import pysiglib
    print(pysiglib.__version__)
    print(pysiglib.BUILT_WITH_CUDA)    # True if CUDA backend loaded
    print(pysiglib.BUILT_WITH_JAX_FFI) # True if JAX FFI available

Install from source
------------------------

If you need a custom build (unsupported platform, alternative CUDA version,
development work), pySigLib can be built from source. This requires a C++
compiler toolchain.

.. tab-set::

   .. tab-item:: Windows

      Requires MSVC. Once installed, run:

      .. code-block:: console

          pip install pysiglib --no-binary pysiglib

      pySigLib will automatically detect CUDA, provided the ``CUDA_PATH`` environment variable is set correctly.
      To manually disable CUDA and build pySigLib for CPU only, set the ``CUSIG`` environment variable to ``0``:

      .. code-block:: console

          set CUSIG=0
          pip install pysiglib --no-binary pysiglib

   .. tab-item:: Linux

      Requires GCC. Once installed, run:

      .. code-block:: console

          pip install pysiglib --no-binary pysiglib

      pySigLib will automatically detect CUDA, provided the ``CUDA_PATH`` environment variable is set correctly.
      On most systems, this path will be ``/usr/lib/nvidia-cuda-toolkit`` and one can set it manually by running:

      .. code-block:: bash

          export CUDA_PATH=/usr/lib/nvidia-cuda-toolkit

      To manually disable CUDA and build pySigLib for CPU only, set the ``CUSIG`` environment variable to ``0``:

      .. code-block:: bash

          export CUSIG=0
          pip install pysiglib --no-binary pysiglib

   .. tab-item:: macOS

      Requires the Xcode Command Line Tools (``xcode-select --install``). Once installed, run:

      .. code-block:: console

          pip install pysiglib --no-binary pysiglib

      pySigLib does not support CUDA on macOS, and will build without it when installed.

JAX support (source builds)
++++++++++++++++++++++++++++++

When building from source, pySigLib automatically detects JAX and builds the
XLA FFI bindings if JAX is installed. Requires **jaxlib >= 0.5.0** (Python
3.10+):

.. code-block:: console

    pip install jax
    pip install pysiglib --no-binary pysiglib

If JAX is not installed at build time, the FFI bindings are skipped and the
rest of pySigLib works normally. To verify:

.. code-block:: python

    import pysiglib
    print(pysiglib.BUILT_WITH_JAX_FFI)

Build options
++++++++++++++++++++++++++++++

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

Editable installs
++++++++++++++++++++++++++++++

pySigLib supports editable installs for development:

.. code-block:: console

    pip install -e .
