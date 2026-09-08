Installation
========================

.. meta::
   :description: Install pySigLib for fast path signatures, log signatures, and signature kernels on CPU and CUDA GPU.

Install from PyPI
------------------------

The recommended way to install pySigLib is from PyPI using pre-built wheels.
No compiler toolchain is required - wheels are published for Windows, Linux,
and macOS (arm64).

.. code-block:: console

    pip install pysiglib                 # NumPy, CPU only
    pip install "pysiglib[torch]"        # PyTorch API
    pip install "pysiglib[jax]"          # JAX API (Python 3.11+)
    pip install "pysiglib[torch,jax]"    # Both frameworks
    pip install "pysiglib[torch,cuda]"   # PyTorch and native CUDA support

The ``[cuda]`` extra installs the companion ``pysiglib-cuda`` plugin, which
ships the CUDA shared library (``cusig``) and the CUDA JAX FFI bindings as a
sibling package. ``pysiglib`` discovers it at import time; if the plugin is
absent, ``pysiglib`` runs CPU-only.

JAX support
++++++++++++++++++++++++++++++

The base NumPy API imports neither Torch nor JAX. The ``[jax]`` extra installs a compatible JAX version; the CPU XLA FFI bindings are included in the wheel. Select the API with ``import pysiglib.jax_api as pysiglib``:

.. code-block:: console

    pip install "pysiglib[jax]"

For GPU-accelerated JAX, use its CUDA variant:

.. code-block:: console

    pip install "pysiglib[jax,cuda]" "jax[cuda12]"

To verify the installation:

.. code-block:: python

    import pysiglib
    print(pysiglib.__version__)
    print(pysiglib.BUILT_WITH_CUDA)    # True if CUDA backend loaded
    print(pysiglib.BUILT_WITH_JAX_FFI) # True if JAX FFI available

Install from source
------------------------

If you need a custom build (unsupported platform, alternative CUDA version,
development work), pySigLib can be built from source. ``pysiglib`` and the
``pysiglib-cuda`` plugin are separate packages with separate builds: install
``pysiglib`` for the CPU core, and additionally install ``pysiglib-cuda``
from the ``plugins/cuda`` subdirectory of a repo checkout for the CUDA
backend.

Base package (CPU only)
+++++++++++++++++++++++++++++++

This requires a C++ compiler toolchain. The base wheel is CPU-only - it does
not build CUDA. Use the ``pysiglib-cuda`` plugin for that (next section).

.. tab-set::

   .. tab-item:: Windows

      Requires MSVC. Once installed, run:

      .. code-block:: console

          pip install pysiglib --no-binary pysiglib

   .. tab-item:: Linux

      Requires GCC >= 10. Once installed, run:

      .. code-block:: console

          pip install pysiglib --no-binary pysiglib

   .. tab-item:: macOS

      Requires the Xcode Command Line Tools (``xcode-select --install``). Once
      installed, run:

      .. code-block:: console

          pip install pysiglib --no-binary pysiglib

CUDA plugin
+++++++++++++++++++++++++++++++

The ``pysiglib-cuda`` plugin is published as wheels only - there is no sdist
on PyPI, because it depends on sources from the parent repository
(``siglib/cusig``). To build the plugin from source, clone the repository
and install the ``plugins/cuda`` subdirectory:

.. code-block:: console

    git clone https://github.com/daniil-shmelev/pySigLib.git
    cd pySigLib
    pip install pysiglib --no-binary pysiglib
    pip install ./plugins/cuda

This requires a working CUDA toolkit (``nvcc`` on ``PATH`` or
``CUDAToolkit_ROOT`` set to its install prefix; on Linux this is typically
``/usr/local/cuda``, and on Windows the ``CUDA_PATH`` environment variable
set by the NVIDIA installer is also picked up).

By default the plugin compiles only for the local GPU's architecture
(``CUDA_ARCH=native``). To target multiple architectures, set ``CUDA_ARCH``
to a semicolon-separated list, ``all-major``, or ``all`` before installing:

.. code-block:: console

    CUDA_ARCH="80;89;90" pip install ./plugins/cuda

JAX support (source builds)
++++++++++++++++++++++++++++++

When building from source, both packages automatically detect JAX and build
their respective XLA FFI bindings if JAX is installed. Requires
**jaxlib >= 0.9.1** (Python 3.11+):

.. code-block:: console

    pip install "pysiglib[jax]"
    pip install pysiglib --no-binary pysiglib
    pip install ./plugins/cuda    # only if you need CUDA

If JAX is not installed at build time, the FFI bindings are skipped and the
rest of pySigLib works normally. To verify:

.. code-block:: python

    import pysiglib
    print(pysiglib.BUILT_WITH_JAX_FFI)

Build options
++++++++++++++++++++++++++++++

The following environment variables control the source build:

.. list-table:: Base ``pysiglib`` build
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``PYSIGLIB_JAX_FFI``
     - ``ON``
     - Set to ``0`` to disable JAX FFI support. When ``ON`` (default), JAX FFI
       is built automatically if JAX is installed, and skipped otherwise.
   * - ``SIGLIB_VEC``
     - ``ON``
     - Set to ``0`` to disable AVX vectorization.

.. list-table:: ``pysiglib-cuda`` plugin build
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``CUDA_ARCH``
     - ``native``
     - CUDA architectures to compile for. Accepts ``native`` (local GPU only),
       ``all`` (all architectures), ``all-major``, or a semicolon-separated list
       (e.g. ``"80;89;90"``). Use ``all`` when building portable wheels.
   * - ``CUDAToolkit_ROOT``
     - (auto)
     - CUDA toolkit prefix. On Windows ``CUDA_PATH`` (set by the NVIDIA
       installer) is picked up automatically; set this only if CMake cannot
       locate ``nvcc`` on its own.

Editable installs
++++++++++++++++++++++++++++++

Both packages support editable installs for development:

.. code-block:: console

    pip install -e .                  # base pysiglib
    pip install -e ./plugins/cuda     # CUDA plugin
