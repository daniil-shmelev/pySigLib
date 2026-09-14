Installation
========================

.. meta::
   :description: Install pySigLib for fast path signatures, log signatures, and signature kernels on CPU and CUDA GPU.

Install from PyPI
------------------------

The recommended way to install pySigLib is from PyPI using pre-built wheels.
No compiler toolchain is required - wheels are published for Windows, Linux,
and macOS (arm64).

Current releases of both ``pysiglib`` and ``pysiglib-cuda`` are published as wheels only. To build from source, use a repository checkout as described below.

.. code-block:: console

    pip install pysiglib              # CPU only
    pip install "pysiglib[cuda]"      # with CUDA GPU support

The ``[cuda]`` extra installs the companion ``pysiglib-cuda`` plugin, which
ships the CUDA shared library (``cusig``) and the CUDA JAX FFI bindings as a
sibling package. ``pysiglib`` discovers it at import time; if the plugin is
absent, ``pysiglib`` runs CPU-only.

JAX support
++++++++++++++++++++++++++++++

Release wheels include XLA FFI bindings for JAX. Install JAX separately in a Python 3.11 or later environment:

.. code-block:: console

    pip install "jax>=0.9.1"

For CUDA 12 JAX on Linux, install:

.. code-block:: console

    pip install "pysiglib[cuda]" "jax[cuda12]>=0.9.1"

JAX does not support CUDA on native Windows. Use Linux or WSL2 for CUDA JAX. See the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_ for supported platforms and CUDA requirements. Native Windows CUDA remains available through the base and Torch APIs.

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

Clone the repository before following the source-build instructions:

.. code-block:: console

    git clone https://github.com/daniil-shmelev/pySigLib.git
    cd pySigLib

Base package (CPU only)
+++++++++++++++++++++++++++++++

This requires a C++20 compiler toolchain. The base wheel is CPU-only - it does
not build CUDA. Use the ``pysiglib-cuda`` plugin for that (next section).

.. tab-set::

   .. tab-item:: Windows

      Requires MSVC. Once installed, run from the repository root:

      .. code-block:: console

          pip install .

   .. tab-item:: Linux

      Requires GCC >= 10. Once installed, run from the repository root:

      .. code-block:: console

          pip install .

   .. tab-item:: macOS

      Requires the Xcode Command Line Tools (``xcode-select --install``). Once
      installed, run from the repository root:

      .. code-block:: console

          pip install .

CUDA plugin
+++++++++++++++++++++++++++++++

The ``pysiglib-cuda`` plugin depends on sources from the parent repository
(``siglib/cusig``). From the repository root, install the base package and
the ``plugins/cuda`` subdirectory:

.. code-block:: console

    pip install .
    pip install ./plugins/cuda

Install both packages from the same checkout so their versions match.

This requires a working CUDA toolkit (``nvcc`` on ``PATH`` or
``CUDAToolkit_ROOT`` set to its install prefix; on Linux this is typically
``/usr/local/cuda``, and on Windows the ``CUDA_PATH`` environment variable
set by the NVIDIA installer is also picked up).

By default the plugin compiles only for the local GPU's architecture
(``CUDA_ARCH=native``). To target multiple architectures, set ``CUDA_ARCH``
to a semicolon-separated list, ``all-major``, or ``all`` before installing:

.. tab-set::

   .. tab-item:: Linux shell

      .. code-block:: console

          CUDA_ARCH="80;89;90" pip install ./plugins/cuda

   .. tab-item:: PowerShell

      .. code-block:: powershell

          $env:CUDA_ARCH = "80;89;90"
          python -m pip install ./plugins/cuda

JAX support (source builds)
++++++++++++++++++++++++++++++

For Python 3.11 and later, both packages declare ``jaxlib >= 0.9.1`` as an isolated build dependency. A standard ``pip install`` therefore provides the FFI headers even if JAX is absent from the active environment. Install JAX in that environment to use the API. Run these commands from the repository root:

.. code-block:: console

    pip install "jax>=0.9.1"
    pip install .
    pip install ./plugins/cuda    # only if you need CUDA

When build isolation is disabled, install ``jaxlib >= 0.9.1`` in the build environment yourself. Builds on Python 3.9 or 3.10 omit the FFI dependency. The CMake build skips FFI if compatible headers are unavailable or ``PYSIGLIB_JAX_FFI=OFF``. To verify the installed base package:

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
     - Set to ``OFF`` to disable JAX FFI support. When ``ON``, build FFI if compatible ``jaxlib`` headers are available.
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
