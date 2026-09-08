JAX API
========================

.. meta::
   :description: Use pySigLib path signatures, log signatures, and signature kernels with JAX jit, grad, and vmap.

.. versionadded:: v3.0.0

Install ``pysiglib[jax]`` to use this API without installing Torch. Array inputs and outputs are ``jax.Array``, with annotations for static checking. It supports JAX transformations (``jax.jit``, ``jax.grad``, ``jax.vmap``). Select JAX explicitly:

.. code-block:: python

    import pysiglib.jax_api as pysiglib

The JAX API uses `XLA FFI <https://jax.readthedocs.io/en/latest/ffi.html>`_ to call the same underlying
C++/CUDA library, so performance is on par with the base API. All functions support ``jax.jit`` compilation
and ``jax.grad`` differentiation.

.. note::

    pySigLib must be built with JAX FFI support enabled (requires jaxlib >= 0.9.1, Python 3.11+).
    Check ``pysiglib.BUILT_WITH_JAX_FFI`` to verify.

Custom static kernels implement ``__call__(x: jax.Array, y: jax.Array) -> jax.Array`` on batched paths and return the double-differenced Gram matrix. JAX differentiates the callable; a manual backward context is unnecessary. Use the kernel and streaming classes exported by this namespace.
