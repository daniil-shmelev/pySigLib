JAX API
========================

.. versionadded:: v3.0.0

pySigLib provides a JAX API which exposes all the same functions, but makes them compatible with JAX transformations
(``jax.jit``, ``jax.grad``, ``jax.vmap``). Just import

.. code-block:: python

    import pysiglib.jax_api as pysiglib

The JAX API uses `XLA FFI <https://jax.readthedocs.io/en/latest/ffi.html>`_ to call the same underlying
C++/CUDA library, so performance is on par with the base API. All functions support ``jax.jit`` compilation
and ``jax.grad`` differentiation.

.. note::

    pySigLib must be built with JAX FFI support enabled (requires jaxlib >= 0.5.0, Python 3.10+).
    Check ``pysiglib.BUILT_WITH_JAX_FFI`` to verify.
    ``sig_to_log_sig`` with ``method=3`` is not supported — use ``log_sig(method=3)`` instead.
