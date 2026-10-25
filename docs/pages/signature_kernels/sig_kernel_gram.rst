pysiglib.sig_kernel_gram
==========================

.. versionadded:: v0.2.1

.. autofunction:: pysiglib.sig_kernel_gram

The Chebyshev-Neumann method is available with ``method="chebyshev"`` and the
same ``order`` values as :doc:`sig_kernel`. It supports cross and symmetric Gram
matrices, normalization, batching, and reverse differentiation.

.. code-block:: python

    gram = pysiglib.sig_kernel_gram(
        paths, paths, method="chebyshev", order=(6, 4), max_batch=8,
    )
