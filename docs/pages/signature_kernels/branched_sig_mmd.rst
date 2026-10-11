pysiglib.branched_sig_mmd
=========================

Compute the unbiased estimate of squared maximum mean discrepancy using the depth-truncated branched kernel. Each sample must contain at least two paths; the unbiased estimate can be negative. ``method="polynomial", order=p`` selects the native product-integration solver with ``p + 1`` nodes per coordinate. Alternatively, pass ``dyadic_order`` for the default finite-difference method.

The same function is exposed by ``pysiglib.torch_api`` and ``pysiglib.jax_api`` for differentiation on CPU and CUDA. Leading batch dimensions are flattened into distribution samples, and ``max_batch`` controls pairwise Gram chunking.

.. autofunction:: pysiglib.branched_sig_mmd
