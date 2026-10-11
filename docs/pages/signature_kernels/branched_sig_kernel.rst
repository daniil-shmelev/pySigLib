pysiglib.branched_sig_kernel
================================

``pysiglib.branched_sig_kernel`` computes the depth-truncated BCK branched
signature kernel for one pair of paths or a batch of path pairs. It uses the
same ``static_kernel`` interface as ``pysiglib.sig_kernel``, so linear, scaled
linear, RBF, Matern, rational quadratic, and custom static kernels can be used.

The ``depth`` parameter truncates by rooted-tree depth. Select ``method="finite_difference"`` with ``dyadic_order`` for grid refinement, or ``method="polynomial"`` with ``order`` for Gauss–Legendre product integration. Polynomial order ``p`` uses ``p + 1`` nodes per coordinate, with supported orders 0 through 64.

``return_grid=True`` returns the final recursion grid instead of only the endpoint value: the refined grid for finite differences, or values at the original path vertices for the polynomial method. Both methods support explicit backpropagation, PyTorch autograd and JAX differentiation on CPU and CUDA.

.. autofunction:: pysiglib.branched_sig_kernel
