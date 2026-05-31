pysiglib.branched_sig_kernel
================================

``pysiglib.branched_sig_kernel`` computes the depth-truncated BCK branched
signature kernel for one pair of paths or a batch of path pairs. It uses the
same ``static_kernel`` interface as ``pysiglib.sig_kernel``, so linear, scaled
linear, RBF, Matern, rational quadratic, and custom static kernels can be used.

The ``depth`` parameter truncates by rooted-tree depth. The optional
``dyadic_order`` parameter controls the finite-difference grid refinement, and
``return_grid=True`` returns the final recursion grid instead of only the
endpoint value.

.. autofunction:: pysiglib.branched_sig_kernel
