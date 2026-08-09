pysiglib.sig_kernel_gram
==========================

.. versionadded:: v0.2.1

The ``method``, ``order``, device support, and workspace behavior are the same as
for :doc:`sig_kernel`.

``max_batch`` limits the number of path pairs passed to one JAX solver invocation.
Reducing it lowers peak intermediate memory use without changing the returned Gram
matrix.

With ``return_grid=True``, finite differences return the dyadically refined PDE
grid. The polynomial method returns values on the vertices of the transformed input
paths.

.. autofunction:: pysiglib.sig_kernel_gram
