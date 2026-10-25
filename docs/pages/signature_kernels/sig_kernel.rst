pysiglib.sig_kernel
========================

.. autofunction:: pysiglib.sig_kernel

Chebyshev-Neumann method
------------------------

Use ``method="chebyshev"`` to propagate polynomial boundary data in the shifted
Chebyshev basis. The solver uses a finite Neumann sum in each cell, interpolates
the outgoing boundaries at Chebyshev-Lobatto nodes, and uses compensated arithmetic
for shared corner values. This is an approximation method.

The parameter ``order=6`` sets both the boundary degree and Neumann order to 6.
Use ``order=(6, 4)`` to set them separately. The permitted values satisfy
``p >= 1``, ``K >= 0`` and ``p + K <= 31``. Increase both orders to check convergence.
Agreement between two orders is an error indicator, not a rigorous error bound.

.. code-block:: python

    value = pysiglib.sig_kernel(x, y, method="chebyshev", order=(6, 4))
    grid = pysiglib.sig_kernel(x, y, method="chebyshev", order=6, return_grid=True)

The returned grid contains one value per pair of transformed path vertices.
The base API supports explicit backpropagation. The Torch and JAX APIs support
reverse differentiation of the selected finite solver.
