pysiglib.sig_kernel
========================

Solver methods
--------------

``method="finite_difference"`` is the default and uses ``dyadic_order``.
``method="polynomial"`` instead uses ``order=N``, where ``N`` is the highest
retained polynomial degree. The native solver therefore carries ``N + 1``
coefficients.

The monomial polysigkernel recurrence follows the
`polysigkernel paper <https://arxiv.org/abs/2502.08470>`_ and the pinned
`monomial approximation source <https://github.com/FrancescoPiatti/polysigkernel/blob/813f1b9c86a4cb2df886faf857f4765b60eccf95/polysigkernel/monomial_approximation_solver.py>`_.

The polynomial solver supports CPU NumPy, Torch, and JAX computations,
including reverse mode. With ``return_grid=True``, it returns the scalar
kernel value at every original path vertex. Reverse mode retains the incoming
boundary coefficients for each tile, or regenerates them when no saved state
is available. Its forward solver workspace is linear in the number of
increments of the second path, but the increment Gram matrix and saved reverse
state remain quadratic in the two path lengths. CUDA tensors are not supported.

.. autofunction:: pysiglib.sig_kernel
