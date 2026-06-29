Volterra Signatures
========================

.. meta::
   :description: Compute Volterra signatures in Python with pySigLib using finite state-space and fractional (rough) kernels.

This section covers Volterra signature computation in pySigLib. A Volterra
signature generalizes the path signature by driving the path through a Volterra
kernel before forming iterated integrals. The kernel is specified by a kernel
object, which is prepared once for a given truncation level and time step and
then reused across many signature evaluations.

pySigLib evaluates Volterra signatures on CPU using an exact finite state-space
scheme. The kernel

.. math::

    K(t,s) = \sum_p \left(1^T e^{-\Lambda (t-s)} b_p\right) A_p

is represented through a state matrix :math:`\Lambda`, readout vectors
:math:`b_p` and coefficient tensors :math:`A_p`. Two kernel families are
provided: :class:`~pysiglib.VolterraFSSK`, a finite state-space kernel covering
diagonal, dense, oscillatory and defective (Jordan) realizations of
:math:`\Lambda`, and :class:`~pysiglib.VolterraFractionalKernel`, a fractional
(rough) kernel approximated by a sum of exponentials.

.. code-block:: python

    import numpy as np
    import pysiglib

    # Diagonal finite state-space kernel: Lambda is a vector of decay rates.
    Lambda = np.array([0.5, 1.5, 3.0])   # (R,)
    A = np.eye(3)[None]                  # (q, m, d)
    b = np.ones((1, 3))                  # (q, R)
    kernel = pysiglib.VolterraFSSK(Lambda, A, b)

    path = np.random.rand(100, 3)        # (length, dimension)

    # Prepare once per (degree, dt, dtype), then evaluate.
    kernel.prepare(degree=3, dt=0.01)
    sig = pysiglib.volterra_sig(path, degree=3, kernel=kernel, dt=0.01)

.. toctree::
   :titlesonly:
   :maxdepth: 2

   /pages/volterra_signatures/volterra_kernels
   /pages/volterra_signatures/prepare_volterra_sig
   /pages/volterra_signatures/volterra_sig
