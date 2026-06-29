Volterra Kernels
========================

A Volterra signature is defined relative to a kernel object. The kernel encodes
the state matrix :math:`\Lambda`, readout vectors :math:`b_p` and coefficient
tensors :math:`A_p` of

.. math::

    K(t,s) = \sum_p \left(1^T e^{-\Lambda (t-s)} b_p\right) A_p.

All kernels derive from :class:`~pysiglib.VolterraKernel`, which provides the
:meth:`~pysiglib.VolterraKernel.prepare` and
:meth:`~pysiglib.VolterraKernel.clear_cache` workflow. A kernel is prepared once
for a given truncation level, time step and dtype, and the cached preparation is
then reused by every :func:`~pysiglib.volterra_sig` call with matching
parameters.

.. code-block:: python

    import numpy as np
    import pysiglib

    A = np.eye(3)[None]   # (q, m, d)

    # Finite state-space kernel built from a Jordan realization.
    kernel = pysiglib.VolterraFSSK.from_jordan(
        A=A, b=np.ones((1, 3)), real_rates=[1.0], real_sizes=[3])
    kernel.prepare(degree=3, dt=0.01)

    # Fractional (rough) kernel, Hurst H = beta - 1/2.
    rough = pysiglib.VolterraFractionalKernel(A, beta=0.7, R=8)
    rough.prepare(degree=3, dt=0.01)

Base Class
------------------

.. autoclass:: pysiglib.VolterraKernel
   :members:

Finite State-Space Kernel
---------------------------

.. autoclass:: pysiglib.VolterraFSSK
   :members:

Fractional Kernel
------------------

.. autoclass:: pysiglib.VolterraFractionalKernel
   :members:
