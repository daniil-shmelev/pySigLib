Computing Branched Log Signatures
=================================

Let :math:`\mathcal{H}` be a Hopf algebra with counit :math:`\varepsilon`.
For an element :math:`x \in \mathcal{H}` the logarithm is defined by

.. math::

    \log_*(x)
      = \sum_{k \geq 1} \frac{(-1)^{k-1}}{k}
        (x-\varepsilon)^{*k}.

Applying this logarithm map to the branched signature yields the `branched log
signature`. The logarithm is taken in the same Hopf algebra as the branched
signature itself: the BCK algebra when ``planar=False`` and the MKW ordered
forest algebra when ``planar=True``.

Preparing for Branched Log Signature Computations
-------------------------------------------------

Before computing a branched log signature, ``pysiglib`` requires a call to
``pysiglib.prepare_branched_sig``. This will pre-compute and cache the tree or
ordered-forest basis and the coproduct tables required for the computation. This
function should be run only once before the computation, for each required
``(dimension, degree, planar)`` combination.

.. code-block:: python

    import numpy as np
    import pysiglib

    pysiglib.prepare_branched_sig(2, 3, planar=False)
    pysiglib.prepare_branched_sig(2, 3, planar=True)

    for i in range(10):
        X = np.random.rand(200, 2)

        X_bck_logsig = pysiglib.branched_log_sig(X, 3)
        X_mkw_logsig = pysiglib.branched_log_sig(X, 3, planar=True)

The prepared object depends on the value of ``planar``. Preparing with
``planar=False`` is not sufficient for ``planar=True``, and conversely.

