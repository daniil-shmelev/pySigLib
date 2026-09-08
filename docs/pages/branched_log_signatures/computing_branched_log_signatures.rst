Computing Branched Log Signatures
=================================

Let :math:`\mathcal{H}` be a connected graded Hopf algebra with counit :math:`\varepsilon`. For a character :math:`X: \mathcal{H} \to \mathbb{R}`, its convolution logarithm is

.. math::

    \log_*(X) = \sum_{k \geq 1} \frac{(-1)^{k-1}}{k} (X-\varepsilon)^{*k}.

Here :math:`*` is convolution of linear maps induced by the coproduct. The series is finite at each truncation degree. Applying it to a branched signature gives its branched log signature. The coproduct is BCK when ``planar=False`` and MKW when ``planar=True``.

Preparing for Branched Log Signature Computations
-------------------------------------------------

Before computing a branched log signature, ``pysiglib`` requires a call to
``pysiglib.prepare_branched_log_sig``. This pre-computes and caches the tree or
ordered-forest basis, coproduct tables, and derived logarithm tables required for
the computation. This function should be run only once before the computation,
for each required ``(dimension, degree, method, planar)`` combination.

.. code-block:: python

    import numpy as np
    import pysiglib

    pysiglib.prepare_branched_log_sig(2, 3, method=0, planar=False)
    pysiglib.prepare_branched_log_sig(2, 3, method=1, planar=True)

    for i in range(10):
        X = np.random.rand(200, 2)

        X_bck_logsig = pysiglib.branched_log_sig(X, 3, method=0)
        X_mkw_logsig = pysiglib.branched_log_sig(X, 3, planar=True, method=1)

The prepared object depends on the value of ``planar``. Preparing with
``planar=False`` is not sufficient for ``planar=True``, and conversely.

Methods and output formats
--------------------------

.. list-table:: Branched log-signature methods
   :header-rows: 1
   :widths: 15 25 60

   * - Method
     - Basis
     - Behavior
   * - ``0``
     - Non-planar or planar
     - Expanded logarithm in the tree or ordered-forest basis.
   * - ``1``
     - Planar only
     - Compressed weighted Lyndon-forest coordinates.
   * - ``2``
     - Planar only
     - Compressed coordinates in the corresponding Lie basis.
   * - ``3``
     - Planar only
     - Direct BCH computation from the path, with the same output coordinates as method 2.

If ``method`` is omitted from a computation, the default is 0 for non-planar inputs and 1 for planar inputs.
