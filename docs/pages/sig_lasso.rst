Signature Lasso
==============

.. autofunction:: pysiglib.sig_lasso

.. autofunction:: pysiglib.sig_lasso_predict

Feature dictionaries
--------------------

``feature_type="sig"`` is the default and retains the signature dictionary.
``"log"`` uses tensor log-signature coefficients indexed by Lyndon words,
as in ``log_sig(method=1)``. ``"hall"`` uses canonical log coefficients in
the reverse-degree Hall basis: higher degrees precede lower degrees, with
recursive left-right order within a degree and the original channel order
on letters. A bracket ``(u,v)`` is admitted when ``u<v`` and either ``u``
is a letter or ``v<=right(u)``. Returned Hall words are foliage words.
This Hall basis differs from the Lyndon basis returned by
``log_sig(method=2)`` and ``log_sig(method=3)``. The dual Hall recursion is
given in `Salvi et al., Theorem 5.24 <https://arxiv.org/pdf/2212.00134>`_.

All three dictionaries accept NumPy arrays and CPU Torch tensors. For a word
of degree ``k``, the column scale is ``k!/V**k`` after path transforms. Changing
the dictionary changes the penalised problem.

Working-set fit
---------------

Choose ``loss="squared"``, ``"logistic"``, ``"poisson"`` or ``"huber"``. The
default squared loss includes the factor one half. Logistic labels are zero
or one, Poisson labels are nonnegative counts, and the Huber threshold is one.
For all four losses, prediction returns the intercept plus the feature matrix
times the coefficients. Apply the sigmoid or exponential to obtain the fitted
mean for logistic or Poisson regression respectively.

The CPU solver fits the current columns by cyclic coordinate descent and drops
zero coefficients. For squared loss it can take a Cholesky step on the current
nonzero coefficients when this decreases the restricted objective. Targets can
reuse selected columns and their inner products. Each target has its own
penalty, residual and optimality certificate.

The signature search shares prefix polynomials and suffix bounds. For the final
one or two letters it tests norm bounds before computing signed correlations.
The tensor-log search applies the same principle to admissible Lyndon words,
using their prefix period to restrict the possible final letters. Both searches
retain a bounded shortlist. Prediction evaluates selected words in sample tiles
and reuses their common prefixes.

For the default double-precision, multi-response squared-loss search at degrees
two through four, targets can share a traversal. The signature search builds
a bound for each path and tests it against the individual penalty-scaled
residuals. On larger dictionaries, both solvers can first search through one
degree below the requested truncation. A lower-degree violation gives a valid
working-set update. When this first search finds no candidates for a target,
the solver checks the remaining degree at the same residual and combines the
two bounds. Subsequent searches cover the full dictionary.

Each target keeps its own signed correlations and dual certificate. A 16-bit
approximation and its rounding error can prioritise candidates between
searches. The solver evaluates activated columns in double precision. On
wider inputs it keeps high-energy channels in memory and reconstructs
neighbouring cold channels together. For time panels, signature increment
maxima use upward-rounded bfloat16 and tensor-log norms use upward-rounded
8-bit integers with a scale per channel. The tensor-log search can exclude
all admissible final letters using group norms before fetching their paths.
Workers share read-only geometry. Between restricted solves, the solver can
retain the lower triangle of the Gram matrix for surviving terms. An explicit
memory limit uses the individually budgeted search.

``kkt_violation`` gives the maximum stationarity error or a conservative upper
bound. Squared fits also return a primal-dual gap. Each target has its own
diagnostics and penalty. ``stats`` includes the native workspace peak in bytes
and the preparation, search, restricted solve and certification times in seconds.

Example
-------

.. code-block:: python

    import numpy as np
    import pysiglib

    rng = np.random.default_rng(42)
    paths = rng.normal(size=(100, 12, 3)).cumsum(axis=1)
    labels = (paths[:, -1, 0] > paths[:, 0, 0]).astype(float)
    model = pysiglib.sig_lasso(paths[:80], labels[:80], 3, alpha=0.01,
                               feature_type="log", loss="logistic")
    test_scores = pysiglib.sig_lasso_predict(paths[80:], model)
    predicted_labels = test_scores >= 0
