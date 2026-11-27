Signature and log-signature Lasso
=================================

.. autofunction:: pysiglib.sig_lasso

.. autofunction:: pysiglib.log_sig_lasso

.. autofunction:: pysiglib.sig_lasso_predict

Feature dictionaries
--------------------

``sig_lasso`` uses signature coordinates indexed by words. ``log_sig_lasso``
uses coefficients of the tensor logarithm indexed by Lyndon words, as in
``log_sig(method=1)``. Both fitting functions accept NumPy arrays and CPU Torch
tensors. For a word of degree ``k``, the column scale is ``k!/V**k``, where
``V`` is the sum of the infinity norms of the transformed path increments.

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
retain a bounded shortlist. At each update the solver admits up to 16 inactive
words with the strongest available violations. A larger pool lets it rescore
unused candidates after the restricted fit before searching the tree again.
Prediction evaluates selected words in sample tiles and reuses their common
prefixes.

Signature tree orders
---------------------

The default prefix tree removes the final letter to obtain the parent.
With ``tree_order="lyndon"``, a Lyndon word has its longest proper Lyndon
prefix as parent, and a single letter has the empty root as parent.
For any other word, write its unique nonincreasing Lyndon factorisation
:math:`w=f_1\cdots f_m` and delete the last factor. For example, on
:math:`1<2`, the parent of :math:`212=2\,|\,12` is :math:`2`. The
column at :math:`212` is its ordinary normalised signature coefficient.

At a non-Lyndon node ending in factor :math:`f`, the allowable suffixes
are precisely those bounded above by the corresponding prefix of
:math:`fff\cdots`. Appending a larger first Lyndon factor crosses this
boundary. The search carries the two sides as lexicographic intervals,
uses the prefix-suffix bound on admissible channel groups, and shares the
signed contraction when two groups use the same terminal prefix state.
The search obtains the last factor in linear time in the prefix length using
`Duval's algorithm <https://doi.org/10.1016/0196-6774(83)90017-2>`_.

All tree orders use ``terminal_bound="norm"`` by default. The option
``terminal_bound="projected"`` adds a signed bound for channel groups. For a fixed terminal
state, let :math:`h_i` be its segment weights and :math:`d_i^a` the vector
of normalised increments in channel :math:`a`. The unscaled column is
:math:`\langle h_i,d_i^a\rangle`. Partition the segments into four consecutive
panels :math:`J_p` and let :math:`P` replace entries by their panel
means. For a channel group :math:`A`, let :math:`m_{ip}\pm e_{ip}`
enclose the channel means on panel :math:`J_p`. Set
:math:`R_i=\max_{a\in A}\|(I-P)d_i^a\|_2` and
:math:`s_{ip}=\sum_{j\in J_p}h_{ij}`. The degree-:math:`k` correlations satisfy

.. math::

   \max_{a\in A}|c_a(r)| \le \frac{k!}{N}\left(\left|\sum_{i,p} r_i s_{ip} m_{ip}\right| + \sum_i |r_i|\left(\sum_p |s_{ip}|e_{ip} + \|(I-P)h_i\|_2 R_i\right)\right).

The first term retains the sign across paths. The other two terms follow
by bounding each panel mean and applying Cauchy-Schwarz to the orthogonal
remainder. This is the sharp bound given only the independent intervals
and remainder radii for each path. The extremum is obtained by choosing
the interval endpoints and aligning the orthogonal remainders with the
corresponding terminal weights, with signs agreeing across paths. The
implementation also tests the original norm bound and computes the exact
signed correlation if needed. At a single channel it goes directly to the
signed contraction when the panel tests across responses would require
more inner products per sample than the number of path segments.

Other grouped searches
----------------------

All the following orders use the ordinary normalised signature. Returned words
and warm-start words use the original channel labels and chronological order.

* ``suffix`` fixes the last letter first. Reversing the sequence of increments
  gives :math:`S^w(D_{L-1},\ldots,D_0)=S^{\mathrm{rev}(w)}(D_0,\ldots,D_{L-1})`.
  The implementation reverses the segment index during packing and applies the
  prefix search in these coordinates. The same grouped bounds then apply to
  suffixes. See the time-reversal identity in
  `Chevyrev and Kormilitzin, Theorem 2.25 <https://link.springer.com/chapter/10.1007/978-3-031-97239-3_1>`_.
* ``grouped`` uses the input channel order and also tests groups in the last
  one or two positions. ``balanced`` uses these groups and divides the parallel
  root search by estimated work. It splits the group with the largest sum of
  training increment energies and schedules the most energetic groups first.
* ``energy`` sorts channels by decreasing :math:`\sum_{i,j}|D_{ij}^a|^2`
  before forming groups. ``profile`` forms a balanced tree on eight-dimensional
  channel norm profiles. For a sample block :math:`G`, the corresponding
  component is :math:`(\sum_{i\in G,j}|D_{ij}^a|^2)^{1/2}`. At each split
  it takes the median on the component with the largest spread. Both orders
  use energy-based work division. The profiles determine the partition.
  Screening uses the bounds over all training samples in each resulting group.
  Widest-axis median splitting follows the construction of
  `Friedman, Bentley and Finkel <https://doi.org/10.1145/355744.355745>`_.
* ``forward`` changes which letter is fixed in a terminal pair :math:`uab`.
  Write :math:`I_j=\int_0^1 P_j^u(s)\,ds` and
  :math:`J_j=\int_0^1 sP_j^u(s)\,ds` for the integrals of the prefix
  coefficient on segment :math:`j`. Fixing :math:`a` gives
  :math:`S^{uab}=\sum_j D_j^b (\sum_{t<j}I_tD_t^a+(I_j-J_j)D_j^a)`.
  One forward scan supplies the weights for every final channel :math:`b`.
  The standard terminal routine fixes :math:`b` and scans backwards.
* ``adaptive`` compares the input, energy and profile partitions using the
  training profiles :math:`e_{ag}`. It minimises the total excess introduced
  by replacing individual channel norms with a group maximum,
  :math:`\sum_{A,g} (|A|\max_{a\in A}e_{ag}-\sum_{a\in A}e_{ag})`,
  where :math:`A` runs over internal channel groups. This score is a proxy
  for pruning work. When a worker receives only a subset of the possible
  penultimate letters, it fixes those letters first. Otherwise it uses the
  backward terminal scan. The group choice and parallel work division use
  training paths alone.

The outside-in bound
^^^^^^^^^^^^^^^^^^^^

With ``tree_order="outside"``, a group specifies the first channel :math:`a`,
the possible final channels :math:`B` and the degree :math:`n=k+2\ge3`.
For one normalised path, let :math:`p_a(t)=x^a(t)-x^a(0)`,
:math:`q_b(t)=x^b(T)-x^b(t)` and let :math:`v(t)` be its cumulative
variation in the infinity norm, with :math:`V=v(T)`. Define

.. math::

   L_{a,k}^2=k\int p_a(t)^2(V-v(t))^{k-1}\,dv(t),\qquad
   R_{b,k}^2=k\int q_b(t)^2v(t)^{k-1}\,dv(t).

For every interior word :math:`w` of length :math:`k`,
:math:`n!|S^{awb}|\le n(n-1)L_{a,k}R_{b,k}`. Integrating the two outside
letters gives :math:`p_a(t_1)q_b(t_k)` on the ordered simplex of the
remaining times. Replacing each interior differential by :math:`dv` and
applying Cauchy-Schwarz on that simplex gives the displayed factors from
the marginal measures of its first and last times.

Across samples, a fixed endpoint pair therefore has the correlation bound
:math:`n(n-1)\sum_i|r_i|L_{i,a,k}R_{i,b,k}/N`. Sample-block Cauchy-Schwarz
first screens whole groups of final channels, and a surviving leaf uses the
samplewise sum. The search reuses the admissible final channels across interior
prefixes at the same residual. In a terminal pair, a worker assigned a proper
subset of the penultimate letters fixes those letters first and contracts only
the surviving final channels. On each linear segment it integrates the squared
excursion against the chord of the variation weight. This is exact at degrees
three and four and conservative at higher degrees by convexity. Only degrees
that can reach the terminal routines need these tables. The tables use
upward-rounded single-precision factors.

The primitive terminal bound
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``terminal_bound="primitive"``, put :math:`T_j^b=\sum_{s\ge j}D_s^b`.
For either orientation of the terminal contraction, summation by parts gives

.. math::

   \sum_{j=0}^{L-1}h_jD_j^b=h_0T_0^b+\sum_{j=1}^{L-1}(h_j-h_{j-1})T_j^b.

For a group of channels, precompute the maxima of :math:`|T_0^b|` and
:math:`\|(T_1^b,\ldots,T_{L-1}^b)\|_2` separately for each path. Multiplying
these maxima by :math:`|h_0|` and :math:`\|\Delta h\|_2`, respectively,
and adding bounds the contraction. Each sample takes the minimum of this
quantity and the ordinary norm bound before the residual-weighted sum.
The discrete identity is given in the
`NIST Digital Library of Mathematical Functions, Section 2.10(ii) <https://dlmf.nist.gov/2.10.ii>`_.

Multiple responses and numerical checks
---------------------------------------

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
    model = pysiglib.log_sig_lasso(paths[:80], labels[:80], 3, alpha=0.01, loss="logistic")
    test_scores = pysiglib.sig_lasso_predict(paths[80:], model)
    predicted_labels = test_scores >= 0
