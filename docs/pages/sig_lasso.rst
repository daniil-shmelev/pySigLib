Signature Lasso
==============

.. autofunction:: pysiglib.sig_lasso

Example
-------

.. code-block:: python

    import numpy as np
    import pysiglib

    rng = np.random.default_rng(42)
    paths = rng.normal(size=(100, 12, 6)).cumsum(axis=1) * 0.1
    y = paths[:, -1, 0] - paths[:, 0, 0]
    model = pysiglib.sig_lasso(paths, y, 4, alpha=0.01)
    prediction = model['signatures'] @ model['coef'] + model['intercept']

The same call accepts CUDA Torch tensors. It returns tensors on the input
device, with no autograd connection. A fit needs at least one sample.
The variation and normalization use float64 after time augmentation and
lead-lag transforms. Apply the same transforms and conversion before
normalizing paths for prediction. A constant
path has zero nonempty signature features. The degree is limited to 170,
because the factorial weights must be finite in float64.

Solver and certificate
----------------------

Both methods solve the same weighted, centred Lasso objective with feature-sign
search. The solver activates one violating word, solves the signed active
equations, and moves to the stationary point or the first zero crossing.
Zero coefficients leave the active set and can enter again later. A candidate
pool avoids a complete dictionary search after each correction.

The active columns are scaled by their Euclidean norms for QR factorisation.
This is a change of variables; it does not change the Lasso penalty. Singular
systems use SVD, including nullspace descent when required. Updates must pass
accuracy and objective-descent checks. A failed check triggers bounded
refinement and a QR rebuild before the solver reports numerical failure.

``max_iter`` limits signed correction steps per response. ``n_iter`` reports
this count, including boundary and nullspace steps. An empty optimum needs
zero steps. ``batch_size`` limits the number of candidates returned by one
search, not the active-set size. ``initial_words`` seeds the candidate pool;
it does not force their coefficients to be nonzero.

The sparse method integrates the absolute prefix against the remaining-variation bound
``|S(x)_{t,T}^v| <= V(t)**q / q!``, where ``q = len(v)`` and
``V(t) = integral_t^T ||dx(s)||_infinity``. A backward sum computes the
remaining variation at segment endpoints. Two integration methods are available:

* ``bound_method="approximate"`` (default) uses the trapezoidal rule at all
  path nodes. Exact time-prefix values are computed through Chen's relation,
  without polynomial arrays. The proxy can underestimate a branch score and
  miss words. It is used throughout search and stopping, with no exact
  verification or repair pass. ``dual_gap`` is an estimate and ``converged``
  means that the approximate stopping test passed.
* ``bound_method="exact"`` integrates the absolute prefix polynomial in
  Bernstein form. De Casteljau subdivision resolves sign changes. Sign-definite
  intervals are integrated analytically for all required suffix degrees.
  Numerically unresolved intervals cause branch expansion.

Full fitting ignores ``bound_method``. Both methods compute exact feature
columns up to floating-point error. Binary alphabet blocks and signed
one-letter energy bounds tighten the search. Sparse fitting never calls the
full signature transform or ``sig_coef``. A final search checks the residual formed from the
returned coefficients, including float32 rounding. Residual-based and
active-equality dual candidates receive full-dictionary feasibility checks.
The actual dual vector is checked again after scaling, using the selected
integral method. ``converged`` requires
the primal-dual gap to be at most ``tol * max(1, mean(centered_y**2))``.
An iteration limit or output rounding can prevent a requested tolerance.
In that case the fit returns ``converged=False`` with a warning.

For full and exact sparse fits, this is a floating-point certificate with conservative bound padding, not an
interval-arithmetic proof. Correlated features can give nonunique coefficients.
Use predictions, objectives and gaps to compare fits.

Resources and statistics
------------------------

Use ``search_method`` to compare sparse-search optimisations with the same
paths, penalty and tolerance:

.. list-table:: Sparse search modes
   :header-rows: 1

   * - Value
     - Channel grouping
     - Signed terminal blocks
   * - ``"plain"``
     - No
     - No
   * - ``"grouped"`` (default)
     - Yes
     - No
   * - ``"signed"``
     - No
     - Yes
   * - ``"grouped_signed"``
     - Yes
     - Yes

``method="full"`` ignores this setting. Grouped search includes the existing
one-letter energy tightening. Channels keep their input order. Signed checks
evaluate the one- and two-letter suffix correlations in the final two levels,
without storing descendant columns. They run when the integral bound fails
to prune. Roundoff padding uses absolute terms and a variation majorant;
uncertain blocks retain the integral bound. These checks can cost more than
the work they save, so compare total runtime as well as constructed columns.

``terminal_queries`` counts signed-block checks, ``terminal_pruned`` counts
successful checks, and ``terminal_correlations`` counts evaluated signed
correlations. These counters include certification searches. They are zero
when signed checks are disabled and for full fits.

``benchmarks/benchmark_sig_lasso_search.py --device both`` runs five
interleaved repetitions after warm-up and checks predictions against a full
fit outside the timed sections. It reports fit runtime, peak native workspace,
constructed columns and signed-check counters. Input transfers are excluded.

All fitting state is private to a call. CPU work uses Highway and oneTBB.
The native entry points are ``sig_lasso_f`` and ``sig_lasso_d`` on CPU,
and ``sig_lasso_cuda_f`` and ``sig_lasso_cuda_d`` on CUDA. Each fits and
copies the complete result in one call. A caller-supplied allocator receives
the result shape and provides output buffers; no result handle or release
call is needed. Python supplies this allocator automatically.
Native fitting buffers use float64, also for float32 inputs.
CPU factorisation uses incremental thin QR, Givens deletions, and Eigen QR/SVD
for rebuilds. CUDA uses device buffers, fused updates, cuBLAS and cuSOLVER on
the caller's stream. Host control transfers contain compact status records.
Full and sparse responses share stored columns. Independent CPU solves run
against a fixed store; new columns are inserted between solve phases.
With an explicit memory limit, CPU tree frontiers run serially to bound live
traversal storage. Correlation and independent-response work can still run in parallel.

``stats['columns']`` counts admitted solver columns. ``constructed`` counts
distinct word nodes with a constructed terminal feature vector, including
prefixes and initial words. It excludes scalar contractions used only inside
an energy bound. ``evaluated`` counts node evaluations, including recomputation
after eviction. ``dictionary_size``
is a Python integer and can exceed 64-bit indexing limits.

``activations`` and ``drops`` count active-set changes. ``qr_appends``,
``qr_deletions``, ``qr_rebuilds``, ``singular_steps`` and ``certificate_passes``
count the corresponding operations, summed over responses.
``numerical_failures`` counts failed active corrections after a rebuilt retry.
``full_transform_calls`` records native full-signature kernel calls; it is
zero for sparse fits.

Phase times cover preparation, search, column insertion, solves and final
certification. ``factorization_seconds`` is included in solve time, except
for factor work during certification. Parallel factor times sum worker time.
CUDA subphase times measure host calls and can include asynchronous work;
use a GPU profiler for kernel durations. Total fitting time includes completion
and feature export. ``output_seconds`` measures export, and ``output_bytes``
counts output buffers. ``signatures`` has shape ``(*path_batch_shape, n_words)``
and contains uncentred ``k! * S(x)^w / V(x)**k`` values in returned-word order.
It contains one matrix shared across responses, or an empty last axis when
the support is empty. Values are gathered from stored columns without recomputation.
``total_seconds`` is native fitting time; Python validation and transfer costs
are additional. ``peak_bytes`` is the peak accounted workspace, including
outputs, conservative metadata and existing-kernel scratch allowances. It is not process
RSS. ``memory_limit`` applies to this accounting, including owned CPU and CUDA
buffers, factors, candidate metadata, queried library scratch and documented
handle-workspace allowances. Caller-owned input arrays and the CUDA runtime
context are outside it. Optional caches are evicted when allocations need room.
QR storage grows as ``O(N*s + s**2)`` for active size ``s``; SVD can need
additional quadratic storage. Sparse fitting can thus be costly when the
optimal model has a large support.

Reference implementation
------------------------

The algorithm follows the feature-sign method of
`Lee et al. (2006), including singular-system handling in footnote 3
<https://papers.nips.cc/paper_files/paper/2006/file/2d71b2ae158c7c5912cc0bbde2bb9d95-Paper.pdf>`_.
The native implementation has no Celer dependency.
Stopping uses the Lasso primal-dual gap, as in
`Fercoq et al. (2015) <https://proceedings.mlr.press/v37/fercoq15.html>`_.

``sig_branch_and_bound/experiments/feature_sign_benchmark.py`` measures
CPU/CUDA full and sparse fits on explicit matrices, ENSO, NATOPS and multilevel
synthetic paths. It uses five interleaved repetitions after warm-up, with memory
probes and independent correctness audits outside timed regions.
``native_sig_lasso_benchmark.py`` also compares Celer on explicit matrices.
