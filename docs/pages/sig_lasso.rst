Signature Lasso
==============

.. autofunction:: pysiglib.sig_lasso

Example
-------

.. code-block:: python

    import math
    import numpy as np
    import pysiglib

    rng = np.random.default_rng(42)
    paths = rng.normal(size=(100, 12, 6)).cumsum(axis=1) * 0.1
    y = paths[:, -1, 0] - paths[:, 0, 0]
    model = pysiglib.sig_lasso(paths, y, 4, alpha=0.01)
    prediction = np.full(len(paths), model['intercept'])
    if model['words']:
        variation = np.max(np.abs(np.diff(paths, axis=1)), axis=2).sum(axis=1)
        scale = np.where(variation > 0, variation, 1.)
        normalized = (paths - paths[:, :1]) / scale[:, None, None]
        features = pysiglib.sig_coef(normalized, model['words'])
        weights = [math.factorial(len(w)) for w in model['words']]
        prediction += (features * weights) @ model['coef']

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

The sparse method bounds all omitted words with Bernstein prefix polynomials
and integral suffix envelopes. Binary alphabet blocks, a variation cap and
signed one-letter energy bounds tighten the search. It never calls the full
signature transform. A final search checks the residual formed from the
returned coefficients, including float32 rounding. Residual-based and
active-equality dual candidates receive full-dictionary feasibility checks.
The actual dual vector is checked again after scaling. ``converged`` requires
the primal-dual gap to be at most ``tol * max(1, mean(centered_y**2))``.
An iteration limit or output rounding can prevent a requested tolerance.
In that case the fit returns ``converged=False`` with a warning.

This is a floating-point certificate with conservative bound padding, not an
interval-arithmetic proof. Correlated features can give nonunique coefficients.
Use predictions, objectives and gaps to compare fits.

Resources and statistics
------------------------

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
use a GPU profiler for kernel durations. Total fitting time includes completion.
``total_seconds`` is native fitting time; Python validation and transfer costs
are additional. ``peak_bytes`` is the peak accounted workspace, including
conservative metadata and existing-kernel scratch allowances. It is not process
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
