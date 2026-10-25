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
        features = pysiglib.sig_coef(paths, model['words'])
        weights = [math.sqrt(math.factorial(len(w))) for w in model['words']]
        prediction += (features * weights) @ model['coef']

The same call accepts CUDA Torch tensors. It returns tensors on the input
device, with no autograd connection. A fit needs at least one sample.

Solver and certificate
----------------------

Both methods solve the same weighted, centred Lasso objective. They share
coordinate descent, Gap Safe screening, partial working-set selection and
residual extrapolation. Inner accuracy is ``max(restricted_tol, 0.3 * gap)``.
Inner dual points are checked again against the available columns before use
in restricted screening. Dictionary growth resets screening and residual
history, and revalidates the saved dual point.

The sparse method bounds all omitted words with Bernstein prefix polynomials
and integral suffix envelopes. Binary alphabet blocks, a variation cap and
signed one-letter energy bounds tighten the search. It never calls the full
signature transform. A final search checks the residual formed from the
returned coefficients, including float32 rounding. ``converged`` refers to
the full dictionary gap, not only the restricted problem.
An iteration limit or output rounding can prevent a requested tolerance.
In that case the fit returns ``converged=False`` with a warning.

This is a floating-point certificate with conservative bound padding, not an
interval-arithmetic proof. Correlated features can give nonunique coefficients.
Use predictions, objectives and gaps to compare fits.

Resources and statistics
------------------------

All fitting state is private to a call. CPU work uses Highway and oneTBB.
Native fitting buffers use float64, also for float32 inputs.
CUDA traversal, candidate values, coordinate updates and residuals stay on
the device. Working-set control and the final result use host transfers.
Independent full-method CPU responses can run in parallel. Sparse responses
share admitted columns and are processed in sequence.

``stats['columns']`` counts admitted solver columns. ``constructed`` counts
distinct word nodes with a constructed terminal feature vector, including
prefixes and initial words. It excludes scalar contractions used only inside
an energy bound. ``evaluated`` counts node evaluations, including recomputation
after eviction. ``dictionary_size``
is a Python integer and can exceed 64-bit indexing limits.

Phase times cover preparation, search, column insertion and restricted solves.
``total_seconds`` is native fitting time; Python validation and transfer costs
are additional. ``peak_bytes`` is the peak accounted workspace, including
conservative metadata and existing-kernel scratch allowances. It is not process
RSS. ``memory_limit`` applies to this accounting, including owned CPU and CUDA
buffers; framework input storage, driver contexts and vendor-library internals
are outside it. Optional caches are evicted when allocations need room.

Reference implementation
------------------------

The optimisation reference is Celer revision
``eef3bb5f239cf1a0b3a6d0fc8997a20b5e73e21a``, reviewed in
`lasso_fast.pyx <https://github.com/mathurinm/celer/blob/eef3bb5f239cf1a0b3a6d0fc8997a20b5e73e21a/celer/lasso_fast.pyx>`_,
`cython_utils.pyx <https://github.com/mathurinm/celer/blob/eef3bb5f239cf1a0b3a6d0fc8997a20b5e73e21a/celer/cython_utils.pyx>`_
and `homotopy.py <https://github.com/mathurinm/celer/blob/eef3bb5f239cf1a0b3a6d0fc8997a20b5e73e21a/celer/homotopy.py>`_.
Celer is not a runtime dependency.

The benchmark script is
``sig_branch_and_bound/experiments/native_sig_lasso_benchmark.py``. It compares
Celer on explicit matrices, runs controlled solver ablations, and measures
full and sparse CPU/CUDA fits in five interleaved repetitions after warm-up.
Dense correctness audits are outside timed regions.
