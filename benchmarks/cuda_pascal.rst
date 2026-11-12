CUDA inverse bidiagonal Pascal coefficients
===========================================

``method="pascal"`` now selects inverse bidiagonal Pascal factors on CUDA.
The forward and backward kernels use one thread per path and word, fixed
degree specialisations through 20, and reversible updates. ``auto`` keeps
Horner on CUDA. CPU selection is unchanged.

The factorisation follows the inverse form in Section 3.1 of Potter and
Duraiswami, `Fast and Stable Pascal Matrix Algorithms
<https://arxiv.org/abs/1711.08453>`_. The weighted factors use no division by
path increments. The paper gives the weighted identity and reverse order.

Optimisation study
------------------

The measurements used an RTX 4060 Laptop GPU, CUDA 12.9.86, MSVC 19.44 and
driver 621.14. The launch study tested blocks of 32, 64, 128 and 256 threads,
both dtypes, degrees 2, 4, 8, 12, 16 and 20, and batches 32, 512 and 4096.
The workload used eight words and 256 segments. Further checks covered
dimensions 1, 2, 4, 8 and 32, short paths, and the grouping threshold.

The retained changes are:

* 32 threads per block. Larger blocks lost on small batches and did not
  give a consistent gain on larger batches.
* A single forward finite-value scan after the segment loop. An infinite
  or NaN accumulator cannot become finite through later additive updates.
* A loop in the guarded forward fallback, reducing register pressure in
  the ordinary forward kernel.
* One allocation for metadata and the backward retry flag.
* Separate grouped-gradient kernels for float32, no augmentation, degree
  at least 8, dimension at most 2, and at least 4096 path-word jobs.
  Repeated letters share an atomic write. Smaller batches and higher
  dimensions use ordinary writes. A nonfinite grouped sum causes a retry
  with guarded arithmetic and ordinary writes.

Keeping path endpoints in registers slowed the tested backward kernels and
was rejected. Grouping at 1024 jobs or in higher dimensions also caused
regressions. The final threshold excludes those cases. The study does not
establish an optimum on other GPU architectures.

Nsight Compute measured 186 registers per thread in the initial float32
degree-16 forward kernel and 157 in backward. The initial backward profile
spent about 51% of warp issue latency waiting on L1TEX operations. Profiles
and raw tuning records are retained under ``tmp/cuda_inverse_pascal/``.

Production timings
------------------

The measured source revision is ``d5c0b9645bab01e6604db2da4af632b3a136f5d3``.
The native library SHA-256 is
``7f4c5ff2aa9add355f8a3e64d206c2142407f04306fb3bcda7d8732a128a2dd9``.
All comparisons are against current Horner, with native allocations,
metadata transfers and synchronisation included. Input/output buffers and
the final prefix coefficients for backward are prepared outside timing.
The reported speedup is the median of paired Horner/Pascal ratios.

Figure 5 uses float32, 512 paths, 256 segments and eight words. Its 150
forward ratios range from 0.97 to 1.30, with median 1.12; 149 exceed one.
Its 150 backward ratios range from 0.93 to 1.65, with median 1.15; 147 exceed
one. Each cell has nine alternating paired samples targeting 50 ms.

The separate float64 confirmation uses the same batch, segment and word
counts, dimension 4, and fifteen paired samples per case:

.. list-table:: Paired median speedup over Horner
   :header-rows: 1

   * - Word length
     - Forward
     - Backward
   * - 4
     - 1.36
     - 1.20
   * - 8
     - 1.79
     - 1.68
   * - 16
     - 2.23
     - 2.20
   * - 20
     - 2.31
     - 2.37

The complete final timing records and provenance are archived with the
manuscript under ``sig_algo_overleaf/data/cuda_pascal_20261112/``. These
results support an explicit CUDA method, not an unconditional default.

Validation
----------

The coefficient and API suite passed 628 tests, with 28 skips. The skips
cover the unavailable JAX GPU backend and degrees beyond the CUDA limit.
Twelve native coefficient tests passed under Compute Sanitizer with zero
memory errors. The CUDA JAX FFI compiled, but JAX GPU execution could not
be tested on this Windows environment; the JAX CPU transform tests passed.

Tests include independent Chen sums and complex-step directional
derivatives, both dtypes, all prefix adjoints, repeated and zero increments,
augmentation, noncontiguous arrays, empty batches, mixed degrees, nondefault
streams, range fallback, and clearing partially written gradients before a
retry. The new grouped kernel is checked against Horner at large batches.
All 300 figure cells pass atol=2e-4 and rtol=1e-3 against Horner. Floating-point
cancellation in reverse reconstruction remains possible; agreement with
Horner is not an independent accuracy guarantee.

Reproduction
------------

Build the production CUDA target and install its DLL into the CUDA plugin.
Run the affected tests from the repository root:

::

    python -m pytest tests/test_pascal_coefficients.py tests/test_sig_coef_backprop.py tests/test_sig_coef.py tests/test_apis_match.py -q

The standalone launch benchmark includes the production kernel header. From
a configured CUDA/MSVC shell, build and run it with:

::

    nvcc -std=c++20 -O3 -arch=sm_89 -shared -I siglib/shared benchmarks/cuda_pascal_tuning.cu -o tmp/cuda_pascal_tuning.dll -Xcompiler /MT -Xcompiler /EHsc
    python benchmarks/cuda_pascal_tuning.py --library tmp/cuda_pascal_tuning.dll --output tmp/cuda_pascal_tuning.json

The Figure 5 benchmark and plot commands are in the manuscript's
``data/README.txt``. The CUDA update verifies that the CPU panels stay
pixel-identical.
