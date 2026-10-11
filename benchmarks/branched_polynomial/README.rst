Native CPU and CUDA comparison
==============================

``compare_native_cpu_cuda.py`` benchmarks the integrated library entry points and produces ``results/accuracy_runtime_cpu_cuda.png`` and its PDF counterpart: CPU in the top row and CUDA in the bottom row, each showing Brownian cross-pairs, Brownian self-pairs and oscillatory cross-pairs. Build/install the library and CUDA plugin, then run from the repository root with NumPy, PyTorch and Matplotlib available:

.. code-block:: bash

   PYTHONPATH=. python benchmarks/branched_polynomial/compare_native_cpu_cuda.py

The default inputs match the original figure: 32 segments, depth 3, seed 17, three coordinates and float64. Polynomial labels are quadrature node counts ``q``, corresponding to public API ``order=q-1``; finite-difference labels are dyadic orders 0 through 6. The CPU is pinned to one logical core. The recorded CUDA device is an NVIDIA GeForce RTX 5080 under WSL2.

Each point reports median wall time over seven samples for a scalar-pair forward and backward call. Both backward passes reconstruct their forward state. Native allocation, launches, synchronization and ctypes dispatch are included; Gram construction, transfers and the common Gram-to-path reverse chain are excluded. Quadrature tables are cached. CPU scratch is reused; CUDA scratch uses the library's per-call memory-aware batch workspace. These are single-pair latency measurements, not batched GPU throughput or full MMD timings. The PyTorch/JAX paired APIs can save polynomial forward state for differentiation, whereas this comparison measures the common recomputation workflow.

Accuracy is the maximum of relative kernel error and relative L2 error in the concatenated gradients of both paths. The reference uses 20 polynomial nodes, checked against 16 nodes with a required relative difference below 1e-11. The independent quadrature, complex-step gradient and closed-form checks are covered by ``tests/test_branched_polynomial_sig_kernel.py`` and the earlier experiment below. ``measurements_cpu_cuda.csv`` records every point, ``references_cpu_cuda.csv`` records reference convergence, and ``metadata_cpu_cuda.json`` records hardware, timing scope and source hashes.

Batched kernels and full MMD
===========================

``profile_batches.py`` measures the public PyTorch API, including static Gram construction, both path gradients, allocation and CUDA synchronization. It covers batches of 64 path pairs with 32, 128 and 512 segments, and full MMD workloads with 8 or 32 paths per sample. Both float32 and float64 use depth 3 and order 3. CPU measurements use one core. The MMD calls exercise Gram chunking through ``max_batch``; this is distinct from the native single-pair timing above.

.. code-block:: bash

   PYTHONPATH=. python benchmarks/branched_polynomial/profile_batches.py --label before
   # Rebuild/install the changed native libraries.
   PYTHONPATH=. python benchmarks/branched_polynomial/profile_batches.py --label after
   python benchmarks/branched_polynomial/summarize_batches.py

The JSON files retain every timing range and the loaded native library hashes. The NPZ files retain the kernel/MMD values and both path gradients. ``summarize_batches.py`` verifies numerical agreement before producing ``results/batches_report.txt`` and ``results/batches_comparison.json``. These benchmarks can take several minutes on one CPU core, particularly for long paths and full MMD.

The recorded cleanup comparison covers 24 cases. Values and both gradients agree within the recorded dtype tolerances. Most timings are close to baseline; an isolated repeat of the largest float64 paired CPU case has a median about 17% slower, with wide, overlapping timing ranges. ``cpu_long_repeat.json`` preserves that repeat separately. The CUDA implementation retains the simple row traversal: an experimental warp scan regressed float64 performance in these workloads. The measurements support that implementation choice, not a general speedup claim for the cleanup.

Original prototype comparison
=============================

This experiment compares a Gauss–Legendre polynomial product-integration prototype with the repository's existing branched finite-difference solver. It computes the scalar depth-truncated BCK kernel and its gradient with respect to the cell Gram increments. Both implementations are native C++, compiled together with identical optimisation flags; the finite-difference implementation is included directly from the current repository source without modification.

Run from the repository root using Python with NumPy and Matplotlib installed. A local CMake build under ``build/`` must supply oneTBB headers and its shared library. The benchmark currently targets Linux and requires a C++20 compiler.

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/branched_polynomial/compare.py
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/branched_polynomial/check_references.py benchmarks/branched_polynomial/results

The local recorded run used ``/home/luke/sig-benchmarks/.venv/bin/python``. Add ``--quick`` for a small sweep or ``--validate-only`` for numerical validation. ``--output`` selects another results directory. ``check_references.py`` performs additional, substantially more memory-intensive finite-difference checks, reaching 8192 refined segments per coordinate in the recorded run.

The polynomial parameter is the number of nodes per coordinate, q, so the interpolated integrand has degree q-1 in each coordinate. Integration uses fixed matrices derived from Legendre polynomials and cumulative sums between cells. The exponential is evaluated pointwise. The last depth uses scalar quadrature, and depth one uses its exact exponential-of-summed-increments formula. The reverse pass applies the transposed integration matrices and reverse cumulative sums.

``compare.py`` validates the native implementations against separate NumPy formulations and complex-step derivatives, checks both path derivatives, verifies the single-cell depth-two closed form, and checks that the finite-difference adapter agrees with the public C entry point. Reference results are accepted only when increasing polynomial nodes from 12 to 16, or further if needed, changes the kernel and gradient norms by less than 1e-10 relatively. Finite-difference Richardson extrapolation supplies an independent check; it is used only for validation, never as a timed competitor.

Timing is single-core float64, for one path pair at a time. It excludes static-kernel Gram construction, the common conversion from Gram gradients to path gradients, Python overhead, and thread-pool dispatch. Each timed backward pass reconstructs forward state in both methods, matching the existing uncached reverse workflow. Allocation and reuse within the native routines are timed. There are five measurement samples after warm-up, each targeting at least two milliseconds, and the median is reported. No fast-math flag is used.

The full sweep covers 8, 32 and 128 segments; depths 2, 3 and 5; three seeds; and Brownian cross-pairs, Brownian self-pairs and oscillatory cross-pairs in three dimensions. Path scales are controlled; kernel values range from approximately 0.59 to 4.99. These cases do not establish behaviour near overflow, for very large amplitudes, on GPUs, or in float32. Self-pair path gradients are the two partial derivatives evaluated at equal paths.

Polynomial node counts are 1, 2, 3, 4, 6 and 8. Finite-difference dyadic orders are 0 through 6, subject to a maximum of 2048 refined segments per coordinate in the timed sweep. Missing finite-difference matches at stringent tolerances mean the tolerance was not reached within this sweep, not that further refinement could not reach it. The supplementary reference checks go beyond this timed limit.

The reported memory is a calculated peak numeric workspace payload for warmed calls, not measured process RSS. It includes the existing finite-difference reverse pass's temporary stack-building workspace and retained forward buffers. It excludes allocator overhead, inputs, outputs, quadrature tables and shared-library memory. For depth D greater than one and square inputs, polynomial payloads contain 3M^2+M doubles for forward and (D+3)M^2+M for forward plus backward, where M=q times the segment count. Finite-difference payloads contain 2R^2+R and (D+7)R^2+2R doubles, respectively, where R=2^r times the segment count plus one.

Results are in ``results/report.txt``. ``measurements.csv`` contains all configurations; ``matched_accuracy.csv`` selects the fastest tested configuration satisfying each tolerance; ``references.csv``, ``validation.json`` and ``extra_reference_checks.csv`` record numerical verification; ``metadata.json`` records compiler flags, CPU, source hashes and summary statistics. The PNG and PDF plots show accuracy versus time and matched-accuracy speedups. Runtime ratios are finite-difference time divided by polynomial time, so values above one favour polynomial integration.
