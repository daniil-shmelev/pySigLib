# Pascal direct reversal

This is the historical report before CUDA Pascal was removed and CPU prefix-suffix backward was replaced. See [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for the final implementation. The CUDA Pascal reproduction commands require the earlier implementation. The separate Overleaf repository and measured binaries are not included in this PR.

The backward Pascal implementation starts from the final prefix coefficients passed through the existing `coefs` argument. It undoes each elementary update, accumulates the adjoints, and recovers the sum `q` during the reverse traversal. There is no preliminary forward sweep, checkpoint trajectory, or triangular tape. Public Python and C interfaces are unchanged. Torch and JAX use their existing saved coefficients. No `_return_state` flag was added.

## Storage and range protection

The shared recurrence uses four arrays of length `m + 1`: current coefficients, increments, adjoints, and increment gradients. CPU scalar and SIMD paths and CUDA thread and warp paths use this recurrence or its lane equivalent. Auxiliary storage per active path and word is O(m), independent of path length. Packed word metadata is independent of path length. Inputs and output gradients are excluded from the bound.

CPU word workers accumulate into the output with atomic additions and combine repeated letters before each scatter. They do not allocate private path-sized gradient arrays. Unsafe factorial scaling or loss of a tiny seed causes a restart in the guarded unscaled recurrence from the saved final coefficients. This restart does not run the forward algorithm. The first segment uses the known tensor unit and a linear-size product recurrence to avoid reconstruction drift at that boundary.

Algorithm 4 uses O(n m^2) arithmetic. In exact arithmetic, its per-segment counts are `3*m*(m-1)/2 - 1` multiplications and `(3*m^2 + m - 2)/2` additions or subtractions, with multiplication by one omitted. Initial scaling costs O(m). The implementation has additional range checks and special cases. The proof and counts describe the stated mathematical algorithm.

The reversal principle is also used for signatures in [Signatory, Appendix C](https://arxiv.org/pdf/2001.00706). Here it is applied to the elementary Pascal updates. Subtraction can amplify rounding error even when every intermediate value is in range.

## Validation

The validation records are in `tmp/reversible_backward_20261110/` from the repository root.

- The Python and API suite passed 667 tests, with four skips at the CUDA degree limit. This includes Torch and JAX CPU. Native tests passed seven CPU and four CUDA cases. They cover both dtypes, mixed and empty words, empty batches, views, arbitrary leading batch dimensions, zero increments, repeated letters, time augmentation, lead-lag, CPU threading, and CUDA streams. See `python_tests_final.txt`, `native_cpu_tests.txt`, and `native_cuda_tests.txt`.
- `check_pascal.py` passes 60 exact rational checks against independently differentiated Chen polynomials. These also check recovery of the tensor unit. Degrees 2 through 6 and one through three segments are included, with terminal-only and all-prefix seeds.
- Fixed 4096-byte workspace tests use 16, 256, and 4096 segments. Repeated backward calls preserve the coefficient and seed arrays exactly. Additional CPU word-worker tests cover bounded storage, transformations, and guarded restarts.
- The range checks include unsafe scaling, subnormal adjoints, generic CPU degrees 32, 35, 64, and 171, and the CUDA degree limit of 20. Existing tolerances were not increased.

The separate stress study passes 42 of 44 cases. Two float32 Brownian paths with 4096 segments fail the fixed pointwise tolerance `3e-5 * (1 + abs(reference))`. The reference uses float64 Horner on the same rounded inputs and cotangents. Float32 Horner also fails this tolerance on both cases. All tested float64 cases pass `2e-11 * (1 + abs(reference))`.

| Device | Pascal maximum absolute error | Pascal relative L2 error | Horner maximum absolute error | Horner relative L2 error |
|---|---:|---:|---:|---:|
| CPU | 4.718e-4 | 1.571e-5 | 9.582e-4 | 2.064e-5 |
| CUDA | 1.271e-3 | 3.107e-5 | 9.582e-4 | 1.962e-5 |

These are observed numerical limits, not passing checks. The full records, including cancellation and zero-increment cases, are in `pascal_accuracy.json`.

## Backward measurements

Final prefix coefficients are prepared outside the timed region for both Horner and Pascal. All remaining reconstruction work is timed. No forward or prefix-suffix paper workload was rerun.

The native heatmaps use 256 segments, batch 512, eight words per path, float32, dimensions 2/3/4/6/8/10/12/16/24/32, and lengths 2 through 16. CPU calls use 20 threads. Each cell has ten warm-up calls and nine alternating paired samples, targeting 50 ms per sample. CUDA wall time is synchronized. Seed reset is included. Inputs and outputs are preallocated; native temporary allocation remains inside the call.

| Device | Minimum Horner/Pascal ratio | Median ratio | Maximum ratio | Cells faster with Pascal |
|---|---:|---:|---:|---:|
| CPU | 2.3081 | 3.9864 | 10.0592 | 150/150 |
| CUDA | 0.0832 | 0.8438 | 1.4354 | 22/150 |

The CPU public-API runtime workload uses dimension 4 with the same segment, batch, and word counts. It includes validation and allocation, three warm-up calls, and nine alternating samples. At length 16, Horner takes 105.8900 ms and Pascal takes 12.7657 ms.

Raw data are `pascal_heatmap_cpu.csv`, `pascal_heatmap_cuda.csv`, and `pascal_backward_cpu.json`; sidecar JSON files retain settings and native library hashes. `pascal_backward_summary.csv` contains the heatmap summary. The measured DLLs are retained in `measured/`. Subsequent source cleanup removed unused checkpoint/tape declarations and changed CPU small-batch word-worker output accumulation. The plotted batch-512 workload does not enter that word-worker path; its measured recurrence is unchanged.

## Local paper

`sig_algo_overleaf/main.tex` updates Algorithm 4, its correctness proof, arithmetic counts, storage bound, range note, backward results, and captions. Algorithm 2 and production prefix-suffix and Horner are unchanged. The changes start from Overleaf revision `5414981f353d921142a0104b2ce54b206049e893` and preserve its concurrent Horner input clarification.

Only the two Pascal backward heatmaps and the backward panels of the two combined Pascal figures were replaced. The original forward panel content was copied into the combined PDFs. The update script verifies exact equality of the preserved forward pixels at 144 dpi and hashes all unchanged forward and prefix-suffix figure files. See `preserved_panels.json`.

The final paper is `sig_algo_overleaf/main.pdf`, with 17 pages. It was compiled twice and checked for warnings and unresolved references. All pages were visually inspected. No commit or push was made to Overleaf.

## Reproduction

From the repository root in PowerShell, after building and staging the Release CPU and CUDA libraries:

```powershell
$env:PYTHONPATH = "$PWD\plugins\cuda;$PWD"
python -m pytest tests/test_pascal_coefficients.py tests/test_sig_coef.py tests/test_sig_coef_backprop.py tests/test_apis_match.py -q -rs --disable-warnings
out/build/windows-cpu/siglib/cpsig_unit_testing/Release/cpsig_tests.exe --gtest_filter='*Pascal*:*SigCoef*:*SignatureCoefficient*'
out/build/windows-cuda/cusig_unit_testing/Release/cusig_tests.exe --gtest_filter='*Pascal*:*SigCoef*:*SignatureCoefficient*'
python benchmarks/reversal/check_pascal.py --output tmp/reversible_backward_20261110/pascal_accuracy.json
python sig_algo_overleaf/scripts/benchmark_pascal_heatmaps.py --library-dir pysiglib --device cpu --phase backward --output tmp/reversible_backward_20261110/pascal_heatmap_cpu.csv
python sig_algo_overleaf/scripts/benchmark_pascal_heatmaps.py --library-dir plugins/cuda/pysiglib_cuda --device cuda --phase backward --output tmp/reversible_backward_20261110/pascal_heatmap_cuda.csv
python sig_algo_overleaf/scripts/benchmark_runtime_by_degree.py --repo . --algorithm pascal --phase backward --device cpu --degrees 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --batch 512 --dimension 4 --output tmp/reversible_backward_20261110/pascal_backward_cpu.json
python sig_algo_overleaf/scripts/update_pascal_backward.py --results-dir tmp/reversible_backward_20261110 --old-heatmap-dir tmp/pascal_cpu20_batch512_20261021 --old-runtime-dir tmp/runtime_degree_20261108
Push-Location sig_algo_overleaf
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

The separate prefix-suffix study and its reproduction commands are in `benchmarks/reversal/REPORT.md`. Its results do not change the paper's prefix-suffix performance claims.
