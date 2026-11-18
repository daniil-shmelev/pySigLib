# Signature and tensor-log branch-and-bound benchmarks

On the 14 configurations in the paper, the sum of median fit times fell from **5.40 s to 4.41 s**, a **1.23× speedup** over the preceding optimized solver. Across the three repetitions of all 26 configurations, the two builds selected identical word sets and agreed on every held-out class label. The fits used the tree throughout and recorded zero full-transform calls. [Individual measurements and numerical checks](measurements.json)

Including the 12 additional tensor-log experiments, the combined signature fit time improved by 1.23× and the combined tensor-log fit time by 2.32×. Every configuration had a shorter median fit time. Tracked peak native allocation fell on 22 configurations and tied on four. Additional process peak RSS fell on 19 configurations and rose on seven. The largest increase was 3.30 MiB on the degree-two Speech Commands signature fit.

## Paper experiments

Each fit used the same split, per-class penalties, 600 training paths and ten responses in both builds. Times are medians of three fresh-process fits on an eight-core Apple M3 with eight feature workers and one BLAS thread. Build order alternated and the second repetition reversed the order of the configurations. Fit time excludes dataset preparation, penalty calibration, test prediction and independent checks. Native MiB measures the fit's tracked high-water allocation, including output arrays. Additional RSS measures the rise in the process high-water mark from just before the fit. The machine, compiler, binary hashes and source hashes are recorded with the results. [Paper protocol](../../../../overleaf/main_2.tex) · [Preceding benchmark](../round2/README.md)

### Signature classification

| Dataset | Degree | Fit s, preceding → current | Speedup | Native MiB, preceding → current | Additional RSS MiB, preceding → current | Test correct | Selected words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.0030 → 0.0024 | 1.23× | 0.23 → 0.23 | 0.20 → 0.31 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.0412 → 0.0321 | 1.29× | 5.48 → 5.38 | 4.91 → 5.55 | 1,907/2,200 | 153 |
| Spoken Arabic Digit | 3 | 0.0702 → 0.0566 | 1.24× | 8.72 → 8.63 | 9.42 → 8.66 | 2,038/2,200 | 204 |
| Spoken Arabic Digit | 4 | 0.610 → 0.509 | 1.20× | 15.81 → 15.25 | 24.80 → 23.95 | 2,093/2,200 | 611 |
| InsectWingbeat | 1 | 0.0138 → 0.0116 | 1.19× | 0.98 → 0.98 | 1.08 → 1.11 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 0.101 → 0.0972 | 1.04× | 24.26 → 14.71 | 23.45 → 19.09 | 346/1,000 | 300 |
| InsectWingbeat | 3 | 0.370 → 0.293 | 1.26× | 35.65 → 25.91 | 38.58 → 30.62 | 375/1,000 | 504 |
| InsectWingbeat | 4 | 2.216 → 1.702 | 1.30× | 39.87 → 29.98 | 50.94 → 40.30 | 397/1,000 | 606 |
| Speech Commands | 1 | 0.0065 → 0.0062 | 1.05× | 0.61 → 0.61 | 0.67 → 0.64 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.0983 → 0.0792 | 1.24× | 10.21 → 9.48 | 9.52 → 12.81 | 463/1,000 | 183 |
| Speech Commands | 3 | 0.790 → 0.682 | 1.16× | 20.73 → 19.00 | 31.00 → 30.91 | 519/1,000 | 633 |
| Speech Commands | 4 | 0.856 → 0.738 | 1.16× | 21.02 → 19.97 | 28.27 → 27.81 | 556/1,000 | 463 |

### Degree-four dictionary comparison on Spoken Arabic Digit

Each response used 0.03 times its maximum centred training correlation over the respective dictionary. This separate calibration accounts for the difference from the Arabic classification row above. [Per-response penalties](../protocol.json)

| Dictionary | Fit s, preceding → current | Speedup | Native MiB, preceding → current | Additional RSS MiB, preceding → current | Test correct | Selected words |
|---|---:|---:|---:|---:|---:|---:|
| Signature | 0.102 → 0.0864 | 1.18× | 11.65 → 11.61 | 13.48 → 13.86 | 1,969/2,200 | 114 |
| Tensor log | 0.126 → 0.114 | 1.11× | 4.05 → 3.94 | 9.84 → 7.91 | 1,937/2,200 | 46 |

## Additional tensor-log experiments

These fits use the classification rows’ penalty vectors on the same paths and labels. The dictionary consists of tensor-log coefficients indexed by Lyndon words.

| Dataset | Degree | Fit s, preceding → current | Speedup | Native MiB, preceding → current | Additional RSS MiB, preceding → current | Test correct | Selected words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.0025 → 0.0023 | 1.10× | 0.23 → 0.23 | 0.34 → 0.33 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.0230 → 0.0213 | 1.08× | 2.91 → 2.51 | 3.56 → 3.03 | 1,916/2,200 | 75 |
| Spoken Arabic Digit | 3 | 0.0767 → 0.0693 | 1.11× | 3.11 → 2.90 | 4.27 → 4.22 | 1,993/2,200 | 73 |
| Spoken Arabic Digit | 4 | 0.494 → 0.449 | 1.10× | 4.82 → 4.61 | 12.06 → 10.31 | 2,062/2,200 | 180 |
| InsectWingbeat | 1 | 0.0149 → 0.0122 | 1.22× | 0.98 → 0.98 | 1.09 → 0.92 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 0.118 → 0.107 | 1.11× | 12.85 → 6.73 | 12.56 → 6.39 | 203/1,000 | 102 |
| InsectWingbeat | 3 | 0.550 → 0.159 | 3.46× | 17.05 → 10.17 | 18.45 → 15.25 | 344/1,000 | 136 |
| InsectWingbeat | 4 | 3.615 → 1.218 | 2.97× | 18.34 → 11.45 | 26.17 → 22.02 | 344/1,000 | 136 |
| Speech Commands | 1 | 0.0067 → 0.0057 | 1.17× | 0.61 → 0.61 | 0.67 → 0.61 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.0634 → 0.0483 | 1.31× | 5.31 → 4.60 | 5.86 → 4.86 | 458/1,000 | 96 |
| Speech Commands | 3 | 0.566 → 0.232 | 2.44× | 8.48 → 7.95 | 11.67 → 12.20 | 492/1,000 | 251 |
| Speech Commands | 4 | 1.232 → 0.526 | 2.34× | 8.10 → 7.70 | 17.09 → 18.41 | 478/1,000 | 113 |

## Speech Commands calibration

The original degree-four calibration sample was unavailable. The reconstruction uses the original training split, all 65 letters, all pure powers and 4,096 uniformly sampled distinct words at each degree from two through four. The generator is NumPy PCG64 with seed 311923. Over the resulting 12,484 distinct words, each penalty is 0.01 times the largest absolute correlation between the factorial-scaled signature coordinate and the centred training indicator for that class. Recomputing the maxima with the words in sorted order reproduced the ten saved penalties. [Full-precision vector and class order](../protocol.json) · [Calibration code](../../../benchmark_prefix_suffix.py)

| Class | Penalty | Class | Penalty |
|---|---:|---|---:|
| down | 3.9297771e-05 | on | 4.2777383e-05 |
| go | 5.5683777e-05 | right | 3.2422925e-05 |
| left | 0.00012465765 | stop | 5.0925912e-05 |
| no | 6.9956242e-05 | up | 4.0414342e-05 |
| off | 3.5752232e-05 | yes | 0.00010442768 |

On this reconstruction, both builds classified **556 of 1,000** test clips correctly with **463** selected words. The paper reports **557 of 1,000** with **469** words. The other 13 paper configurations reproduced the published numbers of correct classifications and selected words.

## Implementation and trials

- The signature and tensor-log searches compute channel energies and variation bounds in sample order. A small set of high-energy channels stays in memory. When a terminal search needs neighbouring cold channels, it can recover both in one pass through the input. An active worker keeps its buffer valid while other workers recycle idle buffers.
- Double-precision SIMD reductions use four independent sums in terminal bounds and restricted Gram calculations. Exact feature correlations still use compensated summation. Tensor-log stores the norms of six-segment panels as upward-rounded 16-bit integers with a per-channel scale, so each decoded norm remains an upper bound.
- With three letters left, the tensor-log search can assign surviving Lyndon prefixes to workers. Each worker shares read-only geometry and keeps its own two-letter contraction workspace. A limit of four concurrent subtrees on the widest inputs bounds the memory cost. A prefix with too few surviving children stays on the existing traversal.
- Between restricted solves, each retained term keeps its Gram index. The solver can store the lower triangle for the surviving coefficients, dropping entries for removed terms. Subsequent solves compute Gram entries for newly active terms and verify stationarity using the fitted residual.

On the degree-four InsectWingbeat signature fit, a stricter initial screening threshold used 27.70 MiB and took 3.67 s over 11 tree passes. A time-contraction screen using 16-bit inputs reduced the number of complete terminal correlations to 7,579 and took 3.43 s. Using 8-bit candidate proposals increased the number of tree passes to seven. The retained solver uses double-precision terminal contractions and 16-bit candidate proposals. These are individual development trials. The tables use separate repeated timings. [Trial measurements](measurements.json)

## Numerical checks

All 156 measured fits converged for all ten responses at the original tolerance of `1e-8`. For each configuration, a separate evaluator reconstructed all selected columns, training and held-out scores, objective, intercept and selected-coordinate stationarity. It used small coordinate batches and expanded the tensor logarithm over contiguous word partitions. The largest selected-coordinate violation was 3.78e-09, the largest held-out score discrepancy was 2.95e-14, and the largest objective difference across the two builds and all repetitions was 2.78e-17. [Per-row audits](measurements.json)

The evaluator streamed every dictionary coordinate on the 20 configurations below the paper’s 4 GiB hypothetical design cutoff. The largest independent stationarity violation was 3.78e-09 and the largest discrepancy between independent and reported dual gaps was 1.13e-12. On the six larger dictionaries, the tree supplied the inactive-coordinate certificate. The degree-four InsectWingbeat signature fit selected 606 words from a dictionary of 1,640,402,004.

The installed checkout reproduced the support and held-out scores of all 26 audited fits. The Release build passed 200 native and 3,870 Python tests. The CPU environment skipped 373 Python tests. Regression cases cover parallel Lyndon searches, independently calculated dual gaps, subnormal variations and Gram reuse after feature removal and insertion. [Regression tests](../../../../tests/test_prefix_suffix.py)

## Reproduction

Prepare the original three datasets and splits as in the [data preparation instructions](../README.md#reproduction). To rebuild the preceding solver, apply the archived native patch in a separate checkout of `5ae817781668dffeb6d8947ba7b5bf470bc20215`. Its nine patched native files were verified against the preceding benchmark’s hashes. Build both libraries with matching Release settings. [Preceding native patch](preceding-native.patch.gz)

```sh
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data prepare \
  --experiments /path/to/sig_branch_and_bound
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data compare \
  --preceding-build /path/to/preceding-package-parent \
  --current-build /path/to/current-package-parent --output /tmp/paired-fits
```

Each build directory contains an importable `pysiglib` package and its native library. The comparison writes three per-build fits for each configuration followed by the independent numerical checks. [Benchmark runner](../../../benchmark_prefix_suffix.py)
