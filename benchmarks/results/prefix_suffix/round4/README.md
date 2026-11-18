# Signature and tensor-log branch-and-bound benchmarks

On the 14 configurations in the paper, the sum of median fit times fell from **4.43 s to 3.42 s**, a **1.29× speedup** over the preceding optimized solver. Both builds produced the same selected words and assigned the same class to every held-out recording across all three repetitions of all 26 configurations. The fits used the tree throughout and recorded zero full-transform calls. [Individual measurements and numerical checks](measurements.json)

Including the 12 additional tensor-log experiments, the speedup in summed median times was 1.29× for signatures and 1.44× for tensor logs. Two InsectWingbeat tensor-log fits took longer. The degree-two median rose from 0.102 to 0.103 s and the degree-three median from 0.153 to 0.163 s. Tracked native peak memory fell on 17 configurations, tied on six and rose on three. The largest increase was 0.72 MiB on the Arabic degree-four tensor-log classification fit. Additional process peak RSS fell on 21 configurations and rose on five.

## Paper experiments

Each fit used the same split, per-class penalties, 600 training paths and ten responses in both builds. Timing and memory entries are medians of three fresh-process fits on an eight-core Apple M3 with eight feature workers and one BLAS thread. Build order alternated between configurations, and the second repetition ran the configurations in reverse. Fit time excludes dataset preparation, penalty calibration, test prediction and independent checks. Native MiB measures the fit's tracked high-water allocation, including output arrays. Additional RSS measures the rise in the process high-water mark from just before the fit. The machine, compiler, binary hashes and source hashes are recorded with the results. [Frozen protocol](../protocol.json) · [Preceding benchmark](../round3/README.md)

### Signature classification

| Dataset | Degree | Fit s, preceding → current | Speedup | Native MiB, preceding → current | Additional RSS MiB, preceding → current | Test correct | Selected words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.0024 → 0.0022 | 1.12× | 0.23 → 0.23 | 0.38 → 0.33 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.0322 → 0.0283 | 1.14× | 5.38 → 4.94 | 5.94 → 4.45 | 1,907/2,200 | 153 |
| Spoken Arabic Digit | 3 | 0.0553 → 0.0498 | 1.11× | 8.63 → 7.87 | 9.20 → 8.73 | 2,038/2,200 | 204 |
| Spoken Arabic Digit | 4 | 0.502 → 0.406 | 1.24× | 15.26 → 15.13 | 24.91 → 22.86 | 2,093/2,200 | 611 |
| InsectWingbeat | 1 | 0.0115 → 0.0099 | 1.15× | 0.98 → 0.98 | 0.52 → 1.39 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 0.0965 → 0.0931 | 1.04× | 14.71 → 14.22 | 20.98 → 20.09 | 346/1,000 | 300 |
| InsectWingbeat | 3 | 0.291 → 0.251 | 1.16× | 25.91 → 20.31 | 34.28 → 24.62 | 375/1,000 | 504 |
| InsectWingbeat | 4 | 1.727 → 1.225 | 1.41× | 30.07 → 24.18 | 39.75 → 33.41 | 397/1,000 | 606 |
| Speech Commands | 1 | 0.0055 → 0.0046 | 1.19× | 0.61 → 0.61 | 0.56 → 0.62 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.0780 → 0.0730 | 1.07× | 9.37 → 9.04 | 12.12 → 13.12 | 463/1,000 | 183 |
| Speech Commands | 3 | 0.674 → 0.630 | 1.07× | 19.00 → 16.85 | 30.69 → 29.27 | 519/1,000 | 633 |
| Speech Commands | 4 | 0.750 → 0.500 | 1.50× | 19.86 → 18.27 | 27.70 → 27.53 | 556/1,000 | 463 |

### Degree-four dictionary comparison on Spoken Arabic Digit

Each response used 0.03 times its maximum centred training correlation over the respective dictionary. This separate calibration accounts for the difference from the Arabic classification row above. [Per-response penalties](../protocol.json)

| Dictionary | Fit s, preceding → current | Speedup | Native MiB, preceding → current | Additional RSS MiB, preceding → current | Test correct | Selected words |
|---|---:|---:|---:|---:|---:|---:|
| Signature | 0.0856 → 0.0677 | 1.26× | 11.61 → 11.19 | 13.55 → 12.81 | 1,969/2,200 | 114 |
| Tensor log | 0.115 → 0.0824 | 1.39× | 3.94 → 3.99 | 8.38 → 7.95 | 1,937/2,200 | 46 |

## Additional tensor-log experiments

These fits use the classification rows’ penalty vectors on the same paths and labels. The dictionary consists of tensor-log coefficients indexed by Lyndon words.

| Dataset | Degree | Fit s, preceding → current | Speedup | Native MiB, preceding → current | Additional RSS MiB, preceding → current | Test correct | Selected words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.0024 → 0.0022 | 1.06× | 0.23 → 0.23 | 0.44 → 0.34 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.0213 → 0.0204 | 1.04× | 2.51 → 1.83 | 3.30 → 1.77 | 1,916/2,200 | 75 |
| Spoken Arabic Digit | 3 | 0.0688 → 0.0447 | 1.54× | 2.90 → 3.04 | 2.47 → 2.59 | 1,993/2,200 | 73 |
| Spoken Arabic Digit | 4 | 0.459 → 0.355 | 1.29× | 4.61 → 5.33 | 9.22 → 11.64 | 2,062/2,200 | 180 |
| InsectWingbeat | 1 | 0.0122 → 0.0120 | 1.02× | 0.98 → 0.98 | 1.05 → 0.91 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 0.102 → 0.103 | 0.99× | 6.73 → 5.32 | 6.58 → 4.72 | 203/1,000 | 102 |
| InsectWingbeat | 3 | 0.153 → 0.163 | 0.93× | 10.17 → 8.87 | 13.95 → 12.69 | 344/1,000 | 136 |
| InsectWingbeat | 4 | 1.209 → 0.700 | 1.73× | 11.45 → 10.06 | 23.31 → 21.84 | 344/1,000 | 136 |
| Speech Commands | 1 | 0.0062 → 0.0055 | 1.12× | 0.61 → 0.61 | 0.61 → 0.58 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.0516 → 0.0494 | 1.04× | 4.60 → 3.47 | 4.72 → 3.77 | 458/1,000 | 96 |
| Speech Commands | 3 | 0.267 → 0.204 | 1.31× | 7.95 → 7.63 | 13.27 → 11.36 | 492/1,000 | 251 |
| Speech Commands | 4 | 0.521 → 0.331 | 1.57× | 7.70 → 7.19 | 16.30 → 16.09 | 478/1,000 | 113 |

## Speech Commands calibration

The original degree-four calibration sample was unavailable. The reconstruction uses the original training split, all 65 letters, all pure powers and 4,096 uniformly sampled distinct words at each degree from two through four. The generator is NumPy PCG64 with seed 311923. Over the resulting 12,484 distinct words, each penalty is 0.01 times the largest absolute correlation between the factorial-scaled signature coordinate and the centred training indicator for that class. Recomputing the maxima with the words in sorted order reproduced all ten reconstructed penalties exactly. [Full-precision vector and class order](../protocol.json) · [Calibration code](../../../benchmark_prefix_suffix.py)

On this reconstruction, both builds classified **556 of 1,000** test clips correctly with **463** selected words. The paper reports **557 of 1,000** with **469** words. The other 13 paper configurations reproduced the published numbers of correct classifications and selected words.

## Implementation and trials

- The search starts some multi-response fits with a dictionary truncated one degree earlier. When lower-degree candidates run out, it searches the missing degree at the same residual and combines the two bounds. Subsequent searches cover the full dictionary. Iteration-limited fits also check the full dictionary.
- For signatures, workers share the root integrals and store time-panel maxima as upward-rounded bfloat16. A bound calculated once per path is tested against each target’s own scaled residual. Exact terminal correlations use compensated SIMD sums.
- For tensor logs, the search stores upward-rounded 8-bit panel norms with a per-channel scale and 32-bit whole-channel norms. Cauchy bounds on groups of admissible Lyndon endings can rule out every one- or two-letter completion before reconstructing channel increments. Degree-one and degree-two prefix integrals have direct formulas.
- The restricted solver reuses correlations from its latest exact gradient refresh. Once coefficient signs stabilize, it can attempt the safeguarded Cholesky step earlier and only fill the triangle used by the factorization. Joint solves use tighter internal refinement for weakly active coordinates and return the requested convergence certificate.

Individual development trials exposed a memory–runtime tradeoff. Keeping 32 warm signature channels used 22.87 MiB on degree-four InsectWingbeat and took 1.373 s. The 48-channel pilot used 24.11 MiB and took 1.215 s. Seeding with the two highest-energy channels took 1.317 s for the signature and 0.981 s for the tensor log. In a three-run comparison, going straight to degree three on the wide InsectWingbeat tensor-log fit reduced the median from 0.199 s to 0.174 s. On the narrower Arabic degree-four tensor-log fit, using six responses per traversal reduced the native peak from 5.33 to 4.92 MiB but raised the median from 0.356 s to 0.446 s. The retained solver uses direct search for the first case and wider response batches for the second. The tables report separate runs of the retained implementation. [Trial measurements](measurements.json)

## Numerical checks

All ten responses converged in each of the 156 measured fits at the original tolerance of `1e-8`. For each configuration, a separate evaluator reconstructed all selected columns, training and held-out scores, and the objective. It checked stationarity for the intercept and each selected coordinate. It used small coordinate batches and expanded the tensor logarithm over contiguous word partitions. The largest selected-coordinate violation was 1.62e-15, the largest held-out score discrepancy was 2.95e-14, and the largest objective difference across the two builds and all repetitions was 4.34e-15. [Per-row audits](measurements.json)

The evaluator streamed every dictionary coordinate on the 20 configurations below the paper’s 4 GiB hypothetical design cutoff. The largest independent stationarity violation was 1.65e-15 and the largest discrepancy between independent and reported dual gaps was 1.14e-12. On the six larger dictionaries, the tree supplied the inactive-coordinate certificate. The degree-four InsectWingbeat signature fit selected 606 words from a dictionary of 1,640,402,004.

The installed checkout reproduced the support and held-out scores of all 26 audited fits. The Release build passed 200 native and 3,874 Python tests. The CPU environment skipped 373 Python tests. Regression cases cover a target whose only signal is at the highest degree, full-dictionary certificates at the iteration limit, tensor-log searches and independently calculated dual gaps. [Regression tests](../../../../tests/test_prefix_suffix.py)

## Reproduction

Prepare the original three datasets and splits, then check out `8f38d775d0feb341b7c62bacbc14967ad4dead27` in a separate directory for the preceding solver. Use matching Release settings when building both packages. [Data preparation instructions](../README.md#reproduction)

```sh
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data prepare \
  --experiments /path/to/sig_branch_and_bound
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data compare \
  --preceding-build /path/to/preceding-package-parent \
  --current-build /path/to/current-package-parent --output /tmp/paired-fits
```

Each build directory contains an importable `pysiglib` package and its native library. The comparison writes three per-build fits for each configuration followed by the independent numerical checks. [Benchmark runner](../../../benchmark_prefix_suffix.py)
