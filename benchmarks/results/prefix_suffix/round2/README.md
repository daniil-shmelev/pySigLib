# Multi-response branch and bound on the audio experiments

A subsequent comparison measures the next set of runtime and memory improvements on the same 26 configurations. [Latest benchmark](../round3/README.md)

The new solver cut the combined fit time for the 14 paper configurations by a factor of **2.05**. Median fit time and tracked native peak fell on all 26 configurations, including the additional tensor-log fits. The increase in process peak RSS fell on 21 configurations, including every degree-three and degree-four fit. All six fits per configuration selected the same words and assigned the same class to every test recording. [Individual measurements and certificates](measurements.json)

Each entry compares the preceding optimized build with the new build on the same Apple M3, using three fresh processes per build, eight feature workers and one BLAS thread. Build order alternated and the second repetition traversed the configurations in reverse. Each fit covers ten responses and 600 training paths. Data preparation, penalty calibration, held-out prediction and the independent checks were outside the timed call. Native peak counts tracked allocations, including returned arrays. The RSS column gives the rise in the process high-water mark during the fit, measured after loading the training paths. [Preceding benchmark](../README.md)

## Signature fits from the paper

| Dataset | Degree | Fit s, preceding → new | Speedup | Native MiB, preceding → new | RSS increase MiB, preceding → new | Test correct | Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.0032 → 0.0025 | 1.32× | 0.29 → 0.23 | 0.30 → 0.31 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.0696 → 0.0410 | 1.70× | 5.59 → 5.48 | 5.98 → 5.88 | 1,907/2,200 | 153 |
| Spoken Arabic Digit | 3 | 0.130 → 0.0699 | 1.87× | 9.00 → 8.72 | 12.38 → 8.61 | 2,038/2,200 | 204 |
| Spoken Arabic Digit | 4 | 0.993 → 0.610 | 1.63× | 17.32 → 15.81 | 29.67 → 25.94 | 2,093/2,200 | 611 |
| InsectWingbeat | 1 | 0.0165 → 0.0140 | 1.18× | 1.03 → 0.98 | 1.00 → 0.88 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 0.319 → 0.0993 | 3.21× | 31.63 → 24.26 | 31.39 → 24.36 | 346/1,000 | 300 |
| InsectWingbeat | 3 | 0.900 → 0.385 | 2.34× | 42.73 → 35.65 | 46.73 → 39.28 | 375/1,000 | 504 |
| InsectWingbeat | 4 | 5.726 → 2.389 | 2.40× | 47.43 → 39.87 | 54.03 → 45.09 | 397/1,000 | 606 |
| Speech Commands | 1 | 0.0108 → 0.0061 | 1.78× | 0.66 → 0.61 | 0.64 → 0.56 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.132 → 0.100 | 1.32× | 13.24 → 10.21 | 13.64 → 10.25 | 463/1,000 | 183 |
| Speech Commands | 3 | 0.937 → 0.783 | 1.20× | 21.51 → 20.73 | 34.86 → 31.73 | 519/1,000 | 633 |
| Speech Commands | 4 | 1.658 → 0.850 | 1.95× | 23.55 → 21.01 | 36.22 → 28.22 | 556/1,000 | 463 |

## Arabic degree-four dictionary comparison from the paper

Each class used 0.03 times its largest absolute centred training correlation over the respective dictionary. Across the two dictionaries the penalty vectors differ. [Penalty vectors](../protocol.json)

| Dictionary | Fit s, preceding → new | Speedup | Native MiB, preceding → new | RSS increase MiB, preceding → new | Test correct | Words |
|---|---:|---:|---:|---:|---:|---:|
| Signature | 0.239 → 0.102 | 2.35× | 12.08 → 11.65 | 16.67 → 13.16 | 1,969/2,200 | 114 |
| Tensor log | 0.296 → 0.126 | 2.34× | 5.09 → 4.05 | 17.39 → 8.89 | 1,937/2,200 | 46 |

## Additional tensor-log fits

These fits use the signature experiment’s per-class penalty vectors at each degree. They measure the tensor-log search on the same three data splits. The response fits use the tensor-log coefficients of Lyndon words.

| Dataset | Degree | Fit s, preceding → new | Speedup | Native MiB, preceding → new | RSS increase MiB, preceding → new | Test correct | Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.0032 → 0.0025 | 1.28× | 0.29 → 0.23 | 0.33 → 0.34 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.0387 → 0.0224 | 1.73× | 3.22 → 2.91 | 3.00 → 3.20 | 1,916/2,200 | 75 |
| Spoken Arabic Digit | 3 | 0.107 → 0.0755 | 1.42× | 3.79 → 3.11 | 6.89 → 3.94 | 1,993/2,200 | 73 |
| Spoken Arabic Digit | 4 | 0.797 → 0.493 | 1.62× | 6.37 → 4.82 | 19.03 → 10.72 | 2,062/2,200 | 180 |
| InsectWingbeat | 1 | 0.0194 → 0.0145 | 1.34× | 1.03 → 0.98 | 0.73 → 1.17 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 0.233 → 0.119 | 1.95× | 24.77 → 12.85 | 25.05 → 12.50 | 203/1,000 | 102 |
| InsectWingbeat | 3 | 1.450 → 0.538 | 2.69× | 25.60 → 17.05 | 29.36 → 17.33 | 344/1,000 | 136 |
| InsectWingbeat | 4 | 15.192 → 3.577 | 4.25× | 26.97 → 18.34 | 39.17 → 22.03 | 344/1,000 | 136 |
| Speech Commands | 1 | 0.0108 → 0.0071 | 1.52× | 0.66 → 0.61 | 0.47 → 0.64 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.0923 → 0.0615 | 1.50× | 9.16 → 5.31 | 9.03 → 5.62 | 458/1,000 | 96 |
| Speech Commands | 3 | 0.939 → 0.574 | 1.64× | 10.80 → 8.48 | 17.22 → 12.84 | 492/1,000 | 251 |
| Speech Commands | 4 | 3.427 → 1.180 | 2.90× | 11.21 → 8.10 | 23.30 → 15.31 | 478/1,000 | 113 |

Process peak RSS increased on 5 configurations, by 0.02 to 0.44 MiB. The primary degree-one Arabic signature medians were 3.24 and 2.46 ms. A separate set of 12 alternating fresh-process pairs gave 3.19 and 2.48 ms. The tables retain the original three repetitions for every configuration.

## Speech Commands degree-four reconstruction

The experiment archive supplied the split and preparation code. The paper reports an all-degree sampled calibration and a multiplier of 0.01. To reconstruct the missing sample, the benchmark uses NumPy PCG64 with seed 311923, includes all 65 letters and all pure powers, and draws 4,096 distinct words uniformly at each of degrees two through four. Their union contains 12,484 distinct words. For each class the reconstructed penalty is 0.01 times the maximum absolute correlation between those scaled signature columns and the centred training indicator. The ten values in class order are stored at full precision. [Protocol and class order](../protocol.json) · [Calibration implementation](../../../benchmark_prefix_suffix.py)

Both builds scored **556/1,000 with 463 selected words**, compared with **557/1,000 and 469 words** in the paper. The remaining 11 signature rows and both Arabic dictionary-comparison rows reproduce the paper’s correct counts and support sizes.

## How the search changed

- Each response keeps its own restricted solve, residual and stopping test. When several responses need a tree search, they share a traversal. For residuals `r_t` and thresholds `h_t`, set `h = min(h_t)`. The common envelope at sample `i` is `max_t |r_t[i]| h/h_t`. A positive upper bound below `h` rules out a branch for every waiting response. The implementation pads the envelope for roundoff and tests each surviving word against each response’s threshold. For the dual gap it tracks correlations above the exact penalty, including values within the activation tolerance.
- The signature search refines terminal bounds over groups of samples and responses. The tensor-log search contracts the final two letters directly, restricts extensions using the Lyndon prefix period, and can stop a signed sum once the remaining samples cannot change the decision. On degree-four InsectWingbeat, signature tree traversals fell from 43 to 6 and tensor-log traversals from 23 to 3.
- A bounded shortlist retains 16-bit candidate values with a measured rounding error. These copies prioritise proposals. When a word enters a restricted problem, the solver gets a double-precision column from another response or reconstructs it using Chen’s identity. Tensor-log reconstruction needs only the contiguous subwords of that word. The restricted solver shares live columns and a bounded cache of Gram entries across responses, and accepts a Cholesky step only after checking the penalised objective decreases.
- Wide inputs are normalised a channel at a time. The cache retains high-energy channels and recycles free buffers for other channels. Every channel remains eligible for search. Each worker reuses its terminal buffers. The batched path covers default float64 multi-response squared fits at degrees two through four. Other supported configurations use individual tree traversals.

During development, a coarse shared traversal took 3.78 s for the degree-four InsectWingbeat signature fit and used 52.90 MiB of tracked memory. A trial that precomputed candidate geometry took 6.76 s. Response-specific sample-group bounds brought the fit to 2.38 s. A later prototype held about 82 MiB of additional process memory on this fit despite its lower tracked peak. Streaming the input channels and reusing terminal buffers brought the final three-run median to 45.09 MiB. The tables give the repeated comparisons for the retained implementation.

## Independent checks

The fit used the tree and recorded zero full-transform calls in all 156 measured runs. For every configuration, the second new-build fit was checked by recomputing selected training columns, training and test scores, objective, intercept stationarity and selected-coordinate stationarity. The evaluator requests batches of signature coordinates and evaluates the tensor logarithm by summing products over contiguous partitions. Its largest held-out score discrepancy was 2.92 × 10⁻¹⁴. Across both builds and all repetitions, the largest difference from the objective of the audited fit was 6.74 × 10⁻¹³. [Per-row checks](measurements.json)

The evaluator also checked every dictionary coordinate on the 20 configurations whose hypothetical training design fit within the paper’s 4 GiB cutoff. The largest independent stationarity violation was 3.78 × 10⁻⁹ against the requested tolerance of 10⁻⁸. The maximum difference between independent and reported dual gaps on these 20 rows was 1.13 × 10⁻¹². On the six larger dictionaries the tree supplied the inactive-coordinate certificate. The largest signature dictionary contained 1,640,402,004 words, equivalent to a 7.16 TiB dense training matrix, and the fit retained 606.

The installed checkout reproduced the support and held-out scores of all 26 audited models. Recalculating the Speech Commands penalties from the training paths, with the sampled words visited in sorted order, reproduced all ten values. The Release CPU build passed 200 native and 3,865 Python tests, with 373 Python tests skipped in the CPU environment.

## Reproduction

Prepare the three archives and original splits as described in the [data preparation instructions](../README.md#reproduction). Build the current checkout and the preceding CPU library with matching Release settings. For the preceding library, apply the archived native-source patch to a separate checkout of `5ae817781668dffeb6d8947ba7b5bf470bc20215` before building. The patch reproduces all eight native files modified before this comparison. The measurements include source and library hashes. [Preceding native patch](preceding-native.patch.gz)

```sh
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data prepare \
  --experiments /path/to/sig_branch_and_bound
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data compare \
  --preceding-build /path/to/preceding-package-parent \
  --current-build /path/to/current-package-parent --output /tmp/paired-fits
```

Each build directory contains its own importable `pysiglib` package and native library. The compare command saves all 156 fits, including their predictions, followed by the 26 independent audits. [Benchmark runner](../../../benchmark_prefix_suffix.py)
