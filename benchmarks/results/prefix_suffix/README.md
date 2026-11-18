# CPU prefix–suffix search on the audio experiments

The subsequent [multi-response benchmarks](round2/README.md) compare the next implementation with the optimized build measured here.

Each fit uses all ten responses on the original split. For the paper configurations, fit times and tracked native peaks are medians of three fresh processes per build. The original build is `5ae8177`. Both builds used the same reconstructed protocol, Apple M3 CPU, eight workers and one BLAS thread. Native peak includes the solver workspace and returned arrays. The measurement file also records the rise in process peak RSS, each individual timing, prediction time, support count and certificate. [Measurements](measurements.json) · [Protocol and penalties](protocol.json)

## Signature fits

| Dataset | Degree | Original s | Optimized s | Speedup | Original MiB | Optimized MiB | Test correct | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.008 | 0.004 | 2.3× | 6.42 | 0.29 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.131 | 0.071 | 1.9× | 13.34 | 5.52 | 1,907/2,200 | 153 |
| Spoken Arabic Digit | 3 | 0.607 | 0.131 | 4.6× | 27.39 | 9.00 | 2,038/2,200 | 204 |
| Spoken Arabic Digit | 4 | 5.308 | 1.011 | 5.3× | 53.84 | 16.84 | 2,093/2,200 | 611 |
| InsectWingbeat | 1 | 0.071 | 0.016 | 4.4× | 74.67 | 1.03 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 2.335 | 0.325 | 7.2× | 81.79 | 31.62 | 346/1,000 | 300 |
| InsectWingbeat | 3 | 8.623 | 0.883 | 9.8× | 104.75 | 42.73 | 375/1,000 | 504 |
| InsectWingbeat | 4 | 34.411 | 6.145 | 5.6× | 125.06 | 47.43 | 397/1,000 | 606 |
| Speech Commands | 1 | 0.031 | 0.010 | 3.0× | 24.90 | 0.66 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.678 | 0.136 | 5.0× | 35.29 | 13.24 | 463/1,000 | 183 |
| Speech Commands | 3 | 9.256 | 0.966 | 9.6× | 51.98 | 21.51 | 519/1,000 | 633 |
| Speech Commands | 4 | 8.834 | 1.600 | 5.5× | 79.78 | 23.55 | 556/1,000 | 463 |

## Dictionary comparison on Spoken Arabic Digit

These degree-four rows use a separate penalty for each response and dictionary, equal to 0.03 times the largest centred training correlation over that dictionary. Calibration and test prediction are outside fit time.

| Dictionary | Original s | Optimized s | Speedup | Original MiB | Optimized MiB | Test correct | Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Signature | 0.958 | 0.246 | 3.9× | 47.18 | 12.09 | 1,969/2,200 | 114 |
| Tensor log | 1.266 | 0.311 | 4.1× | 5.74 | 5.09 | 1,937/2,200 | 46 |

The median rise in process peak RSS also fell on every paper configuration. On InsectWingbeat degree four it fell from 198.16 to 54.45 MiB. On the Arabic tensor-log comparison it fell from 19.47 to 16.66 MiB. Median held-out prediction speedups across the 14 rows ranged from 1.8× to 3.8×. [Individual process and prediction measurements](measurements.json)

## Reconstruction and checks

The archive checksums, split ID checksums, class order and penalty vectors are recorded for all three datasets. For Speech Commands degree four, the original all-degree calibration sample was unavailable. The reconstruction uses all 65 letters, all pure powers, and up to 4,096 distinct words sampled uniformly at each degree from two to four with NumPy PCG64 seed 311923. Each penalty is 0.01 times the maximum absolute centred **training** correlation over that union. [Reconstructed protocol](protocol.json)

The other 11 signature rows and both Arabic dictionary-comparison rows reproduce the published correct counts and support sizes. On the reconstructed Speech Commands row both builds give 556 correct, versus 557 in the paper. The original build selects 462 words, the optimized build selects 463, and the paper reports 469. The additional word in the optimized fit is `(50, 64, 64)`, with coefficient magnitude 5.34e-05. All six measured fits for each paper row agree on every held-out class label.

All 140 response fits passed the solver’s stationarity checks and recorded zero full-transform calls. The independent evaluator uses selected signature coordinates and expands the tensor logarithm over ordered contiguous subwords. Across the 14 paper configurations, the maximum training-column error was 1.89e-15, the maximum held-out score error was 1.28e-14, and the maximum objective error was 2.08e-17. The largest test-score difference between the original and optimized builds was 3.62e-04.

The nine feasible signature rows and both Arabic dictionary-comparison rows were checked against every dictionary coordinate in batches. The largest independently measured KKT violation was 4.00e-09. For the three larger signature dictionaries the independent audit covered every selected column, the stationarity condition for each selected coefficient in each response, the intercept, and training and test scores. The inactive search certificate came from the tree. [Per-row audit results](measurements.json)

## Search changes

- Signature prefixes retain the segment start and scaled monomial coefficients. Only reachable suffix-envelope entries are stored. Segment panels share an upper bound on absolute increments. With one or two letters remaining, the search constructs the integrals needed for the signed correlation directly.
- For the last two letters, each sample has a time-by-time operator. The code bounds its action using its Frobenius norm and channel increment norms, then refines on disjoint sample groups before evaluating surviving signed correlations. Tensor-log uses the additional displacement term from the log-prefix recurrence and the prefix period to enumerate admissible Lyndon extensions.
- For wide log dictionaries, grouped Hölder bounds can reject a prefix before its segment polynomial is constructed. For small alphabets, up to 1 MiB of centred column-radius summaries can be reused across residuals. Each summary is a scalar calculated while streaming a coordinate.
- Targets reuse columns from earlier sparse supports. A safeguarded Cholesky step accelerates squared-loss solves once the signs stabilise. Prediction walks the selected prefixes in order and processes samples in tiles. The optimized degree-four signature predictor handled all 1,000 InsectWingbeat test paths in one call under a 4 MiB native limit. The tensor-log predictor handled all 2,200 Arabic test paths under the same limit. The original build exceeded the limit in both cases.

Development trials on InsectWingbeat degree four measured 11.31 s for the first fused terminal search, 8.77 s after constructing only terminal moments and switching to one-node traversal, then 6.05 s after adding per-group Cauchy bounds for each remaining letter. Reducing the parallel frontier from 32 partitions to 16 took 12.35 s at the middle stage. These were individual tuning runs. The tables above use separate repeated measurements.

## Additional tensor-log fits

Each row below is one optimized ten-response run with the signature experiment’s penalty vector. Two matched original-build runs measured 0.886 s and 45.44 MiB on InsectWingbeat degree two, and 5.880 s and 14.90 MiB on Speech Commands degree three. The corresponding optimized fits were 3.8× and 6.1× faster. The calibration differs from the paper’s Arabic dictionary comparison.

| Dataset | Degree | Fit s | Native MiB | Test correct | Words |
|---|---:|---:|---:|---:|---:|
| Spoken Arabic Digit | 1 | 0.006 | 0.29 | 1,335/2,200 | 14 |
| Spoken Arabic Digit | 2 | 0.040 | 3.22 | 1,916/2,200 | 75 |
| Spoken Arabic Digit | 3 | 0.109 | 3.79 | 1,993/2,200 | 73 |
| Spoken Arabic Digit | 4 | 0.819 | 6.37 | 2,062/2,200 | 180 |
| InsectWingbeat | 1 | 0.019 | 1.03 | 197/1,000 | 79 |
| InsectWingbeat | 2 | 0.236 | 24.77 | 203/1,000 | 102 |
| InsectWingbeat | 3 | 1.476 | 25.60 | 344/1,000 | 136 |
| InsectWingbeat | 4 | 15.503 | 26.97 | 344/1,000 | 136 |
| Speech Commands | 1 | 0.011 | 0.66 | 312/1,000 | 50 |
| Speech Commands | 2 | 0.095 | 9.16 | 458/1,000 | 96 |
| Speech Commands | 3 | 0.966 | 10.80 | 492/1,000 | 251 |
| Speech Commands | 4 | 3.650 | 11.21 | 478/1,000 | 113 |

## Software checks

The final library in the checkout reproduced the measured support and test predictions on all 14 paper configurations. The native Release build passed all 200 CTest cases. The Python suite passed 3,859 tests and skipped 373 in the CPU environment with the project-required Kauri 2.3.0. New regression cases cover prenecklace suffixes, grouped bounds, non-divisible sample counts, multiple responses, shared prediction prefixes and explicit memory limits. The SciPy reference fit uses a larger line-search budget to meet its existing tolerance on the Huber test.

## Reproduction

Build and install the CPU library in a separate environment for each revision. Obtain the raw archives and run the preparation commands in each dataset README at experiment revision `fef23b0`. The benchmark imports the original splits and transformations, verifies the raw archive hashes, and writes a protocol for its own prepared files. [Original experiment scripts](https://github.com/daniil-shmelev/sig_branch_and_bound/tree/fef23b053a0b2d781d0ae0975c450e696edae7f8)

From the pySigLib repository, the following commands reproduce the reconstructed Speech Commands row and its independent checks. The data and model directories can be outside the repository.

```sh
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data prepare --experiments /path/to/sig_branch_and_bound
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data fit --dataset speech --feature sig --degree 4 --output /tmp/audio-results/speech4
python benchmarks/benchmark_prefix_suffix.py --data /tmp/audio-data check --fit /tmp/audio-results/speech4 --output /tmp/audio-results/speech4-check.json
```

Use `--dictionary-comparison` with `--dataset arabic --degree 4` for the paper’s two dictionary rows. Fit each repetition in a fresh process, selecting the installed build for that run. The check command accepts `--baseline` with a second fit stem to compare held-out decisions. [Benchmark runner](../../benchmark_prefix_suffix.py)
