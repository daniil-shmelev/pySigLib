# Expanded prefix-suffix replacement benchmark

This is the historical experiment report. The later CPU replacement and its validation are recorded in [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

This study compares the current full-path implementation with the bounded two-sweep prototype. Production code and paper figures remain unchanged. A fixed block of 512 is the primary replacement candidate; blocks 32 and 128 are also measured for paths above 16 segments. The block is capped at path length. This is a comparison of whole-algorithm replacement, with no selection between the old and new algorithms.

Each device has 368 screen configurations: 23 dimension/depth pairs, segments 16/256/4096 with batches 1/32, plus segments 64/1024 with batch 8; every shape uses both dtypes. The pairs are listed in `expanded.py`. The screen uses one warm-up and three alternating samples. Twelve cases per device are repeated with three warm-ups and nine samples: four largest measured regressions, four nearest 10% overhead, and four fastest cases. Screen calls excluded between two and five seconds are also confirmed on CUDA. Samples are synchronized on CUDA. Inputs and cotangents are identical; saved final signatures are prepared outside timing. Native reconstruction and allocations are included. Timings cover untransformed paths without correction, with the scalar term included. They do not establish runtime for time augmentation, lead-lag, or corrected paths.

In the screen, an initial call above two seconds is recorded but excluded from repeated timing. Confirmation uses a five-second limit. Errors, allocation failures, and excluded calls are listed in `expanded_exclusions.json`. Such calls do not enter the runtime statistics. Raw JSON includes their first-call times and gradient checks when available. CPU float64 production prefix-suffix is the gradient reference on the same rounded inputs. Fixed pointwise tolerances are 3e-5 for float32 and 1e-10 for float64. Performance statistics include measured cases that fail these tolerances; failures are counted separately.

Workspace is native algorithm-owned numeric temporary storage, excluding input and output arrays, registers, and opaque CPU BLAS/TBB allocations. CUDA includes its explicit 4 MiB cuBLAS workspace. CUDA baseline buffers retain the production cache; prototype buffers are allocated and freed each call. The measured peaks are not total process or GPU memory.

## Whole-algorithm replacement: block 512

Each row weights measured configurations equally. Ratios above one are slowdowns. Median and 90th-percentile ratios do not represent a distribution of real user workloads.

| Device | Timed cases | Median ratio | 90th percentile | Worst ratio | Within 10% | Median workspace saving | Accuracy failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 368/368 | 1.04 | 1.22 | 1.71 | 230/368 | 43.4% | 15 |
| CUDA | 367/368 | 1.88 | 8.64 | 92.05 | 42/367 | 30.3% | 14 |

## Runtime by segments

| Device | Segments | Cases | Median ratio | 90th percentile | Worst ratio |
|---|---|---:|---:|---:|---:|
| CPU | 16 | 92 | 1.08 | 1.20 | 1.60 |
| CPU | 64 | 46 | 1.10 | 1.18 | 1.22 |
| CPU | 256 | 92 | 1.11 | 1.25 | 1.71 |
| CPU | 1024 | 46 | 0.96 | 1.18 | 1.29 |
| CPU | 4096 | 92 | 0.92 | 1.19 | 1.70 |
| CUDA | 16 | 92 | 1.13 | 1.61 | 2.98 |
| CUDA | 64 | 46 | 1.71 | 2.44 | 3.66 |
| CUDA | 256 | 92 | 2.42 | 6.53 | 8.22 |
| CUDA | 1024 | 46 | 3.25 | 9.99 | 18.38 |
| CUDA | 4096 | 91 | 2.56 | 42.40 | 92.05 |

## Runtime by batch

| Device | Batch | Cases | Median ratio | 90th percentile | Worst ratio |
|---|---|---:|---:|---:|---:|
| CPU | 1 | 138 | 1.06 | 1.28 | 1.70 |
| CPU | 8 | 92 | 1.05 | 1.18 | 1.29 |
| CPU | 32 | 138 | 1.01 | 1.19 | 1.71 |
| CUDA | 1 | 138 | 2.58 | 24.09 | 92.05 |
| CUDA | 8 | 92 | 2.12 | 6.68 | 18.38 |
| CUDA | 32 | 137 | 1.51 | 2.74 | 5.65 |

## Runtime by dimension

| Device | Dimension | Cases | Median ratio | 90th percentile | Worst ratio |
|---|---|---:|---:|---:|---:|
| CPU | 2 | 112 | 1.06 | 1.22 | 1.71 |
| CPU | 3 | 80 | 1.06 | 1.22 | 1.70 |
| CPU | 4 | 64 | 1.01 | 1.21 | 1.52 |
| CPU | 8 | 64 | 1.00 | 1.22 | 1.39 |
| CPU | 16 | 48 | 1.01 | 1.16 | 1.28 |
| CUDA | 2 | 112 | 2.02 | 12.47 | 92.05 |
| CUDA | 3 | 79 | 1.89 | 8.19 | 50.78 |
| CUDA | 4 | 64 | 1.83 | 6.91 | 46.76 |
| CUDA | 8 | 64 | 1.82 | 4.76 | 25.47 |
| CUDA | 16 | 48 | 1.96 | 4.95 | 33.12 |

## Runtime by depth

| Device | Depth | Cases | Median ratio | 90th percentile | Worst ratio |
|---|---|---:|---:|---:|---:|
| CPU | 2 | 80 | 1.09 | 1.27 | 1.71 |
| CPU | 3 | 32 | 1.03 | 1.19 | 1.28 |
| CPU | 4 | 80 | 1.03 | 1.22 | 1.70 |
| CPU | 5 | 16 | 0.93 | 1.14 | 1.26 |
| CPU | 6 | 48 | 1.06 | 1.19 | 1.50 |
| CPU | 8 | 48 | 1.01 | 1.16 | 1.41 |
| CPU | 10 | 32 | 1.00 | 1.17 | 1.60 |
| CPU | 12 | 16 | 1.06 | 1.16 | 1.28 |
| CPU | 14 | 16 | 1.01 | 1.16 | 1.22 |
| CUDA | 2 | 80 | 1.55 | 2.49 | 2.69 |
| CUDA | 3 | 32 | 2.27 | 18.13 | 33.12 |
| CUDA | 4 | 80 | 2.45 | 12.39 | 58.11 |
| CUDA | 5 | 16 | 1.69 | 3.69 | 9.33 |
| CUDA | 6 | 48 | 2.05 | 16.88 | 92.05 |
| CUDA | 8 | 48 | 1.86 | 10.99 | 70.17 |
| CUDA | 10 | 31 | 1.88 | 7.13 | 38.01 |
| CUDA | 12 | 16 | 1.70 | 9.77 | 22.93 |
| CUDA | 14 | 16 | 1.55 | 8.28 | 18.66 |

## Runtime by dtype

| Device | Dtype | Cases | Median ratio | 90th percentile | Worst ratio |
|---|---|---:|---:|---:|---:|
| CPU | float32 | 184 | 1.06 | 1.23 | 1.71 |
| CPU | float64 | 184 | 1.00 | 1.18 | 1.31 |
| CUDA | float32 | 184 | 2.06 | 12.35 | 92.05 |
| CUDA | float64 | 183 | 1.82 | 6.74 | 64.63 |

## Block size comparison

| Device | Block | Timed cases | Median ratio | 90th percentile | Median workspace saving |
|---|---:|---:|---:|---:|---:|
| CPU | 32 | 276 | 1.16 | 2.43 | 95.5% |
| CPU | 128 | 276 | 1.07 | 1.33 | 82.9% |
| CPU | 512 | 276 | 1.02 | 1.22 | 57.1% |
| CUDA | 32 | 275 | 3.13 | 16.01 | 74.0% |
| CUDA | 128 | 275 | 2.36 | 13.16 | 63.9% |
| CUDA | 512 | 275 | 2.39 | 12.99 | 40.4% |

## Confirmation and limits

- CPU: 368 screened cases and 12 confirmation cases. See `expanded_summary.csv` for every method, case, timing, workspace, and accuracy result.
- CUDA: 368 screened cases and 13 confirmation cases. See `expanded_summary.csv` for every method, case, timing, workspace, and accuracy result.
- There are 3 method or case exclusions across both stages. The full list is `expanded_exclusions.json`. These cases must not be treated as successful timings.
- Direct reversal can amplify floating-point error on long paths. An accuracy failure remains a failure even if its timing is favorable. Review raw errors before a production replacement.

Accuracy of the baseline and block-512 candidate on the screen:

| Device | Both pass | Baseline only fails | Candidate only fails | Both fail | Unchecked |
|---|---:|---:|---:|---:|---:|
| CPU | 353 | 0 | 10 | 5 | 0 |
| CUDA | 354 | 0 | 11 | 3 | 0 |

Largest confirmed block-512 slowdowns:

| Device | (d,m), segments, batch, dtype | Old ms | New ms | Ratio | Gradient passes |
|---|---|---:|---:|---:|---|
| CPU | (3,4), 4096, 1, float32 | 2.030 | 3.172 | 1.56 | True |
| CPU | (3,10), 16, 1, float32 | 0.768 | 0.981 | 1.28 | True |
| CPU | (2,2), 256, 32, float32 | 0.220 | 0.279 | 1.27 | True |
| CPU | (3,2), 256, 1, float32 | 0.035 | 0.042 | 1.17 | True |
| CUDA | (2,6), 4096, 1, float32 | 1.400 | 124.514 | 88.92 | True |
| CUDA | (2,8), 4096, 1, float32 | 2.612 | 183.183 | 70.14 | True |
| CUDA | (2,6), 4096, 1, float64 | 3.269 | 212.549 | 65.01 | True |
| CUDA | (2,4), 4096, 1, float32 | 0.785 | 44.016 | 56.10 | True |

## Reproduction

From the pySigLib root, after building the experiment DLLs as described in `REPORT.md`:

```powershell
foreach ($device in @('cpu', 'cuda')) {
    python benchmarks/reversal/expanded.py --device $device --stage screen --output tmp/reversible_backward_20261110/expanded_$device.json
    python benchmarks/reversal/expanded.py --device $device --stage confirm --output tmp/reversible_backward_20261110/expanded_$device.json
}
python benchmarks/reversal/summarize_expanded.py --results-dir tmp/reversible_backward_20261110 --report benchmarks/reversal/EXPANDED_REPORT.md
```

The JSON files retain the individual samples, errors, exclusions, settings, and native library hashes. Re-running a stage resumes at the first missing configuration.
