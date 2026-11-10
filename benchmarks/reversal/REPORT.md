# Prefix-suffix deconstruction study

This is the historical experiment report. The later CPU replacement and its validation are recorded in [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

Production prefix-suffix and its paper figures are unchanged. The prototypes use two directional sweeps and bounded blocks. Final signatures are prepared outside the timed region. The reverse sweep reconstructs prefixes and differentiates the left factors and prefix recurrence. The forward sweep reconstructs suffixes and differentiates the right factors and suffix recurrence.

The grid contains seven dimension/depth pairs, 16/256/4096 segments, batches 1/32, and float32/float64. Block 0 is the current implementation; the candidates use blocks 1, 8, 32, 128, and 512, capped at the path length. Native automatic split selection is unchanged. Each completed timing has three warm-up calls and nine alternating samples, with synchronized CUDA wall time. Reconstruction, allocation, and free costs are included. CUDA baseline buffers use the existing cache; each prototype call allocates and frees its numeric buffers.

## Memory and error definitions

Native temporary memory counts algorithm-owned numeric buffers. CPU peaks are measured from live allocations in the same path scheduler. CUDA peaks use actual reserved buffer capacity, including metadata and the explicit 4 MiB cuBLAS workspace. Inputs, output gradients, stacks/registers, library code, and opaque CPU BLAS/TBB allocations are excluded. Input and output sizes are recorded separately. Process peak working set is a cumulative process metric and is not used to claim algorithm memory savings. GPU process memory is unavailable under the Windows WDDM driver (nvidia-smi reports N/A).

Grid errors use CPU float64 production prefix-suffix on identical rounded paths and cotangents. Separate correctness checks use independent Chen autograd, all tested explicit splits, both scalar conventions, zero increments, and partial blocks. A candidate qualifies only if its gradient passes the fixed pointwise tolerance: 3e-5 for float32, 1e-10 for float64, used as both absolute and relative tolerances. Failed accuracy checks are retained in the raw results and excluded from qualifying savings.

## Savings within runtime limits

The table selects the smallest workspace for each configuration within each runtime limit. Counts require positive memory savings and valid gradients. The median and best savings are over qualifying configurations. Each device has 84 configurations.

| Device | Maximum overhead | Qualifying configurations | Median saving | Best saving |
|---|---:|---:|---:|---:|
| CPU | 5% | 48/84 | 92.59% | 99.88% |
| CPU | 10% | 58/84 | 92.40% | 99.88% |
| CPU | 25% | 77/84 | 89.93% | 99.88% |
| CUDA | 5% | 9/84 | 1.50% | 17.63% |
| CUDA | 10% | 14/84 | 2.69% | 97.69% |
| CUDA | 25% | 23/84 | 3.40% | 97.69% |

Representative best savings:

| Device | Overhead limit | (d,m), n, batch, dtype | Block | Baseline / candidate ms | Baseline / candidate MiB | Saving |
|---|---:|---|---:|---:|---:|---:|
| cpu | 5% | (2,12), 4096, 32, float64 | 8 | 538.825 / 432.536 | 1280.194 / 1.543 | 99.88% |
| cuda | 5% | (3,6), 16, 32, float64 | 32 | 0.890 / 0.907 | 6.648 / 5.477 | 17.63% |
| cuda | 10% | (3,10), 4096, 32, float64 | 32 | 1942.794 / 2101.426 | 1624.868 / 37.539 | 97.69% |

## Failures and limits

Runtime/allocation exclusions: 0. Accuracy failures among completed timings: 52, from 1008 method measurements. The complete case lists are in `prefix_suffix_exclusions.json`. The initial 10-second call limit excluded one CUDA case. That configuration was rerun with a 30-second limit; the original attempt is retained in the raw JSON retry history.

| Device | Baseline accuracy failures | Prototype accuracy failures | Runtime/allocation exclusions |
|---|---:|---:|---:|
| CPU | 1 | 25 | 0 |
| CUDA | 1 | 25 | 0 |

Direct reversal can amplify rounding error over long paths. The production baseline also fails the fixed tolerance on some long float32 cases. Both the baseline and candidate must pass before a result can qualify. A memory reduction alone is not sufficient to recommend a production change. The candidates remain experimental. Results apply to this Windows CPU/GPU and these native kernels; they are not general bounds on the best possible implementation.

## Validation and reproduction

- CPU: 402 correctness and workspace checks passed. Every block size is checked for constant workspace at three path lengths up to 4096 segments, with the shortest path at least as long as the block.
- CUDA: 402 correctness and workspace checks passed. Every block size is checked for constant workspace at three path lengths up to 4096 segments, with the shortest path at least as long as the block.

From the pySigLib root in PowerShell:

```powershell
cmake -S benchmarks/reversal -B out/build/reversal -G 'Visual Studio 17 2022' -A x64
cmake --build out/build/reversal --config Release -j 2
foreach ($device in @('cpu', 'cuda')) {
    python benchmarks/reversal/run.py --device $device --check --output tmp/reversible_backward_20261110/prefix_suffix_checks_$device.json
    python benchmarks/reversal/run.py --device $device --output tmp/reversible_backward_20261110/prefix_suffix_$device.json
}
python benchmarks/reversal/run.py --device cuda --retry-excluded --max-call-seconds 30 --output tmp/reversible_backward_20261110/prefix_suffix_cuda.json
python benchmarks/reversal/summarize.py --results-dir tmp/reversible_backward_20261110 --report benchmarks/reversal/REPORT.md
```

CMake uses existing Release dependencies. Override `CPU_BUILD`, `DEPS_SOURCE`, and `CUBLAS_ROOT` if their locations differ. The raw JSON files retain every timing sample, repeat count, gradient error, workspace peak, software version, and native library hash. `prefix_suffix_summary.csv` contains all method results; `prefix_suffix_memory_frontier.csv` contains the per-configuration choices at 5%, 10%, and 25% overhead.
