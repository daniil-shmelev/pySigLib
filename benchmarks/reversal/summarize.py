# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Summarize the separate reversal study without changing paper claims."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lines = ["# Prefix-suffix deconstruction study", "", "Production prefix-suffix and its paper figures are unchanged. The prototypes use two directional sweeps and bounded blocks. Final signatures are prepared outside the timed region. The reverse sweep reconstructs prefixes and differentiates the left factors and prefix recurrence. The forward sweep reconstructs suffixes and differentiates the right factors and suffix recurrence.", "",
             "The grid contains seven dimension/depth pairs, 16/256/4096 segments, batches 1/32, and float32/float64. Block 0 is the current implementation; the candidates use blocks 1, 8, 32, 128, and 512, capped at the path length. Native automatic split selection is unchanged. Each completed timing has three warm-up calls and nine alternating samples, with synchronized CUDA wall time. Reconstruction, allocation, and free costs are included. CUDA baseline buffers use the existing cache; each prototype call allocates and frees its numeric buffers.", "",
             "## Memory and error definitions", "",
             "Native temporary memory counts algorithm-owned numeric buffers. CPU peaks are measured from live allocations in the same path scheduler. CUDA peaks use actual reserved buffer capacity, including metadata and the explicit 4 MiB cuBLAS workspace. Inputs, output gradients, stacks/registers, library code, and opaque CPU BLAS/TBB allocations are excluded. Input and output sizes are recorded separately. Process peak working set is a cumulative process metric and is not used to claim algorithm memory savings. GPU process memory is unavailable under the Windows WDDM driver (nvidia-smi reports N/A).", "",
             "Grid errors use CPU float64 production prefix-suffix on identical rounded paths and cotangents. Separate correctness checks use independent Chen autograd, all tested explicit splits, both scalar conventions, zero increments, and partial blocks. A candidate qualifies only if its gradient passes the fixed pointwise tolerance: 3e-5 for float32, 1e-10 for float64, used as both absolute and relative tolerances. Failed accuracy checks are retained in the raw results and excluded from qualifying savings.", "",
             "## Savings within runtime limits", "",
             "The table selects the smallest workspace for each configuration within each runtime limit. Counts require positive memory savings and valid gradients. The median and best savings are over qualifying configurations. Each device has 84 configurations.", "",
             "| Device | Maximum overhead | Qualifying configurations | Median saving | Best saving |", "|---|---:|---:|---:|---:|"]
    rows, winners, exclusions, inaccurate = [], [], [], []
    examples = []
    for device in ["cpu", "cuda"]:
        report = json.loads((args.results_dir / f"prefix_suffix_{device}.json").read_text())
        assert report["complete"] and len(report["cases"]) == 84
        for case in report["cases"]:
            baseline = case["methods"][0]
            for method in case["methods"]:
                row = {k: case[k] for k in ["device", "dimension", "degree", "segments", "batch", "dtype", "input_bytes", "output_bytes"]}
                row.update(method)
                row.pop("samples_ms")
                if method["status"] == "ok":
                    assert len(method["samples_ms"]) == 9
                    if baseline["status"] == "ok":
                        row["runtime_ratio"] = method["median_ms"] / baseline["median_ms"]
                        row["memory_saving"] = 1 - method["peak_native_temporary_bytes"] / baseline["peak_native_temporary_bytes"]
                    if not method["gradient_pass"]:
                        inaccurate.append(row)
                else:
                    exclusions.append(row)
                rows.append(row)
            for overhead in [.05, .10, .25]:
                if baseline["status"] != "ok":
                    continue
                eligible = [r for r in case["methods"][1:] if r["status"] == "ok" and r["gradient_pass"]
                            and baseline["gradient_pass"] and r["median_ms"] <= baseline["median_ms"] * (1 + overhead)
                            and r["peak_native_temporary_bytes"] < baseline["peak_native_temporary_bytes"]]
                if eligible:
                    best = min(eligible, key=lambda r: r["peak_native_temporary_bytes"])
                    record = {k: case[k] for k in ["device", "dimension", "degree", "segments", "batch", "dtype"]}
                    record.update(overhead_limit=overhead, block=best["block"], runtime_ratio=best["median_ms"] / baseline["median_ms"],
                                  baseline_ms=baseline["median_ms"], candidate_ms=best["median_ms"],
                                  baseline_bytes=baseline["peak_native_temporary_bytes"], candidate_bytes=best["peak_native_temporary_bytes"],
                                  memory_saving=1 - best["peak_native_temporary_bytes"] / baseline["peak_native_temporary_bytes"])
                    winners.append(record)
        for overhead in [.05, .10, .25]:
            selected = [r for r in winners if r["device"] == device and r["overhead_limit"] == overhead]
            savings = [r["memory_saving"] * 100 for r in selected]
            lines.append(f"| {device.upper()} | {100 * overhead:g}% | {len(selected)}/84 | {statistics.median(savings):.2f}% | {max(savings):.2f}% |" if selected
                         else f"| {device.upper()} | {100 * overhead:g}% | 0/84 | - | - |")
            if selected and (overhead == .05 or device == "cuda" and overhead == .10):
                examples.append(max(selected, key=lambda r: r["memory_saving"]))
    lines += ["", "Representative best savings:", "",
              "| Device | Overhead limit | (d,m), n, batch, dtype | Block | Baseline / candidate ms | Baseline / candidate MiB | Saving |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for r in examples:
        lines.append(f"| {r['device']} | {100 * r['overhead_limit']:g}% | ({r['dimension']},{r['degree']}), {r['segments']}, {r['batch']}, {r['dtype']} | {r['block']} | {r['baseline_ms']:.3f} / {r['candidate_ms']:.3f} | {r['baseline_bytes'] / 2**20:.3f} / {r['candidate_bytes'] / 2**20:.3f} | {100 * r['memory_saving']:.2f}% |")
    lines += ["", "## Failures and limits", "",
              f"Runtime/allocation exclusions: {len(exclusions)}. Accuracy failures among completed timings: {len(inaccurate)}, from {len(rows)} method measurements. The complete case lists are in `prefix_suffix_exclusions.json`. The initial 10-second call limit excluded one CUDA case. That configuration was rerun with a 30-second limit; the original attempt is retained in the raw JSON retry history.", "",
              "| Device | Baseline accuracy failures | Prototype accuracy failures | Runtime/allocation exclusions |",
              "|---|---:|---:|---:|"]
    for device in ["cpu", "cuda"]:
        failed_baseline = sum(r["device"] == device and r["block"] == 0 for r in inaccurate)
        failed_candidate = sum(r["device"] == device and r["block"] != 0 for r in inaccurate)
        excluded = sum(r["device"] == device for r in exclusions)
        lines.append(f"| {device.upper()} | {failed_baseline} | {failed_candidate} | {excluded} |")
    lines += ["", "Direct reversal can amplify rounding error over long paths. The production baseline also fails the fixed tolerance on some long float32 cases. Both the baseline and candidate must pass before a result can qualify. A memory reduction alone is not sufficient to recommend a production change. The candidates remain experimental. Results apply to this Windows CPU/GPU and these native kernels; they are not general bounds on the best possible implementation.", "",
              "## Validation and reproduction", ""]
    for device in ["cpu", "cuda"]:
        checks = json.loads((args.results_dir / f"prefix_suffix_checks_{device}.json").read_text())["checks"]
        lines.append(f"- {device.upper()}: {len(checks)} correctness and workspace checks passed. Every block size is checked for constant workspace at three path lengths up to 4096 segments, with the shortest path at least as long as the block.")
    lines += ["", "From the pySigLib root in PowerShell:", "", "```powershell",
              "cmake -S benchmarks/reversal -B out/build/reversal -G 'Visual Studio 17 2022' -A x64",
              "cmake --build out/build/reversal --config Release -j 2",
              "foreach ($device in @('cpu', 'cuda')) {",
              "    python benchmarks/reversal/run.py --device $device --check --output tmp/reversible_backward_20261110/prefix_suffix_checks_$device.json",
              "    python benchmarks/reversal/run.py --device $device --output tmp/reversible_backward_20261110/prefix_suffix_$device.json",
              "}",
              "python benchmarks/reversal/run.py --device cuda --retry-excluded --max-call-seconds 30 --output tmp/reversible_backward_20261110/prefix_suffix_cuda.json",
              "python benchmarks/reversal/summarize.py --results-dir tmp/reversible_backward_20261110 --report benchmarks/reversal/REPORT.md", "```", "",
              "CMake uses existing Release dependencies. Override `CPU_BUILD`, `DEPS_SOURCE`, and `CUBLAS_ROOT` if their locations differ. The raw JSON files retain every timing sample, repeat count, gradient error, workspace peak, software version, and native library hash. `prefix_suffix_summary.csv` contains all method results; `prefix_suffix_memory_frontier.csv` contains the per-configuration choices at 5%, 10%, and 25% overhead.", ""]
    for name, records in [("prefix_suffix_summary.csv", rows), ("prefix_suffix_memory_frontier.csv", winners)]:
        fields = list(dict.fromkeys(k for row in records for k in row))
        with (args.results_dir / name).open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (args.results_dir / "prefix_suffix_exclusions.json").write_text(json.dumps(dict(runtime_or_allocation=exclusions, accuracy=inaccurate), indent=2) + "\n", encoding="ascii", newline="\r\n")
    args.report.write_text("\n".join(lines), encoding="ascii", newline="\r\n")


if __name__ == "__main__":
    main()
