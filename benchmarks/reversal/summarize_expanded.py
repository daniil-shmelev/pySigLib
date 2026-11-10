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

"""Report replacement costs without selecting only favorable cases."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, exclusions = [], []
    reports = {device: json.loads((args.results_dir / f"expanded_{device}.json").read_text()) for device in ["cpu", "cuda"]}
    for device, report in reports.items():
        assert report["metadata"].get("screen_finished_utc") and report["metadata"].get("confirm_finished_utc")
        for stage in ["screen", "confirm"]:
            for case in report[stage]:
                identity = {key: case[key] for key in ["device", "dimension", "degree", "segments", "batch", "dtype"]}
                identity["stage"] = stage
                if "methods" not in case:
                    exclusions.append({**identity, "reason": case["reason"]})
                    continue
                baseline = case["methods"][0]
                for method in case["methods"]:
                    row = {**identity, "input_bytes": case["input_bytes"], "output_bytes": case["output_bytes"]}
                    row.update({k: v for k, v in method.items() if k != "samples_ms"})
                    if method["status"] == baseline["status"] == "ok":
                        row["runtime_ratio"] = method["median_ms"] / baseline["median_ms"]
                        row["memory_saving"] = 1 - method["peak_native_temporary_bytes"] / baseline["peak_native_temporary_bytes"]
                    if method["status"] != "ok":
                        exclusions.append({**identity, "block": method["block"], "reason": method.get("reason"),
                                           "first_call_ms": method.get("first_call_ms")})
                    rows.append(row)
    lines = ["# Expanded prefix-suffix replacement benchmark", "",
             "This study compares the current full-path implementation with the bounded two-sweep prototype. Production code and paper figures remain unchanged. A fixed block of 512 is the primary replacement candidate; blocks 32 and 128 are also measured for paths above 16 segments. The block is capped at path length. This is a comparison of whole-algorithm replacement, with no selection between the old and new algorithms.", "",
             "Each device has 368 screen configurations: 23 dimension/depth pairs, segments 16/256/4096 with batches 1/32, plus segments 64/1024 with batch 8; every shape uses both dtypes. The pairs are listed in `expanded.py`. The screen uses one warm-up and three alternating samples. Twelve cases per device are repeated with three warm-ups and nine samples: four largest measured regressions, four nearest 10% overhead, and four fastest cases. Screen calls excluded between two and five seconds are also confirmed on CUDA. Samples are synchronized on CUDA. Inputs and cotangents are identical; saved final signatures are prepared outside timing. Native reconstruction and allocations are included. Timings cover untransformed paths without correction, with the scalar term included. They do not establish runtime for time augmentation, lead-lag, or corrected paths.", "",
             "In the screen, an initial call above two seconds is recorded but excluded from repeated timing. Confirmation uses a five-second limit. Errors, allocation failures, and excluded calls are listed in `expanded_exclusions.json`. Such calls do not enter the runtime statistics. Raw JSON includes their first-call times and gradient checks when available. CPU float64 production prefix-suffix is the gradient reference on the same rounded inputs. Fixed pointwise tolerances are 3e-5 for float32 and 1e-10 for float64. Performance statistics include measured cases that fail these tolerances; failures are counted separately.", "",
             "Workspace is native algorithm-owned numeric temporary storage, excluding input and output arrays, registers, and opaque CPU BLAS/TBB allocations. CUDA includes its explicit 4 MiB cuBLAS workspace. CUDA baseline buffers retain the production cache; prototype buffers are allocated and freed each call. The measured peaks are not total process or GPU memory.", "",
             "## Whole-algorithm replacement: block 512", "",
             "Each row weights measured configurations equally. Ratios above one are slowdowns. Median and 90th-percentile ratios do not represent a distribution of real user workloads.", "",
             "| Device | Timed cases | Median ratio | 90th percentile | Worst ratio | Within 10% | Median workspace saving | Accuracy failures |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for device in ["cpu", "cuda"]:
        group = [r for r in rows if r["stage"] == "screen" and r["device"] == device and r["block"] == 512 and "runtime_ratio" in r]
        ratios = [r["runtime_ratio"] for r in group]
        savings = [r["memory_saving"] for r in group]
        lines.append(f"| {device.upper()} | {len(group)}/368 | {np.median(ratios):.2f} | {np.quantile(ratios, .9):.2f} | {max(ratios):.2f} | {sum(x <= 1.1 for x in ratios)}/{len(group)} | {100*np.median(savings):.1f}% | {sum(not r['gradient_pass'] for r in group)} |")
    for field, label in [("segments", "Segments"), ("batch", "Batch"), ("dimension", "Dimension"), ("degree", "Depth"), ("dtype", "Dtype")]:
        lines += ["", f"## Runtime by {label.lower()}", "", f"| Device | {label} | Cases | Median ratio | 90th percentile | Worst ratio |", "|---|---|---:|---:|---:|---:|"]
        for device in ["cpu", "cuda"]:
            group = [r for r in rows if r["stage"] == "screen" and r["device"] == device and r["block"] == 512 and "runtime_ratio" in r]
            for value in sorted({r[field] for r in group}):
                ratios = [r["runtime_ratio"] for r in group if r[field] == value]
                lines.append(f"| {device.upper()} | {value} | {len(ratios)} | {np.median(ratios):.2f} | {np.quantile(ratios, .9):.2f} | {max(ratios):.2f} |")
    lines += ["", "## Block size comparison", "", "| Device | Block | Timed cases | Median ratio | 90th percentile | Median workspace saving |", "|---|---:|---:|---:|---:|---:|"]
    for device in ["cpu", "cuda"]:
        for block in [32, 128, 512]:
            group = [r for r in rows if r["stage"] == "screen" and r["device"] == device and r["block"] == block and r["segments"] > 16 and "runtime_ratio" in r]
            ratios = [r["runtime_ratio"] for r in group]
            savings = [r["memory_saving"] for r in group]
            if group:
                lines.append(f"| {device.upper()} | {block} | {len(group)} | {np.median(ratios):.2f} | {np.quantile(ratios, .9):.2f} | {100*np.median(savings):.1f}% |")
    lines += ["", "## Confirmation and limits", ""]
    for device, report in reports.items():
        lines.append(f"- {device.upper()}: {len(report['screen'])} screened cases and {len(report['confirm'])} confirmation cases. See `expanded_summary.csv` for every method, case, timing, workspace, and accuracy result.")
    lines += [f"- There are {len(exclusions)} method or case exclusions across both stages. The full list is `expanded_exclusions.json`. These cases must not be treated as successful timings.",
              "- Direct reversal can amplify floating-point error on long paths. An accuracy failure remains a failure even if its timing is favorable. Review raw errors before a production replacement.", "",
              "Accuracy of the baseline and block-512 candidate on the screen:", "",
              "| Device | Both pass | Baseline only fails | Candidate only fails | Both fail | Unchecked |",
              "|---|---:|---:|---:|---:|---:|"]
    for device, report in reports.items():
        counts = [0] * 5
        for case in report["screen"]:
            if "methods" not in case or "gradient_pass" not in case["methods"][0] or "gradient_pass" not in case["methods"][-1]:
                counts[4] += 1
                continue
            baseline = case["methods"][0]["gradient_pass"]
            candidate = case["methods"][-1]["gradient_pass"]
            counts[0 if baseline and candidate else 1 if candidate else 2 if baseline else 3] += 1
        lines.append(f"| {device.upper()} | {' | '.join(str(n) for n in counts)} |")
    lines += ["",
              "Largest confirmed block-512 slowdowns:", "", "| Device | (d,m), segments, batch, dtype | Old ms | New ms | Ratio | Gradient passes |",
              "|---|---|---:|---:|---:|---|"]
    for device in ["cpu", "cuda"]:
        group = sorted([r for r in rows if r["stage"] == "confirm" and r["device"] == device and r["block"] == 512 and "runtime_ratio" in r],
                       key=lambda r: r["runtime_ratio"], reverse=True)
        for row in group[:4]:
            lines.append(f"| {device.upper()} | ({row['dimension']},{row['degree']}), {row['segments']}, {row['batch']}, {row['dtype']} | {row['median_ms']/row['runtime_ratio']:.3f} | {row['median_ms']:.3f} | {row['runtime_ratio']:.2f} | {row['gradient_pass']} |")
    lines += ["",
              "## Reproduction", "", "From the pySigLib root, after building the experiment DLLs as described in `REPORT.md`:", "", "```powershell",
              "foreach ($device in @('cpu', 'cuda')) {",
              "    python benchmarks/reversal/expanded.py --device $device --stage screen --output tmp/reversible_backward_20261110/expanded_$device.json",
              "    python benchmarks/reversal/expanded.py --device $device --stage confirm --output tmp/reversible_backward_20261110/expanded_$device.json",
              "}",
              "python benchmarks/reversal/summarize_expanded.py --results-dir tmp/reversible_backward_20261110 --report benchmarks/reversal/EXPANDED_REPORT.md", "```", "",
              "The JSON files retain the individual samples, errors, exclusions, settings, and native library hashes. Re-running a stage resumes at the first missing configuration.", ""]
    with (args.results_dir / "expanded_summary.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    (args.results_dir / "expanded_exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n", encoding="ascii", newline="\r\n")
    args.report.write_text("\n".join(lines), encoding="ascii", newline="\r\n")


if __name__ == "__main__":
    main()
