"""Verify saved values/gradients and summarize a before/after batch profile."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    root = args.results
    before = json.loads((root / "batches_before.json").read_text())
    after = json.loads((root / "batches_after.json").read_text())
    old_arrays = np.load(root / "batches_before.npz")
    new_arrays = np.load(root / "batches_after.npz")
    errors, relative_errors = {}, {}
    for key in old_arrays.files:
        old, new = old_arrays[key], new_arrays[key]
        tolerance = 5e-6 if old.dtype == np.float32 else 5e-12
        np.testing.assert_allclose(new, old, atol=tolerance, rtol=tolerance, err_msg=key)
        errors[key] = float(np.max(np.abs(new - old)))
        relative_errors[key] = float(np.linalg.norm(new.astype(float) - old.astype(float)) /
                                     max(np.linalg.norm(old.astype(float)), 1e-30))
        if key.endswith(("_dx", "_dy")):
            assert relative_errors[key] < (5e-5 if old.dtype == np.float32 else 5e-12), key
    old_rows = {row["case"]: row for row in before["measurements"]}
    lines = ["Branched polynomial batch profile: before and after cleanup",
             "Depth 3, order 3 (four Gauss nodes), three input coordinates.",
             after["scope"], "",
             "Every saved value and both path gradients agree within the dtype tolerance.",
             "Tolerances: float32 atol=rtol=5e-6; float64 atol=rtol=5e-12.", "",
             "case                                     before ms   after ms   speedup"]
    summary = []
    for row in after["measurements"]:
        previous = old_rows[row["case"]]
        speedup = previous["median_ms"] / row["median_ms"]
        lines.append(f"{row['case']:40} {previous['median_ms']:10.3f} {row['median_ms']:10.3f} {speedup:8.2f}x")
        summary.append(dict(case=row["case"], before_ms=previous["median_ms"],
                            after_ms=row["median_ms"], speedup=speedup))
    lines += ["", "Speedups describe these measured workloads on this machine, not all workloads.",
              "Long paths and MMD batches remain quadratic in the number of path segments/pairs."]
    repeat_path = root / "cpu_long_repeat.json"
    repeat = json.loads(repeat_path.read_text()) if repeat_path.exists() else None
    if repeat is not None:
        baseline = old_rows[repeat["case"]]
        slowdown = 100 * (repeat["median_ms"] / baseline["median_ms"] - 1)
        lines += ["", f"Isolated repeat of {repeat['case']}:",
                  f"Baseline median {baseline['median_ms']:.3f} ms; range {baseline['min_ms']:.3f}–{baseline['max_ms']:.3f} ms.",
                  f"Final repeat median {repeat['median_ms']:.3f} ms; range {min(repeat['samples_ms']):.3f}–{max(repeat['samples_ms']):.3f} ms.",
                  f"The repeat median is {slowdown:.1f}% slower than baseline; the wide, overlapping ranges leave the cause unresolved.",
                  "The repeat is supplementary and does not replace the original sweep measurements."]
    if after.get("measurement_provenance"):
        lines += ["", after["measurement_provenance"]]
    for dtype in ("float32", "float64"):
        worst = max(value for key, value in relative_errors.items()
                    if key.startswith(dtype) and key.endswith(("_dx", "_dy")))
        lines.append(f"Largest {dtype} relative L2 change in a path gradient: {worst:.3e}.")
    (root / "batches_report.txt").write_text("\n".join(lines) + "\n")
    (root / "batches_comparison.json").write_text(json.dumps(
        dict(measurements=summary, maximum_absolute_errors=errors,
             relative_l2_errors=relative_errors, isolated_cpu_repeat=repeat), indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
