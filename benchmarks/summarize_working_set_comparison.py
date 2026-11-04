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

"""Summarise the final CPU benchmark build and its paired references."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path)
    parser.add_argument("library", type=Path, help="Native library of the build to report")
    args = parser.parse_args()
    build = hashlib.sha256(args.library.read_bytes()).hexdigest()
    rows = {}
    for line in args.trials.read_text().splitlines():
        row = json.loads(line)
        if row["case"] == "arabic" and row.get("workload_version") != 2:
            continue
        if row["variant"] == "native" and "error" not in row and row.get("build_id") != build:
            continue
        if row["variant"] == "prototype" and "error" not in row and row.get("prediction_backend") != "baseline":
            continue
        key = tuple(row[k] for k in ("case", "mode", "loss", "jobs", "variant"))
        rows[key] = row
    native = [key for key in rows if key[-1] == "native"]
    if len(native) != 64:
        raise ValueError(f"Expected 64 final-build configurations, found {len(native)}")
    for variant, expected in (("legacy", 26), ("prototype", 64)):
        count = sum(key[-1] == variant for key in rows)
        if count != expected:
            raise ValueError(f"Expected {expected} {variant} configurations, found {count}")
    folder = args.trials.parent
    flat = []
    failures = []
    for key, r in sorted(rows.items()):
        if r.get("error"):
            failures.append(f"{key} {r['error']}")
            continue
        f = {k:r[k] for k in ("variant", "case", "mode", "loss", "jobs", "support", "kkt", "objective", "converged", "prediction_error")}
        for name in ("fit", "predict"):
            f[name+"_p25_ms"], f[name+"_median_ms"], f[name+"_p75_ms"] = r[name+"_ms"]
        f["rss_mib"] = r["rss"]/2**20
        f["native_workspace_mib"] = r["stats"]["peak_bytes"]/2**20
        for name in ("preparation", "search", "insertion", "solve", "certification", "output"):
            f[name+"_ms"] = 1000*r["stats"].get(name+"_seconds", 0) if r["variant"]!="prototype" else None
        f["engine"] = r["stats"].get("engine")
        flat.append(f)
        if not r["converged"] or r["kkt"]>1.05e-7 or r["prediction_error"]>1e-8:
            failures.append(f"{key} convergence={r['converged']}, independent KKT={r['kkt']:.3g}, prediction difference={r['prediction_error']:.3g}")
    with (folder/"measurements.csv").open("w",newline="") as out:
        writer = csv.DictWriter(out,fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    with (folder/"final_trials.jsonl").open("w") as out:
        for _, row in sorted(rows.items()):
            out.write(json.dumps(row)+"\n")

    lines = ["# CPU working-set Lasso", "",
             f"Windows Release build `{build[:12]}`. Intel Core i7-13700H, float64, one or eight workers and one BLAS thread. "
             "Each configuration used one warm-up and five timed fits in a fresh Python process. "
             "Fit includes preparation and certification. Prediction uses the training paths and returns the linear predictor. "
             "A complete independent design was constructed after both timings and the process memory reading.", "",
             "The Gaussian case has 64 paths, 3 channels and degree 3. The anisotropic case has 128 paths, 8 channels and degree 3. "
             "The commutator case has 64 paths, 4 channels and degree 4. The Hall tree case has 48 paths, 2 channels and degree 6. "
             "The Arabic case uses 128 seeded training examples, all 14 channels and degree 2. Fourteen samples are digit 4. "
             "The penalty is 0.003 for Gaussian, anisotropic and degree-six paths, 0.00001 for commutators and 0.001 for Arabic. "
             "The tolerance is 0.0000001. The baseline uses its exact bound and grouped signed search.", "",
             "## Paired timings", "",
             "Ratios are geometric means across matched configurations that passed the independent checks. "
             "A speedup above one means that the native working-set implementation was faster.", "",
             "| Reference | Pairs | Fit speedup | Prediction speedup | Native workspace ratio, new / reference | Process peak ratio, new / reference |",
             "|---|---:|---:|---:|---:|---:|"]
    def valid(r):
        return r and not r.get("error") and r["converged"] and r["kkt"]<=1.05e-7 and r["prediction_error"]<=1e-8
    def gm(x):
        return math.exp(sum(map(math.log,x))/len(x))
    maximum_diff = 0.
    support_differences = 0
    timing_notes = []
    for variant in ("legacy", "prototype"):
        pairs = [(rows[key],rows.get((*key[:-1],variant))) for key in native]
        pairs = [(a,b) for a,b in pairs if valid(a) and valid(b)]
        if not pairs:
            continue
        maximum_diff = max(maximum_diff,max(abs(a["objective"]-b["objective"]) for a,b in pairs))
        support_differences += sum(a["support"] != b["support"] for a,b in pairs)
        for a,b in pairs:
            if abs(a["objective"]-b["objective"])>3e-8*max(1,abs(a["objective"])):
                failures.append(f"Objective mismatch {variant} {a['case']} {a['mode']} {a['loss']} {a['jobs']}")
        speed = [gm([b[k][1]/a[k][1] for a,b in pairs]) for k in ("fit_ms","predict_ms")]
        mem = f"{gm([a['stats']['peak_bytes']/b['stats']['peak_bytes'] for a,b in pairs]):.2f}" if variant=="legacy" else "n/a"
        lines.append(f"| {variant} | {len(pairs)} | {speed[0]:.2f}x | {speed[1]:.2f}x | {mem} | {gm([a['rss']/b['rss'] for a,b in pairs]):.3f} |")
        faster = sum(a["fit_ms"][1] < b["fit_ms"][1] for a,b in pairs)
        timing_notes.append(f"The replacement fitted faster in {faster} of the {len(pairs)} {variant} comparisons.")
    lines.extend(["", " ".join(timing_notes), "",
                  "The prototype uses the frozen legacy prediction helper. Its prediction ratio compares the two native helpers. "
                  "Native workspace is the allocator high-water mark including reservations for third-party workspace and output. "
                  "The prototype measures only its native tree workspace, so its workspace ratio is omitted. "
                  "Process peak is the OS high-water mark, which includes Python, Torch and any loaded numerical libraries.", "",
                  "## Eight-worker examples", "", "Times are milliseconds. Brackets give the 25th and 75th percentiles across the five timed fits.", "",
                  "| Case | Dictionary | Loss | Engine | Fit, ms [P25, P75] | Prediction, ms [P25, P75] | Native MiB | Process MiB | Nonzeros |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|"])
    for case, loss in (("gaussian","squared"),("anisotropic","huber"),("commutator","huber"),("arabic","logistic"),("hall6","poisson")):
        for mode in ("sig","log","hall"):
            r = rows.get((case,mode,loss,8,"native"))
            if not r or r.get("error"):
                continue
            def quartiles(values):
                return f"{values[1]:.3f} [{values[0]:.3f}, {values[2]:.3f}]"
            lines.append(f"| {case} | {mode} | {loss} | {r['stats']['engine']} | {quartiles(r['fit_ms'])} | {quartiles(r['predict_ms'])} | {r['stats']['peak_bytes']/2**20:.2f} | {r['rss']/2**20:.1f} | {r['support']} |")
    good=[rows[k] for k in native if not rows[k].get("error")]
    phases = ("preparation", "search", "solve", "certification")
    lines.extend(["", "## Native phase times", "",
                  "Medians across configurations, using the final timed fit in each. All times are milliseconds.", "",
                  "| Engine | Configurations | Preparation | Search | Restricted solve | Certification |",
                  "|---|---:|---:|---:|---:|---:|"])
    for engine in ("packed", "streamed", "tree"):
        group = [r["stats"] for r in good if r["stats"]["engine"] == engine]
        if group:
            times = [median(1000*s[phase+"_seconds"] for s in group) for phase in phases]
            lines.append(f"| {engine} | {len(group)} | " + " | ".join(f"{t:.3f}" for t in times) + " |")
    lines.extend(["", "## Slower squared-loss fits", "",
                  "Fit times are medians. The restricted solve and the combined search and certification times come from the final timed fit. All times are milliseconds.", "",
                  "| Case | Dictionary | Workers | Old fit | New fit | Old solve | New solve | Old search + certificate | New search + certificate |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for key in sorted(native):
        a, b = rows[key], rows.get((*key[:-1], "legacy"))
        if valid(a) and valid(b) and a["fit_ms"][1] > b["fit_ms"][1]:
            values = [b["fit_ms"][1], a["fit_ms"][1]]
            values.extend(1000*r["stats"]["solve_seconds"] for r in (b, a))
            values.extend(1000*(r["stats"]["search_seconds"] + r["stats"]["certification_seconds"]) for r in (b, a))
            lines.append(f"| {a['case']} | {a['mode']} | {a['jobs']} | " + " | ".join(f"{t:.3f}" for t in values) + " |")
    lines.extend(["", "## Independent checks", "",
                  f"{len(good)} native fits returned. The largest independently measured KKT residual was {max(r['kkt'] for r in good):.3g}. "
                  f"The largest prediction difference against independent columns was {max(r['prediction_error'] for r in good):.3g}. "
                  f"The largest paired objective difference was {maximum_diff:.3g}. "
                  f"Support sizes differed in {support_differences} paired comparisons. "
                  f"There were {len(failures)} failures or timeouts across all methods and audits.", *failures, "",
                  "The CSV includes every worker count, quartiles, support size, objective, KKT residual, both memory measures and native phase times from the last timed fit. "
                  "The JSONL file includes all five individual fit and prediction times.", "",
                  "## Reproduction", "",
                  "Build the Release CPU library and use it in the local Python package. For the historical comparison, "
                  "point the baseline directory at a separate package containing the source and CPU library from commit `fe32561e`. "
                  "The prototype uses the research code in `benchmarks/general_loss_working_set.py` and its original tree bridge in "
                  "`output/unified_general/bridge.dll`. Build that bridge with `benchmarks/build_lookahead_search.ps1` using the Release build directory of the baseline checkout.", "",
                  "```powershell", "python -m benchmarks.run_working_set_comparison output/working_set_comparison/trials.jsonl --baseline output/working_set_baseline --prototype",
                  "python -m benchmarks.summarize_working_set_comparison output/working_set_comparison/trials.jsonl pysiglib/cpsig.dll", "```", ""])
    (folder/"report.md").write_bytes("\r\n".join(lines).encode("ascii"))


if __name__ == "__main__":
    main()
