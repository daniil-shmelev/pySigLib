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

"""Aggregate independent worker records without treating incomplete runs as measurements."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from datasets import CASES

NAMES = {
    "gapped": "GunPoint, gapped patterns",
    "ecg-gapped": "ECG200, gapped patterns",
    "wafer-gapped": "Wafer, gapped patterns",
    "mushroom": "Mushroom, categorical rules",
    "intervals": "Bike Sharing, intervals",
    "sms": "SMS Spam, contiguous n-grams",
    "arabic": "Arabic, integrated Haar",
    "arabic-frames": "Arabic, frame Haar",
    "arabic-gapped": "Arabic, integrated symbol patterns",
    "arabic-frame-words": "Arabic, frame symbol patterns",
    "haar": "GunPoint, Haar",
    "shapelets": "GunPoint, RBF shapelets",
    "shapelet-distance": "GunPoint, shapelet distance",
    "legendre": "Ishigami, Legendre products",
    "splines": "Friedman 1, spline products",
    "motifs": "MUTAG, induced motifs",
    "graph-scattering": "MUTAG, graph scattering",
    "landmarks": "Spatial simulation, RBF landmarks",
    "tanks": "Cascaded Tanks, Volterra",
    "walsh": "MONK-2, Walsh features",
}
MIB = 2**20


def read(path):
    return json.loads(path.read_text())


def quant(rows, field, scale=1):
    return np.quantile([x[field] * scale for x in rows], [0.25, 0.50, 0.75])


def stat(rows, field, scale=1, spec=".3g"):
    lo, m, hi = quant(rows, field, scale)
    return f"{m:{spec}} [{lo:{spec}}, {hi:{spec}}]"


def format_score(row, case):
    if "accuracy" in row:
        return f"{100 * row['accuracy']:.2f}%"
    factor = 1000 if case == "intervals" else 1
    return f"{factor * row['rmse']:.4g}"


def plot_comparisons(rows, destination):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    rows.sort(key=lambda row: -row[2])
    y = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(10.7, 7.8), sharey=True)
    for ax, offset, title in zip(axes, (1, 3), ("Fit time", "Additional peak fit memory")):
        full = [r[offset] for r in rows]
        best = [r[offset + 1] for r in rows]
        ax.set_xscale("log")
        ax.grid(axis="x", which="major", color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.axvline(1, color="0.35", linewidth=1)
        ax.hlines(y, best, full, color="0.72", linewidth=1)
        ax.scatter(full, y, s=40, color="#257497", label="Full Celer / tree", zorder=3)
        ax.scatter(
            best, y, s=21, marker="x", color="#8a3f0f", label="Best measured control / tree", zorder=4
        )
        lo, hi = min(best + full), max(best + full)
        ticks = [v for v in (0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000) if lo / 1.4 < v < hi * 1.4]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Ratio to tree")
        ax.set_title(title)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(y, labels=[r[0] for r in rows], fontsize=9)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.59, 0.99))
    figure.subplots_adjust(left=0.28, right=0.98, bottom=0.08, top=0.9, wspace=0.2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)


def summarize(root, cases, reps, plot=None):
    comparisons = []
    head = [
        "# Runtime, memory and predictive results",
        "",
        (
            "Training times include feature preparation and fitting. Memory is incremental peak process RSS in MiB. "
            "Each entry is the median of independent fresh-process runs. Full uses Celer on an explicit design. "
            "The working-set control uses the same small restricted fits with exhaustive matrix correlations. "
            "The data, training features and penalty are identical within a row."
        ),
        "",
        "| Problem and dictionary | Train / test | Observed columns | Tree s | Full s | Full / tree | Tree MiB | Full MiB | Test tree / full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    detail = [
        "## Timing and memory dispersion",
        "",
        (
            "Each cell gives median [first quartile, third quartile]. With three repetitions these are descriptive order statistics. "
            "Absolute process high-water includes Python and the loaded data. Prediction uses selected features for all solvers "
            "and includes the small cost of closing the fit memory sampler."
        ),
        "",
        "| Case | Method | Fits | Fit s | Prepare s | Search and solve s | Predict ms | Increment MiB | Absolute peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    audit = [
        "## Sparsity and agreement",
        "",
        (
            "Union support counts features used by any response. The largest objective difference is taken across responses. "
            "The held-out score difference precedes thresholding or argmax. Classification disagreement counts are absolute."
        ),
        "",
        "| Case | Penalty ratio | Union tree / full | Largest KKT error (all solvers) | Objective difference | Held-out score difference | Different labels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    controls = [
        "## Strongest measured alternatives",
        "",
        "| Case | Tree s | Matrix working set s / MiB | Specialised scan s / MiB |",
        "|---|---:|---:|---:|",
    ]
    path = [
        "## Four-penalty paths",
        "",
        (
            "Calibration, search preparation, four warm fits and four validation predictions are timed. "
            "The explicit design is constructed once. Memory here is the additional OS high-water above the pre-path high-water, "
            "which can undercount growth if an earlier peak masks it."
        ),
        "",
        "| Case | Method | Repeats | Path s | Prepare and calibrate s | New high-water MiB | Worst KKT |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    val = [
        "## Validation curves",
        "",
        (
            "Single-fit comparisons use the penalty chosen on the explicit solver's validation curve. "
            "The reported score is accuracy or RMSE. Bike Sharing is expressed in rentals per hour."
        ),
        "",
        "| Case | 0.30 | 0.10 | 0.03 | 0.01 | Chosen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    external = [
        "## Other learning algorithms on the same splits",
        "",
        (
            "These models solve different problems. Each uses the same held-out examples as its Lasso row. "
            "Prediction time is for the entire test set. Selection of parameters is outside fit time."
        ),
        "",
        "| Case | Task baseline | Test score | Fit s | Prediction ms | Increment MiB | Absolute peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    miners = [
        "## Published safe pattern pruning implementation",
        "",
        (
            "Each run computes the maximum penalty and fits the validation-selected ratio. The tree includes "
            "feature preparation. PMOpt starts from a symbolic CSV and its clock includes process launch, loading "
            "and model writing. CSV export is measured separately. RSS is the entire process high-water. "
            "PMOpt runs as native C++ and the tree process includes the warmed Python runtime. The accuracy "
            "audit uses the same independent full-dictionary KKT tolerance of 1e-7 for both. Time and RSS "
            "include censored attempts where indicated. PMOpt uses relative dual-gap tolerances of 1e-12 "
            "(strict) and 1e-4 (published default)."
        ),
        "",
        "| Case | Solver | Pass / attempted | Outcome | Fit s | Absolute peak MiB | Export s | Worst KKT | Test accuracy |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    names = {
        "gapped": "Euclidean 1-NN",
        "ecg-gapped": "Euclidean 1-NN",
        "wafer-gapped": "Euclidean 1-NN",
        "mushroom": "Decision tree",
        "sms": "Linear SVM",
        "intervals": "Histogram gradient boosting",
        "motifs": "WL subtree SVM",
        "tanks": "Linear ridge FIR",
    }
    for case in cases:
        d = root / case
        cfg = read(d / "validation.json")
        methods = ["tree", "full", "matrix-ws"]
        if case == "intervals":
            methods.append("histogram")
        if case == "walsh":
            methods.append("fwht")
        if case in ("gapped", "ecg-gapped", "wafer-gapped", "arabic-gapped", "arabic-frame-words"):
            methods.append("distinct")
        runs = {m: [read(d / f"{m}-{i}.json") for i in range(reps)] for m in methods}
        if not all(
            np.all(np.isfinite([r["fit_s"], r["peak_rss"], r["kkt"]])) and r["kkt"] <= 1.01e-7
            for rows in runs.values()
            for r in rows
        ):
            raise ValueError(f"failed fit in {case}")
        t, f, mw = (runs[m] for m in methods[:3])
        name = NAMES[case]
        tf, ff = quant(t, "fit_s")[1], quant(f, "fit_s")[1]
        tm, fm = quant(t, "peak_above_start", 1 / MIB)[1], quant(f, "peak_above_start", 1 / MIB)[1]
        if plot:
            alternatives = [runs[m] for m in methods[1:]]
            if tf > 0 and tm > 0:
                comparisons.append(
                    (
                        name,
                        ff / tf,
                        min(quant(rows, "fit_s")[1] for rows in alternatives) / tf,
                        fm / tm,
                        min(quant(rows, "peak_above_start", 1 / MIB)[1] for rows in alternatives) / tm,
                    )
                )
        head.append(
            f"| {name} | {cfg['n_train']} / {cfg['n_test']} | {cfg['n_features']:,} | {tf:.4g} | {ff:.4g} | "
            f"{ff / tf:.2f} | {tm:.2f} | {fm:.2f} | {format_score(t[0], case)} / {format_score(f[0], case)} |"
        )
        extra = " "
        if len(methods) > 3:
            z = runs[methods[3]]
            extra = f"{quant(z, 'fit_s')[1]:.4g} / {quant(z, 'peak_above_start', 1 / MIB)[1]:.2f}"
        controls.append(
            f"| {name} | {tf:.4g} | {quant(mw, 'fit_s')[1]:.4g} / "
            f"{quant(mw, 'peak_above_start', 1 / MIB)[1]:.2f} | {extra} |"
        )
        for m, rows in runs.items():
            cells = [
                stat(rows, key, scale)
                for key, scale in (
                    ("fit_s", 1),
                    ("prepare_s", 1),
                    ("solve_search_s", 1),
                    ("predict_s", 1000),
                    ("peak_above_start", 1 / MIB),
                    ("peak_rss", 1 / MIB),
                )
            ]
            detail.append(f"| {name} | {m} | {len(rows)} | " + " | ".join(cells) + " |")
        a, b = (np.load(d / f"{m}-0.npy") for m in ("tree", "full"))
        difference = (
            int(np.sum((a >= 0.5) != (b >= 0.5)))
            if a.shape[1] == 1
            else int(np.sum(a.argmax(1) != b.argmax(1)))
        )
        objectives = np.max(np.abs(np.array(t[0]["objective"]) - f[0]["objective"]))
        worst = max(r["kkt"] for rows in runs.values() for r in rows)
        label_cell = str(difference) if "accuracy" in t[0] else " "
        audit.append(
            f"| {name} | {cfg['ratio']} | {t[0]['union']} / {f[0]['union']} | {worst:.2g} | "
            f"{objectives:.2g} | {np.max(np.abs(a - b)):.3g} | {label_cell} |"
        )
        val.append(
            f"| {name} | "
            + " | ".join(format_score(z, case) for z in cfg["curves"])
            + f" | {cfg['ratio']} |"
        )
        for m in ("tree", "full"):
            rows = [read(z) for z in sorted(d.glob(f"path-{m}-*.json"))]
            if rows:
                path.append(
                    f"| {name} | {m} | {len(rows)} | {stat(rows, 'full_path_s')} | {stat(rows, 'prepare_calibrate_s')} | "
                    f"{stat(rows, 'new_highwater', 1 / MIB)} | {max(x['worst_kkt'] for x in rows):.2g} |"
                )
        if case in names:
            rows = [read(z) for z in sorted(d.glob("external-*.json"))]
            if rows:
                external.append(
                    f"| {name} | {names[case]} | {format_score(rows[0], case)} | {stat(rows, 'fit_s')} | "
                    f"{stat(rows, 'predict_s', 1000)} | {stat(rows, 'peak_above_start', 1 / MIB)} | "
                    f"{stat(rows, 'peak_rss', 1 / MIB)} |"
                )
        for mode, label in (
            ("tree", "Tree"),
            ("pmopt", "PMOpt strict"),
            ("pmopt-default", "PMOpt default"),
        ):
            records = [read(f) for f in sorted(d.glob(f"miner-{mode}-[0-9]*.json"))]
            if records:
                outcome = ", ".join(sorted({p["status"] for p in records}))
                passed = sum(p["status"] == "ok" for p in records)
                errors = [p["kkt"] for p in records if "kkt" in p]
                acc = [p["accuracy"] for p in records if "accuracy" in p]
                kkt = f"{max(errors):.2g}" if errors else ""
                score = f"{100 * np.median(acc):.2f}%" if acc else ""
                export = stat(records, "export_s") if "export_s" in records[0] else ""
                miners.append(
                    f"| {name} | {label} | {passed} / {len(records)} | {outcome} | {stat(records, 'fit_s')} | "
                    f"{stat(records, 'peak_rss', 1 / MIB)} | {export} | {kkt} | {score} |"
                )
    out = root / "MEASUREMENTS.md"
    out.write_text(
        "\n".join(
            head
            + [""]
            + controls
            + [""]
            + audit
            + [""]
            + external
            + [""]
            + miners
            + [""]
            + detail
            + [""]
            + path
            + [""]
            + val
        )
        + "\n",
        encoding="ascii",
    )
    if plot and comparisons:
        plot_comparisons(comparisons, plot)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threads": 1,
        "repetitions": args.repeat,
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "scipy",
                "scikit-learn",
                "numba",
                "celer",
                "psutil",
                "threadpoolctl",
                "networkx",
                "matplotlib",
            )
        },
    }
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="ascii"
    )
    print(summarize(args.output, args.cases, args.repeat, args.plot))
    if args.bundle:
        args.bundle.parent.mkdir(exist_ok=True, parents=True)
        with ZipFile(args.bundle, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(args.output.rglob("*")):
                if path.suffix in (".json", ".npy", ".md") and not path.name.endswith("-failure.json"):
                    archive.write(path, path.relative_to(args.output))
