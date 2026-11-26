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

"""Compare ordinary signature tree orders on the frozen audio experiments."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


FOLDERS = dict(arabic="spoken_arabic_digit", insect="insect_wingbeat", speech="speech_commands")


def peak_rss():
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def fit(args, case, order, repeat):
    import pysiglib
    from threadpoolctl import threadpool_limits

    dataset, degree = case.split("-")
    degree = int(degree)
    records = json.loads(args.paper.read_text())
    if dataset == "dictionary":
        ref = records["dictionary_comparison"]["sig"][0]
        dataset = "arabic"
    else:
        ref = next(r["tree"] for r in records["classification"] if r["dataset"] == dataset and r["degree"] == degree)
    filename = args.data / FOLDERS[dataset] / "results/normalized/integral_comparison/prepared.npz"
    with np.load(filename) as data:
        x, y = np.array(data["train"], order="C", copy=True), np.array(data["response"], order="C", copy=True)
        tree, bound = order.split("_")
        with threadpool_limits(limits=1):
            before = peak_rss()
            start = perf_counter()
            model = pysiglib.sig_lasso(x, y, degree, ref["alpha"], tree_order=tree, terminal_bound=bound, n_jobs=8,
                                       fit_intercept=True, tol=1e-8, max_iter=3000)
            elapsed = perf_counter() - start
            peak = peak_rss()
            assert np.all(model["converged"]) and model["stats"]["full_transform_calls"] == 0
            prediction = pysiglib.sig_lasso_predict(data["test"], model, n_jobs=8)
            correct = int(np.sum(prediction.argmax(axis=1) == data["test_labels"]))
            train = model["signatures"] @ model["coef"] + model["intercept"]
            objective = .5 * ((y - train)**2).mean(axis=0) + ref["alpha"] * np.abs(model["coef"]).sum(axis=0)
            np.testing.assert_allclose(objective, model["objective"], rtol=1e-11, atol=1e-13)
    stem = args.output / f"{case}-{order}-{repeat}"
    np.savez(stem.with_suffix(".npz"), prediction=prediction, train=train,
             words=np.array([",".join(map(str, w)) for w in model["words"]]), coef=model["coef"])
    record = dict(case=case, order=order, repeat=repeat, seconds=elapsed, rss=peak, rss_increase=peak-before,
                  correct=correct, paper_correct=ref["correct"], test_samples=len(prediction),
                  words=len(model["words"]), max_kkt=float(np.max(model["kkt_violation"])),
                  objective=objective.tolist(), stats=model["stats"])
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(case, order, repeat, f"{elapsed:.3f}s", correct, flush=True)


def compare(args):
    for case in args.cases:
        for repeat in range(args.repeats):
            shift = repeat % len(args.orders)
            orders = args.orders[shift:] + args.orders[:shift]
            for order in orders:
                subprocess.run([sys.executable, __file__, "--data", str(args.data), "--paper", str(args.paper),
                                "--output", str(args.output), "--worker", case, order, str(repeat)], check=True)
            reference = args.output / f"{case}-{orders[0]}-{repeat}.npz"
            with np.load(reference) as first:
                for other in orders[1:]:
                    with np.load(args.output / f"{case}-{other}-{repeat}.npz") as second:
                        np.testing.assert_array_equal(first["words"], second["words"])
                        np.testing.assert_array_equal(first["prediction"].argmax(1), second["prediction"].argmax(1))
                        for key in ("prediction", "train", "coef"):
                            np.testing.assert_allclose(first[key], second[key], atol=1e-8, rtol=1e-8)


def audit_scores(args, case, order):
    from math import factorial
    import pysiglib

    dataset = case.split("-")[0]
    folder = FOLDERS["arabic" if dataset == "dictionary" else dataset]
    with np.load(args.data / folder / "results/normalized/integral_comparison/prepared.npz") as data, \
            np.load(args.output / f"{case}-{order}-0.npz") as fit:
        words = [tuple(map(int, w.split(","))) for w in fit["words"]]
        coeff = fit["coef"] * np.array([factorial(len(w)) for w in words])[:, None]
        differences = []
        for xkey, skey in (("train", "train"), ("test", "prediction")):
            x = data[xkey]
            idx = np.linspace(0, len(x) - 1, 16, dtype=int)
            x = x[idx]
            variation = abs(np.diff(x, axis=1)).max(axis=2).sum(axis=1)
            x = np.ascontiguousarray((x - x[:, :1]) / np.where(variation, variation, 1.)[:, None, None])
            independent = pysiglib.sig_coef(x, words, n_jobs=1) @ coeff
            differences.append(fit[skey][idx] - independent)
        # Each difference must equal the same intercept on both splits.
        error = np.max(abs(np.concatenate(differences) - differences[0][:1]))
        assert error < 1e-9, (case, error)
        return error


def report(args):
    records = [json.loads(p.read_text()) for p in args.output.glob("*.json") if p.stem.rsplit("-", 1)[-1].isdigit()]
    lines = ["# Grouped searches for ordinary signature features", "",
             "The comparison varies the signed terminal projection and the Lyndon factor ordering independently. "
             f"Each cell uses {args.repeats} fresh processes in rotated order, eight feature workers, "
             "one BLAS thread, float64, squared loss, an intercept and tolerance 1e-8. "
             "Times cover the complete fit with resident inputs. Loading, test prediction and auditing take place outside the timer.",
             "", f"Platform {platform.platform()}, Python {platform.python_version()}.", "",
             "The three audio tasks use the saved training and test splits. The penalty vectors come from the "
             "paper's archived measurements at revision 02da5ef18, including the reconstructed degree-four "
             "Speech Commands calibration. The separate Arabic dictionary comparison uses its own penalty.", "",
             "## Fit time", "", "Median seconds with the first and third quartiles in parentheses.", "",
             "| Task | " + " | ".join(s.replace("_", " / ") for s in args.orders) + " | Words | Test accuracy |",
             "|---|" + "---:|" * (len(args.orders) + 2)]
    details = ["## Memory and search work", "",
               "Memory columns give medians in MiB. RSS increase is the change in the process resident-memory "
               "high-water mark from immediately before fitting to its return. Native peak counts tracked fitting allocations. "
               "Terminal contractions count full signed feature columns evaluated in the final one or two degrees.", "",
               "| Task | Order / bound | Native peak | RSS increase | Peak process RSS | Terminal contractions |",
               "|---|---|---:|---:|---:|---:|"]
    totals = dict.fromkeys(args.orders, 0.)
    score_error = coef_error = obj_error = kkt = independent_error = 0.
    for case in args.cases:
        row = [case]
        for order in args.orders:
            group = sorted((r for r in records if r["case"] == case and r["order"] == order), key=lambda r: r["repeat"])
            if [r["repeat"] for r in group] != list(range(args.repeats)):
                raise ValueError(f"Incomplete measurements for {case}, {order}")
            lo, mid, hi = np.percentile([r["seconds"] for r in group], [25, 50, 75])
            totals[order] += mid
            row.append(f"{mid:.4f} ({lo:.4f}, {hi:.4f})")
            memory = [np.median([r["stats"]["peak_bytes"] for r in group]),
                      np.median([r["rss_increase"] for r in group]), np.median([r["rss"] for r in group])]
            info = [case, order.replace("_", " / "), *(f"{v / 2**20:.2f}" for v in memory),
                    f'{int(np.median([r["stats"]["terminal_correlations"] for r in group])):,}']
            details.append("| " + " | ".join(info) + " |")
            for r in group:
                assert r["correct"] == r["paper_correct"]
                kkt = max(kkt, r["max_kkt"])
                ref = args.output / f"{case}-{args.orders[0]}-{r['repeat']}"
                paired = args.output / f"{case}-{order}-{r['repeat']}.npz"
                with np.load(ref.with_suffix(".npz")) as a, np.load(paired) as b:
                    np.testing.assert_array_equal(a["words"], b["words"])
                    np.testing.assert_array_equal(a["coef"] != 0, b["coef"] != 0)
                    np.testing.assert_array_equal(a["prediction"].argmax(1), b["prediction"].argmax(1))
                    score_error = max(score_error, np.max(abs(a["prediction"] - b["prediction"])))
                    coef_error = max(coef_error, np.max(abs(a["coef"] - b["coef"]), initial=0.))
                other = json.loads(ref.with_suffix(".json").read_text())
                obj_error = max(obj_error, np.max(abs(np.asarray(r["objective"]) - other["objective"])))
        row += [str(r["words"]), f'{r["correct"] / r["test_samples"]:.2%}']
        lines.append("| " + " | ".join(row) + " |")
        independent_error = max(independent_error, audit_scores(args, case, args.orders[0]))
    lines.append("| Sum of medians | " + " | ".join(f"{v:.4f}" for v in totals.values()) + " | | |")
    lines.extend(["", f"All {len(args.cases)*len(args.orders)*args.repeats} fits converged and recorded zero full transforms. "
                  f"All paired supports and test labels agreed. Maximum paired absolute differences were {coef_error:.2g} "
                  f"in coefficients, {score_error:.2g} in test scores and {obj_error:.2g} in recomputed objectives. "
                  f"The largest solver-reported KKT violation was {kkt:.2g}. An independent signature coefficient "
                  f"kernel reproduced scores up to a common intercept on 16 training and 16 held-out recordings "
                  f"per configuration to {independent_error:.2g}.", "", *details])
    lines.extend(["", "## Reproduction", "", "From the built branch checkout, run the following in PowerShell.", "",
                  "```powershell", "git show 02da5ef18:benchmarks/results/prefix_suffix/paper_rerun/measurements.json | Set-Content -Encoding ascii paper.json",
                  "python benchmarks/benchmark_signature_tree_order.py --data PATH/TO/sig_branch_and_bound_experiments --paper paper.json --output tree_results",
                  "```", "", "SHA-256 hashes of the prepared input archives", ""])
    for name, folder in FOLDERS.items():
        file = args.data / folder / "results/normalized/integral_comparison/prepared.npz"
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        lines.append(f"- {name} `{digest.hexdigest()}`")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="sig_branch_and_bound_experiments directory")
    parser.add_argument("--paper", type=Path, required=True, help="paper_rerun/measurements.json at revision 02da5ef18")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--cases", nargs="+", default=[f"{name}-{d}" for name in FOLDERS for d in range(1, 5)] + ["dictionary-4"])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--orders", nargs="+", default=["prefix_norm", "lyndon_projected", "prefix_projected", "lyndon_norm"],
                        choices=["prefix_norm", "lyndon_projected", "prefix_projected", "lyndon_norm"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        fit(args, args.worker[0], args.worker[1], int(args.worker[2]))
    else:
        if not args.report_only:
            compare(args)
        report(args)
