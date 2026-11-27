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
import os
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


FOLDERS = dict(arabic="spoken_arabic_digit", insect="insect_wingbeat", speech="speech_commands")
ORDERS = ("prefix", "lyndon", "suffix", "energy", "profile", "forward", "grouped", "outside", "balanced", "adaptive")
BOUNDS = ("norm", "projected", "primitive")


def peak_rss():
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def fit(args, case, order, repeat):
    import pysiglib
    from pysiglib.load_siglib import CPSIG
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
        permutation = np.random.default_rng(28011).permutation(x.shape[-1]) if args.symmetry == "permute" else None

        def transform(paths):
            if permutation is not None:
                return np.take(paths, permutation, axis=-1)
            if args.symmetry == "reverse":
                return np.ascontiguousarray(-paths[:, ::-1, :])
            return paths

        x = transform(x)
        search = {}
        if order != "original":
            tree, bound = order.split("_")
            search = dict(tree_order=tree, terminal_bound=bound)
        with threadpool_limits(limits=1):
            before = peak_rss()
            start = perf_counter()
            model = pysiglib.sig_lasso(x, y, degree, ref["alpha"], n_jobs=args.jobs, fit_intercept=True,
                                       tol=1e-8, max_iter=3000, **search)
            elapsed = perf_counter() - start
            peak = peak_rss()
            assert np.all(model["converged"]) and model["stats"]["full_transform_calls"] == 0
            prediction = pysiglib.sig_lasso_predict(transform(data["test"]), model, n_jobs=args.jobs)
            correct = int(np.sum(prediction.argmax(axis=1) == data["test_labels"]))
            train = model["signatures"] @ model["coef"] + model["intercept"]
            objective = .5 * ((y - train)**2).mean(axis=0) + ref["alpha"] * np.abs(model["coef"]).sum(axis=0)
            np.testing.assert_allclose(objective, model["objective"], rtol=1e-11, atol=1e-13)
    stem = args.output / f"{case}-{order}-{repeat}"
    words, coef = model["words"], model["coef"]
    if args.symmetry != "identity":
        words = ([tuple(permutation[a] for a in w) for w in words] if permutation is not None
                 else [w[::-1] for w in words])
        rows = sorted(range(len(words)), key=lambda j: (len(words[j]), words[j]))
        words, coef = [words[j] for j in rows], coef[rows]
    np.savez(stem.with_suffix(".npz"), prediction=prediction, train=train,
             words=np.array([",".join(map(str, w)) for w in words]), coef=coef)
    record = dict(case=case, order=order, repeat=repeat, symmetry=args.symmetry, jobs=args.jobs, seconds=elapsed, rss=peak, rss_increase=peak-before,
                  native_sha256=hashlib.sha256(Path(CPSIG._name).read_bytes()).hexdigest(),
                  correct=correct, paper_correct=ref["correct"], test_samples=len(prediction),
                  words=len(model["words"]), max_kkt=float(np.max(model["kkt_violation"])),
                  objective=objective.tolist(), stats=model["stats"])
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(case, order, repeat, f"{elapsed:.3f}s", correct, flush=True)


def compare(args):
    for case in args.cases:
        for repeat in range(args.repeats):
            shift = (repeat % len(args.orders) if args.repeats >= len(args.orders)
                     else repeat * len(args.orders) // args.repeats)
            orders = args.orders[shift:] + args.orders[:shift]
            for order in orders:
                environment = dict(os.environ)
                if order == "original" and args.original_pythonpath:
                    environment["PYTHONPATH"] = str(args.original_pythonpath.resolve())
                subprocess.run([sys.executable, __file__, "--data", str(args.data), "--paper", str(args.paper),
                                "--output", str(args.output), "--jobs", str(args.jobs), "--symmetry", args.symmetry,
                                "--worker", case, order, str(repeat)],
                               check=True, env=environment)
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


def audit_endpoint_pairs(args):
    rows = []
    for case in args.cases:
        dataset, level = case.split("-")
        if dataset == "dictionary" or int(level) not in (3, 4):
            continue
        with np.load(args.data / FOLDERS[dataset] / "results/normalized/integral_comparison/prepared.npz") as data:
            x, y = data["train"], data["response"]
        ref = next(r["tree"] for r in json.loads(args.paper.read_text())["classification"]
                   if r["dataset"] == dataset and r["degree"] == int(level))
        alpha = np.asarray(ref["alpha"])
        with np.load(args.output / f"{case}-prefix_norm-0.npz") as baseline:
            residuals = {"initial": y - y.mean(axis=0), "fitted": y - baseline["train"]}
        dv = np.max(np.abs(np.diff(x, axis=1)), axis=-1)
        total = dv.sum(axis=1)
        scale = np.where(total > 0., total, 1.)
        dv /= scale[:, None]
        x = (x - x[:, :1]) / scale[:, None, None]
        before = np.cumsum(dv, axis=1) - dv
        after = np.cumsum(dv[:, ::-1], axis=1)[:, ::-1] - dv
        k = int(level) - 2

        def marginal(z, start, end):
            a, b = z[:, :-1], z[:, 1:]
            both = (a + b)**2
            return np.sqrt((k / 12 * dv[:, :, None] *
                           ((2*a*a + both) * start[:, :, None]**(k-1) +
                            (2*b*b + both) * end[:, :, None]**(k-1))).sum(axis=1))

        left = marginal(x, after + dv, after)
        right = marginal(x[:, -1:] - x, before, before + dv)
        for stage, r in residuals.items():
            bound = np.zeros((x.shape[-1], x.shape[-1]))
            for t in range(r.shape[1]):
                products = (np.abs(r[:, t:t+1]) * left).T @ right
                np.maximum(bound, (k+2)*(k+1) * products / (len(x)*alpha[t]), out=bound)
            rows.append(dict(case=case, stage=stage, excluded=int(np.sum(bound < 1)), pairs=bound.size))
    (args.output / "endpoint-audit.json").write_text(json.dumps(rows, indent=2))


def synthetic(args):
    import pysiglib
    from threadpoolctl import threadpool_limits

    cases = ("baseline", "samples", "length", "channels", "degree", "anisotropic")
    records = []
    with threadpool_limits(limits=1):
        for case in cases:
            folder = args.synthetic_data / f"{case}_7201"
            degree = json.loads((folder / "prepared.json").read_text())["degree"]
            words = {tuple(w): j for j, w in enumerate(json.loads((folder / "sig_words.json").read_text()))}
            design = np.load(folder / "sig.npy", mmap_mode="r")
            held = np.load(folder / "sig_test.npy", mmap_mode="r")
            with np.load(folder / "data.npz") as data, np.load(folder / "squared_response.npz") as responses:
                x, test = data["train"].copy(), data["test"].copy()
                y = responses["y"][:, 1].copy()
            alpha = .03 * np.max(abs(design.T @ (y - y.mean()))) / len(y)
            reference = None
            for rep in range(-1, args.repeats):
                for j in range(len(args.orders)):
                    order = args.orders[(j + rep + 1) % len(args.orders)]
                    tree, bound = order.split("_")
                    start = perf_counter()
                    model = pysiglib.sig_lasso(x, y, degree, alpha, tree_order=tree, terminal_bound=bound,
                                              tol=1e-8, max_iter=3000, n_jobs=args.jobs)
                    elapsed = perf_counter() - start
                    cols = [words[w] for w in model["words"]]
                    assert model["converged"] and model["stats"]["full_transform_calls"] == 0
                    np.testing.assert_allclose(model["signatures"], design[:, cols], atol=1e-12)
                    pred = held[:, cols] @ model["coef"] + model["intercept"]
                    np.testing.assert_allclose(pysiglib.sig_lasso_predict(test, model, n_jobs=args.jobs), pred, atol=1e-11)
                    residual = y - design[:, cols] @ model["coef"] - model["intercept"]
                    correlations = design.T @ residual / len(y)
                    kkt = max(np.max(abs(correlations)) - alpha, abs(residual.mean()),
                              np.max(abs(correlations[cols] - alpha * np.sign(model["coef"])), initial=0.))
                    assert kkt <= 1.01e-8
                    if reference is None:
                        reference = model["words"], pred
                    assert model["words"] == reference[0]
                    np.testing.assert_allclose(pred, reference[1], atol=1e-9)
                    if rep >= 0:
                        records.append(dict(case=case, degree=degree, shape=list(x.shape), order=order, repeat=rep,
                                            seconds=elapsed, words=len(cols), alpha=float(alpha), kkt=float(kkt),
                                            stats=model["stats"]))
    (args.output / "synthetic.json").write_text(json.dumps(records, indent=2))


def report_synthetic(args):
    records = json.loads((args.output / "synthetic.json").read_text())
    lines = ["# Single-response signature regression", "",
             f"Seed 7201, sparse signature target, 0.03 times the training maximum correlation, {args.jobs} worker(s). "
             "One warm-up per method and case precedes the timed fits. The dense reference matrices are outside fit timing. "
             "The memory column is the tracked native peak.", "",
             "| Case | Order / bound | Median fit ms (Q1, Q3) | Native MiB | Words | Max dense KKT violation |",
             "|---|---|---:|---:|---:|---:|"]
    for case in dict.fromkeys(r["case"] for r in records):
        for order in args.orders:
            group = [r for r in records if r["case"] == case and r["order"] == order]
            if len(group) != args.repeats:
                raise ValueError(f"Incomplete synthetic measurements for {case}, {order}")
            lo, mid, hi = 1000 * np.percentile([r["seconds"] for r in group], [25, 50, 75])
            peak = np.median([r["stats"]["peak_bytes"] for r in group]) / 2**20
            lines.append(f"| {case} | {order} | {mid:.2f} ({lo:.2f}, {hi:.2f}) | {peak:.2f} | "
                         f"{group[0]['words']} | {max(r['kkt'] for r in group):.2g} |")
    lines.extend(["", "The selected training columns and test predictions were checked against the full reference "
                  "designs. Each order recovered the same support within each case."])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def report(args):
    records = [json.loads(p.read_text()) for p in args.output.glob("*.json") if p.stem.rsplit("-", 1)[-1].isdigit()]
    if any(r.get("symmetry", "identity") != args.symmetry or r.get("jobs", 8) != args.jobs for r in records):
        raise ValueError("The measurement directory mixes path symmetries or worker counts")
    lines = ["# Grouped searches for ordinary signature features", "",
             "Each configuration specifies its word or channel order and its terminal bound. "
             f"Each cell uses {args.repeats} fresh processes in rotated order, {args.jobs} feature worker(s), "
             "one BLAS thread, float64, squared loss, an intercept and tolerance 1e-8. "
             "Times cover the complete fit with resident inputs. Loading, test prediction and auditing take place outside the timer.",
             "", f"Platform {platform.platform()}, Python {platform.python_version()}. Path symmetry: {args.symmetry}.", "",
             "All updated variants share the register-based signed terminal contraction. "
             "An original column, when present, runs the pre-change prefix implementation from a separate library.", "",
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
    case_medians = {order: [] for order in args.orders}
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
            if not case.startswith("dictionary-"):
                case_medians[order].append([mid, *memory])
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
                  f"per experimental setting to {independent_error:.2g}."])
    if case_medians[args.orders[0]]:
        lines += ["", "### Classification tasks", "",
                  "Time is the sum of the case medians. Each memory entry is the largest case median. "
                  "The extra Arabic dictionary penalty is excluded from this summary.", "",
                  "| Order / bound | Fit s | Native peak MiB | RSS increase MiB | Peak process RSS MiB |",
                  "|---|---:|---:|---:|---:|"]
        for order, rows in case_medians.items():
            array = np.asarray(rows)
            mem = " | ".join(f"{v:.2f}" for v in np.max(array[:, 1:], axis=0) / 2**20)
            lines.append(f"| {order} | {array[:, 0].sum():.4f} | {mem} |")
    lines.extend(["", *details])
    lines.extend(["", "## Reproduction", "", "From the built branch checkout, run the following in PowerShell.", "",
                  "```powershell", "git show 02da5ef18:benchmarks/results/prefix_suffix/paper_rerun/measurements.json | Set-Content -Encoding ascii paper.json",
                  "python benchmarks/benchmark_signature_tree_order.py --data PATH/TO/sig_branch_and_bound_experiments "
                  f"--paper paper.json --output tree_results --repeats {args.repeats} --orders " +
                  " ".join(o for o in args.orders if o != "original") + " --cases " + " ".join(args.cases) +
                  f" --symmetry {args.symmetry} --jobs {args.jobs}" +
                  (" --original-pythonpath PATH/TO/PRECHANGE/PACKAGE_PARENT" if args.original_pythonpath else ""),
                  "```", "", "SHA-256 hashes of the prepared input archives", ""])
    for name, folder in FOLDERS.items():
        file = args.data / folder / "results/normalized/integral_comparison/prepared.npz"
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        lines.append(f"- {name} `{digest.hexdigest()}`")
    hashes = {order: {r["native_sha256"] for r in records if r["order"] == order} for order in args.orders}
    if any(len(h) != 1 for h in hashes.values()):
        raise ValueError("Each configuration must use one native library throughout")
    updated = set.union(*(h for order, h in hashes.items() if order != "original"))
    if len(updated) != 1 or ("original" in hashes and hashes["original"] == updated):
        raise ValueError("Expected a shared updated native library and a distinct original control")
    lines += ["", f"Updated native library SHA-256 `{next(iter(updated))}`."]
    if "original" in hashes:
        lines.append(f"Pre-change native library SHA-256 `{next(iter(hashes['original']))}`.")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="sig_branch_and_bound_experiments directory")
    parser.add_argument("--paper", type=Path, help="paper_rerun/measurements.json at revision 02da5ef18")
    parser.add_argument("--synthetic-data", type=Path, help="Run the saved single-response synthetic benchmarks instead")
    parser.add_argument("--symmetry", choices=("identity", "reverse", "permute"), default="identity",
                        help="Relabel the audio task by reversing increments or permuting channels with seed 28011")
    parser.add_argument("--original-pythonpath", type=Path, help="Include a separate pre-change package in the audio comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--endpoint-audit", action="store_true",
                        help="Audit the endpoint bound at the initial and fitted residuals of saved prefix fits")
    parser.add_argument("--cases", nargs="+", default=[f"{name}-{d}" for name in FOLDERS for d in range(1, 5)] + ["dictionary-4"])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8, help="CPU workers per fit")
    parser.add_argument("--orders", nargs="+", default=["prefix_norm", "lyndon_projected", "prefix_projected", "lyndon_norm"],
                        choices=[f"{order}_{bound}" for order in ORDERS for bound in BOUNDS])
    args = parser.parse_args()
    if args.original_pythonpath and not args.worker:
        args.orders = ["original", *args.orders]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.synthetic_data:
        if args.symmetry != "identity" or args.original_pythonpath:
            parser.error("symmetry and original-library controls apply to the audio experiments")
        if not args.report_only:
            synthetic(args)
        report_synthetic(args)
    elif args.data is None or args.paper is None:
        parser.error("the audio comparison requires --data and --paper")
    elif args.endpoint_audit:
        audit_endpoint_pairs(args)
    elif args.worker:
        fit(args, args.worker[0], args.worker[1], int(args.worker[2]))
    else:
        if not args.report_only:
            compare(args)
        report(args)
