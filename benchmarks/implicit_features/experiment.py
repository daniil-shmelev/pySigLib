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

"""Reproducible single-core timing, validation and independent full-dictionary audits."""

import argparse
import gc
import json
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from datasets import CASES, dictionary_factory, problem, take
from graphs import canonical
from intervals import HistogramScan, Intervals
from products import Products
from scattering import Haar
from scipy import sparse
from sequences import Sequences
from solver import Fit, audit, cd, fit_tree, predict
from subsequences import Subsequences
from threadpoolctl import threadpool_limits
from walsh import Walsh, WalshTransform

RATIOS = (0.3, 0.1, 0.03, 0.01)
METHODS = ("tree", "full", "matrix-ws", "histogram", "fwht", "distinct")


class ExplicitSearch:
    def __init__(self, matrix, keys):
        self.matrix, self.keys = matrix, keys
        self.index = {k: j for j, k in enumerate(keys)}
        self.n = matrix.shape[0]

    def search(self, r, floor, k):
        scores = np.abs(self.matrix.T @ r)
        hits = np.flatnonzero(scores > floor)
        if len(hits) > k:
            hits = hits[np.argpartition(scores[hits], -k)[-k:]]
        return sorted(((scores[j], self.keys[j]) for j in hits), reverse=True), len(scores), 0

    def columns(self, keys):
        x = self.matrix[:, [self.index[k] for k in keys]]
        return x.toarray() if sparse.issparse(x) else x


def full(matrix, keys, y, alpha):
    coef, b, r = cd(matrix, y, alpha)
    support = np.flatnonzero(coef)
    value = 0.5 * np.mean(r**2) + alpha * np.abs(coef).sum()
    model = Fit([keys[j] for j in support], coef[support], b, value)
    return model


def fit(method, reference, y, alpha):
    if method in ("full", "distinct"):
        matrix, keys = reference
        return [full(matrix, keys, v, a) for v, a in zip(y.T, alpha)]
    return [fit_tree(reference, v, a) for v, a in zip(y.T, alpha)]


def score(kind, y, pred):
    mse = float(np.mean((y - pred) ** 2))
    if kind == "binary":
        accuracy = np.mean((pred >= 0.5) == y)
    elif kind == "multiclass":
        accuracy = np.mean(pred.argmax(axis=1) == y.argmax(axis=1))
    else:
        return {"rmse": np.sqrt(mse), "mse": mse}
    return {"accuracy": float(accuracy), "mse": mse}


def rss_peak():
    mem = psutil.Process().memory_info()
    if hasattr(mem, "peak_wset"):
        return mem.peak_wset
    import resource

    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


class FitMemory:
    def __init__(self):
        self.process = psutil.Process()
        self.base = self.process.memory_info().rss
        self.previous_peak = rss_peak()
        self.sampled = self.base
        self.done = threading.Event()
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()

    def _sample(self):
        while not self.done.wait(0.002):
            self.sampled = max(self.sampled, self.process.memory_info().rss)

    def finish(self):
        peak = rss_peak()
        self.done.set()
        self.worker.join()
        observed = max(
            self.sampled, self.process.memory_info().rss, peak if peak > self.previous_peak else 0
        )
        return {
            "peak_rss": peak,
            "peak_above_start": max(0, observed - self.base),
            "new_highwater": peak - self.previous_peak,
            "rss_at_start": self.base,
        }


def prime():
    # Compile on unrelated tiny data before timing. Do not retain target feature states.
    x = np.array([[1.0, 0.0, -0.2, 0.3], [0.4, 1.0, 0.6, 0.3]])
    for d in (
        Products(x, [0, 1], [1, 1], 2),
        Sequences([[2], [1], [2, 1], [1, 2]], 2, 3),
        Haar(x, 2, 2),
        Subsequences([[0, 1], [1], [0], [1, 0]], 2, 2),
        Intervals([[1, 0, 1, 2]], [3]),
    ):
        r = np.arange(d.n, dtype=float) - 0.5
        hits, _, _ = d.search(r, 0, 2)
        d.columns([key for _, key in hits])
        d.explicit()
        if isinstance(d, Subsequences):
            d.distinct()
    parity = Walsh(np.array([[0, 1, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1]]))
    parity.search(np.array([0.5, -0.5]), 0.0, 1)
    WalshTransform(parity).search(np.array([0.5, -0.5]), 0.0, 1)
    parity.explicit()
    full(x.T.copy(order="F"), [0, 1], np.array([0.0, 1.0, 1.0, 0.0]), 0.02)
    canonical.cache_clear()
    gc.collect()


def penalty(dictionary, y):
    out = []
    for col in y.T:
        best, _, _ = dictionary.search(col - col.mean(), 0.0, 1)
        out.append(best[0][0] / len(col) if best else 0.0)
    if min(out) <= 0:
        raise ValueError("degenerate response or dictionary")
    return np.array(out)


def prepare_search(method, dictionary):
    if method == "tree":
        return dictionary
    if method == "histogram":
        return HistogramScan(dictionary.bins, dictionary.sizes)
    if method == "fwht":
        return WalshTransform(dictionary)
    if method == "distinct":
        return dictionary.distinct()
    matrix, keys = dictionary.explicit()
    return (matrix, keys) if method == "full" else ExplicitSearch(matrix, keys)


def tune(p, case):
    x = take(p.x, p.train)
    y = p.y[p.train]
    factory = dictionary_factory(case, x)
    train = factory(x)
    val = factory(take(p.x, p.valid))
    matrix, keys = train.explicit()
    amax = np.max(np.abs(matrix.T @ (y - y.mean(axis=0))), axis=0) / len(y)
    curves = []
    for ratio in RATIOS:
        ms = fit("full", (matrix, keys), y, amax * ratio)
        err = max(audit(matrix, keys, v, a, m) for v, a, m in zip(y.T, amax * ratio, ms))
        if err > 1.01e-7:
            raise ArithmeticError(f"validation fit failed KKT {err:g}")
        pred = predict(val, ms)
        curves.append(dict(ratio=ratio, **score(p.kind, p.y[p.valid], pred), kkt=err))
    chosen = min(curves, key=lambda z: (-z.get("accuracy", 0.0), z["mse"], -z["ratio"]))
    idx = np.r_[p.train, p.valid] if p.refit else p.train
    x, y = take(p.x, idx), p.y[idx]
    train = dictionary_factory(case, x)(x)
    matrix, keys = train.explicit()
    amax = np.max(np.abs(matrix.T @ (y - y.mean(axis=0))), axis=0) / len(y)
    tree_max = penalty(train, y)
    np.testing.assert_allclose(tree_max, amax, rtol=1e-11, atol=1e-13)
    return {
        "curves": curves,
        "ratio": chosen["ratio"],
        "alpha": (amax * chosen["ratio"]).tolist(),
        "alpha_max": amax.tolist(),
        "n_features": len(keys),
        "n_train": len(y),
        "n_test": len(p.test),
        "storage_bytes": (
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            if sparse.issparse(matrix)
            else matrix.nbytes
        ),
    }


def measure(p, case, method, config, save):
    idx = np.r_[p.train, p.valid] if p.refit else p.train
    x, y, alpha = take(p.x, idx), p.y[idx], config["alpha"]
    xt = take(p.x, p.test)
    gc.collect()
    sample = FitMemory()
    t = perf_counter()
    factory = dictionary_factory(case, x)
    d = factory(x)
    search = prepare_search(method, d)
    t1 = perf_counter()
    models = fit(method, search, y, alpha)
    t2 = perf_counter()
    memory = sample.finish()
    pred = predict(factory(xt), models)
    t3 = perf_counter()
    # Exhaustive feature-wise audit outside both timings, including active signs.
    matrix, keys = (
        d.explicit()
        if method in ("tree", "histogram", "fwht", "distinct")
        else ((search.matrix, search.keys) if method == "matrix-ws" else search)
    )
    err = [audit(matrix, keys, v, a, m) for v, a, m in zip(y.T, alpha, models)]
    result = dict(
        method=method,
        fit_s=t2 - t,
        prepare_s=t1 - t,
        solve_search_s=t2 - t1,
        predict_s=t3 - t2,
        **memory,
        kkt=max(err),
        objective=[m.objective for m in models],
        support=[len(m.keys) for m in models],
        union=len({k for m in models for k in m.keys}),
        visited=sum(m.visited for m in models),
        pruned=sum(m.pruned for m in models),
        rounds=sum(m.rounds for m in models),
        distinct_columns=len(search[1]) if method == "distinct" else None,
        **score(p.kind, p.y[p.test], pred),
    )
    if max(err) > 1.01e-7 or not np.isfinite(pred).all():
        raise ArithmeticError(f"full dictionary KKT or finite check failed: {result}")
    np.save(save, pred, allow_pickle=False)
    return result


def run(args):
    p = problem(args.case, args.data or args.output / "data", args.arabic, args.fold)
    dest = args.output / (args.case + (f"-f{args.fold}" if args.fold else ""))
    dest.mkdir(parents=True, exist_ok=True)
    config = dest / "validation.json"
    with threadpool_limits(limits=1):
        prime()
        if args.stage == "tune":
            res, path = tune(p, args.case), config
        else:
            path = dest / f"{args.method}-{args.rep}.json"
            res = measure(
                p, args.case, args.method, json.loads(config.read_text()), path.with_suffix(".npy")
            )
    path.write_text(json.dumps(res, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(args.case, path.stem, res.get("fit_s", res.get("ratio")), flush=True)


def suite(args):
    failures = []
    for case in args.cases:
        dest = args.output / (case + (f"-f{args.fold}" if args.fold else ""))
        jobs = [("tune", "full", 0, "validation")]
        methods = [
            m
            for m in args.methods
            if (m != "histogram" or case == "intervals")
            and (m != "fwht" or case == "walsh")
            and (
                m != "distinct"
                or case in ("gapped", "wafer-gapped", "ecg-gapped", "arabic-gapped", "arabic-frame-words")
            )
        ]
        for i in range(args.repeat):
            for j in range(len(methods)):
                m = methods[(i + j) % len(methods)]
                jobs.append(("measure", m, i, f"{m}-{i}"))
        for stage, method, rep, key in jobs:
            if (dest / (key + ".json")).exists():
                continue
            cmd = [
                sys.executable,
                __file__,
                "--stage",
                stage,
                "--method",
                method,
                "--rep",
                str(rep),
                "--case",
                case,
                "--fold",
                str(args.fold),
                "--output",
                str(args.output),
            ]
            if args.arabic:
                cmd += ["--arabic", str(args.arabic)]
            if args.data:
                cmd += ["--data", str(args.data)]
            try:
                subprocess.run(cmd, check=True, timeout=args.timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f"FAILED {case} {key}: {exc}", flush=True)
                dest.mkdir(parents=True, exist_ok=True)
                failures.append((case, key))
                info = {
                    "case": case,
                    "stage": stage,
                    "method": method,
                    "repetition": rep,
                    "error": str(exc),
                }
                (dest / (key + "-failure.json")).write_text(
                    json.dumps(info, indent=2) + "\n", encoding="ascii"
                )
                if stage == "tune":
                    break
    if failures:
        raise SystemExit(f"incomplete benchmarking jobs: {failures}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("suite", "tune", "measure"), default="suite")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--arabic", type=Path)
    ap.add_argument("--cases", choices=CASES, nargs="+", default=CASES)
    ap.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--method", choices=METHODS, default="tree")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if args.stage == "suite":
        suite(args)
    else:
        run(args)
