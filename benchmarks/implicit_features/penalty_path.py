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

"""Warm four-penalty validation paths, including calibration and feature construction."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from datasets import dictionary_factory, problem, take
from experiment import RATIOS, penalty, prime, rss_peak, score
from solver import Fit, audit, cd, fit_tree, predict
from threadpoolctl import threadpool_limits


def timed(args):
    p = problem(args.case, args.data, args.arabic)
    x, y = take(p.x, p.train), p.y[p.train]
    xv = take(p.x, p.valid)
    prime()
    start_rss, old_peak = psutil.Process().memory_info().rss, rss_peak()
    t0 = perf_counter()
    factory = dictionary_factory(args.case, x)
    train, validation = factory(x), factory(xv)
    if args.method == "tree":
        amax = penalty(train, y)
    else:
        matrix, keys = train.explicit()
        amax = np.max(np.abs(matrix.T @ (y - y.mean(axis=0))), axis=0) / len(y)
    t1 = perf_counter()
    models, warm = [None] * y.shape[1], [None] * y.shape[1]
    history, fits = [], []
    for ratio in RATIOS:
        t = perf_counter()
        for j, v in enumerate(y.T):
            a = ratio * amax[j]
            if args.method == "tree":
                models[j] = fit_tree(train, v, a, previous=models[j])
            else:
                beta, b, r = cd(matrix, v, a, warm[j])
                warm[j] = beta
                idx = np.flatnonzero(beta)
                obj = 0.5 * np.mean(r * r) + a * np.abs(beta).sum()
                models[j] = Fit([keys[i] for i in idx], beta[idx], b, obj)
        pred = predict(validation, models)
        history.append(
            dict(ratio=ratio, fit_predict_s=perf_counter() - t, **score(p.kind, p.y[p.valid], pred))
        )
        fits.append(models.copy())
    elapsed = perf_counter() - t0
    peak = rss_peak()
    if args.method == "tree":
        matrix, keys = train.explicit()
    worst = max(
        audit(matrix, keys, v, ratio * a, model)
        for ratio, fit in zip(RATIOS, fits)
        for v, a, model in zip(y.T, amax, fit)
    )
    if worst > 1.01e-7:
        raise ArithmeticError(f"path KKT check failed: {worst:g}")
    return {
        "method": args.method,
        "full_path_s": elapsed,
        "prepare_calibrate_s": t1 - t0,
        "peak_rss": peak,
        "new_highwater": peak - old_peak,
        "rss_at_start": start_rss,
        "curves": history,
        "worst_kkt": worst,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arabic", type=Path)
    ap.add_argument("--case")
    ap.add_argument(
        "--cases", nargs="+", default=["gapped", "arabic-gapped", "sms", "mushroom", "intervals", "haar"]
    )
    ap.add_argument("--method", choices=["tree", "full"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    if args.case:
        with threadpool_limits(limits=1):
            result = timed(args)
        path = args.output / args.case / f"path-{args.method}-{args.rep}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
        print(args.case, args.method, result["full_path_s"], flush=True)
    else:
        for name in args.cases:
            for i in range(args.repeat):
                for method in ("tree", "full") if i % 2 == 0 else ("full", "tree"):
                    if (args.output / name / f"path-{method}-{i}.json").exists():
                        continue
                    cmd = [
                        sys.executable,
                        __file__,
                        "--case",
                        name,
                        "--method",
                        method,
                        "--rep",
                        str(i),
                        "--output",
                        str(args.output),
                        "--data",
                        str(args.data),
                    ]
                    if args.arabic:
                        cmd += ["--arabic", str(args.arabic)]
                    subprocess.run(cmd, check=True, timeout=600)
