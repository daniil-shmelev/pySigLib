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

"""Compare the working-set search with Yoshida et al.'s native safe pattern pruning."""

import argparse
import csv
import gc
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

import numpy as np
import psutil
from datasets import dictionary_factory, problem, take
from experiment import FitMemory, fit, penalty, prime, score
from solver import Fit, audit, predict
from threadpoolctl import threadpool_limits

CASES = ("gapped", "ecg-gapped", "wafer-gapped", "mushroom", "sms")


def write_input(case, d, y, path):
    if case == "mushroom":
        rows = (np.flatnonzero(col) + 1 for col in d.factors.T)
        kind, base, skip = "itemset", len(d.factors) + 1, 0
    elif case == "sms":
        docs = np.split(d.tokens, np.flatnonzero(np.diff(d.doc)) + 1)
        rows = (row[row != 0] for row in docs)
        kind, base, skip = "sequence", d.base, 0
    else:
        # The token at position j is the unique symbol with next endpoint j + 1.
        rows = (
            (d.table[a:b] == np.arange(a + 1, b + 1)[:, None]).argmax(axis=1) + 1
            for a, b in zip(d.starts, d.ends)
        )
        kind, base, skip = "sequence", d.base, int(max(d.ends - d.starts))
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow((kind, "objective"))
        writer.writerows((" ".join(map(str, row)), float(v)) for row, v in zip(rows, y))
    return base, skip


def execute(cmd, directory, timeout, limit):
    start = perf_counter()
    peak = 0
    with (directory / "log.txt").open("w") as logfile:
        proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)
        child = psutil.Process(proc.pid)
        status = "ok"
        while proc.poll() is None:
            try:
                info = child.memory_info()
                peak = max(peak, getattr(info, "peak_wset", info.rss))
            except psutil.NoSuchProcess:
                break
            if peak > limit or perf_counter() - start > timeout:
                status = "memory limit" if peak > limit else "timeout"
                proc.kill()
                break
            sleep(0.001)
        code = proc.wait()
    seconds = perf_counter() - start
    if code and status == "ok":
        raise RuntimeError((directory / "log.txt").read_text(errors="replace")[-3000:])
    return {"status": status, "fit_s": seconds, "peak_rss": peak}


def measure(args):
    p = problem(args.case, args.data)
    idx = np.r_[p.train, p.valid]
    x, y = take(p.x, idx), p.y[idx]
    case_path = args.output / args.case
    cfg = json.loads((case_path / "validation.json").read_text())
    prime()
    gc.collect()
    if args.method == "tree":
        mem = FitMemory()
        start = perf_counter()
        factory = dictionary_factory(args.case, x)
        d = factory(x)
        alpha = penalty(d, y) * cfg["ratio"]
        m = fit("tree", d, y, alpha)[0]
        record = dict(status="ok", fit_s=perf_counter() - start, **mem.finish())
    else:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            start = perf_counter()
            factory = dictionary_factory(args.case, x)
            d = factory(x)
            base, skip = write_input(args.case, d, y[:, 0], folder / "train.csv")
            export_s = perf_counter() - start
            tolerance = 1e-4 if args.method == "pmopt-default" else args.tolerance
            cmd = [
                str(args.binary.resolve()),
                "mine",
                "predictive",
                str(folder / "train.csv"),
                "-o",
                str(folder),
                "--loss",
                "squared",
                "--no-l2",
                "--maxpat",
                str(d.degree),
                "--maxiter",
                "200000",
                "--maxtrial",
                "2",
                "--tol",
                str(tolerance),
                "--seed",
                "603",
                "--l1-step",
                str(-np.log10(cfg["ratio"])),
                "--l1-min",
                str(0.99 * cfg["ratio"]),
                "--verbose",
                "0",
            ]
            if skip:
                cmd += ["--max-skip", str(skip)]
            record = execute(cmd, folder, args.timeout, args.memory_mib * 2**20)
            record.update(export_s=export_s, tolerance=tolerance)
            if record["status"] != "ok":
                return record
            with (folder / "models_0.csv").open() as stream:
                rows = [row for row in csv.DictReader(stream) if int(row["id"]) == 1]
            weights, intercept = {}, None
            for row in rows:
                coef = float(row["coefficient"])
                digits = list(map(int, row["pattern"].split()))
                if not digits:
                    intercept = coef
                    continue
                if args.case == "mushroom":
                    digits.sort()
                key = 0
                for digit in digits:
                    key = key * base + digit
                if coef:
                    weights[key] = coef
            if intercept is None:
                raise ArithmeticError("missing final PMOpt intercept")
            m = Fit(list(weights), np.fromiter(weights.values(), float), intercept, 0.0)
            alpha = np.array(cfg["alpha"])
    np.testing.assert_allclose(alpha, cfg["alpha"], rtol=1e-10)
    matrix, keys = d.explicit()
    err = audit(matrix, keys, y[:, 0], alpha[0], m)
    residual = y[:, 0] - predict(d, [m])[:, 0]
    objective = residual @ residual / (2 * len(y)) + alpha[0] * np.abs(m.coef).sum()
    test = predict(factory(take(p.x, p.test)), [m])
    record.update(kkt=err, objective=objective, support=len(m.keys), **score(p.kind, p.y[p.test], test))
    if not np.isfinite(err) or err > 1.01e-7:
        record["status"] = "KKT tolerance exceeded"
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    ap.add_argument("--case", choices=CASES)
    methods = ("tree", "pmopt", "pmopt-default")
    ap.add_argument("--method", choices=methods)
    ap.add_argument("--methods", nargs="+", choices=methods, default=methods[:2])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--memory-mib", type=int, default=2048)
    ap.add_argument("--tolerance", type=float, default=1e-12)
    args = ap.parse_args()
    if args.case:
        with threadpool_limits(1):
            result = measure(args)
        name = f"miner-{args.method}-{args.rep}.json"
        (args.output / args.case / name).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(args.case, args.method, result["status"], result["fit_s"], flush=True)
    else:
        for rep in range(args.repeat):
            for case in args.cases:
                for method in args.methods if rep % 2 == 0 else reversed(args.methods):
                    dest = args.output / case / f"miner-{method}-{rep}.json"
                    if dest.exists():
                        continue
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--binary",
                            str(args.binary),
                            "--output",
                            str(args.output),
                            "--data",
                            str(args.data),
                            "--case",
                            case,
                            "--method",
                            method,
                            "--rep",
                            str(rep),
                            "--timeout",
                            str(args.timeout),
                            "--memory-mib",
                            str(args.memory_mib),
                            "--tolerance",
                            str(args.tolerance),
                        ],
                        check=True,
                    )
