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

"""Run matched working-set benchmarks. Each configuration gets a fresh process."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, help="Directory containing the old pysiglib package and CPU library")
    parser.add_argument("--prototype", action="store_true", help="Include the research prototype and its original tree bridge")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="+", choices=["gaussian", "anisotropic", "commutator", "arabic", "hall6"])
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.baseline:
        env["PYSIGLIB_BASELINE"] = str(args.baseline.resolve())
    tasks = []
    for case, extra in (("gaussian", ["logistic", "poisson", "huber"]), ("anisotropic", ["huber"]),
                        ("commutator", ["huber"]), ("arabic", ["logistic"]), ("hall6", ["poisson"])):
        if args.cases and case not in args.cases:
            continue
        modes = ["hall"] if case == "hall6" else ["sig", "log", "hall"]
        for mode in modes:
            for loss in ["squared", *extra]:
                for jobs in (1, 8):
                    tasks.append(("native", case, mode, loss, jobs))
                    if args.prototype:
                        tasks.append(("prototype", case, mode, loss, jobs))
                    if args.baseline and loss == "squared":
                        tasks.append(("legacy", case, mode, loss, jobs))
    random.Random(7203).shuffle(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    filename = {"win32": "cpsig.dll", "darwin": "libcpsig.dylib"}.get(sys.platform, "libcpsig.so")
    with Path(importlib.util.find_spec("pysiglib").origin).with_name(filename).open("rb") as library:
        build_id = hashlib.sha256(library.read()).hexdigest()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if row["case"] == "arabic" and row.get("workload_version") != 2:
                continue
            if row["variant"] == "native" and row.get("build_id") != build_id:
                continue
            if row["variant"] == "prototype" and args.baseline and row.get("prediction_backend") != "baseline":
                continue
            done.add(tuple(row[k] for k in ("variant", "case", "mode", "loss", "jobs")))
    with args.output.open("a", encoding="utf8") as out:
        for task in tasks:
            if task in done:
                continue
            command = [sys.executable, "-m", "benchmarks.benchmark_working_set", *map(str, task), "--repeats", str(args.repeats)]
            try:
                run = subprocess.run(command, text=True, capture_output=True, timeout=3600, env=env)
                if run.returncode:
                    raise RuntimeError(run.stderr[-2000:])
                row = json.loads(run.stdout.splitlines()[-1])
            except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                row = dict(zip(("variant", "case", "mode", "loss", "jobs"), task))
                row["error"] = str(exc)
            out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
