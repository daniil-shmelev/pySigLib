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

"""Run the batch-32 prefix-suffix length grids with bounded cell runtime."""

import argparse
import ast
import contextlib
import importlib.util
import json
import multiprocessing
import os
import queue
from pathlib import Path
import sys
import time
import traceback

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "tmp" / "prefix_suffix_dispatch" / "heatmaps_simple"
DIMENSIONS = (2, 3, 4, 6, 8, 10, 12, 16, 24, 32)
DEGREES = tuple(range(2, 17))
LIMIT = 120


def cell_worker(script, requests, results):
    module = None
    while True:
        request = requests.get()
        if request is None: return
        arguments, log_path = request
        code = 0
        with open(log_path, 'w', encoding='ascii') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                sys.argv = arguments
                if module is None:
                    specification = importlib.util.spec_from_file_location('heatmap_cell', script)
                    module = importlib.util.module_from_spec(specification)
                    specification.loader.exec_module(module)
                module.main()
            except BaseException:
                traceback.print_exc()
                code = 1
        results.put(code)


class CellWorker:
    def __init__(self, script):
        self.script = script
        self.process = None

    def close(self):
        if self.process is None: return
        if self.process.is_alive():
            self.requests.put(None)
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.process.join()
        self.requests.close()
        self.results.close()
        self.process = None

    def run(self, arguments, log_path, timeout):
        if self.process is None:
            context = multiprocessing.get_context('spawn')
            self.requests, self.results = context.Queue(), context.Queue()
            self.process = context.Process(target=cell_worker, args=(str(self.script), self.requests, self.results), daemon=True)
            self.process.start()
        self.requests.put((arguments, str(log_path)))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.process.terminate()
                self.process.join()
                self.close()
                raise TimeoutError
            try:
                return self.results.get(timeout=min(1, remaining))
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(f'Benchmark worker exited with code {self.process.exitcode}')


def save(path, report):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19: raise
            time.sleep(0.1)


def estimate_function(script):
    tree = ast.parse(script.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "estimate")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["estimate"]


def run_grid(segments, kind, device):
    jobs = 20 if device == "cpu" else 1
    folder = ROOT / f"segments{segments}"
    target_dir = folder / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{device}.json"
    script = REPO / "benchmarks" / ("benchmark_signature_methods.py" if kind == "forward"
                                    else "benchmark_signature_backprop.py")
    worker = CellWorker(script)
    estimate = estimate_function(script)
    report = json.loads(target.read_text()) if target.exists() else dict(metadata=None, cases=[], skipped=[], complete=False)
    if report["complete"]:
        print(f"{segments} {kind} {device}: already complete", flush=True)
        return
    recorded = {(c["dimension"], c["degree"]) for c in report["cases"] + report["skipped"]}
    for case in report["cases"]:
        case.setdefault("reference_jobs", report["metadata"]["settings"]["reference_jobs"])
    assert len(recorded) == len(report["cases"]) + len(report["skipped"])
    cell_dir = folder / "cells" / kind / device
    cell_dir.mkdir(parents=True, exist_ok=True)
    previous_elapsed = sum(c.get("cell_elapsed_seconds", c.get("elapsed_seconds", 0)) for c in report["cases"] + report["skipped"])
    started = time.perf_counter()
    for dimension in DIMENSIONS:
        for degree in DEGREES:
            if (dimension, degree) in recorded:
                continue
            case = dict(batch=32, segments=segments, dimension=dimension, degree=degree,
                        dtype="float32", n_jobs=jobs)
            if kind == "forward":
                output, workspace, work = estimate(32, segments, dimension, degree, "float32")
                live = 8 * output + 4 * 32 * (segments + 1) * dimension * 4 + workspace * min(32, jobs)
            else:
                values = estimate(32, segments, dimension, degree, "float32", device, jobs)
                live, work = values["estimated_live_bytes"], values["estimated_gemm_flops"]
            if live > 4096 * 2**20 or work > 1e12:
                reason = "estimated live memory exceeds 4096 MiB" if live > 4096 * 2**20 else "estimated work exceeds 1e12"
                report["skipped"].append(case | dict(reason=reason, estimated_live_bytes=live,
                                                     estimated_gemm_flops=work))
                recorded.add((dimension, degree))
                save(target, report)
                continue
            stem = f"d{dimension}_m{degree}"
            case_file, child_file = cell_dir / f"{stem}_case.json", cell_dir / f"{stem}.json"
            case_file.write_text(json.dumps([[32, segments, dimension, degree]]) + "\n",
                                 encoding="ascii", newline="\r\n")
            command = [sys.executable, str(script), "--device", device, "--dtype", "float32",
                       "--n-jobs", str(jobs), "--reference-jobs", "20", "--case-file", str(case_file), "--sample-seconds", "0.02",
                       "--max-memory-mib", "4096", "--cuda-memory-fraction", "0.65",
                       "--max-work", "1e12", "--output", str(child_file)]
            command += ["--repeats", "9"] if kind == "forward" else [
                "--batch", "32", "--segments", str(segments), "--samples", "9"]
            cell_started = time.perf_counter()
            timed_out = False
            child = None
            if child_file.exists():
                try:
                    child = json.loads(child_file.read_text())
                except json.JSONDecodeError:
                    pass
            if not child or not child.get("complete"):
                for unfinished in (child_file, child_file.with_suffix(".partial.json")):
                    if unfinished.exists():
                        unfinished.replace(unfinished.with_name(f"{stem}.interrupted.{time.time_ns()}.json"))
                try:
                    process = worker.run(command[1:], cell_dir / f"{stem}.log", LIMIT)
                except TimeoutError:
                    timed_out = True
                    process = None
            else:
                process = None
            elapsed = time.perf_counter() - cell_started
            child = json.loads(child_file.read_text()) if child_file.exists() else None
            checkpoint = child_file.with_suffix('.partial.json')
            if (not child or not child.get('complete')) and checkpoint.exists():
                candidate = json.loads(checkpoint.read_text())
                if candidate.get('complete'):
                    child = candidate
                    save(child_file, child)
                    process = None
                    print(f'{stem}: recovered completed checkpoint', flush=True)
            if report["metadata"] is None and child:
                report["metadata"] = child["metadata"]
                report["metadata"]["settings"]["case_file"] = str(folder / "cases.json")
                report["metadata"]["settings"]["output"] = str(target)
            if child and report["metadata"]["native_sha256"][device] != child["metadata"]["native_sha256"][device]:
                raise RuntimeError("Native libraries changed during the benchmark")
            if timed_out:
                report["skipped"].append(case | dict(reason="120-second cell runtime limit",
                                                     timeout_seconds=LIMIT, elapsed_seconds=elapsed))
                print(f"{segments} {kind} {device} {stem}: timed out after {elapsed:.1f}s", flush=True)
            elif process is not None and process != 0:
                raise RuntimeError(f"Benchmark failed: {segments} {kind} {device} {stem}; inspect the cell log")
            elif not child or not child["complete"]:
                raise RuntimeError(f"Incomplete benchmark: {segments} {kind} {device} {stem}")
            else:
                assert len(child["cases"]) + len(child["skipped"]) == 1
                for item in child["cases"] + child["skipped"]:
                    item["cell_elapsed_seconds"] = elapsed
                    item["reference_jobs"] = child["metadata"]["settings"]["reference_jobs"]
                report["cases"].extend(child["cases"])
                report["skipped"].extend(child["skipped"])
                print(f"{segments} {kind} {device} {stem}: complete in {elapsed:.1f}s", flush=True)
            recorded.add((dimension, degree))
            report["metadata"]["cell_timeout_seconds"] = LIMIT
            report["metadata"]["execution"] = "Each grid uses a reusable worker, restarted after a 120-second cell timeout."
            report["metadata"]["reference_thread_counts"] = sorted({c["reference_jobs"] for c in report["cases"]})
            save(target, report)
    worker.close()
    assert len(recorded) == 150
    report["complete"] = True
    report["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
    save(target, report)
    with (ROOT / "timing.csv").open("a", encoding="ascii", newline="") as output:
        output.write(f"{segments},{kind},{device},{report['elapsed_seconds']}\r\n")
    print(f"{segments} {kind} {device}: {len(report['cases'])} measured, "
          f"{len(report['skipped'])} excluded", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, choices=(16, 256, 4096), nargs="+", default=[16, 256, 4096])
    args = parser.parse_args()
    os.environ['PYTHONPATH'] = os.pathsep.join((str(REPO), str(REPO / 'plugins/cuda')))
    for name in ('PYSIGLIB_PS_DISPATCH', 'PYSIGLIB_PS_SPLIT_A', 'PYSIGLIB_PS_ENDPOINT_BLOCK'):
        os.environ.pop(name, None)
    for segments in args.segments:
        folder = ROOT / f"segments{segments}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cases.json").write_text(json.dumps([[32, segments, d, m]
                                                        for d in DIMENSIONS for m in DEGREES], indent=2) + "\n",
                                             encoding="ascii", newline="\r\n")
        for kind, device in (("forward", "cuda"), ("backward", "cuda"),
                             ("forward", "cpu"), ("backward", "cpu")):
            run_grid(segments, kind, device)


if __name__ == "__main__":
    main()
