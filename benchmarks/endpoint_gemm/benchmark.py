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

"""Native end-to-end comparison of endpoint GEMM, Horner, and prefix-suffix."""

import argparse
import ctypes
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CASES = [(1, 4096, 4, 6), (1, 8192, 8, 4), (1, 4096, 16, 4),
         (1, 4096, 32, 3), (1, 2048, 64, 3), (1, 8192, 256, 2),
         (1, 4096, 512, 2), (32, 256, 2, 10), (32, 256, 4, 6),
         (32, 256, 8, 4), (32, 256, 32, 3), (32, 256, 64, 3)]


class Native:
    def __init__(self):
        cpu = ROOT / "out/build/windows-cpu/siglib/cpsig/Release/cpsig.dll"
        gpu = ROOT / "out/build/windows-cuda/cusig_build/Release/cusig.dll"
        experiment = ROOT / "out/build/endpoint-gemm/Release"
        directories = [cpu.parent, gpu.parent, experiment,
                       ROOT / "out/build/windows-cpu/msvc_19.44_cxx20_64_mt_release",
                       Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin")]
        self.directories = [os.add_dll_directory(str(p)) for p in directories if p.exists()]
        self.paths = {"cpu": cpu, "cuda": gpu, "endpoint_cpu": experiment / "endpoint_cpu.dll",
                      "endpoint_cuda": experiment / "endpoint_cuda.dll"}
        self.libs = {name: ctypes.CDLL(str(path)) for name, path in self.paths.items()}
        for name in ("endpoint_cpu", "endpoint_cuda"):
            self.libs[name].endpoint_error.restype = ctypes.c_char_p

    def operation(self, path, degree, method, jobs=1, block=256):
        gpu = isinstance(path, torch.Tensor)
        dtype = str(path.dtype).split(".")[-1]
        scalar = ctypes.c_float if dtype == "float32" else ctypes.c_double
        pointer = ctypes.POINTER(scalar)
        batch, points, d = path.shape
        size = sum(d ** k for k in range(degree + 1))
        output = torch.empty((batch, size), dtype=path.dtype, device=path.device) if gpu else np.empty((batch, size), dtype=path.dtype)
        ptr = lambda x: ctypes.cast(x.data_ptr() if gpu else x.ctypes.data, pointer)
        suffix = "f" if dtype == "float32" else "d"
        if method == "endpoint":
            lib = self.libs["endpoint_cuda" if gpu else "endpoint_cpu"]
            fn = getattr(lib, "endpoint_" + suffix)
            fn.argtypes = [pointer, pointer] + [ctypes.c_uint64] * 5 + [ctypes.c_void_p if gpu else ctypes.c_int]
            args = (ptr(path), ptr(output), batch, d, points, degree, block,
                    torch.cuda.current_stream().cuda_stream if gpu else jobs)
        else:
            lib = self.libs["cuda" if gpu else "cpu"]
            symbol = "signature" + ("_prefix_suffix" if method == "prefix_suffix" else "") + ("_cuda" if gpu else "")
            fn = getattr(lib, symbol + "_" + suffix)
            types = [pointer, pointer] + [ctypes.c_uint64] * 4 + [ctypes.c_bool, ctypes.c_bool, scalar]
            args = (ptr(path), ptr(output), batch, d, points, degree, False, False, scalar(1))
            if method == "prefix_suffix":
                types += [ctypes.c_bool, ctypes.c_void_p if gpu else ctypes.c_int]
                args += (True, torch.cuda.current_stream().cuda_stream if gpu else jobs)
            else:
                types += [ctypes.c_bool, ctypes.c_bool] + ([] if gpu else [ctypes.c_int]) + [pointer] + [ctypes.c_uint64] * 3
                args += (True, True) + (() if gpu else (jobs,)) + (None, 0, 0, 0)
            fn.argtypes = types
        fn.restype = ctypes.c_int

        def run():
            code = fn(*args)
            if code:
                message = lib.endpoint_error().decode() if method == "endpoint" else f"status {code}"
                raise RuntimeError(f"{method}: {message}")
            return output
        run.path = path
        return run


def measure(run, count, cuda):
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(count):
        run()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / count / 1e6


def errors(result, reference, d, degree):
    if isinstance(result, torch.Tensor):
        result = result.cpu().numpy()
    if not np.isfinite(result).all():
        raise AssertionError("non-finite result")
    delta = result.astype(np.float64) - reference
    ratios, offset = [], 1
    for k in range(1, degree + 1):
        denominator = np.linalg.norm(reference[:, offset:offset + d ** k], axis=1)
        numerator = np.linalg.norm(delta[:, offset:offset + d ** k], axis=1)
        ratios.append(float(np.max(numerator / np.where(denominator > 0, denominator, 1))))
        offset += d ** k
    return {"max_abs": float(np.max(np.abs(delta))), "levelwise_relative_or_zero_absolute": max(ratios)}


def validate(native):
    cases = [(1, 0, 2, 3), (1, 1, 2, 3), (2, 7, 2, 5), (2, 9, 3, 4), (1, 5, 1, 6)]
    tested = 0
    rng = np.random.default_rng(178)
    for batch, n, d, degree in cases:
        increments = rng.integers(-2, 3, (batch, n, d)) / 8
        if n > 1:
            increments[:, 1] = 0
        vertices = np.concatenate([np.zeros((batch, 1, d)), np.cumsum(increments, axis=1)], axis=1)
        expected = []
        for dx in increments:
            state = [np.array([Fraction(1)], dtype=object)] + [np.full(d ** k, Fraction(0), dtype=object) for k in range(1, degree + 1)]
            for v in dx:
                segment = [np.array([Fraction(1)], dtype=object)]
                for k in range(1, degree + 1):
                    segment.append(np.kron(segment[-1], [Fraction(float(x)) for x in v]) / k)
                state = [sum((np.kron(state[j], segment[k - j]) for j in range(k + 1)),
                             np.full(d ** k, Fraction(0), dtype=object)) for k in range(degree + 1)]
            expected.append(np.concatenate(state).astype(np.float64))
        expected = np.array(expected)
        for dtype in (np.float32, np.float64):
            for device in ("cpu", "cuda"):
                path = vertices.astype(dtype)
                if device == "cuda":
                    path = torch.as_tensor(path, device="cuda")
                for block in (1, 3, 8, 256):
                    run = native.operation(path, degree, "endpoint", jobs=20, block=block)
                    stream = torch.cuda.Stream() if device == "cuda" else None
                    if stream:
                        with torch.cuda.stream(stream):
                            stream.wait_stream(torch.cuda.default_stream())
                            run = native.operation(path, degree, "endpoint", block=block)
                            actual = run().cpu().numpy()
                    else:
                        actual = run()
                    np.testing.assert_allclose(actual, expected, rtol=2e-5 if dtype == np.float32 else 2e-12,
                                               atol=2e-6 if dtype == np.float32 else 2e-14)
                    tested += 1
    # Empty batches, zero-point paths, and exact retracing paths.
    for device in ("cpu", "cuda"):
        for shape in ((0, 5, 2), (1, 0, 2), (1, 1, 2)):
            path = np.zeros(shape, dtype=np.float64)
            if device == "cuda": path = torch.as_tensor(path, device="cuda")
            result = native.operation(path, 4, "endpoint")()
            actual = result.cpu().numpy() if device == "cuda" else result
            if shape[0]: np.testing.assert_array_equal(actual, [[1] + [0] * 30])
            tested += 1
        vertices = np.array([[[0, 0], [1, 0], [1, 1], [1, 0], [0, 0]]], dtype=np.float64)
        path = torch.as_tensor(vertices, device="cuda") if device == "cuda" else vertices
        result = native.operation(path, 5, "endpoint", block=2)()
        actual = result.cpu().numpy() if device == "cuda" else result
        np.testing.assert_allclose(actual, [[1] + [0] * 62], atol=2e-14)
        tested += 1
    for d, degree in ((8, 6), (4, 7)):
        vertices = rng.normal(scale=0.05, size=(2, 8, d))
        for dtype in (np.float32, np.float64):
            source = vertices.astype(dtype)
            reference = native.operation(source.astype(np.float64), degree, "horner")().copy()
            for device in ("cpu", "cuda"):
                path = torch.as_tensor(source, device="cuda") if device == "cuda" else source
                result = native.operation(path, degree, "endpoint", block=3)()
                error = errors(result, reference, d, degree)
                assert error["levelwise_relative_or_zero_absolute"] < (2e-5 if dtype == np.float32 else 2e-12)
                tested += 1
    for device in ("cpu", "cuda"):
        source = np.zeros((1, 2, 2), dtype=np.float64)
        path = torch.as_tensor(source, device="cuda") if device == "cuda" else source
        try:
            native.operation(path, 3, "endpoint", block=1 << 30)()
        except RuntimeError:
            tested += 1
        else:
            raise AssertionError("workspace limit was not enforced")
    print(f"Validation passed: {tested} cases", flush=True)
    return tested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--case", nargs=4, type=int, action="append")
    parser.add_argument("--blocks", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sample-ms", type=float, default=10)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/endpoint_gemm/results.json")
    args = parser.parse_args()
    native = Native()
    if args.validate_only:
        validate(native)
        return
    torch.set_num_threads(args.jobs)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "date": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "libraries": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in native.paths.items()},
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "numpy": np.__version__,
              "timing": "native calls with preallocated input/output; includes internal preparation, GEMM, merges and synchronization",
              "precision": "one dtype throughout; pedantic cuBLAS; no TF32", "cases": []}
    dtype = np.dtype(args.dtype)
    cuda = args.device == "cuda"
    if cuda:
        warm = torch.randn((1024, 1024), device="cuda")
        until = time.perf_counter() + 2
        while time.perf_counter() < until:
            torch.mm(warm, warm)
            torch.cuda.synchronize()
        del warm
    for case in args.case or CASES:
        batch, n, d, degree = case
        entry = {"shape": case, "tuning": [], "samples": [], "errors": [], "first_call_ms": {}}
        candidates = sorted(set(min(n, b) for b in args.blocks) | {n})
        rng = np.random.default_rng(987)
        dx = rng.normal(size=(batch, n, d)) / np.sqrt(n)
        vertices = np.concatenate([np.zeros((batch, 1, d)), np.cumsum(dx, axis=1)], axis=1).astype(dtype)
        path = torch.as_tensor(vertices, device="cuda") if cuda else vertices
        reference = native.operation(vertices.astype(np.float64), degree, "horner", jobs=args.jobs)().copy()
        for block in candidates:
            try:
                run = native.operation(path, degree, "endpoint", args.jobs, block)
                first = measure(run, 1, cuda)
                error = errors(run(), reference, d, degree)
                if error["levelwise_relative_or_zero_absolute"] > (2e-4 if args.dtype == "float32" else 2e-10):
                    raise AssertionError(str(error))
                samples = [measure(run, 1, cuda) for _ in range(3)]
                entry["tuning"].append({"block": block, "ms": samples, "first_ms": first, "error": error})
            except RuntimeError as exc:
                entry["tuning"].append({"block": block, "excluded": str(exc)})
        valid = [r for r in entry["tuning"] if "ms" in r]
        if not valid:
            raise RuntimeError(f"no feasible endpoint block for {case}")
        selected = min(valid, key=lambda r: np.median(r["ms"]))["block"]
        entry["block"] = selected
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            dx = rng.normal(size=(batch, n, d)) / np.sqrt(n)
            vertices = np.concatenate([np.zeros((batch, 1, d)), np.cumsum(dx, axis=1)], axis=1).astype(dtype)
            path = torch.as_tensor(vertices, device="cuda") if cuda else vertices
            reference = native.operation(vertices.astype(np.float64), degree, "horner", jobs=args.jobs)().copy()
            operations = {method: native.operation(path, degree, method, args.jobs, selected)
                          for method in ("horner", "prefix_suffix", "endpoint")}
            counts = {}
            for method, run in operations.items():
                first = measure(run, 1, cuda)
                entry["first_call_ms"].setdefault(method, first)
                error = errors(run(), reference, d, degree)
                if error["levelwise_relative_or_zero_absolute"] > (2e-4 if args.dtype == "float32" else 2e-10):
                    raise AssertionError((case, method, error))
                entry["errors"].append({"seed": seed, "method": method, **error})
                warm = measure(run, 3, cuda)
                counts[method] = max(1, min(1000, int(args.sample_ms / max(warm, 0.001))))
            for repeat in range(args.repeats):
                for method in rng.permutation(list(operations)):
                    ms = measure(operations[method], counts[method], cuda)
                    entry["samples"].append({"seed": seed, "repeat": repeat, "method": method, "ms": ms,
                                             "iterations": counts[method]})
        report["cases"].append(entry)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")
        medians = {method: np.median([v["ms"] for v in entry["samples"] if v["method"] == method])
                   for method in ("horner", "prefix_suffix", "endpoint")}
        print(f"{case} block={selected}: {medians}", flush=True)


if __name__ == "__main__":
    main()
