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

"""Isolated prefix-suffix reversal experiments, with native workspace accounting."""

import argparse
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plugins/cuda"))
import pysiglib

PAIRS = [(2, 6), (2, 12), (3, 6), (3, 10), (4, 6), (8, 4), (16, 3)]
BLOCKS = [0, 1, 8, 32, 128, 512]


class ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in ["peak_working_set", "working_set", "peak_paged", "paged",
                                           "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"]]


class Native:
    def __init__(self, device, directory):
        self.device = device
        self.path = directory / f"reversal_{device}.dll"
        self.library = ctypes.CDLL(str(self.path.resolve()), winmode=0)
        self.library.experiment_error.restype = ctypes.c_char_p
        self.functions = {}
        for dtype, suffix in [(torch.float32, "f"), (torch.float64, "d")]:
            f = getattr(self.library, f"experiment_{suffix}")
            f.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint64] * 6 + [ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
            if device == "cuda":
                f.argtypes += [ctypes.c_void_p]
            f.restype = ctypes.c_int
            self.functions[dtype] = f

    def call(self, path, final, seeds, out, degree, split, block, scalar, threads):
        batch, length, d = path.shape
        peak = ctypes.c_uint64()
        args = [x.data_ptr() for x in (path, final, seeds, out)]
        args += [batch, d, length - 1, degree, split, block, scalar, threads, ctypes.byref(peak)]
        if self.device == "cuda":
            args += [torch.cuda.current_stream().cuda_stream]
        status = self.functions[path.dtype](*args)
        if status:
            raise RuntimeError(self.library.experiment_error().decode())
        return peak.value

    def clear(self):
        if self.device == "cuda":
            self.library.experiment_clear()


def chen(path, degree, scalar):
    levels = [path.new_ones(1)] + [path.new_zeros(path.shape[-1] ** k) for k in range(1, degree + 1)]
    for increment in path[1:] - path[:-1]:
        exponential = [path.new_ones(1)]
        for k in range(1, degree + 1):
            exponential.append(torch.outer(exponential[-1], increment).flatten() / k)
        levels = [sum(torch.outer(levels[j], exponential[k - j]).flatten() for j in range(k + 1)) for k in range(degree + 1)]
    return torch.cat(levels[not scalar:])


def check(native, threads):
    records = []
    cases = [(d, m, 9, split, scalar, block) for d, m in [(2, 3), (2, 6), (3, 4)]
             for split in range(m + 1) for scalar in [False, True] for block in BLOCKS]
    cases += [(2, 4, n, 0, True, b) for n, b in [(35, 32), (129, 128), (513, 512), (4096, 8)]]
    for dtype in [torch.float32, torch.float64]:
        for d, m, n, split, scalar, block in cases:
            rng = np.random.default_rng(9101 + n + d + m)
            x = torch.tensor(rng.normal(scale=.03, size=(n + 1, d)).cumsum(0), dtype=dtype)
            x[1] = x[0]
            exact_path = x.double().requires_grad_()
            if n < 20:
                exact = chen(exact_path, m, scalar)
                seed = torch.tensor(rng.normal(size=exact.shape), dtype=dtype)
                reference, = torch.autograd.grad((exact * seed.double()).sum(), exact_path)
                final = exact.detach().to(dtype)
            else:
                final64 = pysiglib.sig(x.double(), m, method="horner", scalar_term=scalar)
                seed = torch.tensor(rng.normal(size=final64.shape), dtype=dtype)
                reference = pysiglib.sig_backprop(x.double(), final64, seed.double(), m, method="horner")
                final = final64.to(dtype)
            path, final, seed = [v.unsqueeze(0).to(native.device).contiguous() for v in (x, final, seed)]
            out = torch.empty_like(path)
            peak = native.call(path, final, seed, out, m, split, block, scalar, threads)
            tol = 3e-5 if dtype == torch.float32 else 1e-10
            torch.testing.assert_close(out[0].cpu().double(), reference, atol=tol, rtol=tol,
                                       msg=f"d={d} m={m} n={n} split={split} scalar={scalar} block={block} {dtype}")
            records.append(dict(device=native.device, dtype=str(dtype), dimension=d, degree=m, segments=n,
                                split=split, scalar=scalar, block=block, peak_bytes=peak,
                                max_absolute=float((out[0].cpu().double() - reference).abs().max())))
        for block in BLOCKS[1:]:
            peaks = []
            lengths = [max(64, block), max(256, 2 * block), 4096]
            for n in lengths:
                path = torch.zeros((1, n + 1, 2), dtype=dtype, device=native.device)
                final = torch.zeros((1, 31), dtype=dtype, device=native.device)
                final[:, 0] = 1
                seeds = torch.ones_like(final)
                out = torch.empty_like(path)
                peaks.append(native.call(path, final, seeds, out, 4, 2, block, True, 1))
            assert len(set(peaks)) == 1, (dtype, block, peaks)
            records.append(dict(device=native.device, dtype=str(dtype), block=block,
                                bounded_workspace_segments=lengths, peak_bytes=peaks))
    native.clear()
    return records


def measure(native, d, m, n, batch, dtype, args):
    blocks = getattr(args, "blocks", BLOCKS)
    native.clear()
    rng = np.random.default_rng(args.seed + d * 1000 + m * 100 + n + batch)
    host = torch.tensor(rng.normal(scale=.03, size=(batch, n + 1, d)).cumsum(1), dtype=dtype)
    path = host.to(native.device)
    final = pysiglib.sig(path, m, method="ps", scalar_term=True, n_jobs=args.threads)
    seeds = torch.tensor(rng.normal(scale=.05, size=final.shape), dtype=dtype, device=native.device)
    final64 = pysiglib.sig(host.double(), m, method="ps", scalar_term=True, n_jobs=args.threads)
    reference = torch.empty_like(host, dtype=torch.float64)
    baseline = native if native.device == "cpu" else Native("cpu", native.path.parent)
    baseline.call(host.double(), final64, seeds.cpu().double(), reference, m, 0, 0, True, args.threads)
    out = torch.empty_like(path)
    if native.device == "cuda":
        from pysiglib.load_siglib import CUSIG
        if CUSIG.clear_cache_cuda(False):
            raise RuntimeError("Could not release the separate reference library workspace")
    records, repeats = {}, {}
    sync = torch.cuda.synchronize if native.device == "cuda" else lambda: None
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    memory_query = ctypes.windll.psapi.GetProcessMemoryInfo
    memory_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_uint32]
    for block in blocks:
        record = dict(block=block, effective_block=min(block, n), status="ok", samples_ms=[], peak_native_temporary_bytes=0)
        records[block] = record
        try:
            sync()
            start = time.perf_counter()
            peak = native.call(path, final, seeds, out, m, 0, block, True, args.threads)
            sync()
            elapsed = time.perf_counter() - start
            error = out.cpu().double() - reference
            tolerance = 3e-5 if dtype == torch.float32 else 1e-10
            record.update(first_call_ms=elapsed * 1000, max_absolute=float(error.abs().max()),
                          relative_l2=float(error.norm() / reference.norm().clamp_min(1e-300)),
                          gradient_pass=bool(torch.all(error.abs() <= tolerance * (1 + reference.abs()))),
                          peak_native_temporary_bytes=peak)
            if elapsed > args.max_call_seconds:
                record.update(status="excluded", reason=f"Initial call exceeds {args.max_call_seconds:g} seconds")
                continue
            for _ in range(args.warmup):
                native.call(path, final, seeds, out, m, 0, block, True, args.threads)
            repeats[block] = max(1, min(1000, int(args.sample_ms / max(elapsed * 1000, .001))))
        except Exception as exc:
            record.update(status="failed", reason=str(exc))
    for sample in range(getattr(args, "samples", 9)):
        for block in (blocks if sample % 2 == 0 else blocks[::-1]):
            record = records[block]
            if record["status"] != "ok":
                continue
            sync()
            start = time.perf_counter_ns()
            for _ in range(repeats[block]):
                peak = native.call(path, final, seeds, out, m, 0, block, True, args.threads)
                record["peak_native_temporary_bytes"] = max(record["peak_native_temporary_bytes"], peak)
            sync()
            record["samples_ms"].append((time.perf_counter_ns() - start) / (1e6 * repeats[block]))
            memory = ProcessMemory()
            memory.cb = ctypes.sizeof(memory)
            if not memory_query(process(), ctypes.byref(memory), memory.cb):
                raise ctypes.WinError()
            record["process_peak_working_set_bytes"] = memory.peak_working_set
            record["process_peak_commit_bytes"] = memory.peak_pagefile
    for block, record in records.items():
        if record["samples_ms"]:
            record.update(median_ms=float(np.median(record["samples_ms"])), repetitions=repeats[block])
    native.clear()
    return dict(device=native.device, dimension=d, degree=m, segments=n, batch=batch,
                dtype=str(dtype).split(".")[-1], input_bytes=sum(t.numel() * t.element_size() for t in (path, final, seeds)),
                output_bytes=out.numel() * out.element_size(), methods=list(records.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--library-dir", type=Path, default=ROOT / "out/build/reversal/Release")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--retry-excluded", action="store_true")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sample-ms", type=float, default=20)
    parser.add_argument("--max-call-seconds", type=float, default=10)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--segments", type=int, nargs="+", default=[16, 256, 4096])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    args = parser.parse_args()
    native = Native(args.device, args.library_dir)
    report = dict(metadata=dict(device=args.device, platform=platform.platform(), python=sys.version, torch=torch.__version__,
                                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                environment={k: v for k, v in os.environ.items() if k.startswith(("PYSIGLIB_PS_", "OMP_", "OPENBLAS_"))},
                                gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                                library_sha256=hashlib.sha256(native.path.read_bytes()).hexdigest(),
                                settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                workspace="Native numeric temporary allocations; excludes inputs, output, library code, stacks, and opaque CPU BLAS/TBB allocations. CUDA includes its explicit 4 MiB BLAS workspace. Process peak is reported separately.",
                                reference="CPU float64 production prefix-suffix on the same rounded inputs and cotangents; separate small-case checks use independent Chen autograd.",
                                gpu_process_memory="Unavailable under this Windows WDDM driver: nvidia-smi reports N/A. Native owned workspace is measured separately.",
                                timing="Nine alternating synchronized samples. Reconstruction and native allocation/free are timed. CUDA baseline retains its production cache; prototypes allocate/free their numeric workspace on each call."),
                  cases=[], complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        report["checks"] = check(native, args.threads)
        print(f"{args.device}: {len(report['checks'])} correctness checks passed", flush=True)
    elif args.retry_excluded:
        report = json.loads(args.output.read_text(encoding="ascii"))
        assert report["metadata"]["device"] == args.device
        assert report["metadata"]["library_sha256"] == hashlib.sha256(native.path.read_bytes()).hexdigest()
        for key in ["threads", "warmup", "sample_ms", "seed"]:
            assert report["metadata"]["settings"][key] == getattr(args, key), key
        report["complete"] = False
        for index, case in enumerate(report["cases"]):
            if all(method["status"] == "ok" for method in case["methods"]):
                continue
            replacement = measure(native, case["dimension"], case["degree"], case["segments"],
                                  case["batch"], getattr(torch, case["dtype"]), args)
            report.setdefault("retries", []).append(dict(previous_case=case, max_call_seconds=args.max_call_seconds,
                                                        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            report["cases"][index] = replacement
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")
            print(f"Retried {args.device} configuration {index + 1}", flush=True)
    else:
        for (d, m), n, batch, dtype in itertools.product(PAIRS, args.segments, args.batches, [torch.float32, torch.float64]):
            report["cases"].append(measure(native, d, m, n, batch, dtype, args))
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")
            print(f"{args.device} d={d} m={m} n={n} batch={batch} {dtype}", flush=True)
    report["complete"] = True
    report["metadata"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")


if __name__ == "__main__":
    main()
