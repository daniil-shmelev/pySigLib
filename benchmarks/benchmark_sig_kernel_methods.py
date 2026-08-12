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

import argparse
import csv
import ctypes
from ctypes import POINTER, c_bool, c_double, c_int, c_size_t, c_uint64, c_void_p
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np


SEGMENTS = (16, 32, 64, 128, 256, 512, 1024)
METHODS = ("finite_difference", "polynomial")
COLORS = {
    "finite_difference": "#4c78a8",
    "polynomial": "#54a24b",
}
LABELS = {
    "finite_difference": "Finite difference",
    "polynomial": "Polynomial",
}


if platform.system() == "Windows":
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = (
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", c_size_t),
            ("WorkingSetSize", c_size_t),
            ("QuotaPeakPagedPoolUsage", c_size_t),
            ("QuotaPagedPoolUsage", c_size_t),
            ("QuotaPeakNonPagedPoolUsage", c_size_t),
            ("QuotaNonPagedPoolUsage", c_size_t),
            ("PagefileUsage", c_size_t),
            ("PeakPagefileUsage", c_size_t),
            ("PrivateUsage", c_size_t),
        )

    MEMORY_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    MEMORY_PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
    MEMORY_KERNEL32.GetCurrentProcess.argtypes = ()
    MEMORY_KERNEL32.GetCurrentProcess.restype = c_void_p
    MEMORY_PSAPI.GetProcessMemoryInfo.argtypes = (
        c_void_p, ctypes.POINTER(ProcessMemoryCountersEx), ctypes.c_ulong)
    MEMORY_PSAPI.GetProcessMemoryInfo.restype = c_bool
    MEMORY_PROCESS = MEMORY_KERNEL32.GetCurrentProcess()


def load_cpsig(repo_root):
    library_dir = repo_root / "pysiglib"
    if platform.system() == "Windows":
        tbb_path = library_dir / "tbb12.dll"
        if tbb_path.exists():
            ctypes.CDLL(str(tbb_path), winmode=0)
        library_path = library_dir / "cpsig.dll"
        library = ctypes.CDLL(str(library_path), winmode=0)
    elif platform.system() == "Darwin":
        library_path = library_dir / "libcpsig.dylib"
        library = ctypes.CDLL(str(library_path))
    else:
        library_path = library_dir / "libcpsig.so"
        library = ctypes.CDLL(str(library_path))

    library.sig_kernel_d.argtypes = (
        POINTER(c_double), POINTER(c_double), c_uint64, c_uint64,
        c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
    library.sig_kernel_d.restype = c_int
    polynomial_args = (
        POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64,
        c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
    library.sig_kernel_poly_d.argtypes = polynomial_args
    library.sig_kernel_poly_d.restype = c_int
    library.signature_d.argtypes = (
        POINTER(c_double), POINTER(c_double), c_uint64, c_uint64,
        c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool,
        c_bool, c_int, POINTER(c_double), c_uint64, c_uint64, c_uint64)
    library.signature_d.restype = c_int
    return library


def pin_logical_cpu(requested_cpu):
    if platform.system() == "Windows":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = c_void_p
        kernel32.GetProcessAffinityMask.argtypes = (
            c_void_p, POINTER(c_size_t), POINTER(c_size_t))
        kernel32.GetProcessAffinityMask.restype = c_bool
        kernel32.SetProcessAffinityMask.argtypes = (c_void_p, c_size_t)
        kernel32.SetProcessAffinityMask.restype = c_bool
        process = kernel32.GetCurrentProcess()
        process_mask = c_size_t()
        system_mask = c_size_t()
        if not kernel32.GetProcessAffinityMask(
                process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
        allowed = [cpu for cpu in range(c_size_t(-1).value.bit_length())
                   if process_mask.value & (1 << cpu)]
        cpu = requested_cpu if requested_cpu in allowed else allowed[0]
        if not kernel32.SetProcessAffinityMask(process, c_size_t(1 << cpu)):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        return cpu

    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        cpu = requested_cpu if requested_cpu in allowed else allowed[0]
        os.sched_setaffinity(0, {cpu})
        return cpu

    raise RuntimeError("logical CPU pinning is not supported on this platform")


def cpu_model():
    if platform.system() == "Windows":
        import winreg
        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except ImportError:
        return platform.processor() or platform.machine()


def compiler_name(repo_root):
    config_path = repo_root / "pysiglib" / "_config.py"
    spec = importlib.util.spec_from_file_location("pysiglib_build_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "CXX_COMPILER", "unknown")


def git_revision(repo_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        check=True, capture_output=True, text=True)
    return result.stdout.strip()


def brownian_paths(seed):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(SEGMENTS[-1])
    increments1 = rng.normal(scale=scale, size=(SEGMENTS[-1], 2))
    increments2 = rng.normal(scale=scale, size=(SEGMENTS[-1], 2))
    path1 = np.vstack([np.zeros(2), np.cumsum(increments1, axis=0)])
    path2 = np.vstack([np.zeros(2), np.cumsum(increments2, axis=0)])
    paths = {}
    for segments in SEGMENTS:
        stride = SEGMENTS[-1] // segments
        paths[segments] = (
            np.ascontiguousarray(path1[::stride], dtype=np.float64),
            np.ascontiguousarray(path2[::stride], dtype=np.float64),
        )
    return paths


def direct_signature_kernel(library, path1, path2, degree=21):
    signature_length = (1 << (degree + 1)) - 1
    sig1 = np.empty(signature_length, dtype=np.float64)
    sig2 = np.empty(signature_length, dtype=np.float64)
    args = (1, 2, path1.shape[0], degree, False, False, 1.0, True, True, 1,
            None, 0, 0, 0)
    status = library.signature_d(
        path1.ctypes.data_as(POINTER(c_double)),
        sig1.ctypes.data_as(POINTER(c_double)), *args)
    if status:
        raise RuntimeError("signature_d failed with status " + str(status))
    status = library.signature_d(
        path2.ctypes.data_as(POINTER(c_double)),
        sig2.ctypes.data_as(POINTER(c_double)), *args)
    if status:
        raise RuntimeError("signature_d failed with status " + str(status))
    return float(np.dot(sig1, sig2))


def make_solver_call(library, method, gram, segments, parameter):
    gram_ptr = gram.ctypes.data_as(POINTER(c_double))
    out = np.empty(1, dtype=np.float64)
    out_ptr = out.ctypes.data_as(POINTER(c_double))
    if method == "finite_difference":
        def call():
            status = library.sig_kernel_d(
                gram_ptr, out_ptr, 1, 2, segments + 1, segments + 1,
                parameter, parameter, False, 1)
            if status:
                raise RuntimeError("sig_kernel_d failed with status " + str(status))
            return out[0]
    else:
        native = library.sig_kernel_poly_d

        def call():
            status = native(
                gram_ptr, out_ptr, None, 1, 2, segments + 1, segments + 1,
                parameter, False, 1)
            if status:
                raise RuntimeError(method + " failed with status " + str(status))
            return out[0]
    return call


def benchmark_call(call, warmups, samples, minimum_seconds):
    repetitions = 1
    while True:
        start = time.perf_counter()
        for _ in range(repetitions):
            call()
        elapsed = time.perf_counter() - start
        if elapsed >= minimum_seconds:
            break
        repetitions *= max(2, int(np.ceil(minimum_seconds / max(elapsed, 1e-9))))

    for _ in range(warmups):
        for _ in range(repetitions):
            call()

    timings = np.empty(samples, dtype=np.float64)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for sample in range(samples):
            start = time.perf_counter()
            for _ in range(repetitions):
                call()
            timings[sample] = (time.perf_counter() - start) / repetitions
    finally:
        if gc_enabled:
            gc.enable()
    return timings, repetitions


def process_memory_bytes():
    if platform.system() == "Windows":
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not MEMORY_PSAPI.GetProcessMemoryInfo(
                MEMORY_PROCESS, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.PrivateUsage), "private_commit"

    if platform.system() == "Linux":
        with Path("/proc/self/statm").open(encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE"), "resident_set"

    raise RuntimeError("process memory measurement is supported on Windows and Linux")


def logical_memory_bytes(method, segments, parameter):
    scalar_bytes = np.dtype(np.float64).itemsize
    gram_bytes = segments * segments * scalar_bytes
    if method == "finite_difference":
        refined_length = (segments << parameter) + 1
        solver_bytes = (2 * segments * segments + 3 * refined_length) * scalar_bytes
    else:
        coefficient_count = parameter + 1
        solver_bytes = (
            (segments + 4) * coefficient_count
            + segments
            + 2 * coefficient_count * coefficient_count
        ) * scalar_bytes
    return gram_bytes, solver_bytes, gram_bytes + solver_bytes + scalar_bytes


def memory_worker(method, segments, parameter, duration, seed):
    repo_root = Path(__file__).resolve().parents[1]
    library = load_cpsig(repo_root)
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(segments)
    increments1 = np.ascontiguousarray(
        rng.normal(scale=scale, size=(segments, 2)), dtype=np.float64)
    increments2 = np.ascontiguousarray(
        rng.normal(scale=scale, size=(segments, 2)), dtype=np.float64)

    stop = threading.Event()
    ready = threading.Event()
    solver_phase = threading.Event()
    peak_total = [0]
    peak_solver = [0]

    def sample_memory():
        ready.set()
        while not stop.is_set():
            current, _ = process_memory_bytes()
            if current > peak_total[0]:
                peak_total[0] = current
            if solver_phase.is_set() and current > peak_solver[0]:
                peak_solver[0] = current
            stop.wait(0.0005)

    sampler = threading.Thread(target=sample_memory)
    sampler.start()
    ready.wait()
    baseline_total, metric = process_memory_bytes()
    peak_total[0] = baseline_total

    gram = np.ascontiguousarray(increments1 @ increments2.T)
    gc.collect()
    baseline_solver, _ = process_memory_bytes()
    peak_solver[0] = baseline_solver
    call = make_solver_call(library, method, gram, segments, parameter)
    solver_phase.set()
    deadline = time.perf_counter() + duration
    calls = 0
    while time.perf_counter() < deadline:
        call()
        calls += 1
    current, _ = process_memory_bytes()
    peak_total[0] = max(peak_total[0], current)
    peak_solver[0] = max(peak_solver[0], current)
    stop.set()
    sampler.join()

    gram_bytes, solver_bytes, total_bytes = logical_memory_bytes(
        method, segments, parameter)
    return {
        "memory_metric": metric,
        "peak_total_bytes": max(0, peak_total[0] - baseline_total),
        "peak_solver_bytes": max(0, peak_solver[0] - baseline_solver),
        "gram_bytes": gram_bytes,
        "logical_solver_bytes": solver_bytes,
        "logical_total_bytes": total_bytes,
        "calls": calls,
    }


def benchmark_memory(repo_root, args, metadata):
    rows = []
    for segments in SEGMENTS:
        for method in METHODS:
            parameter = 0 if method == "finite_difference" else 7
            measurements = []
            for _ in range(args.memory_samples):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--memory-worker", method,
                    "--memory-segments", str(segments),
                    "--memory-parameter", str(parameter),
                    "--memory-duration", str(args.memory_duration),
                    "--seed", str(args.seed),
                ]
                result = subprocess.run(
                    command, cwd=repo_root, check=True, capture_output=True,
                    text=True)
                measurements.append(json.loads(result.stdout.strip().splitlines()[-1]))

            peak_total = np.array(
                [measurement["peak_total_bytes"] for measurement in measurements],
                dtype=np.float64)
            peak_solver = np.array(
                [measurement["peak_solver_bytes"] for measurement in measurements],
                dtype=np.float64)
            first = measurements[0]
            rows.append({
                "method": method,
                "path_length": segments,
                "degree": parameter if method != "finite_difference" else "",
                "dyadic_order": parameter if method == "finite_difference" else "",
                "peak_total_median_bytes": float(np.median(peak_total)),
                "peak_total_q1_bytes": float(np.quantile(peak_total, 0.25)),
                "peak_total_q3_bytes": float(np.quantile(peak_total, 0.75)),
                "peak_solver_median_bytes": float(np.median(peak_solver)),
                "peak_solver_q1_bytes": float(np.quantile(peak_solver, 0.25)),
                "peak_solver_q3_bytes": float(np.quantile(peak_solver, 0.75)),
                "gram_bytes": first["gram_bytes"],
                "logical_solver_bytes": first["logical_solver_bytes"],
                "logical_total_bytes": first["logical_total_bytes"],
                "memory_metric": first["memory_metric"],
                "samples": args.memory_samples,
                "dtype": "float64",
                "seed": metadata["seed"],
                "logical_cpu": metadata["logical_cpu"],
                "compiler": metadata["compiler"],
                "cpu_model": metadata["cpu_model"],
                "git_revision": metadata["git_revision"],
            })
            print("memory", method, segments, int(np.median(peak_total)))
    return rows


def summarize_row(panel, method, segments, parameter, timings, repetitions,
                  reference, value, metadata, status="ok"):
    row = {
        "panel": panel,
        "method": method,
        "path_length": segments,
        "selected_degree": parameter if method != "finite_difference" else "",
        "dyadic_order": parameter if method == "finite_difference" else "",
        "median_seconds": "",
        "q1_seconds": "",
        "q3_seconds": "",
        "reference_value": reference,
        "value": "" if value is None else value,
        "absolute_error": "" if value is None else abs(value - reference),
        "inner_repetitions": repetitions,
        "samples": metadata["samples"],
        "status": status,
        "dtype": "float64",
        "seed": metadata["seed"],
        "logical_cpu": metadata["logical_cpu"],
        "compiler": metadata["compiler"],
        "cpu_model": metadata["cpu_model"],
        "git_revision": metadata["git_revision"],
    }
    if timings is not None:
        row["median_seconds"] = float(np.median(timings))
        row["q1_seconds"] = float(np.quantile(timings, 0.25))
        row["q3_seconds"] = float(np.quantile(timings, 0.75))
    return row


def meets_tolerance(value, reference):
    return np.isclose(value, reference, atol=1e-10, rtol=1e-8)


def fixed_degree_rows(library, grams, references, args, metadata):
    rows = []
    for segments in SEGMENTS:
        for method in METHODS:
            parameter = 0 if method == "finite_difference" else 7
            call = make_solver_call(library, method, grams[segments], segments, parameter)
            value = float(call())
            timings, repetitions = benchmark_call(
                call, args.warmups, args.samples, args.minimum_sample_seconds)
            rows.append(summarize_row(
                "fixed_degree", method, segments, parameter, timings, repetitions,
                references[segments], value, metadata))
            print("fixed", method, segments, float(np.median(timings)))
    return rows


def accuracy_matched_rows(library, grams, references, args, metadata):
    rows = []
    for segments in SEGMENTS:
        reference = references[segments]
        for method in METHODS:
            selected = None
            value = None
            last_parameter = None
            if method == "finite_difference":
                dyadic_order = 0
                while (((segments << dyadic_order) + 1) ** 2) <= 25_000_000:
                    last_parameter = dyadic_order
                    call = make_solver_call(
                        library, method, grams[segments], segments, dyadic_order)
                    candidate = float(call())
                    value = candidate
                    if meets_tolerance(candidate, reference):
                        selected = dyadic_order
                        break
                    dyadic_order += 1
                missing_status = "out_of_budget"
            else:
                for degree in range(2, 33):
                    last_parameter = degree
                    call = make_solver_call(library, method, grams[segments], segments, degree)
                    candidate = float(call())
                    value = candidate
                    if meets_tolerance(candidate, reference):
                        selected = degree
                        break
                missing_status = "degree_limit"

            if selected is None:
                rows.append(summarize_row(
                    "accuracy_matched", method, segments, last_parameter, None, 0,
                    reference, value, metadata, missing_status))
                print("accuracy", method, segments, missing_status)
                continue

            call = make_solver_call(library, method, grams[segments], segments, selected)
            timings, repetitions = benchmark_call(
                call, args.warmups, args.samples, args.minimum_sample_seconds)
            rows.append(summarize_row(
                "accuracy_matched", method, segments, selected, timings, repetitions,
                reference, value, metadata))
            print("accuracy", method, segments, selected, float(np.median(timings)))
    return rows


def plot_results(rows, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    panels = (
        ("fixed_degree", "Fixed degree 7"),
        ("accuracy_matched", "Accuracy matched"),
    )
    for axis, (panel, title) in zip(axes, panels):
        panel_rows = [row for row in rows if row["panel"] == panel]
        for method in METHODS:
            method_rows = [row for row in panel_rows
                           if row["method"] == method and row["status"] == "ok"]
            x = np.array([row["path_length"] for row in method_rows])
            median = np.array([row["median_seconds"] for row in method_rows])
            q1 = np.array([row["q1_seconds"] for row in method_rows])
            q3 = np.array([row["q3_seconds"] for row in method_rows])
            if x.size:
                axis.plot(x, median, marker="o", color=COLORS[method], label=LABELS[method])
                axis.fill_between(x, q1, q3, color=COLORS[method], alpha=0.18)

        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Path segments")
        axis.set_ylabel("Native solver time (seconds)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

        if panel == "accuracy_matched":
            y_top = axis.get_ylim()[1]
            for method in METHODS:
                missing = [row for row in panel_rows
                           if row["method"] == method and row["status"] != "ok"]
                if missing:
                    axis.scatter(
                        [row["path_length"] for row in missing],
                        [y_top / 1.4] * len(missing), marker="x", s=55,
                        color=COLORS[method])
            axis.text(
                0.98, 0.98, "x: tolerance not met within limit",
                transform=axis.transAxes, ha="right", va="top", fontsize=8)

    png_path = output_dir / "sig_kernel_methods_cpu.png"
    svg_path = output_dir / "sig_kernel_methods_cpu.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def plot_memory(rows, output_dir):
    import matplotlib.pyplot as plt

    mib = 1024.0 * 1024.0
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    plot_methods = (
        ("finite_difference", "Finite difference"),
        ("polynomial", "Polynomial"),
    )
    panels = (
        ("logical_solver_bytes", "Native solver workspace"),
        ("logical_total_bytes", "Gram plus native buffers"),
        ("peak_total_median_bytes", "Measured process-memory increase"),
    )
    for axis, (field, title) in zip(axes, panels):
        for method, label in plot_methods:
            method_rows = [row for row in rows if row["method"] == method]
            x = np.array([row["path_length"] for row in method_rows])
            y = np.array([row[field] for row in method_rows], dtype=np.float64) / mib
            positive = y > 0
            axis.plot(
                x[positive], y[positive], marker="o", color=COLORS[method],
                label=label)
            if field == "peak_total_median_bytes":
                q1 = np.array(
                    [row["peak_total_q1_bytes"] for row in method_rows],
                    dtype=np.float64) / mib
                q3 = np.array(
                    [row["peak_total_q3_bytes"] for row in method_rows],
                    dtype=np.float64) / mib
                band = positive & (q1 > 0) & (q3 > 0)
                axis.fill_between(
                    x[band], q1[band], q3[band], color=COLORS[method], alpha=0.18)

        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Path segments")
        axis.set_ylabel("Memory (MiB)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

    axes[2].text(
        0.03, 0.03, "Zero increments below allocator resolution are omitted",
        transform=axes[2].transAxes, ha="left", va="bottom", fontsize=8)
    png_path = output_dir / "sig_kernel_methods_cpu_memory.png"
    svg_path = output_dir / "sig_kernel_methods_cpu_memory.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--minimum-sample-seconds", type=float, default=0.2)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=25022025)
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--memory-samples", type=int, default=3)
    parser.add_argument("--memory-duration", type=float, default=0.1)
    parser.add_argument("--memory-worker", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--memory-segments", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--memory-parameter", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.memory_worker is not None:
        result = memory_worker(
            args.memory_worker, args.memory_segments, args.memory_parameter,
            args.memory_duration, args.seed)
        print(json.dumps(result, sort_keys=True))
        return

    if not args.memory_only and args.samples < 15:
        raise ValueError("samples must be at least 15")
    if not args.memory_only and args.warmups < 3:
        raise ValueError("warmups must be at least 3")
    if not args.memory_only and args.minimum_sample_seconds < 0.2:
        raise ValueError("minimum-sample-seconds must be at least 0.2")
    if args.memory_samples < 1:
        raise ValueError("memory-samples must be at least 1")
    if args.memory_duration <= 0:
        raise ValueError("memory-duration must be positive")
    if args.memory_only and args.skip_memory:
        raise ValueError("memory-only and skip-memory cannot be used together")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or repo_root / "out" / "benchmarks" / "sig_kernel_methods_cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    library = load_cpsig(repo_root)
    logical_cpu = pin_logical_cpu(args.cpu)
    metadata = {
        "samples": args.samples,
        "seed": args.seed,
        "logical_cpu": logical_cpu,
        "compiler": compiler_name(repo_root),
        "cpu_model": cpu_model(),
        "git_revision": git_revision(repo_root),
    }

    if not args.memory_only:
        paths = brownian_paths(args.seed)
        grams = {
            segments: np.ascontiguousarray(
                np.diff(path1, axis=0) @ np.diff(path2, axis=0).T)
            for segments, (path1, path2) in paths.items()
        }
        references = {
            segments: direct_signature_kernel(library, path1, path2)
            for segments, (path1, path2) in paths.items()
        }

        rows = fixed_degree_rows(library, grams, references, args, metadata)
        rows.extend(accuracy_matched_rows(library, grams, references, args, metadata))
        csv_path = output_dir / "sig_kernel_methods_cpu.csv"
        with csv_path.open("w", newline="", encoding="ascii") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_results(rows, output_dir)
        print(csv_path)

    if not args.skip_memory:
        memory_rows = benchmark_memory(repo_root, args, metadata)
        memory_csv_path = output_dir / "sig_kernel_methods_cpu_memory.csv"
        with memory_csv_path.open("w", newline="", encoding="ascii") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(memory_rows[0]))
            writer.writeheader()
            writer.writerows(memory_rows)
        plot_memory(memory_rows, output_dir)
        print(memory_csv_path)


if __name__ == "__main__":
    main()
