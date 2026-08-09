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
from ctypes import POINTER, c_size_t, c_void_p
import gc
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def pin_logical_cpu(requested_cpu):
    if platform.system() == "Windows":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = c_void_p
        kernel32.GetProcessAffinityMask.argtypes = (
            c_void_p, POINTER(c_size_t), POINTER(c_size_t))
        kernel32.GetProcessAffinityMask.restype = ctypes.c_bool
        kernel32.SetProcessAffinityMask.argtypes = (c_void_p, c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
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
    raise RuntimeError("logical CPU pinning is unavailable")


def cpu_model():
    if platform.system() == "Windows":
        import winreg
        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def benchmark(call, warmups, samples, minimum_sample_seconds):
    repetitions = 1
    while True:
        start = time.perf_counter()
        for _ in range(repetitions):
            call()
        elapsed = time.perf_counter() - start
        if elapsed >= minimum_sample_seconds:
            break
        repetitions *= max(
            2, int(minimum_sample_seconds / max(elapsed, 1e-12)) + 1)
    for _ in range(warmups):
        for _ in range(repetitions):
            call()
    timings = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            start = time.perf_counter()
            for _ in range(repetitions):
                call()
            timings.append((time.perf_counter() - start) / repetitions)
    finally:
        if gc_enabled:
            gc.enable()
    timings.sort()
    return {
        "median_seconds": statistics.median(timings),
        "q1_seconds": statistics.median(timings[:len(timings) // 2]),
        "q3_seconds": statistics.median(timings[(len(timings) + 1) // 2:]),
        "repetitions": repetitions,
    }


def block_tree(jax, value):
    for leaf in jax.tree_util.tree_leaves(value):
        leaf.block_until_ready()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", type=int, default=7)
    parser.add_argument("--lengths", type=int, nargs="+", default=(16, 32, 64, 128, 256, 512))
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--minimum-sample-seconds", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=25022025)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(1, str(args.upstream_root.resolve()))
    sys.modules["pysiglib_cuda"] = None

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from polysigkernel import SigKernel
    from pysiglib import jax_api as pysiglib_jax

    if not hasattr(jax.lib, "xla_bridge"):
        jax.lib.xla_bridge = SimpleNamespace(
            get_backend=lambda platform_name: SimpleNamespace(
                device_count=lambda: len(jax.devices(platform_name))))

    logical_cpu = pin_logical_cpu(args.cpu)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    max_length = max(args.lengths)
    increments1 = rng.normal(scale=1 / np.sqrt(max_length), size=(max_length, 2))
    increments2 = rng.normal(scale=1 / np.sqrt(max_length), size=(max_length, 2))
    full_path1 = np.vstack((np.zeros(2), np.cumsum(increments1, axis=0)))
    full_path2 = np.vstack((np.zeros(2), np.cumsum(increments2, axis=0)))
    upstream_solver = SigKernel(
        order=args.order, static_kernel="linear", solver="monomial_approx")
    rows = []

    for segments in args.lengths:
        if max_length % segments:
            raise ValueError("each length must divide the maximum length")
        stride = max_length // segments
        path1 = jax.device_put(jnp.asarray(full_path1[::stride][None]))
        path2 = jax.device_put(jnp.asarray(full_path2[::stride][None]))

        def polynomial_objective(left, right):
            return jnp.sum(pysiglib_jax.sig_kernel(
                left, right, method="polynomial", order=args.order))

        def goursat_objective(left, right):
            return jnp.sum(pysiglib_jax.sig_kernel(
                left, right, dyadic_order=0, method="finite_difference"))

        def upstream_objective(left, right):
            return jnp.sum(upstream_solver.kernel_matrix(left, right))

        methods = {
            "pysiglib_polynomial": jax.jit(jax.value_and_grad(
                polynomial_objective, argnums=(0, 1))),
            "pysiglib_goursat": jax.jit(jax.value_and_grad(
                goursat_objective, argnums=(0, 1))),
            "polysigkernel": jax.jit(jax.value_and_grad(
                upstream_objective, argnums=(0, 1))),
        }
        outputs = {}
        for name, call_grad in methods.items():
            compiled = call_grad(path1, path2)
            block_tree(jax, compiled)
            outputs[name] = compiled

            def call():
                block_tree(jax, call_grad(path1, path2))

            timing = benchmark(
                call, args.warmups, args.samples, args.minimum_sample_seconds)
            polynomial_grads = outputs["pysiglib_polynomial"][1]
            grad_error = max(
                float(jnp.max(jnp.abs(compiled[1][0] - polynomial_grads[0]))),
                float(jnp.max(jnp.abs(compiled[1][1] - polynomial_grads[1]))),
            )
            rows.append({
                "implementation": name,
                "segments": segments,
                "order": args.order if name != "pysiglib_goursat" else "",
                "dyadic_order": 0 if name == "pysiglib_goursat" else "",
                "median_seconds": timing["median_seconds"],
                "q1_seconds": timing["q1_seconds"],
                "q3_seconds": timing["q3_seconds"],
                "repetitions": timing["repetitions"],
                "samples": args.samples,
                "value": float(compiled[0]),
                "max_gradient_difference_from_polynomial": grad_error,
            })
            print(
                name, segments,
                "median_ms=", 1000 * timing["median_seconds"],
                "gradient_difference=", grad_error)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        check=True, capture_output=True, text=True).stdout.strip()
    metadata = {
        "logical_cpu": logical_cpu,
        "cpu_model": cpu_model(),
        "git_revision": revision,
        "jax_version": jax.__version__,
        "seed": args.seed,
    }
    for row in rows:
        row.update(metadata)
    csv_path = output_dir / "backprop_runtime.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    styles = {
        "pysiglib_polynomial": ("pySigLib polynomial", "o"),
        "pysiglib_goursat": ("pySigLib Goursat (dyadic 0)", "s"),
        "polysigkernel": ("polysigkernel", "^"),
    }
    for name, (label, marker) in styles.items():
        selected = [row for row in rows if row["implementation"] == name]
        x = np.asarray([row["segments"] for row in selected])
        median = np.asarray([row["median_seconds"] for row in selected])
        q1 = np.asarray([row["q1_seconds"] for row in selected])
        q3 = np.asarray([row["q3_seconds"] for row in selected])
        axis.plot(x, 1000 * median, marker=marker, label=label)
        axis.fill_between(x, 1000 * q1, 1000 * q3, alpha=0.18)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Path segments")
    axis.set_ylabel("Forward and backward runtime (ms)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "backprop_runtime.png", dpi=180)
    fig.savefig(output_dir / "backprop_runtime.svg")


if __name__ == "__main__":
    main()
