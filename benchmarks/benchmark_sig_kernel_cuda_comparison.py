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

import argparse
import csv
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METHOD_ORDER = {
    "pysiglib_polynomial": 0,
    "polysigkernel_jax": 1,
    "pysiglib_goursat": 2,
}


def brownian_paths(segments, seed):
    rng = np.random.default_rng(seed + segments)
    scale = np.float32(1.0 / math.sqrt(segments))
    increments1 = rng.normal(size=(1, segments, 2)).astype(np.float32) * scale
    increments2 = rng.normal(size=(1, segments, 2)).astype(np.float32) * scale
    zeros = np.zeros((1, 1, 2), dtype=np.float32)
    path1 = np.concatenate(
        (zeros, np.cumsum(increments1, axis=1, dtype=np.float32)), axis=1)
    path2 = np.concatenate(
        (zeros, np.cumsum(increments2, axis=1, dtype=np.float32)), axis=1)
    return path1, path2


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def benchmark(function, synchronize, warmups, samples, min_sample_seconds):
    start = time.perf_counter()
    result = function()
    synchronize(result)
    first_call_seconds = time.perf_counter() - start
    for _ in range(warmups - 1):
        result = function()
        synchronize(result)

    repetitions = 1
    while True:
        start = time.perf_counter()
        for _ in range(repetitions):
            result = function()
        synchronize(result)
        elapsed = time.perf_counter() - start
        if elapsed >= min_sample_seconds:
            break
        repetitions = max(
            repetitions + 1,
            math.ceil(repetitions * min_sample_seconds * 1.1 / max(elapsed, 1e-9)))

    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(repetitions):
            result = function()
        synchronize(result)
        timings.append((time.perf_counter() - start) / repetitions)

    return result, first_call_seconds, repetitions, timings


def pysiglib_cases(path1, path2, order, operation):
    import pysiglib
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available through PyTorch")
    device_path1 = torch.from_numpy(path1).cuda().requires_grad_(operation == "backprop")
    device_path2 = torch.from_numpy(path2).cuda().requires_grad_(operation == "backprop")
    torch.cuda.synchronize()

    if operation == "backprop":
        def polynomial():
            result = pysiglib.torch_api.sig_kernel(
                device_path1, device_path2, method="polynomial", order=order)
            gradients = torch.autograd.grad(result.sum(), (device_path1, device_path2))
            return result, gradients

        def goursat():
            result = pysiglib.torch_api.sig_kernel(
                device_path1, device_path2,
                method="finite_difference", dyadic_order=0)
            gradients = torch.autograd.grad(result.sum(), (device_path1, device_path2))
            return result, gradients

        polynomial_function = polynomial
        goursat_function = goursat
    else:
        polynomial_function = lambda: pysiglib.sig_kernel(
            device_path1, device_path2, method="polynomial", order=order)
        goursat_function = lambda: pysiglib.sig_kernel(
            device_path1, device_path2,
            method="finite_difference", dyadic_order=0)

    return [
        (
            "pysiglib_polynomial",
            "pySigLib polynomial",
            polynomial_function,
            lambda result: torch.cuda.synchronize(),
            torch.cuda.get_device_name(),
            torch.__version__,
        ),
        (
            "pysiglib_goursat",
            "pySigLib Goursat",
            goursat_function,
            lambda result: torch.cuda.synchronize(),
            torch.cuda.get_device_name(),
            torch.__version__,
        ),
    ]


def polysigkernel_cases(path1, path2, order, operation):
    import jax

    if not hasattr(jax.lib, "xla_bridge"):
        jax.lib.xla_bridge = types.SimpleNamespace(
            get_backend=lambda backend: types.SimpleNamespace(
                device_count=lambda: len(jax.devices(backend))))
    import jax.numpy as jnp
    from polysigkernel import SigKernel

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("CUDA is not available through JAX")
    device_path1 = jax.device_put(jnp.asarray(path1), devices[0])
    device_path2 = jax.device_put(jnp.asarray(path2), devices[0])
    kernel = SigKernel(order=order, static_kernel="linear")

    if operation == "backprop":
        def objective(left, right):
            return jnp.sum(kernel.kernel_matrix(left, right))

        function = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
        case_function = lambda: function(device_path1, device_path2)
    else:
        case_function = lambda: kernel.kernel_matrix(device_path1, device_path2)

    return [
        (
            "polysigkernel_jax",
            "polysigkernel JAX",
            case_function,
            lambda result: jax.block_until_ready(result),
            str(devices[0].device_kind),
            jax.__version__,
        )
    ]


def result_value(result):
    if isinstance(result, (tuple, list)):
        return result_value(result[0])
    if hasattr(result, "detach"):
        result = result.detach().cpu()
    array = np.asarray(result)
    return float(array.reshape(-1)[0])


def run_benchmarks(args):
    rows = []
    revision = args.revision or git_revision()
    for segments in args.lengths:
        path1, path2 = brownian_paths(segments, args.seed)
        if args.backend == "pysiglib":
            cases = pysiglib_cases(path1, path2, args.order, args.operation)
        else:
            cases = polysigkernel_cases(path1, path2, args.order, args.operation)

        for method, label, function, synchronize, device, framework in cases:
            result, first_call, repetitions, timings = benchmark(
                function, synchronize, args.warmups, args.samples,
                args.min_sample_seconds)
            q1, median, q3 = np.quantile(timings, (0.25, 0.5, 0.75))
            row = {
                "method": method,
                "label": label,
                "segments": segments,
                "dimension": 2,
                "dtype": "float32",
                "order": args.order if method != "pysiglib_goursat" else "",
                "dyadic_order": 0 if method == "pysiglib_goursat" else "",
                "median_seconds": f"{median:.12g}",
                "q1_seconds": f"{q1:.12g}",
                "q3_seconds": f"{q3:.12g}",
                "minimum_seconds": f"{min(timings):.12g}",
                "maximum_seconds": f"{max(timings):.12g}",
                "first_call_seconds": f"{first_call:.12g}",
                "inner_repetitions": repetitions,
                "samples": args.samples,
                "value": f"{result_value(result):.12g}",
                "scope": "end_to_end",
                "operation": args.operation,
                "device": device,
                "framework_version": framework,
                "platform": platform.platform(),
                "revision": revision,
            }
            rows.append(row)
            print(
                f"{label}: segments={segments}, median={1e3 * median:.3f} ms, "
                f"IQR=[{1e3 * q1:.3f}, {1e3 * q3:.3f}] ms, "
                f"repetitions={repetitions}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            rows.extend(csv.DictReader(source))
    rows.sort(key=lambda row: (METHOD_ORDER[row["method"]], int(row["segments"])))
    return rows


def plot_benchmarks(args):
    import matplotlib.pyplot as plt

    rows = load_rows(args.input)
    methods = list(dict.fromkeys(row["method"] for row in rows))
    colors = {
        "pysiglib_polynomial": "#0072B2",
        "polysigkernel_jax": "#D55E00",
        "pysiglib_goursat": "#009E73",
    }
    markers = {
        "pysiglib_polynomial": "o",
        "polysigkernel_jax": "s",
        "pysiglib_goursat": "^",
    }

    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        x = np.array([int(row["segments"]) for row in selected])
        median = 1e3 * np.array([float(row["median_seconds"]) for row in selected])
        q1 = 1e3 * np.array([float(row["q1_seconds"]) for row in selected])
        q3 = 1e3 * np.array([float(row["q3_seconds"]) for row in selected])
        axis.plot(
            x, median, marker=markers[method], linewidth=2,
            color=colors[method], label=selected[0]["label"])
        axis.fill_between(x, q1, q3, color=colors[method], alpha=0.16)

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Path segments per path")
    axis.set_ylabel("End-to-end runtime (ms)")
    operation = rows[0].get("operation", "forward")
    title_operation = "forward and backward" if operation == "backprop" else "forward"
    axis.set_title(
        "CUDA signature kernel " + title_operation + " runtime, one path pair, float32")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)

    args.png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png, dpi=180)
    figure.savefig(args.svg)
    plt.close(figure)

    if args.output_csv:
        with args.output_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", choices=("pysiglib", "polysigkernel"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--lengths", type=int, nargs="+", default=(16, 32, 64, 128, 256, 512, 1024))
    run_parser.add_argument("--order", type=int, default=7)
    run_parser.add_argument("--seed", type=int, default=25022025)
    run_parser.add_argument("--warmups", type=int, default=3)
    run_parser.add_argument("--samples", type=int, default=15)
    run_parser.add_argument("--min-sample-seconds", type=float, default=0.2)
    run_parser.add_argument("--revision")
    run_parser.add_argument("--operation", choices=("forward", "backprop"), default="forward")

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--input", type=Path, nargs="+", required=True)
    plot_parser.add_argument("--output-csv", type=Path)
    plot_parser.add_argument("--png", type=Path, required=True)
    plot_parser.add_argument("--svg", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "run":
        run_benchmarks(arguments)
    else:
        plot_benchmarks(arguments)
