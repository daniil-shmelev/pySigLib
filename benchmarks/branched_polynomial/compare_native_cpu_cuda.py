"""Compare the canonical native CPU/CUDA branched solvers on identical inputs.

Run from a built/installed checkout with NumPy, PyTorch and Matplotlib:
    python benchmarks/branched_polynomial/compare_native_cpu_cuda.py

Both backward passes reconstruct their forward state. Timings cover the native
forward and backward entry points, including allocation and synchronization,
but exclude Gram construction, transfers and the common Gram-to-path chain.
"""
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bsk-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from compare import make_paths, path_gradients, relative_norm, write_csv
from pysiglib.error_codes import err_msg
from pysiglib.load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FAMILIES = ("brownian_cross", "brownian_self", "oscillatory_cross")


def pointer(tensor):
    return ct.cast(tensor.data_ptr(), ct.POINTER(ct.c_double))


class NativeSolver:
    def __init__(self, gram, depth, method, parameter, device):
        self.device = device
        self.gram = torch.tensor(gram, dtype=torch.float64, device=device)
        self.out = torch.empty(1, dtype=torch.float64, device=device)
        self.grad = torch.empty_like(self.gram)
        self.seed = torch.ones_like(self.out)
        nx, ny = gram.shape
        lib = CPSIG if device == "cpu" else CUSIG
        suffix = "_d" if device == "cpu" else "_cuda_d"
        tail = (1,) if device == "cpu" else ()
        sizes = (1, 3, nx+1, ny+1, depth)
        if method == "polynomial":
            # Labels use q nodes, whereas the public API accepts degree q-1.
            self.forward = getattr(lib, "branched_sig_kernel_poly" + suffix)
            self.backward = getattr(lib, "branched_sig_kernel_poly_backprop" + suffix)
            self.forward_args = (pointer(self.gram), pointer(self.out), None, *sizes, parameter-1, False, *tail)
            self.backward_args = (pointer(self.gram), pointer(self.grad), pointer(self.seed), None, *sizes, parameter-1, False, *tail)
        else:
            self.forward = getattr(lib, "branched_sig_kernel" + suffix)
            self.backward = getattr(lib, "branched_sig_kernel_backprop" + suffix)
            self.forward_args = (pointer(self.gram), pointer(self.out), *sizes, parameter, parameter, False, *tail)
            self.backward_args = (pointer(self.gram), pointer(self.grad), pointer(self.seed), None, *sizes, parameter, parameter, False, *tail)

    def __call__(self):
        for fn, args in ((self.forward, self.forward_args), (self.backward, self.backward_args)):
            status = fn(*args)
            if status:
                raise RuntimeError(err_msg(status, self.device))

    def evaluate(self):
        self()
        return self.out.item(), self.grad.cpu().numpy().copy()


def benchmark(solver, samples, minimum_ms):
    for _ in range(3):
        solver()
    start = time.perf_counter()
    solver()
    elapsed_ms = (time.perf_counter()-start)*1e3
    repetitions = min(10000, max(1, math.ceil(minimum_ms/max(elapsed_ms, 1e-4))))
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(repetitions):
            solver()
        times.append((time.perf_counter()-start)*1e3/repetitions)
    return dict(forward_backward_ms=float(np.median(times)),
                minimum_ms=min(times), maximum_ms=max(times), repetitions=repetitions)


def plot(rows, output, segments, depth, gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    for row, device in enumerate(("cpu", "cuda")):
        for column, family in enumerate(FAMILIES):
            ax = axes[row, column]
            for method, color, marker, label in (
                ("finite_difference", "#bf5143", "o", "finite difference"),
                ("polynomial", "#246db3", "s", "polynomial"),
            ):
                selected = sorted((r for r in rows if r["device"] == device and r["family"] == family and r["method"] == method), key=lambda r: r["parameter"])
                ax.loglog([r["forward_backward_ms"] for r in selected],
                          [max(r["joint_error"], 1e-16) for r in selected],
                          marker=marker, color=color, linewidth=2, markersize=7, label=label)
                for r in selected:
                    ax.annotate(str(r["parameter"]),
                                (r["forward_backward_ms"], max(r["joint_error"], 1e-16)),
                                xytext=(5, 5), textcoords="offset points", fontsize=10)
            ax.set_title(("CPU" if device == "cpu" else "GPU") + " · " + family.replace("_", " "))
            ax.set_xlabel("Forward + backward (ms)")
            ax.grid(True, which="both", alpha=.22)
            ax.legend(fontsize=10)
    fig.supylabel("Max relative error: kernel, both path gradients", fontsize=13)
    fig.suptitle(f"Branched kernel: {segments} segments, depth {depth}, float64\n"
                 f"CPU: one core · GPU: {gpu}\n"
                 "Labels: FD dyadic order / polynomial nodes per coordinate (order + 1)", fontsize=17)
    fig.subplots_adjust(left=.075, right=.985, bottom=.075, top=.84, hspace=.32, wspace=.08)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"accuracy_runtime_cpu_cuda.{extension}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--minimum-ms", type=float, default=10.)
    args = parser.parse_args()
    if not BUILT_WITH_CUDA or not torch.cuda.is_available():
        raise RuntimeError("This six-panel comparison requires a CUDA build and an available GPU")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    gpu = torch.cuda.get_device_name()
    rows, references = [], []
    start = time.monotonic()
    for family in FAMILIES:
        dx, dy = make_paths(family, args.segments, args.seed)
        gram = dx @ dy.T
        low, low_grad = NativeSolver(gram, args.depth, "polynomial", 16, "cpu").evaluate()
        ref, ref_grad = NativeSolver(gram, args.depth, "polynomial", 20, "cpu").evaluate()
        ref_path = path_gradients(ref_grad, dx, dy)
        value_gap = abs(low-ref)/abs(ref)
        path_gap = relative_norm(path_gradients(low_grad, dx, dy), ref_path)
        if max(value_gap, path_gap) > 1e-11:
            raise RuntimeError(f"Reference did not converge for {family}")
        references.append(dict(family=family, kernel=ref, nodes=20,
                               q16_q20_kernel_gap=value_gap, q16_q20_path_gradient_gap=path_gap))
        for device in ("cpu", "cuda"):
            settings = [("finite_difference", r) for r in range(7)] + [("polynomial", q) for q in (1, 2, 3, 4, 6, 8)]
            # Fixed shuffled order avoids a systematic method/warmth bias.
            np.random.default_rng(args.seed).shuffle(settings)
            for method, parameter in settings:
                solver = NativeSolver(gram, args.depth, method, parameter, device)
                timing = benchmark(solver, args.samples, args.minimum_ms)
                value, gradient = solver.evaluate()
                kernel_error = abs(value-ref)/abs(ref)
                path_error = relative_norm(path_gradients(gradient, dx, dy), ref_path)
                rows.append(dict(device=device, family=family, segments=args.segments,
                                 depth=args.depth, seed=args.seed, method=method, parameter=parameter,
                                 api_order=parameter-1 if method == "polynomial" else "",
                                 kernel=value, kernel_relative_error=kernel_error,
                                 gram_gradient_relative_error=relative_norm(gradient, ref_grad),
                                 path_gradient_relative_error=path_error,
                                 joint_error=max(kernel_error, path_error), **timing))
            write_csv(output / "measurements_cpu_cuda.csv", rows)
            print(f"{device:4s} {family}: complete ({time.monotonic()-start:.1f}s)", flush=True)
    write_csv(output / "references_cpu_cuda.csv", references)
    plot(rows, output, args.segments, args.depth, gpu)
    sources = [Path(__file__), HERE/"compare.py", ROOT/"siglib/shared/branched_polynomial.h",
               ROOT/"siglib/shared/preparation/polynomial_sig_kernel/branched_polynomial_tables.h",
               ROOT/"siglib/cpsig/cp_branched_polynomial_sig_kernel.cpp",
               ROOT/"siglib/cusig/cu_branched_polynomial_sig_kernel.cu",
               ROOT/"siglib/cpsig/cp_branched_sig_kernel.cpp",
               ROOT/"siglib/cusig/cu_branched_sig_kernel.cu"]
    metadata = dict(git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    cpu=next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
                    logical_cpu=cpu, gpu=gpu, platform=platform.platform(), torch=torch.__version__,
                    cuda=torch.version.cuda, numpy=np.__version__, dtype="float64",
                    segments=args.segments, depth=args.depth, seed=args.seed,
                    samples=args.samples, minimum_sample_ms=args.minimum_ms,
                    elapsed_seconds=time.monotonic()-start,
                    timing_scope="Median wall time for native scalar-pair forward + backward. Both backward passes reconstruct forward state. Includes native allocation, launches, synchronization and ctypes dispatch. Excludes Gram construction, device transfers and Gram-to-path chain. Cached quadrature tables; CPU scratch is reused, CUDA scratch is allocated per call. CPU pinned to one core, n_jobs=1.",
                    accuracy="Maximum of relative kernel error and relative L2 error in concatenated gradients of both input paths. CPU q=20 reference checked against q=16. Plot floors errors at 1e-16.",
                    parameter="FD labels are dyadic orders; polynomial labels are q nodes per coordinate, i.e. API order=q-1.")
    (output / "metadata_cpu_cuda.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(output / "accuracy_runtime_cpu_cuda.png", flush=True)


if __name__ == "__main__":
    main()
