"""Reproducible native CPU comparison of branched kernel solvers.

Run with a Python environment containing numpy and matplotlib:
    python benchmarks/branched_polynomial/compare.py --output /tmp/bsk-comparison

The prototype and the unchanged repository finite-difference source are compiled
in one translation unit. The local CMake build supplies oneTBB headers/library.
Timing excludes Gram construction and the common Gram-to-path reverse chain.
Both backward calls reconstruct their forward state; no saved-state advantage
is given to the prototype. This is an experiment, not a public API addition.
"""

import argparse
import csv
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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOUBLE_PTR = ct.POINTER(ct.c_double)


def pointer(array):
    return array.ctypes.data_as(DOUBLE_PTR)


def tables(q):
    """Integrated Lagrange basis via discrete Legendre orthogonality."""
    t, w = np.polynomial.legendre.leggauss(q)
    w = np.ascontiguousarray(w / 2)
    v = np.polynomial.legendre.legvander(t, q)
    integ = np.empty((q, q))
    integ[:, 0] = (t + 1) / 2
    for k in range(1, q):
        integ[:, k] = (v[:, k + 1] - v[:, k - 1]) / (2 * (2 * k + 1))
    A = (integ * (2 * np.arange(q) + 1)) @ v[:, :q].T
    return np.ascontiguousarray(A * w[None, :]), w


def build(output, compiler):
    includes = sorted((ROOT / "build").glob("*/_deps/onetbb-src/include"))
    libraries = sorted((ROOT / "build").glob("**/libtbb.so.12.*"))
    if not includes or not libraries:
        raise RuntimeError("Build cpsig with CMake first to supply oneTBB headers and library")
    library = libraries[0].resolve()
    target = output / "libbranched_comparison.so"
    command = [compiler, "-std=c++20", "-O3", "-DNDEBUG", "-march=native",
               "-fPIC", "-shared", "-pthread", "-I" + str(includes[0]),
               str(HERE / "prototype.cpp"), str(library),
               "-Wl,-rpath," + str(library.parent), "-o", str(target)]
    subprocess.run(command, check=True)
    return target, command


def load(path):
    lib = ct.CDLL(str(path))
    lib.bsk_reset.argtypes = []
    lib.bsk_reset.restype = None
    lib.bsk_bench.argtypes = [ct.c_int, ct.c_int, DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR,
                             ct.c_uint64, ct.c_uint64, ct.c_uint64, ct.c_uint64,
                             DOUBLE_PTR, DOUBLE_PTR, ct.c_uint64]
    lib.bsk_bench.restype = ct.c_double
    lib.branched_sig_kernel_d.argtypes = [DOUBLE_PTR, DOUBLE_PTR] + [ct.c_uint64] * 7 + [ct.c_bool, ct.c_int]
    lib.branched_sig_kernel_d.restype = ct.c_int
    return lib


class Solver:
    def __init__(self, lib, gram, depth, method, parameter):
        self.lib = lib
        self.gram = np.ascontiguousarray(gram, dtype=np.float64)
        self.out = np.zeros(1)
        self.grad = np.zeros_like(self.gram)
        self.A, self.w = tables(parameter) if method == "polynomial" else (np.zeros(1), np.zeros(1))
        self.method = 1 if method == "polynomial" else 0
        self.depth = depth
        self.parameter = parameter

    def __call__(self, mode=2, repetitions=1):
        return self.lib.bsk_bench(self.method, mode, pointer(self.gram), pointer(self.out),
                                  pointer(self.grad), *self.gram.shape, self.depth,
                                  self.parameter, pointer(self.A), pointer(self.w), repetitions)

    def evaluate(self):
        self.lib.bsk_reset()
        self()
        return float(self.out[0]), self.grad.copy()


def numpy_reference(gram, depth, q):
    """Independent dense cumulative operators; also supports complex-step AD."""
    A, w = tables(q)
    nx, ny = gram.shape
    def cumulative(n):
        return np.kron(np.eye(n), A) + np.kron(np.tril(np.ones((n, n)), -1), np.tile(w, (q, 1)))
    Qx, Qy = cumulative(nx), cumulative(ny)
    C = np.repeat(np.repeat(gram, q, 0), q, 1)
    K = np.ones_like(C)
    for m in range(1, depth + 1):
        F = C * K
        if m == depth:
            return np.exp(np.tile(w, nx) @ F @ np.tile(w, ny))
        K = np.exp(Qx @ F @ Qy.T)
    return 1.0


def trapezoid_reference(gram, depth, refinement):
    factor = 2**refinement
    C = np.repeat(np.repeat(gram, factor, 0), factor, 1) / factor**2
    K = np.ones((C.shape[0] + 1, C.shape[1] + 1), dtype=gram.dtype)
    for _ in range(depth):
        F = C * (K[:-1, :-1] + K[1:, :-1] + K[:-1, 1:] + K[1:, 1:]) / 4
        K = np.ones_like(K)
        K[1:, 1:] = np.exp(np.cumsum(np.cumsum(F, 0), 1))
    return K[-1, -1]


def path_gradients(grad, dx, dy):
    d_dx, d_dy = grad @ dy, grad.T @ dx
    gx = np.zeros((len(dx) + 1, dx.shape[1]))
    gy = np.zeros((len(dy) + 1, dy.shape[1]))
    gx[:-1] -= d_dx
    gx[1:] += d_dx
    gy[:-1] -= d_dy
    gy[1:] += d_dy
    return np.concatenate((gx.ravel(), gy.ravel()))


def relative_norm(actual, reference):
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-300))


def validate(lib):
    rng = np.random.default_rng(712)
    gram = rng.normal(scale=0.07, size=(2, 3))
    worst_value, worst_grad = 0.0, 0.0
    checks = 0
    for method, parameters, reference in (
            ("polynomial", (1, 2, 4, 7), numpy_reference),
            ("finite_difference", (0, 1, 3), trapezoid_reference)):
        for depth in (0, 1, 2, 4):
            for parameter in parameters:
                value, grad = Solver(lib, gram, depth, method, parameter).evaluate()
                expected = reference(gram, depth, parameter)
                np.testing.assert_allclose(value, expected, rtol=3e-13, atol=3e-14)
                expected_grad = np.zeros_like(gram)
                for idx in np.ndindex(gram.shape):
                    perturbed = gram.astype(complex)
                    perturbed[idx] += 1e-30j
                    expected_grad[idx] = np.imag(reference(perturbed, depth, parameter)) / 1e-30
                np.testing.assert_allclose(grad, expected_grad, rtol=2e-12, atol=5e-13)
                worst_value = max(worst_value, abs(value - expected))
                worst_grad = max(worst_grad, float(np.max(abs(grad - expected_grad))))
                checks += 1
    # Closed form for a single cell at depth two:
    # log K_2(rho) = sum_{k>=1} rho^k / (k k!).
    exact_errors = []
    for rho in (-2.0, -0.2, 0.0, 0.5, 2.0):
        expected = math.exp(sum(rho**k / (k * math.factorial(k)) for k in range(1, 100)))
        expected_grad = expected * (math.expm1(rho) / rho if rho else 1.0)
        value, grad = Solver(lib, np.array([[rho]]), 2, "polynomial", 16).evaluate()
        np.testing.assert_allclose([value, grad[0, 0]], [expected, expected_grad], rtol=5e-13, atol=5e-13)
        exact_errors.append(abs(value - expected))
    # Native adapter agrees with the actual public C entry point.
    out = np.empty(1)
    for depth in (1, 3):
        for refinement in (0, 2):
            expected, _ = Solver(lib, gram, depth, "finite_difference", refinement).evaluate()
            status = lib.branched_sig_kernel_d(pointer(gram), pointer(out), 1, 3, 3, 4,
                                               depth, refinement, refinement, False, 1)
            assert status == 0
            np.testing.assert_allclose(out[0], expected, rtol=1e-14)
    # Check the common chain rule against perturbations of both paths.
    dx, dy = rng.normal(scale=.1, size=(2, 3)), rng.normal(scale=.1, size=(3, 3))
    x = np.vstack((np.zeros(3), dx.cumsum(0)))
    y = np.vstack((np.zeros(3), dy.cumsum(0)))
    _, grad = Solver(lib, dx @ dy.T, 3, "polynomial", 4).evaluate()
    analytic = path_gradients(grad, dx, dy)
    numerical = []
    for target in (x, y):
        for idx in np.ndindex(target.shape):
            xc, yc = x.astype(complex), y.astype(complex)
            (xc if target is x else yc)[idx] += 1e-30j
            numerical.append(np.imag(numpy_reference(np.diff(xc, axis=0) @ np.diff(yc, axis=0).T, 3, 4)) / 1e-30)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-12, atol=5e-13)
    return {"discrete_value_and_gradient_cases": checks,
            "max_value_absolute_error": worst_value,
            "max_gram_gradient_absolute_error": worst_grad,
            "single_cell_exact_max_absolute_error": max(exact_errors),
            "path_gradient_complex_step_relative_error": relative_norm(analytic, np.array(numerical)),
            "public_c_entry_point_checks": 4}


def make_paths(family, segments, seed):
    rng = np.random.default_rng(seed)
    d = 3
    if family.startswith("brownian"):
        scale = .7 / math.sqrt(segments * d)
        dx = rng.normal(scale=scale, size=(segments, d))
        dy = rng.normal(scale=scale, size=(segments, d))
    else:
        # Frequent reversals with appreciable cell coefficients; intentionally
        # harder for low polynomial order than finely sampled Brownian paths.
        direction = np.array([1.0, .5, -.3])
        signs = (-1.0)**np.arange(segments)
        dx = .45 * signs[:, None] * direction + rng.normal(scale=.08 / math.sqrt(segments), size=(segments, d))
        dy = .4 * signs[:, None] * np.array([.8, -.3, .5]) + rng.normal(scale=.1 / math.sqrt(segments), size=(segments, d))
    for inc in (dx, dy):
        max_norm = np.max(np.linalg.norm(inc.cumsum(0), axis=1))
        if max_norm > 1.0:
            inc /= max_norm
    if family == "brownian_self":
        dy = dx.copy()
    return dx, dy


def benchmark(solver, mode, samples, minimum_ms):
    elapsed = solver(mode)
    reps = min(100000, max(1, math.ceil(minimum_ms / max(elapsed, .00001))))
    solver(mode, reps)
    times = [solver(mode, reps) for _ in range(samples)]
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def memory_bytes(method, segments, depth, parameter, backward):
    """Peak numeric workspace payload for warmed calls, excluding inputs/outputs.

    Includes retained forward buffers and the FD reverse pass's temporary
    build-stack workspace. Does not claim to measure process RSS.
    """
    if depth <= 1:
        return 0
    if method == "polynomial":
        width = segments * parameter
        count = (depth + 3 if backward else 3) * width**2 + width
    else:
        width = segments * 2**parameter + 1
        count = (depth + 7) * width**2 + 2 * width if backward else 2 * width**2 + width
    return 8 * count


def reference_case(lib, gram, depth, dx, dy):
    previous = None
    for q in (12, 16, 20, 24):
        current = Solver(lib, gram, depth, "polynomial", q).evaluate()
        if not np.isfinite(current[0]) or not np.isfinite(current[1]).all():
            raise RuntimeError("Reference overflow: select a smaller path scale")
        if previous is not None:
            value_gap = abs(current[0] - previous[0]) / abs(current[0])
            grad_gap = relative_norm(current[1], previous[1])
            path_gap = relative_norm(path_gradients(current[1], dx, dy), path_gradients(previous[1], dx, dy))
            if max(value_gap, grad_gap, path_gap) < 1e-10:
                return current, q, value_gap, grad_gap, path_gap
        previous = current
    raise RuntimeError("Reference did not converge to the required tolerance")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matched_accuracy(rows):
    matches = []
    case_ids = sorted({row["case"] for row in rows})
    for case in case_ids:
        case_rows = [row for row in rows if row["case"] == case]
        for tolerance in (1e-4, 1e-6, 1e-8):
            for metric, error_key, time_key, memory_key in (
                    ("forward", "kernel_relative_error", "forward_ms", "forward_workspace_bytes"),
                    ("forward_backward", "joint_error", "forward_backward_ms", "forward_backward_workspace_bytes")):
                best = {}
                for method in ("finite_difference", "polynomial"):
                    eligible = [r for r in case_rows if r["method"] == method and r[error_key] <= tolerance]
                    best[method] = min(eligible, key=lambda r: r[time_key]) if eligible else None
                fd, poly = best["finite_difference"], best["polynomial"]
                matches.append({"case": case, "family": case_rows[0]["family"],
                                "segments": case_rows[0]["segments"], "depth": case_rows[0]["depth"],
                                "metric": metric, "tolerance": tolerance,
                                "fd_parameter": fd["parameter"] if fd else "",
                                "polynomial_nodes": poly["parameter"] if poly else "",
                                "fd_ms": fd[time_key] if fd else "",
                                "polynomial_ms": poly[time_key] if poly else "",
                                "speedup": fd[time_key] / poly[time_key] if fd and poly else "",
                                "workspace_ratio_fd_over_poly": fd[memory_key] / poly[memory_key] if fd and poly else ""})
    return matches


def plot_results(rows, matches, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = sorted({r["family"] for r in rows})
    lengths = sorted({r["segments"] for r in rows})
    middle = lengths[len(lengths) // 2]
    seed = min(r["seed"] for r in rows)
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4.4), squeeze=False)
    for ax, family in zip(axes[0], families):
        for method, color, marker in (("finite_difference", "#b85042", "o"), ("polynomial", "#2166ac", "s")):
            subset = sorted([r for r in rows if r["family"] == family and r["segments"] == middle
                             and r["seed"] == seed and r["depth"] == 3 and r["method"] == method], key=lambda r: r["parameter"])
            ax.loglog([r["forward_backward_ms"] for r in subset], [max(r["joint_error"], 1e-15) for r in subset],
                      marker=marker, color=color, label=method.replace("_", " "))
            for r in subset:
                ax.annotate(str(r["parameter"]), (r["forward_backward_ms"], max(r["joint_error"], 1e-15)), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set_title(family.replace("_", " "))
        ax.set_xlabel("Forward + backward (ms)")
        ax.set_ylabel("Max relative error: kernel, both path gradients")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Branched kernel: {middle} segments, depth 3, float64, one CPU core\nLabels: FD dyadic order / polynomial nodes per coordinate")
    fig.tight_layout()
    fig.savefig(output / "accuracy_runtime.png", dpi=180)
    fig.savefig(output / "accuracy_runtime.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric in zip(axes, ("forward", "forward_backward")):
        for family, color in zip(families, ("#2166ac", "#1b9e77", "#b85042")):
            ys = []
            for tolerance in (1e-4, 1e-6, 1e-8):
                speeds = [r["speedup"] for r in matches if r["metric"] == metric and r["family"] == family
                          and r["tolerance"] == tolerance and r["speedup"] != ""]
                ys.append(float(np.median(speeds)) if speeds else np.nan)
            ax.plot(["1e-4", "1e-6", "1e-8"], ys, "o-", color=color, label=family.replace("_", " "))
        ax.axhline(1, color="black", linestyle="--", linewidth=.8)
        ax.set_yscale("log")
        ax.set_xlabel("Relative error tolerance")
        ax.set_ylabel("FD time / polynomial time (median)")
        ax.set_title(metric.replace("_", " + "))
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Matched-accuracy speedup; only cases where both methods reach tolerance")
    fig.tight_layout()
    fig.savefig(output / "matched_accuracy.png", dpi=180)
    fig.savefig(output / "matched_accuracy.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--minimum-ms", type=float, default=2.0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    binary, command = build(output, args.compiler)
    lib = load(binary)
    validation = validate(lib)
    print("Validation:", json.dumps(validation), flush=True)
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if args.validate_only:
        return
    lengths = (8, 32) if args.quick else (8, 32, 128)
    depths = (2, 3) if args.quick else (2, 3, 5)
    seeds = (17,) if args.quick else (17, 43, 91)
    families = ("brownian_cross", "brownian_self", "oscillatory_cross")
    metadata = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (HERE / "prototype.cpp", HERE / "compare.py", ROOT / "siglib/cpsig/cp_branched_sig_kernel.cpp")},
        "build_command": command, "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
        "cpu": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        "logical_cpu": cpu, "platform": platform.platform(), "numpy": np.__version__, "dtype": "float64",
        "lengths_in_segments": lengths, "depths": depths, "seeds": seeds, "families": families,
        "polynomial_nodes": [1, 2, 3, 4, 6, 8], "fd_dyadic_orders": list(range(7)),
        "max_fd_refined_segments_per_axis": 2048, "timing_samples": args.samples, "minimum_sample_ms": args.minimum_ms,
        "timing_scope": "Native scalar pair; Gram construction and Gram-to-path chain excluded for both methods. Both backward passes recompute forward state. Warm workspace, cached quadrature matrices. Median over samples.",
        "memory_scope": "Calculated peak numeric workspace payload for warmed calls; includes FD temporary stack-building buffers, excludes inputs, outputs, quadrature tables, allocator overhead and process RSS.",
        "reference": "Independent order checks q=12 and q=16 (then 20/24 if needed), requiring <1e-10 relative change in kernel, Gram gradient, and both-path gradient norm. Independent FD Richardson checks included.",
    }
    rows, refs = [], []
    started = time.monotonic()
    total_cases = len(lengths) * len(depths) * len(seeds) * len(families)
    for family in families:
        for segments in lengths:
            for seed in seeds:
                dx, dy = make_paths(family, segments, seed)
                gram = np.ascontiguousarray(dx @ dy.T)
                for depth in depths:
                    case = f"{family}_n{segments}_d{depth}_s{seed}"
                    (ref, grad_ref), ref_q, value_gap, grad_gap, path_gap = reference_case(lib, gram, depth, dx, dy)
                    path_ref = path_gradients(grad_ref, dx, dy)
                    max_refinement = min(6, int(math.log2(2048 // segments)))
                    # Independent FD extrapolation checks the collocation reference.
                    # Three levels also estimate the FD truncation error remaining.
                    fd_refs = [Solver(lib, gram, depth, "finite_difference", r).evaluate()
                               for r in range(max(0, max_refinement - 2), max_refinement + 1)]
                    rich = (4 * fd_refs[-1][0] - fd_refs[-2][0]) / 3
                    rich_grad = (4 * fd_refs[-1][1] - fd_refs[-2][1]) / 3
                    rich_prev = (4 * fd_refs[-2][0] - fd_refs[-3][0]) / 3
                    refs.append({"case": case, "value": ref, "q": ref_q,
                                 "polynomial_value_gap": value_gap, "polynomial_gram_gradient_gap": grad_gap,
                                 "polynomial_path_gradient_gap": path_gap,
                                 "fd_finest_dyadic_order": max_refinement,
                                 "fd_finest_relative_error": abs(fd_refs[-1][0] - ref) / abs(ref),
                                 "fd_richardson_relative_error": abs(rich - ref) / abs(ref),
                                 "fd_richardson_previous_relative_error": abs(rich_prev - ref) / abs(ref),
                                 "fd_richardson_gram_gradient_error": relative_norm(rich_grad, grad_ref)})
                    settings = [("finite_difference", r) for r in range(max_refinement + 1)]
                    settings += [("polynomial", q) for q in (1, 2, 3, 4, 6, 8)]
                    # Alternate method order across cases to reduce ordering bias.
                    if len(refs) % 2 == 0:
                        settings.reverse()
                    for method, parameter in settings:
                        solver = Solver(lib, gram, depth, method, parameter)
                        lib.bsk_reset()
                        forward_ms, forward_min, forward_max = benchmark(solver, 0, args.samples, args.minimum_ms)
                        fb_ms, fb_min, fb_max = benchmark(solver, 2, args.samples, args.minimum_ms)
                        value, grad = float(solver.out[0]), solver.grad.copy()
                        kernel_error = abs(value - ref) / abs(ref)
                        path_error = relative_norm(path_gradients(grad, dx, dy), path_ref)
                        rows.append({"case": case, "family": family, "segments": segments, "depth": depth, "seed": seed,
                                     "method": method, "parameter": parameter, "kernel": value,
                                     "kernel_relative_error": kernel_error, "gram_gradient_relative_error": relative_norm(grad, grad_ref),
                                     "path_gradient_relative_error": path_error, "joint_error": max(kernel_error, path_error),
                                     "forward_ms": forward_ms, "forward_min_ms": forward_min, "forward_max_ms": forward_max,
                                     "forward_backward_ms": fb_ms, "forward_backward_min_ms": fb_min, "forward_backward_max_ms": fb_max,
                                     "forward_workspace_bytes": memory_bytes(method, segments, depth, parameter, False),
                                     "forward_backward_workspace_bytes": memory_bytes(method, segments, depth, parameter, True)})
                    write_csv(output / "measurements.csv", rows)
                    write_csv(output / "references.csv", refs)
                    print(f"[{len(refs)}/{total_cases}] {case}: K={ref:.6g}, reference q={ref_q}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    matches = matched_accuracy(rows)
    write_csv(output / "matched_accuracy.csv", matches)
    plot_results(rows, matches, output)
    summary = []
    for metric in ("forward", "forward_backward"):
        for tol in (1e-4, 1e-6, 1e-8):
            selected = [r for r in matches if r["metric"] == metric and r["tolerance"] == tol]
            paired = [r for r in selected if r["speedup"] != ""]
            entry = {"metric": metric, "tolerance": tol, "cases": len(selected), "both_reached": len(paired),
                     "fd_reached": sum(r["fd_ms"] != "" for r in selected),
                     "polynomial_reached": sum(r["polynomial_ms"] != "" for r in selected),
                     "polynomial_faster": sum(r["speedup"] > 1 for r in paired),
                     "median_speedup": float(np.median([r["speedup"] for r in paired])) if paired else None,
                     "median_workspace_ratio_fd_over_poly": float(np.median([r["workspace_ratio_fd_over_poly"] for r in paired])) if paired else None}
            summary.append(entry)
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["summary"] = summary
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
