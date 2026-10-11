"""Extra independent FD refinement on the hardest reference in each family."""

import argparse
import csv
from pathlib import Path

import numpy as np

from compare import Solver, load, make_paths, path_gradients, relative_norm, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    lib = load(results / "libbranched_comparison.so")
    references = list(csv.DictReader((results / "references.csv").open()))
    records = []
    for family in ("brownian_cross", "brownian_self", "oscillatory_cross"):
        selected = max((r for r in references if r["case"].startswith(family)),
                       key=lambda r: float(r["fd_richardson_gram_gradient_error"]))
        case = selected["case"]
        _, n, d, s = case.rsplit("_", 3)
        segments, depth, seed = int(n[1:]), int(d[1:]), int(s[1:])
        dx, dy = make_paths(family, segments, seed)
        gram = dx @ dy.T
        value_ref, grad_ref = Solver(lib, gram, depth, "polynomial", 16).evaluate()
        path_ref = path_gradients(grad_ref, dx, dy)
        r0 = int(selected["fd_finest_dyadic_order"])
        previous = Solver(lib, gram, depth, "finite_difference", r0 - 1).evaluate()
        for r in range(r0, r0 + 3):
            current = Solver(lib, gram, depth, "finite_difference", r).evaluate()
            rich_value = (4 * current[0] - previous[0]) / 3
            rich_grad = (4 * current[1] - previous[1]) / 3
            records.append({"case": case, "dyadic_order": r,
                            "refined_segments": segments * 2**r,
                            "kernel_relative_error": abs(current[0] - value_ref) / abs(value_ref),
                            "path_gradient_relative_error": relative_norm(path_gradients(current[1], dx, dy), path_ref),
                            "richardson_kernel_relative_error": abs(rich_value - value_ref) / abs(value_ref),
                            "richardson_gram_gradient_relative_error": relative_norm(rich_grad, grad_ref),
                            "richardson_path_gradient_relative_error": relative_norm(path_gradients(rich_grad, dx, dy), path_ref)})
            print(records[-1], flush=True)
            write_csv(results / "extra_reference_checks.csv", records)
            previous = current
        assert records[-1]["richardson_kernel_relative_error"] < 1e-9
        assert records[-1]["richardson_path_gradient_relative_error"] < 1e-8


if __name__ == "__main__":
    main()
