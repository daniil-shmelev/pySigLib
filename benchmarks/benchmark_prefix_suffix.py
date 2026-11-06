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

"""Compare public full-signature methods for one path shape."""

import argparse
from statistics import median
from time import perf_counter

import numpy as np
import torch

import pysiglib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--segments", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--split", type=int, nargs=2, metavar=("A", "B"), action="append", default=[])
    args = parser.parse_args()
    if min(args.batch, args.segments, args.dimension, args.degree, args.repeats) < 1:
        parser.error("shape parameters and repeats must be positive")
    if any(a < 1 or b < 0 or a + b != args.degree for a, b in args.split):
        parser.error("each split requires a >= 1, b >= 0 and a + b == degree")

    rng = np.random.default_rng(719)
    increments = rng.normal(size=(args.batch, args.segments, args.dimension)) / np.sqrt(args.segments)
    path = np.zeros((args.batch, args.segments + 1, args.dimension), dtype=args.dtype)
    np.cumsum(increments, axis=1, out=path[:, 1:])
    if args.device == "cuda":
        path = torch.as_tensor(path, device="cuda")
    signature = pysiglib.sig(path, args.degree, n_jobs=args.n_jobs)
    seeds = rng.normal(size=signature.shape).astype(args.dtype)
    if args.device == "cuda":
        seeds = torch.as_tensor(seeds, device="cuda")

    methods = [("horner", None, None), ("ps", "ps", None), ("auto", "auto", None)]
    methods.extend((f"ps({a},{b})", "ps", (a, b)) for a, b in args.split)
    print("pass      method        median ms   speedup vs Horner")
    for phase in ("forward", "backward"):
        samples = {name: [] for name, _, _ in methods}
        for i in range(args.repeats + 1):
            for name, method, split in methods if i % 2 else reversed(methods):
                options = dict(method=method, ps_split=split, n_jobs=args.n_jobs)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = perf_counter()
                if phase == "forward":
                    pysiglib.sig(path, args.degree, **options)
                else:
                    pysiglib.sig_backprop(path, signature, seeds, args.degree, **options)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                if i:
                    samples[name].append((perf_counter() - start) * 1000)
        baseline = median(samples["horner"])
        for name, values in samples.items():
            runtime = median(values)
            print(f"{phase:<9} {name:<11} {runtime:10.3f} {baseline / runtime:18.2f}")


if __name__ == "__main__":
    main()
