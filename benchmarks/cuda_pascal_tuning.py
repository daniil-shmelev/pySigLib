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

"""Compare launch sizes for the production CUDA Pascal kernels."""

import argparse
import ctypes as ct
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

import pysiglib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 512, 4096])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[4])
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 8, 12, 16, 20])
    parser.add_argument("--dtypes", nargs="+", choices=["float32", "float64"], default=["float32", "float64"])
    parser.add_argument("--segments", type=int, default=256)
    parser.add_argument("--threads", type=int, nargs="+", default=[32, 64, 128, 256])
    args = parser.parse_args()
    library = ct.CDLL(str(args.library.resolve()))
    records = []
    for dtype, suffix in [(torch.float32, "f"), (torch.float64, "d")]:
        if str(dtype).split(".")[-1] not in args.dtypes:
            continue
        function = getattr(library, "pascal_tune_" + suffix)
        function.argtypes = [ct.c_void_p] * 6 + [ct.c_uint64] * 5 + [ct.c_bool, ct.c_uint]
        function.restype = ct.c_int
        for batch, dimension in product(args.batches, args.dimensions):
            for degree in args.degrees:
                rng = np.random.default_rng(9321 + degree)
                points = rng.normal(0, .03, (batch, args.segments + 1, dimension)).cumsum(axis=1)
                path = torch.tensor(points, device="cuda", dtype=dtype)
                words = rng.integers(dimension, size=(8, degree), dtype=np.uint64)
                word_list = [tuple(int(c) for c in word) for word in words]
                order = np.arange(8, dtype=np.uint64)
                packed = np.concatenate((order * degree, order * degree, order, words.ravel()))
                metadata = torch.tensor(packed, device="cuda", dtype=torch.uint64)
                coefs = pysiglib.sig_coef(path, word_list, prefixes=True, method="pascal")
                seeds = torch.ones_like(coefs)
                gradient = pysiglib.sig_coef_backprop(path, word_list, coefs, seeds, method="pascal")
                retry = torch.zeros(1, dtype=torch.int32, device="cuda")
                for backward in [False, True]:
                    output = torch.empty_like(gradient if backward else coefs)
                    pointers = [x.data_ptr() for x in [path, output, coefs, seeds, metadata, retry]]
                    samples = {threads: [] for threads in args.threads}
                    for sample in range(args.samples):
                        order = args.threads if sample % 2 == 0 else list(reversed(args.threads))
                        for threads in order:
                            for _ in range(3):
                                assert function(*pointers, batch, 8, dimension, args.segments + 1, degree, backward, threads) == 0
                            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(10):
                                assert function(*pointers, batch, 8, dimension, args.segments + 1, degree, backward, threads) == 0
                            stop.record()
                            stop.synchronize()
                            samples[threads].append(start.elapsed_time(stop) / 10)
                            assert retry.item() == 0
                            tol = 3e-5 if dtype == torch.float32 else 2e-11
                            torch.testing.assert_close(output, gradient if backward else coefs, rtol=tol, atol=tol)
                    records.append(dict(dtype=str(dtype), batch=batch, dimension=dimension, degree=degree,
                                        segments=args.segments, backward=backward, samples=samples))
                    print(dtype, batch, dimension, degree, backward, {k: round(float(np.median(v)), 4) for k, v in samples.items()}, flush=True)
    result = dict(records=records, gpu=torch.cuda.get_device_name(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2), encoding="ascii", newline="\r\n")


if __name__ == "__main__":
    main()
