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

"""Measure shared traversal on initial and final loss-gradient queries."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lasso_features import write_json, process_peak_bytes
from benchmarks.benchmark_lasso_optimization import select_method


def query(paths, residuals, thresholds, degree, batch, jobs=8):
    from pysiglib._lasso_types import LassoOptions
    from pysiglib.load_siglib import CPSIG
    paths = np.ascontiguousarray(paths, dtype=np.float64)
    residuals = np.ascontiguousarray(residuals.T, dtype=np.float64)
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
    n, length, dimension = paths.shape
    targets = len(residuals)
    options = LassoOptions(n, length, dimension, degree, targets, 3000, 0, 1e-8, jobs, 1, 0, 0)
    scores = np.zeros((targets, 2))
    diagnostics = np.zeros(7)
    pointer = ctypes.POINTER(ctypes.c_double)
    function = CPSIG.sig_working_set_query_benchmark_d
    function.argtypes = (pointer, pointer, pointer, ctypes.c_uint32, ctypes.POINTER(LassoOptions), pointer, pointer)
    function.restype = ctypes.c_int
    before = process_peak_bytes()
    started = time.perf_counter()
    status = function(paths.ctypes.data_as(pointer), residuals.ctypes.data_as(pointer), thresholds.ctypes.data_as(pointer),
                      batch, ctypes.byref(options), scores.ctypes.data_as(pointer), diagnostics.ctypes.data_as(pointer))
    seconds = time.perf_counter()-started
    if status:
        raise RuntimeError(f'Shared query failed with status {status}')
    result = dict(zip(('prepare', 'search', 'native_bytes', 'constructed', 'correlations', 'probes', 'pruned'), diagnostics.tolist()))
    result.update(seconds=seconds, process_peak_bytes=process_peak_bytes(), process_before_bytes=before, batch=batch,
                  scores=scores.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('output/optimization_round2/final_fresh'))
    parser.add_argument('--output', type=Path, default=Path('output/optimization_round3/queries.json'))
    parser.add_argument('--cases', nargs='+', default=['arabic4', 'speech3', 'insect3'])
    parser.add_argument('--batches', nargs='+', type=int, default=[0, 1, 2, 4, 10])
    parser.add_argument('--method', default='optimized')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    args = parser.parse_args()
    records = []
    native = hashlib.sha256(Path('pysiglib/cpsig.dll').read_bytes()).hexdigest()
    with threadpool_limits(limits=1):
        for case in args.cases:
            folder = args.data/case
            with np.load(folder/'data.npz') as data:
                paths, y, alpha = data['paths'], data['y'], data['alpha']
            with np.load(folder/'optimized_model.npz') as model:
                final = y-model['training']
            spec = json.loads((folder/'spec.json').read_text())
            select_method(args.method, spec['degree'])
            for phase, residual in [('initial', y-y.mean(0)), ('final', final)]:
                threshold = alpha+.9*np.minimum(1e-8, 1e-4*alpha)
                reference = None
                for repeat in range(-args.warmups, args.repeats):
                    order = args.batches[repeat % len(args.batches):]+args.batches[:repeat % len(args.batches)]
                    for batch in order:
                        row = query(paths, residual, threshold, spec['degree'], batch, args.jobs)
                        row.update(case=case, phase=phase, repeat=repeat, method=args.method, native_sha256=native,
                                   memory_note='Process peak is cumulative within this query suite.')
                        scores = np.asarray(row['scores'])[:, 0]
                        if reference is None: reference = scores
                        np.testing.assert_allclose(scores, reference, atol=1e-11, rtol=1e-9)
                        records.append(row)
                        write_json(args.output, records)
                        print(case, phase, repeat, batch, round(row['seconds'], 4), round(row['native_bytes']/2**20, 2),
                              int(row['constructed']), flush=True)


if __name__ == '__main__':
    main()
