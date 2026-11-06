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

"""Screen splits and confirm choices on independent paths with paired timings."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plugins/cuda'))
import pysiglib


def configure(candidate):
    if candidate is None:
        for name in ('PYSIGLIB_PS_DISPATCH', 'PYSIGLIB_PS_SPLIT_A', 'PYSIGLIB_PS_ENDPOINT_BLOCK'):
            os.environ.pop(name, None)
        return
    os.environ['PYSIGLIB_PS_DISPATCH'] = '0'
    os.environ['PYSIGLIB_PS_SPLIT_A'] = str(candidate[0])
    os.environ['PYSIGLIB_PS_ENDPOINT_BLOCK'] = str(candidate[1])


def operations(shape, device, dtype, seed, jobs=None):
    batch, n, d, m = shape
    rng = np.random.default_rng(seed)
    array = np.concatenate([np.zeros((batch, 1, d)),
                            np.cumsum(rng.normal(size=(batch, n, d)), axis=1) / np.sqrt(n)], axis=1).astype(dtype)
    path = torch.as_tensor(array, device=device) if device == 'cuda' else array
    jobs = (20 if device == 'cpu' else 1) if jobs is None else jobs
    signature = pysiglib.sig(path, m, n_jobs=jobs)
    seeds = (rng.normal(size=signature.shape) / np.sqrt(signature.shape[-1])).astype(dtype)
    deriv = torch.as_tensor(seeds, device=device) if device == 'cuda' else seeds
    def forward():
        return pysiglib.sig(path, m, method='prefix_suffix_gemm', n_jobs=jobs)
    def backward():
        return pysiglib.sig_backprop(path, signature, deriv, m, method='prefix_suffix_gemm', n_jobs=jobs)
    return {'forward': forward, 'backward': backward}


def timing(run, candidate, gpu):
    configure(candidate)
    if gpu: torch.cuda.synchronize()
    started = time.perf_counter_ns()
    result = run()
    if gpu: torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / 1e6, result


def check(actual, expected, dtype):
    if isinstance(actual, torch.Tensor):
        actual, expected = actual.cpu().numpy(), expected.cpu().numpy()
    tolerance = 2e-4 if dtype == 'float32' else 2e-10
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def run_case(shape, device, dtype, samples):
    batch, n, d, m = shape
    gpu = device == 'cuda'
    runs = operations(shape, device, dtype, 719)
    candidates = [(0, 0)]
    for a in range(1, m + 1):
        scan = max(m // 2, a - 1, m - a)
        if 4 * (n + 1) * sum(d ** k for k in range(scan + 1)) * np.dtype(dtype).itemsize <= (256 << 20):
            candidates.append((a, 0))
    if d ** (m - 1) <= 32768:
        candidates.extend((0, b) for b in sorted({min(n, b) for b in (16, 64, 256, 1024)}))
    result = {'shape': shape, 'dtype': dtype, 'screen': {}, 'confirmed': {}}
    for phase, run in runs.items():
        _, expected = timing(run, (0, 0), gpu)
        timings = []
        for candidate in candidates:
            if phase == 'backward' and candidate[1]: continue
            try:
                first, actual = timing(run, candidate, gpu)
                check(actual, expected, dtype)
                if first > 1500:
                    timings.append({'candidate': candidate, 'excluded': 'call exceeds 1.5 seconds'})
                    continue
                values = [timing(run, candidate, gpu)[0] for _ in range(3)]
                timings.append({'candidate': candidate, 'samples_ms': values, 'median_ms': float(np.median(values))})
            except AssertionError:
                raise
            except Exception as error:
                timings.append({'candidate': candidate, 'excluded': str(error)})
        result['screen'][phase] = timings
    del runs
    runs = operations(shape, device, dtype, 1931)
    for phase, run in runs.items():
        screened = [entry for entry in result['screen'][phase] if 'median_ms' in entry]
        if not screened: continue
        winner = min(screened, key=lambda entry: entry['median_ms'])['candidate']
        if winner == (0, 0): continue
        baseline = next((entry['median_ms'] for entry in screened if entry['candidate'] == (0, 0)), None)
        if baseline is None or min(entry['median_ms'] for entry in screened) > baseline * .95: continue
        _, expected = timing(run, (0, 0), gpu)
        _, actual = timing(run, winner, gpu)
        check(actual, expected, dtype)
        pairs = []
        for i in range(samples):
            order = (winner, (0, 0)) if i % 2 else ((0, 0), winner)
            times = {tuple(candidate): timing(run, candidate, gpu)[0] for candidate in order}
            pairs.append([times[(0, 0)], times[tuple(winner)]])
        pairs = np.asarray(pairs)
        rng = np.random.default_rng(51)
        draw = rng.integers(0, len(pairs), (5000, len(pairs)))
        bootstrap = np.median(pairs[draw, 0], axis=1) / np.median(pairs[draw, 1], axis=1)
        interval = np.quantile(bootstrap, [.025, .975])
        result['confirmed'][phase] = dict(candidate=winner, pairs_ms=pairs.tolist(),
                                          speedup=float(np.median(pairs[:, 0]) / np.median(pairs[:, 1])),
                                          ci95=interval.tolist(), accepted=bool(interval[0] > 1.05))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--samples', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pairs = ((2, 6), (2, 8), (2, 12), (3, 5), (3, 8), (4, 4), (4, 6),
             (6, 3), (6, 4), (8, 3), (8, 4), (10, 3), (12, 3), (16, 3), (24, 3), (32, 3))
    report = json.loads(args.output.read_text()) if args.output.exists() else {
        'device': args.device, 'dtype': args.dtype, 'scope': 'complete Python API', 'cases': []}
    recorded = {tuple(case['shape']) for case in report['cases']}
    report['prior_seconds'] = report.get('elapsed_seconds', 0)
    report['complete'] = False
    started = time.perf_counter()
    for n in (16, 256, 4096):
        for d, m in pairs:
            shape = (32, n, d, m)
            if shape in recorded: continue
            configure((0, 0))
            result = run_case(shape, args.device, args.dtype, args.samples)
            report['cases'].append(result)
            report['elapsed_seconds'] = report.get('prior_seconds', 0) + time.perf_counter() - started
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='ascii', newline='\r\n')
            print(shape, result['confirmed'], flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2), encoding='ascii', newline='\r\n')


if __name__ == '__main__':
    main()
