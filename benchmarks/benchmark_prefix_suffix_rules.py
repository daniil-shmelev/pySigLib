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

"""Check simple dispatch rules on shapes outside the original calibration grid."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_prefix_suffix_dispatch import operations, timing, check, configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--dtype', choices=('float32', 'float64'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--suite', choices=('validation', 'boundary'), default='validation')
    parser.add_argument('--jobs', type=int, nargs='+')
    args = parser.parse_args()
    gpu = args.device == 'cuda'
    pairs = ((2, 3), (12, 3), (20, 3), (24, 3), (32, 3), (48, 3), (2, 7), (2, 9), (3, 6), (4, 5)) if gpu else ((8, 3), (12, 3), (24, 3), (32, 3), (16, 4))
    shapes = ((1, 513), (16, 513), (96, 513), (1, 2049), (16, 2049), (96, 2049))
    if args.suite == 'boundary':
        pairs, shapes = ((8, 3), (14, 3), (18, 3)), ((5, 641), (40, 3073))
    report = json.loads(args.output.read_text()) if args.output.exists() else {
        'device': args.device, 'dtype': args.dtype, 'cases': []}
    report['complete'] = False
    for batch, n in shapes:
        for d, m in pairs:
            shape = [batch, n, d, m]
            for jobs in (args.jobs or ((1,) if gpu else ((1, 20) if batch <= 8 else (20,)))):
                if any(c['shape'] == shape and c['jobs'] == jobs for c in report['cases']): continue
                configure((0, 0))
                runs = operations(shape, args.device, args.dtype, 9431, jobs)
                case = dict(shape=shape, jobs=jobs, phases={})
                for phase in (('forward', 'backward') if gpu else ('forward',)):
                    run = runs[phase]
                    _, expected = timing(run, (0, 0), gpu)
                    _, actual = timing(run, None, gpu)
                    check(actual, expected, args.dtype)
                    pairs_ms = []
                    for i in range(args.samples):
                        order = (None, (0, 0)) if i % 2 else ((0, 0), None)
                        times = {candidate: timing(run, candidate, gpu)[0] for candidate in order}
                        pairs_ms.append([times[(0, 0)], times[None]])
                    values = np.asarray(pairs_ms)
                    draw = np.random.default_rng(91).integers(0, len(values), (5000, len(values)))
                    interval = np.quantile(np.median(values[draw, 0], axis=1) / np.median(values[draw, 1], axis=1), [.025, .975])
                    case['phases'][phase] = dict(pairs_ms=pairs_ms, ci95=interval.tolist(),
                        speedup=float(np.median(values[:, 0]) / np.median(values[:, 1])))
                report['cases'].append(case)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2), encoding='ascii', newline='\r\n')
                print(shape, jobs, {k: round(v['speedup'], 3) for k, v in case['phases'].items()}, flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2), encoding='ascii', newline='\r\n')


if __name__ == '__main__':
    main()
