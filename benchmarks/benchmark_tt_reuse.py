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

"""Compare suffix envelopes and reuse of working-set search candidates."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lasso_features import write_json
from benchmarks.benchmark_tt_bound import worker

METHODS = {'tt': (1, 0), 'block2': (3, 0), 'suffix': (4, 0),
           'pool64': (1, 64), 'pool256': (1, 256), 'suffix_pool64': (4, 64), 'suffix_pool256': (4, 256)}


def select_method(name):
    from pysiglib.load_siglib import CPSIG
    bound, capacity = METHODS[name]
    for symbol, value in (('sig_working_set_experimental_bound', bound), ('sig_working_set_experimental_pool', capacity)):
        function = getattr(CPSIG, symbol)
        function.argtypes, function.restype = (ctypes.c_uint32,), ctypes.c_int
        if function(value):
            raise RuntimeError(f'Experimental native option rejected: {symbol}')


def summarize(output):
    rows = []
    for folder in sorted(output.iterdir()):
        if not (folder / 'audit.json').exists():
            continue
        records = json.loads((folder / 'timings.json').read_text())
        audits = json.loads((folder / 'audit.json').read_text())
        for mode in audits:
            values = [r for r in records if r['mode'] == mode and r['repeat'] >= 0]
            times = [r['fit_seconds'] for r in values]
            row = dict(case=folder.name, mode=mode, median_seconds=float(np.median(times)),
                       iqr_seconds=float(np.subtract(*np.percentile(times, [75, 25]))),
                       native_peak_mib=float(np.median([r['stats']['peak_bytes'] for r in values])) / 2**20,
                       support=values[0]['nonzeros'], union_support=values[0]['union_support'], audit=audits[mode])
            for key in ('preparation_seconds', 'search_seconds', 'solve_seconds', 'certification_seconds',
                        'constructed', 'block_queries', 'terminal_correlations', 'search_passes', 'activations', 'drops'):
                row[key] = float(np.median([r['stats'][key] for r in values]))
            rows.append(row)
            print(f"{row['case']:12} {mode:15} {row['median_seconds']:9.4f} ({row['iqr_seconds']:.4f}) {row['native_peak_mib']:9.2f} MiB {row['search_passes']:5.0f} searches")
    write_json(output / 'summary.json', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('output/tt_bound'))
    parser.add_argument('--output', type=Path, default=Path('output/tt_reuse'))
    parser.add_argument('--cases', nargs='+', default=['baseline', 'degree', 'arabic3', 'arabic4'])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.worker:
            spec = json.loads((args.worker / 'spec.json').read_text())
            worker(args.worker, tuple(spec['methods']), select_method, 'tt')
            return
        if args.summary:
            summarize(args.output)
            return
        if 'tt' not in args.methods:
            raise ValueError('The TT control is required')
        for case in args.cases:
            folder = args.output / case
            folder.mkdir(parents=True, exist_ok=True)
            spec = json.loads((args.data / case / 'spec.json').read_text())
            spec.update(data_folder=str((args.data / case).resolve()), methods=args.methods, repeats=args.repeats, jobs=args.jobs,
                        native_sha256=hashlib.sha256(Path('pysiglib/cpsig.dll').read_bytes()).hexdigest())
            write_json(folder / 'spec.json', spec)
            subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_tt_reuse', '--worker', str(folder.resolve())], check=True, timeout=3600)
        summarize(args.output)


if __name__ == '__main__':
    main()
