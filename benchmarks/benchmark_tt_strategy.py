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

"""Test saved certificates, adaptive solves and permanent subtree screening."""

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
from benchmarks.benchmark_tt_reuse import summarize as base_summary

METHODS = {'tt': 0, 'certificates': 1, 'adaptive': 2, 'certificates_adaptive': 3,
           'screening': 4, 'certificates_screening': 5, 'adaptive_screening': 6, 'all': 7}
COUNTERS = ('retained_checks', 'retained_hits', 'norm_calls', 'permanent_hits', 'screened_entries',
            'screen_passes', 'inner_sweeps', 'adaptive_refinements', 'final_audits', 'audit_repairs', 'screen_seconds')


def select_method(name):
    from pysiglib.load_siglib import CPSIG
    for symbol, value in (('sig_working_set_experimental_bound', 1), ('sig_working_set_experimental_pool', 0),
                          ('sig_working_set_experimental_strategy', METHODS[name])):
        function = getattr(CPSIG, symbol)
        function.argtypes, function.restype = (ctypes.c_uint32,), ctypes.c_int
        if function(value):
            raise RuntimeError(f'Experimental native option rejected: {symbol}')


def diagnostics():
    from pysiglib.load_siglib import CPSIG
    output = np.empty(len(COUNTERS), dtype=np.float64)
    function = CPSIG.sig_working_set_experimental_stats
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes, function.restype = (pointer,), ctypes.c_int
    if function(output.ctypes.data_as(pointer)):
        raise RuntimeError('Cannot read experimental native diagnostics')
    return dict(zip(COUNTERS, output.tolist()))


def summarize(output, validation=None):
    base_summary(output)
    rows = json.loads((output / 'summary.json').read_text())
    for row in rows:
        records = json.loads((output / row['case'] / 'timings.json').read_text())
        records = [r for r in records if r['mode'] == row['mode'] and r['repeat'] >= 0]
        row['experiment'] = {k: float(np.median([r['experiment'][k] for r in records])) for k in COUNTERS}
        row['native_peak_iqr_mib'] = float(np.subtract(*np.percentile([r['stats']['peak_bytes'] / 2**20 for r in records], [75, 25])))
        if validation is not None:
            folder = validation / row['case']
            checks = json.loads((folder / 'timings.json').read_text())
            check = next(r for r in checks if r['mode'] == row['mode'] and r['repeat'] == 0)
            original = next(r for r in records if r['repeat'] == 0)
            keys = ('objective', 'gap', 'nonzeros', 'union_support', 'n_iter', 'converged')
            counts = ('constructed', 'evaluated', 'block_queries', 'terminal_correlations', 'search_passes')
            row['validation_results_match'] = all(original[k] == check[k] for k in keys)
            row['validation_search_counts_match'] = all(original['stats'][k] == check['stats'][k] for k in counts)
            row['validation_audit'] = json.loads((folder / 'audit.json').read_text())[row['mode']]
    write_json(output / 'summary.json', rows)
    if validation is not None and not all(r['validation_results_match'] and r['validation_search_counts_match']
                                          and r['validation_audit']['passed'] for r in rows):
        raise AssertionError('Final-source validation does not match the frozen experiment')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('output/tt_bound'))
    parser.add_argument('--output', type=Path, default=Path('output/tt_strategy'))
    parser.add_argument('--cases', nargs='+', default=['baseline', 'degree', 'arabic3', 'arabic4'])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.worker:
            spec = json.loads((args.worker / 'spec.json').read_text())
            worker(args.worker, tuple(spec['methods']), select_method, 'tt', diagnostics)
            return
        if args.summary:
            summarize(args.output, args.validation)
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
            subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_tt_strategy', '--worker', str(folder.resolve())],
                           check=True, timeout=3600)
        summarize(args.output)


if __name__ == '__main__':
    main()
