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

"""Measure branch-and-bound candidate, traversal, and workspace policies."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lasso_features import write_json
from benchmarks.benchmark_tt_bound import worker
from benchmarks.benchmark_tt_strategy import diagnostics, summarize

DEFAULTS = dict(bound=1, pool=0, strategy=0, batch=32, rank=0, order=0, parts=0, search=4, cache=16 << 20, refresh=0.)
METHODS = {
    'bernstein': dict(bound=0),
    'tt': {}, 'pool64': dict(pool=64), 'pool256': dict(pool=256),
    'rank16': dict(pool=256, rank=16), 'rank64': dict(pool=256, rank=64),
    'rank16_b16': dict(pool=256, rank=16, batch=16),
    'rank16_b8': dict(pool=256, rank=16, batch=8),
    'rank16_b4': dict(pool=256, rank=16, batch=4),
    'rank16_b1': dict(pool=256, rank=16, batch=1),
    'rank16_order': dict(pool=256, rank=16, order=1),
    'shared': dict(order=2), 'pool_shared': dict(pool=256, order=2),
    'rank16_shared': dict(pool=256, rank=16, order=2),
    'rank16_shared_b8': dict(pool=256, rank=16, order=2, batch=8),
    'rank16_shared_b4': dict(pool=256, rank=16, order=2, batch=4),
    'rank16_shared_order': dict(pool=256, rank=16, order=3),
    'rank16_parts8': dict(pool=256, rank=16, parts=8),
    'rank16_parts4': dict(pool=256, rank=16, parts=4),
    'rank16_cache4': dict(pool=256, rank=16, cache=4 << 20),
    'rank16_cache0': dict(pool=256, rank=16, cache=0),
    'rank16_adaptive': dict(pool=256, rank=16, strategy=2),
    'rank16_saved': dict(pool=256, rank=16, strategy=1),
    'rank16_both': dict(pool=256, rank=16, strategy=3),
    'compact': dict(order=4), 'pool_compact': dict(pool=256, order=4),
    'rank16_compact': dict(pool=256, rank=16, order=4),
    'rank16_compact_adaptive': dict(pool=256, rank=16, order=4, strategy=2),
    'rank16_compact_saved': dict(pool=256, rank=16, order=4, strategy=1),
    'rank16_compact_both': dict(pool=256, rank=16, order=4, strategy=3),
    'rank16_b8_adaptive': dict(pool=256, rank=16, batch=8, strategy=2),
    'rank16_b4_adaptive': dict(pool=256, rank=16, batch=4, strategy=2),
    'rank16_b4_compact': dict(pool=256, rank=16, batch=4, order=4),
    'rank16_b4_compact_adaptive': dict(pool=256, rank=16, batch=4, order=4, strategy=2),
    'rank16_parts8_compact': dict(pool=256, rank=16, parts=8, order=4),
    'rank16_shared_compact': dict(pool=256, rank=16, order=6),
    'rank16_shared_compact_adaptive': dict(pool=256, rank=16, order=6, strategy=2),
    'compact_refresh25': dict(pool=256, rank=16, order=4, refresh=.25),
    'compact_refresh50': dict(pool=256, rank=16, order=4, refresh=.5),
    'compact_adaptive_refresh25': dict(pool=256, rank=16, order=4, strategy=2, refresh=.25),
    'compact_adaptive_refresh50': dict(pool=256, rank=16, order=4, strategy=2, refresh=.5),
    'compact_b4_adaptive_refresh25': dict(pool=256, rank=16, order=4, strategy=2, batch=4, refresh=.25),
    'b4_pool64': dict(pool=64, rank=16, batch=4),
    'b4_exactpool': dict(pool=256, batch=4),
    'b4_rank1': dict(pool=256, rank=1, batch=4),
    'b4_rank4': dict(pool=256, rank=4, batch=4),
    'b4_rank32': dict(pool=256, rank=32, batch=4),
    'b4_cache4': dict(pool=256, rank=16, batch=4, cache=4 << 20),
    'b4_cache0': dict(pool=256, rank=16, batch=4, cache=0),
    'b4_parts8': dict(pool=256, rank=16, batch=4, parts=8),
    'b4_parts4': dict(pool=256, rank=16, batch=4, parts=4),
    'b4_plain': dict(pool=256, rank=16, batch=4, search=0),
    'b4_grouped': dict(pool=256, rank=16, batch=4, search=1),
    'b4_signed': dict(pool=256, rank=16, batch=4, search=2),
    'b4_grouped_signed': dict(pool=256, rank=16, batch=4, search=3),
    'b4_bernstein': dict(pool=256, rank=16, batch=4, bound=0),
    'b4_combined': dict(pool=256, rank=16, batch=4, bound=2),
    'b4_block2': dict(pool=256, rank=16, batch=4, bound=3),
    'b4_suffix': dict(pool=256, rank=16, batch=4, bound=4),
    'compact_cache0': dict(pool=256, rank=16, batch=4, order=4, cache=0),
    'compact_cache0_adaptive': dict(pool=256, rank=16, batch=4, order=4, cache=0, strategy=2),
    'compact_b1_cache0': dict(pool=256, rank=16, batch=1, order=4, cache=0),
}


def select_method(name):
    from pysiglib.load_siglib import CPSIG
    config = DEFAULTS | METHODS[name]
    for symbol, key in (('sig_working_set_experimental_bound', 'bound'), ('sig_working_set_experimental_pool', 'pool'),
                        ('sig_working_set_experimental_strategy', 'strategy')):
        function = getattr(CPSIG, symbol)
        function.argtypes, function.restype = (ctypes.c_uint32,), ctypes.c_int
        if function(config[key]):
            raise RuntimeError(f'Experimental native option rejected: {symbol}')
    function = CPSIG.sig_working_set_experimental_search
    function.argtypes, function.restype = (ctypes.c_uint32,) * 5 + (ctypes.c_uint64,), ctypes.c_int
    if function(*(config[k] for k in ('batch', 'rank', 'order', 'parts', 'search', 'cache'))):
        raise RuntimeError('Experimental native search configuration rejected')
    function = CPSIG.sig_working_set_experimental_refresh
    function.argtypes, function.restype = (ctypes.c_double,), ctypes.c_int
    if function(config['refresh']):
        raise RuntimeError('Experimental native pool refresh rejected')


def run_worker(folder):
    spec = json.loads((folder / 'spec.json').read_text())
    source = Path(spec['data_folder'])
    design = np.load(source / 'design.npy', mmap_mode='r')
    lookup = {tuple(w): j for j, w in enumerate(json.loads((source / 'words.json').read_text()))}
    data = np.load(source / 'data.npz')
    y, alpha = data['y'], data['alpha']
    means = design.mean(0)
    tau = np.minimum(spec['tol'], 1e-4 * alpha)

    def check_fit(model):
        indices = [lookup[w] for w in model['words']]
        selected = design[:, indices]
        prediction = selected @ model['coef'] + model['intercept']
        residual = y - prediction
        corr = design.T @ residual / len(y) - means[:, None] * residual.mean(0)
        error = np.maximum(0, abs(corr).max(0) - alpha)
        for j, index in enumerate(indices):
            active = model['coef'][j] != 0
            error = np.maximum(error, np.where(active, abs(corr[index] - alpha * np.sign(model['coef'][j])), 0))
        objective = .5 * np.mean(residual**2, axis=0) + alpha * abs(model['coef']).sum(0)
        columns_error = float(np.max(abs(selected - model['signatures']))) if indices else 0.
        objective_error = float(np.max(abs(objective - model['objective'])))
        passed = bool(np.all(model['converged']) and np.all(error <= tau + 5e-12)
                      and np.max(abs(residual.mean(0))) <= spec['tol'] + 5e-12
                      and columns_error < 1e-10 and objective_error < 1e-10)
        return dict(passed=passed, kkt=error.tolist(), columns_error=columns_error, objective_error=objective_error)

    worker(folder, tuple(spec['methods']), select_method, 'tt', diagnostics, check_fit)


def memory_worker(folder, method, repeat):
    import pysiglib
    if sys.platform != 'win32':
        raise NotImplementedError('This process-memory measurement uses Windows PSAPI')
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32)] + [(name, ctypes.c_size_t) for name in
                    ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                     'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    current = ctypes.WinDLL('kernel32', use_last_error=True).GetCurrentProcess
    current.argtypes, current.restype = (), ctypes.c_void_p
    read = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    read.argtypes, read.restype = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32), ctypes.c_int
    spec = json.loads((folder / 'spec.json').read_text())
    data = np.load(Path(spec['data_folder']) / 'data.npz')
    paths, y, alpha = data['paths'], data['y'], data['alpha']
    select_method(method)
    before = Counters(cb=ctypes.sizeof(Counters))
    if not read(current(), ctypes.byref(before), before.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    started = time.perf_counter()
    model = pysiglib.sig_lasso(paths, y, spec['degree'], alpha, n_jobs=spec['jobs'], tol=spec['tol'], max_iter=spec['max_iter'])
    seconds = time.perf_counter() - started
    after = Counters(cb=ctypes.sizeof(Counters))
    if not read(current(), ctypes.byref(after), after.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    result = dict(method=method, repeat=repeat, fit_seconds=seconds, native_peak_bytes=model['stats']['peak_bytes'],
                  initial_rss=before.WorkingSetSize, initial_peak_rss=before.PeakWorkingSetSize,
                  process_peak_rss=after.PeakWorkingSetSize, process_peak_commit=after.PeakPagefileUsage,
                  converged=np.asarray(model['converged']).tolist())
    write_json(folder / f'memory_{method}_{repeat}.json', result)
    print(spec['case'], method, repeat, round(after.PeakWorkingSetSize / 2**20, 2), 'MiB process peak', flush=True)
    if not np.all(model['converged']):
        raise AssertionError('Fresh-process memory fit did not converge')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('output/tt_bound'))
    parser.add_argument('--output', type=Path, default=Path('output/bb_tradeoff'))
    parser.add_argument('--cases', nargs='+', default=['baseline', 'degree', 'arabic3', 'arabic4'])
    parser.add_argument('--methods', nargs='+', choices=METHODS,
                        default=['tt', 'pool256', 'rank16', 'rank64', 'rank16_b8', 'rank16_b4', 'rank16_b1',
                                 'rank16_order', 'shared', 'rank16_shared', 'rank16_shared_b8', 'rank16_parts8'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--memory-worker', type=Path)
    parser.add_argument('--memory-method', choices=METHODS)
    parser.add_argument('--memory-repeat', type=int, default=0)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.memory_worker:
            memory_worker(args.memory_worker, args.memory_method, args.memory_repeat)
            return
        if args.worker:
            run_worker(args.worker)
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
                        configurations={name: DEFAULTS | METHODS[name] for name in args.methods},
                        native_sha256=hashlib.sha256(Path('pysiglib/cpsig.dll').read_bytes()).hexdigest())
            write_json(folder / 'spec.json', spec)
            if args.memory:
                for repeat in range(args.repeats):
                    for method in args.methods:
                        subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_bb_tradeoff', '--memory-worker', str(folder.resolve()),
                                        '--memory-method', method, '--memory-repeat', str(repeat)], check=True, timeout=3600)
            else:
                subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_bb_tradeoff', '--worker', str(folder.resolve())],
                               check=True, timeout=3600)
        if not args.memory:
            summarize(args.output)


if __name__ == '__main__':
    main()
