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

"""Compare terminal bounds, matrix blocks and accelerated restricted solves."""

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

from benchmarks.benchmark_bb_tradeoff import select_method as select_search
from benchmarks.benchmark_lasso_features import process_peak_bytes, write_json
from benchmarks.benchmark_tt_strategy import diagnostics

METHODS = {'balanced': (0, 0), 'optimized': (7, 7), 'anderson': (0, 1), 'gram': (0, 2), 'anderson_gram': (0, 3),
           'block8': (1, 0), 'block16': (2, 0), 'block32': (3, 0), 'block_late': (4, 0),
           'anderson_block16': (2, 1), 'anderson_block_late': (4, 1), 'grouped': (0, 0),
           'block16_grouped': (2, 0), 'anderson_block16_grouped': (2, 1),
           'eager_gram': (0, 6), 'anderson_eager_gram': (0, 7), 'cauchy': (5, 0), 'cauchy_grouped': (5, 0),
           'anderson_cauchy': (5, 1), 'anderson_gram_cauchy': (5, 3), 'segment': (6, 0), 'segment_grouped': (6, 0),
           'anderson_eager_cauchy': (5, 7), 'anderson_eager_cauchy_grouped': (5, 7),
           'anderson_eager_segment': (6, 7), 'anderson_eager_segment_grouped': (6, 7),
           'operator': (7, 0), 'operator_grouped': (7, 0),
           'anderson_eager_operator': (7, 7), 'anderson_eager_operator_grouped': (7, 7),
           'operator_segment': (8, 0), 'operator_segment_grouped': (8, 0),
           'anderson_eager_operator_segment': (8, 7), 'anderson_eager_operator_segment_grouped': (8, 7),
           'lazy_grouped': (9, 7), 'summary16_grouped': (7, 7), 'summary64_grouped': (7, 7),
           'gamma16_grouped': (7, 7), 'energy16_grouped': (7, 7), 'lazy_summary16_grouped': (9, 7),
           'norm_ascending_grouped': (7, 7), 'norm_descending_grouped': (7, 7), 'random_order_grouped': (7, 7),
           'sample_operator_grouped': (10, 7), 'sample_both_grouped': (11, 7),
           'scalar1_grouped': (7, 7), 'scalar16_grouped': (7, 7), 'sample_scalar_grouped': (11, 7),
           'sample_norm_descending_grouped': (11, 7), 'sample_signed_grouped': (12, 7), 'sample_letters_grouped': (13, 7),
           'optimized_next': (14, 7), 'sample_b1_grouped': (11, 7), 'sample_b2_grouped': (11, 7),
           'sample_fast_grouped': (14, 7), 'suffix_groups_grouped': (15, 7), 'ordered_grouped': (14, 15), 'cost_grouped': (14, 23)}
COUNTERS = ('anderson_attempts', 'anderson_accepted', 'gram_builds', 'gram_reused_entries', 'gram_seconds',
            'matrix_blocks', 'matrix_correlations', 'matrix_seconds', 'summary_hits', 'summary_misses',
            'summary_pruned', 'segment_refinements')


def select_method(name, degree=3):
    from pysiglib.load_siglib import CPSIG
    if name in ('optimized', 'optimized_next'):
        name = ('balanced' if degree == 1 else 'anderson_eager_gram' if degree == 2
                else 'sample_fast_grouped' if name == 'optimized_next' else 'anderson_eager_operator_grouped')
    select_search('compact_cache0')
    summaries = getattr(CPSIG, 'sig_working_set_experimental_summaries', None)
    flags = 4 if 'scalar' in name else 1 if name.startswith('gamma') else 2 if name.startswith('energy') else 3 if 'summary' in name else 0
    megabytes = 1 if name == 'scalar1_grouped' else 64 if '64' in name else 16
    if summaries is not None:
        summaries.argtypes, summaries.restype = (ctypes.c_uint32, ctypes.c_uint64), ctypes.c_int
    if (summaries is None and flags) or (summaries is not None and summaries(flags, megabytes << 20 if flags else 0)):
        raise RuntimeError('Experimental summary configuration rejected')
    function = CPSIG.sig_working_set_experimental_kernels
    function.argtypes, function.restype = (ctypes.c_uint32, ctypes.c_uint32), ctypes.c_int
    if function(*METHODS[name]):
        raise RuntimeError('Experimental kernel configuration rejected')
    if name == 'grouped' or name.endswith('_grouped'):
        batch = 1 if '_b1_' in name else 2 if '_b2_' in name else 4
        if CPSIG.sig_working_set_experimental_search(batch, 16, 4, 0, 3, 0):
            raise RuntimeError('Experimental grouped search rejected')


def optimization_stats():
    from pysiglib.load_siglib import CPSIG
    output = np.zeros(len(COUNTERS), dtype=np.float64)
    function = CPSIG.sig_working_set_optimization_stats
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes, function.restype = (pointer, ctypes.c_uint64), ctypes.c_int
    if function(output.ctypes.data_as(pointer), len(output)) and function(output.ctypes.data_as(pointer), 8):
        raise RuntimeError('Cannot read kernel diagnostics')
    return dict(zip(COUNTERS, output.tolist()))


def prepare(args, case, folder):
    if case in ('baseline', 'degree'):
        source = args.data / case
        spec = json.loads((source / 'spec.json').read_text())
        with np.load(source / 'data.npz') as data:
            arrays = {key: data[key] for key in data.files}
        spec['reference_folder'] = str(source.resolve())
    else:
        from benchmarks.rerun_audio_working_set import DATASETS, digest
        name, degree = case[:-1], int(case[-1])
        protocol = args.experiment_root / 'results/optimized_degree4_20261028/protocol.json'
        saved = json.loads(protocol.read_text())['datasets'][name]
        prepared = args.experiment_root / DATASETS[name] / 'results/normalized/integral_comparison/prepared.npz'
        assert digest(prepared) == saved['prepared_sha256']
        with np.load(prepared) as data:
            arrays = dict(paths=data['train'], y=data['response'], test=data['test'], labels=data['test_labels'],
                          alpha=np.asarray(saved['choices'][str(degree)]['alpha']))
        spec = dict(case=case, degree=degree, prepared_sha256=saved['prepared_sha256'],
                    protocol_sha256=digest(protocol))
    if args.response is not None:
        arrays['y'] = arrays['y'][:, args.response:args.response + 1]
        arrays['alpha'] = arrays['alpha'][args.response:args.response + 1]
        arrays.pop('labels', None)
    if args.loss != 'squared':
        if case not in ('baseline', 'degree'):
            raise ValueError('The paper protocol uses squared loss')
        from scipy.special import expit
        from scipy.optimize import brentq
        rng = np.random.default_rng(9417)
        y = arrays['y']
        if args.loss == 'logistic':
            y = rng.binomial(1, expit(y)).astype(float)
            residual = y-y.mean(0)
        elif args.loss == 'poisson':
            y = rng.poisson(np.exp(.5*np.clip(y, -4, 4))).astype(float)
            residual = y-y.mean(0)
        else:
            y = y + 3*rng.normal(size=y.shape)*(rng.random(y.shape) < .05)
            intercept = np.array([brentq(lambda b: np.clip(y[:, j]-b, -1, 1).mean(), y[:, j].min()-1, y[:, j].max()+1)
                                  for j in range(y.shape[1])])
            residual = np.clip(y-intercept, -1, 1)
        design = np.load(Path(spec['reference_folder'])/'design.npy', mmap_mode='r')
        arrays['y'] = y
        arrays['alpha'] = .03*np.max(abs(design.T@residual/len(y)), axis=0)
    np.savez(folder / 'data.npz', **arrays)
    spec.update(jobs=args.jobs, response=args.response, loss=args.loss, tol=1e-8, max_iter=3000,
                shape=list(arrays['paths'].shape), targets=arrays['y'].shape[1], methods=args.methods,
                repeats=args.repeats, warmups=args.warmups, reuse_process=args.reuse_process,
                native_sha256=hashlib.sha256(Path('pysiglib/cpsig.dll').read_bytes()).hexdigest())
    write_json(folder / 'spec.json', spec)


def worker(folder, method, repeat, fresh=True):
    import pysiglib
    from benchmarks.rerun_audio_working_set import selected_predictions, exhaustive_kkt
    spec = json.loads((folder / 'spec.json').read_text())
    loss = spec.get('loss', 'squared')
    with np.load(folder / 'data.npz') as data:
        paths, y, alpha = data['paths'].copy(order='C'), data['y'].copy(order='C'), data['alpha'].copy()
        test = data['test'].copy(order='C') if 'test' in data else paths
        labels = data['labels'] if 'labels' in data else None
    select_method(method, spec['degree'])
    before = process_peak_bytes()
    started = time.perf_counter()
    permutation = None
    ordered = paths
    ordering = method.removeprefix('sample_')
    if ordering.startswith('norm_') or ordering.startswith('random_order'):
        if ordering.startswith('norm_'):
            variation = np.zeros(len(paths))
            for j in range(1, paths.shape[1]):
                variation += abs(paths[:, j]-paths[:, j-1]).max(1)
            squared = np.zeros(paths.shape[-1])
            for j in range(1, paths.shape[1]):
                increment = (paths[:, j]-paths[:, j-1])/np.maximum(variation, 1e-300)[:, None]
                squared += (increment**2).sum(0)
            permutation = np.argsort(squared, kind='stable')
            if ordering.startswith('norm_descending'): permutation = permutation[::-1]
        else:
            permutation = np.random.default_rng(1107).permutation(paths.shape[-1])
        ordered = np.ascontiguousarray(paths[:, :, permutation])
    order_seconds = time.perf_counter()-started
    model = pysiglib.sig_lasso(ordered, y, spec['degree'], alpha, loss=loss, n_jobs=spec['jobs'], tol=spec['tol'], max_iter=spec['max_iter'])
    if permutation is not None:
        words = [tuple(int(permutation[a]) for a in word) for word in model['words']]
        order = sorted(range(len(words)), key=lambda j: (len(words[j]), words[j]))
        model['words'] = [words[j] for j in order]
        model['coef'] = model['coef'][order]
        model['signatures'] = model['signatures'][:, order]
    seconds = time.perf_counter() - started
    peak = process_peak_bytes()
    record = dict(method=method, repeat=repeat, fit_seconds=seconds, fresh_process=fresh, process_peak_bytes=peak, process_peak_before=before,
                  extra_peak_bytes=peak-before, stats=model['stats'], experiment=diagnostics(), optimization=optimization_stats(),
                  objective=model['objective'].tolist(), converged=model['converged'].tolist(),
                  gap=model['dual_gap'].tolist() if model['dual_gap'] is not None else None,
                  kkt=model['kkt_violation'].tolist(), nonzeros=np.count_nonzero(model['coef'], axis=0).tolist(),
                  union_support=len(model['words']), n_iter=model['n_iter'].tolist(), native_sha256=spec['native_sha256'])
    record['channel_order_seconds'] = order_seconds
    if permutation is not None: record['channel_permutation'] = permutation.tolist()
    lengths = np.asarray([len(word) for word in model['words']])
    record['support_by_degree'] = {str(k): np.count_nonzero(model['coef'][lengths == k], axis=0).tolist()
                                   for k in range(1, spec['degree']+1)}
    dest = folder / f'{method}_{repeat}.json'
    write_json(dest, record)
    print(spec['case'], method, repeat, round(seconds, 4), record['union_support'], record['optimization'], flush=True)
    training = selected_predictions(paths, model['words'], model['coef'], model['intercept'])
    residual = y-training
    if loss == 'squared':
        values = .5*residual**2
    elif loss == 'logistic':
        from scipy.special import expit
        residual = y-expit(training)
        values = np.logaddexp(0, training)-y*training
    elif loss == 'poisson':
        residual = y-np.exp(training)
        values = np.exp(training)-y*training
    else:
        clipped = np.clip(residual, -1, 1)
        values = clipped*(residual-.5*clipped)
        residual = clipped
    objective = np.mean(values, axis=0)+alpha*np.sum(abs(model['coef']), axis=0)
    active = model['coef'] != 0
    active_error = np.max(np.where(active, abs(model['signatures'].T@residual/len(y)-alpha*np.sign(model['coef'])), 0), axis=0, initial=0.)
    tau = np.minimum(spec['tol'], 1e-4*alpha)
    record['training_feature_error'] = float(abs(training-(model['signatures']@model['coef']+model['intercept'])).max())
    record['objective_error'] = float(abs(objective-model['objective']).max())
    record['active_kkt'] = active_error.tolist()
    record['intercept_kkt'] = abs(residual.mean(0)).tolist()
    passed = bool(np.all(model['converged']) and np.all(active_error <= tau+1e-12)
                  and np.max(abs(residual.mean(0))) <= spec['tol']+1e-12
                  and record['training_feature_error'] < 1e-10 and record['objective_error'] < 1e-11
                  and model['stats']['engine'] == 'tree' and model['stats']['full_transform_calls'] == 0)
    p = sum(paths.shape[-1]**k for k in range(1, spec['degree']+1))
    if 'reference_folder' in spec:
        source = Path(spec['reference_folder'])
        design = np.load(source/'design.npy', mmap_mode='r')
        words = [tuple(w) for w in json.loads((source/'words.json').read_text())]
        indices = [words.index(w) for w in model['words']]
        corr = design.T@residual/len(y)
        error = np.maximum(0, abs(corr).max(0)-alpha)
        error = np.maximum(error, np.max(np.where(active, abs(corr[indices]-alpha*np.sign(model['coef'])), 0), axis=0, initial=0.))
        record['exhaustive_kkt'] = error.tolist()
        passed &= bool(np.all(error <= tau+1e-12))
    elif 8*len(y)*p <= 4 << 30:
        error = exhaustive_kkt(paths, -residual/len(y), spec['degree'], alpha, model['words'], model['coef'])
        record['exhaustive_kkt'] = error.tolist()
        passed &= bool(np.all(error <= tau+1e-12))
    started = time.perf_counter()
    predicted = pysiglib.sig_lasso_predict(test, model, n_jobs=spec['jobs'])
    record['prediction_seconds'] = time.perf_counter()-started
    independently_predicted = selected_predictions(test, model['words'], model['coef'], model['intercept'])
    record['prediction_error'] = float(abs(predicted-independently_predicted).max())
    passed &= record['prediction_error'] < 1e-10
    if labels is not None:
        record['accuracy'] = float(np.mean(predicted.argmax(1) == labels))
    record['passed'] = passed
    write_json(dest, record)
    if repeat == 0:
        np.savez(folder/f'{method}_model.npz', training=training, predictions=predicted, coef=model['coef'],
                 intercept=model['intercept'], words=np.asarray([','.join(map(str, w)) for w in model['words']]))
    if not passed:
        raise AssertionError(f'Independent fit audit failed: {folder}, {method}, {repeat}')


def summarize(output):
    rows = []
    for folder in sorted(output.iterdir()):
        if not (folder/'spec.json').exists():
            continue
        spec = json.loads((folder/'spec.json').read_text())
        for method in spec['methods']:
            records = [json.loads(p.read_text()) for p in folder.glob(f'{method}_*.json')]
            records = [r for r in records if r['repeat'] >= 0 and r['method'] == method]
            if not records:
                continue
            times = [r['fit_seconds'] for r in records]
            row = dict(case=folder.name, method=method, repeats=len(records), seconds=float(np.median(times)),
                       iqr_seconds=float(np.subtract(*np.percentile(times, [75, 25]))),
                       native_mib=float(np.median([r['stats']['peak_bytes'] for r in records]))/2**20,
                       extra_process_mib=float(np.median([r['extra_peak_bytes'] for r in records]))/2**20 if all(r.get('fresh_process', True) for r in records) else None,
                       process_mib=float(np.median([r['process_peak_bytes'] for r in records]))/2**20 if all(r.get('fresh_process', True) for r in records) else None,
                       support=[r['union_support'] for r in records], passed=all(r.get('passed', False) for r in records),
                       nonzeros=records[0]['nonzeros'], accuracy=records[0].get('accuracy'))
            prediction = [r['prediction_seconds'] for r in records]
            row['prediction_seconds'] = float(np.median(prediction))
            row['prediction_iqr_seconds'] = float(np.subtract(*np.percentile(prediction, [75, 25])))
            for name in ('preparation_seconds', 'search_seconds', 'solve_seconds', 'certification_seconds', 'terminal_correlations', 'search_passes'):
                row[name] = float(np.median([r['stats'][name] for r in records]))
            if (folder/'balanced_model.npz').exists() and (folder/f'{method}_model.npz').exists():
                with np.load(folder/'balanced_model.npz') as baseline, np.load(folder/f'{method}_model.npz') as model:
                    row['training_difference'] = float(abs(model['training']-baseline['training']).max())
                    row['prediction_difference'] = float(abs(model['predictions']-baseline['predictions']).max())
            rows.append(row)
            print(row['case'], method, round(row['seconds'], 4), round(row['native_mib'], 2), row['passed'])
    write_json(output/'summary.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/optimization_round2'))
    parser.add_argument('--data', type=Path, default=Path('output/tt_bound'))
    parser.add_argument('--experiment-root', type=Path, default=Path('../../sig_branch_and_bound_experiments'))
    parser.add_argument('--cases', nargs='+', default=['baseline', 'degree', 'arabic4', 'speech3', 'insect3'])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['balanced', 'anderson', 'gram', 'block16'])
    parser.add_argument('--response', type=int)
    parser.add_argument('--loss', choices=['squared', 'logistic', 'poisson', 'huber'], default='squared')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--reuse-process', action='store_true')
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.worker:
            if args.suite:
                spec = json.loads((args.worker/'spec.json').read_text())
                for repeat in range(-spec['warmups'], spec['repeats']):
                    methods = spec['methods']
                    order = methods[repeat % len(methods):]+methods[:repeat % len(methods)]
                    for method in order:
                        worker(args.worker, method, repeat, fresh=False)
                return
            worker(args.worker, args.method, args.repeat)
            return
        if args.summary:
            summarize(args.output)
            return
        for case in args.cases:
            folder = args.output/case
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'spec.json').exists():
                raise FileExistsError(folder/'spec.json')
            prepare(args, case, folder)
            if args.reuse_process:
                subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_lasso_optimization', '--worker', str(folder.resolve()), '--suite'],
                               check=True, timeout=args.timeout)
                continue
            for repeat in range(-args.warmups, args.repeats):
                order = args.methods[repeat % len(args.methods):]+args.methods[:repeat % len(args.methods)]
                for method in order:
                    subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_lasso_optimization', '--worker', str(folder.resolve()),
                                    '--method', method, '--repeat', str(repeat)], check=True, timeout=args.timeout)
        summarize(args.output)


if __name__ == '__main__':
    main()
