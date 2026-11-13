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

"""Measure general CPU Lasso changes against independent feature dictionaries."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lasso_features import CASES, MODES, bind_native, prepare, process_peak_bytes, write_json


VARIANTS = {'control': 0, 'log_simd': 1, 'hall_reuse': 2, 'hall_direct': 4, 'groups': 8,
            'kernels': 7, 'all': 15, 'log_terminal': 16, 'log_combined': 17,
            'kernels_terminal': 23, 'groups_terminal': 31, 'sig_groups': 0,
            'log_moments': 55, 'hall_first': 87, 'combined': 119, 'combined_groups': 127,
            'ordered': 0, 'combined_ordered': 119, 'previous': 0, 'partial': 375,
            'hall_global': 247, 'hall_simd': 631, 'hall_global_simd': 759, 'final_trial': 1015, 'cost': 119,
            'hall_single': 695, 'hall_global_groups': 767, 'recommended': 2039, 'default': None, 'hall_greedy': 2039}


def configure(name, degree, cache_bytes=0):
    if name == 'default':
        if cache_bytes: raise ValueError('The public default has no optional search cache')
        return
    from pysiglib.load_siglib import CPSIG
    from benchmarks.benchmark_lasso_optimization import select_method
    select_method('optimized_next', degree)
    feature = getattr(CPSIG, 'sig_working_set_experimental_features', None)
    if feature is None:
        if VARIANTS[name]: raise RuntimeError('The DLL has no feature optimisation interface')
    else:
        feature.argtypes, feature.restype = (ctypes.c_uint32,), ctypes.c_int
        if feature(VARIANTS[name]): raise RuntimeError('Feature configuration rejected')
    if CPSIG.sig_working_set_experimental_search(4, 16, 4, 0, 3, cache_bytes):
        raise RuntimeError('Search configuration rejected')
    if name == 'sig_groups' and CPSIG.sig_working_set_experimental_kernels(15, 7):
        raise RuntimeError('Terminal configuration rejected')
    if name in ('ordered', 'combined_ordered') and CPSIG.sig_working_set_experimental_kernels(14, 15):
        raise RuntimeError('Solver configuration rejected')
    if name == 'cost' and CPSIG.sig_working_set_experimental_kernels(14, 23):
        raise RuntimeError('Solver configuration rejected')


def prepare_case(args, case):
    from benchmarks.lasso_feature_reference import dictionary
    folder = args.data / f'{case}_{args.seed}'
    if case in CASES:
        source = prepare(SimpleNamespace(output=args.data, native_dir=args.native_dir), case, args.seed)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        saved = args.paper_data / case
        source_file = saved / 'data.npz'
        if (folder / 'prepared.json').exists():
            source = json.loads((folder / 'prepared.json').read_text())
        else:
            with np.load(source_file) as data:
                arrays = dict(train=data['paths'], response=data['y'], test=data['test'],
                              test_response=np.eye(data['y'].shape[1])[data['labels'].astype(int)])
            np.savez(folder / 'data.npz', **arrays)
            source = dict(data=str((folder / 'data.npz').resolve()), degree=int(case[-1]), classification=True,
                          case=case, seed=args.seed, designs={}, provenance=dict(source=str(source_file.resolve()),
                          sha256=hashlib.sha256(source_file.read_bytes()).hexdigest()))
            for mode in args.modes:
                words, design = dictionary(arrays['train'], source['degree'], mode)
                np.save(folder / f'{mode}.npy', design)
                write_json(folder / f'{mode}_words.json', words)
                source['designs'][mode] = dict(design=str((folder / f'{mode}.npy').resolve()), words=str((folder / f'{mode}_words.json').resolve()))
            write_json(folder / 'prepared.json', source)
    if source['classification'] and args.loss != 'squared':
        raise ValueError('The saved paper comparison uses squared loss')
    with np.load(source['data']) as data:
        y = data['response'].copy()
        test_response = data['test_response'].copy()
        for mode in args.modes:
            if mode not in source['designs']:
                words, design = dictionary(data['train'], source['degree'], mode)
                np.save(folder / f'{mode}.npy', design)
                write_json(folder / f'{mode}_words.json', words)
                source['designs'][mode] = dict(design=str((folder / f'{mode}.npy').resolve()), words=str((folder / f'{mode}_words.json').resolve()))
                write_json(folder / 'prepared.json', source)
            target = folder / f'{mode}_test.npy'
            if not target.exists():
                words, design = dictionary(data['test'], source['degree'], mode)
                expected = json.loads(Path(source['designs'][mode]['words']).read_text())
                assert [list(w) for w in words] == expected
                np.save(target, design)
    rng = np.random.default_rng(args.seed + 2216)
    if args.loss == 'logistic':
        y = rng.binomial(1, expit(y)).astype(float)
        test_response = rng.binomial(1, expit(test_response)).astype(float)
        residual = y-y.mean(0)
    elif args.loss == 'poisson':
        y = rng.poisson(np.exp(.5*np.clip(y, -4, 4))).astype(float)
        test_response = rng.poisson(np.exp(.5*np.clip(test_response, -4, 4))).astype(float)
        residual = y-y.mean(0)
    elif args.loss == 'huber':
        from scipy.optimize import brentq
        y += 3*rng.normal(size=y.shape)*(rng.random(y.shape) < .05)
        intercept = np.array([brentq(lambda b: np.clip(y[:, j]-b, -1, 1).mean(), y[:, j].min()-1, y[:, j].max()+1)
                              for j in range(y.shape[1])])
        residual = np.clip(y-intercept, -1, 1)
    else:
        residual = y-y.mean(0)
    response = folder / f'{args.loss}_response.npz'
    if response.exists():
        with np.load(response) as saved:
            if not np.array_equal(saved['y'], y) or not np.array_equal(saved['test'], test_response):
                raise ValueError(f'Response data changed: {response}')
    else:
        np.savez(response, y=y, test=test_response)
    specifications = []
    for mode in args.modes:
        features = source['designs'][mode]
        design = np.load(features['design'], mmap_mode='r')
        alpha = np.maximum(1e-15, args.ratio*np.max(abs(design.T@residual/len(y)), axis=0))
        destination = args.output / f'{case}_{mode}'
        destination.mkdir(parents=True, exist_ok=True)
        spec = dict(case=case, mode=mode, degree=source['degree'], data=source['data'], response=str(response.resolve()),
                    design=features['design'], words=features['words'], test_design=str((folder / f'{mode}_test.npy').resolve()),
                    classification=source['classification'], loss=args.loss, alpha=alpha.tolist(), ratio=args.ratio,
                    jobs=args.jobs, cache_bytes=args.cache_mib << 20, memory_limit=args.memory_mib << 20,
                    repeats=args.repeats, warmups=args.warmups, variants=args.variants,
                    feature_flags={name: VARIANTS[name] for name in args.variants}, native_dir=str(args.native_dir.resolve()),
                    native_sha256=hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest())
        spec['seed'] = args.seed
        with np.load(source['data']) as data:
            spec['training_shape'], spec['test_shape'] = list(data['train'].shape), list(data['test'].shape)
        spec['response_sha256'] = hashlib.sha256(response.read_bytes()).hexdigest()
        if args.reference_dir:
            spec['reference_native_dir'] = str(args.reference_dir.resolve())
            spec['reference_sha256'] = hashlib.sha256((args.reference_dir / 'cpsig.dll').read_bytes()).hexdigest()
        path = destination / 'spec.json'
        if path.exists(): raise FileExistsError(path)
        write_json(path, spec)
        specifications.append(path)
    return specifications


def worker(path, name, repeat, fresh):
    import pysiglib
    from benchmarks.benchmark_lasso_optimization import optimization_stats
    spec = json.loads(path.read_text())
    if name == 'previous':
        spec['native_dir'] = spec['reference_native_dir']
        spec['native_sha256'] = spec['reference_sha256']
    with np.load(spec['data']) as data, np.load(spec['response']) as target:
        paths, test = data['train'].copy(order='C'), data['test'].copy(order='C')
        y, expected = target['y'], target['test']
    configure(name, spec['degree'], spec['cache_bytes'])
    before = process_peak_bytes()
    start = time.perf_counter()
    model = pysiglib.sig_lasso(paths, y, spec['degree'], spec['alpha'], feature_type=spec['mode'], loss=spec['loss'],
                              n_jobs=spec['jobs'], memory_limit=spec['memory_limit'], tol=1e-8, max_iter=3000)
    seconds = time.perf_counter()-start
    peak = process_peak_bytes()
    optimization = optimization_stats()
    design = np.load(spec['design'], mmap_mode='r')
    words = [tuple(w) for w in json.loads(Path(spec['words']).read_text())]
    lookup = {w: j for j, w in enumerate(words)}
    indices = [lookup[w] for w in model['words']]
    eta = design[:, indices]@model['coef']+model['intercept']
    residual = y-eta
    if spec['loss'] == 'logistic':
        residual = y-expit(eta)
        values = np.logaddexp(0, eta)-y*eta
    elif spec['loss'] == 'poisson':
        residual = y-np.exp(eta)
        values = np.exp(eta)-y*eta
    elif spec['loss'] == 'huber':
        residual = np.clip(residual, -1, 1)
        values = residual*(y-eta-.5*residual)
    else:
        values = .5*residual**2
    alpha = np.asarray(spec['alpha'])
    objective = values.mean(0)+alpha*abs(model['coef']).sum(0)
    corr = design.T@residual/len(y)
    error = np.maximum(0, abs(corr).max(0)-alpha)
    active = model['coef'] != 0
    error = np.maximum(error, np.max(np.where(active, abs(corr[indices]-alpha*np.sign(model['coef'])), 0), axis=0, initial=0.))
    column_error = float(np.max(abs(model['signatures']-design[:, indices]), initial=0.))
    intercept_error = float(abs(residual.mean(0)).max())
    objective_error = float(abs(objective-model['objective']).max())
    start = time.perf_counter()
    predicted = pysiglib.sig_lasso_predict(test, model, n_jobs=spec['jobs'], memory_limit=spec['memory_limit'])
    prediction_seconds = time.perf_counter()-start
    held_design = np.load(spec['test_design'], mmap_mode='r')
    prediction_error = float(abs(predicted-(held_design[:, indices]@model['coef']+model['intercept'])).max())
    predicted_response = expit(predicted) if spec['loss'] == 'logistic' else np.exp(predicted) if spec['loss'] == 'poisson' else predicted
    passed = bool(np.all(model['converged']) and np.all(error <= np.minimum(1e-8, 1e-4*alpha)+1e-11)
                  and intercept_error <= 1e-8+1e-11 and column_error < 1e-10 and prediction_error < 1e-9
                  and objective_error < 1e-10 and model['stats']['full_transform_calls'] == 0)
    record = dict(spec, variant=name, repeat=repeat, fit_seconds=seconds, prediction_seconds=prediction_seconds,
                  fresh_process=fresh, process_peak_bytes=peak, extra_peak_bytes=peak-before, stats=model['stats'], optimization=optimization,
                  objective=model['objective'].tolist(), gap=model['dual_gap'].tolist() if model['dual_gap'] is not None else None,
                  converged=model['converged'].tolist(), kkt=model['kkt_violation'].tolist(), exhaustive_kkt=error.tolist(),
                  column_error=column_error, prediction_error=prediction_error, intercept_error=intercept_error,
                  objective_error=objective_error, nonzeros=np.count_nonzero(model['coef'], axis=0).tolist(),
                  union_support=len(model['words']), support_by_degree={str(k): sum(len(w) == k for w in model['words'])
                  for k in range(1, spec['degree']+1)}, test_rmse=np.sqrt(np.mean((predicted_response-expected)**2, axis=0)).tolist(), passed=passed)
    if spec['loss'] == 'logistic':
        record['binary_accuracy'] = np.mean((predicted_response >= .5) == expected, axis=0).tolist()
    if spec['classification']:
        record['accuracy'] = float(np.mean(predicted.argmax(1) == expected.argmax(1)))
    write_json(path.parent / f'{name}_{repeat}.json', record)
    if repeat == 0:
        np.savez(path.parent / f'{name}_model.npz', words=np.asarray([str(w) for w in model['words']]),
                 coef=model['coef'], intercept=model['intercept'], predictions=predicted, training=eta)
    print(spec['case'], spec['mode'], name, repeat, f'{seconds:.4f}', f'{model["stats"]["peak_bytes"]/2**20:.2f}',
          len(model['words']), passed, flush=True)
    if not passed: raise RuntimeError('Independent audit failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, default=Path('pysiglib'))
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--data', type=Path, default=Path('output/optimization_round4/data'))
    parser.add_argument('--paper-data', type=Path, default=Path('output/optimization_round3/build_fresh'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['baseline', 'degree'])
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['control', 'all'])
    parser.add_argument('--loss', choices=['squared', 'logistic', 'poisson', 'huber'], default='squared')
    parser.add_argument('--seed', type=int, default=7201)
    parser.add_argument('--ratio', type=float, default=.03)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--cache-mib', type=int, default=0)
    parser.add_argument('--memory-mib', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--reuse-process', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    if args.worker:
        spec = json.loads(args.worker.read_text())
        bind_native(Path(spec['reference_native_dir'] if args.variant == 'previous' else spec['native_dir']))
    else:
        if 'previous' in args.variants and (not args.reference_dir or args.reuse_process):
            raise ValueError('Saved build comparisons require --reference-dir and fresh processes')
        if 'default' in args.variants and args.reuse_process and len(args.variants) > 1:
            raise ValueError('Default-policy comparisons require fresh processes')
        bind_native(args.native_dir)
    with threadpool_limits(limits=1):
        if args.worker:
            if args.suite:
                for repeat in range(-spec['warmups'], spec['repeats']):
                    variants = spec['variants']
                    for name in variants[repeat % len(variants):]+variants[:repeat % len(variants)]:
                        worker(args.worker, name, repeat, False)
            else:
                worker(args.worker, args.variant, args.repeat, True)
            return
        for case in args.cases:
            for path in prepare_case(args, case):
                spec = json.loads(path.read_text())
                commands = [[sys.executable, '-m', 'benchmarks.benchmark_lasso_general', '--output', str(args.output),
                             '--worker', str(path.resolve()), '--suite']] if args.reuse_process else []
                if not commands:
                    for repeat in range(-args.warmups, args.repeats):
                        for name in args.variants[repeat % len(args.variants):]+args.variants[:repeat % len(args.variants)]:
                            commands.append([sys.executable, '-m', 'benchmarks.benchmark_lasso_general', '--output', str(args.output),
                                             '--worker', str(path.resolve()), '--variant', name, '--repeat', str(repeat)])
                for command in commands:
                    try:
                        subprocess.run(command, check=True, timeout=args.timeout)
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                        write_json(path.parent / 'failure.json', dict(command=command, error=str(error)))
                        raise


if __name__ == '__main__':
    main()
