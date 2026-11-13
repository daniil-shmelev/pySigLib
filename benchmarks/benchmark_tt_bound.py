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

"""Full-solver comparison of Bernstein, TT and combined signature bounds."""

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lasso_features import CASES, write_json

BOUNDS = ('bernstein', 'tt', 'combined')


def select_bound(name):
    from pysiglib.load_siglib import CPSIG
    function = CPSIG.sig_working_set_experimental_bound
    function.argtypes, function.restype = (ctypes.c_uint32,), ctypes.c_int
    if function(BOUNDS.index(name)):
        raise RuntimeError('Experimental native bound selection failed')


def prepare(args, case):
    from benchmarks.lasso_feature_reference import dictionary
    folder = args.output / case
    folder.mkdir(parents=True, exist_ok=True)
    if case.startswith('arabic'):
        degree = int(case[-1])
        protocol = json.loads((args.source / 'sig_branch_and_bound_experiments/results/optimized_degree4_20261028/protocol.json').read_text())
        item = protocol['datasets']['arabic']
        source = Path(item['prepared'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['prepared_sha256']
        data = np.load(source)
        paths = np.asarray(data['train'], dtype=np.float64)
        y = np.asarray(data['response'], dtype=np.float64)
        alpha = np.array(item['choices'][str(degree)]['alpha'])
        provenance = dict(source=str(source), sha256=item['prepared_sha256'], ratio=item['choices'][str(degree)]['ratio'])
    else:
        n, length, dimension, degree = CASES[case]
        rng = np.random.default_rng(7201)
        arrays = {}
        for name, count in (('train', n), ('validation', 128), ('test', 512)):
            increments = rng.normal(size=(count, length - 1, dimension)) / np.sqrt(length - 1)
            if case == 'anisotropic':
                increments[:, :, 4:] *= .05
            arrays[name] = np.concatenate((np.zeros((count, 1, dimension)), increments.cumsum(1)), axis=1)
        targets = []
        source_words = {'sig': [(0, 0, 0), (0, 1, 1), (1, 2, 3)],
                        'log': [(0, 0, 1), (0, 1, 1), (0, 1, 2)],
                        'hall': [(0, 1, 0), (0, 1, 1), (0, 2, 1)]}
        for mode in ('sig', 'log', 'hall'):
            sources = []
            for name in ('train', 'validation', 'test'):
                words, design = dictionary(arrays[name], 3, mode)
                sources.append(design[:, [words.index(w) for w in source_words[mode]]])
            mean, scale = sources[0].mean(0), sources[0].std(0)
            combined = [((z - mean) / scale) @ np.array([1., -.8, .6]) for z in sources]
            targets.append([z / combined[0].std() for z in combined])
        endpoint = []
        for name in ('train', 'validation', 'test'):
            _, z = dictionary(arrays[name], 1, 'sig')
            endpoint.append(z[:, 0] - z[:, 1])
        mean, scale = endpoint[0].mean(), endpoint[0].std()
        targets.insert(0, [(z - mean) / scale for z in endpoint])
        responses = []
        for j in range(3):
            response = np.column_stack([target[j] for target in targets])
            responses.append(response + rng.normal(scale=.05, size=response.shape))
        paths, y = arrays['train'], responses[0]
        provenance = dict(seed=7201, ratio=args.ratio)
        alpha = None
    words, design = dictionary(paths, degree, 'sig')
    if alpha is None:
        alpha = args.ratio * np.max(np.abs((design - design.mean(0)).T @ (y - y.mean(0))), axis=0) / len(y)
    np.savez(folder / 'data.npz', paths=paths, y=y, alpha=alpha)
    np.save(folder / 'design.npy', design)
    write_json(folder / 'words.json', words)
    spec = dict(case=case, degree=degree, shape=list(paths.shape), targets=y.shape[1], alpha=alpha.tolist(),
                jobs=args.jobs, repeats=args.repeats, provenance=provenance, tol=1e-8, max_iter=3000,
                native_sha256=hashlib.sha256(Path('pysiglib/cpsig.dll').read_bytes()).hexdigest())
    write_json(folder / 'spec.json', spec)
    return folder


def worker(folder, modes=BOUNDS, selector=select_bound, reference='bernstein', diagnostics=None, fit_auditor=None):
    import pysiglib
    from benchmarks.lasso_feature_reference import audit
    spec = json.loads((folder / 'spec.json').read_text())
    data_folder = Path(spec.get('data_folder', str(folder)))
    data = np.load(data_folder / 'data.npz')
    paths, y, alpha = data['paths'], data['y'], data['alpha']
    records, retained = [], {}
    for repeat in range(-1, spec['repeats']):
        order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
        for mode in order:
            selector(mode)
            gc.collect()
            started = time.perf_counter()
            model = pysiglib.sig_lasso(paths, y, spec['degree'], alpha, n_jobs=spec['jobs'], tol=spec['tol'], max_iter=spec['max_iter'])
            seconds = time.perf_counter() - started
            record = dict(mode=mode, repeat=repeat, fit_seconds=seconds, stats=model['stats'],
                          converged=np.asarray(model['converged']).tolist(), objective=np.asarray(model['objective']).tolist(),
                          gap=np.asarray(model['dual_gap']).tolist(), nonzeros=np.count_nonzero(model['coef'], axis=0).tolist(),
                          union_support=len(model['words']), n_iter=np.asarray(model['n_iter']).tolist())
            if diagnostics is not None:
                record['experiment'] = diagnostics()
            if fit_auditor is not None and repeat >= 0:
                record['fit_audit'] = fit_auditor(model)
            records.append(record)
            write_json(folder / 'timings.json', records)
            print(spec['case'], mode, repeat, round(seconds, 4), record['union_support'], flush=True)
            if 'fit_audit' in record and not record['fit_audit']['passed']:
                raise AssertionError(f'Independent repeated-fit audit failed: {folder}, {mode}, {repeat}')
            if repeat == 0:
                retained[mode] = model
            else:
                if repeat > 0:
                    np.testing.assert_allclose(model['objective'], retained[mode]['objective'], atol=1e-8 if fit_auditor else 1e-12, rtol=1e-12)
                del model
    design = np.load(data_folder / 'design.npy', mmap_mode='r')
    words = [tuple(w) for w in json.loads((data_folder / 'words.json').read_text())]
    audits, predictions = {}, {}
    for mode, model in retained.items():
        result = audit(model, design, words, y, alpha)
        predictions[mode] = model['signatures'] @ model['coef'] + model['intercept']
        selected_prediction = pysiglib.sig_lasso_predict(paths, model, n_jobs=spec['jobs'])
        result['prediction_error'] = float(np.max(np.abs(predictions[mode] - selected_prediction)))
        result['reference'] = reference
        result['same_support_as_reference'] = model['words'] == retained[reference]['words']
        result['objective_error'] = float(np.max(np.abs(model['objective'] - retained[reference]['objective'])))
        residual = y - predictions[mode]
        correlations = (design - design.mean(0)).T @ residual / len(y)
        violation = np.maximum(0, np.max(np.abs(correlations), axis=0) - alpha)
        lookup = {w: j for j, w in enumerate(words)}
        for j, word in enumerate(model['words']):
            active = model['coef'][j] != 0
            error = np.abs(correlations[lookup[word]] - alpha * np.sign(model['coef'][j]))
            violation = np.maximum(violation, np.where(active, error, 0))
        result['kkt_violation'] = violation.tolist()
        result['gap_error'] = float(np.max(np.abs(np.array(result['gap']) - model['dual_gap'])))
        gap_passed = result['gap_error'] < 1e-9
        if diagnostics is not None:
            theta = (residual - residual.mean(0)) / len(y)
            tau = np.minimum(spec['tol'], 1e-4 * alpha)
            lower = theta * (alpha / (alpha + tau + 1e-13))
            dual_high = np.sum(y * theta, axis=0) - len(y) * np.sum(theta * theta, axis=0) / 2
            dual_low = np.sum(y * lower, axis=0) - len(y) * np.sum(lower * lower, axis=0) / 2
            upper_gap = np.asarray(result['objective']) - np.minimum(dual_high, dual_low)
            result['gap_upper_limit'] = upper_gap.tolist()
            result['gap_underestimate'] = float(max(0, np.max(np.array(result['gap']) - model['dual_gap'])))
            result['gap_upper_error'] = float(max(0, np.max(model['dual_gap'] - upper_gap)))
            gap_passed = result['gap_underestimate'] < 1e-9 and result['gap_upper_error'] < 1e-9
        result['passed'] = bool(np.all(model['converged']) and max(violation) < 2e-8 and gap_passed and result['columns_error'] < 1e-10
                                and result['prediction_error'] < 1e-10 and result['objective_error'] < 1e-8)
        audits[mode] = result
    for mode in modes:
        audits[mode]['fit_prediction_difference'] = float(np.max(np.abs(predictions[mode] - predictions[reference])))
    write_json(folder / 'audit.json', audits)
    if not all(value['passed'] for value in audits.values()):
        raise AssertionError(f'Independent fit audit failed: {folder}')


def summarize(output):
    rows = []
    for folder in sorted(output.iterdir()):
        if not (folder / 'audit.json').exists():
            continue
        records = json.loads((folder / 'timings.json').read_text())
        audits = json.loads((folder / 'audit.json').read_text())
        for mode in BOUNDS:
            values = [r for r in records if r['mode'] == mode and r['repeat'] >= 0]
            times = [r['fit_seconds'] for r in values]
            row = dict(case=folder.name, mode=mode, median_seconds=float(np.median(times)),
                       iqr_seconds=float(np.subtract(*np.percentile(times, [75, 25]))),
                       native_peak_mib=float(np.median([r['stats']['peak_bytes'] for r in values])) / 2**20,
                       support=values[0]['nonzeros'], union_support=values[0]['union_support'], audit=audits[mode])
            for key in ('preparation_seconds', 'search_seconds', 'solve_seconds', 'certification_seconds',
                        'constructed', 'evaluated', 'pruned', 'block_queries', 'terminal_correlations', 'search_passes'):
                if key in values[0]['stats']:
                    row[key] = float(np.median([r['stats'][key] for r in values]))
            rows.append(row)
    write_json(output / 'summary.json', rows)
    for row in rows:
        print(f"{row['case']:12} {row['mode']:10} {row['median_seconds']:9.4f} ({row['iqr_seconds']:.4f}) {row['native_peak_mib']:9.2f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('output/tt_bound'))
    parser.add_argument('--source', type=Path, default=Path('../..').resolve())
    parser.add_argument('--cases', nargs='+', default=list(CASES) + ['arabic3', 'arabic4'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--ratio', type=float, default=.03)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.worker:
            worker(args.worker)
            return
        if args.summary:
            summarize(args.output)
            return
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'environment.json', dict(python=sys.version, platform=platform.platform(),
                   source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   native_sha256=hashlib.sha256(Path('pysiglib/cpsig.dll').read_bytes()).hexdigest()))
        for case in args.cases:
            folder = prepare(args, case)
            command = [sys.executable, '-m', 'benchmarks.benchmark_tt_bound', '--worker', str(folder.resolve())]
            subprocess.run(command, check=True, timeout=3600)
        summarize(args.output)


if __name__ == '__main__':
    main()
