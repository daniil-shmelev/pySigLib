"""Rerun the classification and dictionary comparisons reported in main_2.tex.

From the repository root, run python -m benchmarks.benchmark_paper_audio
--data PATH --output PATH on the datasets prepared by benchmark_prefix_suffix.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks import benchmark_prefix_suffix as audio
from benchmarks.rerun_audio_working_set import dense_fit, selected_predictions
from pysiglib.load_siglib import CPSIG


def dense(args):
    spec = json.loads((args.data / 'protocol.json').read_text())['datasets'][args.dataset]
    alpha = np.asarray(spec['choices'][str(args.degree)]['alpha'])
    n, _, d = spec['shape']
    size = sum(d**k for k in range(1, args.degree + 1))
    if n * size * 8 > 4 << 30:
        raise ValueError('Full training design exceeds the paper cutoff')
    with np.load(args.data / spec['prepared']) as source:
        train, y = source['train'], source['response']
        before = audio.process_peak_bytes()
        start = perf_counter()
        model, matrix, means = dense_fit(train, y, args.degree, alpha, size)
        seconds = perf_counter() - start
        peak = audio.process_peak_bytes()
        test, labels = source['test'], source['test_labels']
    beta = model['coef']
    grad = (model['signatures'] @ beta + model['intercept'] - y) / n
    total = grad.sum(axis=0)
    corr = matrix.T @ grad + means[:, None] * total
    excess = (abs(corr).max(axis=0) - alpha).clip(min=0.)
    selected = model['signatures'].T @ grad
    active = np.where(beta != 0, abs(selected + alpha * np.sign(beta)), 0.).max(axis=0, initial=0.)
    feature_error = np.maximum(excess, active)
    assert np.all(feature_error <= np.minimum(audio.TOL, 1e-4 * alpha) + 1e-13)
    assert np.all(abs(total) <= audio.TOL)
    pred = selected_predictions(test, model['words'], beta, model['intercept'])
    audio.save_json(args.output.with_suffix('.json'), dict(seconds=seconds, extra_rss_bytes=peak-before,
        support=len(beta), correct=int((pred.argmax(1) == labels).sum()), n=len(test), kkt=feature_error.tolist(),
        intercept_kkt=abs(total).tolist(), iterations=list(map(int, model['n_iter'])), alpha=alpha.tolist()))
    np.savez(args.output.with_suffix('.npz'), pred=pred)


def experiment(data, output):
    cfg = json.loads((data / 'protocol.json').read_text())
    frozen = Path(__file__).resolve().parent / 'results/prefix_suffix/protocol.json'
    assert audio.digest(frozen) == audio.digest(data / 'protocol.json')
    for spec in cfg['datasets'].values():
        for key in ('prepared', 'sample_ids'):
            assert audio.digest(data / spec[key]) == spec[key + '_sha256']

    def run(method, name, degree, stem, feature='sig', comparison=False):
        cmd = [sys.executable, '-m', 'benchmarks.benchmark_paper_audio', '--data', str(data),
               '--output', str(stem), '--worker', method, '--dataset', name, '--degree', str(degree),
               '--feature', feature]
        if comparison:
            cmd.append('--dictionary-comparison')
        subprocess.run(cmd, check=True)
        print(method, stem.name, flush=True)

    for degree in range(1, 5):
        for i, (name, spec) in enumerate(cfg['datasets'].items()):
            n, _, d = spec['shape']
            size = sum(d**k for k in range(1, degree + 1))
            methods = ['tree']
            if n * size * 8 <= 4 << 30:
                methods = ['tree', 'dense'] if (i + degree) % 2 else ['dense', 'tree']
            for method in methods:
                run(method, name, degree, output / f'{name}-{degree}-{method}')
    for rep in range(4):
        for mode in (('sig', 'log') if rep % 2 == 0 else ('log', 'sig')):
            run('tree', 'arabic', 4, output / f'dictionary-{mode}-{rep}', mode, True)
    summarise(data, output)


def summarise(data, output):
    """Audit the saved models and collect the observations used in the tables."""
    cfg = json.loads((data / 'protocol.json').read_text())

    def load(stem):
        return json.loads(stem.with_suffix('.json').read_text())

    def audit(stem, baseline=None):
        audit_file = stem.with_name(stem.name + '-audit.json')
        audio.check(SimpleNamespace(data=data, fit=stem, baseline=baseline, output=audit_file))
        return json.loads(audit_file.read_text())

    classification = []
    for degree in range(1, 5):
        for dataset, spec in cfg['datasets'].items():
            n, _, d = spec['shape']
            size = sum(d**k for k in range(1, degree + 1))
            stem = output / f'{dataset}-{degree}-tree'
            tree = load(stem)
            assert tree['stats']['engine'] == 'tree' and tree['stats']['full_transform_calls'] == 0
            assert all(tree['converged']) and tree['stats']['dictionary_size'] == size
            record = dict(dataset=dataset, degree=degree, dictionary_size=size, design_bytes=n*size*8, tree=tree)
            full = None
            if record['design_bytes'] <= 4 << 30:
                full = output / f'{dataset}-{degree}-dense'
                record['dense'] = load(full)
                assert record['dense']['alpha'] == tree['alpha']
            record['audit'] = audit(stem, full)
            classification.append(record)
    dictionaries = {}
    for mode in ('sig', 'log'):
        reps = []
        for rep in range(1, 4):
            stem = output / f'dictionary-{mode}-{rep}'
            fit = load(stem)
            assert all(fit['converged']) and fit['stats']['engine'] == 'tree' and fit['stats']['full_transform_calls'] == 0
            fit['audit'] = audit(stem)
            with np.load(stem.with_suffix('.npz')) as z, np.load(output / f'dictionary-{mode}-1.npz') as first:
                np.testing.assert_array_equal(z['words'], first['words'])
                np.testing.assert_allclose(z['pred'], first['pred'], atol=1e-10, rtol=1e-10)
            reps.append(fit)
        dictionaries[mode] = reps
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
    audio.save_json(output / 'measurements.json', dict(protocol_sha256=audio.digest(data / 'protocol.json'),
        native_sha256=audio.digest(Path(CPSIG._name)), checkout_revision=revision, platform=platform.platform(),
        python=platform.python_version(), numpy=np.__version__, scikit_learn=version('scikit-learn'),
        feature_workers=audio.JOBS, blas_threads=1, classification_fits_per_method=1,
        dictionary_warmups=1, dictionary_repetitions=3, classification=classification, dictionary_comparison=dictionaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('tree', 'dense'))
    parser.add_argument('--dataset', choices=audio.FOLDERS)
    parser.add_argument('--degree', type=int, choices=range(1, 5))
    parser.add_argument('--feature', choices=('sig', 'log'), default='sig')
    parser.add_argument('--dictionary-comparison', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.worker == 'tree':
            audio.fit(args)
        elif args.worker == 'dense':
            dense(args)
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            experiment(args.data.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
