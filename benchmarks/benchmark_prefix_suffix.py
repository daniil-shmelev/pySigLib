"""Reproduce the audio fits and audit them using batches of selected coordinates.

Run `python benchmark_prefix_suffix.py --help` for the available commands. Each
invocation of "fit" times one ten-response fit in the installed CPU build.
"""
import argparse
import hashlib
from itertools import islice, product
import json
from math import factorial
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
import pysiglib
if __package__:
    from .benchmark_lasso_features import process_peak_bytes
else:
    from benchmark_lasso_features import process_peak_bytes


FOLDERS = {'arabic': 'spoken_arabic_digit', 'insect': 'insect_wingbeat', 'speech': 'speech_commands'}
RATIOS = {'arabic': (.03, .003, .01, .003), 'insect': (.03, .1, .1, .1), 'speech': (.01, .01, .003, .01)}
JOBS, TOL = 8, 1e-8


def normalize(paths):
    variation = np.abs(np.diff(paths, axis=1)).max(axis=2).sum(axis=1)
    return np.ascontiguousarray((paths - paths[:, :1]) / np.where(variation, variation, 1)[:, None, None])


def batches(words, size):
    words = iter(words)
    while batch := list(islice(words, size)):
        yield batch


def dictionary(d, degree, mode):
    for k in range(1, degree + 1):
        for w in product(range(d), repeat=k):
            if mode == 'sig' or all(w < w[i:] for i in range(1, k)):
                yield w


def partitions(word):
    for mask in range(1 << (len(word) - 1)):
        cuts = [0] + [i for i in range(1, len(word)) if mask & (1 << (i - 1))] + [len(word)]
        yield tuple(word[a:b] for a, b in zip(cuts, cuts[1:]))


def columns(paths, words, mode):
    """Factorial-scaled columns from sig_coef and the tensor logarithm series."""
    scales = [factorial(len(w)) for w in words]
    if mode == 'sig':
        return pysiglib.sig_coef(paths, words, n_jobs=JOBS) * scales
    expansions = [list(partitions(w)) for w in words]
    pieces = sorted({u for expansion in expansions for factors in expansion for u in factors})
    sig = pysiglib.sig_coef(paths, pieces, n_jobs=JOBS)
    by_word = {u: sig[:, j] for j, u in enumerate(pieces)}
    result = np.zeros((len(paths), len(words)))
    for j, expansion in enumerate(expansions):
        for factors in expansion:
            value = by_word[factors[0]]
            for u in factors[1:]:
                value = value * by_word[u]
            result[:, j] += ((-1.) ** (len(factors) - 1) / len(factors)) * value
    return result * scales


def calibration_words(d, degree):
    rng = np.random.default_rng(311923)
    yield from ((a,) for a in range(d))
    for k in range(2, degree + 1):
        codes = rng.choice(d**k, min(4096, d**k), replace=False)
        for code in codes:
            yield tuple(int(a) for a in np.unravel_index(code, (d,) * k))
        yield from ((a,) * k for a in range(d))


def calibrate(paths, response, words, mode):
    residual = response - response.mean(axis=0)
    maximum = np.zeros(response.shape[1])
    for batch in batches(words, min(96, max(1, paths.shape[-1] // 2))):
        scores = columns(paths, batch, mode).T @ residual / len(paths)
        maximum = np.maximum(maximum, np.abs(scores).max(axis=0))
    return maximum


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def prepare(data, experiments):
    data.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='reconstruction', split_seed=311923, n_jobs=JOBS, tolerance=TOL,
                    max_iter=3000, datasets={}, dictionary_comparison={})
    for name, folder in FOLDERS.items():
        root = (experiments / folder).resolve()
        sources = root / 'data' / ('sources.json' if name == 'arabic' else f'{name}_source.json')
        archive = json.loads(sources.read_text())
        if name == 'arabic':
            archive = dict(archive['arabic.zip'], filename='arabic.zip')
        if digest(root / 'data' / archive['filename']) != archive['sha256']:
            raise ValueError(f'Archive checksum mismatch for {name}')
        with np.load(root / 'results' / f'{name}.npz') as source:
            classes = source['classes'].tolist()
            if name == 'arabic':
                sys.path.insert(0, str(root))
                try:
                    import search  # Original resampling, PCA transform and official test labels.
                    samples, labels = search.load(name, 'test')
                    test = search.transform(search.resample(samples, 24), name, source)
                finally:
                    sys.path.pop(0)
                train, y = source['paths'][:600], source['target'][:600]
                ids = dict(train=source['training_indices'].tolist(), validation=source['tuning_indices'].tolist(),
                           test=list(range(len(test))))
            else:
                train, test, labels = source['train_paths'], source['test_paths'], source['test_labels']
                y = np.eye(10)[source['train_labels']]
                ids = {part: source[part + '_ids'].tolist() for part in ('train', 'validation', 'test')}
        x = normalize(train)
        residual = y - y.mean(axis=0)
        d = x.shape[-1]
        first = np.zeros(y.shape[1])
        for a in range(d):
            first = np.maximum(first, np.abs(x[:, -1, a] @ residual) / len(x))
        prepared, splits = data / f'{name}.npz', data / f'{name}-sample-ids.json'
        np.savez(prepared, train=train, response=y, test=test, test_labels=labels, alpha=first)
        save_json(splits, ids)
        choices = {str(k): dict(ratio=ratio, alpha=(ratio * first).tolist(), calibration='degree_one')
                   for k, ratio in enumerate(RATIOS[name], start=1)}
        if name == 'speech':
            words = dict.fromkeys(calibration_words(d, 4))
            choices['4'].update(alpha=(.01 * calibrate(x, y, words, 'sig')).tolist(),
                                calibration='reconstructed_sample', numpy_generator='PCG64', seed=311923,
                                samples_per_degree=4096, includes_all_letters_and_pure_powers=True)
        if name == 'arabic':
            for mode in ('sig', 'log'):
                alpha = .03 * calibrate(x, y, dictionary(d, 4, mode), mode)
                protocol['dictionary_comparison'][mode] = dict(dataset=name, degree=4, ratio=.03,
                                                                calibration='full_dictionary', alpha=alpha.tolist())
        protocol['datasets'][name] = dict(classes=classes, shape=list(train.shape), archive=archive, choices=choices,
                                          prepared=prepared.name, prepared_sha256=digest(prepared),
                                          sample_ids=splits.name, sample_ids_sha256=digest(splits))
    save_json(data / 'protocol.json', protocol)

def save_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fit(args):
    data, name, mode, degree = args.data, args.dataset, args.feature, args.degree
    protocol = json.loads((data / 'protocol.json').read_text())
    spec = protocol['datasets'][name]
    if args.dictionary_comparison:
        choice = protocol['dictionary_comparison'][mode]
        if (name, degree) != (choice['dataset'], choice['degree']):
            raise ValueError('The dictionary-comparison calibration requires Arabic degree four')
    else:
        choice = spec['choices'][str(degree)]
    alpha = np.asarray(choice['alpha'])
    with np.load(data / spec['prepared']) as source:
        train, y = source['train'], source['response']
        before = process_peak_bytes()
        start = perf_counter()
        model = pysiglib.sig_lasso(train, y, degree, alpha, feature_type=mode, n_jobs=JOBS, tol=TOL, max_iter=3000)
        seconds = perf_counter() - start
        peak = process_peak_bytes()
        assert np.all(model['converged']) and model['stats']['full_transform_calls'] == 0
        test, labels = source['test'], source['test_labels']
    start = perf_counter()
    pred = np.concatenate([pysiglib.sig_lasso_predict(chunk.copy(), model, n_jobs=JOBS)
                           for chunk in np.array_split(test, (len(test) + 63) // 64)])
    pred_seconds = perf_counter() - start
    record = dict(dataset=name, mode=mode, degree=degree, alpha=alpha.tolist(), seconds=seconds,
                  pred_seconds=pred_seconds, extra_rss_bytes=peak - before, correct=int(np.sum(pred.argmax(1) == labels)), n=len(labels),
                  support=len(model['words']), kkt=model['kkt_violation'].tolist(), gap=model['dual_gap'].tolist(),
                  objective=model['objective'].tolist(), converged=model['converged'].tolist(),
                  iterations=model['n_iter'].tolist(), stats=model['stats'], dictionary_comparison=args.dictionary_comparison)
    save_json(args.output.with_suffix('.json'), record)
    np.savez(args.output.with_suffix('.npz'), pred=pred, signatures=model['signatures'],
             train=model['signatures'] @ model['coef'] + model['intercept'],
             coef=model['coef'], intercept=model['intercept'], words=[' '.join(map(str, w)) for w in model['words']])


def check(args):
    info = json.loads(args.fit.with_suffix('.json').read_text())
    with np.load(args.fit.with_suffix('.npz')) as model, np.load(args.data / (info['dataset'] + '.npz')) as source:
        words = [tuple(map(int, w.split())) for w in model['words']]
        x, xt = normalize(source['train']), normalize(source['test'])
        y, beta, alpha = source['response'], model['coef'], np.asarray(info['alpha'])
        train = np.broadcast_to(model['intercept'], y.shape).copy()
        test = np.broadcast_to(model['intercept'], (len(xt), y.shape[1])).copy()
        columns_error = 0.
        selected_kkt = np.zeros_like(alpha)
        stationarity_tol = np.minimum(TOL, 1e-4 * alpha)
        mode, degree, d = info['mode'], info['degree'], x.shape[-1]
        width = min(128, max(1, info['stats']['dictionary_size'] // 2))
        for pos in range(0, len(words), width):
            group = words[pos:pos + width]
            design = columns(x, group, mode)
            end = pos + len(group)
            columns_error = max(columns_error, np.max(np.abs(design - model['signatures'][:, pos:end])))
            train += design @ beta[pos:end]
            test += columns(xt, group, mode) @ beta[pos:end]
        residual = y - train
        for pos in range(0, len(words), width):
            group = words[pos:pos + width]
            coeff = beta[pos:pos + len(group)]
            corr = columns(x, group, mode).T @ residual / len(x)
            defect = np.where(coeff != 0, abs(corr - alpha * np.sign(coeff)), np.maximum(abs(corr) - alpha, 0.))
            selected_kkt = np.maximum(selected_kkt, defect.max(axis=0))
        assert np.all(selected_kkt <= stationarity_tol + 1e-13)
        objective = .5 * np.mean(residual**2, axis=0) + alpha * np.abs(beta).sum(axis=0)
        record = dict(columns_error=float(columns_error), training_score_error=float(np.max(abs(train - model['train']))),
                      test_score_error=float(np.max(abs(test - model['pred']))), selected_kkt=float(max(selected_kkt)),
                      intercept_kkt=float(np.max(abs(residual.mean(axis=0)))),
                      objective_error=float(np.max(abs(objective - info['objective']))),
                      label_disagreements=int(np.sum(test.argmax(1) != model['pred'].argmax(1))))
        assert max(record.values()) < TOL, record
        # Same 4 GiB feasibility cutoff as the paper. The audit itself streams
        # even those dictionaries, retaining one small coordinate batch at a time.
        if 8 * len(x) * info['stats']['dictionary_size'] <= 4 << 30:
            maximum = np.zeros_like(alpha)
            centered_max = np.zeros_like(alpha)
            mean = residual.mean(axis=0)
            count = 0
            for group in batches(dictionary(d, degree, mode), width):
                design = columns(x, group, mode)
                corr = design.T @ residual / len(x)
                maximum = np.maximum(maximum, np.abs(corr).max(axis=0))
                centered_corr = corr - design.mean(axis=0)[:, None] * mean
                centered_max = np.maximum(centered_max, np.abs(centered_corr).max(axis=0))
                count += len(group)
            assert count == info['stats']['dictionary_size']
            stationarity = np.maximum(maximum - alpha, selected_kkt)
            assert np.all(stationarity <= stationarity_tol + 1e-13), stationarity
            theta = (residual - mean) / len(x) * np.minimum(1., alpha / np.maximum(centered_max, 1e-300))
            dual = np.sum(y * theta - .5 * len(x) * theta**2, axis=0)
            gap = objective - dual
            assert np.min(gap) >= -1e-13
            gap_error = float(np.max(np.abs(gap - info['gap'])))
            assert gap_error <= 1e-10, gap_error
            record.update(audited_words=count, global_kkt=max(float(max(stationarity)), record['intercept_kkt']),
                          independent_dual_gap=float(max(0., max(gap))), dual_gap_error=gap_error)
        if args.baseline:
            with np.load(args.baseline.with_suffix('.npz')) as base:
                record.update(baseline_test_score_delta=float(np.max(abs(test - base['pred']))),
                              baseline_label_disagreements=int(np.sum(test.argmax(1) != base['pred'].argmax(1))))
            assert record['baseline_label_disagreements'] == 0, record
    save_json(args.output, record)


def compare(args):
    """Run three paired fresh-process fits and an independent audit on each row."""
    builds = {'preceding': args.preceding_build.resolve(), 'current': args.current_build.resolve()}
    rows = [(name, mode, k, False) for name in FOLDERS for mode in ('sig', 'log') for k in range(1, 5)]
    rows.extend(('arabic', mode, 4, True) for mode in ('sig', 'log'))
    output = args.output.resolve()

    def run(build, command, *arguments):
        env = os.environ.copy()
        path = str(builds[build])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [path, env.get('PYTHONPATH')]))
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--data', str(args.data.resolve()),
                        command, *map(str, arguments)], env=env, check=True)

    def stem(row):
        name, mode, k, comparison = row
        return ('dict-' if comparison else '') + f'{name}-{mode}{k}'

    for build, directory in builds.items():
        if not (directory / 'pysiglib' / '__init__.py').is_file():
            raise ValueError(f'{build} must point to a directory containing the built pysiglib package')
    for rep in (1, 2, 3):
        for i, row in enumerate(reversed(rows) if rep == 2 else rows):
            name, mode, k, comparison = row
            for build in (('current', 'preceding') if (rep + i) % 2 else ('preceding', 'current')):
                dest = output / f'{build}{rep}' / stem(row)
                run(build, 'fit', '--dataset', name, '--feature', mode, '--degree', k,
                    *(['--dictionary-comparison'] if comparison else []), '--output', dest)
                print(f'Fit {rep}/3 {build} {stem(row)}', flush=True)
    for row in rows:
        name = stem(row)
        run('current', 'check', '--fit', output / 'current2' / name, '--baseline', output / 'preceding2' / name,
            '--output', output / 'audit' / f'{name}.json')
        print(f'Checked {name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Directory for prepared data and penalty vectors')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Import the three original prepared datasets and compute training-only penalties')
    prep.add_argument('--experiments', type=Path, required=True)
    run = sub.add_parser('fit', help='Run one fit in this process and save its model arrays and diagnostics')
    run.add_argument('--dataset', choices=FOLDERS, required=True)
    run.add_argument('--feature', choices=('sig', 'log'), required=True)
    run.add_argument('--degree', type=int, choices=range(1, 5), required=True)
    run.add_argument('--dictionary-comparison', action='store_true')
    run.add_argument('--output', type=Path, required=True, help='Output stem for paired .npz and .json files')
    audit = sub.add_parser('check', help='Recompute selected scores and audit all coordinates where feasible')
    audit.add_argument('--fit', type=Path, required=True)
    audit.add_argument('--baseline', type=Path)
    audit.add_argument('--output', type=Path, required=True, help='Audit JSON file')
    paired = sub.add_parser('compare', help='Repeat all 26 fits on two builds and audit the second pair')
    paired.add_argument('--preceding-build', type=Path, required=True, help='Directory containing the preceding pysiglib package')
    paired.add_argument('--current-build', type=Path, required=True, help='Directory containing the new pysiglib package')
    paired.add_argument('--output', type=Path, required=True, help='Directory for measurements and fitted models')
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.command == 'prepare':
            prepare(args.data, args.experiments)
        elif args.command == 'fit':
            fit(args)
        elif args.command == 'check':
            check(args)
        else:
            compare(args)


if __name__ == '__main__':
    main()
