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

"""Repeat the paper's frozen audio protocols with the public CPU signature Lasso."""

import argparse
import csv
import hashlib
import json
from math import factorial
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import warnings

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import pysiglib
from pysiglib.load_siglib import CPSIG
from benchmarks.benchmark_lasso_features import process_peak_bytes
from sig_branch_and_bound_experiments.spoken_arabic_digit.feature_sign_benchmark import normalize_paths


ROOT = Path('sig_branch_and_bound_experiments')
PROTOCOL = ROOT/'results/optimized_degree4_20261028/protocol.json'
DATASETS = {'arabic': 'spoken_arabic_digit', 'speech': 'speech_commands', 'insect': 'insect_wingbeat'}
TOL = 1e-8


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def tensor_word(j, dimension):
    word = []
    code = int(j)+1
    while code:
        code, a = divmod(code-1, dimension)
        word.append(a)
    return tuple(reversed(word))


def dense_fit(train, y, degree, alpha, p):
    n, _, d = train.shape
    matrix = np.empty((n, p), order='F')
    scales = np.repeat([factorial(k) for k in range(1, degree+1)], [d**k for k in range(1, degree+1)])
    tile = max(1, min(n, (64 << 20)//(8*p)))
    for i in range(0, n, tile):
        values = pysiglib.sig(normalize_paths(train[i:i+tile]), degree, n_jobs=8)
        values *= scales
        matrix[i:i+len(values)] = values
    means = matrix.mean(axis=0)
    matrix -= means
    ymean = y.mean(axis=0)
    norms = np.sqrt(np.einsum('ij,ij->j', matrix, matrix))
    coefs = np.zeros((p, y.shape[1]))
    iterations = []
    for j, lam in enumerate(alpha):
        target = y[:, j]-ymean[j]
        ids = np.arange(p)
        design = matrix
        coef = coefs[:, j]
        sweeps = 0
        for batch in (1, 8, 64, 512, 4096, 100000, 250000):
            nz = np.flatnonzero(coef)
            residual = target-matrix[:, nz]@coef[nz]
            q = (residual-residual.mean())/n
            corr = matrix.T@q
            error = np.maximum(abs(corr)-lam, 0.)
            error[nz] = abs(corr[nz]-lam*np.sign(coef[nz]))
            if error.max() <= .05*min(TOL, 1e-4*lam):
                break
            # Gap Safe ball for the squared loss with a free intercept.
            guard = 64*n*np.finfo(float).eps
            scale = min(1., lam/(abs(corr).max()+guard*norms.max()*np.linalg.norm(q)))
            theta = scale*q
            primal = .5*(residual@residual)/n+lam*abs(coef[nz]).sum()
            dual = target@theta-.5*n*(theta@theta)
            radius = np.sqrt(2*(max(0., primal-dual)+guard*(1+abs(primal)+abs(dual)))/n)
            keep = (abs(corr[ids])*scale+norms[ids]*(radius+guard) >= lam) | (coef[ids] != 0)
            if keep.sum() < .8*len(ids):
                ids = ids[keep]
                design = matrix[:, ids]
            solver = Lasso(alpha=lam, tol=1e-13, max_iter=batch, fit_intercept=False, copy_X=False,
                           warm_start=True, precompute=False)
            solver.coef_ = coef[ids].copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                solver.fit(design, target)
            coef[ids] = solver.coef_
            sweeps += solver.n_iter_
        iterations.append(sweeps)
    ids = np.flatnonzero(np.any(coefs != 0, axis=1))
    beta = coefs[ids]
    return (dict(words=[tensor_word(j, d) for j in ids], coef=beta, intercept=ymean-means[ids]@beta,
                 signatures=matrix[:, ids]+means[ids], n_iter=iterations), matrix, means)


def selected_predictions(paths, words, beta, intercept):
    predictions = np.broadcast_to(intercept, (len(paths), len(intercept))).copy()
    for i in range(0, len(paths), 64):
        block = normalize_paths(paths[i:i+64])
        part = predictions[i:i+len(block)]
        for j in range(0, len(words), 128):
            batch = words[j:j+128]
            cols = pysiglib.sig_coef(block, batch, n_jobs=8)
            cols *= [factorial(len(w)) for w in batch]
            part += cols @ beta[j:j+len(batch)]
    return predictions


def exhaustive_kkt(train, gradient, degree, alpha, words, beta):
    n, _, d = train.shape
    p = sum(d**k for k in range(1, degree+1))
    corr = np.zeros((p, len(alpha)))
    tile = max(1, min(n, (32 << 20)//(8*p)))
    for i in range(0, n, tile):
        block = pysiglib.sig(normalize_paths(train[i:i+tile]), degree, n_jobs=8)
        corr += block.T @ gradient[i:i+len(block)]
    offset = 0
    for k in range(1, degree+1):
        corr[offset:offset+d**k] *= factorial(k)
        offset += d**k
    for j, word in enumerate(words):
        code = 0
        for ch in word:
            code = d*code+ch+1
        active = beta[j] != 0
        corr[code-1, active] = abs(corr[code-1, active]+alpha[active]*np.sign(beta[j, active]))+alpha[active]
    return np.maximum(0., np.abs(corr).max(axis=0)-alpha)


def worker(out, name, degree, method):
    spec = json.loads(PROTOCOL.read_text())['datasets'][name]
    folder = ROOT/DATASETS[name]/'results/normalized/integral_comparison'
    prepared = folder/'prepared.npz'
    assert digest(prepared) == spec['prepared_sha256'] and digest(folder/'sample_ids.json') == spec['sample_ids_sha256']
    choice = spec['choices'][str(degree)]
    alpha = np.asarray(choice['alpha'])
    stem = f'{name}_d{degree}_{method}'
    dest = out/(stem+'.json')
    if dest.exists():
        raise FileExistsError(dest)
    n, _, d = spec['shape']
    p = sum(d**k for k in range(1, degree+1))
    record = dict(dataset=name, degree=degree, method=method, ratio=choice['ratio'], alpha=alpha.tolist(),
                  dictionary_size=p, design_bytes=8*n*p, prepared_sha256=spec['prepared_sha256'],
                  library_sha256=digest(CPSIG._name), protocol_sha256=digest(PROTOCOL))
    if method == 'dense' and 8*n*p > 4 << 30:
        record['status'] = 'design_exceeds_4_GiB'
    else:
        with np.load(prepared) as data:
            train, y = data['train'].copy(order='C'), data['response'].copy(order='C')
        assert train.dtype == y.dtype == np.dtype('float64')
        if method == 'balanced':
            from benchmarks.benchmark_bb_tradeoff import DEFAULTS, METHODS, select_method
            select_method('compact_cache0')
            record['search_configuration'] = DEFAULTS | METHODS['compact_cache0']
        before = process_peak_bytes()
        start = perf_counter()
        if method == 'dense':
            model, matrix, means = dense_fit(train, y, degree, alpha, p)
        else:
            model = pysiglib.sig_lasso(train, y, degree, alpha, loss='squared', feature_type='sig', n_jobs=8,
                                      tol=TOL, max_iter=3000)
        record['fit_seconds'] = perf_counter()-start
        peak = process_peak_bytes()
        record.update(process_peak_before=before, process_peak=peak, extra_peak=peak-before)
        if method == 'balanced':
            print(stem, 'fit finished', record['fit_seconds'], flush=True)
        words, beta, intercept = model['words'], model['coef'], model['intercept']
        training = model['signatures'] @ beta + intercept
        grad = (training-y)/n
        objective = .5*n*np.sum(grad**2, axis=0) + alpha*np.sum(abs(beta), axis=0)
        active = beta != 0
        active_kkt = np.max(np.where(active, abs(model['signatures'].T@grad+alpha*np.sign(beta)), 0.), axis=0, initial=0.)
        tau = np.minimum(TOL, 1e-4*alpha)
        ok = (abs(grad.sum(axis=0)) <= TOL+1e-14) & (active_kkt <= tau+1e-14)
        if method == 'dense':
            maximum = np.max(abs(matrix.T@grad+means[:, None]*grad.sum(axis=0)), axis=0)
            ok &= maximum <= alpha+tau+1e-14
            record['max_correlation'] = maximum.tolist()
        else:
            assert model['stats']['engine'] == 'tree' and model['stats']['full_transform_calls'] == 0
            ok &= model['converged']
            np.testing.assert_allclose(objective, model['objective'], atol=2e-12, rtol=2e-12)
            record.update(stats=model['stats'], kkt=model['kkt_violation'].tolist(), gap=model['dual_gap'].tolist())
        with np.load(prepared) as data:
            test, labels = data['test'].copy(order='C'), data['test_labels']
        scores = selected_predictions(test, words, beta, intercept)
        if method != 'dense':
            predicted = np.concatenate([pysiglib.sig_lasso_predict(test[i:i+64].copy(), model, n_jobs=8)
                                        for i in range(0, len(test), 64)])
            np.testing.assert_allclose(scores, predicted, rtol=1e-10, atol=1e-10)
        if method == 'balanced':
            independently_fitted = selected_predictions(train, words, beta, intercept)
            np.testing.assert_allclose(training, independently_fitted, rtol=1e-10, atol=1e-10)
            record['training_audit_max_error'] = float(np.max(abs(training-independently_fitted)))
            record['prediction_audit_max_error'] = float(np.max(abs(scores-predicted)))
            if 8*n*p <= 4 << 30:
                error = exhaustive_kkt(train, (independently_fitted-y)/n, degree, alpha, words, beta)
                record['exhaustive_kkt'] = error.tolist()
                ok &= error <= tau+1e-12
        record.update(status='complete' if ok.all() else 'not_converged', certified=ok.tolist(),
                      active_kkt=active_kkt.tolist(), intercept_kkt=abs(grad.sum(axis=0)).tolist(),
                      objective=objective.tolist(), nnz=active.sum(axis=0).tolist(), support=len(words),
                      iterations=np.asarray(model['n_iter']).tolist(), correct=int((scores.argmax(1)==labels).sum()),
                      test_count=len(labels))
        np.savez_compressed(out/(stem+'.npz'), training=training, scores=scores, coef=beta, intercept=intercept,
                            words=[','.join(map(str, w)) for w in words])
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='ascii')
    tmp.replace(dest)
    print(stem, record['status'], record.get('fit_seconds', ''), flush=True)


def report(out, tables):
    summary = []
    for name, dirname in DATASETS.items():
        rows = []
        estimated_memory = False
        with np.load(ROOT/dirname/'results/normalized/integral_comparison/prepared.npz') as data:
            train, y, labels = data['train'], data['response'], data['test_labels']
        for degree in range(1, 5):
            prefix = f'{name}_d{degree}'
            working, dense = [json.loads((out/f'{prefix}_{kind}.json').read_text()) for kind in ('working', 'dense')]
            assert working['status'] == 'complete' and all(working['certified'])
            assert working['stats']['engine'] == 'tree' and working['stats']['full_transform_calls'] == 0
            assert working['alpha'] == dense['alpha'] and working['library_sha256'] == dense['library_sha256']
            with np.load(out/f'{prefix}_working.npz') as model:
                words = [tuple(map(int, w.split(','))) for w in model['words']]
                training = selected_predictions(train, words, model['coef'], model['intercept'])
                np.testing.assert_allclose(training, model['training'], rtol=1e-10, atol=1e-10)
                assert np.count_nonzero(model['scores'].argmax(1)==labels) == working['correct']
            row = dict(dataset=name, degree=degree, dimension=train.shape[-1], engine=working['stats']['engine'],
                       p=working['dictionary_size'], words=working['support'], correct=working['correct'],
                       n_test=len(labels), accuracy=working['correct']/len(labels), seconds=working['fit_seconds'],
                       extra_mib=working['extra_peak']/2**20, native_mib=working['stats']['peak_bytes']/2**20,
                       reported_kkt=max(working['kkt']), max_dual_gap=max(working['gap']), dense_status=dense['status'],
                       dense_design_bytes=dense['design_bytes'],
                       dense_seconds='', dense_extra_mib='', dense_accuracy='', objective_diff='', max_train_diff='',
                       max_test_diff='', label_disagreements='', exhaustive_kkt='')
            cells = [str(degree)]
            if dense['status'] == 'design_exceeds_4_GiB':
                assert dense['design_bytes'] > 4 << 30
                size = dense['design_bytes']
                estimate = f'{size/2**40:.2f} TiB' if size >= 2**40 else f'{size/2**30:.1f} GiB'
                cells += ['--', '--', estimate + r'$^{*}$']
                estimated_memory = True
            else:
                assert dense['status'] == 'complete' and all(dense['certified'])
                with np.load(out/f'{prefix}_working.npz') as a, np.load(out/f'{prefix}_dense.npz') as b:
                    residual = (a['training']-y)/len(y)
                    alpha = np.array(working['alpha'])
                    words = [tuple(map(int, w.split(','))) for w in a['words']]
                    error = exhaustive_kkt(train, residual, degree, alpha, words, a['coef'])
                    assert np.all(error <= np.minimum(TOL, 1e-4*alpha)+1e-12)
                    row.update(max_train_diff=float(abs(a['training']-b['training']).max()),
                               max_test_diff=float(abs(a['scores']-b['scores']).max()),
                               label_disagreements=int((a['scores'].argmax(1)!=b['scores'].argmax(1)).sum()),
                               exhaustive_kkt=float(error.max()))
                row.update(dense_seconds=dense['fit_seconds'], dense_extra_mib=dense['extra_peak']/2**20,
                           dense_accuracy=dense['correct']/len(labels),
                           objective_diff=float(abs(np.array(working['objective'])-dense['objective']).max()))
                cells += [f"{row['dense_accuracy']*100:.2f}", f"{row['dense_seconds']:.3f}", f"{row['dense_extra_mib']:.2f} MiB"]
            cells += [f"{row['accuracy']*100:.2f}", f"{row['seconds']:.3f}", f"{row['extra_mib']:.2f}", str(row['words'])]
            rows.append(' & '.join(cells)+r' \\')
            summary.append(row)
        lines = [r'\begin{center}', r'\small', r'\setlength{\tabcolsep}{3pt}',
                 r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}c|rrr|rrrr}',
                 r'& \multicolumn{3}{c|}{Full coordinate descent} & \multicolumn{4}{c}{Branch and bound} \\',
                 r'$M$ & Acc. (\%) & Time (s) & Memory & Acc. (\%) & Time (s) & MiB & Words \\',
                 r'\hline', *rows, r'\end{tabular*}']
        if estimated_memory:
            lines.append(r'\par\smallskip{\footnotesize $^{*}$Estimated minimum for the full fit, from the training matrix alone.}')
        lines.append(r'\end{center}')
        (tables/f'{name}_table.tex').write_text('\n'.join(lines)+'\n', encoding='ascii', newline='\r\n')
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


def compare_with_paper(out, tables):
    rows = []
    for name in DATASETS:
        table = tables/f'{name}_table.tex'
        cells = {}
        for line in table.read_text().splitlines():
            fields = [field.strip() for field in line.removesuffix(r'\\').split('&')]
            if fields[0] in ('1', '2', '3', '4') and len(fields) == 8:
                cells[int(fields[0])] = fields[1:]
        for degree in range(1, 5):
            record = json.loads((out/f'{name}_d{degree}_balanced.json').read_text())
            full_acc, full_time, full_memory, paper_acc, paper_time, paper_memory, words = cells[degree]
            full_memory = full_memory.replace(r'$^{*}$', '*')
            stats = record['stats']
            accuracy = 100*record['correct']/record['test_count']
            extra_mib = record['extra_peak']/2**20
            rows.append(dict(dataset=name, degree=degree, dictionary_size=record['dictionary_size'],
                full_accuracy_percent=full_acc, full_fit_seconds=full_time, full_memory=full_memory,
                paper_accuracy_percent=float(paper_acc), paper_fit_seconds=float(paper_time),
                paper_extra_peak_mib=float(paper_memory), paper_words=int(words),
                balanced_accuracy_percent=accuracy, balanced_fit_seconds=record['fit_seconds'],
                balanced_extra_peak_mib=extra_mib, balanced_words=record['support'],
                correct=record['correct'], test_count=record['test_count'],
                time_ratio=record['fit_seconds']/float(paper_time),
                extra_peak_ratio=extra_mib/float(paper_memory),
                native_peak_mib=stats['peak_bytes']/2**20, process_peak_mib=record['process_peak']/2**20,
                preparation_seconds=stats['preparation_seconds'], search_seconds=stats['search_seconds'],
                solve_seconds=stats['solve_seconds'], certification_seconds=stats['certification_seconds'],
                search_passes=stats['search_passes'], full_transform_calls=stats['full_transform_calls'],
                status=record['status'], certified_responses=sum(record['certified']),
                reported_kkt=max(record['kkt']), exhaustive_kkt=max(record.get('exhaustive_kkt', []), default=''),
                max_dual_gap=max(record['gap']), training_audit_max_error=record['training_audit_max_error'],
                prediction_audit_max_error=record['prediction_audit_max_error'],
                penalty_ratio=record['ratio'], source_table_sha256=digest(table)))
    with (out/'comparison.csv').open('w', encoding='ascii', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    revision = subprocess.check_output(['git', '-C', str(tables), 'rev-parse', 'HEAD'], text=True).strip()
    faster = sum(r['time_ratio'] < 1 for r in rows)
    less_memory = sum(r['extra_peak_ratio'] < 1 for r in rows)
    accuracy_matches = sum(round(r['balanced_accuracy_percent'], 2) == r['paper_accuracy_percent'] for r in rows)
    support_matches = sum(r['balanced_words'] == r['paper_words'] for r in rows)
    lines = [
        'Balanced branch and bound on the three paper experiments',
        '',
        f'Paper revision {revision}',
        f"Native binary SHA256 {record['library_sha256']}",
        f"Saved protocol SHA256 {record['protocol_sha256']}",
        'Intel Core i7-13700H, Windows, 32 GiB installed RAM.',
        'One fresh CPU process per row, eight workers, one BLAS thread, float64.',
        'Each fit uses the paper splits, paths, response penalties, intercept,',
        'tolerance 1e-8 and limit of 3000 working-set updates per response.',
        'Fit times exclude data loading, prediction and independent audits.',
        'Memory is the increase in peak process resident memory during fit.',
        'Native peak and total process peak are separate columns in comparison.csv.',
        'Full-design entries come from the paper. A star denotes an estimated',
        'minimum from the full training design alone.',
        '',
        'The balanced configuration uses TT bounds, a candidate pool of 256,',
        'rank 16, probe batches of four, early release of terminal states and',
        'no optional cache. Terminal checks use the existing degree policy.',
        '',
        '                    Published full design         Published branch and bound       Balanced branch and bound',
        'Dataset  Degree     Acc.%    Fit s         Memory    Acc.%    Fit s      MiB   Words    Acc.%    Fit s      MiB   Words',
    ]
    for r in rows:
        lines.append(
            f"{r['dataset']:8} {r['degree']:>4} "
            f"{r['full_accuracy_percent']:>10} {r['full_fit_seconds']:>10} {r['full_memory']:>14} "
            f"{r['paper_accuracy_percent']:>8.2f} {r['paper_fit_seconds']:>9.3f} "
            f"{r['paper_extra_peak_mib']:>8.2f} {r['paper_words']:>7} "
            f"{r['balanced_accuracy_percent']:>8.2f} {r['balanced_fit_seconds']:>9.3f} "
            f"{r['balanced_extra_peak_mib']:>8.2f} {r['balanced_words']:>7}"
        )
    exhaustive = [r['exhaustive_kkt'] for r in rows if r['exhaustive_kkt'] != '']
    lines += [
        '',
        f'Faster than the published branch and bound in {faster} of {len(rows)} rows.',
        f'Lower peak resident memory increase in {less_memory} of {len(rows)} rows.',
        f'Published test accuracy matches in {accuracy_matches} of {len(rows)} rows.',
        f'Published union support count matches in {support_matches} of {len(rows)} rows.',
        f"Certified responses: {sum(r['certified_responses'] for r in rows)} of {10*len(rows)}.",
        f"Full signature transform calls during fitting: {sum(r['full_transform_calls'] for r in rows)}.",
        f"Exhaustive KKT audits: {len(exhaustive)} rows, maximum error {max(exhaustive):.4g}.",
        f"Maximum independent training-score difference: {max(r['training_audit_max_error'] for r in rows):.4g}.",
        f"Maximum independent test-score difference: {max(r['prediction_audit_max_error'] for r in rows):.4g}.",
        'The three largest dictionaries use the native tree certificate and',
        'independent checks of the selected features, intercept and objective.',
        '',
        'Raw JSON files contain penalties, objective and KKT values per class,',
        'nonzero counts, working-set updates, allocation and timing counters,',
        'data hashes and the native binary hash. NPZ files contain the fitted',
        'words, coefficients, intercepts, training scores and test scores.',
        'The paper and new fit times are single runs from separate sessions.',
        '',
        'Reproduce from the tmp/tt_lasso_bound worktree in PowerShell.',
        '$env:PYTHONPATH = (Resolve-Path ../..).Path',
        'python -m benchmarks.rerun_audio_working_set --experiment-root ../../sig_branch_and_bound_experiments --output output/balanced_paper_audio_repeat --method balanced',
        'python -m benchmarks.rerun_audio_working_set --output output/balanced_paper_audio_repeat --compare-paper-tables ../../sig_branch_and_bound/tables',
        '',
    ]
    (out/'REPORT.txt').write_text('\n'.join(lines), encoding='ascii', newline='\r\n')


def main():
    global ROOT, PROTOCOL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--experiment-root', type=Path, default=ROOT)
    ap.add_argument('--dataset', choices=DATASETS)
    ap.add_argument('--degree', type=int, choices=range(1, 5))
    ap.add_argument('--method', choices=['working', 'dense', 'balanced'])
    ap.add_argument('--tables', type=Path, help='Audit completed trials and write the three LaTeX tables')
    ap.add_argument('--compare-paper-tables', type=Path, help='Read existing paper tables and write comparison.csv')
    args = ap.parse_args()
    ROOT = args.experiment_root.resolve()
    PROTOCOL = ROOT/'results/optimized_degree4_20261028/protocol.json'
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compare_paper_tables:
        compare_with_paper(args.output, args.compare_paper_tables)
        return
    if args.tables:
        with threadpool_limits(1, user_api='blas'):
            report(args.output, args.tables)
        return
    if args.dataset:
        if (args.output/'STOP').exists():
            raise SystemExit('This trial directory has been closed')
        if not args.degree or not args.method:
            ap.error('Provide --degree and --method for one fit')
        with threadpool_limits(1, user_api='blas'):
            worker(args.output, args.dataset, args.degree, args.method)
        return
    for degree in range(1, 5):
        for name in DATASETS:
            for method in ((args.method,) if args.method else ('working', 'dense')):
                dest = args.output/f'{name}_d{degree}_{method}.json'
                if dest.exists():
                    old = json.loads(dest.read_text())
                    if old['library_sha256'] != digest(CPSIG._name) or old['protocol_sha256'] != digest(PROTOCOL):
                        raise ValueError('Existing row uses a different build or protocol')
                    continue
                print('Starting', name, degree, method, flush=True)
                subprocess.run([sys.executable, '-m', 'benchmarks.rerun_audio_working_set', '--output', str(args.output),
                                '--experiment-root', str(ROOT), '--dataset', name,
                                '--degree', str(degree), '--method', method], check=True)


if __name__ == '__main__':
    main()
