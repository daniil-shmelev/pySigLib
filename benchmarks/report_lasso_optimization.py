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

"""Consolidate the terminal-bound and restricted-solver experiments."""

import argparse
import json
from pathlib import Path

import numpy as np


def runs(folder, method):
    records = [json.loads(path.read_text()) for path in folder.glob(f'{method}_*.json')]
    records = [r for r in records if r['method'] == method]
    if not records or not all(r.get('passed', False) for r in records):
        raise ValueError(f'Missing or failed audit: {folder}, {method}')
    spec = json.loads((folder/'spec.json').read_text())
    if sorted(r['repeat'] for r in records) != list(range(-spec['warmups'], spec['repeats'])):
        raise ValueError(f'Incomplete repetitions: {folder}, {method}')
    return sorted((r for r in records if r['repeat'] >= 0), key=lambda r: r['repeat'])


def statistics(records):
    times = [r['fit_seconds'] for r in records]
    prediction = [r['prediction_seconds'] for r in records if 'prediction_seconds' in r]
    result = dict(repeats=len(records), fit_seconds=float(np.median(times)),
                  fit_iqr=float(np.subtract(*np.percentile(times, [75, 25]))) if len(times) > 1 else None,
                  prediction_seconds=float(np.median(prediction)) if prediction else None,
                  prediction_iqr=float(np.subtract(*np.percentile(prediction, [75, 25]))) if len(prediction) > 1 else None,
                  native_mib=float(np.median([r['stats']['peak_bytes'] for r in records]))/2**20)
    for key in ('preparation_seconds', 'search_seconds', 'solve_seconds', 'certification_seconds',
                'terminal_correlations', 'search_passes'):
        result[key] = float(np.median([r['stats'][key] for r in records]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/optimization_round2'))
    parser.add_argument('--historical', type=Path, default=Path('output/balanced_paper_audio'))
    args = parser.parse_args()
    rows, audits = [], []
    for name in ('arabic', 'speech', 'insect'):
        for degree in range(1, 5):
            case = f'{name}{degree}'
            historical = case == 'insect4'
            folder = args.root/('final_largest' if historical else 'final_fresh')/case
            optimized = runs(folder, 'optimized')[0]
            repeated = args.root/'final_repeated'/case
            timing = repeated if repeated.exists() else folder
            optimized_runs = runs(timing, 'optimized')
            audits.extend(optimized_runs)
            if historical:
                baseline = json.loads((args.historical/f'{name}_d{degree}_balanced.json').read_text())
                if baseline['status'] != 'complete':
                    raise ValueError('The historical control did not complete')
                baseline['union_support'] = baseline['support']
                baseline['nonzeros'] = baseline['nnz']
                baseline['accuracy'] = baseline['correct']/baseline['test_count']
                baseline['process_peak_bytes'] = baseline['process_peak']
                baseline['extra_peak_bytes'] = baseline['extra_peak']
                baseline_runs = [baseline]
                baseline_model = args.historical/f'{name}_d{degree}_balanced.npz'
            else:
                baseline = runs(folder, 'balanced')[0]
                baseline_runs = runs(timing, 'balanced')
                audits.extend(baseline_runs)
                baseline_model = folder/'balanced_model.npz'
            before, after = statistics(baseline_runs), statistics(optimized_runs)
            for summary, raw in ((before, baseline), (after, optimized)):
                summary.update(process_mib=raw['process_peak_bytes']/2**20,
                               extra_process_mib=raw['extra_peak_bytes']/2**20,
                               fresh_native_mib=raw['stats']['peak_bytes']/2**20,
                               support=raw['union_support'], nonzeros=raw['nonzeros'], accuracy=raw['accuracy'],
                               objective=raw['objective'])
            with np.load(baseline_model) as old, np.load(folder/'optimized_model.npz') as new:
                old_prediction = old['scores'] if historical else old['predictions']
                same_support = all(set(old['words'][old['coef'][:, j] != 0]) == set(new['words'][new['coef'][:, j] != 0])
                                   for j in range(new['coef'].shape[1]))
                comparison = dict(same_support=bool(same_support), training_difference=float(abs(old['training']-new['training']).max()),
                                  prediction_difference=float(abs(old_prediction-new['predictions']).max()),
                                  objective_difference=float(abs(np.asarray(before['objective'])-after['objective']).max()))
            rows.append(dict(case=case, baseline=before, optimized=after, comparison=comparison,
                             historical_baseline=historical, native_sha256=optimized['native_sha256'],
                             support_by_degree=optimized['support_by_degree']))
    if len({row['native_sha256'] for row in rows}) != 1:
        raise ValueError('Final results use different native builds')
    with (args.root/'comparison.json').open('w', encoding='ascii', newline='\r\n') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
        stream.write('\n')

    lines = ['CPU signature Lasso: optimisation study', '',
             'Final native SHA256: '+rows[0]['native_sha256'],
             'Intel Core i7-13700H; Windows; float64; eight CPU workers; one BLAS thread.',
             'The paper splits, transforms, normalisation and per-response penalties are unchanged.',
             'Each paper fit has 600 training samples, 25 points and ten responses. Channel counts are 14, 65 and 201.',
             'Test samples: Arabic 2200, Speech 1000, Insect 1000. Tolerance 1e-8; at most 3000 working-set updates.',
             'Fits use no full signature transform. Independent audits and prediction are outside fit timing.',
             'No explicit fit memory limit was set. The optional cache is zero; exhaustive audits are limited to 4 GiB designs.', '',
             'Balanced means the earlier TT configuration: pool 256, rank 16, probe batch 4, early terminal-state release.',
             'Optimized retains balanced at degree 1; adds Anderson and Gram reuse at degree 2; at degree >=3 it also uses',
             'grouped signed terminal checks, the operator prebound and the flat Cauchy bound. This is an experimental',
             'benchmark profile. Public API defaults and the paper were not changed.', '',
             'The degree-1 profiles are identical; their timing differences are measurement variation.',
             'Fit seconds are medians. IQR is the 75th percentile minus the 25th percentile.',
             'Arabic4, Speech3 and Insect3 use one warm-up and three timed repetitions in rotated order.',
             'Other full paper rows use one fresh process per method. The Insect4 balanced control is historical:',
             '3188.6 seconds from a different session and native build. Its ratio is not a paired timing estimate.', '',
             'Case       Balanced s (IQR)    Optimized s (IQR)   Ratio   Support   Accuracy %']
    for row in rows:
        b, a = row['baseline'], row['optimized']
        mark = '*' if row['historical_baseline'] else ''
        before_iqr = f"{b['fit_iqr']:.4f}" if b['fit_iqr'] is not None else 'single'
        after_iqr = f"{a['fit_iqr']:.4f}" if a['fit_iqr'] is not None else 'single'
        lines.append(f"{row['case']:<10} {b['fit_seconds']:9.4f} ({before_iqr}) {a['fit_seconds']:10.4f} ({after_iqr}) "
                     f"{b['fit_seconds']/a['fit_seconds']:6.2f}{mark:<1} {a['support']:7} {100*a['accuracy']:11.2f}")
    lines += ['', 'Fresh-process memory, MiB. Native means accounted native allocations. Process peak includes Python,',
              'Torch and other loaded libraries. Increase means the rise in process peak during fit. Reused-process',
              'memory is excluded from these columns. Native timing-run medians are also stored in comparison.json.',
              'Case       Native B/O       Process peak B/O       Peak increase B/O']
    for row in rows:
        b, a = row['baseline'], row['optimized']
        lines.append(f"{row['case']:<10} {b['fresh_native_mib']:7.2f}/{a['fresh_native_mib']:7.2f} "
                     f"{b['process_mib']:10.2f}/{a['process_mib']:8.2f} {b['extra_process_mib']:10.2f}/{a['extra_process_mib']:7.2f}")
    lines += ['', 'Costs in seconds. Each phase is a median over the timing runs used above.',
              'Case/method                   Prepare    Search     Solve    Certify    Predict (IQR)']
    for row in rows:
        for method in ('baseline', 'optimized'):
            r = row[method]
            prediction_iqr = f"{r['prediction_iqr']:.4f}" if r['prediction_iqr'] is not None else 'single'
            prediction = f"{r['prediction_seconds']:.4f} ({prediction_iqr})" if r['prediction_seconds'] is not None else 'not recorded'
            lines.append(f"{row['case']+'/'+method:<29} {r['preparation_seconds']:8.4f} {r['search_seconds']:9.4f} "
                         f"{r['solve_seconds']:9.4f} {r['certification_seconds']:9.4f} {prediction}")
    lines += ['', 'Per-response nonzero coefficients; response order matches the saved paper protocol.']
    for row in rows:
        lines.append(f"{row['case']:<10} {row['optimized']['nonzeros']}")
    lines += ['', f"Identical per-response support sets: {sum(row['comparison']['same_support'] for row in rows)} of 12.",
              f"Largest cross-method objective difference: {max(row['comparison']['objective_difference'] for row in rows):.3e}.",
              f"Largest cross-method training-score difference: {max(row['comparison']['training_difference'] for row in rows):.3e}.",
              f"Largest cross-method test-score difference: {max(row['comparison']['prediction_difference'] for row in rows):.3e}.",
              f"Largest independent feature reconstruction error: {max(r['training_feature_error'] for r in audits):.3e}.",
              f"Largest independent prediction error: {max(r['prediction_error'] for r in audits):.3e}.",
              f"Largest exhaustive KKT error: {max(max(r['exhaustive_kkt']) for r in audits if 'exhaustive_kkt' in r):.3e}.",
              f"Largest recorded squared-loss dual gap in the final paper runs: {max(max(r['gap']) for r in audits):.3e}.",
              'All final fits passed the native convergence test and independent active KKT, intercept, objective and prediction checks.',
              'Nine paper dictionaries also received exhaustive KKT checks. Speech4, Insect3 and Insect4 use the tree certificate',
              'with independent selected-coordinate checks. These are floating-point checks, not interval-arithmetic proofs.', '',
              'Synthetic checks: 128 samples, 13 points, four responses. Baseline uses 8 channels and degree 4;',
              'degree uses 4 channels and degree 6. Nonquadratic responses use seed 9417 and training-only alpha_max.',
              'Squared-loss timings have five repeats; other losses have three. All use one warm-up and rotated order.',
              'Suite/case/method                            Fit s (IQR)       Native MiB  Union']
    for suite in ('final_synthetic', 'final_one_worker', 'final_logistic', 'final_poisson', 'final_huber'):
        for case in ('baseline', 'degree'):
            for method in ('balanced', 'optimized'):
                records = runs(args.root/suite/case, method)
                r = statistics(records)
                lines.append(f"{suite+'/'+case+'/'+method:<44} {r['fit_seconds']:.5f} ({r['fit_iqr']:.5f}) {r['native_mib']:9.2f} {records[0]['union_support']:6}")
    lines += ['', 'Component pilots below use response 0 only, one warm-up and two timed repetitions.',
              'They use earlier builds and are separate from the final comparisons. The v6 underflow issue described',
              'below affects very high degrees; these pilot cases use degrees 3 and 4.',
              'Study/case/method                                         Fit s (IQR)      Native MiB']
    for suite in ('solver_v3', 'matrix_pilot', 'bounds_v6', 'grouping_v6'):
        for r in json.loads((args.root/suite/'summary.json').read_text()):
            lines.append(f"{suite+'/'+r['case']+'/'+r['method']:<57} {r['seconds']:8.4f} ({r['iqr_seconds']:.4f}) {r['native_mib']:9.2f}")
    lines += ['', 'Additional single-response Insect4 pilot: one fit per method, no warm-up, within one process.']
    for r in json.loads((args.root/'norm_pilot/summary.json').read_text()):
        if r['case'] == 'insect4':
            lines.append(f"{r['method']:<20} {r['seconds']:.4f} seconds; {r['native_mib']:.2f} native MiB.")
    lines += ['', 'The flat/operator norm checks and safeguarded Anderson solve give the useful combined result.',
              'Persistent Gram reuse gives a smaller extra gain. Matrix blocks add workspace and give only a small',
              'speed gain in the Speech4 pilot. Segment refinement is tighter, but faster only on some cases.',
              'Grouping at degree 3 improves speed with these checks, at a memory cost. It is not forced below degree 3.',
              'Faster runs can evaluate more terminal correlations because grouping changes reuse and vectorisation.',
              'The results do not establish a universal fastest rule. Shared traversal across responses was not implemented.', '',
              'Validation: 927 Python tests and eight native tests passed, including the degree-170 underflow regression.',
              'Coverage includes both input dtypes, one/eight workers, multiple losses, all three feature dictionaries for',
              'the accelerated solver, cancellation, memory limits, prediction and API compatibility. CUDA was not evaluated.',
              'There were two rejected prototypes. v2 failed a convergence audit because of unsigned arithmetic in an',
              'acceptance guard; this was fixed in v3. v6 passed the low-degree checks but failed the added degree-170',
              'norm-underflow regression; this was fixed in v7. The corresponding failure logs are retained.',
              'No final fit failed or timed out. The scope of the performance tables is the standard signature dictionary.', '',
              'METHOD.txt gives the formulas and literature references. comparison.json stores the consolidated results;',
              'the final_* directories store raw fit records and models. All sources and the measured DLL are in measured_source.',
              'COMMANDS.txt contains the build, validation and benchmark commands.']
    with (args.root/'REPORT.txt').open('w', encoding='ascii', newline='\r\n') as stream:
        stream.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[:38]))


if __name__ == '__main__':
    main()
