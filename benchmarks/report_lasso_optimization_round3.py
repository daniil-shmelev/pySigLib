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

"""Consolidate the third CPU Lasso optimisation study."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.benchmark_lasso_features import write_json
from benchmarks.report_lasso_optimization import runs, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/optimization_round3'))
    args = parser.parse_args()
    rows, checks = [], []
    for spec_path in sorted(args.root.glob('*/*/spec.json')):
        folder = spec_path.parent
        spec = json.loads(spec_path.read_text())
        for method in spec['methods']:
            records = runs(folder, method)
            row = statistics(records)
            row.update(suite=folder.parent.name, case=spec['case'], method=method, response=spec['response'],
                       loss=spec['loss'], jobs=spec['jobs'], native_sha256=records[0]['native_sha256'],
                       support=records[0]['union_support'], nonzeros=records[0]['nonzeros'],
                       accuracy=records[0].get('accuracy'), support_by_degree=records[0]['support_by_degree'],
                       constructed=float(np.median([r['stats']['constructed'] for r in records])),
                       passed=all(r['passed'] for r in records))
            fresh = all(r.get('fresh_process', True) for r in records)
            row['process_mib'] = max(r['process_peak_bytes'] for r in records)/2**20 if fresh else None
            row['extra_process_mib'] = max(r['extra_peak_bytes'] for r in records)/2**20 if fresh else None
            row['optimization'] = {key: float(np.median([r['optimization'][key] for r in records]))
                                   for key in records[0]['optimization']}
            reference = folder/'optimized_model.npz'
            model_path = folder/f'{method}_model.npz'
            if reference.exists() and model_path.exists():
                with np.load(reference) as old, np.load(model_path) as new:
                    row['same_support'] = all(set(old['words'][old['coef'][:, j] != 0]) == set(new['words'][new['coef'][:, j] != 0])
                                              for j in range(new['coef'].shape[1]))
                    row['prediction_difference'] = float(abs(old['predictions']-new['predictions']).max())
                    row['training_difference'] = float(abs(old['training']-new['training']).max())
                old_record = json.loads((folder/'optimized_0.json').read_text())
                row['objective_difference'] = float(abs(np.asarray(old_record['objective'])-records[0]['objective']).max())
            checks.extend(records)
            rows.append(row)
    queries = []
    for source in sorted(args.root.glob('queries*.json')):
        records = json.loads(source.read_text())
        groups = sorted({(r['case'], r['phase'], r['batch']) for r in records})
        for case, phase, batch in groups:
            selected = [r for r in records if (r['case'], r['phase'], r['batch']) == (case, phase, batch) and r['repeat'] >= 0]
            times = [r['seconds'] for r in selected]
            queries.append(dict(source=source.name, case=case, phase=phase, batch=batch, repeats=len(times),
                                method=selected[0]['method'], native_sha256=selected[0]['native_sha256'],
                                seconds=float(np.median(times)), iqr=float(np.subtract(*np.percentile(times, [75, 25]))),
                                native_mib=float(np.median([r['native_bytes'] for r in selected]))/2**20,
                                constructed=float(np.median([r['constructed'] for r in selected]))))
    primary = []
    for case in [f'{name}{degree}' for name in ('arabic', 'speech', 'insect') for degree in range(1, 5)]:
        fresh = {r['method']: r for r in rows if r['suite'] == 'build_fresh' and r['case'] == case}
        repeated = {r['method']: r for r in rows if r['suite'] == 'build_repeated' and r['case'] == case}
        if not fresh: continue
        timing = repeated or fresh
        primary.append(dict(case=case, before=timing['optimized'], after=timing['optimized_next'],
                            fresh_before=fresh['optimized'], fresh_after=fresh['optimized_next']))
    write_json(args.root/'comparison.json', dict(primary=primary, fits=rows, queries=queries))
    lines = ['CPU signature Lasso: third optimisation study', '',
             'Intel Core i7-13700H; Windows; float64; eight workers unless stated; one BLAS thread; tolerance 1e-8.',
             'Paper fits: 600 training paths, 25 points, ten responses; 14/65/201 channels for Arabic/Speech/Insect.',
             'Test paths: 2200/1000/1000. At most 3000 working-set updates; no explicit fit memory limit.',
             'The paper data, transforms, penalties and test split are unchanged. No full signature is built during fitting.',
             'Optimized is the previous experimental profile. Optimized_next is the new profile.',
             'The build_fresh/build_repeated suites are the main comparison: original saved DLL versus final DLL, in rotated fresh processes.',
             'The final_* suites are earlier full-fit checks of kernel 13 in build v5. Candidate_* suites use the final kernel 14 in build v6.',
             'Times are medians; IQR is the 75th minus 25th percentile. Single measurements have no IQR.',
             'Native memory counts library allocations. Process memory includes Python and loaded libraries.',
             'Process memory is reported only for fresh processes. Audits are outside fit timing.',
             'Pilots use response 0; final paper fits use all ten responses. See METHOD.txt and COMMANDS.txt.', '']
    lines.extend(['Main comparison: original saved build versus final build.',
                  'Arabic4, Speech3 and Insect3: one warm-up and three timed repetitions. Other cases: one timed fit.',
                  'The memory columns use the separate build_fresh measurements.',
                  'Case       Previous s (IQR)       New s (IQR)   Speedup   Native MiB old/new   Process MiB old/new  Union  Accuracy %'])
    for r in primary:
        b, a, fb, fa = (r[key] for key in ('before', 'after', 'fresh_before', 'fresh_after'))
        biqr = f'{b["fit_iqr"]:.4f}' if b['fit_iqr'] is not None else 'single'
        aiqr = f'{a["fit_iqr"]:.4f}' if a['fit_iqr'] is not None else 'single'
        lines.append(f'{r["case"]:10} {b["fit_seconds"]:9.4f} ({biqr:>6}) {a["fit_seconds"]:9.4f} ({aiqr:>6}) '
                     f'{b["fit_seconds"]/a["fit_seconds"]:7.2f} {fb["native_mib"]:9.2f}/{fa["native_mib"]:7.2f} '
                     f'{fb["process_mib"]:10.2f}/{fa["process_mib"]:7.2f} {a["support"]:6} {100*a["accuracy"]:11.2f}')
    lines.extend(['', 'Degree-one and degree-two profiles use the same algorithm. Their timing differences do not show a bound improvement.', ''])
    lines.extend(['Main comparison search counts, from build_fresh.',
                  'Case       Prefixes old/new         Terminal correlations old/new'])
    for r in primary:
        b, a = r['fresh_before'], r['fresh_after']
        lines.append(f'{r["case"]:10} {b["constructed"]:9.0f}/{a["constructed"]:<9.0f} {b["terminal_correlations"]:16.0f}/{a["terminal_correlations"]:.0f}')
    lines.append('')
    for suite in sorted({r['suite'] for r in rows}):
        lines.extend([suite, 'Case       Method                               Fit s    IQR s  Native MiB  Process MiB  Union  Accuracy %'])
        for r in rows:
            if r['suite'] != suite: continue
            iqr = f'{r["fit_iqr"]:.4f}' if r['fit_iqr'] is not None else 'single'
            process = f'{r["process_mib"]:.2f}' if r['process_mib'] is not None else 'reused'
            accuracy = f'{100*r["accuracy"]:.2f}' if r['accuracy'] is not None else 'n/a'
            lines.append(f'{r["case"]:10} {r["method"]:35} {r["fit_seconds"]:8.4f} {iqr:>8} {r["native_mib"]:11.2f} {process:>12} {r["support"]:6} {accuracy:>11}')
        lines.append('')
    lines.extend(['Final fit phase costs, seconds.', 'Suite/case/method                                                  Prepare    Search     Solve    Certify    Predict (IQR)'])
    for r in rows:
        if not r['suite'].startswith(('final', 'build_', 'candidate_')): continue
        piqr = f'{r["prediction_iqr"]:.4f}' if r['prediction_iqr'] is not None else 'single'
        lines.append(f'{r["suite"]+"/"+r["case"]+"/"+r["method"]:66} {r["preparation_seconds"]:8.4f} {r["search_seconds"]:9.4f} {r["solve_seconds"]:9.4f} {r["certification_seconds"]:10.4f} {r["prediction_seconds"]:10.4f} ({piqr})')
    lines.extend(['', 'Shared traversal query replay, not full fitting.',
                  'Batch 0 is the current native search; batch 1 is the shared prototype with one response.',
                  'Larger batches share prefix construction, with a separate pruning mask for each response.',
                  'Both initial and final gradients come from the saved paper fits.',
                  'Source              Case       Phase    Batch  Seconds      IQR  Native MiB  Prefixes'])
    for r in queries:
        lines.append(f'{r["source"]:19} {r["case"]:10} {r["phase"]:8} {r["batch"]:5} {r["seconds"]:8.4f} {r["iqr"]:8.4f} {r["native_mib"]:11.2f} {r["constructed"]:9.0f}')
    final = [r for r in rows if r['suite'].startswith(('final', 'build_', 'candidate_')) and r['method'] != 'optimized']
    lines.extend(['', f'Completed timed fits: {len(checks)}. All passed: {all(r["passed"] for r in checks)}.',
                  f'Final comparisons with identical per-response supports: {sum(r.get("same_support", False) for r in final)} of {len(final)}.',
                  f'Largest final objective difference: {max((r.get("objective_difference", 0) for r in final), default=0):.3e}.',
                  f'Largest final prediction difference: {max((r.get("prediction_difference", 0) for r in final), default=0):.3e}.',
                  f'Largest independent feature error: {max((r["training_feature_error"] for r in checks), default=0):.3e}.',
                  f'Largest independent prediction error: {max((r["prediction_error"] for r in checks), default=0):.3e}.',
                  f'Largest exhaustive KKT error: {max((max(r["exhaustive_kkt"]) for r in checks if "exhaustive_kkt" in r), default=0):.3e}.',
                  f'Largest reported KKT error: {max((max(r["kkt"]) for r in checks), default=0):.3e}.',
                  f'Largest reported squared-loss dual gap: {max((max(r["gap"]) for r in checks if r.get("gap") is not None), default=0):.3e}.',
                  f'Reported numerical failures: {sum(r["stats"]["numerical_failures"] for r in checks)}.',
                  f'Full signature transforms during fitting: {sum(r["stats"]["full_transform_calls"] for r in checks)}.',
                  f'Recorded timeout or subprocess failure files: {len(list(args.root.glob("*/*/failure.json")))}.',
                  'Large dictionaries use the tree certificate plus independent selected-feature, objective, gradient and prediction checks.',
                  'The convergence tolerance controls KKT error, not the dual gap.',
                  'These are floating-point checks, not interval-arithmetic proofs.', '',
                  'Recommendation and limits', '',
                  'Keep the per-sample operator and terminal bounds, individual-channel checks and fast positive reductions as the new experimental profile.',
                  'The main benefit is in large searches with many channels. Small synthetic fits can be slower because the extra tests cost more than they save.',
                  'Keep optional summary caches disabled and retain the original channel order and four-probe batches. The trials do not show a consistent combined gain from those changes.',
                  'Shared response traversal remains a query prototype. It reduces repeated prefix construction, but its timing gains vary by data and gradient. It is not a measured full-fit improvement.',
                  'The largest degree-four fits have one timed observation per build. The three repeated paper cases have three observations per build; their IQRs describe this small sample.',
                  'The new kernels are for standard-signature features. Log-signature and Hall behaviour passed regression tests; their runtime was not studied here.',
                  'The final build passed 1071 Python tests and eight native Lasso tests. See the validation logs and measured_source/manifest.json.',
                  'The experimental worktree contains the changes. Public defaults, paper text, git staging and remotes were not changed.', ''])
    with (args.root/'REPORT.txt').open('w', encoding='ascii', newline='\r\n') as output:
        output.write('\n'.join(lines))
    print(args.root/'REPORT.txt')


if __name__ == '__main__':
    main()
