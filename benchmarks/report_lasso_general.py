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

"""Summarize repeated CPU feature-search experiments and saved-build comparisons."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lasso_features import write_json


def summarize(root):
    groups = defaultdict(list)
    for spec_path in sorted(root.rglob('spec.json')):
        spec = json.loads(spec_path.read_text())
        if 'variants' not in spec or 'mode' not in spec:
            continue
        for variant in spec['variants']:
            records = []
            for path in spec_path.parent.glob(f'{variant}_*.json'):
                record = json.loads(path.read_text())
                if record.get('variant') == variant and record['repeat'] >= 0:
                    records.append(record)
            if records:
                groups[(str(spec_path.parent.relative_to(root)), variant)] = records
    rows = []
    for (folder, variant), records in groups.items():
        first = records[0]
        row = dict(folder=folder, variant=variant, case=first['case'], mode=first['mode'], loss=first['loss'],
                   degree=first['degree'], jobs=first['jobs'], cache_bytes=first['cache_bytes'], ratio=first['ratio'],
                   repeats=len(records), passed=all(r['passed'] for r in records), native_sha256=first['native_sha256'],
                   nonzeros=first['nonzeros'], union_support=first['union_support'], support_by_degree=first['support_by_degree'],
                   test_rmse=first['test_rmse'], accuracy=first.get('accuracy'), binary_accuracy=first.get('binary_accuracy'),
                   objective=first['objective'],
                   max_exhaustive_kkt=max(max(r['exhaustive_kkt']) for r in records),
                   max_reported_kkt=max(max(r['kkt']) for r in records),
                   max_reported_gap=max(max(r['gap']) for r in records) if first['gap'] is not None else None,
                   max_intercept_error=max(r['intercept_error'] for r in records),
                   max_objective_error=max(r['objective_error'] for r in records),
                   max_column_error=max(r['column_error'] for r in records),
                   max_prediction_error=max(r['prediction_error'] for r in records))
        for field in ('fit_seconds', 'prediction_seconds'):
            values = [r[field] for r in records]
            row[field] = float(np.median(values))
            row[field + '_iqr'] = float(np.subtract(*np.percentile(values, [75, 25])))
        row['native_mib'] = float(np.median([r['stats']['peak_bytes'] for r in records]))/2**20
        row['process_mib'] = float(np.median([r['process_peak_bytes'] for r in records]))/2**20 if all(r['fresh_process'] for r in records) else None
        row['process_extra_mib'] = float(np.median([r['extra_peak_bytes'] for r in records]))/2**20 if row['process_mib'] is not None else None
        for field in ('preparation_seconds', 'search_seconds', 'insertion_seconds', 'solve_seconds', 'certification_seconds',
                      'search_passes', 'terminal_correlations', 'block_queries', 'constructed', 'full_transform_calls'):
            row[field] = float(np.median([r['stats'][field] for r in records]))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', default='previous')
    parser.add_argument('--candidate', default='recommended')
    args = parser.parse_args()
    rows = summarize(args.root)
    lookup = {(r['folder'], r['variant']): r for r in rows}
    comparisons = []
    for row in rows:
        if row['variant'] != args.candidate or (row['folder'], args.reference) not in lookup:
            continue
        old = lookup[(row['folder'], args.reference)]
        comparison = dict(folder=row['folder'], case=row['case'], mode=row['mode'], loss=row['loss'],
                          speedup=old['fit_seconds']/row['fit_seconds'], prediction_speedup=old['prediction_seconds']/row['prediction_seconds'],
                          native_ratio=row['native_mib']/old['native_mib'],
                          objective_difference=max(abs(np.asarray(old['objective'])-row['objective'])),
                          support_equal=old['nonzeros'] == row['nonzeros'] and old['union_support'] == row['union_support'])
        with np.load(args.root/row['folder']/f'{args.reference}_model.npz') as a, np.load(args.root/row['folder']/f'{args.candidate}_model.npz') as b:
            comparison['support_equal'] = bool(comparison['support_equal'] and np.array_equal(a['words'], b['words'])
                                               and np.array_equal(a['coef'] != 0, b['coef'] != 0))
            comparison['coefficient_difference'] = float(abs(a['coef']-b['coef']).max(initial=0.)) if comparison['support_equal'] else None
            comparison['training_difference'] = float(abs(a['training']-b['training']).max())
            comparison['prediction_difference'] = float(abs(a['predictions']-b['predictions']).max())
            if row['loss'] == 'squared':
                spec = json.loads((args.root/row['folder']/'spec.json').read_text())
                design = np.load(spec['design'], mmap_mode='r')
                with np.load(spec['response']) as targets:
                    y = targets['y']
                alpha = np.asarray(spec['alpha'])
                comparison['independent_gaps'] = {}
                comparison['reported_gap_underestimate'] = 0.
                for label, model in ((args.reference, a), (args.candidate, b)):
                    residual = y-model['training']
                    theta = residual-residual.mean(0)
                    with threadpool_limits(limits=1):
                        maximum = abs(design.T@theta/len(y)).max(0)
                    theta *= np.minimum(1., alpha/np.maximum(maximum, 1e-300))/len(y)
                    primal = .5*np.mean(residual**2, axis=0)+alpha*abs(model['coef']).sum(0)
                    gap = primal-np.sum(y*theta, axis=0)+.5*len(y)*np.sum(theta**2, axis=0)
                    comparison['independent_gaps'][label] = gap.tolist()
                    recorded = json.loads((args.root/row['folder']/f'{label}_0.json').read_text())
                    comparison['reported_gap_underestimate'] = max(comparison['reported_gap_underestimate'],
                                                                  float(np.max(gap-recorded['gap'])))
        comparisons.append(comparison)
    write_json(args.root/'summary.json', dict(rows=rows, comparisons=comparisons))
    lines = ['Repeated CPU feature-search measurements', '',
             'Times are medians; IQR is the 75th percentile minus the 25th percentile.',
             'Native memory includes tracked native allocations. Process memory is the peak working set in fresh workers.',
             'Independent audits and dense reference construction are outside fit timing.', '',
             'case mode variant fit_s IQR_s predict_s predict_IQR_s native_MiB process_MiB support passed']
    for row in rows:
        process = f'{row["process_mib"]:.2f}' if row['process_mib'] is not None else 'reused'
        lines.append(f'{row["case"]} {row["mode"]} {row["variant"]} {row["fit_seconds"]:.6f} {row["fit_seconds_iqr"]:.6f} '
                     f'{row["prediction_seconds"]:.6f} {row["prediction_seconds_iqr"]:.6f} {row["native_mib"]:.3f} '
                     f'{process} {row["union_support"]} {row["passed"]}')
    lines += ['', f'Comparisons: {args.reference} -> {args.candidate}',
              'case mode fit_speedup prediction_speedup native_ratio objective_difference prediction_difference same_support']
    for row in comparisons:
        lines.append(f'{row["case"]} {row["mode"]} {row["speedup"]:.3f} {row["prediction_speedup"]:.3f} {row["native_ratio"]:.3f} '
                     f'{row["objective_difference"]:.3g} {row["prediction_difference"]:.3g} {row["support_equal"]}')
    (args.root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='ascii', newline='\r\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
