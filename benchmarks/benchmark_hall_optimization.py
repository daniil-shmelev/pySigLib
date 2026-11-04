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

"""Compare sparse Hall kernels using the saved validation-selected penalties."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmarks.benchmark_lasso_features import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('output/lasso_features_results'))
    parser.add_argument('--output', type=Path, default=Path('output/hall_optimization/results'))
    parser.add_argument('--cases', nargs='+', default=['baseline_7201', 'baseline_7202', 'baseline_7203',
        'samples_7201', 'length_7201', 'channels_7201', 'degree_7201', 'anisotropic_7201', 'arabic_4'])
    parser.add_argument('--searches', nargs='+', choices=['plain', 'grouped', 'signed', 'grouped_signed'], default=['grouped_signed'])
    parser.add_argument('--bound', choices=['exact', 'approximate'], default='exact')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    if args.repeats < 1 or args.cache_mib < 0:
        parser.error('repeats must be positive and cache-mib must be nonnegative')
    native_hash = hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in args.cases:
        source = args.reference / case
        specification = json.loads((source / 'hall_measured_0_job.json').read_text())
        baseline = json.loads((source / 'hall_measured_0.json').read_text())
        for repeat in range(args.repeats):
            order = args.searches[repeat % len(args.searches):] + args.searches[:repeat % len(args.searches)]
            for search in order:
                folder = args.output / case
                folder.mkdir(exist_ok=True)
                output = folder / f'hall_{search}_measured_{repeat}.json'
                job = output.with_name(output.stem + '_job.json')
                specification.update(output=str(output.resolve()), native_sha256=native_hash, search=search,
                    jobs=args.jobs, cache_bytes=args.cache_mib * 1024**2, bound=args.bound, test=True)
                write_json(job, specification)
                cached = json.loads(output.read_text()) if output.exists() else {}
                if not all(cached.get(k) == specification[k] for k in ('native_sha256', 'search', 'jobs', 'cache_bytes', 'bound', 'test')):
                    command = [sys.executable, '-m', 'benchmarks.benchmark_lasso_features', '--native-dir',
                        str(args.native_dir), '--worker', str(job)]
                    with output.with_suffix('.log').open('w', encoding='ascii', errors='backslashreplace', newline='\r\n') as log:
                        try:
                            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                            if completed.returncode:
                                write_json(output, dict(specification, status='failed', returncode=completed.returncode))
                        except subprocess.TimeoutExpired:
                            write_json(output, dict(specification, status='timeout'))
                result = json.loads(output.read_text())
                result['reference_file'] = str((source / 'hall_measured_0.json').resolve())
                if 'objective' in result:
                    result['maximum_objective_difference'] = float(np.max(np.abs(np.asarray(result['objective']) - baseline['objective'])))
                write_json(output, result)
                records.append(result)
                print(case, search, repeat, result['status'], result.get('fit_seconds'), flush=True)
    summary = []
    for case in args.cases:
        before = [json.loads(p.read_text()) for p in (args.reference / case).glob('hall_measured_[0-9].json')]
        for search in args.searches:
            group = [r for r in records if Path(r['data']).parent.name == case and r['search'] == search]
            good = [r for r in group if r['status'] == 'ok']
            row = dict(case=case, search=search, outcomes=[r['status'] for r in group])
            if good:
                times = np.asarray([r['fit_seconds'] for r in good])
                row.update(median_seconds=float(np.median(times)), iqr_seconds=float(np.diff(np.percentile(times, [25, 75]))[0]),
                    previous_median_seconds=float(np.median([r['fit_seconds'] for r in before])),
                    maximum_gap=max(max(r['audit']['gap']) for r in good),
                    maximum_column_error=max(r['audit']['columns_error'] for r in good),
                    maximum_objective_difference=max(r['maximum_objective_difference'] for r in good),
                    native_peak_bytes=max(r['stats']['peak_bytes'] for r in good),
                    union_support=good[0]['union_support'])
                row['speedup'] = row['previous_median_seconds'] / row['median_seconds']
            summary.append(row)
    write_json(args.output / 'summary.json', dict(native_sha256=native_hash, cases=summary))
    if any(r['status'] != 'ok' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
