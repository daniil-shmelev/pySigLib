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

"""Compare sparse and full CPU fits at saved validation-selected penalties."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmarks.benchmark_lasso_features import MODES, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('output/lasso_features_results/samples_7201'))
    parser.add_argument('--output', type=Path, default=Path('output/hall_optimization/larger_full_comparison'))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error('repeats and timeout must be positive')
    native_hash = hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = [(mode, method) for mode in MODES for method in ('sparse', 'full')]
    records = []
    for repeat in range(args.repeats):
        order = settings[repeat % len(settings):] + settings[:repeat % len(settings)]
        for mode, method in order:
            specification = json.loads((args.reference / f'{mode}_measured_0_job.json').read_text())
            output = args.output / f'{mode}_{method}_measured_{repeat}.json'
            job = output.with_name(output.stem + '_job.json')
            specification.update(output=str(output.resolve()), native_sha256=native_hash, method=method,
                search='grouped_signed', jobs=8, cache_bytes=256 * 1024**2, bound='exact', test=True)
            write_json(job, specification)
            command = [sys.executable, '-m', 'benchmarks.benchmark_lasso_features', '--native-dir', str(args.native_dir), '--worker', str(job)]
            with output.with_suffix('.log').open('w', encoding='ascii', errors='backslashreplace', newline='\r\n') as log:
                try:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    if completed.returncode:
                        write_json(output, dict(specification, status='failed', returncode=completed.returncode))
                except subprocess.TimeoutExpired:
                    write_json(output, dict(specification, status='timeout'))
            result = json.loads(output.read_text())
            records.append(result)
            print(mode, method, repeat, result['status'], result.get('fit_seconds'), flush=True)
    rows, agreement = [], []
    for mode, method in settings:
        group = [r for r in records if r['mode'] == mode and r['method'] == method]
        good = [r for r in group if r['status'] == 'ok']
        row = dict(mode=mode, method=method, statuses=[r['status'] for r in group])
        if good:
            times = np.asarray([r['fit_seconds'] for r in good])
            row.update(median_fit_seconds=float(np.median(times)), fit_iqr_seconds=float(np.diff(np.percentile(times, [25, 75]))[0]),
                native_peak_mib=max(r['stats']['peak_bytes'] for r in good) / 1024**2,
                fit_process_peak_mib=max(r['fit_process_peak_bytes'] for r in good) / 1024**2,
                dictionary_size=good[0]['stats']['dictionary_size'], union_support=good[0]['union_support'],
                maximum_gap=max(max(r['audit']['gap']) for r in good),
                maximum_column_error=max(r['audit']['columns_error'] for r in good),
                median_prediction_seconds=float(np.median([r['prediction_seconds'] for r in good])))
        rows.append(row)
    for mode in MODES:
        pair = [next((r for r in records if r['mode'] == mode and r['method'] == method and 'model_file' in r), None) for method in ('sparse', 'full')]
        if any(r is None for r in pair):
            continue
        coefficients, predictions, word_sets = [], [], []
        words = [tuple(w) for w in json.loads(Path(pair[0]['words']).read_text())]
        lookup = {word: index for index, word in enumerate(words)}
        design = np.load(pair[0]['design'], mmap_mode='r')
        for record in pair:
            with np.load(record['model_file']) as model:
                metadata = json.loads(str(model['metadata']))
                selected = [lookup[tuple(w)] for w in metadata['words']]
                dense = np.zeros((len(words), model['coef'].shape[1]))
                dense[selected] = model['coef']
                coefficients.append(dense)
                word_sets.append(metadata['words'])
                predictions.append(design @ dense + model['intercept'])
        agreement.append(dict(mode=mode, same_words=word_sets[0] == word_sets[1],
            maximum_coefficient_difference=float(np.max(np.abs(coefficients[0] - coefficients[1]))),
            maximum_training_prediction_difference=float(np.max(np.abs(predictions[0] - predictions[1]))),
            maximum_objective_difference=float(np.max(np.abs(np.asarray(pair[0]['objective']) - pair[1]['objective']))),
            maximum_test_rmse_difference=float(np.max(np.abs(np.asarray(pair[0]['test_rmse']) - pair[1]['test_rmse'])))))
    with np.load(args.reference / 'data.npz') as data:
        shapes = {key: list(data[key].shape) for key in ('train', 'response', 'test')}
    summary = dict(native_sha256=native_hash, reference=str(args.reference.resolve()), input_shapes=shapes,
        repeats=args.repeats, jobs=8, dtype='float64', degree=records[0]['degree'], bound='exact', search='grouped_signed',
        rows=rows, sparse_full_agreement=agreement)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if any(r['status'] != 'ok' for r in records) or len(agreement) != 3 or any(
            r['maximum_training_prediction_difference'] > 1e-7 or r['maximum_objective_difference'] > 1e-9 for r in agreement):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
