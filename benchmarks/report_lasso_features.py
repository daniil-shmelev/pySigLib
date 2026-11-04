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

"""Summarize the reproducible Lasso feature study and plot validation curves."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--report', type=Path, default=Path('benchmarks/lasso_features.rst'))
    args = parser.parse_args()
    environment = json.loads((args.results / 'environment.json').read_text())
    records = []
    for path in args.results.glob('*/*.json'):
        record = json.loads(path.read_text())
        if isinstance(record, dict) and 'status' in record and record.get('native_sha256') == environment['native_sha256'] and record.get('protocol_version') == 3:
            record['result_path'] = str(path)
            records.append(record)
    groups = defaultdict(list)
    for record in records:
        if Path(record['result_path']).stem.startswith(record['mode'] + '_measured_') and not record.get('warmup'):
            groups[(record['case'], record['seed'], record['degree'], record['mode'])].append(record)
    lines = ['CPU Lasso feature comparison', '============================', '',
        'The three dictionaries use factorial/variation scaling, squared loss, and separately selected penalties. '
        'All primary fits use the CPU working-set solver with eight workers, float64, tolerance 1e-8, '
        'max_iter=3000 and a 4 GiB native memory limit. '
        'An equal one-hour worker limit includes warm-up and checks. Validation selects penalties; test labels do not enter selection.', '',
        'Each timed fit runs in an isolated worker after one unmeasured fit in that worker. '
        'Separate warm-up records are retained. Selected-model tables use five timed fits. Tuning uses one fit per penalty.', '',
        'Synthetic responses are endpoint, signature, tensor-log and Hall targets, in that order. '
        'Arabic uses the saved 600/300/2200 split, all 14 channels and ten one-hot responses. '
        'Training is not repeated on validation data. Nonzero coefficients and union support measure different quantities.', '',
        'The independent Hall reference expands the dual PBW elements using '
        '`Salvi et al., Theorem 5.24 <https://arxiv.org/pdf/2212.00134>`_. '
        'The implicit Hall search uses the centred polynomial recurrence.', '',
        f'Host: {environment.get("cpu_name", environment["processor"])}. '
        f'Compiler: {environment.get("compiler", "MSVC Release")}. '
        f'Python {environment["python"].split()[0]}; NumPy {environment.get("numpy", "unknown")}; '
        f'Torch {environment.get("torch", "unknown")}. Native SHA256: ``{environment["native_sha256"]}``.', '',
        'Selected configurations', '-----------------------', '',
        '.. list-table:: Median fit time with IQR; prediction time is measured separately.', '   :header-rows: 1', '',
        '   * - Case / seed / degree', '     - Mode', '     - Fit s (IQR)', '     - Predict s (IQR)', '     - Native MiB',
        '     - Process MiB', '     - Union support', '     - Nonzeros by response', '     - Test accuracy or RMSE']
    summary = []
    for key, entries in sorted(groups.items()):
        good = [r for r in entries if r['status'] == 'ok']
        if not good:
            continue
        times = np.array([r['fit_seconds'] for r in good])
        r = good[0]
        prediction_path = Path(r['result_path']).with_name(r['mode'] + '_prediction.json')
        prediction_result = json.loads(prediction_path.read_text()) if prediction_path.exists() else None
        if prediction_result and prediction_result.get('native_sha256') != environment['native_sha256']:
            prediction_result = None
        prediction_median = prediction_result['median_seconds'] if prediction_result else float(np.median([x['prediction_seconds'] for x in good]))
        prediction_iqr = prediction_result['iqr_seconds'] if prediction_result else float(np.subtract(*np.percentile([x['prediction_seconds'] for x in good], [75, 25])))
        prepared = json.loads(Path(r['result_path']).with_name('prepared.json').read_text())
        ratios = (np.asarray(r['alpha']) / np.asarray(prepared['designs'][r['mode']]['alpha_max'])).tolist()
        metric = f"{r['test_accuracy']:.4f}" if r['classification'] else ', '.join(f'{v:.4f}' for v in r['test_rmse'])
        values = [f'{key[0]} / {key[1]} / {key[2]}', key[3], f'{np.median(times):.5f} ({np.subtract(*np.percentile(times, [75, 25])):.5f})',
            f"{prediction_median:.5f} ({prediction_iqr:.5f})", f"{max(x['stats']['peak_bytes'] for x in good)/2**20:.1f}",
            f"{max(x['process_peak_bytes'] for x in good)/2**20:.1f}", str(r['union_support']), str(r['nonzeros']), metric]
        lines += [''] + [('   * - ' if j == 0 else '     - ') + value for j, value in enumerate(values)]
        summary.append(dict(case=key[0], seed=key[1], degree=key[2], mode=key[3], repeats=len(good),
            fit_median=float(np.median(times)), fit_iqr=float(np.subtract(*np.percentile(times, [75, 25]))),
            prediction_median=prediction_median, prediction_iqr=prediction_iqr, selected_ratios=ratios,
            prediction_peak_bytes=prediction_result['process_peak_bytes'] if prediction_result else None, union_support=r['union_support'],
            nonzeros=r['nonzeros'], test_accuracy=r.get('test_accuracy'), test_rmse=r['test_rmse'],
            validation_accuracy=r.get('validation_accuracy'), validation_mse=r['validation_mse']))
    lines += ['', 'Baseline target comparison', '--------------------------', '',
        'These are mean test RMSE values across the three baseline seeds. Each response uses its own validation-selected penalty. '
        'Changing the dictionary changes the linear model class and its L1 penalty.', '',
        '.. list-table:: Baseline test errors by source target.', '   :header-rows: 1', '',
        '   * - Mode', '     - Endpoint', '     - Signature', '     - Tensor log', '     - Hall']
    for mode in ('sig', 'log', 'hall'):
        data = [r['test_rmse'] for r in summary if r['case'] == 'baseline' and r['mode'] == mode]
        if data:
            values = [mode] + [f'{v:.5f}' for v in np.mean(data, axis=0)]
            lines += [''] + [('   * - ' if j == 0 else '     - ') + v for j, v in enumerate(values)]
    lines += ['', 'Selected penalties', '------------------', '',
        'Ratios below multiply each response\'s training-only full-dictionary alpha_max. '
        'Synthetic ratios are selected per response; Arabic uses a common ratio across responses.', '',
        '.. list-table:: Validation-selected ratios.', '   :header-rows: 1', '',
        '   * - Case / seed / degree / mode', '     - Ratios by response']
    for r in summary:
        values = [' / '.join(str(r[key]) for key in ('case', 'seed', 'degree', 'mode')),
            ', '.join(f'{value:.3g}' for value in r['selected_ratios'])]
        lines += [''] + [('   * - ' if j == 0 else '     - ') + v for j, v in enumerate(values)]
    lines += ['', 'Phase costs', '-----------', '', '.. list-table:: Median native phase times in seconds.', '   :header-rows: 1', '',
        '   * - Case / seed / degree / mode', '     - Prepare', '     - Search', '     - Solve', '     - Certify', '     - Columns', '     - Support by degree']
    for key, entries in sorted(groups.items()):
        good = [r for r in entries if r['status'] == 'ok']
        if not good:
            continue
        values = [' / '.join(map(str, key))] + [f'{np.median([r["stats"][name] for r in good]):.5f}' for name in
            ('preparation_seconds', 'search_seconds', 'solve_seconds', 'certification_seconds')]
        values += [str(good[0]['stats']['columns']), str(good[0]['support_by_degree'])]
        lines += [''] + [('   * - ' if j == 0 else '     - ') + v for j, v in enumerate(values)]
    lines += ['', 'One-worker baseline', '-------------------', '',
        '.. list-table:: Five repetitions after warm-up, with the eight-worker selected penalties.', '   :header-rows: 1', '',
        '   * - Seed / mode', '     - Fit s (IQR)', '     - Union support', '     - Maximum independent gap']
    for seed in (7201, 7202, 7203):
        for mode in ('sig', 'log', 'hall'):
            good = [r for r in records if r['status'] == 'ok' and r['case'] == 'baseline' and r['seed'] == seed and r['mode'] == mode
                and '_serial_measured_' in r['result_path'] and not r.get('warmup')]
            if good:
                times = [r['fit_seconds'] for r in good]
                values = [f'{seed} / {mode}', f'{np.median(times):.5f} ({np.subtract(*np.percentile(times, [75, 25])):.5f})',
                    str(good[0]['union_support']), f'{max(max(r["audit"]["gap"]) for r in good):.3g}']
                lines += [''] + [('   * - ' if j == 0 else '     - ') + v for j, v in enumerate(values)]
    lines += ['', 'Additional baseline probes', '--------------------------', '',
        'Each probe below is one fit at baseline seed 7201 with the selected penalties. Main tables use five repetitions. '
        'Gaps are audited independently with the complete dictionary.', '',
        '.. list-table:: Single-worker timing.', '   :header-rows: 1', '',
        '   * - Probe', '     - Fit s', '     - Union support', '     - Independent gap', '     - Constructed', '     - Terminal correlations']
    for r in sorted(records, key=lambda r: r['result_path']):
        name = Path(r['result_path']).stem
        if r['case'] != 'baseline' or r['seed'] != 7201 or r['status'] != 'ok' or '_serial_measured_' in name or '_serial' not in name:
            continue
        values = [name, f'{r["fit_seconds"]:.5f}', str(r['union_support']), f'{max(r["audit"]["gap"]):.3g}',
            str(r['stats']['constructed']), str(r['stats']['terminal_correlations'])]
        lines += [''] + [('   * - ' if j == 0 else '     - ') + v for j, v in enumerate(values)]
    selected_arabic = {}
    for mode in ('sig', 'log', 'hall'):
        candidates = [r for r in summary if r['case'] == 'arabic' and r['mode'] == mode]
        if candidates:
            selected_arabic[mode] = min(candidates, key=lambda r: (-r['validation_accuracy'], np.mean(r['validation_mse']), r['degree']))
    lines += ['', 'Validation-selected Arabic models', '---------------------------------', '']
    for mode, r in selected_arabic.items():
        lines += [f'* {mode}: degree {r["degree"]}, test accuracy {r["test_accuracy"]:.4f}, '
                  f'median fit {r["fit_median"]:.5f} s, union support {r["union_support"]}.']
    lines += ['', 'Validation curves', '-----------------', '',
        'Curve points use single tuning fits; selected-model timing tables use five warmed fits. '
        'Each curve follows ratios 1, 0.1, 0.03, 0.01 and 0.003. Baseline curves use seed 7201 and '
        'the mean validation MSE across four responses. Arabic curves show each degree. '
        'All selected Arabic penalties reach the lower end of this grid; a wider grid can change the comparison.', '']
    for case in ('baseline', 'arabic'):
        relative = Path(os.path.relpath(args.results / f'{case}_validation.png', args.report.parent)).as_posix()
        lines += [f'.. image:: {relative}', f'   :alt: {case} validation error or accuracy against fit time and union support.', '   :width: 100%', '']
    before_path, after_path = args.results / 'signature_before.json', args.results / 'signature_after.json'
    if before_path.exists() and after_path.exists():
        before, after = json.loads(before_path.read_text()), json.loads(after_path.read_text())
        lines += ['', 'Signature compatibility timing', '------------------------------', '',
            'The original DLL and the new DLL each use one warm-up and 15 timed fits on two fixed signature workloads. '
            'These timings are separate from penalty selection and the main study.', '',
            '.. list-table:: Existing signature CPU path before and after the search interface change.', '   :header-rows: 1', '',
            '   * - Channels / degree', '     - Before s (IQR)', '     - After s (IQR)', '     - Objective difference']
        for dimension, degree in ((8, 4), (24, 3)):
            old = [r for r in before['records'] if r['dimension'] == dimension]
            new = [r for r in after['records'] if r['dimension'] == dimension]
            values = [f'{dimension} / {degree}']
            for data in (old, new):
                times = [r['seconds'] for r in data]
                values.append(f'{np.median(times):.6f} ({np.subtract(*np.percentile(times, [75, 25])):.6f})')
            values.append(f'{abs(old[0]["objective"] - new[0]["objective"]):.3g}')
            lines += [''] + [('   * - ' if j == 0 else '     - ') + v for j, v in enumerate(values)]
    profile_root = args.results.parent / 'lasso_features_profiles'
    if all((profile_root / f'{mode}_hotspots.csv').exists() for mode in ('sig', 'log', 'hall')):
        lines += ['', 'Native profile', '--------------', '',
            'VTune 2025.0.1 software sampling profiles the Arabic degree-four fits at ratio 0.003. '
            'The table shows the largest effective CPU self times within cpsig.dll. It excludes OS waits and spin time. '
            'CPU times sum across threads and must not be read as fit wall times. Inlined accessors and generic math functions have separate attribution. '
            'The trace also includes validation prediction outside the fit timer.', '',
            '.. list-table:: Largest native sampled costs and named feature operations in seconds.', '   :header-rows: 1', '',
            '   * - Mode', '     - Function', '     - Effective CPU s', '     - Source']
        for mode in ('sig', 'log', 'hall'):
            with (profile_root / f'{mode}_hotspots.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = [r for r in csv.DictReader(stream, delimiter='\t') if r['Module'] == 'cpsig.dll']
            rows.sort(key=lambda r: float(r['CPU Time:Effective Time']), reverse=True)
            highlighted = rows[:6] + [r for r in rows[6:] if any(name in r['Function'] for name in
                ('::hall_state', '::log_child', '::hall_envelope', '::hall_bound', '::hall_group_bound', '::log_bound', '::terminals'))]
            for row in highlighted:
                match = re.search(r'(PolynomialSearch|LassoCPU)::([A-Za-z_]+)', row['Function'])
                name = '::'.join(match.groups()) if match else row['Function'].replace('`', '')[:90]
                values = [mode, name, f'{float(row["CPU Time:Effective Time"]):.3f}', row['Source File']]
                lines += [''] + [('   * - ' if j == 0 else '     - ') + value for j, value in enumerate(values)]
        lines += ['', 'Hall state polynomial updates are the main sampled cost in the Hall fit. '
            'Signed terminal contractions are the main sampled feature operation in the log fit. '
            'The full per-function reports and worker specifications are retained in output/lasso_features_profiles.', '']
    lines += ['', 'Checks and failures', '-------------------', '', f'Recorded outcomes: {dict(Counter(r["status"] for r in records))}.', '',
        'The final Python run has 483 passed tests, two baseline assertion failures and one skipped test. '
        'The native Lasso tests have 11 passes and one baseline assertion failure. '
        'All 39 new Python tests pass. They include independent dense and rational references, active KKT conditions, '
        'held-out prediction, transforms, both dtypes, multiple responses, resource limits and CPU-only validation. '
        'The native exhaustive descendant and group-bound audit also passes through degree six.', '',
        'Three pre-existing assertions remain. The Python sparse CPU and CUDA limit tests expect at least two certificate passes; '
        'the original CPU DLL, a CUDA build with the original shared headers, and the new libraries all report one. '
        'baseline_failure.log and baseline_cuda_failure.log repeat these baseline checks. '
        'The native TerminalCacheChecksWordsAndPrefixDegree test expects a cache miss for a different prefix split; '
        'the existing cache keys the completed word. Its source and sig_lasso_math.h are unchanged. '
        'See python_tests.log, new_features_tests.log and native_tests.log for the complete test outcomes.', '']
    successful = [r for r in records if r['status'] == 'ok']
    if successful:
        lines += [f'Maximum independent full-dictionary dual gap: {max(max(r["audit"]["gap"]) for r in successful):.3g}. '
                  f'Maximum selected-column error: {max(r["audit"]["columns_error"] for r in successful):.3g}.', '']
    failures = [r for r in records if r['status'] != 'ok']
    for r in failures:
        lines += [f'* {r["result_path"]}: {r["status"]}.']
    lines += ['', 'Process memory includes Python, Torch, the dense reference and mapped calibration arrays. '
        'Native memory is the fitting workspace accounting. Dense calibration and KKT audits are excluded from fit timings.', '',
        'Prediction is measured in separate workers by case using saved selected models, one warm-up call per mode and five timed calls in rotated mode order. '
        'Only selected features and their dependencies are evaluated, with up to 64 MiB of private state reuse. '
        'The prediction process peak in summary.json is shared across modes within a case; fitting process peaks are measured per mode.', '',
        'The fits report a squared-loss duality gap and a global KKT check.', '',
        'The degree limit of 170 is an allocation and arithmetic limit. Independent general-coordinate tests '
        'cover small dictionaries through degree six; degree-170 tests cover one-channel and constant paths. '
        'Conservative floating-point padding is not an interval proof.', '',
        'Reproduction', '------------', '', '::', '',
        '   cmake --preset windows-cpu', '   cmake --build out/build/windows-cpu --config Release --target cpsig cpsig_tests -j 2',
        f'   python -m benchmarks.benchmark_lasso_features --native-dir out/build/windows-cpu/siglib/cpsig/Release --arabic --output {args.results.as_posix()}',
        f'   python -m benchmarks.benchmark_lasso_features --native-dir out/build/windows-cpu/siglib/cpsig/Release --predictions-only --output {args.results.as_posix()}',
        f'   python -m benchmarks.benchmark_lasso_features --native-dir out/build/windows-cpu/siglib/cpsig/Release --serial-only --output {args.results.as_posix()}',
        f'   python -m benchmarks.report_lasso_features {args.results.as_posix()}', '']
    lines += ['The saved profiling jobs can be repeated with VTune, for example:', '', '::', '',
        '   vtune -collect hotspots -knob sampling-mode=sw -result-dir output/lasso_features_profiles/hall_repeat -- python -m benchmarks.benchmark_lasso_features --worker output/lasso_features_profiles/hall_job.json --native-dir out/build/windows-cpu/siglib/cpsig/Release', '',
        'Correctness commands:', '', '::', '',
        '   python -m pytest tests/test_sig_lasso_features.py -q',
        '   python -m pytest tests/test_sig_lasso.py tests/test_working_set_losses.py tests/test_apis_match.py -q',
        '   out/build/windows-cpu/siglib/cpsig_unit_testing/Release/cpsig_tests.exe --gtest_filter=LassoFeatures.*:sigLassoTest.*:LassoActiveCPU.*', '']
    args.report.write_text('\n'.join(lines), encoding='ascii', newline='\r\n')
    (args.results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='ascii', newline='\r\n')
    (args.results / 'selected_arabic.json').write_text(json.dumps(selected_arabic, indent=2) + '\n', encoding='ascii', newline='\r\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for case in ('baseline', 'arabic'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for mode, color in zip(('sig', 'log', 'hall'), ('tab:blue', 'tab:orange', 'tab:green')):
            data = [r for r in records if r['status'] == 'ok' and r['case'] == case and r['mode'] == mode and 'ratio' in r and (case != 'baseline' or r['seed'] == 7201)]
            for degree in sorted({r['degree'] for r in data}):
                selected = sorted([r for r in data if r['degree'] == degree], key=lambda r: r['ratio'], reverse=True)
                metric = [r['validation_accuracy'] if case == 'arabic' else np.mean(r['validation_mse']) for r in selected]
                style = ('solid', 'dashed', 'dashdot', 'dotted')[(degree - 1) % 4]
                marker = ('o', 's', '^', 'D')[(degree - 1) % 4]
                axes[0].plot([r['fit_seconds'] for r in selected], metric, marker=marker, linestyle=style, color=color, label=f'{mode}, M={degree}')
                axes[1].plot([r['union_support'] for r in selected], metric, marker=marker, linestyle=style, color=color)
        axes[0].set_xscale('log'); axes[1].set_xscale('symlog', linthresh=1)
        for axis in axes:
            axis.set_ylabel('Validation accuracy' if case == 'arabic' else 'Mean validation MSE'); axis.grid(alpha=.25)
        axes[0].set_xlabel('Fit seconds'); axes[1].set_xlabel('Union support'); axes[0].legend(fontsize=8)
        fig.savefig(args.results / f'{case}_validation.png', dpi=180)
        plt.close(fig)
    print(args.report)


if __name__ == '__main__':
    main()
