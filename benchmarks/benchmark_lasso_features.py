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

"""Resumable CPU comparison of signature, tensor-log and Hall Lasso."""

import argparse
import ctypes
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

MODES = ('sig', 'log', 'hall')
RATIOS = (1., .1, .03, .01, .003)
PROTOCOL_VERSION = 3
CASES = {'baseline': (128, 13, 8, 4), 'samples': (512, 13, 8, 4),
         'length': (128, 65, 8, 4), 'channels': (128, 13, 24, 3),
         'degree': (128, 13, 4, 6), 'anisotropic': (128, 13, 8, 4)}


def bind_native(directory):
    os.environ['PATH'] = str(Path('pysiglib').resolve()) + os.pathsep + os.environ['PATH']
    original = ctypes.CDLL
    def load(name, *args, **kwargs):
        if Path(name).name == 'cpsig.dll':
            name = str((directory / 'cpsig.dll').resolve())
        return original(name, *args, **kwargs)
    ctypes.CDLL = load


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='ascii', newline='\r\n')
    temporary.replace(path)


def process_peak_bytes():
    if os.name == 'nt':
        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [(name, ctypes.c_size_t) for name in
                ('peak_working', 'working', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return counters.peak_working
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def worker(args):
    bind_native(args.native_dir)
    import pysiglib
    from benchmarks.lasso_feature_reference import audit
    specification = json.loads(args.worker.read_text())
    data = np.load(specification['data'])
    mode, degree = specification['mode'], specification['degree']
    design = np.load(specification['design'], mmap_mode='r')
    words = [tuple(w) for w in json.loads(Path(specification['words']).read_text())]
    alpha = np.asarray(specification['alpha'])
    paths, response = data['train'], data['response']
    def fit():
        return pysiglib.sig_lasso(paths, response, degree, alpha, feature_type=mode, n_jobs=specification.get('jobs', 8),
                                 tol=1e-8, max_iter=3000, memory_limit=4 * 1024**3)
    if specification.get('test', False):
        warmup = fit()
        del warmup
    started = time.perf_counter()
    model = fit()
    fit_seconds = time.perf_counter() - started
    fit_process_peak_bytes = process_peak_bytes()
    result = dict(specification, status='ok', fit_seconds=fit_seconds, protocol_version=PROTOCOL_VERSION,
        converged=np.asarray(model['converged']).tolist(), stats=model['stats'], fit_process_peak_bytes=fit_process_peak_bytes,
        reported_dual_gap=np.asarray(model['dual_gap']).tolist(), objective=np.asarray(model['objective']).tolist(),
        n_iter=np.asarray(model['n_iter']).tolist(),
        nonzeros=np.count_nonzero(model['coef'], axis=0).tolist(), union_support=len(model['words']),
        support_by_degree={str(k): sum(len(w) == k for w in model['words']) for k in range(1, degree + 1)})
    result['audit'] = audit(model, design, words, data['response'], alpha)
    if not np.all(model['converged']) or max(result['audit']['gap']) > 5e-8 or result['audit']['columns_error'] > 1e-8:
        result['status'] = 'nonconverged'
    validation = pysiglib.sig_lasso_predict(data['validation'], model, n_jobs=specification.get('jobs', 8))
    result['validation_mse'] = np.mean((validation - data['validation_response'])**2, axis=0).tolist()
    if specification['classification']:
        result['validation_accuracy'] = float(np.mean(validation.argmax(1) == data['validation_response'].argmax(1)))
    if specification.get('test', False):
        test_paths = data['test']
        started = time.perf_counter()
        prediction = pysiglib.sig_lasso_predict(test_paths, model, n_jobs=specification.get('jobs', 8))
        result['prediction_jobs'] = specification.get('jobs', 8)
        result['prediction_seconds'] = time.perf_counter() - started
        result['test_rmse'] = np.sqrt(np.mean((prediction - data['test_response'])**2, axis=0)).tolist()
        if specification['classification']:
            result['test_accuracy'] = float(np.mean(prediction.argmax(1) == data['test_response'].argmax(1)))
        if Path(specification['output']).stem.endswith('_measured_0'):
            model_path = Path(specification['output']).with_name(Path(specification['output']).stem + '_model.npz')
            metadata = {key: model[key] for key in ('words', 'feature_type', 'feature_convention', 'normalization', 'degree', 'dimension', 'time_aug', 'lead_lag', 'end_time')}
            np.savez_compressed(model_path, coef=model['coef'], intercept=model['intercept'], metadata=json.dumps(metadata))
            result['model_file'] = str(model_path)
    result['process_peak_bytes'] = process_peak_bytes()
    write_json(Path(specification['output']), result)


def prediction_worker(args):
    bind_native(args.native_dir)
    import pysiglib
    records, models = {}, {}
    native_hash = hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest()
    for mode in MODES:
        record = json.loads((args.prediction_worker / f'{mode}_measured_0.json').read_text())
        if record['status'] != 'ok' or record['native_sha256'] != native_hash:
            continue
        records[mode] = record
        saved = np.load(record['model_file'])
        model = json.loads(str(saved['metadata']))
        model['coef'], model['intercept'] = saved['coef'], saved['intercept']
        models[mode] = model
    data = np.load(next(iter(records.values()))['data'])
    paths, response = data['test'], data['test_response']
    for mode, model in models.items():
        pysiglib.sig_lasso_predict(paths, model, n_jobs=8)
    times, predictions = {mode: [] for mode in models}, {}
    for repeat in range(5):
        for mode in MODES[repeat % 3:] + MODES[:repeat % 3]:
            if mode in models:
                started = time.perf_counter()
                predictions[mode] = pysiglib.sig_lasso_predict(paths, models[mode], n_jobs=8)
                times[mode].append(time.perf_counter() - started)
    for mode, record in records.items():
        error = np.sqrt(np.mean((predictions[mode] - response)**2, axis=0))
        np.testing.assert_allclose(error, record['test_rmse'], atol=1e-11, rtol=1e-11)
        output = args.prediction_worker / f'{mode}_prediction.json'
        write_json(output, dict(status='ok', case=record['case'], degree=record['degree'], mode=mode,
            times=times[mode], median_seconds=float(np.median(times[mode])), iqr_seconds=float(np.subtract(*np.percentile(times[mode], [75, 25]))),
            process_peak_bytes=process_peak_bytes(), shared_process_peak=True, test_rmse=error.tolist(), prediction_protocol=3,
            native_sha256=native_hash))


def prediction_study(args):
    native_hash = hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest()
    pending = set()
    for path in sorted(args.output.glob('*/*_measured_0.json')):
        record = json.loads(path.read_text())
        if path.stem != record['mode'] + '_measured_0' or record['status'] != 'ok' or record['native_sha256'] != native_hash:
            continue
        output = path.with_name(record['mode'] + '_prediction.json')
        if output.exists():
            previous = json.loads(output.read_text())
            if previous.get('prediction_protocol') == 3 and previous.get('native_sha256') == native_hash:
                continue
        pending.add(path.parent)
    for folder in sorted(pending):
        command = [sys.executable, '-m', 'benchmarks.benchmark_lasso_features', '--prediction-worker', str(folder), '--native-dir', str(args.native_dir)]
        subprocess.run(command, check=True, timeout=args.timeout)
        print(folder.name, 'prediction complete', flush=True)


def serial_study(args):
    for seed in (7201, 7202, 7203):
        specification = prepare(args, 'baseline', seed)
        selected = {}
        for mode in MODES:
            record = json.loads((Path(specification['data']).parent / f'{mode}_measured_0.json').read_text())
            if record['status'] == 'ok':
                selected[mode] = record['alpha']
        for repeat in range(-1, args.repeats):
            for mode in MODES[repeat % 3:] + MODES[:repeat % 3]:
                if mode in selected:
                    run_job(args, specification, mode, f'serial_measured_{repeat}', selected[mode], jobs=1, test=repeat >= 0, warmup=repeat < 0)


def prepare(args, case, seed, degree=None):
    bind_native(args.native_dir)
    from benchmarks.lasso_feature_reference import dictionary
    folder = args.output / (f'{case}_{seed}' if degree is None else f'{case}_{degree}')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'prepared.json').exists():
        return json.loads((folder / 'prepared.json').read_text())
    started = time.perf_counter()
    if case == 'arabic':
        source = Path('sig_branch_and_bound_experiments/spoken_arabic_digit/results/normalized/integral_comparison/prepared.npz')
        saved = np.load(source)
        arrays = {name: np.array(saved[name], dtype=np.float64) for name in ('train', 'validation', 'test')}
        arrays['response'] = np.array(saved['response'], dtype=np.float64)
        classes = arrays['response'].shape[1]
        arrays['validation_response'] = np.eye(classes)[saved['labels'].astype(int)]
        arrays['test_response'] = np.eye(classes)[saved['test_labels'].astype(int)]
        provenance = dict(source=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    else:
        n, length, dimension, degree = CASES[case]
        rng = np.random.default_rng(seed)
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
        for mode in MODES:
            sources = []
            for name in ('train', 'validation', 'test'):
                words, design = dictionary(arrays[name], 3, mode)
                sources.append(design[:, [words.index(w) for w in source_words[mode]]])
            mean, scale = sources[0].mean(0), sources[0].std(0)
            combined = [((z - mean) / scale) @ np.array([1., -.8, .6]) for z in sources]
            target_scale = combined[0].std()
            targets.append([z / target_scale for z in combined])
        endpoint = []
        for name in ('train', 'validation', 'test'):
            words, z = dictionary(arrays[name], 1, 'sig')
            endpoint.append(z[:, 0] - z[:, 1])
        mean, scale = endpoint[0].mean(), endpoint[0].std()
        targets.insert(0, [(z - mean) / scale for z in endpoint])
        for j, name in enumerate(('response', 'validation_response', 'test_response')):
            response = np.column_stack([target[j] for target in targets])
            arrays[name] = response + rng.normal(scale=.05, size=response.shape)
        provenance = dict(seed=seed)
    np.savez(folder / 'data.npz', **arrays)
    specification = dict(data=str((folder / 'data.npz').resolve()), degree=degree, classification=case == 'arabic', case=case, seed=seed,
                         provenance=provenance, designs={})
    for mode in MODES:
        words, design = dictionary(arrays['train'], degree, mode)
        alpha_max = np.max(np.abs((design - design.mean(0)).T @ (arrays['response'] - arrays['response'].mean(0))), axis=0) / len(design)
        np.save(folder / f'{mode}.npy', design)
        write_json(folder / f'{mode}_words.json', words)
        specification['designs'][mode] = dict(design=str((folder / f'{mode}.npy').resolve()), words=str((folder / f'{mode}_words.json').resolve()), alpha_max=alpha_max.tolist())
    specification['preparation_seconds'] = time.perf_counter() - started
    write_json(folder / 'prepared.json', specification)
    return specification


def run_job(args, specification, mode, label, alpha, **extra):
    folder = Path(specification['data']).parent
    output = folder / f'{mode}_{label}.json'
    native_hash = hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest()
    if output.exists():
        previous = json.loads(output.read_text())
        complete = 'reported_dual_gap' in previous and (label != 'serial' or previous.get('prediction_jobs') == 1) and (label != 'measured_0' or 'model_file' in previous)
        if previous.get('protocol_version') == PROTOCOL_VERSION and previous.get('native_sha256') == native_hash and (previous['status'] == 'timeout' or (previous['status'] != 'error' and complete)):
            return previous
    job = {k: v for k, v in specification.items() if k != 'designs'}
    job.update(specification['designs'][mode], mode=mode, alpha=list(alpha), output=str(output.resolve()), **extra)
    job['native_sha256'] = native_hash
    job['protocol_version'] = PROTOCOL_VERSION
    job_path = folder / f'{mode}_{label}_job.json'
    write_json(job_path, job)
    command = [sys.executable, '-m', 'benchmarks.benchmark_lasso_features', '--worker', str(job_path), '--native-dir', str(args.native_dir)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        if completed.returncode:
            write_json(output, dict(job, status='error', stderr=completed.stderr[-12000:]))
    except subprocess.TimeoutExpired:
        write_json(output, dict(job, status='timeout'))
    result = json.loads(output.read_text())
    print(folder.name, mode, label, result['status'], result.get('fit_seconds'), flush=True)
    return result


def study(args):
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'environment.json', dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
        numpy=np.__version__, torch=version('torch'), threadpoolctl=version('threadpoolctl'),
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), native_sha256=hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest()))
    cases = [(name, seed, None) for name in args.cases for seed in ((7201, 7202, 7203) if name == 'baseline' else (7201,))]
    if args.arabic:
        cases.extend(('arabic', 0, degree) for degree in (1, 2, 3, 4))
    for case, seed, degree in cases:
        specification = prepare(args, case, seed, degree)
        selected = {}
        for mode in MODES:
            maximum = np.asarray(specification['designs'][mode]['alpha_max'])
            candidates = [run_job(args, specification, mode, f'ratio_{ratio}', np.maximum(1e-15, maximum * ratio), ratio=ratio) for ratio in RATIOS]
            valid = [r for r in candidates if r['status'] == 'ok']
            if not valid:
                continue
            if case == 'arabic':
                best = min(valid, key=lambda r: (-r['validation_accuracy'], np.mean(r['validation_mse']), -r['ratio']))
                selected[mode] = best['alpha']
            else:
                selected[mode] = [min(valid, key=lambda r: (round(r['validation_mse'][j], 12), -r['ratio']))['alpha'][j] for j in range(len(maximum))]
        for repeat in range(-1, args.repeats):
            for mode in MODES[repeat % 3:] + MODES[:repeat % 3]:
                if mode not in selected:
                    continue
                run_job(args, specification, mode, f'measured_{repeat}', selected[mode], test=repeat >= 0, warmup=repeat < 0)
        if case == 'baseline' and seed == 7201:
            for mode in MODES:
                if mode in selected:
                    run_job(args, specification, mode, 'serial', selected[mode], jobs=1, test=True)



def signature_regression(args):
    bind_native(args.native_dir)
    import pysiglib
    records = []
    for dimension, degree in ((8, 4), (24, 3)):
        rng = np.random.default_rng(7201)
        paths = rng.normal(size=(128, 13, dimension)).cumsum(1)
        variation = np.abs(np.diff(paths, axis=1)).max(2).sum(1)
        normalized = (paths - paths[:, :1]) / variation[:, None, None]
        response = pysiglib.sig_coef(normalized, [(0,), (0, 1), (0, 1, 2)]) @ np.array([1., 3., 12.])
        for repeat in range(-1, 15):
            started = time.perf_counter()
            model = pysiglib.sig_lasso(paths, response, degree, .002, n_jobs=8)
            seconds = time.perf_counter() - started
            if repeat >= 0:
                records.append(dict(dimension=dimension, degree=degree, seconds=seconds, objective=float(model['objective']),
                    gap=float(model['dual_gap']), words=model['words'], coef=model['coef'].tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(native=str(args.native_dir.resolve()), native_sha256=hashlib.sha256((args.native_dir / 'cpsig.dll').read_bytes()).hexdigest(), records=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('output/lasso_features'))
    parser.add_argument('--cases', nargs='*', choices=list(CASES), default=list(CASES))
    parser.add_argument('--arabic', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--signature-regression', action='store_true')
    parser.add_argument('--predictions-only', action='store_true')
    parser.add_argument('--serial-only', action='store_true')
    parser.add_argument('--prediction-worker', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.prediction_worker:
            prediction_worker(args)
        elif args.predictions_only:
            prediction_study(args)
        elif args.serial_only:
            serial_study(args)
        elif args.signature_regression:
            signature_regression(args)
        elif args.worker:
            worker(args)
        else:
            study(args)


if __name__ == '__main__':
    main()
