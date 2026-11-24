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

"""Compare bounded working-set searches on the frozen manuscript audio fits."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
PAPER = '02da5ef180616028bcf1533d859a4f5310848f6b'
FOLDERS = dict(arabic='spoken_arabic_digit', insect='insect_wingbeat', speech='speech_commands')
CASES = [f'{name}{k}' for name in FOLDERS for k in range(1, 5)] + ['dict_sig', 'dict_log']
POLICIES = {
    'current': {}, 'plain32': {}, 'legacy': {}, 'strongest1': dict(width=1, feedback=1),
    'top16': dict(width=16), 'top32': dict(width=32), 'top64': dict(width=64),
    'bins16': dict(width=16, bins=1, feedback=1), 'bins32': dict(width=32, bins=1, feedback=1),
    'bins64': dict(width=64, bins=1, feedback=1), 'first16': dict(width=16, bins=1, feedback=1, first=1),
    'half32': dict(width=32, bins=1, feedback=1, percent=50),
    'grow64': dict(width=64, bins=1, feedback=1, grow=1),
    'bins32_cache32': dict(width=32, bins=1, feedback=1, pool=32),
    'bins32_pool': dict(width=32, bins=1), 'raw32_feedback': dict(width=32, feedback=1),
    'grow_pool': dict(width=64, grow=1), 'grow_bins_pool': dict(width=64, grow=1, bins=1),
    'quarter_pool': dict(width=64, percent=25), 'top8': dict(width=8),
    'best32_128': dict(width=32, pool=128, unused=1), 'best32_256': dict(width=32, pool=256, unused=1),
    'best32_512': dict(width=32, pool=512, unused=1),
    'group32_256': dict(width=32, pool=256, unused=1, bins=1),
    'group32_512': dict(width=32, pool=512, unused=1, bins=1),
    'group16_256': dict(width=16, pool=256, unused=1, bins=1),
    'peer32': dict(width=32, peers=32), 'peer128': dict(width=32, peers=128),
    'peergroups': dict(width=32, bins=1, peers=32), 'legacytrace': dict(trace=1),
    'first1': dict(width=1, feedback=1, first=1, pool=1), 'all_violators': dict(width=256, all=1),
}
TOL = 1e-8


def paper_file(name):
    return json.loads(subprocess.check_output(['git', 'show', f'{PAPER}:benchmarks/results/prefix_suffix/{name}'], cwd=ROOT))


def verify_data(root, protocol):
    local = json.loads((root / 'results/optimized_degree4_20261028/protocol.json').read_text())['datasets']
    for name, folder in FOLDERS.items():
        directory = root / folder / 'results/normalized/integral_comparison'
        ids = json.loads((directory / 'sample_ids.json').read_text())
        keys = ('training_indices', 'tuning_indices', 'test_ids') if name == 'arabic' else ('train_ids', 'validation_ids', 'test_ids')
        canonical = dict(zip(('train', 'validation', 'test'), (ids[k] for k in keys)))
        digest = hashlib.sha256((json.dumps(canonical, indent=2) + '\n').encode()).hexdigest()
        assert digest == protocol['datasets'][name]['sample_ids_sha256'], name
        with (directory / 'prepared.npz').open('rb') as file:
            assert hashlib.file_digest(file, 'sha256').hexdigest() == local[name]['prepared_sha256']


def case_info(case):
    if case.startswith('dict_'):
        return 'arabic', 4, case[5:]
    return case[:-1], int(case[-1]), 'sig'


def worker(args):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits
    package_root = args.out / (args.policy + '_package') if args.policy in ('current', 'plain32') else ROOT
    sys.path.insert(0, str(package_root))
    sys.modules['pysiglib_cuda'] = None
    import pysiglib
    from pysiglib.load_siglib import CPSIG
    assert Path(CPSIG._name).resolve().parent == package_root / 'pysiglib'
    proc = psutil.Process()

    def peak():
        usage = proc.memory_info()
        if hasattr(usage, 'peak_wset'):
            return usage.peak_wset
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)

    protocol = paper_file('protocol.json')
    print('ready', flush=True)
    with threadpool_limits(1):
        data_id, train, y = None, None, None
        for line in sys.stdin:
            message = json.loads(line)
            case, save = message['case'], message['save']
            dataset, degree, feature = case_info(case)
            prepared = args.data / FOLDERS[dataset] / 'results/normalized/integral_comparison/prepared.npz'
            if data_id != dataset:
                train = y = None
                with np.load(prepared) as z:
                    train, y = z['train'], z['response']
                data_id = dataset
            choice = (protocol['dictionary_comparison'][feature] if case.startswith('dict_')
                      else protocol['datasets'][dataset]['choices'][str(degree)])
            alpha = np.asarray(choice['alpha'])
            fit = pysiglib.sig_lasso if feature == 'sig' else pysiglib.log_sig_lasso
            before = peak()
            start = perf_counter()
            model = fit(train, y, degree, alpha, n_jobs=8, tol=TOL, max_iter=3000)
            seconds = perf_counter() - start
            after = peak()
            st = model['stats']
            X, beta, bias = model['signatures'], model['coef'], model['intercept']
            fitted = X @ beta + bias
            residual = y - fitted
            corr = X.T @ residual / len(y)
            active = beta != 0
            defect = np.where(active, np.abs(corr - alpha * np.sign(beta)), np.maximum(0., abs(corr) - alpha)).max(0, initial=0.)
            primal = .5 * (residual**2).mean(0) + alpha * abs(beta).sum(0)
            ok = (np.all(model['converged']) and np.all(defect <= np.minimum(TOL, 1e-4 * alpha) + 1e-13)
                  and np.max(abs(residual.mean(0))) <= TOL and np.allclose(primal, model['objective'], atol=2e-12, rtol=0)
                  and st['engine'] == 'tree' and st['full_transform_calls'] == st['numerical_failures'] == 0)
            record = dict(case=case, policy=args.policy, seconds=seconds, rss_mib=(after-before)/2**20, peak_mib=after/2**20,
                          native_mib=st['peak_bytes']/2**20, support=len(model['words']), nnz=int(active.sum()),
                          updates=int(np.sum(model['n_iter'])), passes=st['search_passes'], activated=st['activations'],
                          evaluated=st['evaluated'], terminal=st['terminal_correlations'], search_s=st['search_seconds'],
                          solve_s=st['solve_seconds'], cert_s=st['certification_seconds'], ok=bool(ok),
                          kkt=float(max(model['kkt_violation'])), selected_kkt=float(max(defect)),
                          gap=float(max(model['dual_gap'])), objective=primal.tolist())
            if save:
                with np.load(prepared) as z:
                    labels, test = z['test_labels'], z['test']
                scores = np.concatenate([pysiglib.sig_lasso_predict(test[i:i+64], model, n_jobs=8) for i in range(0, len(test), 64)])
                record.update(correct=int(np.sum(np.argmax(scores, axis=1)==labels)), n=len(test))
                dest = args.out / f'{args.policy}-{case}.npz'
                np.savez(dest, fitted=fitted, pred=scores, words=[' '.join(map(str, w)) for w in model['words']],
                         coef=beta, bias=bias, alpha=alpha, nnz=active.sum(0), kkt=model['kkt_violation'], gap=model['dual_gap'])
                del scores, test
            if args.policy == 'legacytrace':
                counts = (ctypes.c_uint64 * 8)()
                CPSIG.study_search_counts(counts)
                record['counts'] = list(counts)
            print(json.dumps(record, allow_nan=False), flush=True)
            del model, X, beta, fitted


def launch(args, policy):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYSIGLIB_STUDY_')}
    env.update({f'PYSIGLIB_STUDY_{k.upper()}': str(v) for k, v in POLICIES[policy].items()})
    env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    p = subprocess.Popen([sys.executable, __file__, '--worker', '--policy', policy, '--data', str(args.data),
                          '--out', str(args.out), '--control', str(args.control)], env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != 'ready':
        p.stdin.close()
        raise RuntimeError(f'{policy} worker failed at startup (exit {p.wait()})')
    return p


def compare(args):
    policies, cases = args.policies.split(','), args.cases.split(',')
    assert set(policies) <= POLICIES.keys() and set(cases) <= set(CASES)
    args.out.mkdir(parents=True, exist_ok=True)
    verify_data(args.data, paper_file('protocol.json'))
    records = args.out / 'fits.jsonl'
    rng = random.Random(args.seed)
    if records.exists():
        raise FileExistsError(records)
    for policy, binary in (('current', args.control), ('plain32', args.plain)):
        if policy not in policies:
            continue
        if binary is None or not binary.is_file():
            raise FileNotFoundError(f'Set the path to the clean {policy} library')
        pkg = args.out / (policy + '_package') / 'pysiglib'
        shutil.copytree(ROOT / 'pysiglib', pkg, ignore=shutil.ignore_patterns('__pycache__', '*.dll', '*.so', '*.dylib'))
        shutil.copy2(binary, pkg / binary.name)
        if (ROOT / 'pysiglib/tbb12.dll').exists():
            shutil.copy2(ROOT / 'pysiglib/tbb12.dll', pkg)
    workers = {}
    try:
        with records.open('x', encoding='ascii') as file:
            for case in cases:
                if args.fresh or not workers:
                    for policy in policies:
                        workers[policy] = launch(args, policy)
                order = list(workers)
                rng.shuffle(order)
                for rep in range(-1, args.reps):
                    if rep >= 0:
                        order = order[1:] + order[:1]
                    for policy in order:
                        p = workers[policy]
                        p.stdin.write(json.dumps(dict(case=case, save=args.save and rep == -1)) + '\n')
                        p.stdin.flush()
                        line = p.stdout.readline()
                        if not line:
                            raise RuntimeError(f'{policy} failed on {case} (exit {p.poll()})')
                        record = json.loads(line)
                        record.update(rep=rep, fresh=bool(args.fresh and rep == -1))
                        file.write(json.dumps(record) + '\n')
                        file.flush()
                print(case, 'complete', flush=True)
                if args.fresh:
                    for p in workers.values():
                        p.stdin.close()
                        p.wait()
                    workers.clear()
    finally:
        for p in workers.values():
            p.stdin.close()
            p.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control', type=Path, default=ROOT / 'out/control/cpsig.dll')
    parser.add_argument('--plain', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--policy', choices=POLICIES)
    parser.add_argument('--policies', default='current,top16,bins16,first16,half32,grow64')
    parser.add_argument('--cases', default=','.join(CASES))
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7231)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    args.data, args.out = args.data.resolve(), args.out.resolve()
    worker(args) if args.worker else compare(args)


if __name__ == '__main__':
    main()
