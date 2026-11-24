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

"""Independent tensor-coordinate, stationarity, and score-tie checks for the search study."""

import argparse
import json
import os
from itertools import product
from math import factorial
from pathlib import Path
import subprocess
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from benchmark_search_policy import CASES, FOLDERS, ROOT, TOL, case_info, paper_file


def normalize(x):
    variation = np.abs(np.diff(x, axis=1)).max(2).sum(1)
    return np.ascontiguousarray((x - x[:, :1]) / np.where(variation, variation, 1)[:, None, None])


def indices(word, d):
    code = 0
    for a in word:
        code = d * code + a + 1
    return code - 1


def tensor_coordinates(sig, words, d, log):
    # The homogeneous tensor logarithm is the signed sum over consecutive cuts.
    n, k = len(sig), len(words[0])
    ix = np.array([indices(w, d) for w in words])
    result = sig[:, ix].copy()
    if log:
        for mask in range(1, 1 << (k - 1)):
            cuts = [0] + [j for j in range(1, k) if mask & (1 << (j - 1))] + [k]
            term = np.ones((n, len(words)))
            for a, b in zip(cuts, cuts[1:]):
                term *= sig[:, [indices(w[a:b], d) for w in words]]
            result += (-1)**(len(cuts) - 2) * term / (len(cuts) - 1)
    return result * factorial(k)


def selected(x, words, log, lib):
    result = np.empty((len(x), len(words)))
    for k in sorted(set(map(len, words))):
        ids = np.array([i for i, w in enumerate(words) if len(w) == k])
        for start in range(0, len(ids), 96):
            batch = ids[start:start+96]
            ws = [words[i] for i in batch]
            if not log:
                result[:, batch] = lib.sig_coef(x, ws, n_jobs=8) * factorial(k)
                continue
            pieces = sorted({w[a:b] for w in ws for a in range(k) for b in range(a+1, k+1)})
            sig = lib.sig_coef(x, pieces, n_jobs=8)
            lookup = {w: i for i, w in enumerate(pieces)}
            values = sig[:, [lookup[w] for w in ws]].copy()
            for mask in range(1, 1 << (k-1)):
                cuts = [0] + [j for j in range(1, k) if mask & (1 << (j-1))] + [k]
                term = np.ones_like(values)
                for a, b in zip(cuts, cuts[1:]):
                    term *= sig[:, [lookup[w[a:b]] for w in ws]]
                values += (-1)**(len(cuts)-2) * term / (len(cuts)-1)
            result[:, batch] = factorial(k) * values
    return result


def audit(args, lib):
    protocol = paper_file('protocol.json')
    paper = paper_file('paper_rerun/measurements.json')
    published = {f"{r['dataset']}{r['degree']}": r['tree'] for r in paper['classification']}
    published.update({f'dict_{k}': v[0] for k, v in paper['dictionary_comparison'].items()})
    records = []
    for case in CASES:
        dataset, k, mode = case_info(case)
        log = mode == 'log'
        with np.load(args.data / FOLDERS[dataset] / 'results/normalized/integral_comparison/prepared.npz') as data:
            x, test, y, labels = normalize(data['train']), normalize(data['test']), data['response'], data['test_labels']
        n, _, d = x.shape
        alpha = np.asarray((protocol['dictionary_comparison'][mode] if case.startswith('dict_')
                            else protocol['datasets'][dataset]['choices'][str(k)])['alpha'])
        policies = args.policies.split(',')
        residuals, defects, row = [], [], []
        previous_words = None
        for policy in policies:
            with np.load(args.models / f'{policy}-{case}.npz') as m:
                ws = [tuple(map(int, s.split())) for s in m['words']]
                np.testing.assert_array_equal(alpha, m['alpha'])
                if ws != previous_words:
                    design = selected(x, ws, log, lib)
                    test_design = selected(test, ws, log, lib)
                    previous_words = ws
                train = design @ m['coef'] + m['bias']
                preds = test_design @ m['coef'] + m['bias']
                train_err, pred_err = np.max(abs(train - m['fitted'])), np.max(abs(preds - m['pred']))
                assert max(train_err, pred_err) < 1e-9
                resid = y - train
                corr = design.T @ resid / n
                beta = m['coef']
                error = np.where(beta != 0., abs(corr - alpha*np.sign(beta)), np.maximum(abs(corr)-alpha, 0)).max(0, initial=0.)
                assert np.all(error <= np.minimum(TOL, 1e-4*alpha) + 1e-13)
                assert abs(resid.mean(0)).max() < TOL
                primal = .5*(resid**2).mean(0) + alpha*abs(beta).sum(0)
                paper_delta = float(max(abs(primal - published[case]['objective'])))
                assert paper_delta < 1e-9
                support = {(ws[i], j) for i, j in zip(*np.nonzero(beta))}
                if policy == policies[0]:
                    base_train, base_preds, base_words, base_objective = train, preds, set(ws), primal
                    base_support = support
                item = dict(case=case, policy=policy, test_correct=int((preds.argmax(1)==labels).sum()), test_n=len(test),
                            labels_changed=int((preds.argmax(1)!=base_preds.argmax(1)).sum()),
                            changed_words=len(set(ws)^base_words), changed_coefficients=len(support ^ base_support),
                            nnz=len(support), union=len(ws),
                            support_by_degree=[sum(len(w)==j for w in ws) for j in range(1,k+1)],
                            fit_score_delta=float(abs(train-base_train).max()), test_score_delta=float(abs(preds-base_preds).max()),
                            objective_delta=float(max(abs(primal-base_objective))), paper_objective_delta=paper_delta,
                            paper_correct=published[case]['correct'], paper_words=published[case]['support'],
                            training_check=float(train_err), test_check=float(pred_err), active_kkt=float(max(error)),
                            intercept_kkt=float(abs(resid.mean(0)).max()), reported_kkt=float(max(m['kkt'])))
                residuals.append(resid)
                defects.append(error)
                row.append(item)
        p = sum(d**j for j in range(1,k+1))
        if 8*n*p <= 4 << 30:
            rhs = np.concatenate(residuals, axis=1)/n
            width = max(1, min(n, (24 << 20)//(8*p)))
            if log:
                groups = [[w for w in product(range(d), repeat=j) if all(w < w[i:] for i in range(1,j))]
                          for j in range(1,k+1)]
                count = sum(map(len, groups))
            else:
                scales = np.repeat([factorial(j) for j in range(1,k+1)], [d**j for j in range(1,k+1)])
                count = p
            corr = np.zeros((count, rhs.shape[1]))
            for start in range(0,n,width):
                sig = lib.sig(x[start:start+width], k, n_jobs=8)
                if log:
                    full = np.concatenate([tensor_coordinates(sig, g, d, True) for g in groups], axis=1)
                else:
                    full = sig
                    full *= scales
                corr += full.T @ rhs[start:start+width]
            for scores, err, item in zip(np.hsplit(corr,len(row)), defects, row):
                error = np.maximum(np.abs(scores).max(0)-alpha, err)
                assert np.all(error <= np.minimum(TOL, 1e-4*alpha)+1e-13), (case,item['policy'],error)
                item.update(audited_words=count, exhaustive_feature_kkt=float(max(error)))
        records.extend(row)
        print(case, 'checked', flush=True)
    (args.models / 'audit.json').write_text(json.dumps(records,indent=2)+'\n', encoding='ascii')


def ties(policy, lib):
    n = 64
    x = np.array([[1. if (i & a).bit_count()%2 == 0 else -1. for a in range(1,6)] for i in range(n)])
    examples = {'duplicates': (x[:, [0]*5+[1,2,3,4]], x @ [10.,9.,8.,7.,6.]),
                'orthogonal': (x.copy(), x @ np.full(5,10.))}
    results = []
    for name,(matrix,y) in examples.items():
        paths = np.stack([np.zeros_like(matrix),matrix],axis=1)
        first = lib.sig_lasso(paths,y,1,1.,tol=TOL,max_iter=1)
        final = lib.sig_lasso(paths,y,1,1.,tol=TOL,max_iter=100)
        expected = x @ (np.array([10.,9.,8.,7.,6.])-1 if name=='duplicates' else np.full(5,9.))
        np.testing.assert_allclose(final['signatures'] @ final['coef'] + final['intercept'], expected, rtol=0,atol=1e-9)
        assert np.all(final['converged'])
        results.append(dict(example=name, policy=policy, first_added=first['stats']['activations'],
                            first_nonzero=len(first['words']), first_words=first['words'], updates=int(final['n_iter'])))
    print(json.dumps(results))


def losses(args, lib):
    rng = np.random.default_rng(24917)
    n, d, k = 96, 3, 3
    paths = np.concatenate([np.zeros((n,1,d)), rng.normal(size=(n,6,d)).cumsum(1)], axis=1)
    x = normalize(paths)
    full = lib.sig(x,k,n_jobs=1)
    target = 2*x[:,-1,0] - 1.3*x[:,-1,1]
    ys = {'squared': target+rng.normal(0,.1,n), 'huber': target+rng.standard_t(2,n)*.3,
          'logistic': rng.binomial(1,1/(1+np.exp(-target))).astype(float),
          'poisson': rng.poisson(np.exp(target)).astype(float)}
    rows = []
    for mode in ('sig','log'):
        groups = [[w for w in product(range(d),repeat=j) if mode=='sig' or all(w<w[i:] for i in range(1,j))]
                  for j in range(1,k+1)]
        words = [w for g in groups for w in g]
        A = np.concatenate([tensor_coordinates(full,g,d,mode=='log') for g in groups],axis=1)
        fit = lib.sig_lasso if mode=='sig' else lib.log_sig_lasso
        for loss,y in ys.items():
            alpha = .1*abs(A.T @ (y-y.mean())/n).max()
            m = fit(paths,y,k,alpha,loss=loss,tol=TOL,max_iter=3000,n_jobs=1)
            b = np.zeros(len(words))
            for w,v in zip(m['words'],m['coef']):
                b[words.index(w)] = v
            eta = A @ b + m['intercept']
            r = {'squared':lambda: y-eta,'huber':lambda: np.clip(y-eta,-1,1),
                 'logistic':lambda: y-1/(1+np.exp(-eta)),'poisson':lambda: y-np.exp(eta)}[loss]()
            corr = A.T @ r/n
            defect = np.where(b!=0,abs(corr-alpha*np.sign(b)),np.maximum(abs(corr)-alpha,0)).max()
            assert np.all(m['converged']) and defect <= min(TOL,alpha*1e-4)+1e-12 and abs(r.mean())<=TOL
            rows.append(dict(mode=mode,loss=loss,kkt=float(defect),intercept=float(abs(r.mean()))))
    (args.models/'losses.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--policies', default='current,top32')
    parser.add_argument('--ties', action='store_true')
    parser.add_argument('--losses', action='store_true')
    parser.add_argument('--tie-worker', choices=('raw','groups'))
    args = parser.parse_args()
    if args.ties:
        results = []
        for name in ('raw','groups'):
            env = dict(os.environ, PYSIGLIB_STUDY_WIDTH='5', PYSIGLIB_STUDY_POOL='5',
                       PYSIGLIB_STUDY_BINS=str(int(name=='groups')), PYSIGLIB_STUDY_FEEDBACK='1')
            output = subprocess.check_output([sys.executable,__file__,'--models',str(args.models),'--tie-worker',name],env=env)
            results.extend(json.loads(output))
        (args.models/'ties.json').write_text(json.dumps(results,indent=2)+'\n',encoding='ascii')
        return
    sys.path.insert(0,str(args.models.resolve()/'plain32_package' if args.losses else ROOT))
    sys.modules['pysiglib_cuda'] = None
    import pysiglib
    with threadpool_limits(1):
        if args.tie_worker:
            ties(args.tie_worker,pysiglib)
        elif args.losses:
            losses(args,pysiglib)
        else:
            audit(args,pysiglib)


if __name__ == '__main__':
    main()
