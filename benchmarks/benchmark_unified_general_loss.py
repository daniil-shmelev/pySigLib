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

"""Benchmark exact general-loss fits. Run as a module from the repository root."""

import argparse
import csv
import gc
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lookahead_search import workload
from benchmarks.benchmark_lasso_violation_batches import intercept_only
from benchmarks.general_loss_working_set import _fit, general_loss_fit
from benchmarks.lasso_feature_reference import dictionary, shuffle


def make_loss(y, name):
    n = len(y)

    def evaluate(z):
        if name == 'logistic':
            return (np.logaddexp(0,z)-y*z).mean(), (expit(z)-y)/n
        if name == 'poisson':
            mu = np.exp(z)
            return (mu-y*z).mean(), (mu-y)/n
        e = z-y
        if name == 'huber':
            g = np.clip(e,-1,1)
            return (g*(e-.5*g)).mean(), g/n
        value, g = .5*(e@e)/n, e/n
        if name == 'coupled':
            delta = np.diff(e)
            value += (delta@delta)/n
            g[:-1] -= 2*delta/n
            g[1:] += 2*delta/n
        return value,g

    return evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--case', choices=['synthetic','anisotropic','arabic','arabic_validation','arabic_unseen','commutator'], default='synthetic')
    p.add_argument('--degree', type=int, default=3)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--digit', type=int, default=4)
    p.add_argument('--modes', nargs='+', default=['sig','log','hall'])
    p.add_argument('--losses', nargs='+', default=['squared','logistic','huber','poisson','coupled'])
    p.add_argument('--ratios', nargs='+', type=float, default=[.03])
    p.add_argument('--methods', nargs='+', default=['auto:adaptive','packed:top16','packed:global','tree:lookahead'])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--cache-mib', type=int, default=16)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.case in ('arabic_validation','arabic_unseen'):
        folder = Path('sig_branch_and_bound_experiments/spoken_arabic_digit/results/normalized/integral_comparison')
        with np.load(folder/'prepared.npz') as saved:
            if args.case == 'arabic_validation':
                paths = saved['validation'].astype(np.float64)
                y = (saved['labels'] == args.digit).astype(np.float64)
            else:
                indices = np.sort(rng.choice(len(saved['test_labels']),args.samples,replace=False))
                paths = saved['test'][indices].astype(np.float64)
                y = (saved['test_labels'][indices] == args.digit).astype(np.float64)
        responses = {key:y for key in ('squared','logistic','huber','poisson')}
    elif args.case == 'commutator':
        if args.degree != 4 or args.channels < 4:
            raise ValueError('The commutator workload uses degree four and at least four channels')
        generators = rng.normal(size=(args.samples,4,args.channels))
        def bracket(a,b):
            return np.concatenate((a,b,-a[:,::-1],-b[:,::-1]),axis=1)
        dx = bracket(bracket(generators[:,:1],generators[:,1:2]),bracket(generators[:,2:3],generators[:,3:4]))
        paths = np.concatenate((np.zeros_like(dx[:,:1]),dx.cumsum(axis=1)),axis=1)
        w,T = dictionary(paths,4,'sig')
        signal = T[:,w.index((0,1,2,3))]-.7*T[:,w.index((0,2,1,3))]
        signal = (signal-signal.mean())/signal.std()
        y = signal+.1*rng.normal(size=len(signal))
        responses = dict(squared=y,huber=y,logistic=rng.binomial(1,expit(signal)),
                         poisson=rng.poisson(np.exp(.3*signal)))
        del T
    else:
        paths, responses = workload(args.case,args.seed,args.samples,args.channels,args.degree,args.digit)
    paths = np.ascontiguousarray(paths)
    responses['coupled'] = responses['squared']
    with threadpool_limits(1), args.output.open('x',newline='') as out:
        writer = None
        for mode in args.modes:
            words,X = dictionary(paths,args.degree,mode)
            ids = {w:j for j,w in enumerate(words)}
            for name in args.losses:
                y = responses[name]
                loss = make_loss(y,name)
                _,g0 = loss(np.full(len(y),intercept_only(y,'squared' if name == 'coupled' else name)))
                for ratio in args.ratios:
                    alpha = ratio * max(abs(g0 @ X))
                    reference = None
                    for rep in range(-1,args.repeats):
                        for method in rng.permutation(args.methods):
                            engine,admission = method.split(':')
                            shuffle.cache_clear()
                            if rep < 0:
                                gc.collect()
                                tracemalloc.start()
                            try:
                                start = perf_counter()
                                if method == 'auto:adaptive':
                                    m = general_loss_fit(paths,args.degree,mode,loss,alpha,jobs=args.jobs)
                                else:
                                    m = _fit(paths,args.degree,mode,loss,alpha,args.jobs,1e-7,engine,admission,args.cache_mib << 20)
                                ms = (perf_counter()-start)*1000
                                peak = tracemalloc.get_traced_memory()[1] if rep < 0 else 0
                            finally:
                                if rep < 0:
                                    tracemalloc.stop()
                            support = [ids[w] for w in m['words']]
                            eta = m['intercept'] + X[:,support] @ m['coef']
                            value,g = loss(eta)
                            corr = g @ X
                            error = np.maximum(abs(corr)-alpha,0)
                            error[support] = abs(corr[support]+alpha*np.sign(m['coef']))
                            kkt = max(abs(g.sum()),max(error))
                            obj = value+alpha*abs(m['coef']).sum()
                            if reference is None:
                                reference = obj,eta,set(support)
                            diff = abs(obj-reference[0])
                            if not (np.isfinite(kkt) and kkt <= 1.2e-7 and diff < 2e-7*max(1.,abs(obj))):
                                raise ArithmeticError((mode,name,ratio,method,kkt,diff))
                            native = m['stats'].peak_bytes
                            row = dict(case=args.case,seed=args.seed,digit=args.digit,n=len(y),d=paths.shape[-1],
                                       degree=args.degree,mode=mode,loss=name,ratio=ratio,method=method,
                                       engine=m['engine'],rep=rep,ms=ms if rep>=0 else None,
                                       workspace_mib=(peak+native)/2**20 if rep<0 else None,
                                       native_mib=native/2**20,scans=m['scans'],admitted=m['admitted'],
                                       peak_set=m['peak_set'],nnz=len(support),kkt=kkt,obj=obj,obj_diff=diff,
                                       pred_diff=max(abs(eta-reference[1])),support_diff=len(set(support)^reference[2]))
                            if writer is None:
                                writer = csv.DictWriter(out,fieldnames=row)
                                writer.writeheader()
                            writer.writerow(row)
                            out.flush()


if __name__ == '__main__':
    main()
