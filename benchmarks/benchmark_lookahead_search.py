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

"""Run with python -m benchmarks.benchmark_lookahead_search after building the bridge."""

import argparse
import csv
import ctypes as ct
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_lasso_violation_batches import fit, intercept_only, residual, select_candidates
from benchmarks.lasso_feature_reference import dictionary
from benchmarks._prototype_lasso_types import LassoOptions, LassoStats


SETTINGS = ('plain', 'grouped', 'signed', 'grouped_signed')
MODES = ('sig', 'log', 'hall')
TOL = 1e-7


def workload(case, seed, samples, channels, degree, digit):
    if case == 'arabic':
        source = Path('sig_branch_and_bound_experiments/spoken_arabic_digit/results/normalized/integral_comparison')
        with np.load(source / 'prepared.npz') as saved:
            paths = saved['train'].astype(np.float64)
            y = (saved['response'].argmax(axis=1) == digit).astype(float)
        return paths, {k: y for k in ('squared', 'logistic', 'poisson', 'huber')}
    rng = np.random.default_rng(seed)
    dx = rng.normal(size=(samples, 12, channels))
    if case == 'anisotropic':
        dx[:, :, 3:] *= .2
    paths = np.concatenate((np.zeros((samples, 1, channels)), dx.cumsum(axis=1)), axis=1)
    words, X = dictionary(paths, degree, 'sig')
    Z = X[:, [words.index(w) for w in [(0,), (1, 2), (2, 1, 0)]]]
    signal = ((Z - Z.mean(axis=0)) / Z.std(axis=0)) @ np.array([.9, -.7, .45])
    binary = rng.binomial(1, expit(signal))
    counts = rng.poisson(np.exp(.45 * signal))
    continuous = signal + .5 * rng.standard_t(3, samples)
    return paths, dict(squared=continuous, logistic=binary, poisson=counts, huber=continuous)


def trial(lib, paths, degree, mode, setting, X, codes, order, y, loss, lam, jobs, policy, bound, verify, reference_trace):
    options = LassoOptions(*paths.shape, degree, 1, 3000, len(codes), 256 << 20, 4 << 30,
                           TOL, jobs, 0, 1, 0, SETTINGS.index(setting), bound == 'exact')
    out = np.empty_like(codes)
    stats = LassoStats()
    trace = []
    search_ms = 0.
    start = perf_counter()
    handle = lib.lookahead_open(ct.byref(options), paths.ctypes.data, MODES.index(mode))
    if not handle:
        raise RuntimeError('Native search preparation failed')

    def scan(X, unused, r, eligible, lam, tol, policy, active, norm2, state):
        nonlocal search_ms
        tick = perf_counter()
        excluded = codes[~eligible]
        capacity = len(out) if policy == 'lookahead' else min(16, len(out))
        size = lib.lookahead_query(handle, r.ctypes.data, lam + tol, excluded.ctypes.data,
                                   len(excluded), out.ctypes.data, capacity)
        search_ms += 1000 * (perf_counter() - tick)
        if size < 0:
            raise RuntimeError('Native feature search failed')
        ids = np.sort(order[np.searchsorted(codes, out[:size], sorter=order)])
        if verify:
            scores = abs(r @ X / len(r))
            expected = np.flatnonzero(eligible & (scores > lam + tol))
            if policy == 'top16':
                expected = np.sort(expected[np.argsort(-scores[expected], kind='stable')[:16]])
            np.testing.assert_array_equal(ids, expected)
        selected, signs = select_candidates(X, r, ids, lam, tol, policy, active, norm2, state)
        trace.append((ids.tobytes(), selected.tobytes()))
        return selected, signs

    try:
        b, beta, scans, admitted, batch = fit(X, None, y, loss, lam, policy, TOL, search=scan)
    finally:
        lib.lookahead_close(handle, ct.byref(stats))
    elapsed = 1000 * (perf_counter() - start)
    if reference_trace is not None and trace != reference_trace:
        raise AssertionError('Different violation sets or lookahead batches')
    eta = b + X @ beta
    r = residual(y, eta, loss)
    corr = r @ X / len(y)
    kkt = max(abs(r.mean()), np.where(beta == 0, np.maximum(abs(corr) - lam, 0),
                                      abs(corr - lam * np.sign(beta))).max())
    if not np.isfinite(kkt) or kkt > TOL:
        raise AssertionError(f'Full KKT error {kkt:g}')
    return dict(fit_ms=elapsed, search_ms=search_ms, native_mib=stats.peak_bytes / 2**20,
                scans=scans, admitted=admitted, max_batch=batch,
                max_returned=max(len(ids) // np.dtype(np.intp).itemsize for ids, _ in trace),
                nnz=np.count_nonzero(beta), kkt=kkt,
                blocks=stats.block_queries, terminal=stats.terminal_queries, evaluated=stats.evaluated), trace, beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, default=Path('output/lookahead_search/bridge.dll'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', choices=('synthetic', 'anisotropic', 'arabic'), default='synthetic')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--losses', nargs='+', choices=('squared', 'logistic', 'poisson', 'huber'),
                        default=['squared', 'logistic', 'poisson', 'huber'])
    parser.add_argument('--ratios', nargs='+', type=float, default=[.12, .03])
    parser.add_argument('--policy', choices=('lookahead', 'top16'), default='lookahead')
    parser.add_argument('--bound', choices=('exact', 'approximate'), default='exact')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7201)
    parser.add_argument('--digit', type=int, default=4)
    parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--degree', type=int, choices=(3, 4), default=3)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.channels < 3 or args.repeats < 1 or any(not 0 < r < 1 for r in args.ratios):
        parser.error('Use at least three channels, one repetition and ratios in (0, 1).')
    with os.add_dll_directory(str(Path('out/build/windows-cpu/siglib/cpsig/Release').resolve())):
        lib = ct.CDLL(str(args.dll.resolve()))
    lib.lookahead_open.argtypes = [ct.POINTER(LassoOptions), ct.c_void_p, ct.c_uint32]
    lib.lookahead_open.restype = ct.c_void_p
    lib.lookahead_query.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_void_p,
                                    ct.c_size_t, ct.c_void_p, ct.c_size_t]
    lib.lookahead_query.restype = ct.c_int64
    lib.lookahead_close.argtypes = [ct.c_void_p, ct.POINTER(LassoStats)]
    lib.lookahead_close.restype = None
    with threadpool_limits(limits=1), args.output.open('x', newline='') as stream:
        paths, responses = workload(args.case, args.seed, args.samples, args.channels, args.degree, args.digit)
        paths = np.ascontiguousarray(paths)
        writer = None
        for mode in args.modes:
            words, X = dictionary(paths, args.degree, mode)
            X = np.asfortranarray(X - X.mean(axis=0))
            codes = np.zeros(len(words), dtype=np.uint64)
            for j, w in enumerate(words):
                for a in w:
                    codes[j] = codes[j] * paths.shape[2] + a + 1
            order = np.argsort(codes)
            for loss in args.losses:
                y = responses[loss]
                r = residual(y, intercept_only(y, loss), loss)
                for ratio in args.ratios:
                    lam = ratio * abs(r @ X / len(y)).max()
                    trace = baseline = None
                    for rep in range(args.repeats + 1):
                        for setting in np.roll(SETTINGS, rep % 4):
                            row, observed, beta = trial(lib, paths, args.degree, mode, setting, X, codes, order, y, loss,
                                                        lam, args.jobs, args.policy, args.bound, rep == 0, trace)
                            if trace is None:
                                trace, baseline = observed, beta
                            np.testing.assert_array_equal(beta, baseline)
                            if rep:
                                row = dict(case=args.case, mode=mode, loss=loss, ratio=ratio, setting=setting,
                                           rep=rep, seed=args.seed, digit=args.digit, jobs=args.jobs,
                                           n=len(y), d=paths.shape[2], degree=args.degree,
                                           policy=args.policy, bound=args.bound, p=len(codes), **row)
                                if writer is None:
                                    writer = csv.DictWriter(stream, fieldnames=row)
                                    writer.writeheader()
                                writer.writerow(row)
                                stream.flush()


if __name__ == '__main__':
    main()
