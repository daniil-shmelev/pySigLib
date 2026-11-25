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
"""Explore expected signatures of random flights. Run as a module from the repo root."""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from math import comb, factorial

import numpy as np
from scipy.special import hyp1f1


def flights(counts, rng, min_knots=0):
    """Return (time, position) at the exact switches, with independent fair initial signs."""
    counts = np.asarray(counts)
    knots = max(int(counts.max()), min_knots)
    paths = np.zeros((len(counts), knots + 2, 2))
    signs = rng.choice([-1., 1.], size=len(counts))
    for count in np.unique(counts):
        rows = np.flatnonzero(counts == count)
        size = max(count, min_knots)
        times = rng.random((len(rows), size))
        order = np.argsort(times, axis=1)
        times = np.take_along_axis(times, order, axis=1)
        bounds = np.column_stack((np.zeros(len(rows)), times, np.ones(len(rows))))
        switches = np.column_stack((np.zeros(len(rows), dtype=int), order < count))
        velocity = signs[rows, None] * (1 - 2 * (np.cumsum(switches, axis=1) % 2))
        positions = np.cumsum(np.diff(bounds, axis=1) * velocity, axis=1)
        paths[rows, :size + 2, 0] = bounds
        paths[rows, 1:size + 2, 1] = positions
        paths[rows, size + 2:, :] = paths[rows, size + 1:size + 2, :]
    return paths


def sample(case, per_class, seed):
    rng = np.random.default_rng(seed)
    labels = np.repeat([-1., 1.], per_class)
    if case == 'fixed':
        return flights(np.where(labels > 0, 2, 1), rng, min_knots=2), labels
    if case == 'mixture':
        pairs = rng.choice([1, 2, 4], size=len(labels), p=[0.25, 0.5, 0.25])
        return flights(2 * pairs - (labels < 0), rng, min_knots=8), labels
    counts = rng.poisson(np.where(labels > 0, 0.6, 3.0))
    return flights(counts, rng), labels


def parity_gaps(word):
    positions = [j for j, letter in enumerate(word) if letter]
    if len(positions) % 2:
        return None
    return sum(right - left for left, right in zip(positions[::2], positions[1::2]))


def fixed_mean(n, k, switches):
    """Exact beta expectation of (1 - 2 B)**switches."""
    if k is None:
        return Fraction(0)
    if k == 0:
        return Fraction(1)
    return sum(Fraction(comb(switches, j) * (-2)**j * comb(k + j - 1, j),
                        comb(n + j, j)) for j in range(switches + 1))


def exact_contrast(case, word):
    n, k = len(word), parity_gaps(word)
    if k is None or k == 0:
        return 0.
    if case == 'fixed':
        return float(fixed_mean(n, k, 2) - fixed_mean(n, k, 1))
    if case == 'mixture':
        return sum(p * float(fixed_mean(n, k, 2 * j) - fixed_mean(n, k, 2 * j - 1))
                   for j, p in [(1, 0.25), (2, 0.5), (4, 0.25)])
    return float(hyp1f1(k, n + 1, -1.2) - hyp1f1(k, n + 1, -6.0))


def empirical_contrast(values, labels):
    pos, neg = values[labels > 0], values[labels < 0]
    error = np.sqrt(pos.var(ddof=1) / len(pos) + neg.var(ddof=1) / len(neg))
    return float(pos.mean() - neg.mean()), float(error)


def fit(args):
    import pysiglib as ps

    test, test_y = sample(args.case, args.test, args.test_seed)
    for seed in args.seeds:
        train, train_y = sample(args.case, args.train, seed)
        if args.permute:
            train_y = np.random.default_rng(seed + 1000000).permutation(train_y)
        for degree in args.degrees:
            model = ps.sig_lasso(train, train_y, degree, args.alpha, n_jobs=args.jobs,
                                 tol=1e-8, memory_limit=2**30)
            words = model['words']
            pred = ps.sig_lasso_predict(test, model, n_jobs=args.jobs)
            stats = model['stats']
            selected = []
            for i, word in enumerate(words):
                scale = float(factorial(len(word)))
                # Use the separate selected-coefficient kernel on untouched paths.
                values = ps.sig_coef(test, word, n_jobs=args.jobs).reshape(-1) * scale
                contrast, se = empirical_contrast(values, test_y)
                selected.append(dict(word=''.join('x' if a else 't' for a in word),
                                     k=parity_gaps(word), coefficient=float(model['coef'][i]),
                                     exact=exact_contrast(args.case, word), test=contrast, se=se))
            record = dict(case=args.case, seed=seed, test_seed=args.test_seed, degree=degree,
                          train_per_class=args.train, test_per_class=args.test, alpha=args.alpha,
                          permuted=args.permute, support=selected, kkt=float(model['kkt_violation']),
                          converged=bool(model['converged']), test_accuracy=float(np.mean((pred >= 0) == (test_y > 0))),
                          columns=stats['dictionary_size'], full_transforms=stats['full_transform_calls'],
                          native_seconds=stats['total_seconds'], native_peak_bytes=stats['peak_bytes'])
            print(json.dumps(record), flush=True)
            if not model['converged'] or stats['full_transform_calls']:
                raise RuntimeError('The branch-and-bound fit did not certify the result')


def deep_probes(args):
    import pysiglib as ps

    paths, labels = sample('fixed', args.test, args.test_seed)
    times = paths[:, :, 0]
    directions = np.sign(np.diff(paths[:, :, 1], axis=1))
    for n in args.degrees:
        word = (1,) + (0,) * (n - 2) + (1,)
        values = ps.sig_coef(paths, word, n_jobs=args.jobs).reshape(-1) * float(factorial(n))
        # Integrate the joint law of the minimum and maximum of n uniform times.
        reference = np.ones(len(paths))
        for i in range(directions.shape[1]):
            for j in range(i + 1, directions.shape[1]):
                mass = ((times[:, j + 1] - times[:, i])**n - (times[:, j + 1] - times[:, i + 1])**n
                        - (times[:, j] - times[:, i])**n + (times[:, j] - times[:, i + 1])**n)
                reference -= 2 * mass * (directions[:, i] != directions[:, j])
        worst = float(np.max(np.abs(reference - values)))
        if worst > 1e-10:
            raise ArithmeticError((n, worst))
        contrast, se = empirical_contrast(values, labels)
        print(json.dumps(dict(degree=n, test=contrast, se=se, exact=exact_contrast('fixed', word),
                              sign_accuracy=float(np.mean((values >= 0) == (labels > 0))),
                              pathwise_error=worst, test_per_class=args.test, test_seed=args.test_seed)), flush=True)


def interleaved_mean(word, switches):
    """Count signed interleavings of signature times and switch times, exactly."""
    row = [1] * (switches + 1)
    for letter in word:
        for j in range(1, switches + 1):
            sign = -1 if letter and j % 2 else 1
            row[j] = sign * row[j] + row[j - 1]
    initial_sign_mean = int(sum(word) % 2 == 0)
    return Fraction(row[-1] * initial_sign_mean, comb(len(word) + switches, switches))


def audit(args):
    comparisons = 0
    reconstructed = 0
    moments = {}
    for n in range(1, args.degrees[0] + 1):
        counts = Counter()
        means = {(k, m): fixed_mean(n, k, m) for k in [None, *range(n)]
                 for m in range(args.switches + 1)}
        for word in product((0, 1), repeat=n):
            k = parity_gaps(word)
            counts[k] += 1
            for m in range(args.switches + 1):
                direct = interleaved_mean(word, m)
                if direct != means[k, m]:
                    raise ArithmeticError((word, m, direct, means[k, m]))
                comparisons += 1
        if counts[None] != 2**(n - 1) or any(counts[k] != comb(n - 1, k) for k in range(n)):
            raise ArithmeticError(counts)
        if n >= 2:
            endpoint_word = (1,) + (0,) * (n - 2) + (1,)
            for m in range(args.switches + 1):
                moments[n - 2, m] = interleaved_mean(endpoint_word, m) / (n * (n - 1))
                for k in range(1, n):
                    degree = n - k - 1
                    value = k * comb(n, k) * sum((-1)**s * comb(degree, s) * moments[k - 1 + s, m]
                                                 for s in range(degree + 1))
                    if value != means[k, m]:
                        raise ArithmeticError((n, k, m, value))
                    reconstructed += 1
        if n >= 3:
            for m in range(2, args.switches + 1, 2):
                contrasts = [abs(means[k, m] - means[k, m - 1]) for k in range(n)]
                if max(contrasts[:-1]) >= contrasts[-1]:
                    raise ArithmeticError((n, m, contrasts))
    print(json.dumps(dict(exact_comparisons=comparisons, reconstructed_classes=reconstructed, degree=args.degrees[0],
                          max_switches=args.switches)), flush=True)


def matched_counts(args):
    import pysiglib as ps

    # Coefficients of (1-z) P'_7(z), up to a common multiplier.
    difference = np.array([-5, 5, 135, -135, -495, 495, 429, -429])
    normalizer = int(difference.clip(min=0).sum())
    for n in range(1, 9):
        largest = max(abs(sum(int(a) * interleaved_mean(word, m) for m, a in enumerate(difference))
                          / normalizer) for word in product((0, 1), repeat=n))
        target = Fraction(16, 171171) if n == 8 else 0
        if largest != target:
            raise ArithmeticError((n, largest))
    rng = np.random.default_rng(args.test_seed)
    labels = np.repeat([-1., 1.], args.test)
    counts = np.concatenate([rng.choice(8, size=args.test, p=(s * difference).clip(min=0) / normalizer)
                             for s in [-1, 1]])
    paths = flights(counts, rng)
    word = (1, 0, 0, 0, 0, 1, 0, 0)
    values = factorial(8) * ps.sig_coef(paths, word, n_jobs=args.jobs).reshape(-1)
    contrast, se = empirical_contrast(values, labels)
    print(json.dumps(dict(first_separating_degree=8, exact_maximum=str(largest), word='xttttxtt',
                          test=contrast, se=se, test_per_class=args.test, test_seed=args.test_seed)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'probe', 'audit', 'matched'])
    parser.add_argument('--case', choices=['fixed', 'poisson', 'mixture'], default='fixed')
    parser.add_argument('--degrees', nargs='+', type=int, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[47601, 47602, 47603])
    parser.add_argument('--test-seed', type=int, default=93470)
    parser.add_argument('--train', type=int, default=384, help='independent paths per class')
    parser.add_argument('--test', type=int, default=8192, help='independent paths per class')
    parser.add_argument('--alpha', type=float, default=0.24)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--switches', type=int, default=8, help='largest count in the exact audit')
    parser.add_argument('--permute', action='store_true', help='shuffle training labels only')
    args = parser.parse_args()
    if args.degrees is None:
        args.degrees = {'fit': [6, 9, 12], 'probe': [6, 9, 12, 24, 48, 96], 'audit': [12], 'matched': [8]}[args.action]
    if min(args.degrees) < (1 if args.action != 'probe' else 2) or max(args.degrees) > 170:
        parser.error('choose a supported degree from 1 through 170 (2 through 170 for probes)')
    if args.action == 'audit' and len(args.degrees) != 1:
        parser.error('audit takes a single maximum degree')
    if args.train < 1 or args.test < 2 or args.switches < 0:
        parser.error('training needs a path per class, standard errors need two, and counts must be nonnegative')
    {'fit': fit, 'probe': deep_probes, 'audit': audit, 'matched': matched_counts}[args.action](args)
