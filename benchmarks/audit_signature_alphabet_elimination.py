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

"""Audit degree-four alphabet elimination against the existing endpoint screen."""

import argparse
import json
from itertools import product
from math import comb
from pathlib import Path

import numpy as np


def pivot_bounds(increments, order):
    remaining = np.zeros(increments.shape[:2])
    nodes = (.5 - .5 / np.sqrt(3.), .5 + .5 / np.sqrt(3.))
    for pivot in reversed(order):
        dx = increments[:, :, pivot]
        remaining = np.maximum(remaining, dx)
        before = np.cumsum(remaining, axis=1) - remaining
        after = np.cumsum(remaining[:, ::-1], axis=1)[:, ::-1] - remaining
        cap = remaining.sum(axis=1)**4
        bounds = np.zeros((len(dx), 4))
        for t in nodes:
            left = before + t * remaining
            right = after + (1. - t) * remaining
            for pos in range(4):
                bounds[:, pos] += 2 * comb(3, pos) * (dx * left**pos * right**(3-pos)).sum(axis=1)
        yield pivot, np.minimum(bounds, cap[:, None]), cap


def endpoint_bounds(x, dv):
    before = np.cumsum(dv, axis=1) - dv
    after = np.cumsum(dv[:, ::-1], axis=1)[:, ::-1] - dv

    def marginal(z, start, end):
        a, b = z[:, :-1], z[:, 1:]
        both = (a + b)**2
        return np.sqrt((dv[:, :, None] / 6 * (
            (2*a*a + both) * start[:, :, None] + (2*b*b + both) * end[:, :, None])).sum(axis=1))

    return marginal(x, after + dv, after), marginal(x[:, -1:] - x, before, before + dv)


def endpoint_counts(endpoint, order):
    # Counts for using the pivot at neither, first, last, or both endpoints.
    grid = endpoint[np.ix_(order, order)]
    counts = []
    neither = 0
    for j in reversed(range(len(order))):
        first = int(grid[j, j + 1:].sum())
        last = int(grid[j + 1:, j].sum())
        both = int(grid[j, j])
        counts.append((neither, first, last, both))
        neither += first + last + both
    return counts[::-1]


def survivors(counts, others, viable):
    neither, first, last, both = counts
    pairs = neither + first * viable[0] + last * viable[3] + both * (viable[0] and viable[3])
    return pairs * (others + int(viable[1])) * (others + int(viable[2])) - neither * others**2


def small_check():
    import pysiglib

    rng = np.random.default_rng(7216)
    x = rng.normal(size=(70, 10, 4)).cumsum(axis=1)
    x[0] = 0.
    x[1::5, -1] = x[1::5, 0]
    total = np.abs(np.diff(x, axis=1)).max(axis=2).sum(axis=1)
    x = np.ascontiguousarray((x - x[:, :1]) / np.where(total, total, 1.)[:, None, None])
    dx = np.abs(np.diff(x, axis=1))
    words = list(product(range(4), repeat=4))
    coeff = 24 * np.abs(pysiglib.sig(x, 4)[:, -len(words):])
    left, right = endpoint_bounds(x, dx.max(axis=2))
    for i, word in enumerate(words):
        assert np.all(coeff[:, i] <= 12 * left[:, word[0]] * right[:, word[3]] + 1e-13)
    for order in (np.arange(4), rng.permutation(4)):
        rank = np.argsort(order)
        endpoint = rng.random((4, 4)) > .35
        counts = endpoint_counts(endpoint, order)
        kept = 0
        direct = 0
        for pivot, bounds, _ in pivot_bounds(dx, order):
            allowed = rng.integers(0, 2, 4, dtype=bool)
            j = rank[pivot]
            kept += survivors(counts[j], 3 - j, allowed)
            for i, word in enumerate(words):
                if min(word, key=lambda c: rank[c]) != pivot:
                    continue
                positions = [p for p in range(4) if word[p] == pivot]
                for p in positions:
                    assert np.all(coeff[:, i] <= bounds[:, p] + 1e-13)
                direct += int(endpoint[word[0], word[3]] and allowed[positions].all())
        assert direct == kept


def audit(args):
    from benchmark_signature_tree_order import FOLDERS

    small_check()
    paper = json.loads(args.paper.read_text())
    all_rows = []
    cutoff = 1. - 1e-10
    for name, folder in FOLDERS.items():
        with np.load(args.data / folder / 'results/normalized/integral_comparison/prepared.npz') as raw:
            x, y = raw['train'], raw['response']
        with np.load(args.fits / f'{name}-4-prefix_norm-0.npz') as model:
            prediction = model['train']
        reference = next(r for r in paper['classification'] if r['dataset'] == name and r['degree'] == 4)
        alpha = np.asarray(reference['tree']['alpha'])
        dx = np.abs(np.diff(x, axis=1))
        total = dx.max(axis=2).sum(axis=1)
        scale = np.where(total, total, 1.)[:, None, None]
        dx /= scale
        left, right = endpoint_bounds((x - x[:, :1]) / scale, dx.max(axis=2))
        residuals = {'initial': y - y.mean(axis=0), 'fitted': y - prediction}
        weights = {key: np.abs(r) / (len(x) * alpha) for key, r in residuals.items()}
        endpoints = {}
        for key, w in weights.items():
            endpoint = np.zeros((dx.shape[2], dx.shape[2]), dtype=bool)
            for t in range(w.shape[1]):
                endpoint |= 12 * ((w[:, t:t+1] * left).T @ right) >= cutoff
            endpoints[key] = endpoint

        d = dx.shape[2]
        weak = np.argsort(np.einsum('ija,ija->a', dx, dx), kind='stable')
        alternating = np.empty_like(weak)
        alternating[::2] = weak[::-1][:(d + 1) // 2]
        alternating[1::2] = weak[:d // 2]
        orders = {'weak-first': weak, 'strong-first': weak[::-1], 'alternating': alternating, 'input': np.arange(d)}
        for label, order in orders.items():
            rank = np.argsort(order)
            counts = {key: endpoint_counts(e, order) for key, e in endpoints.items()}
            kept = {key: [0, 0, 0] for key in weights}
            for pivot, bounds, cap in pivot_bounds(dx, order):
                j = rank[pivot]
                for key, w in weights.items():
                    alphabet_ok = np.full(4, np.max(w.T @ cap) >= cutoff)
                    positions_ok = (w.T @ bounds).max(axis=0) >= cutoff
                    for i, test in enumerate((alphabet_ok, np.full(4, positions_ok.any()), positions_ok)):
                        kept[key][i] += int(survivors(counts[key][j], d - j - 1, test))
            for key in weights:
                base = int(endpoints[key].sum()) * d**2
                alphabet, anywhere, position = kept[key]
                assert 0 <= position <= anywhere <= alphabet <= base
                all_rows.append(dict(dataset=name, stage=key, order=label, outside_survivors=base,
                                     alphabet_only=alphabet, pivot_anywhere=anywhere, pivot_position=position))
                print(name, key, label, f'{position:,} of {base:,} words survive')
    if args.output:
        args.output.write_text(json.dumps(all_rows, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Prepared audio data root')
    parser.add_argument('--paper', type=Path, required=True, help='Saved classification penalties')
    parser.add_argument('--fits', type=Path, required=True, help='Outputs from benchmark_signature_tree_order.py')
    parser.add_argument('--output', type=Path, help='Save per-order word counts to JSON')
    audit(parser.parse_args())
