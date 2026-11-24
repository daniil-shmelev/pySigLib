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

"""Independent small dictionaries and exhaustive residual-oracle checks."""

import itertools
import unittest

import networkx as nx
import numpy as np
from graphs import Graphs
from intervals import HistogramScan, Intervals
from products import Products
from scattering import GraphScattering, Haar
from scipy import sparse
from sequences import Sequences
from solver import fit_tree, kkt_error, predict
from subsequences import Subsequences
from templates import Templates
from walsh import Walsh, WalshTransform


def pack(word, base):
    total = 0
    for letter in word:
        total = base * total + letter
    return total


class TestBounds(unittest.TestCase):
    def compare(self, d, reference):
        x, keys = d.explicit()
        x = x.toarray() if sparse.issparse(x) else x
        self.assertEqual(set(keys), set(reference))
        values = np.column_stack([reference[k] for k in keys])
        np.testing.assert_allclose(x, values, atol=2e-13)
        np.testing.assert_allclose(d.columns(list(keys)), values, atol=2e-13)
        self.assertEqual(d.columns([]).shape, (d.n, 0))
        rng = np.random.default_rng(731)
        for r in [np.zeros(d.n), np.ones(d.n), *rng.normal(size=(12, d.n))]:
            scores = np.abs(values.T @ r)
            for floor in (0.0, 0.19 * max(scores), 1.05 * max(scores)):
                for k in (1, 3, 12):
                    result, _, _ = d.search(r, floor, k)
                    self.assertEqual(len({key for _, key in result}), len(result))
                    expected = sorted((z for z in scores if z > floor + 1e-12), reverse=True)[:k]
                    actual = [z for z, key in result if z > floor + 1e-12]
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
                    for z, key in result:
                        self.assertAlmostEqual(z, abs(reference[key] @ r), places=10)
        return list(keys), values

    def test_products(self):
        x = np.random.default_rng(31).uniform(-1.0, 1.0, (7, 12))
        x[0] = 0.0
        groups = [0, 0, 1, 2, 2, 3, 4]
        costs = [1, 2, 1, 1, 2, 1, 1]
        for a in (x, np.abs(x), 3.2 * x, 2.1 * np.abs(x)):
            ref = {}
            for n in range(1, 5):
                for inds in itertools.combinations(range(len(a)), n):
                    ids = list(inds)
                    if len({groups[i] for i in ids}) < n or sum(costs[i] for i in ids) > 4:
                        continue
                    val = a[ids].prod(axis=0)
                    if val.any():
                        ref[pack([i + 1 for i in ids], 8)] = val
            self.compare(Products(a, groups, costs, 4, sparse=True), ref)

    def test_strings(self):
        docs = [[1, 2, 1, 3, 2], [2, 1], [3, 1, 1], [], [3, 2, 1, 1, 3, 3]]
        for gaps in (False, True):
            ref = {}
            for i, doc in enumerate(docs):
                for n in range(1, 5):
                    words = (
                        itertools.combinations(doc, n)
                        if gaps
                        else (doc[j : j + n] for j in range(len(doc) - n + 1))
                    )
                    for word in words:
                        key = pack([t + int(gaps) for t in word], 5 if gaps else 4)
                        ref.setdefault(key, np.zeros(len(docs)))[i] = 1.0
            d = Subsequences(docs, 4, 4) if gaps else Sequences(docs, 4, 4)
            self.compare(d, ref)
            if gaps:
                x, keys = d.distinct()
                classes = {tuple(col) for col in ref.values()}
                self.assertEqual(classes, set(map(tuple, x.toarray().T)))
                self.assertEqual(len(keys), len(classes))
                np.testing.assert_array_equal(x.toarray(), d.columns(list(keys)))

    def test_haar(self):
        rng = np.random.default_rng(203)
        x = rng.normal(size=(8, 15))
        x[0] = 0.75
        for increasing in (True, False):
            ref = {}
            states = {}
            for m in range(1, 4):
                for path in itertools.product(range(3), repeat=m):
                    if increasing and any(b <= a for a, b in zip(path, path[1:])):
                        continue
                    state = x
                    for j in path:
                        h = 2**j
                        kernel = np.r_[np.ones(h), -np.ones(h)] / (2 * h)
                        val = np.fft.ifft(np.fft.fft(state) * np.fft.fft(kernel, 15), axis=1)
                        state = np.abs(val.real)
                    ref[(0,) + path] = state.mean(axis=1)
                    if len(path) > 1:
                        parent = states[(0,) + path[:-1]]
                        envelope = np.minimum(parent.mean(axis=1), np.std(parent, axis=1))
                        self.assertTrue(np.all(ref[(0,) + path] <= envelope + 1e-12))
                    states[(0,) + path] = state
            self.compare(Haar(x, 3, 3, increasing), ref)

    def test_graphs(self):
        rng = np.random.default_rng(217)
        original, graphs = [], []
        for i in range(8):
            g = nx.gnp_random_graph(6, 0.3 + i / 30, seed=i)
            for v in g:
                g.nodes[v]["label"] = int(rng.integers(2))
            for a, b in g.edges:
                g[a][b]["bond"] = int(rng.integers(1, 3))
            original.append(g)
            graphs.append(
                ([g.nodes[v]["label"] for v in g], [{w: g[v][w]["bond"] for w in g[v]} for v in g])
            )
        d = Graphs(graphs, 4)
        _, keys = d.explicit()
        patterns = []
        for key in keys:
            g = nx.Graph()
            for j, step in enumerate(key):
                g.add_node(j, label=step[0])
                for i, bond in enumerate(step[1:]):
                    if bond:
                        g.add_edge(i, j, bond=bond)
            patterns.append(g)

        def same(a, b):
            return nx.is_isomorphic(
                a,
                b,
                node_match=lambda p, q: p["label"] == q["label"],
                edge_match=lambda p, q: p["bond"] == q["bond"],
            )

        ref = {key: np.zeros(d.n) for key in keys}
        for i, g in enumerate(original):
            for n in range(1, 5):
                for ids in itertools.combinations(g.nodes, n):
                    sub = g.subgraph(ids)
                    if nx.is_connected(sub):
                        hits = [k for k, h in zip(keys, patterns) if n == len(h) and same(sub, h)]
                        self.assertEqual(len(hits), 1)
                        ref[hits[0]][i] = 1.0
        self.compare(d, ref)
        spectral = {}
        for i, g in enumerate(original):
            a = nx.to_numpy_array(g, weight=None)
            w, q = np.linalg.eigh(
                np.eye(len(a)) - (np.diag(a.sum(axis=1)) - a) / (2 * max(1, a.sum(axis=1).max()))
            )
            bank = [(q * (1 - w)) @ q.T, (q * (w - w**2)) @ q.T, (q * (w**2 - w**4)) @ q.T]
            for chan in (0, 1):
                for m in range(1, 4):
                    for path in itertools.product(range(3), repeat=m):
                        v = np.array([float(g.nodes[t]["label"] == chan) for t in g])
                        for j in path:
                            v = np.abs(bank[j] @ v)
                        spectral.setdefault((chan,) + path, np.zeros(d.n))[i] = v.mean()
        self.compare(GraphScattering(graphs, (0, 1), 3, 3), spectral)

    def test_metric_intervals(self):
        rng = np.random.default_rng(73)
        w, c = rng.normal(size=(14, 6, 3)), rng.normal(size=(23, 3))
        d = Templates(w, c, 0.7, leaf=3, cache=2)
        ref = {i: np.exp(-0.7 * ((w - centre) ** 2).sum(axis=2).min(axis=1)) for i, centre in enumerate(c)}
        self.compare(d, ref)

        def check(ball):
            indices = (
                ball.indices
                if not ball.children
                else np.concatenate([check(child) for child in ball.children])
            )
            e = np.sqrt(-np.log(ref[ball.centre]) / 0.7)
            low = np.exp(-0.7 * (e + ball.radius) ** 2)
            high = np.exp(-0.7 * np.maximum(e - ball.radius, 0.0) ** 2)
            for j in indices:
                self.assertTrue(np.all((low <= ref[j] + 1e-12) & (ref[j] <= high + 1e-12)))
            return indices

        check(d.root)
        raw = {i: np.linalg.norm(w - centre, axis=2).min(axis=1) for i, centre in enumerate(c)}
        self.compare(Templates(w, c, 1.0, leaf=3, profile="distance"), raw)
        bins = rng.integers(0, 5, (3, 32))
        ref = {
            (j, a, b): ((v >= a) & (v < b)).astype(float)
            for j, v in enumerate(bins)
            for a in range(5)
            for b in range(a + 1, 6)
        }
        for cls in (Intervals, HistogramScan):
            keys, X = self.compare(cls(bins, [5] * 3), ref)
        y = rng.normal(size=32)
        model = fit_tree(Intervals(bins, [5] * 3), y, 0.05, batch=3)
        beta = np.zeros(len(keys))
        for key, coef in zip(model.keys, model.coef):
            beta[keys.index(key)] = coef
        r = y - X @ beta - model.intercept
        self.assertLess(kkt_error(X.T @ r / len(y), beta, 0.05, r), 1.01e-7)
        test = rng.integers(0, 5, (3, 24))
        pred = np.full(24, model.intercept)
        for (j, a, b), coef in zip(keys, beta):
            pred += coef * ((test[j] >= a) & (test[j] < b))
        np.testing.assert_allclose(predict(Intervals(test, [5] * 3), [model])[:, 0], pred)

    def test_walsh(self):
        rng = np.random.default_rng(731)
        bits = rng.integers(0, 2, (25, 8))
        bits[1] = bits[0]
        ref = {}
        for mask in range(1, 256):
            signs = 1 - 2 * bits[:, [b for b in range(8) if mask & (1 << b)]]
            ref[mask] = signs.prod(axis=1)
        dic = Walsh(bits)
        self.compare(dic, ref)
        self.compare(WalshTransform(dic), ref)


if __name__ == "__main__":
    unittest.main()
