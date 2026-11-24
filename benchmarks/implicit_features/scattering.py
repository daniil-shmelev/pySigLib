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

"""Globally pooled periodic Haar scattering and graph diffusion scattering."""

import math

import numpy as np
from numba import njit
from products import offer
from scipy import sparse
from scipy.sparse import csgraph
from solver import Candidates, bound_slack, make_matrix


@njit(cache=True)
def _haar_step(low, shift):
    n, t = low.shape
    next_low, modulus = np.empty_like(low), np.empty_like(low)
    shift %= t
    for i in range(n):
        for j in range(t):
            other = j - shift
            if other < 0:
                other += t
            a, b = low[i, j], low[i, other]
            next_low[i, j] = 0.5 * (a + b)
            modulus[i, j] = 0.5 * abs(a - b)
    return next_low, modulus


@njit(cache=True)
def _moments(state):
    n, t = state.shape
    mean, sd = np.empty(n), np.empty(n)
    for i in range(n):
        a = state[i].sum() / t
        mean[i] = a
        # Direct differences avoid cancellation for nearly constant states.
        sd[i] = np.sqrt(np.sum((state[i] - a) ** 2) / t)
    return mean, sd


@njit(cache=True)
def _haar_walk(roots, scales, degree, increasing, r, floor, top, scores, slack, full, keys, design):
    channels, n, t = roots.shape
    work = np.empty((2, degree + 1, n, t))
    scale = np.zeros(degree + 1, np.int64)
    first = np.zeros(degree + 1, np.int64)
    slot = np.zeros(degree + 1, np.int64)
    code = np.zeros(degree + 1, np.int64)
    threshold = floor
    used = visited = cuts = 0
    for ch in range(channels):
        work[0, 0] = roots[ch]
        level = 0
        scale[0], slot[0], first[0], code[0] = 0, 0, 0, (ch + 1) * (scales + 1)
        while level >= 0:
            j = scale[level]
            if j == scales:
                level -= 1
                continue
            scale[level] += 1
            cur = slot[level]
            slot[level] = 1 - cur
            shift = (2**j) % t
            evaluate = j >= first[level]
            deeper = level + 1 < degree and (not increasing or j + 1 < scales)
            upper_pos = upper_neg = corr = 0.0
            for i in range(n):
                total = squares = 0.0
                for ti in range(t):
                    prev = ti - shift
                    if prev < 0:
                        prev += t
                    a, b = work[cur, level, i, ti], work[cur, level, i, prev]
                    work[1 - cur, level, i, ti] = 0.5 * (a + b)
                    if evaluate:
                        h = 0.5 * abs(a - b)
                        if deeper:
                            work[0, level + 1, i, ti] = h
                        total += h
                        if deeper and not full:
                            squares += h * h
                if evaluate:
                    mean = total / t
                    if full:
                        design[visited, i] = mean
                    else:
                        corr += mean * r[i]
                        if deeper:
                            energy = squares / t
                            roundoff = 16 * t * np.finfo(np.float64).eps * (energy + mean * mean)
                            sd = np.sqrt(max(0.0, energy - mean * mean) + roundoff)
                            v = min(mean, sd) * r[i]
                            if v > 0:
                                upper_pos += v
                            else:
                                upper_neg -= v
            if not evaluate:
                continue
            child_code = code[level] * (scales + 1) + j + 1
            if full:
                keys[visited] = child_code
            else:
                used, threshold = offer(scores, top, threshold, floor, used, abs(corr), child_code)
            visited += 1
            if deeper:
                b = max(upper_pos, upper_neg)
                if full or b + slack * (1.0 + b) >= threshold:
                    level += 1
                    first[level] = j + 1 if increasing else 0
                    slot[level], scale[level], code[level] = 0, 0, child_code
                else:
                    cuts += 1
    return used, visited, cuts


class Scattering:
    def __init__(self, roots, scales, degree, increasing):
        self.roots, self.scales, self.degree = roots, scales, degree
        self.increasing = increasing

    def _walk(self, visit, wanted=None):
        def walk(key, state):
            start = key[-1] + 1 if self.increasing and len(key) > 1 else 0
            indices = range(start, self.scales) if wanted is None else wanted.get(key, ())
            for j, child in self.children(state, indices):
                child_key = key + (j,)
                mean, envelope = self.measure(child) if wanted is None else (self.pool(child), None)
                if visit(child_key, mean, envelope) and len(child_key) <= self.degree:
                    walk(child_key, child)

        for i, state in enumerate(self.roots):
            walk((i,), state)

    def search(self, r, floor, k):
        out, slack = Candidates(floor, k), bound_slack(r)
        pos, neg = np.maximum(r, 0.0), np.maximum(-r, 0.0)

        def visit(key, val, envelope):
            out.visited += 1
            out.add(abs(r @ val), key)
            b = max(pos @ envelope, neg @ envelope)
            keep = b + slack * (1 + b) >= out.threshold
            out.pruned += int(not keep and len(key) <= self.degree)
            return keep

        self._walk(visit)
        return out.result()

    def columns(self, keys):
        pos = {key: j for j, key in enumerate(keys)}
        wanted = {}
        for key in keys:
            for j in range(1, len(key)):
                wanted.setdefault(key[:j], set()).add(key[j])
        out = np.zeros((self.n, len(keys)), order="F")

        def visit(key, val, envelope):
            if key in pos:
                out[:, pos[key]] = val
            return key in wanted

        self._walk(visit, wanted)
        return out

    def explicit(self):
        keys, values = [], []

        # A full transform does not need to compute any branch bounds.
        def walk(key, state):
            start = key[-1] + 1 if self.increasing and len(key) > 1 else 0
            for j, child in self.children(state, range(start, self.scales)):
                child_key = key + (j,)
                keys.append(child_key)
                values.append(self.pool(child))
                if len(child_key) <= self.degree:
                    walk(child_key, child)

        for i, state in enumerate(self.roots):
            walk((i,), state)
        return make_matrix(values, self.n), keys


class Haar(Scattering):
    def __init__(self, x, scales, degree, increasing=False):
        x = np.asarray(x, dtype=float)
        self.n = len(x)
        if x.ndim == 2:
            x = x[:, :, None]
        if degree < 1 or scales < 1 or 2 ** (scales - 1) >= x.shape[1] or not np.isfinite(x).all():
            raise ValueError("invalid periodic Haar dictionary")
        # A terminator between the channel and the scale digits makes decoding unambiguous.
        if (x.shape[2] + 1) * (scales + 1) ** (degree + 1) >= 2**63:
            raise ValueError("Haar indices exceed int64")
        super().__init__(np.ascontiguousarray(x.transpose(2, 0, 1)), scales, degree, increasing)

    def _decode(self, code):
        scales = []
        while True:
            code, j = divmod(int(code), self.scales + 1)
            if j == 0:
                return (code - 1, *reversed(scales))
            scales.append(j - 1)

    def search(self, r, floor, k):
        keys, scores = np.empty(k, np.int64), np.zeros(k)
        used, visited, cuts = _haar_walk(
            self.roots,
            self.scales,
            self.degree,
            self.increasing,
            r,
            floor,
            keys,
            scores,
            bound_slack(r),
            False,
            np.empty(0, np.int64),
            np.empty((0, 0)),
        )
        best = [(scores[i], self._decode(keys[i])) for i in range(used)]
        return sorted(best, reverse=True), visited, cuts

    def explicit(self):
        count = len(self.roots) * sum(
            (math.comb(self.scales, h) if self.increasing else self.scales**h)
            for h in range(1, self.degree + 1)
        )
        if count * self.n * 8 > 2**31:
            raise MemoryError("explicit Haar design exceeds 2 GiB benchmark limit")
        design, keys = np.empty((count, self.n)), np.empty(count, np.int64)
        _, visited, _ = _haar_walk(
            self.roots,
            self.scales,
            self.degree,
            self.increasing,
            np.zeros(self.n),
            0.0,
            np.empty(1, np.int64),
            np.empty(1),
            0.0,
            True,
            keys,
            design,
        )
        if visited != count:
            raise ArithmeticError("incorrect Haar enumeration")
        return design.T, [self._decode(k) for k in keys]

    def children(self, state, indices):
        low = state
        for j in range(max(indices, default=-1) + 1):
            low, high = _haar_step(low, 2**j)
            if j in indices:
                yield j, high

    def pool(self, state):
        return state.mean(axis=1)

    def measure(self, state):
        mean, sd = _moments(state)
        return mean, np.minimum(mean, sd)


class GraphScattering(Scattering):
    def __init__(self, graphs, channels, scales, degree):
        blocks, labels, sizes = [], [], []
        for nodes, adj in graphs:
            ii, jj = [], []
            for i, neighbours in enumerate(adj):
                for j in neighbours:
                    ii.append(i)
                    jj.append(j)
            n = len(nodes)
            a = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
            lap = csgraph.laplacian(a).tocsr()
            d = max(1.0, max(map(len, adj)))
            blocks.append(sparse.eye(n, format="csr") - lap / (2 * d))
            sizes.append(n)
            labels.extend(nodes)
        self.n = len(sizes)
        self.sizes = np.array(sizes)
        self.starts = np.cumsum(self.sizes) - self.sizes
        self.p = sparse.block_diag(blocks, format="csr")
        labels = np.array(labels)
        super().__init__([(labels == c).astype(float) for c in channels], scales, degree, False)

    def children(self, state, indices):
        low = state
        for j in range(max(indices, default=-1) + 1):
            blurred = low
            for _ in range(1 if j == 0 else 2 ** (j - 1)):
                blurred = self.p @ blurred
            if j in indices:
                yield j, np.abs(low - blurred)
            low = blurred

    def pool(self, state):
        return np.add.reduceat(state, self.starts) / self.sizes

    def measure(self, state):
        mean = self.pool(state)
        delta = state - np.repeat(mean, self.sizes)
        return mean, np.sqrt(self.pool(delta * delta))
