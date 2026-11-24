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

"""Implicit products of distinct groups of atoms with a total degree budget."""

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix
from solver import bound_slack


@njit(cache=True)
def offer(scores, keys, threshold, floor, used, score, key):
    if score <= threshold:
        return used, threshold
    j = used if used < len(scores) else np.argmin(scores)
    scores[j], keys[j] = score, key
    used = min(used + 1, len(scores))
    return used, max(floor, scores.min()) if used == len(scores) else floor


@njit(cache=True)
def _envelope(factors, groups, costs, degree, bounded):
    n = factors.shape[1]
    count = groups[-1] + 1
    depth = 1 if bounded else degree + 1
    out = np.zeros((count + 1, depth, n))
    for g in range(count - 1, -1, -1):
        out[g] = out[g + 1]
        for f in np.flatnonzero(groups == g):
            a = np.abs(factors[f])
            if bounded:
                out[g, 0] = np.maximum(out[g, 0], a)
            else:
                for b in range(costs[f], degree + 1):
                    further = np.maximum(1.0, out[g + 1, b - costs[f]])
                    out[g, b] = np.maximum(out[g, b], a * further)
    return out


@njit(cache=True)
def _walk(
    factors,
    after,
    groups,
    costs,
    degree,
    env,
    bounded,
    nonnegative,
    r,
    floor,
    scores,
    top,
    slack,
    mode,
    sparse,
    codes,
    ptr,
    rows,
    values,
):
    # mode 0 searches, mode 1 sizes the explicit design, mode 2 fills it.
    m, n = factors.shape
    samples = np.empty((degree + 1, n), dtype=np.int32)
    state = np.empty((degree + 1, n))
    samples[0], state[0] = np.arange(n), 1.0
    size = np.zeros(degree + 1, dtype=np.int64)
    size[0] = n
    next_atom = np.zeros(degree + 1, np.int64)
    budget = np.full(degree + 1, degree, np.int64)
    prefix = np.zeros(degree + 1, np.int64)
    level = used = visited = pruned = ncols = nnz = 0
    threshold = floor
    while level >= 0:
        f = next_atom[level]
        if f == m:
            level -= 1
            continue
        next_atom[level] += 1
        rest = budget[level] - costs[f]
        if rest < 0:
            continue
        visited += 1
        code = prefix[level] * (m + 1) + f + 1
        child = level + 1
        count = 0
        cor = pos = neg = 0.0
        extend = rest > 0 and after[f] < m
        for p in range(size[level]):
            i = samples[level, p]
            v = state[level, p] * factors[f, i]
            if v == 0.0:
                continue
            samples[child, count], state[child, count] = i, v
            count += 1
            if mode == 0:
                t = r[i] * v
                cor += t
                if extend:
                    e = env[groups[f] + 1, 0 if bounded else rest, i]
                    if t > 0.0:
                        pos += t * e
                    else:
                        neg -= t * e
            elif mode == 2:
                if sparse:
                    rows[nnz], values[nnz] = i, v
                    nnz += 1
                else:
                    values[ncols * n + i] = v
        if count == 0:
            if extend:
                pruned += 1
            continue
        if mode == 0:
            used, threshold = offer(scores, top, threshold, floor, used, abs(cor), code)
        else:
            if mode == 2:
                codes[ncols], ptr[ncols + 1] = code, nnz
            else:
                nnz += count
            ncols += 1
        if extend:
            upper = max(pos, neg) if nonnegative else pos + neg
            if mode == 0 and upper + slack * (1 + upper) < threshold:
                pruned += 1
                continue
            size[child], prefix[child] = count, code
            next_atom[child], budget[child] = after[f], rest
            level = child
    return used, visited, pruned, ncols, nnz


class Products:
    def __init__(self, factors, groups, costs, degree, sparse=False):
        self.factors = np.ascontiguousarray(factors, dtype=float)
        self.groups = np.asarray(groups, dtype=np.int64)
        self.costs = np.asarray(costs, dtype=np.int64)
        self.degree = degree
        self.sparse = sparse
        f, self.n = self.factors.shape
        if (
            f == 0
            or len(groups) != f
            or len(costs) != f
            or self.groups[0] != 0
            or np.any(np.diff(self.groups) < 0)
            or np.any(np.diff(self.groups) > 1)
            or np.any(self.costs <= 0)
            or degree < 1
            or not np.isfinite(self.factors).all()
        ):
            raise ValueError("invalid product dictionary")
        if (f + 1) ** min(degree, 1 + int(self.groups[-1])) > 2**63 - 1:
            raise ValueError("packed feature indices exceed int64")
        self.after = np.searchsorted(self.groups, self.groups, side="right").astype(np.int64)
        self.bounded = np.abs(self.factors).max(initial=0.0) <= 1.0
        self.nonnegative = np.all(self.factors >= 0.0)
        self.env = None

    def _run(self, mode, r, floor, scores, keys, codes, ptr, rows, values):
        if mode == 0 and self.env is None:
            self.env = _envelope(self.factors, self.groups, self.costs, self.degree, self.bounded)
            if not np.isfinite(self.env).all():
                raise OverflowError("nonfinite product envelope")
        env = self.env if mode == 0 else np.empty((0, 0, 0))
        return _walk(
            self.factors,
            self.after,
            self.groups,
            self.costs,
            self.degree,
            env,
            self.bounded,
            self.nonnegative,
            r,
            floor,
            scores,
            keys,
            bound_slack(r),
            mode,
            self.sparse,
            codes,
            ptr,
            rows,
            values,
        )

    def search(self, r, floor, k):
        scores, keys = np.zeros(k), np.empty(k, np.int64)
        empty = np.empty(0, np.int64)
        used, visited, cuts, _, _ = self._run(
            0, r, floor, scores, keys, empty, empty, np.empty(0, np.int32), np.empty(0)
        )
        return sorted(zip(scores[:used], keys[:used]), reverse=True), visited, cuts

    def columns(self, keys):
        out = np.ones((self.n, len(keys)), order="F")
        for j, key in enumerate(keys):
            while key:
                key, atom = divmod(int(key), len(self.factors) + 1)
                out[:, j] *= self.factors[atom - 1]
        return out

    def explicit(self):
        empty = np.empty(0, np.int64)
        z = np.zeros(self.n)
        args = (z, 0.0, np.empty(1), np.empty(1, np.int64))
        _, _, _, cols, nnz = self._run(1, *args, empty, empty, np.empty(0, np.int32), np.empty(0))
        length = nnz if self.sparse else cols * self.n
        if (12 if self.sparse else 8) * length > 2**31:
            raise MemoryError("explicit design exceeds 2 GiB benchmark limit")
        values = np.zeros(length)
        rows = np.empty(nnz if self.sparse else 0, np.int32)
        codes, ptr = np.empty(cols, np.int64), np.zeros(cols + 1, np.int64)
        self._run(2, *args, codes, ptr, rows, values)
        if self.sparse:
            matrix = csc_matrix((values, rows, ptr), shape=(self.n, cols))
        else:
            matrix = values.reshape(cols, self.n).T
        return matrix, codes
