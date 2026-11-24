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

"""Presence of contiguous token patterns using intervals of a bounded suffix array."""

import numpy as np
from numba import njit
from products import offer
from scipy.sparse import csc_matrix
from solver import bound_slack


@njit(cache=True)
def _suffix_walk(tokens, doc, starts, degree, base, r, floor, scores, top, slack, mode, codes, ptr, rows):
    stop = np.zeros(degree + 1, np.int64)
    cursor = np.zeros(degree + 1, np.int64)
    prefix = np.zeros(degree + 1, np.int64)
    stop[0] = len(starts)
    seen = np.zeros(len(r), np.int64)
    level = visited = pruned = used = cols = nnz = 0
    threshold = floor
    while level >= 0:
        lo = cursor[level]
        if lo == stop[level]:
            level -= 1
            continue
        symbol = tokens[starts[lo] + level]
        hi = lo + 1
        while hi < stop[level] and tokens[starts[hi] + level] == symbol:
            hi += 1
        cursor[level] = hi
        if symbol == 0:
            continue
        visited += 1
        code = prefix[level] * base + symbol
        pos = neg = 0.0
        for p in range(lo, hi):
            i = doc[starts[p]]
            if seen[i] == visited:
                continue
            seen[i] = visited
            if mode == 0:
                if r[i] > 0:
                    pos += r[i]
                else:
                    neg -= r[i]
            else:
                if mode == 2:
                    rows[nnz] = i
                nnz += 1
        if mode == 0:
            used, threshold = offer(scores, top, threshold, floor, used, abs(pos - neg), code)
        else:
            if mode == 2:
                codes[cols], ptr[cols + 1] = code, nnz
            cols += 1
        if level + 1 < degree:
            if mode == 0 and max(pos, neg) + slack < threshold:
                pruned += 1
            else:
                level += 1
                cursor[level], stop[level], prefix[level] = lo, hi, code
    return used, visited, pruned, cols, nnz


class Sequences:
    def __init__(self, docs, degree, base):
        self.n, self.degree, self.base = len(docs), degree, base
        if degree < 1 or base < 2 or base**degree > 2**63 - 1:
            raise ValueError("invalid suffix index dimensions")
        sizes = np.array([len(row) + degree for row in docs], dtype=np.int64)
        self.tokens = np.zeros(sizes.sum(), np.int32)
        self.doc = np.repeat(np.arange(self.n, dtype=np.int32), sizes)
        pos = 0
        for row in docs:
            self.tokens[pos : pos + len(row)] = row
            pos += len(row) + degree
        if np.any(self.tokens < 0) or np.any(self.tokens >= base):
            raise ValueError("token out of alphabet range")
        starts = np.flatnonzero(self.tokens).astype(np.int64)
        self.starts = starts[np.lexsort(tuple(self.tokens[starts + j] for j in reversed(range(degree))))]

    def _run(self, mode, r, floor, scores, keys, codes, ptr, rows):
        return _suffix_walk(
            self.tokens,
            self.doc,
            self.starts,
            self.degree,
            self.base,
            r,
            floor,
            scores,
            keys,
            bound_slack(r),
            mode,
            codes,
            ptr,
            rows,
        )

    def search(self, r, floor, k):
        scores, keys, empty = np.zeros(k), np.empty(k, np.int64), np.empty(0, np.int64)
        used, visited, pruned, _, _ = self._run(
            0, r, floor, scores, keys, empty, empty, np.empty(0, np.int32)
        )
        return sorted(zip(scores[:used], keys[:used]), reverse=True), visited, pruned

    def columns(self, keys):
        out = np.zeros((self.n, len(keys)), order="F")
        for j, code in enumerate(keys):
            symbols = []
            while code:
                code, symbol = divmod(int(code), self.base)
                symbols.append(symbol)
            lo, hi = 0, len(self.starts)
            for depth, symbol in enumerate(reversed(symbols)):
                # The interval shares this prefix, so the next symbols are sorted.
                lo, hi = _interval(self.tokens, self.starts, depth, lo, hi, symbol)
            out[self.doc[self.starts[lo:hi]], j] = 1.0
        return out

    def explicit(self):
        empty = np.empty(0, np.int64)
        args = (np.zeros(self.n), 0.0, np.empty(1), np.empty(1, np.int64))
        _, _, _, cols, nnz = self._run(1, *args, empty, empty, np.empty(0, np.int32))
        codes, ptr = np.empty(cols, np.int64), np.zeros(cols + 1, np.int64)
        rows = np.empty(nnz, np.int32)
        self._run(2, *args, codes, ptr, rows)
        matrix = csc_matrix((np.ones(nnz), rows, ptr), shape=(self.n, cols))
        matrix.sort_indices()
        return matrix, codes


@njit(cache=True)
def _interval(tokens, starts, level, lo, hi, symbol):
    lower, upper = lo, hi
    while lower < upper:
        mid = (lower + upper) // 2
        if tokens[starts[mid] + level] < symbol:
            lower = mid + 1
        else:
            upper = mid
    first, upper = lower, hi
    while lower < upper:
        mid = (lower + upper) // 2
        if tokens[starts[mid] + level] <= symbol:
            lower = mid + 1
        else:
            upper = mid
    return first, lower
