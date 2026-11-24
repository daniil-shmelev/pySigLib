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

"""All univariate interval indicators with exact signed endpoint bounds."""

import heapq

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix


@njit(cache=True)
def _best_suffix(p):
    n = len(p) - 1
    score = np.empty(n)
    right = np.empty(n, np.int64)
    lo = hi = n
    for left in range(n - 1, -1, -1):
        if p[left + 1] < p[lo]:
            lo = left + 1
        if p[left + 1] > p[hi]:
            hi = left + 1
        right[left] = lo if p[left] - p[lo] > p[hi] - p[left] else hi
        score[left] = abs(p[right[left]] - p[left])
    return score, right


class Intervals:
    def __init__(self, bins, sizes):
        self.bins = np.ascontiguousarray(bins, dtype=np.int32)
        self.sizes = sizes
        self.n = self.bins.shape[1]

    def columns(self, keys):
        out = np.empty((self.n, len(keys)), order="F")
        for col, (j, a, b) in enumerate(keys):
            out[:, col] = (self.bins[j] >= a) & (self.bins[j] < b)
        return out

    def search(self, r, floor, k, exhaustive=False):
        best, hist = [], []
        groups = 0
        for j, b in enumerate(self.sizes):
            p = np.r_[0.0, np.cumsum(np.bincount(self.bins[j], weights=r, minlength=b))]
            hist.append(p)
            if exhaustive:
                a, h = np.triu_indices(b)
                right = h + 1
                s = np.abs(p[right] - p[a])
                selected = np.flatnonzero(s > floor)
                if len(selected) > k:
                    selected = selected[np.argpartition(s[selected], -k)[-k:]]
                best.extend((-s[i], j, a[i], right[i], 0, 0) for i in selected)
                groups += len(s)
            else:
                s, right = _best_suffix(p)
                selected = np.flatnonzero(s > floor)
                if len(selected) > k:
                    selected = selected[np.argpartition(s[selected], -k)[-k:]]
                best.extend((-s[a], j, a, right[a], a + 1, b + 1) for a in selected)
                groups += b
        heap = heapq.nsmallest(k, best)
        heapq.heapify(heap)
        out = []
        while heap and len(out) < k:
            score, j, a, b, lo, hi = heapq.heappop(heap)
            out.append((-score, (j, a, b)))
            if exhaustive or len(out) == k:
                continue
            p = hist[j]
            for first, stop in ((lo, b), (b + 1, hi)):
                if first >= stop:
                    continue
                v = p[first:stop]
                i, h = first + v.argmin(), first + v.argmax()
                right = i if p[a] - p[i] > p[h] - p[a] else h
                s = abs(p[right] - p[a])
                if s > floor:
                    heapq.heappush(heap, (-s, j, a, right, first, stop))
        return out, groups, 0

    def explicit(self):
        keys, lengths, orders, offsets = [], [], [], []
        for j, b in enumerate(self.sizes):
            order = np.argsort(self.bins[j], kind="stable").astype(np.int32)
            off = np.r_[0, np.cumsum(np.bincount(self.bins[j], minlength=b))]
            for a in range(b):
                for right in range(a + 1, b + 1):
                    keys.append((j, a, right))
                    lengths.append(off[right] - off[a])
            orders.append(order)
            offsets.append(off)
        ptr = np.r_[0, np.cumsum(lengths)]
        if 12 * ptr[-1] > 2**31:
            raise MemoryError("explicit interval design exceeds 2 GiB benchmark limit")
        rows = np.empty(ptr[-1], np.int32)
        for col, (j, a, b) in enumerate(keys):
            rows[ptr[col] : ptr[col + 1]] = orders[j][offsets[j][a] : offsets[j][b]]
        x = csc_matrix((np.ones(len(rows)), rows, ptr), shape=(self.n, len(keys)))
        x.sort_indices()
        return x, keys


class HistogramScan(Intervals):
    def search(self, r, floor, k):
        return super().search(r, floor, k, exhaustive=True)
