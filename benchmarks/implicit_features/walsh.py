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

"""Boolean Fourier features with signed folding on the observed suffixes."""

import numpy as np
from numba import njit
from products import offer
from solver import bound_slack


@njit(cache=True)
def _fwht(a):
    step = 1
    while step < len(a):
        for i in range(0, len(a), 2 * step):
            for j in range(i, i + step):
                x, y = a[j], a[j + step]
                a[j], a[j + step] = x + y, x - y
        step *= 2


@njit(cache=True)
def _search(values, suffix, sizes, left, right, d, floor, scores, keys, slack):
    state = np.zeros_like(suffix, dtype=np.float64)
    state[0, : len(values)] = values
    next_bit = np.zeros(d + 1, np.int64)
    mask = np.zeros(d + 1, np.int64)
    threshold = floor
    level = used = visits = cuts = 0
    while level >= 0:
        if d - level <= 5:
            block = np.zeros(1 << (d - level))
            for j in range(sizes[level]):
                block[suffix[level, j]] = state[level, j]
            _fwht(block)
            for high, v in enumerate(block):
                key = mask[level] | (high << level)
                if key:
                    used, threshold = offer(scores, keys, threshold, floor, used, abs(v), key)
                    visits += 1
            level -= 1
            continue
        b = next_bit[level]
        if b == 2:
            level -= 1
            continue
        next_bit[level] += 1
        upper = 0.0
        for j in range(sizes[level + 1]):
            a = state[level, left[level, j]]
            z = state[level, right[level, j]]
            v = a - z if b else a + z
            state[level + 1, j] = v
            upper += abs(v)
        if upper + slack < threshold:
            cuts += 1
        else:
            mask[level + 1] = mask[level] | (b << level)
            level += 1
            next_bit[level] = 0
    return used, visits, cuts


@njit(cache=True)
def _columns(x, keys):
    out = np.empty((len(keys), len(x)))
    for g, mask in enumerate(keys):
        for i in range(len(x)):
            v = x[i] & mask
            # Fold the parity of an int64 down to four bits.
            v ^= v >> 32
            v ^= v >> 16
            v ^= v >> 8
            v ^= v >> 4
            out[g, i] = 1.0 - 2.0 * ((0x6996 >> (v & 15)) & 1)
    return out.T


class Walsh:
    def __init__(self, bits):
        n, d = bits.shape
        if not 1 <= d <= 62 or not np.all((bits == 0) | (bits == 1)):
            raise ValueError("expected one to 62 Boolean inputs")
        self.n, self.d = n, d
        self.x = bits.astype(np.int64) @ (1 << np.arange(d, dtype=np.int64))
        self.folds = None

    def columns(self, keys):
        return _columns(self.x, np.asarray(keys, np.int64))

    def explicit(self):
        cols = (1 << self.d) - 1
        if 8 * self.n * cols > 2**31:
            raise MemoryError("explicit Walsh design exceeds 2 GiB benchmark limit")
        keys = np.arange(1, cols + 1, dtype=np.int64)
        return self.columns(keys), keys

    def search(self, r, floor, k):
        if self.folds is None:
            unique, inverse = np.unique(self.x, return_inverse=True)
            n = len(unique)
            suffix = np.zeros((self.d + 1, n + 1), np.int64)
            sizes = np.empty(self.d + 1, np.int64)
            left = np.full((self.d, n), n, np.int64)
            right = np.full_like(left, n)
            for i in range(self.d + 1):
                suffix[i, : len(unique)], sizes[i] = unique, len(unique)
                if i < self.d:
                    parents, where = np.unique(unique >> 1, return_inverse=True)
                    even = (unique & 1) == 0
                    left[i, where[even]] = np.flatnonzero(even)
                    right[i, where[~even]] = np.flatnonzero(~even)
                    unique = parents
            self.folds = (inverse, suffix, sizes, left, right)
        inverse, suffix, sizes, left, right = self.folds
        scores, keys = np.zeros(k), np.empty(k, np.int64)
        used, visits, cuts = _search(
            np.bincount(inverse, weights=r),
            suffix,
            sizes,
            left,
            right,
            self.d,
            floor,
            scores,
            keys,
            bound_slack(r),
        )
        return sorted(zip(scores[:used], keys[:used]), reverse=True), visits, cuts


class WalshTransform(Walsh):
    def __init__(self, dictionary):
        self.x, self.n, self.d = dictionary.x, dictionary.n, dictionary.d

    def search(self, r, floor, k):
        if self.d > 26:
            raise MemoryError("full-transform baseline limited to 26 bits")
        c = np.bincount(self.x, weights=r, minlength=1 << self.d)
        _fwht(c)
        c = np.abs(c)
        c[0] = 0.0
        idx = np.flatnonzero(c > floor)
        if len(idx) > k:
            idx = idx[np.argpartition(c[idx], -k)[-k:]]
        return sorted(((c[j], j) for j in idx), reverse=True), len(c) - 1, 0
