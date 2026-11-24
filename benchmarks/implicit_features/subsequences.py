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

"""Gapped subsequences with one earliest reachable endpoint per recording."""

import numpy as np
from numba import njit
from products import offer
from scipy.sparse import csc_matrix
from solver import bound_slack


@njit(cache=True)
def _next_table(tokens, offsets, alphabet):
    table = np.full((len(tokens), alphabet), -1, np.int32)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        for t in range(hi - 2, lo - 1, -1):
            table[t] = table[t + 1]
            table[t, tokens[t]] = t + 1
    return table


@njit(cache=True)
def _walk(table, starts, ends, depth, r, floor, scores, top, slack, mode, codes, ptr, rows):
    alphabet = table.shape[1]
    n = len(r)
    ids = np.empty((depth + 1, n), np.int32)
    pos = np.empty_like(ids)
    ids[0], pos[0] = np.arange(n), starts
    size = np.zeros(depth + 1, np.int64)
    size[0] = n
    letter = np.zeros(depth + 1, np.int64)
    prefix = np.zeros(depth + 1, np.int64)
    level = used = visited = pruned = cols = nnz = 0
    threshold = floor
    while level >= 0:
        j = letter[level]
        if j == alphabet:
            level -= 1
            continue
        letter[level] += 1
        visited += 1
        child = level + 1
        code = prefix[level] * (alphabet + 1) + j + 1
        count = 0
        cor = plus = minus = 0.0
        for p in range(size[level]):
            dest = table[pos[level, p], j]
            if dest < 0:
                continue
            i = ids[level, p]
            if mode == 0:
                cor += r[i]
                if dest < ends[i]:
                    if r[i] > 0:
                        plus += r[i]
                    else:
                        minus -= r[i]
            else:
                if mode == 2:
                    rows[nnz] = i
                nnz += 1
            ids[child, count], pos[child, count] = i, dest
            count += 1
        if count == 0:
            continue
        if mode == 0:
            used, threshold = offer(scores, top, threshold, floor, used, abs(cor), code)
        else:
            if mode == 2:
                codes[cols], ptr[cols + 1] = code, nnz
            cols += 1
        if child < depth:
            if mode == 0 and max(plus, minus) + slack < threshold:
                pruned += 1
            else:
                level = child
                letter[level], prefix[level], size[level] = 0, code, count
    return used, visited, pruned, cols, nnz


class Subsequences:
    def __init__(self, docs, alphabet, degree):
        if alphabet < 1 or degree < 1 or (alphabet + 1) ** degree > 2**63 - 1:
            raise ValueError("invalid gapped subsequence dimensions")
        self.n, self.degree, self.base = len(docs), degree, alphabet + 1
        offsets = np.r_[0, np.cumsum([len(row) + 1 for row in docs])].astype(np.int32)
        tokens = np.zeros(offsets[-1], np.int32)
        for begin, row in zip(offsets, docs):
            tokens[begin : begin + len(row)] = row
        if tokens.min(initial=0) < 0 or tokens.max(initial=0) >= alphabet:
            raise ValueError("invalid symbol")
        self.table = _next_table(tokens, offsets, alphabet)
        self.starts, self.ends = offsets[:-1], offsets[1:] - 1

    def _run(self, mode, r, floor, scores, keys, codes, ptr, rows):
        return _walk(
            self.table,
            self.starts,
            self.ends,
            self.degree,
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
        for j, key in enumerate(keys):
            symbols = []
            while key:
                key, tail = divmod(int(key), self.base)
                symbols.append(tail - 1)
            ids = np.arange(self.n)
            pos = self.starts
            for s in reversed(symbols):
                pos = self.table[pos, s]
                keep = pos >= 0
                ids, pos = ids[keep], pos[keep]
            out[ids, j] = 1.0
        return out

    def explicit(self):
        z = np.zeros(self.n)
        empty = np.empty(0, np.int64)
        args = (z, 0.0, np.empty(1), np.empty(1, np.int64))
        _, _, _, cols, nnz = self._run(1, *args, empty, empty, np.empty(0, np.int32))
        if 12 * nnz + 12 * cols > 2**31:
            raise MemoryError("explicit subsequence design exceeds 2 GiB benchmark limit")
        codes, ptr, rows = np.empty(cols, np.int64), np.zeros(cols + 1, np.int64), np.empty(nnz, np.int32)
        self._run(2, *args, codes, ptr, rows)
        return csc_matrix((np.ones(nnz), rows, ptr), shape=(self.n, cols)), codes

    def distinct(self):
        codes, rows, ptr = _distinct(self.table, self.starts, self.degree)
        return csc_matrix((np.ones(len(rows)), rows, ptr), shape=(self.n, len(codes))), codes


@njit(cache=True)
def _distinct(table, starts, depth):
    # A hash hit is verified against the entire sorted support.
    n, alphabet = len(starts), table.shape[1]
    ids = np.empty((depth + 1, n), np.int32)
    pos = np.empty_like(ids)
    ids[0], pos[0] = np.arange(n), starts
    size = np.zeros(depth + 1, np.int64)
    size[0] = n
    letter = np.zeros(depth + 1, np.int64)
    prefix = np.zeros(depth + 1, np.int64)
    buckets = {np.uint64(0): np.int64(0)}
    supports = [np.empty(0, np.int32)]
    codes, chain = [np.int64(0)], [np.int64(0)]
    level = 0
    while level >= 0:
        j = letter[level]
        if j == alphabet:
            level -= 1
            continue
        letter[level] += 1
        child = level + 1
        code = prefix[level] * (alphabet + 1) + j + 1
        count = 0
        fingerprint = np.uint64(14695981039346656037)
        for p in range(size[level]):
            dest = table[pos[level, p], j]
            if dest >= 0:
                i = ids[level, p]
                ids[child, count], pos[child, count] = i, dest
                fingerprint = (fingerprint ^ np.uint64(i + 1)) * np.uint64(1099511628211)
                count += 1
        if not count:
            continue
        # Numba infers an optional index for dict.get.
        first = buckets[fingerprint] if fingerprint in buckets else 0  # noqa: SIM401
        index = first
        while index != 0 and not np.array_equal(supports[index], ids[child, :count]):
            index = chain[index]
        if index == 0:
            buckets[fingerprint] = len(codes)
            codes.append(code)
            supports.append(ids[child, :count].copy())
            chain.append(first)
        if child < depth:
            level = child
            size[level], prefix[level], letter[level] = count, code, 0
    ptr = np.zeros(len(codes), np.int64)
    for i in range(1, len(codes)):
        ptr[i] = ptr[i - 1] + len(supports[i])
    rows = np.empty(ptr[-1], np.int32)
    for i in range(1, len(codes)):
        rows[ptr[i - 1] : ptr[i]] = supports[i]
    return np.array(codes[1:]), rows, ptr
