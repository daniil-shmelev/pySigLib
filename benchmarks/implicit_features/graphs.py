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

"""Connected, vertex-induced, labelled graph motifs via canonical augmentation."""

from collections import defaultdict
from functools import lru_cache
from itertools import permutations

import numpy as np
from scipy.sparse import csc_matrix
from solver import Candidates, bound_slack


@lru_cache(maxsize=8192)
def canonical(code):
    labels = [step[0] for step in code]
    n = len(labels)
    edges = [[0] * n for _ in labels]
    for j, step in enumerate(code):
        for i, edge in enumerate(step[1:]):
            edges[j][i] = edges[i][j] = edge
    best = None
    for perm in permutations(range(n)):
        trial = []
        for j, v in enumerate(perm):
            links = tuple(edges[v][i] for i in perm[:j])
            if j and not any(links):
                break
            trial.append((labels[v], *links))
            if best is not None and trial[-1] > best[j] and tuple(trial[:-1]) == best[:j]:
                break
        if len(trial) == n:
            key = tuple(trial)
            if best is None or key < best:
                best = key
    return best


class Graphs:
    def __init__(self, graphs, degree):
        self.graphs, self.n, self.degree = graphs, len(graphs), degree
        if not 1 <= degree <= 6:
            raise ValueError("this permutation canonicaliser supports one to six vertices")

    def _roots(self):
        roots = defaultdict(list)
        for i, (labels, _) in enumerate(self.graphs):
            for v, label in enumerate(labels):
                roots[((label,),)].append((i, (v,)))
        return roots

    def _extend(self, key, matches, wanted):
        children = defaultdict(list)
        for i, emb in matches:
            labels, adj = self.graphs[i]
            neighbours = set()
            for v in emb:
                neighbours.update(adj[v])
            neighbours.difference_update(emb)
            for v in neighbours:
                step = (labels[v], *(adj[v].get(t, 0) for t in emb))
                child = key + (step,)
                if wanted is not None:
                    if child not in wanted:
                        continue
                elif canonical(child) != child:
                    continue
                children[child].append((i, emb + (v,)))
        return children

    def _visit(self, callback, wanted=None):
        stack = list(self._roots().items())
        while stack:
            key, matches = stack.pop()
            if wanted is not None and key not in wanted:
                continue
            # Projected embeddings are grouped by input graph throughout the traversal.
            ids = np.fromiter((i for i, _ in matches), np.int32, count=len(matches))
            ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
            if callback(key, ids) and len(key) < self.degree:
                stack.extend(self._extend(key, matches, wanted).items())

    def search(self, r, floor, k):
        out, slack = Candidates(floor, k), bound_slack(r)

        def visit(key, ids):
            out.visited += 1
            weights = r[ids]
            total = weights.sum()
            out.add(abs(total), key)
            # max(sum positive, -sum negative) = (l1 + |sum|)/2.
            ok = (np.abs(weights).sum() + abs(total)) / 2 + slack >= out.threshold
            out.pruned += int(not ok and len(key) < self.degree)
            return ok

        self._visit(visit)
        return out.result()

    def columns(self, keys):
        positions = {key: j for j, key in enumerate(keys)}
        prefixes = {key[:d] for key in keys for d in range(1, len(key) + 1)}
        internal = {key[:-1] for key in prefixes}
        out = np.zeros((self.n, len(keys)), order="F")

        def visit(key, ids):
            if key in positions:
                out[ids, positions[key]] = 1.0
            return key in internal

        if keys:
            self._visit(visit, prefixes)
        return out

    def explicit(self):
        keys, indices, ptr = [], [], [0]

        def visit(key, ids):
            keys.append(key)
            indices.append(ids)
            ptr.append(ptr[-1] + len(ids))
            return True

        self._visit(visit)
        idx = np.concatenate(indices) if keys else np.empty(0, np.int32)
        return csc_matrix((np.ones(len(idx)), idx, ptr), shape=(self.n, len(keys))), keys
