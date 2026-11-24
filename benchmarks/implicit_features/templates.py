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

"""Gaussian landmark and shapelet similarities with a ball-tree correlation bound."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from solver import Candidates, bound_slack


@dataclass
class Ball:
    centre: int
    radius: float
    indices: np.ndarray
    children: tuple


class Templates:
    def __init__(self, windows, centres, gamma, leaf=12, cache=64, profile="gaussian"):
        self.windows = np.ascontiguousarray(windows, dtype=float)
        self.centres = np.asarray(centres, dtype=float)
        self.gamma, self.leaf, self.cache_limit = gamma, leaf, cache
        self.profile = profile
        self.n = len(windows)
        self.cache = OrderedDict()
        self.root = None
        if (
            self.windows.ndim != 3
            or self.centres.ndim != 2
            or len(self.centres) == 0
            or self.windows.shape[-1] != self.centres.shape[1]
            or gamma <= 0
            or leaf < 2
            or profile not in ("gaussian", "distance")
        ):
            raise ValueError("invalid metric dictionary")

    def _ball(self, idx):
        points = self.centres[idx]
        mid = int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
        centre = int(idx[mid])
        radius = np.sqrt(np.max(np.sum((points - points[mid]) ** 2, axis=1)))
        if len(idx) <= self.leaf:
            return Ball(centre, radius, idx, ())
        dim = np.argmax(np.ptp(points, axis=0))
        order = idx[np.argsort(points[:, dim], kind="stable")]
        m = len(order) // 2
        return Ball(centre, radius, np.empty(0, np.int64), (self._ball(order[:m]), self._ball(order[m:])))

    def _dist(self, idx):
        n, t, d = self.windows.shape
        squared = cdist(self.windows.reshape(-1, d), self.centres[idx], metric="sqeuclidean")
        return np.sqrt(squared.reshape(n, t, -1).min(axis=1))

    def _pivot(self, j):
        if j in self.cache:
            self.cache.move_to_end(j)
            return self.cache[j]
        dist = self._dist([j])[:, 0]
        if self.cache_limit:
            self.cache[j] = dist
            if len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        return dist

    def columns(self, keys):
        out = np.empty((self.n, len(keys)), order="F")
        # Limit the sample-window by template temporary to about 16 MiB.
        block = max(1, (2**21) // (self.n * self.windows.shape[1]))
        for j in range(0, len(keys), block):
            d = self._dist(keys[j : j + block])
            out[:, j : j + block] = np.exp(-self.gamma * d * d) if self.profile == "gaussian" else d
        return out

    def explicit(self):
        keys = np.arange(len(self.centres))
        return self.columns(keys), keys

    def search(self, r, floor, k):
        if self.root is None:
            self.root = self._ball(np.arange(len(self.centres)))
        out, slack = Candidates(floor, k), bound_slack(r)
        pos, neg = np.maximum(r, 0.0), np.maximum(-r, 0.0)
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = self._pivot(node.centre)
            low, high = np.maximum(d - node.radius, 0.0), d + node.radius
            if self.profile == "gaussian":
                low, high = np.exp(-self.gamma * high**2), np.exp(-self.gamma * low**2)
            upper = max(pos @ high - neg @ low, neg @ high - pos @ low)
            if upper + slack < out.threshold:
                out.pruned += 1
                continue
            if node.children:
                stack.extend(node.children)
            else:
                cor = self.columns(node.indices).T @ r
                out.visited += len(cor)
                for j, value in zip(node.indices, np.abs(cor)):
                    out.add(value, int(j))
        return out.result()
