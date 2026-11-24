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

"""Working-set Lasso with a pluggable exact correlation search."""

import heapq
import warnings
from dataclasses import dataclass

import numpy as np
from celer import Lasso
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning


class Candidates:
    def __init__(self, floor, limit):
        self.floor = floor
        self.limit = limit
        self.heap = []
        self.visited = 0
        self.pruned = 0

    @property
    def threshold(self):
        return self.floor if len(self.heap) < self.limit else self.heap[0][0]

    def add(self, value, key):
        if value <= self.threshold:
            return
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, (value, key))
        else:
            heapq.heapreplace(self.heap, (value, key))

    def result(self):
        return sorted(self.heap, reverse=True), self.visited, self.pruned


def bound_slack(r):
    # The dot products here are unnormalised.
    return 64 * np.finfo(float).eps * (1 + len(r)) * max(1.0, np.sum(np.abs(r)))


@dataclass
class Fit:
    keys: list
    coef: np.ndarray
    intercept: float
    objective: float
    rounds: int = 0
    visited: int = 0
    pruned: int = 0


def cd(matrix, y, alpha, initial=None, stationarity=8e-8):
    estimator = Lasso(alpha=alpha, fit_intercept=True, warm_start=True, max_iter=150, max_epochs=50_000)
    if initial is not None:
        estimator.coef_ = initial
    for tol in (1e-7, 1e-9, 1e-12, 1e-14):
        estimator.tol = tol
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(matrix, y)
        coef = estimator.coef_
        r = y - estimator.intercept_ - matrix @ coef
        corr = np.asarray(matrix.T @ r).ravel() / len(y)
        error = kkt_error(corr, coef, alpha, r)
        if np.isfinite(error) and error <= stationarity:
            return coef, float(estimator.intercept_), r
    raise ArithmeticError(f"coordinate descent KKT error {error:g}")


def kkt_error(corr, coef, alpha, r):
    active = coef != 0.0
    inactive = np.maximum(0.0, np.abs(corr[~active]) - alpha)
    active_error = np.abs(corr[active] - alpha * np.sign(coef[active]))
    return float(max(abs(r.mean()), inactive.max(initial=0.0), active_error.max(initial=0.0)))


def fit_tree(dictionary, y, alpha, batch=16, tol=1e-7, previous=None):
    if not np.isfinite(alpha) or alpha <= 0 or batch < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("expected positive penalty, batch size and numerical tolerance")
    if y.shape != (dictionary.n,) or not np.isfinite(y).all():
        raise ValueError("invalid response")
    keys = [] if previous is None else list(previous.keys)
    matrix = None
    coef = np.empty(0)
    intercept = float(y.mean())
    r = y - intercept
    if keys:
        matrix = dictionary.columns(keys)
        if np.count_nonzero(matrix) < 0.12 * matrix.size:
            matrix = sparse.csc_matrix(matrix)
        coef, intercept, r = cd(matrix, y, alpha, previous.coef.copy(), 0.8 * tol)
    visited = pruned = rounds = 0
    while True:
        candidates, visits, cuts = dictionary.search(r, (alpha + tol) * len(y), batch)
        visited += visits
        pruned += cuts
        if not candidates:
            break
        entering = [key for _, key in candidates if key not in keys]
        if not entering:
            raise ArithmeticError("restricted solver did not remove active violations")
        keys.extend(entering)
        block = dictionary.columns(entering)
        if matrix is None:
            if np.count_nonzero(block) < 0.12 * block.size:
                matrix = sparse.csc_matrix(block)
            else:
                matrix = np.asfortranarray(block)
        elif sparse.issparse(matrix):
            matrix = sparse.hstack((matrix, sparse.csc_matrix(block)), format="csc")
        else:
            matrix = np.asfortranarray(np.column_stack((matrix, block)))
        coef, intercept, r = cd(matrix, y, alpha, np.pad(coef, (0, len(entering))), 0.8 * tol)
        rounds += 1
    keep = np.flatnonzero(coef)
    obj = (r @ r) / (2 * len(y)) + alpha * np.abs(coef).sum()
    return Fit([keys[i] for i in keep], coef[keep], intercept, float(obj), rounds, visited, pruned)


def predict(dictionary, models):
    keys = list(dict.fromkeys(key for m in models for key in m.keys))
    positions = {key: i for i, key in enumerate(keys)}
    weights = np.zeros((len(keys), len(models)))
    for j, model in enumerate(models):
        weights[[positions[k] for k in model.keys], j] = model.coef
    return dictionary.columns(keys) @ weights + [m.intercept for m in models]


def audit(matrix, keys, y, alpha, fit):
    positions = {key: i for i, key in enumerate(keys)}
    coef = np.zeros(matrix.shape[1])
    coef[[positions[k] for k in fit.keys]] = fit.coef
    r = y - fit.intercept - matrix @ coef
    corr = np.asarray(matrix.T @ r).ravel() / len(y)
    return kkt_error(corr, coef, alpha, r)


def make_matrix(columns, n, use_sparse=False):
    if not columns:
        return sparse.csc_matrix((n, 0)) if use_sparse else np.empty((n, 0))
    if use_sparse:
        return sparse.hstack(columns, format="csc", dtype=float)
    return np.asfortranarray(np.column_stack(columns))
