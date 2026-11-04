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

"""Dense references independent of the Lasso polynomial recurrences."""

from functools import lru_cache
from itertools import product
from math import factorial

import numpy as np
import pysiglib


@lru_cache(None)
def shuffle(u, v):
    if not u:
        return {v: 1}
    if not v:
        return {u: 1}
    result = {}
    for prefix, terms in ((u[:1], shuffle(u[1:], v)), (v[:1], shuffle(u, v[1:]))):
        for w, c in terms.items():
            result[prefix + w] = result.get(prefix + w, 0) + c
    return result


def hall_basis(dimension, degree):
    words = [(a,) for a in range(dimension)]
    keys = {w: (-1, w[0]) for w in words}
    factors, runs = {}, {w: 0 for w in words}
    duals = {w: {w: 1.} for w in words}
    for n in range(2, degree + 1):
        additions = []
        for u in words:
            for v in words:
                if len(u) + len(v) != n or keys[u] >= keys[v]:
                    continue
                if u in factors and keys[v] > keys[factors[u][1]]:
                    continue
                w = u + v
                run = runs[u] + 1 if u in factors and factors[u][1] == v else 1
                terms = {}
                for a, ca in duals[u].items():
                    for b, cb in duals[v].items():
                        for tail, count in shuffle(a[1:], b).items():
                            key = a[:1] + tail
                            terms[key] = terms.get(key, 0.) + ca * cb * count / run
                additions.append((w, u, v, run, terms))
        for w, u, v, run, terms in additions:
            if w in keys:
                raise AssertionError('Hall foliage collision')
            keys[w] = (-n, keys[u], keys[v])
            factors[w], runs[w], duals[w] = (u, v), run, terms
            words.append(w)
    return sorted(words, key=lambda w: (len(w), w)), duals


def dictionary(paths, degree, feature_type):
    paths = np.asarray(paths, dtype=np.float64)
    variation = np.abs(np.diff(paths, axis=-2)).max(axis=-1).sum(axis=-1)
    normalized = (paths - paths[..., :1, :]) / np.where(variation > 0, variation, 1)[..., None, None]
    d = paths.shape[-1]
    all_words = [w for n in range(1, degree + 1) for w in product(range(d), repeat=n)]
    if feature_type == 'sig':
        tensor = pysiglib.sig(normalized, degree)
        words = all_words
    else:
        tensor = pysiglib.log_sig(normalized, degree, method=0)
        words = [w for w in all_words if all(w < w[k:] for k in range(1, len(w)))]
    if feature_type == 'hall':
        words, duals = hall_basis(d, degree)
        index = {w: i for i, w in enumerate(all_words)}
        result = np.zeros((*paths.shape[:-2], len(words)))
        for j, w in enumerate(words):
            for term, coefficient in duals[w].items():
                result[..., j] += coefficient * tensor[..., index[term]]
    else:
        indices = {w: i for i, w in enumerate(all_words)}
        result = tensor[..., [indices[w] for w in words]].copy()
    result *= np.array([factorial(len(w)) for w in words])
    return words, result


def audit(model, design, words, y, alpha, fit_intercept=True):
    y = np.asarray(y).reshape(len(design), -1)
    coefficients = np.asarray(model['coef']).reshape(len(model['words']), y.shape[1])
    prediction = np.asarray(model['signatures']) @ coefficients + np.asarray(model['intercept']).reshape(-1)
    residual = y - prediction
    centered = design - design.mean(0) if fit_intercept else design
    corr = centered.T @ residual / len(design)
    alpha = np.broadcast_to(alpha, (y.shape[1],))
    maximum = np.max(np.abs(corr), axis=0)
    theta = residual - residual.mean(0) if fit_intercept else residual.copy()
    theta *= np.minimum(1, alpha / np.maximum(maximum, 1e-300)) / len(design)
    objective = .5 * np.mean(residual**2, axis=0) + alpha * np.abs(coefficients).sum(0)
    gap = objective - np.sum(y * theta, axis=0) + .5 * len(design) * np.sum(theta**2, axis=0)
    lookup = {w: j for j, w in enumerate(words)}
    columns_error = max((float(np.max(np.abs(model['signatures'][..., j] - design[:, lookup[w]]))) for j, w in enumerate(model['words'])), default=0.)
    return dict(gap=gap.tolist(), max_correlation=maximum.tolist(), objective=objective.tolist(), columns_error=columns_error)
