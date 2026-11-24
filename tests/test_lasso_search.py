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

from itertools import product
from math import factorial

import numpy as np
import pytest

import pysiglib


@pytest.mark.parametrize("fit", [pysiglib.sig_lasso, pysiglib.log_sig_lasso])
def test_equal_scores_on_independent_columns(fit):
    dimension = 20
    basis = np.eye(dimension)
    endpoints = np.concatenate((basis, -basis))
    paths = np.stack((np.zeros_like(endpoints), endpoints), axis=1)
    y = endpoints.sum(axis=1)
    alpha = 0.5 / dimension

    with pytest.warns(RuntimeWarning, match="global KKT check"):
        first = fit(paths, y, 1, alpha, max_iter=1)
    assert not first["converged"]
    assert first["words"] == [(j,) for j in range(16)]

    model = fit(paths, y, 1, alpha)
    assert model["converged"]
    assert model["words"] == [(j,) for j in range(dimension)]
    np.testing.assert_allclose(model["coef"], 0.5)


@pytest.mark.parametrize("dtype,jobs,limit,tol", [(np.float64, 2, None, 1e-8), (np.float32, 1, 4 << 20, 1e-6)])
def test_log_variation_search_against_dense_tensor_log(dtype, jobs, limit, tol):
    rng = np.random.default_rng(7201)
    degree, dimension = 5, 4
    paths = np.cumsum(rng.normal(size=(64, 8, dimension)), axis=1).astype(dtype)
    paths[:, :, 0] *= .001
    paths[:, :, 3] = 0
    paths[::4, -1] = paths[::4, 0]
    paths[0] = 0
    x = paths.astype(np.float64)
    variation = np.abs(np.diff(x, axis=1)).max(axis=2).sum(axis=1)
    dense = pysiglib.log_sig(x / np.where(variation > 0, variation, 1)[:, None, None], degree, method=0)
    words, indices = [], []
    column = 0
    for k in range(1, degree + 1):
        for word in product(range(dimension), repeat=k):
            if all(word < word[j:] for j in range(1, k)):
                words.append(word)
                indices.append(column)
            column += 1
    design = dense[:, indices] * [factorial(len(w)) for w in words]
    y = design[:, [words.index((1, 1, 1, 2)), words.index((2,))]].astype(dtype, order="C")
    alpha = .05 * np.abs(design.T @ (y - y.mean(axis=0)) / len(y)).max(axis=0)
    model = pysiglib.log_sig_lasso(paths, y, degree, alpha, n_jobs=jobs, tol=tol, memory_limit=limit)
    assert model["converged"].all()
    assert model["stats"]["pruned"] > 0
    assert model["stats"]["full_transform_calls"] == 0
    selected = [words.index(w) for w in model["words"]]
    np.testing.assert_allclose(model["signatures"], design[:, selected], atol=tol)
    prediction = design[:, selected] @ model["coef"] + model["intercept"]
    np.testing.assert_allclose(pysiglib.sig_lasso_predict(paths, model), prediction, atol=tol)
    residual = y - prediction
    correlations = design.T @ residual / len(y)
    active = model["coef"] != 0
    assert np.max(np.abs(correlations) - alpha) <= 2 * tol
    np.testing.assert_allclose(residual.mean(axis=0), 0., atol=2 * tol)
    np.testing.assert_allclose(correlations[selected][active], (alpha * np.sign(model["coef"]))[active], atol=2 * tol)
