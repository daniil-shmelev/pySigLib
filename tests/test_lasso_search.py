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


def _dense_log_design(paths, degree):
    dimension = paths.shape[-1]
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
    return words, dense[:, indices] * [factorial(len(w)) for w in words]


@pytest.mark.parametrize("degree,dimension,targets,jobs", [(2, 3, 2, 2), (4, 3, 2, 2), (6, 2, 1, 1)])
def test_signature_factor_tree_against_complete_design(degree, dimension, targets, jobs):
    rng = np.random.default_rng(9325)
    paths = rng.normal(size=(180, 10, dimension)).cumsum(axis=1)
    paths[0] = 0.
    variation = abs(np.diff(paths, axis=1)).max(axis=2).sum(axis=1)
    normalized = paths / np.where(variation, variation, 1.)[:, None, None]
    words = [w for k in range(1, degree + 1) for w in product(range(dimension), repeat=k)]
    design = pysiglib.sig(normalized, degree) * [factorial(len(w)) for w in words]
    # Prescribe one unique violating column in each search, including words
    # that tie with a repeated Lyndon factor up to their final letter.
    centered = design - design.mean(axis=0)
    dual, _, rank, _ = np.linalg.lstsq(centered.T, np.eye(len(words)), rcond=None)
    assert rank == len(words)
    for first in range(0, len(words), targets):
        cols = list(range(first, min(first + targets, len(words))))
        y = np.ascontiguousarray(dual[:, cols] * len(paths) + centered[:, cols])
        model = pysiglib.sig_lasso(paths, y, degree, 1., tree_order='lyndon', terminal_bound='projected',
                                  n_jobs=jobs, tol=1e-8)
        assert model['words'] == [words[c] for c in cols]
        np.testing.assert_allclose(model['signatures'], design[:, cols], atol=1e-13)
        np.testing.assert_allclose(model['coef'], np.eye(len(cols)), atol=3e-7)
        assert np.all(model['converged'])
        assert model['stats']['full_transform_calls'] == 0
        np.testing.assert_allclose(pysiglib.sig_lasso_predict(paths, model),
                                   design[:, cols] @ model['coef'] + model['intercept'], atol=1e-12)


@pytest.mark.parametrize("order,bound", [("prefix", "projected"), ("lyndon", "norm"), ("lyndon", "projected")])
@pytest.mark.parametrize("loss", ["squared", "logistic", "poisson", "huber"])
def test_signature_tree_order_dense_kkt(order, bound, loss):
    rng = np.random.default_rng(213)
    paths = rng.normal(size=(64, 8, 3)).cumsum(axis=1).astype(np.float32)
    paths[:8] = paths[:8, -1:, :] * np.linspace(0, 1, 8)[None, :, None]
    paths[8:16, -1] = paths[8:16, 0]
    paths[16] = 0.
    y = rng.normal(size=64).astype(np.float32)
    if loss == "logistic":
        y = (y > 0).astype(np.float32)
    elif loss == "poisson":
        y = rng.poisson(1.5, len(y)).astype(np.float32)
    alpha = .015
    model = pysiglib.sig_lasso(paths, y, 4, alpha, loss=loss, tree_order=order, terminal_bound=bound,
                               memory_limit=4 << 20, tol=1e-6, initial_words=[(2, 2, 0)])
    x = paths.astype(np.float64)
    v = abs(np.diff(x, axis=1)).max(axis=2).sum(axis=1)
    words = [w for k in range(1, 5) for w in product(range(3), repeat=k)]
    design = pysiglib.sig(x / np.where(v, v, 1.)[:, None, None], 4) * [factorial(len(w)) for w in words]
    cols = [words.index(w) for w in model["words"]]
    eta = design[:, cols] @ model["coef"] + model["intercept"]
    if loss == "logistic":
        residual = y - 1 / (1 + np.exp(-eta))
    elif loss == "poisson":
        residual = y - np.exp(eta)
    elif loss == "huber":
        residual = np.clip(y - eta, -1., 1.)
    else:
        residual = y - eta
    assert model["converged"]
    cor = design.T @ residual / len(y)
    assert np.max(abs(cor)) <= alpha + 2e-6
    np.testing.assert_allclose(cor[cols], alpha * np.sign(model["coef"]), atol=2e-6)
    assert abs(residual.mean()) <= 2e-6


def test_signature_tree_arguments():
    x = np.ones((3, 2, 1))
    y = np.ones(3)
    for order in ("prefix", "lyndon"):
        model = pysiglib.sig_lasso(x, y, 2, .1, tree_order=order)
        assert model["stats"]["terminal_bound"] == "norm"
    with pytest.raises(ValueError, match="tree_order"):
        pysiglib.sig_lasso(x, y, 2, .1, tree_order="log")
    with pytest.raises(ValueError, match="terminal_bound"):
        pysiglib.sig_lasso(x, y, 2, .1, terminal_bound="exact")


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_projected_tree_on_short_scalar_paths(length):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(40, length, 1)).cumsum(axis=1)
    y = rng.normal(size=len(x))
    model = pysiglib.sig_lasso(x, y, 5, .03, tree_order="lyndon", terminal_bound="projected")
    total = abs(np.diff(x[:, :, 0], axis=1)).sum(axis=1)
    delta = (x[:, -1, 0] - x[:, 0, 0]) / np.where(total, total, 1.)
    design = delta[:, None] ** np.arange(1, 6)
    cols = [len(w) - 1 for w in model["words"]]
    np.testing.assert_allclose(model["signatures"], design[:, cols], atol=1e-12)
    r = y - design[:, cols] @ model["coef"] - model["intercept"]
    assert model["converged"] and np.max(abs(design.T @ r)) / len(y) <= .03 + 1e-8


@pytest.mark.parametrize("dtype,jobs,limit,tol", [(np.float64, 2, None, 1e-8), (np.float32, 1, 4 << 20, 1e-6)])
def test_log_variation_search_against_dense_tensor_log(dtype, jobs, limit, tol):
    rng = np.random.default_rng(7201)
    degree, dimension = 5, 4
    paths = np.cumsum(rng.normal(size=(64, 8, dimension)), axis=1).astype(dtype)
    paths[:, :, 0] *= .001
    paths[:, :, 3] = 0
    paths[::4, -1] = paths[::4, 0]
    paths[0] = 0
    words, design = _dense_log_design(paths, degree)
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


def test_log_search_recovers_every_binary_lyndon_word():
    rng = np.random.default_rng(9342)
    paths = np.cumsum(rng.normal(size=(128, 10, 2)), axis=1)
    degree = 7
    words, design = _dense_log_design(paths, degree)
    centered = design - design.mean(axis=0)
    scales = design.std(axis=0)
    alpha = .1 * scales
    # Give each response a known single-word solution by prescribing its residual correlations.
    residual, _, rank, _ = np.linalg.lstsq(centered.T, len(paths) * np.diag(alpha), rcond=None)
    assert rank == len(words)
    y = centered / scales + residual
    model = pysiglib.log_sig_lasso(paths, y, degree, alpha, tol=1e-10, n_jobs=2)
    assert model["converged"].all()
    assert model["words"] == words
    assert model["stats"]["full_transform_calls"] == 0
    np.testing.assert_array_equal(model["coef"] != 0, np.eye(len(words), dtype=bool))
    np.testing.assert_allclose(model["coef"].diagonal() * scales, 1., atol=1e-7)
