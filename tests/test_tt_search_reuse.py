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

import ctypes
from itertools import product
from math import factorial

import numpy as np
import pytest
from scipy.special import expit

import pysiglib
from pysiglib._lasso_types import LassoOptions
from pysiglib.load_siglib import CPSIG
from benchmarks.benchmark_tt_reuse import METHODS, select_method
from benchmarks.lasso_feature_reference import audit, dictionary


@pytest.fixture(autouse=True)
def reset_options():
    yield
    CPSIG.sig_working_set_experimental_bound(0)
    CPSIG.sig_working_set_experimental_pool(0)


@pytest.mark.parametrize('degree', [1, 3, 6])
def test_native_suffix_envelopes_and_subtrees(degree):
    n, segments, dimension = 5, 3, 2
    increments = np.random.default_rng(521).integers(-2, 3, size=(n, segments, dimension)).astype(float)
    increments[0] = 0
    paths = np.concatenate((np.zeros((n, 1, dimension)), increments.cumsum(1)), axis=1)
    variation = abs(increments).max(2).sum(1)
    increments /= np.where(variation > 0, variation, 1)[:, None, None]
    width = degree + 1
    matrices = np.zeros((n, dimension, segments * width, segments * width))
    for j in range(segments):
        for a in range(width):
            if a < degree:
                matrices[:, :, j * width + a, j * width + a + 1] = increments[:, j]
            for h in range(j + 1, segments):
                matrices[:, :, j * width + a, h * width + 1] = increments[:, h] / factorial(a)
    end = np.tile([1 / factorial(a) for a in range(width)], segments)
    options = LassoOptions(n, segments + 1, dimension, degree, 1, 3000, 0, 1e-8, 2, 1, 0, 0)
    function = CPSIG.sig_working_set_tt_envelopes_d
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes, function.restype = (pointer, ctypes.POINTER(LassoOptions), pointer), ctypes.c_int
    envelopes = {}
    for mode in ('tt', 'block2', 'suffix'):
        select_method(mode)
        output = np.empty((degree, width, segments, n))
        assert function(paths.ctypes.data_as(pointer), ctypes.byref(options), output.ctypes.data_as(pointer)) == 0
        envelopes[mode] = output.transpose(3, 0, 2, 1)
    positive = np.broadcast_to(end, (n, len(end))).copy()
    for q in range(degree):
        vectors = []
        for word in product(range(dimension), repeat=q):
            value = np.broadcast_to(end, (n, len(end))).copy()
            for c in reversed(word):
                value = np.einsum('nij,nj->ni', matrices[:, c], value)
            vectors.append(abs(value))
        exact = np.max(vectors, axis=0).reshape(n, segments, width)
        if q:
            positive = np.max(np.einsum('ncij,nj->nci', abs(matrices), positive), axis=1)
        valid = slice(1, degree - q + 1)
        np.testing.assert_allclose(envelopes['tt'][:, q, :, valid], positive.reshape(n, segments, width)[:, :, valid], atol=2e-14)
        np.testing.assert_allclose(envelopes['suffix'][:, q, :, valid], exact[:, :, valid], atol=2e-14)
        assert np.all(exact[:, :, valid] <= envelopes['block2'][:, q, :, valid] + 2e-14)
        assert np.all(envelopes['block2'][:, q, :, valid] <= envelopes['tt'][:, q, :, valid] + 2e-14)
    words, design = dictionary(paths, degree, 'sig')
    for word in words:
        state = np.zeros((n, segments * width))
        state[:, 0] = 1
        for c in word:
            state = np.einsum('ni,nij->nj', state, matrices[:, c])
        for q in range(degree - len(word) + 1):
            indices = [j for j, w in enumerate(words) if len(w) == len(word) + q and w[:len(word)] == word]
            actual = np.max(abs(design[:, indices]), axis=1)
            for mode, h in envelopes.items():
                bound = np.sum(abs(state.reshape(n, segments, width)) * h[:, q], axis=(1, 2)) * factorial(len(word) + q)
                assert np.all(actual <= bound + 2e-11), (mode, word, q)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('jobs', [1, 8])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_fit_against_full_dictionary(method, jobs, dtype):
    rng = np.random.default_rng(911)
    paths = rng.normal(size=(80, 6, 3)).cumsum(1).astype(dtype)
    y = rng.normal(size=(80, 2)).astype(dtype)
    words, design = dictionary(paths, 5, 'sig')
    select_method(method)
    tolerance = 1e-5 if dtype == np.float32 else 1e-8
    model = pysiglib.sig_lasso(paths, y, 5, .002, n_jobs=jobs, tol=tolerance)
    assert model['converged'].all()
    result = audit(model, design, words, y, .002)
    assert result['columns_error'] < (2e-7 if dtype == np.float32 else 1e-11)
    residual = y - model['intercept'] - model['signatures'] @ model['coef']
    correlation = (design - design.mean(0)).T @ residual / len(y)
    assert np.max(abs(correlation)) <= .002 + 2 * tolerance
    for j, word in enumerate(model['words']):
        active = model['coef'][j] != 0
        error = abs(correlation[words.index(word)] - .002 * np.sign(model['coef'][j]))
        assert np.max(np.where(active, error, 0)) < 2 * tolerance


@pytest.mark.parametrize('feature_type', ['sig', 'log', 'hall'])
@pytest.mark.parametrize('loss', ['squared', 'logistic', 'poisson', 'huber'])
@pytest.mark.parametrize('intercept', [False, True])
def test_pool_general_losses(feature_type, loss, intercept):
    select_method('pool64')
    rng = np.random.default_rng(481)
    paths = rng.normal(size=(100, 6, 3)).cumsum(1)
    y = rng.normal(size=100)
    if loss == 'logistic':
        y = (y > 0).astype(float)
    elif loss == 'poisson':
        y = rng.poisson(1.5, size=100).astype(float)
    words, design = dictionary(paths, 3, feature_type)
    model = pysiglib.sig_lasso(paths, y, 3, .0005, feature_type=feature_type, loss=loss, fit_intercept=intercept, tol=1e-8)
    assert model['converged']
    eta = model['intercept'] + model['signatures'] @ model['coef']
    residual = y - (expit(eta) if loss == 'logistic' else np.exp(eta) if loss == 'poisson' else eta)
    if loss == 'huber':
        residual = np.clip(residual, -1, 1)
    correlation = design.T @ residual / len(y)
    assert np.max(abs(correlation)) <= .0005 + 1e-8
    for j, word in enumerate(model['words']):
        assert abs(correlation[words.index(word)] - .0005 * np.sign(model['coef'][j])) < 1e-8
    if intercept:
        assert abs(residual.mean()) < 1e-8


@pytest.mark.parametrize('method', ['pool64', 'pool256'])
def test_iteration_limit_still_checks_the_full_tree(method):
    select_method(method)
    rng = np.random.default_rng(409)
    paths = rng.normal(size=(80, 5, 26))
    y = rng.normal(size=80)
    with pytest.warns(RuntimeWarning, match='KKT'):
        model = pysiglib.sig_lasso(paths, y, 1, 1e-5, max_iter=1, n_jobs=1)
    assert not model['converged'] and model['n_iter'] == 1
    assert model['stats']['search_passes'] == 2
    assert model['kkt_violation'] > 1e-8
