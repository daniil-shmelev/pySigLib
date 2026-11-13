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

import numpy as np
import pytest
from scipy.special import expit

import pysiglib
from pysiglib.load_siglib import CPSIG
from benchmarks.benchmark_bb_tradeoff import METHODS, select_method
from benchmarks.lasso_feature_reference import dictionary


@pytest.fixture(autouse=True)
def reset_options():
    yield
    select_method('tt')
    CPSIG.sig_working_set_experimental_bound(0)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('jobs', [1, 8])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_complete_dictionary(method, jobs, dtype):
    rng = np.random.default_rng(927)
    paths = rng.normal(size=(70, 6, 3)).cumsum(1).astype(dtype)
    paths[0] = 0
    y = rng.normal(size=(70, 2)).astype(dtype)
    words, design = dictionary(paths, 5, 'sig')
    select_method(method)
    tol = 1e-5 if dtype == np.float32 else 1e-8
    model = pysiglib.sig_lasso(paths, y, 5, [.002, .007], n_jobs=jobs, tol=tol)
    assert model['converged'].all()
    indices = [words.index(w) for w in model['words']]
    np.testing.assert_allclose(model['signatures'], design[:, indices], atol=2e-7 if dtype == np.float32 else 1e-12)
    residual = y - model['intercept'] - design[:, indices] @ model['coef']
    corr = design.T @ residual / len(y)
    assert np.max(abs(corr) - [.002, .007]) < tol
    for j, index in enumerate(indices):
        assert np.max(np.where(model['coef'][j] != 0, abs(corr[index] - [.002, .007] * np.sign(model['coef'][j])), 0)) < tol
    select_method('tt')
    reference = pysiglib.sig_lasso(paths, y, 5, [.002, .007], n_jobs=jobs, tol=tol)
    np.testing.assert_allclose(model['objective'], reference['objective'], atol=2 * tol**2, rtol=1e-10)


@pytest.mark.parametrize('method', ['rank16_b1', 'rank16_order', 'rank16_shared', 'rank16_both', 'rank16_compact_adaptive', 'compact_adaptive_refresh25', 'compact_cache0_adaptive'])
@pytest.mark.parametrize('loss', ['squared', 'logistic', 'poisson', 'huber'])
@pytest.mark.parametrize('intercept', [False, True])
def test_losses(method, loss, intercept):
    select_method(method)
    rng = np.random.default_rng(441)
    paths = rng.normal(size=(80, 6, 3)).cumsum(1)
    y = rng.normal(size=80)
    if loss == 'logistic': y = (y > 0).astype(float)
    if loss == 'poisson': y = rng.poisson(1.5, size=80).astype(float)
    words, design = dictionary(paths, 3, 'sig')
    model = pysiglib.sig_lasso(paths, y, 3, .001, loss=loss, fit_intercept=intercept, n_jobs=8)
    assert model['converged']
    eta = model['intercept'] + model['signatures'] @ model['coef']
    residual = y - (expit(eta) if loss == 'logistic' else np.exp(eta) if loss == 'poisson' else eta)
    if loss == 'huber': residual = np.clip(residual, -1, 1)
    corr = design.T @ residual / len(y)
    assert max(abs(corr)) <= .001 + 1e-8
    for j, word in enumerate(model['words']):
        assert abs(corr[words.index(word)] - .001 * np.sign(model['coef'][j])) < 1e-8
    if intercept: assert abs(residual.mean()) < 1e-8


@pytest.mark.parametrize('method', ['rank16_b1', 'rank16_shared_b8', 'rank16_parts4', 'rank16_cache0', 'rank16_both', 'compact', 'rank16_compact', 'compact_cache0'])
def test_limits_and_ties(method):
    select_method(method)
    rng = np.random.default_rng(982)
    paths = rng.normal(size=(80, 5, 6)).cumsum(1)
    paths[:, :, 3:] = paths[:, :, :3]
    paths = paths[:, ::2]
    y = rng.normal(size=80)
    words, design = dictionary(paths, 3, 'sig')
    model = pysiglib.sig_lasso(paths, y, 3, .003, memory_limit=8 << 20, n_jobs=8)
    assert model['converged'] and model['stats']['peak_bytes'] <= 8 << 20
    corr = design.T @ (y - model['intercept'] - model['signatures'] @ model['coef']) / len(y)
    assert max(abs(corr)) <= .003 + 1e-8
    wide = rng.normal(size=(80, 4, 26))
    with pytest.warns(RuntimeWarning, match='KKT'):
        result = pysiglib.sig_lasso(wide, y, 1, 1e-5, max_iter=1, n_jobs=8)
    assert not result['converged'] and result['kkt_violation'] > 1e-8
