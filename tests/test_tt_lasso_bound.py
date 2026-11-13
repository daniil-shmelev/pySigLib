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

import pysiglib
from benchmarks.benchmark_tt_bound import BOUNDS, select_bound
from benchmarks.lasso_feature_reference import audit, dictionary


@pytest.fixture(autouse=True)
def reset_bound():
    yield
    select_bound('bernstein')


@pytest.mark.parametrize('degree', range(1, 7))
@pytest.mark.parametrize('jobs', [1, 8])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_full_dictionary_optimality(degree, jobs, dtype):
    rng = np.random.default_rng(891 + degree)
    paths = rng.normal(size=(35, 7, 3)).cumsum(1).astype(dtype)
    paths[0] = 0
    paths[1, -1] = paths[1, 0]
    y = rng.normal(size=(35, 2)).astype(dtype)
    words, design = dictionary(paths, degree, 'sig')
    alpha = .06 * np.max(np.abs((design - design.mean(0)).T @ (y - y.mean(0))), axis=0) / len(y)
    reference = None
    tolerance = 1e-5 if dtype == np.float32 else 1e-8
    for mode in BOUNDS:
        select_bound(mode)
        model = pysiglib.sig_lasso(paths, y, degree, alpha, n_jobs=jobs, tol=tolerance)
        result = audit(model, design, words, y, alpha)
        assert model['converged'].all()
        np.testing.assert_allclose(result['gap'], model['dual_gap'], atol=10 * np.finfo(dtype).eps if dtype == np.float32 else 1e-10, rtol=0)
        residual = y - model['signatures'] @ model['coef'] - model['intercept']
        correlations = (design - design.mean(0)).T @ residual / len(y)
        violation = np.maximum(0, np.max(np.abs(correlations), axis=0) - alpha)
        for j, word in enumerate(model['words']):
            active = model['coef'][j] != 0
            error = np.abs(correlations[words.index(word)] - alpha * np.sign(model['coef'][j]))
            violation = np.maximum(violation, np.where(active, error, 0))
        assert np.max(violation) < 2 * tolerance
        assert result['columns_error'] < (2e-7 if dtype == np.float32 else 1e-11)
        if reference is None:
            reference = model
        np.testing.assert_allclose(model['objective'], reference['objective'], atol=2e-7, rtol=0)


@pytest.mark.parametrize('mode', BOUNDS)
@pytest.mark.parametrize('intercept', [False, True])
def test_views_and_memory_limit(mode, intercept):
    select_bound(mode)
    rng = np.random.default_rng(812)
    paths = rng.normal(size=(25, 9, 4))[:, ::2, ::2]
    y = rng.normal(size=25)
    words, design = dictionary(paths, 6, 'sig')
    fit = pysiglib.sig_lasso(paths, y, 6, .003, fit_intercept=intercept, n_jobs=3, memory_limit=12 << 20, tol=1e-8)
    assert fit['converged']
    assert max(audit(fit, design, words, y, .003, intercept)['gap']) < 5e-8
    prediction = pysiglib.sig_lasso_predict(paths, fit)
    np.testing.assert_allclose(prediction, fit['signatures'] @ fit['coef'] + fit['intercept'], atol=1e-12)
