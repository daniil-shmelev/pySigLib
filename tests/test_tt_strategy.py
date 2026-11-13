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
from benchmarks.benchmark_tt_strategy import METHODS, select_method, diagnostics
from benchmarks.lasso_feature_reference import audit, dictionary


@pytest.fixture(autouse=True)
def reset_options():
    yield
    CPSIG.sig_working_set_experimental_strategy(0)
    CPSIG.sig_working_set_experimental_pool(0)
    CPSIG.sig_working_set_experimental_bound(0)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('jobs', [1, 8])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_independent_dictionary(method, jobs, dtype):
    rng = np.random.default_rng(812)
    paths = rng.normal(size=(80, 6, 3)).cumsum(1).astype(dtype)
    y = rng.normal(size=(80, 2)).astype(dtype)
    paths[0] = 0
    words, design = dictionary(paths, 5, 'sig')
    tolerance = 1e-5 if dtype == np.float32 else 1e-8
    select_method(method)
    model = pysiglib.sig_lasso(paths, y, 5, [.002, .006], n_jobs=jobs, tol=tolerance)
    stats = diagnostics()
    assert model['converged'].all()
    result = audit(model, design, words, y, [.002, .006])
    assert result['columns_error'] < (2e-7 if dtype == np.float32 else 1e-11)
    residual = y - model['intercept'] - model['signatures'] @ model['coef']
    correlation = (design - design.mean(0)).T @ residual / len(y)
    assert np.max(abs(correlation) - [.002, .006]) < 2 * tolerance
    for j, word in enumerate(model['words']):
        active = model['coef'][j] != 0
        error = abs(correlation[words.index(word)] - np.array([.002, .006]) * np.sign(model['coef'][j]))
        assert np.max(np.where(active, error, 0)) < 2 * tolerance
    if METHODS[method] & 4:
        assert stats['final_audits'] >= 2 and stats['screened_entries'] > 0
    select_method('tt')
    reference = pysiglib.sig_lasso(paths, y, 5, [.002, .006], n_jobs=jobs, tol=tolerance)
    np.testing.assert_allclose(model['objective'], reference['objective'], atol=2 * tolerance**2, rtol=1e-10)


@pytest.mark.parametrize('method,feature', [('certificates', 'sig'), ('adaptive', 'sig'), ('certificates_adaptive', 'sig'),
                                          ('adaptive', 'log'), ('adaptive', 'hall')])
@pytest.mark.parametrize('loss', ['squared', 'logistic', 'poisson', 'huber'])
@pytest.mark.parametrize('intercept', [False, True])
def test_general_losses(method, feature, loss, intercept):
    select_method(method)
    rng = np.random.default_rng(481)
    paths = rng.normal(size=(100, 6, 3)).cumsum(1)
    y = rng.normal(size=100)
    if loss == 'logistic':
        y = (y > 0).astype(float)
    elif loss == 'poisson':
        y = rng.poisson(1.5, size=100).astype(float)
    words, design = dictionary(paths, 3, feature)
    model = pysiglib.sig_lasso(paths, y, 3, .0005, feature_type=feature, loss=loss, fit_intercept=intercept, tol=1e-8)
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


@pytest.mark.parametrize('method', ['certificates', 'adaptive', 'screening', 'all'])
@pytest.mark.parametrize('degree', [1, 6])
@pytest.mark.parametrize('intercept', [False, True])
def test_views_limits_and_degrees(method, degree, intercept):
    select_method(method)
    rng = np.random.default_rng(178)
    paths = rng.normal(size=(80, 7, 4)).cumsum(1)[:, ::2, ::2]
    y = rng.normal(size=80)
    words, design = dictionary(paths, degree, 'sig')
    model = pysiglib.sig_lasso(paths, y, degree, .003, fit_intercept=intercept, n_jobs=3, memory_limit=12 << 20)
    assert model['converged'] and model['stats']['peak_bytes'] <= 12 << 20
    correlation = design.T @ (y - model['intercept'] - model['signatures'] @ model['coef']) / len(y)
    assert abs(correlation).max() <= .003 + 1e-8
    prediction = pysiglib.sig_lasso_predict(paths, model, n_jobs=3)
    np.testing.assert_allclose(prediction, model['intercept'] + model['signatures'] @ model['coef'], atol=1e-12)


@pytest.mark.parametrize('method', ['certificates', 'adaptive', 'screening', 'all'])
def test_iteration_limit(method):
    select_method(method)
    rng = np.random.default_rng(409)
    paths = rng.normal(size=(80, 5, 26))
    y = rng.normal(size=80)
    with pytest.warns(RuntimeWarning, match='KKT'):
        model = pysiglib.sig_lasso(paths, y, 1, 1e-5, max_iter=1)
    assert not model['converged'] and model['n_iter'] == 1
    assert model['kkt_violation'] > 1e-8


@pytest.mark.parametrize('method,options', [('screening', {'loss': 'logistic'}), ('screening', {'feature_type': 'hall'}),
                                          ('certificates', {'feature_type': 'log'})])
def test_unsupported_combinations(method, options):
    select_method(method)
    with pytest.raises(ValueError):
        pysiglib.sig_lasso(np.zeros((4, 3, 2)), np.array([0., 1., 0., 1.]), 2, .01, **options)
