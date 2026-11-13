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
from scipy.special import expit

import pysiglib
from pysiglib.load_siglib import CPSIG
from benchmarks.benchmark_bb_tradeoff import select_method as reset_search
from benchmarks.benchmark_lasso_optimization import METHODS, optimization_stats, select_method
from benchmarks.lasso_feature_reference import dictionary


@pytest.fixture(autouse=True)
def reset_options():
    yield
    reset_search('tt')
    CPSIG.sig_working_set_experimental_bound(0)
    CPSIG.sig_working_set_experimental_kernels(0, 0)
    CPSIG.sig_working_set_experimental_summaries(0, 0)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('jobs', [1, 8])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_terminal_and_solver_kkt(method, jobs, dtype):
    rng = np.random.default_rng(1307)
    paths = rng.normal(size=(65, 6, 10)).cumsum(1).astype(dtype)
    paths[0] = 0
    y = rng.normal(size=(65, 2)).astype(dtype)
    words, design = dictionary(paths, 3, 'sig')
    select_method(method)
    if METHODS[method][0]: CPSIG.sig_working_set_experimental_search(4, 16, 4, 0, 3, 0)
    tol = 1e-5 if dtype == np.float32 else 1e-8
    alpha = np.array([.006, .012])
    model = pysiglib.sig_lasso(paths, y, 3, alpha, n_jobs=jobs, tol=tol)
    assert model['converged'].all()
    indices = [words.index(w) for w in model['words']]
    np.testing.assert_allclose(model['signatures'], design[:, indices], atol=2e-7 if dtype == np.float32 else 1e-12)
    prediction = design[:, indices]@model['coef']+model['intercept']
    corr = design.T@(y-prediction)/len(y)
    assert np.max(abs(corr)-alpha) < tol
    for j, index in enumerate(indices):
        assert np.max(np.where(model['coef'][j] != 0, abs(corr[index]-alpha*np.sign(model['coef'][j])), 0)) < tol
    np.testing.assert_allclose(pysiglib.sig_lasso_predict(paths, model, n_jobs=jobs), prediction,
                               atol=2e-6 if dtype == np.float32 else 1e-10, rtol=1e-6 if dtype == np.float32 else 1e-10)
    select_method('balanced')
    baseline = pysiglib.sig_lasso(paths, y, 3, alpha, n_jobs=jobs, tol=tol)
    np.testing.assert_allclose(model['objective'], baseline['objective'], atol=2e-7 if dtype == np.float32 else 2e-11)


@pytest.mark.parametrize('feature_type', ['sig', 'log', 'hall'])
@pytest.mark.parametrize('loss', ['squared', 'logistic', 'poisson', 'huber'])
@pytest.mark.parametrize('intercept', [False, True])
def test_accelerated_losses(feature_type, loss, intercept):
    rng = np.random.default_rng(817)
    paths = rng.normal(size=(90, 7, 3)).cumsum(1)
    y = rng.normal(size=90)
    if loss == 'logistic': y = (y > 0).astype(float)
    if loss == 'poisson': y = rng.poisson(1.5, size=90).astype(float)
    select_method('anderson_gram')
    model = pysiglib.sig_lasso(paths[:, ::2], y, 3, .001, feature_type=feature_type, loss=loss,
                              fit_intercept=intercept, n_jobs=8)
    words, design = dictionary(paths[:, ::2], 3, feature_type)
    assert model['converged']
    indices = [words.index(w) for w in model['words']]
    eta = design[:, indices]@model['coef']+model['intercept']
    residual = y-(expit(eta) if loss == 'logistic' else np.exp(eta) if loss == 'poisson' else eta)
    if loss == 'huber': residual = np.clip(residual, -1, 1)
    corr = design.T@residual/len(y)
    assert max(abs(corr)) <= .001+1e-8
    for j, index in enumerate(indices):
        assert abs(corr[index]-.001*np.sign(model['coef'][j])) < 1e-8
    if intercept: assert abs(residual.mean()) < 1e-8


@pytest.mark.parametrize('method', ['anderson_gram', 'block8', 'block16', 'block32', 'block_late', 'anderson_block16', 'cauchy', 'segment', 'operator'])
def test_ties_limits_and_cancellation(method):
    rng = np.random.default_rng(1113)
    paths = rng.normal(size=(60, 5, 10)).cumsum(1)
    paths[:, -1] = paths[:, 0]
    paths[:, :, 5:] = paths[:, :, :5]
    y = rng.normal(size=60)
    select_method(method)
    if METHODS[method][0]: CPSIG.sig_working_set_experimental_search(4, 16, 4, 0, 3, 0)
    model = pysiglib.sig_lasso(paths, y, 3, .005, n_jobs=8, memory_limit=4 << 20)
    assert model['converged'] and model['stats']['peak_bytes'] <= 4 << 20
    words, design = dictionary(paths, 3, 'sig')
    indices = [words.index(w) for w in model['words']]
    residual = y-model['intercept']-design[:, indices]@model['coef']
    assert np.max(abs(design.T@residual/len(y))) <= .005+1e-8


def test_acceleration_and_matrix_counters():
    rng = np.random.default_rng(97)
    paths = rng.normal(size=(100, 6, 12)).cumsum(1)
    y = rng.normal(size=(100, 2))
    select_method('anderson_block16_grouped')
    result = pysiglib.sig_lasso(paths, y, 3, [.002, .005], n_jobs=8)
    stats = optimization_stats()
    assert result['converged'].all()
    assert stats['anderson_attempts'] > 0 and stats['anderson_accepted'] > 0
    assert stats['matrix_blocks'] > 0
    assert result['stats']['full_transform_calls'] == 0


def test_invalid_native_configuration():
    select_method('balanced')
    assert CPSIG.sig_working_set_experimental_kernels(16, 0) != 0
    assert CPSIG.sig_working_set_experimental_kernels(0, 32) != 0


def test_covariance_acceleration_with_correlated_features():
    rng = np.random.default_rng(113)
    paths = rng.normal(size=(100, 6, 3)).cumsum(1)
    paths[:, :, 1:] = paths[:, :, :1] + .1*paths[:, :, 1:]
    y = rng.normal(size=100)
    words, design = dictionary(paths, 3, 'sig')
    select_method('anderson_eager_gram')
    model = pysiglib.sig_lasso(paths, y, 3, .0001, initial_words=words, n_jobs=1)
    stats = optimization_stats()
    assert stats['gram_builds'] > 0 and stats['anderson_accepted'] > 0
    assert model['converged']
    indices = [words.index(w) for w in model['words']]
    residual = y-design[:, indices]@model['coef']-model['intercept']
    corr = design.T@residual/len(y)
    assert np.max(abs(corr)) <= .0001+1e-8
    assert np.max(abs(corr[indices]-.0001*np.sign(model['coef']))) <= 1e-8


@pytest.mark.parametrize('closed', [False, True])
def test_terminal_norm_inequalities(closed):
    rng = np.random.default_rng(772)
    paths = rng.normal(size=(7, 5, 3)).cumsum(1)
    paths[0] = 0
    paths[:, :, 2] *= .03
    if closed: paths[:, -1] = paths[:, 0]
    variation = abs(np.diff(paths, axis=1)).max(2).sum(1)
    paths = (paths-paths[:, :1])/np.maximum(variation, 1e-30)[:, None, None]
    increments = np.diff(paths, axis=1)
    n, segments, d = increments.shape
    residual = rng.normal(size=n)
    prefixes = [()] + [w for k in range(1, 5) for w in product(range(d), repeat=k)]
    words = [w for k in range(1, 7) for w in product(range(d), repeat=k)]
    lookup = {w: j for j, w in enumerate(words)}
    tensor = pysiglib.sig(paths, 6)
    points, weights = np.polynomial.legendre.leggauss(3)
    points, weights = (points+1)/2, weights/2
    integrals = np.zeros((n, segments, len(prefixes)))
    linear = np.zeros_like(integrals)
    for j in range(segments):
        for u, weight in zip(points, weights):
            partial = np.concatenate((paths[:, :j+1], paths[:, j:j+1]+u*increments[:, j:j+1]), axis=1)
            values = pysiglib.sig(partial, 4, scalar_term=True)
            integrals[:, j] += weight*values
            linear[:, j] += u*weight*values
    tails = increments[:, ::-1].cumsum(1)[:, ::-1]
    channel_norms = np.sqrt((increments**2).sum((0, 1)))
    segment_norms = np.sqrt((increments**2).sum(0))
    for k, prefix in enumerate(prefixes):
        integral, moment = integrals[:, :, k], linear[:, :, k]
        z = residual[:, None, None]*(integral[:, :, None]*tails-moment[:, :, None]*increments)
        scale = factorial(len(prefix)+2)/n
        contraction = scale*np.einsum('nja,njb->ab', increments, z)
        indices = [lookup[prefix+(a, b)] for a in range(d) for b in range(d)]
        exact = scale*(residual@tensor[:, indices]).reshape(d, d)
        np.testing.assert_allclose(contraction, exact, atol=1e-12, rtol=1e-11)
        flat = scale*np.outer(channel_norms, np.sqrt((z**2).sum((0, 1))))
        segment = scale*segment_norms.T@np.sqrt((z**2).sum(0))
        squared = ((segments-1-np.arange(segments))*integral**2+(abs(integral)+abs(moment))**2).sum(1)
        operator = scale*np.max(abs(residual)*np.sqrt(squared))*np.outer(channel_norms, channel_norms)
        sample_norms = np.sqrt((increments**2).sum(1))
        sample_flat = scale*sample_norms.T@np.sqrt((z**2).sum(1))
        sample_operator = scale*sample_norms.T@((abs(residual)*np.sqrt(squared))[:, None]*sample_norms)
        exact_squared = ((segments-1-np.arange(segments))*integral**2+(integral-moment)**2).sum(1)
        signed_operator = scale*sample_norms.T@((abs(residual)*np.sqrt(exact_squared))[:, None]*sample_norms)
        assert np.all(abs(exact) <= flat+1e-12)
        assert np.all(abs(exact) <= segment+1e-12)
        assert np.all(abs(exact) <= operator+1e-12)
        assert np.all(abs(exact) <= sample_flat+1e-12)
        assert np.all(abs(exact) <= sample_operator+1e-12)
        assert np.all(abs(exact) <= signed_operator+1e-12)
        assert np.all(segment <= flat+1e-12)


@pytest.mark.parametrize('method', ['balanced', 'cauchy_grouped', 'segment_grouped', 'operator_grouped', 'operator_segment_grouped',
                                    'lazy_grouped', 'summary16_grouped', 'lazy_summary16_grouped', 'sample_operator_grouped', 'sample_both_grouped',
                                    'sample_signed_grouped', 'sample_letters_grouped', 'sample_fast_grouped', 'scalar16_grouped'])
def test_terminal_norm_underflow(method):
    endpoints = np.repeat([1., .999], 3)
    paths = np.zeros((6, 3, 2))
    paths[:, 1, 0] = 1
    paths[:, 2, 0] = 2*endpoints/(1+endpoints)
    y = np.repeat([1., 0.], 3)
    alpha = .125*(2-.999**169-.999**170)
    select_method(method)
    model = pysiglib.sig_lasso(paths, y, 170, alpha, n_jobs=1)
    assert model['converged']
    assert model['words'] == [(0,)*170]
    design = endpoints[:, None]**np.arange(1, 171)
    assert all(set(word) == {0} for word in model['words'])
    selected = [len(word)-1 for word in model['words']]
    np.testing.assert_allclose(model['signatures'], design[:, selected], atol=1e-12, rtol=1e-12)
    residual = y-model['intercept']-design[:, selected]@model['coef']
    assert np.max(abs(design.T@residual/len(y))) <= alpha+1e-8


@pytest.mark.parametrize('degree', range(1, 7))
@pytest.mark.parametrize('jobs', [1, 8])
def test_shared_query_reference(degree, jobs):
    from benchmarks.benchmark_lasso_response_queries import query
    rng = np.random.default_rng(1121)
    paths = rng.normal(size=(23, 5, 3)).cumsum(1)
    paths[0] = 0
    paths[1, -1] = paths[1, 0]
    residuals = rng.normal(size=(len(paths), 4))
    residuals -= residuals.mean(0)
    _, design = dictionary(paths, degree, 'sig')
    maximum = abs(design.T@residuals/len(paths)).max(0)
    thresholds = .2*maximum
    select_method('sample_both_grouped')
    for batch in [0, 1, 2, 4]:
        result = query(paths, residuals, thresholds, degree, batch, jobs)
        np.testing.assert_allclose(np.asarray(result['scores'])[:, 0], maximum, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize('loss', ['squared', 'logistic', 'poisson', 'huber'])
@pytest.mark.parametrize('method', ['lazy_grouped', 'summary16_grouped', 'sample_operator_grouped', 'sample_both_grouped', 'sample_signed_grouped', 'sample_letters_grouped', 'sample_fast_grouped', 'scalar16_grouped'])
def test_new_bounds_general_losses(loss, method):
    rng = np.random.default_rng(1109)
    paths = rng.normal(size=(51, 6, 4)).cumsum(1)
    y = rng.normal(size=(51, 3))
    if loss == 'logistic': y = (y > 0).astype(float)
    if loss == 'poisson': y = rng.poisson(2, size=y.shape).astype(float)
    select_method(method)
    if 'summary' in method: CPSIG.sig_working_set_experimental_summaries(3, 32768)
    if 'scalar' in method: CPSIG.sig_working_set_experimental_summaries(4, 32768)
    fit = pysiglib.sig_lasso(paths, y, 4, .005, loss=loss, n_jobs=1, memory_limit=8 << 20)
    assert fit['converged'].all() and fit['stats']['peak_bytes'] <= 8 << 20
    _, design = dictionary(paths, 4, 'sig')
    eta = fit['signatures']@fit['coef']+fit['intercept']
    residual = y-(expit(eta) if loss == 'logistic' else np.exp(eta) if loss == 'poisson' else eta)
    if loss == 'huber': residual = np.clip(residual, -1, 1)
    assert np.max(abs(design.T@residual/len(y))) <= .005+1e-8


@pytest.mark.parametrize('flags', [1, 2, 3, 4])
def test_sample_bounds_with_summary_eviction(flags):
    rng = np.random.default_rng(1137)
    paths = rng.normal(size=(47, 6, 7)).cumsum(1)
    paths[:, :, 4:] *= .03
    y = rng.normal(size=(47, 2))
    select_method('sample_signed_grouped')
    CPSIG.sig_working_set_experimental_summaries(flags, 32768)
    model = pysiglib.sig_lasso(paths, y, 4, .003, n_jobs=1)
    assert model['converged'].all()
    _, design = dictionary(paths, 4, 'sig')
    residual = y-model['signatures']@model['coef']-model['intercept']
    assert np.max(abs(design.T@residual/len(y))) <= .003+1e-8


@pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
def test_positive_reduction_extreme_gradients(scale):
    from benchmarks.benchmark_lasso_response_queries import query
    rng = np.random.default_rng(1153)
    paths = rng.normal(size=(35, 6, 4)).cumsum(1)
    residual = rng.normal(size=(35, 3))
    residual -= residual.mean(0)
    _, design = dictionary(paths, 4, 'sig')
    expected = abs(design.T@residual/len(paths)).max(0)
    select_method('sample_fast_grouped')
    result = query(paths, residual*scale, .2*expected*scale, 4, 0)
    np.testing.assert_allclose(np.asarray(result['scores'])[:, 0]/scale, expected, atol=1e-12, rtol=1e-11)
