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
import torch
from scipy.special import expit

import pysiglib
from pysiglib.load_siglib import CPSIG
from benchmarks.benchmark_lasso_general import configure
from benchmarks.benchmark_lasso_optimization import select_method
from benchmarks.lasso_feature_reference import dictionary


@pytest.fixture(autouse=True)
def reset_options():
    yield
    configure('control', 3)
    select_method('balanced')


@pytest.mark.parametrize('mode', ['log', 'hall'])
@pytest.mark.parametrize('degree', [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize('variant', ['log_simd', 'hall_reuse', 'hall_direct', 'kernels', 'groups', 'all', 'kernels_terminal'])
@pytest.mark.parametrize('jobs', [1, 8])
def test_complete_coordinates(mode, degree, variant, jobs):
    rng = np.random.default_rng(12109)
    paths = rng.normal(size=(37, 7, 2)).cumsum(1)
    paths[0] = 0
    paths[1, -1] = paths[1, 0]
    paths[2] = np.arange(7)[:, None]*np.array([1., -.5])
    words, expected = dictionary(paths, degree, mode)
    expected[2, [j for j, word in enumerate(words) if len(word) > 1]] = 0.
    configure(variant, degree)
    model = dict(words=words, coef=np.eye(len(words)), intercept=np.zeros(len(words)),
                 feature_type=mode, degree=degree, dimension=2, time_aug=False, lead_lag=False, end_time=1.,
                 normalization='factorial_variation', feature_convention='reverse_degree_hall_v1' if mode == 'hall' else mode)
    actual = pysiglib.sig_lasso_predict(paths, model, n_jobs=jobs)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('mode', ['sig', 'log', 'hall'])
@pytest.mark.parametrize('loss', ['squared', 'logistic', 'poisson', 'huber'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('variant', ['combined_ordered', 'final_trial', 'cost', 'hall_greedy'])
def test_general_fit(mode, loss, dtype, intercept, variant):
    rng = np.random.default_rng(31073)
    paths = rng.normal(size=(67, 7, 3)).cumsum(1).astype(dtype)
    paths[0] = 0
    words, design = dictionary(paths, 4, mode)
    y = 4*design[:, [0, 3]]+rng.normal(size=(67, 2))*.2
    if loss == 'logistic': y = rng.binomial(1, expit(y)).astype(float)
    if loss == 'poisson': y = rng.poisson(np.exp(y*.5)).astype(float)
    y = y.astype(dtype)
    alpha = np.array([.002, .005])
    tolerance = 1e-5 if dtype == np.float32 else 1e-8
    configure(variant, 4)
    result = pysiglib.sig_lasso(paths, y, 4, alpha, feature_type=mode, loss=loss, n_jobs=8,
                               tol=tolerance, fit_intercept=intercept)
    assert result['converged'].all()
    indices = [words.index(w) for w in result['words']]
    eta = design[:, indices]@result['coef']+result['intercept']
    residual = y-(expit(eta) if loss == 'logistic' else np.exp(eta) if loss == 'poisson' else eta)
    if loss == 'huber': residual = np.clip(residual, -1, 1)
    corr = design.T@residual/len(y)
    assert np.max(abs(corr)-alpha) < tolerance
    active_error = np.where(result['coef'] != 0, abs(corr[indices]-alpha*np.sign(result['coef'])), 0)
    assert np.max(active_error, initial=0.) < tolerance
    if intercept: assert max(abs(residual.mean(0))) < tolerance
    np.testing.assert_allclose(pysiglib.sig_lasso_predict(paths, result, n_jobs=1), eta,
                               atol=2e-6 if dtype == np.float32 else 2e-11, rtol=2e-6 if dtype == np.float32 else 2e-11)
    configure('control', 4)
    previous = pysiglib.sig_lasso(paths, y, 4, alpha, feature_type=mode, loss=loss, n_jobs=8,
                                 tol=tolerance, fit_intercept=intercept)
    np.testing.assert_allclose(result['objective'], previous['objective'], atol=2e-7 if dtype == np.float32 else 2e-10)


@pytest.mark.parametrize('mode', ['log', 'hall'])
@pytest.mark.parametrize('variant', ['log_simd', 'hall_reuse', 'hall_direct', 'kernels', 'groups', 'all', 'log_terminal', 'kernels_terminal', 'groups_terminal', 'log_moments', 'hall_first', 'combined', 'combined_groups', 'partial', 'hall_global', 'hall_simd', 'hall_global_simd', 'final_trial', 'hall_greedy'])
@pytest.mark.parametrize('cache', [0, 1 << 16, 16 << 20])
def test_cache_limits_and_repeated_queries(mode, variant, cache):
    rng = np.random.default_rng(6711)
    paths = rng.normal(size=(41, 9, 3)).cumsum(1)
    words, design = dictionary(paths[:, ::2], 5, mode)
    y = design[:, [12, 34, 69]] + .02*rng.normal(size=(41, 3))
    configure(variant, 5, cache)
    result = pysiglib.sig_lasso(paths[:, ::2], y, 5, .0005, feature_type=mode, n_jobs=1,
                               memory_limit=16 << 20, tol=1e-9)
    assert result['converged'].all() and result['stats']['peak_bytes'] <= 16 << 20
    indices = [words.index(w) for w in result['words']]
    residual = y-design[:, indices]@result['coef']-result['intercept']
    assert np.max(abs(design.T@residual/len(y))) <= .0005+1e-9
    assert result['stats']['full_transform_calls'] == 0


def test_invalid_feature_configuration():
    assert CPSIG.sig_working_set_experimental_features(2048) != 0


@pytest.mark.parametrize('jobs', [1, 8])
@pytest.mark.parametrize('dimension,degree', [(4, 4), (10, 3)])
def test_signature_suffix_groups(jobs, dimension, degree):
    rng = np.random.default_rng(72103)
    paths = rng.normal(size=(51, 7, dimension)).cumsum(1)
    paths[:, :, 3:] *= .07
    words, design = dictionary(paths, degree, 'sig')
    y = rng.normal(size=(51, 2))
    configure('sig_groups', degree)
    result = pysiglib.sig_lasso(paths, y, degree, [.001, .003], n_jobs=jobs, memory_limit=16 << 20)
    assert result['converged'].all()
    indices = [words.index(w) for w in result['words']]
    residual = y-design[:, indices]@result['coef']-result['intercept']
    assert np.max(abs(design.T@residual/len(y))-[.001, .003]) < 1e-8


@pytest.mark.parametrize('mode', ['sig', 'log', 'hall'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('tensor', [False, True])
@pytest.mark.parametrize('time_aug,lead_lag', [(False, False), (True, False), (False, True), (True, True)])
def test_optimized_path_transforms(mode, dtype, tensor, time_aug, lead_lag):
    rng = np.random.default_rng(89301)
    paths = rng.normal(size=(2, 17, 9, 2)).cumsum(2).astype(dtype)[:, :, ::2]
    y = rng.normal(size=(2, 17, 2)).astype(dtype)
    transformed = pysiglib.transform_path(paths.astype(np.float64), time_aug=time_aug, lead_lag=lead_lag)
    words, design = dictionary(transformed.reshape(-1, *transformed.shape[-2:]), 3, mode)
    configure('final_trial', 3)
    inputs = torch.tensor(paths, requires_grad=True) if tensor else paths
    target = torch.tensor(y) if tensor else y
    model = pysiglib.sig_lasso(inputs, target, 3, .01, feature_type=mode, time_aug=time_aug, lead_lag=lead_lag,
                              tol=1e-5 if dtype == np.float32 else 1e-8, n_jobs=8)
    assert np.asarray(model['converged']).all()
    prediction = pysiglib.sig_lasso_predict(inputs, model, n_jobs=1)
    if tensor:
        assert not prediction.requires_grad and prediction.device.type == 'cpu'
        prediction = prediction.numpy()
    indices = [words.index(w) for w in model['words']]
    expected = design[:, indices]@np.asarray(model['coef'])+np.asarray(model['intercept'])
    assert prediction.shape == (2, 17, 2)
    np.testing.assert_allclose(prediction.reshape(-1, 2), expected, atol=2e-6 if dtype == np.float32 else 2e-11, rtol=2e-6 if dtype == np.float32 else 2e-11)
