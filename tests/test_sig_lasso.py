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

import math

import numpy as np
import pytest
import torch
from sklearn.linear_model import Lasso
from benchmarks.lasso_feature_reference import dictionary, audit

import pysiglib


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


@pytest.mark.parametrize('device', ['numpy', 'cpu'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_returned_signatures(device, dtype):
    rng = np.random.default_rng(402)
    paths = rng.normal(size=(2, 12, 10, 4)).astype(dtype)[:, :, ::2, ::2]
    paths[0, 0] = 7
    target = rng.normal(size=(2, 12, 2)).astype(dtype)
    if device != 'numpy':
        paths = torch.tensor(paths, device=device, requires_grad=True)
        target = torch.tensor(target, device=device)
    result = pysiglib.sig_lasso(paths, target, 3, .01,
                               time_aug=True, lead_lag=True, initial_words=[(0, 1)])
    transformed = pysiglib.transform_path(as_numpy(paths).astype(np.float64), time_aug=True, lead_lag=True)
    flat = transformed.reshape(-1, *transformed.shape[-2:])
    variation = np.max(np.abs(np.diff(flat, axis=1)), axis=2).sum(1)
    normalized = (flat - flat[:, :1]) / np.where(variation > 0, variation, 1)[:, None, None]
    expected = pysiglib.sig_coef(normalized, result['words'])
    expected *= [math.factorial(len(word)) for word in result['words']]
    actual = as_numpy(result['signatures'])
    assert actual.shape == (2, 12, len(result['words']))
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=3e-7 if dtype == np.float32 else 5e-13)
    predicted = actual @ as_numpy(result['coef']) + as_numpy(result['intercept'])
    residual = as_numpy(target) - predicted
    objective = .5 * np.mean(residual**2, axis=(0, 1)) + .01 * np.sum(np.abs(as_numpy(result['coef'])), axis=0)
    np.testing.assert_allclose(objective, as_numpy(result['objective']), atol=2e-7 if dtype == np.float32 else 1e-12)
    assert result['stats']['output_bytes'] >= actual.nbytes
    assert result['stats']['peak_bytes'] >= result['stats']['output_bytes']
    assert result['stats']['total_seconds'] >= result['stats']['output_seconds'] >= 0
    if device != 'numpy':
        assert result['signatures'].device == paths.device
        assert not result['signatures'].requires_grad



@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('intercept', [False, True])
def test_rank_deficient_and_per_target_penalties(dtype, intercept):
    rng = np.random.default_rng(721)
    paths = rng.normal(size=(45, 2, 16)).astype(dtype)
    paths[:, :, 7] = paths[:, :, 3]
    paths[:, :, 15] = 0
    y = rng.normal(size=(45, 2)).astype(dtype)
    words, matrix = dictionary(paths, 2, 'sig')
    alpha = np.array([.02, .03])
    fit = pysiglib.sig_lasso(paths, y, 2, alpha, fit_intercept=intercept, initial_words=[(0,), (1, 2)])
    prediction = fit['intercept'] + fit['signatures'] @ fit['coef']
    for j in range(2):
        oracle = Lasso(alpha=alpha[j], fit_intercept=intercept, tol=1e-13, max_iter=100000).fit(matrix, y[:, j])
        np.testing.assert_allclose(prediction[:, j], oracle.predict(matrix), atol=3e-5)
    assert fit['converged'].all()
    assert max(audit(fit, matrix, words, y, alpha, intercept)['gap']) < 1e-5


def test_large_dictionary_and_degree_limit():
    paths = np.zeros((4, 2, 3))
    y = np.array([1., -1., 2., -2.])
    model = pysiglib.sig_lasso(paths, y, 45, 1., memory_limit=8 << 20)
    assert model['stats']['dictionary_size'] > 2**64
    assert model['converged'] and not model['words']
    with pytest.raises(ValueError, match='170'):
        pysiglib.sig_lasso(paths, y, 171, 1.)


def test_float32_output_overflow_raises():
    x = np.zeros((4, 2, 1), dtype=np.float32)
    with pytest.raises(RuntimeError, match='[Nn]onfinite'):
        pysiglib.sig_lasso(x, np.full(4, 1e30, dtype=np.float32), 1, 1., fit_intercept=False)


@pytest.mark.parametrize('bad', [dict(degree=0), dict(alpha=0), dict(tol=0), dict(initial_words=[(3,)]),
                                 dict(max_iter=0), dict(memory_limit=-1), dict(feature_type='unknown')])
def test_validation(bad):
    options = dict(degree=2, alpha=.1)
    options.update(bad)
    with pytest.raises((TypeError, ValueError)):
        pysiglib.sig_lasso(np.zeros((5, 3, 2)), np.zeros(5), **options)


def test_parallel_responses():
    rng = np.random.default_rng(601)
    paths = rng.normal(size=(48, 6, 4))
    y = rng.normal(size=(48, 3))
    models = [pysiglib.sig_lasso(paths, y, 3, [.02, .03, .04], n_jobs=jobs, tol=1e-10) for jobs in (1, 3)]
    assert models[0]['words'] == models[1]['words']
    np.testing.assert_allclose(models[0]['coef'], models[1]['coef'], atol=1e-10)
    assert models[1]['converged'].all()
