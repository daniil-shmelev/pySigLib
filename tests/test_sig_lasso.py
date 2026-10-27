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

import itertools
import math
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.linear_model import Lasso

import pysiglib
from conftest import DEVICES, skip_no_cuda


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def dense_data(paths, degree):
    paths = np.asarray(paths, dtype=np.float64)
    variation = np.max(np.abs(np.diff(paths, axis=1)), axis=2).sum(axis=1)
    normalized = (paths - paths[:, :1]) / np.where(variation > 0, variation, 1.)[:, None, None]
    matrix = pysiglib.sig(normalized, degree)
    weights = [math.factorial(k) for k in range(1, degree + 1)]
    words = [word for k in range(1, degree + 1) for word in itertools.product(range(paths.shape[-1]), repeat=k)]
    return matrix * np.array([weights[len(word) - 1] for word in words]), words


def audit(result, matrix, words, response, alpha, intercept=True):
    coefficients = np.zeros((len(words),) + as_numpy(result['coef']).shape[1:])
    positions = {word: j for j, word in enumerate(words)}
    for j, word in enumerate(result['words']):
        coefficients[positions[word]] = as_numpy(result['coef'])[j]
    predicted = matrix @ coefficients + as_numpy(result['intercept'])
    residual = response - predicted
    n = len(response)
    primal = np.sum(residual**2, axis=0) / (2 * n) + alpha * np.sum(abs(coefficients), axis=0)
    projected = residual - residual.mean(axis=0) if intercept else residual
    maximum = np.maximum(alpha, np.max(abs(matrix.T @ projected), axis=0) / n)
    theta = projected / n * alpha / maximum
    dual = np.sum(response * theta, axis=0) - n * np.sum(theta**2, axis=0) / 2
    np.testing.assert_allclose(primal, as_numpy(result['objective']), atol=2e-7, rtol=2e-7)
    return predicted, primal - dual


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('method', ['full', 'sparse'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_variation_normalization_against_dense_signature(device, method, dtype):
    rng = np.random.default_rng(874)
    paths = np.cumsum(rng.normal(size=(36, 6, 3)), axis=1)
    paths *= np.geomspace(1e-8, 1e8, len(paths))[:, None, None]
    paths[0] = 4.
    paths = paths.astype(dtype)
    matrix, words = dense_data(paths, 3)
    assert np.max(np.abs(matrix)) <= 1. + 2e-6
    response = (matrix[:, 0] + 2 * matrix[:, 15] + .01 * rng.normal(size=len(paths))).astype(dtype)
    x = torch.as_tensor(paths, device=device) if device == 'cuda' else paths
    y = torch.as_tensor(response, device=device) if device == 'cuda' else response
    result = pysiglib.sig_lasso(x, y, 3, .002, method=method, tol=1e-5 if dtype == np.float32 else 1e-10)
    predicted, gap = audit(result, matrix, words, response, .002)
    reference = Lasso(alpha=.002, tol=1e-13, max_iter=100000).fit(matrix, response)
    np.testing.assert_allclose(predicted, reference.predict(matrix), atol=1e-3 if dtype == np.float32 else 2e-5)
    assert bool(result['converged'])
    assert gap < (2e-5 if dtype == np.float32 else 2e-8)
    if method == 'sparse':
        assert result['stats']['full_transform_calls'] == 0


def test_removed_degree_weights():
    with pytest.raises(TypeError, match='degree_weights'):
        pysiglib.sig_lasso(np.zeros((3, 2, 1)), np.zeros(3), 1, .1, degree_weights=[1.])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_explicit_matrix_lasso(device, dtype, method):
    rng = np.random.default_rng(721)
    matrix = rng.normal(size=(45, 16))
    matrix[:, 7] = matrix[:, 3]
    matrix[:, 15] = 0
    y = (matrix[:, 2] - 0.5 * matrix[:, 4] + rng.normal(size=45) * 0.1).astype(dtype)
    paths = np.stack((np.zeros_like(matrix), matrix), axis=1).astype(dtype)
    source = torch.as_tensor(paths, device=device) if device == 'cuda' else paths
    response = torch.as_tensor(y, device=device) if device == 'cuda' else y
    result = pysiglib.sig_lasso(source, response, 1, 0.03, method=method, tol=1e-5 if dtype == np.float32 else 1e-10, batch_size=3)
    matrix, words = dense_data(paths, 1)
    oracle = Lasso(alpha=0.03, tol=1e-13, max_iter=100000).fit(matrix, y.astype(np.float64))
    predicted, gap = audit(result, matrix, words, y, 0.03)
    np.testing.assert_allclose(predicted, oracle.predict(matrix), atol=3e-4 if dtype == np.float32 else 2e-5)
    assert bool(result['converged'])
    assert gap < (2e-5 if dtype == np.float32 else 2e-8)
    assert as_numpy(result['coef']).dtype == dtype


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('cache', [0, 2048, 4 * 1024**2])
def test_sparse_dense_oracle(device, intercept, cache):
    rng = np.random.default_rng(982)
    paths = np.cumsum(rng.normal(size=(28, 7, 3)) * 0.2, axis=1)
    y = rng.normal(size=(28, 2)) + [0.5, -0.3]
    matrix, words = dense_data(paths, 3)
    alpha = np.array([0.02, 0.03])
    x = torch.as_tensor(paths, device=device) if device == 'cuda' else paths
    target = torch.as_tensor(y, device=device) if device == 'cuda' else y
    result = pysiglib.sig_lasso(x, target, 3, alpha, fit_intercept=intercept,
                               cache_bytes=cache, batch_size=4, initial_words=[(0,), (1, 2)], tol=1e-10)
    predicted, gap = audit(result, matrix, words, y, alpha, intercept)
    for j in range(2):
        oracle = Lasso(alpha=alpha[j], fit_intercept=intercept, tol=1e-13, max_iter=100000).fit(matrix, y[:, j])
        np.testing.assert_allclose(predicted[:, j], oracle.predict(matrix), atol=2e-5)
    assert as_numpy(result['converged']).all()
    assert np.max(gap) < 2e-8
    assert result['words'] == sorted(result['words'], key=lambda word: (len(word), word))


@pytest.mark.parametrize('device', DEVICES)
def test_screening_reset_after_growth(device):
    q, _ = np.linalg.qr(np.array([[1., 1.], [-1., 1.], [0., -2.], [0., 0.]]))
    matrix = 2 * q @ np.linalg.cholesky([[1., .8], [.8, 1.]]).T
    response = matrix @ np.linalg.solve([[1., .8], [.8, 1.]], [0., 2.])
    paths = np.stack([np.zeros_like(matrix), matrix], axis=1)
    if device == 'cuda':
        paths = torch.as_tensor(paths, device=device)
        response = torch.as_tensor(response, device=device)
    result = pysiglib.sig_lasso(paths, response, 1, .5, initial_words=[(0,)], batch_size=1, tol=1e-11)
    design, words = dense_data(as_numpy(paths), 1)
    prediction, gap = audit(result, design, words, as_numpy(response), .5)
    oracle = Lasso(alpha=.5, tol=1e-13, max_iter=100000).fit(design, as_numpy(response))
    np.testing.assert_allclose(prediction, oracle.predict(design), atol=2e-5)
    assert gap < 2e-8
    assert bool(result['converged'])


@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_transforms_batch_views_and_torch_cpu(method):
    rng = np.random.default_rng(62)
    x = torch.tensor(rng.normal(size=(2, 3, 5, 4)), dtype=torch.float64)[..., ::2]
    y = torch.tensor(rng.normal(size=(2, 3)), dtype=torch.float64)
    result = pysiglib.sig_lasso(x, y, 2, .3, method=method, time_aug=True, lead_lag=True, n_jobs=2, tol=1e-9)
    transformed = pysiglib.transform_path(x, time_aug=True, lead_lag=True)
    flat = transformed.reshape(6, transformed.shape[-2], transformed.shape[-1]).numpy()
    matrix, words = dense_data(flat, 2)
    _, gap = audit(result, matrix, words, y.numpy().ravel(), .3)
    assert gap < 2e-7
    assert result['coef'].device.type == 'cpu'
    assert not result['coef'].requires_grad


@pytest.mark.parametrize('device', DEVICES)
def test_huge_dictionary_prunes_without_materialising(device):
    x = np.zeros((4, 2, 3), dtype=np.float64)
    y = np.array([1., -1., 2., -2.])
    if device == 'cuda':
        x = torch.as_tensor(x, device=device)
        y = torch.as_tensor(y, device=device)
    with patch.object(pysiglib, 'sig', side_effect=AssertionError('dense transform used')):
        result = pysiglib.sig_lasso(x, y, 45, 1., cache_bytes=0, memory_limit=8 * 1024**2)
    assert result['stats']['dictionary_size'] > 2**64
    assert not result['words']
    assert result['stats']['evaluated'] == 0
    assert bool(result['converged'])


@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_constant_paths_and_response(method):
    result = pysiglib.sig_lasso(np.ones((6, 1, 2)), np.full(6, 4.), 3, .1, method=method)
    assert result['words'] == []
    assert result['coef'].shape == (0,)
    assert result['intercept'] == 4.
    assert result['dual_gap'] < 1e-10


@pytest.mark.parametrize('kwargs', [dict(alpha=0), dict(degree=0), dict(method='beam'), dict(tol=0),
    dict(batch_size=0), dict(initial_words=[(4,)]), dict(cache_bytes=-1)])
def test_validation(kwargs):
    arguments = dict(degree=2, alpha=.1)
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        pysiglib.sig_lasso(np.zeros((5, 3, 2)), np.zeros(5), **arguments)


def test_memory_and_iteration_limits():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(12, 5, 3)) * .2
    y = rng.normal(size=12)
    with pytest.raises(MemoryError):
        pysiglib.sig_lasso(x, y, 4, .01, memory_limit=128)
    with pytest.warns(RuntimeWarning):
        result = pysiglib.sig_lasso(x, y, 3, .001, max_iter=1)
    assert not result['converged']
    assert np.isfinite(result['objective']) and np.isfinite(result['dual_gap'])


@skip_no_cuda
@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_cuda_nondefault_stream(method):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        generator = torch.Generator(device='cuda').manual_seed(739)
        x = torch.randn(25, 6, 3, generator=generator, device='cuda', dtype=torch.float64) * .1
        y = 3 * (x[:, -1, 0] - x[:, 0, 0])
        result = pysiglib.sig_lasso(x, y, 2, .02, method=method, tol=1e-9)
        variation = torch.amax(torch.abs(torch.diff(x, dim=1)), dim=2).sum(dim=1)
        normalized = (x - x[:, :1]) / torch.where(variation > 0, variation, 1.)[:, None, None]
        predicted = result['intercept'] + (pysiglib.sig_coef(normalized, result['words']) * torch.tensor(
            [math.factorial(len(w)) for w in result['words']], device='cuda')) @ result['coef']
    stream.synchronize()
    assert torch.isfinite(predicted).all()
    assert bool(result['converged'])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_threshold_and_zero_columns(device, method):
    matrix = np.array([[-1., 0., 2.], [1., 0., 2.], [-1., 0., 2.], [1., 0., 2.]])
    paths = np.stack([np.zeros_like(matrix), matrix], axis=1)
    response = matrix[:, 0].copy()
    if device == 'cuda':
        paths, response = torch.as_tensor(paths, device=device), torch.as_tensor(response, device=device)
    result = pysiglib.sig_lasso(paths, response, 1, .5, method=method, tol=1e-12)
    assert result['words'] == []
    assert bool(result['converged'])
    result = pysiglib.sig_lasso(paths, response, 1, .5 - 1e-5, method=method, tol=1e-13)
    assert result['words'] == [(0,)]
    np.testing.assert_allclose(as_numpy(result['coef']), [4e-5], atol=1e-10)


@skip_no_cuda
@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_cuda_transforms_and_single_point(method):
    generator = torch.Generator(device='cuda').manual_seed(712)
    x = torch.randn((2, 3, 5, 4), generator=generator, device='cuda', dtype=torch.float32)[..., ::2]
    y = torch.randn((2, 3, 2), generator=generator, device='cuda', dtype=torch.float32)
    gpu = pysiglib.sig_lasso(x, y, 2, .3, method=method, time_aug=True, lead_lag=True)
    cpu = pysiglib.sig_lasso(x.cpu(), y.cpu(), 2, .3, method=method, time_aug=True, lead_lag=True)
    np.testing.assert_allclose(as_numpy(gpu['objective']), as_numpy(cpu['objective']), atol=2e-6)
    result = pysiglib.sig_lasso(torch.zeros(5, 1, 3, device='cuda'), torch.ones(5, device='cuda'), 3, .1, method=method)
    assert bool(result['converged']) and not result['words']
    np.testing.assert_allclose(as_numpy(result['intercept']), 1.)


@skip_no_cuda
def test_cuda_sparse_large_dictionary_and_failure_cleanup():
    x = torch.zeros(5, 3, 3, device='cuda', dtype=torch.float64)
    y = torch.arange(5, device='cuda', dtype=torch.float64)
    with pytest.raises(MemoryError):
        pysiglib.sig_lasso(x, y, 45, .1, memory_limit=128)
    result = pysiglib.sig_lasso(x, y, 45, .1, cache_bytes=0)
    assert bool(result['converged']) and result['stats']['constructed'] == 0
    assert result['stats']['dictionary_size'] > 2**64


@skip_no_cuda
def test_cuda_one_point_tiny_penalty():
    code = '''
import numpy as np
import torch
import pysiglib
for dtype in (torch.float32, torch.float64):
    paths = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]], [[7., 8.]]], dtype=dtype, device="cuda")
    y = torch.tensor([[1., 2.], [2., 4.], [4., 8.], [5., 10.]], dtype=dtype, device="cuda")
    for intercept in (False, True):
        result = pysiglib.sig_lasso(paths, y, 3, [1e-20, 1e-20], method="sparse", fit_intercept=intercept,
                                    initial_words=[(0,), (1, 0)], tol=1e-10)
        assert not result["words"] and bool(result["converged"].all())
        assert bool((result["n_iter"] == 0).all()) and result["stats"]["full_transform_calls"] == 0
        expected_intercept = [3., 6.] if intercept else [0., 0.]
        expected_objective = [1.25, 5.] if intercept else [5.75, 23.]
        np.testing.assert_allclose(result["intercept"].cpu().numpy(), expected_intercept)
        np.testing.assert_allclose(result["objective"].cpu().numpy(), expected_objective)
        assert bool(torch.isfinite(result["dual_gap"]).all())
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


@skip_no_cuda
def test_cuda_sparse_iteration_limit_stays_uncertified():
    rng = np.random.default_rng(7123)
    paths = np.cumsum(rng.normal(size=(24, 5, 4)) * .2, axis=1)
    response = rng.normal(size=24)
    with pytest.warns(RuntimeWarning, match='max_iter'):
        result = pysiglib.sig_lasso(torch.as_tensor(paths, device='cuda'), torch.as_tensor(response, device='cuda'),
                                    3, .001, method='sparse', max_iter=1, tol=1e-10)
    assert not bool(result['converged'])
    assert np.isfinite(as_numpy(result['objective'])) and np.isfinite(as_numpy(result['dual_gap']))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA devices required')
def test_cuda_device_isolation():
    outputs = []
    for index in (1, 0, 1):
        x = torch.arange(60, device=f'cuda:{index}', dtype=torch.float64).reshape(10, 2, 3) / 60
        y = torch.arange(10, device=x.device, dtype=x.dtype)
        result = pysiglib.sig_lasso(x, y, 2, .01)
        assert result['coef'].device == x.device
        outputs.append(float(result['objective']))
    np.testing.assert_allclose(outputs, outputs[0])


def test_thread_count_and_constructed_records():
    rng = np.random.default_rng(391)
    x = np.cumsum(rng.normal(size=(35, 7, 5)) * .08, axis=1)
    y = rng.normal(size=35)
    first = pysiglib.sig_lasso(x, y, 4, .01, cache_bytes=0, batch_size=3, n_jobs=1)
    second = pysiglib.sig_lasso(x, y, 4, .01, cache_bytes=100000, batch_size=3, n_jobs=4)
    np.testing.assert_allclose(first['coef'], second['coef'], atol=1e-10)
    assert first['words'] == second['words']
    assert first['stats']['columns'] <= first['stats']['constructed'] <= sum(5**k for k in range(1, 5))
    assert first['stats']['constructed'] <= first['stats']['evaluated']


@pytest.mark.parametrize('device', DEVICES)
def test_nonfinite_computation_raises(device):
    x = np.array([[[-1e308, 0.], [1e308, 0.]], [[1e308, 0.], [-1e308, 0.]]])
    y = np.array([1., -1.])
    if device == 'cuda':
        x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    with pytest.raises(RuntimeError, match='Nonfinite|nonfinite'):
        pysiglib.sig_lasso(x, y, 3, .1)


@skip_no_cuda
def test_cuda_root_batches_with_memory_pressure():
    rng = np.random.default_rng(572)
    increments = rng.normal(size=(24, 5, 40)) * .025
    increments[:, :, [5, 34, 38]] *= 8
    x = np.concatenate([np.zeros((24, 1, 40)), increments.cumsum(axis=1)], axis=1)
    y = x[:, -1, 34] - .7 * x[:, -1, 5] + .4 * x[:, -1, 38]
    reference = pysiglib.sig_lasso(x, y, 3, .003, method='full', tol=1e-10)
    gpu = pysiglib.sig_lasso(torch.as_tensor(x, device='cuda'), torch.as_tensor(y, device='cuda'),
        3, .003, initial_words=[(34,)], batch_size=2, memory_limit=64 * 1024**2, tol=1e-10)
    np.testing.assert_allclose(as_numpy(gpu['objective']), reference['objective'], atol=1e-10)
    assert bool(gpu['converged'])
    assert gpu['stats']['peak_bytes'] <= 64 * 1024**2


@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_parallel_independent_responses(method):
    rng = np.random.default_rng(601)
    x = rng.normal(size=(48, 6, 4)) * .1
    y = rng.normal(size=(48, 3))
    serial = pysiglib.sig_lasso(x, y, 3, [.02, .03, .04], method=method, n_jobs=1, tol=1e-10)
    parallel = pysiglib.sig_lasso(x, y, 3, [.02, .03, .04], method=method, n_jobs=3, tol=1e-10)
    assert serial['words'] == parallel['words']
    np.testing.assert_allclose(serial['coef'], parallel['coef'], atol=1e-12)
    assert parallel['converged'].all()


@pytest.mark.parametrize('device', DEVICES)
def test_float32_result_overflow_raises(device):
    x = np.zeros((4, 2, 1), dtype=np.float32)
    y = np.full(4, 1e30, dtype=np.float32)
    if device == 'cuda':
        x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    with pytest.raises(RuntimeError, match='Nonfinite|nonfinite'):
        pysiglib.sig_lasso(x, y, 1, 1., fit_intercept=False)


@skip_no_cuda
def test_cuda_long_path_segment_scan():
    rng = np.random.default_rng(772)
    x = np.cumsum(rng.normal(size=(7, 130, 3)) * .02, axis=1)
    y = rng.normal(size=7)
    cpu = pysiglib.sig_lasso(x, y, 3, .001, method='full', tol=1e-10)
    gpu = pysiglib.sig_lasso(torch.as_tensor(x, device='cuda'), torch.as_tensor(y, device='cuda'), 3, .001, tol=1e-10)
    np.testing.assert_allclose(as_numpy(gpu['objective']), cpu['objective'], atol=1e-10)
    assert bool(gpu['converged'])


@pytest.mark.parametrize('device', DEVICES)
def test_factorial_degree_limit(device):
    x, y = np.zeros((1, 2, 1)), np.zeros(1)
    if device == 'cuda':
        x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    with pytest.raises(ValueError, match='at most 170'):
        pysiglib.sig_lasso(x, y, 1000, 1., cache_bytes=0, memory_limit=16 * 1024**2)


@skip_no_cuda
@pytest.mark.parametrize('features', [37, 129])
@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_cuda_large_working_sets(features, method):
    rng = np.random.default_rng(425)
    matrix = rng.normal(size=(192, features))
    matrix -= matrix.mean(axis=0)
    matrix = np.linalg.qr(matrix)[0] * np.sqrt(len(matrix))
    coefficients = np.linspace(.5, 1.5, features)
    paths = np.stack([np.zeros_like(matrix), matrix], axis=1)
    matrix, _ = dense_data(paths, 1)
    response = matrix @ coefficients
    result = pysiglib.sig_lasso(torch.as_tensor(paths, device='cuda'),
        torch.as_tensor(response, device='cuda'), 1, .005, method=method, tol=1e-11)
    oracle = Lasso(alpha=.005, tol=1e-13, max_iter=100000).fit(matrix, response)
    prediction, gap = audit(result, matrix, [(j,) for j in range(features)], response, .005)
    assert bool(result['converged'])
    assert result['words'] == [(j,) for j in range(features)]
    assert gap < 2e-8
    np.testing.assert_allclose(prediction, oracle.predict(matrix), atol=2e-5)


@skip_no_cuda
@pytest.mark.parametrize('method', ['full', 'sparse'])
@pytest.mark.parametrize('perturbation', [1e-3, 1e-7])
def test_cuda_dependent_columns_multiple_responses(method, perturbation):
    rng = np.random.default_rng(1907)
    basis = rng.normal(size=(64, 5))
    matrix = np.column_stack((basis[:, 0], basis[:, 0] + perturbation * basis[:, 1], -basis[:, 0],
                              basis[:, 2], basis[:, 3], basis[:, 4]))
    response = np.column_stack((1.3 * basis[:, 0] - .9 * basis[:, 2] + .1 * basis[:, 3],
                                -.7 * basis[:, 0] + .6 * basis[:, 3] + .2 * basis[:, 4]))
    paths = np.stack((np.zeros_like(matrix), matrix), axis=1)
    matrix, _ = dense_data(paths, 1)
    alpha = np.array([.03, .04])
    reference = pysiglib.sig_lasso(paths, response, 1, alpha, method='full', tol=1e-10)
    result = pysiglib.sig_lasso(torch.as_tensor(paths, device='cuda'), torch.as_tensor(response, device='cuda'),
                                1, alpha, method=method, batch_size=2, tol=1e-10)
    predicted, gap = audit(result, matrix, [(j,) for j in range(matrix.shape[1])], response, alpha)
    reference_predicted, _ = audit(reference, matrix, [(j,) for j in range(matrix.shape[1])], response, alpha)
    assert as_numpy(result['converged']).all() and reference['converged'].all()
    assert np.max(gap) < 2e-8
    np.testing.assert_allclose(predicted, reference_predicted, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(as_numpy(result['objective']), reference['objective'], atol=1e-9, rtol=1e-9)


@skip_no_cuda
@pytest.mark.parametrize('strong_count', [40, 128])
def test_cuda_large_correlated_active_set_with_drops(strong_count):
    rng = np.random.default_rng(17)
    basis = rng.normal(size=(strong_count + 60, strong_count + 10))
    basis -= basis.mean(axis=0)
    basis = np.linalg.qr(basis)[0] * np.sqrt(len(basis))
    mixing = rng.normal(size=(10, 10))
    mixing /= np.linalg.norm(mixing, axis=0)
    matrix = np.column_stack((basis[:, :strong_count], basis[:, strong_count:] @ mixing))
    response = basis[:, :strong_count] @ np.full(strong_count, 2.) + matrix[:, strong_count:] @ (rng.normal(size=10) * .2)
    assert np.max(abs(matrix[:, strong_count:].T @ response) / len(response)) < 2
    scale = max(1., np.max(np.abs(matrix)))
    endpoint = matrix / scale
    excursion = endpoint.copy()
    excursion[:, 0] += (1. - np.max(np.abs(endpoint), axis=1)) / 2.
    paths = np.stack((np.zeros_like(matrix), endpoint, excursion, endpoint), axis=1)
    response = response / scale
    alpha = .03 / scale**2
    matrix, _ = dense_data(paths, 1)
    result = pysiglib.sig_lasso(torch.as_tensor(paths, device='cuda'), torch.as_tensor(response, device='cuda'),
                                1, alpha, method='full', tol=1e-10 / scale**2)
    oracle = Lasso(alpha=alpha, tol=1e-13, max_iter=100000).fit(matrix, response)
    predicted, gap = audit(result, matrix, [(j,) for j in range(matrix.shape[1])], response, alpha)
    assert bool(result['converged']) and len(result['words']) >= strong_count
    assert result['stats']['drops'] > 0
    assert gap < 2e-8
    np.testing.assert_allclose(predicted, oracle.predict(matrix), atol=2e-5, rtol=2e-5)


@skip_no_cuda
@pytest.mark.parametrize('cache', [0, 8 * 1024**2])
def test_cuda_wide_search_multiple_responses(cache):
    rng = np.random.default_rng(1827)
    increments = rng.normal(size=(32, 5, 48)) * .06
    increments[:, :, [3, 34, 41]] *= 3
    paths = np.concatenate((np.zeros((len(increments), 1, increments.shape[-1])),
                            np.cumsum(increments, axis=1)), axis=1)
    matrix, words = dense_data(paths, 2)
    positions = {word: j for j, word in enumerate(words)}
    response = np.column_stack((matrix[:, positions[(34,)]] + .7 * matrix[:, positions[(3, 41)]],
                                -.8 * matrix[:, positions[(41,)]] + .6 * matrix[:, positions[(34, 3)]]))
    alpha = np.array([.004, .005])
    reference = pysiglib.sig_lasso(paths, response, 2, alpha, method='full', tol=1e-10)
    result = pysiglib.sig_lasso(torch.as_tensor(paths, device='cuda'), torch.as_tensor(response, device='cuda'),
                                2, alpha, method='sparse', initial_words=[(34,)], batch_size=3,
                                cache_bytes=cache, memory_limit=128 * 1024**2, tol=1e-10)
    predicted, gap = audit(result, matrix, words, response, alpha)
    reference_predicted, _ = audit(reference, matrix, words, response, alpha)
    assert as_numpy(result['converged']).all() and reference['converged'].all()
    assert np.max(gap) < 2e-8
    assert result['stats']['full_transform_calls'] == 0
    assert result['stats']['peak_bytes'] <= 128 * 1024**2
    np.testing.assert_allclose(predicted, reference_predicted, atol=2e-5, rtol=2e-5)
