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
from sklearn.linear_model import LassoLars
from threadpoolctl import threadpool_limits

import pysiglib
from pysiglib.load_siglib import CPSIG, CUSIG, CUSIG_SIG_LASSO, _lasso_abi_matches


def fit_matrix(matrix, target, alpha, device, **kwargs):
    scale = max(1., np.max(np.abs(matrix)))
    normalized = matrix / scale
    extra = (1. - np.max(np.abs(normalized), axis=1)) / 2.
    excursion = normalized.copy()
    excursion[:, 0] += extra
    paths = np.stack((np.zeros_like(matrix), normalized, excursion, normalized), axis=1)
    target = target / scale
    alpha = alpha / scale**2
    if 'tol' in kwargs:
        kwargs['tol'] /= scale**2
    if device == 'cuda':
        if not torch.cuda.is_available() or CUSIG_SIG_LASSO is None:
            pytest.skip('CUDA Lasso ABI v3 is unavailable')
        paths = torch.as_tensor(paths, device=device)
        target = torch.as_tensor(target, device=device)
    result = pysiglib.sig_lasso(paths, target, 1, alpha, fit_intercept=False, **kwargs)
    beta = np.zeros(matrix.shape[1])
    values = result['coef'].cpu().numpy() if device == 'cuda' else result['coef']
    for word, value in zip(result['words'], values):
        beta[word[0]] = value
    return result, beta


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('method', ['full', 'sparse'])
@pytest.mark.parametrize('seed', range(12))
def test_matrix_reference(device, method, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(64, 85)) * np.exp(rng.uniform(-2, 2, size=85))
    target = matrix[:, :8] @ rng.normal(size=8) + .03 * rng.normal(size=64)
    alpha = .02 * max(abs(matrix.T @ target)) / len(target)
    result, beta = fit_matrix(matrix, target, alpha, device, method=method, tol=1e-10)
    with threadpool_limits(limits=1):
        reference = LassoLars(alpha=alpha, fit_intercept=False, max_iter=10000).fit(matrix, target)
    assert bool(result['converged'])
    np.testing.assert_allclose(matrix @ beta, reference.predict(matrix), atol=2e-7, rtol=2e-8)
    residual = target - matrix @ beta
    assert np.max(abs(matrix.T @ residual)) / len(target) <= alpha + 1e-8
    assert int(result['n_iter']) >= len(result['words'])
    assert (result['stats']['full_transform_calls'] > 0) == (method == 'full')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('seed', range(4))
def test_rank_deficient_matrix(device, seed):
    rng = np.random.default_rng(100 + seed)
    matrix = rng.normal(size=(40, 7)) @ rng.normal(size=(7, 30))
    matrix = np.column_stack((matrix, matrix[:, 0], -matrix[:, 0], np.zeros(40)))
    target = rng.normal(size=40)
    alpha = .03 * max(abs(matrix.T @ target)) / len(target)
    result, beta = fit_matrix(matrix, target, alpha, device, method='full', tol=1e-9)
    residual = target - matrix @ beta
    assert bool(result['converged'])
    assert np.max(abs(matrix.T @ residual)) / len(target) <= alpha + 1e-8
    assert result['stats']['drops'] > 0
    if seed == 0:
        assert result['stats']['singular_steps'] > 0


def test_empty_support_and_removed_options():
    paths = np.zeros((4, 2, 3))
    result = pysiglib.sig_lasso(paths, np.zeros(4), 2, 1.)
    assert result['n_iter'] == 0
    assert result['converged']
    assert result['stats']['activations'] == 0
    assert 'coordinate_sweeps' not in result['stats']
    with pytest.raises(TypeError, match='max_epochs'):
        pysiglib.sig_lasso(paths, np.zeros(4), 2, 1., max_epochs=10)


def test_stale_abi_rejected():
    assert not _lasso_abi_matches(None, '')
    assert not _lasso_abi_matches(object(), '')

    class Library:
        @staticmethod
        def sig_lasso_abi_version():
            return 1

    assert not _lasso_abi_matches(Library(), '')
    Library.sig_lasso_abi_version = staticmethod(lambda: 2)
    assert not _lasso_abi_matches(Library(), '')
    Library.sig_lasso_abi_version = staticmethod(lambda: 3)
    assert not _lasso_abi_matches(Library(), '')


def test_legacy_entrypoints_are_absent():
    for library, suffix in ((CPSIG, ''), (CUSIG, '_cuda')):
        if library is None:
            continue
        assert _lasso_abi_matches(library, suffix)
        assert not hasattr(library, 'sig_lasso' + suffix + '_d')
        assert not hasattr(library, 'sig_lasso_v2' + suffix + '_d')
        assert not hasattr(library, 'sig_lasso_shape' + suffix)
        assert not hasattr(library, 'sig_lasso_release' + suffix)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('method', ['full', 'sparse'])
@pytest.mark.parametrize('batch_size', [1, 128])
def test_equal_candidates_use_word_order(device, method, batch_size):
    vector = np.array([-2., -1., 1., 2.])
    matrix = np.column_stack((vector, vector, -vector, np.zeros(4)))
    result, beta = fit_matrix(matrix, vector, .1, device, method=method,
        initial_words=[(3,), (2,), (1,), (0,)], batch_size=batch_size, tol=1e-12)
    assert result['converged']
    assert result['words'] == [(0,)]
    np.testing.assert_allclose(matrix @ beta, .96 * vector, atol=1e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('method', ['full', 'sparse'])
def test_limit_still_certifies_and_initial_pool_is_not_support(device, method):
    matrix = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    result, beta = fit_matrix(matrix, 2 * matrix[:, 0], .1, device, method=method,
                             max_iter=1, initial_words=[(0,), (1,)], tol=1e-12)
    assert result['converged']
    assert result['n_iter'] == 1
    assert result['words'] == [(0,)]
    assert result['stats']['certificate_passes'] >= 2
    np.testing.assert_allclose(beta, [1.8, 0], atol=1e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_large_dictionary_has_no_dense_native_allocation(device):
    paths, target = np.zeros((3, 2, 24)), np.zeros(3)
    if device == 'cuda':
        if not torch.cuda.is_available() or CUSIG_SIG_LASSO is None:
            pytest.skip('CUDA Lasso ABI v3 is unavailable')
        paths, target = torch.as_tensor(paths, device=device), torch.as_tensor(target, device=device)
    result = pysiglib.sig_lasso(paths, target, 20, 1., memory_limit=32 * 1024**2, cache_bytes=0)
    assert result['converged']
    assert result['stats']['full_transform_calls'] == 0
    assert result['stats']['constructed'] == 0
    assert result['stats']['peak_bytes'] < 32 * 1024**2
    assert result['stats']['dictionary_size'] > 2**64


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_output_accuracy_failure_is_not_convergence(device):
    matrix = np.array([[1.], [2.], [-1.], [-2.]])
    with pytest.warns(RuntimeWarning, match='numerical accuracy'):
        result, _ = fit_matrix(matrix, matrix[:, 0], .1, device, tol=1e-30)
    assert not result['converged']
    assert result['dual_gap'] > 1e-30
