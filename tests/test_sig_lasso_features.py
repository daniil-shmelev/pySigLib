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

import pysiglib
from benchmarks.lasso_feature_reference import dictionary, audit


@pytest.mark.parametrize('mode', ['log', 'hall'])
@pytest.mark.parametrize('dimension,degree', [(2, 6), (3, 4)])
def test_all_coordinates(mode, dimension, degree):
    rng = np.random.default_rng(593)
    paths = rng.normal(size=(8, 6, dimension)).cumsum(1)
    paths[0] = 0
    paths[1, 1:] = paths[1, :1]
    paths[2, -1] = paths[2, 0]
    paths[3] = np.arange(6)[:, None] * np.arange(1, dimension + 1)
    words, expected = dictionary(paths, degree, mode)
    model = dict(words=words, coef=np.eye(len(words)), intercept=np.zeros(len(words)),
                 feature_type=mode, degree=degree, dimension=dimension, time_aug=False,
                 lead_lag=False, end_time=1., normalization='factorial_variation',
                 feature_convention='reverse_degree_hall_v1' if mode == 'hall' else mode)
    actual = pysiglib.sig_lasso_predict(paths, model, n_jobs=2)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    complete = pysiglib.sig_lasso(paths, np.zeros(len(paths)), degree, 1., feature_type=mode, initial_words=words)
    assert complete['converged']
    assert complete['stats']['dictionary_size'] == len(words)


@pytest.mark.parametrize('mode', ['sig', 'log', 'hall'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('intercept', [False, True])
def test_fit_and_prediction(mode, dtype, intercept):
    rng = np.random.default_rng(70)
    paths = rng.normal(size=(40, 7, 3)).cumsum(1).astype(dtype)
    paths[0] = 0
    y = rng.normal(size=(40, 2)).astype(dtype)
    words, design = dictionary(paths, 3, mode)
    options = dict(feature_type=mode, tol=1e-5 if dtype == np.float32 else 1e-9,
                   fit_intercept=intercept, max_iter=3000, n_jobs=2)
    sparse = pysiglib.sig_lasso(paths, y, 3, .01, **options)
    assert np.all(sparse['converged'])
    prediction = pysiglib.sig_lasso_predict(paths, sparse)
    np.testing.assert_allclose(prediction, sparse['signatures'] @ sparse['coef'] + sparse['intercept'], atol=2e-6)
    check = audit(sparse, design, words, y, .01, intercept)
    assert max(check['gap']) < (5e-5 if dtype == np.float32 else 2e-8)
    np.testing.assert_allclose(sparse['objective'], check['objective'], atol=2e-6 if dtype == np.float32 else 2e-11)
    selected = design[:, [words.index(w) for w in sparse['words']]]
    if intercept:
        selected = selected - selected.mean(0)
    active_error = selected.T @ (y - prediction) / len(paths) - .01 * np.sign(sparse['coef'])
    assert np.max(np.abs(active_error)[sparse['coef'] != 0], initial=0.) < (5e-5 if dtype == np.float32 else 2e-8)
    held_out = rng.normal(size=(7, 9, 3)).cumsum(1).astype(dtype)
    held_words, held_design = dictionary(held_out, 3, mode)
    expected = held_design[:, [held_words.index(w) for w in sparse['words']]] @ sparse['coef'] + sparse['intercept']
    np.testing.assert_allclose(pysiglib.sig_lasso_predict(held_out, sparse), expected,
                               atol=2e-6 if dtype == np.float32 else 2e-11, rtol=2e-6 if dtype == np.float32 else 2e-11)
    np.testing.assert_allclose(pysiglib.sig_lasso_predict(held_out[0].copy(), sparse), expected[0],
                               atol=2e-6 if dtype == np.float32 else 2e-11, rtol=2e-6 if dtype == np.float32 else 2e-11)


@pytest.mark.parametrize('jobs', [1, 3])
def test_hall_repeated_search(jobs):
    rng = np.random.default_rng(7103)
    paths = rng.normal(size=(37, 7, 3)).cumsum(1)
    words, design = dictionary(paths, 5, 'hall')
    y = design[:, [12, 34, 69]] + .02 * rng.normal(size=(37, 3))
    model = pysiglib.sig_lasso(paths, y, 5, .0005, feature_type='hall',
        n_jobs=jobs,
        tol=1e-10, max_iter=3000)
    assert np.all(model['converged'])
    assert model['stats']['search_passes'] > 1
    check = audit(model, design, words, y, .0005)
    assert max(check['gap']) < 2e-9
    assert check['columns_error'] < 2e-11
    expected = design[:, [words.index(w) for w in model['words']]] @ model['coef'] + model['intercept']
    np.testing.assert_allclose(pysiglib.sig_lasso_predict(paths, model, n_jobs=jobs), expected, atol=2e-11, rtol=2e-11)


def test_hall_simd_targets():
    import ctypes
    from pysiglib.load_siglib import CPSIG
    if not hasattr(CPSIG, 'cpsig_test_supported_targets'):
        pytest.skip('Native dispatch test hooks are unavailable')
    CPSIG.cpsig_test_supported_targets.restype = ctypes.c_int64
    CPSIG.cpsig_test_set_target.argtypes = (ctypes.c_int64,)
    paths = np.random.default_rng(7104).normal(size=(37, 7, 3)).cumsum(1)
    words, expected = dictionary(paths, 5, 'hall')
    model = dict(words=words, coef=np.eye(len(words)), intercept=np.zeros(len(words)),
        feature_type='hall', degree=5, dimension=3, time_aug=False, lead_lag=False,
        end_time=1., normalization='factorial_variation', feature_convention='reverse_degree_hall_v1')
    remaining = CPSIG.cpsig_test_supported_targets()
    try:
        while remaining:
            target = remaining & -remaining
            remaining &= ~target
            assert CPSIG.cpsig_test_set_target(target) == 0
            actual = pysiglib.sig_lasso_predict(paths, model, n_jobs=3)
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
            fitted = pysiglib.sig_lasso(paths, expected[:, [12, 34]], 5, .0005, feature_type='hall',
                n_jobs=3, tol=1e-10)
            assert np.all(fitted['converged'])
            assert max(audit(fitted, expected, words, expected[:, [12, 34]], .0005)['gap']) < 2e-9
    finally:
        assert CPSIG.cpsig_test_set_target(0) == 0


@pytest.mark.parametrize('mode', ['sig', 'log', 'hall'])
def test_transforms_torch_and_empty_prediction(mode):
    rng = np.random.default_rng(52)
    paths = torch.tensor(rng.normal(size=(2, 3, 5, 2)), dtype=torch.float64).transpose(0, 1)
    y = torch.tensor(rng.normal(size=(3, 2)), dtype=torch.float64)
    model = pysiglib.sig_lasso(paths, y, 2, .02, feature_type=mode, time_aug=True, lead_lag=True)
    prediction = pysiglib.sig_lasso_predict(paths, model)
    torch.testing.assert_close(prediction, (model['signatures'].reshape(6, -1) @ model['coef'] + model['intercept']).reshape(3, 2))
    assert pysiglib.sig_lasso_predict(paths[:0], model).shape == (0, 2)


@pytest.mark.parametrize('mode,word', [('log', (1, 0)), ('hall', (1, 0)), ('hall', (0, 0))])
def test_invalid_initial_features(mode, word):
    with pytest.raises(ValueError):
        pysiglib.sig_lasso(np.zeros((4, 3, 2)), np.zeros(4), 3, .1, feature_type=mode, initial_words=[word])


@pytest.mark.parametrize('mode', ['log', 'hall'])
def test_exact_rational_coordinates(mode):
    from fractions import Fraction
    from itertools import product
    from math import factorial
    from benchmarks.lasso_feature_reference import hall_basis
    degree = 6
    signature = {(): Fraction(1)}
    for increment in ((Fraction(1, 3), 0), (0, Fraction(2, 3))):
        exponential = {(): Fraction(1)}
        for n in range(1, degree + 1):
            for word in product(range(2), repeat=n):
                value = Fraction(1, factorial(n))
                for a in word:
                    value *= increment[a]
                exponential[word] = value
        combined = {}
        for u, a in signature.items():
            for v, b in exponential.items():
                if len(u) + len(v) <= degree:
                    combined[u + v] = combined.get(u + v, 0) + a * b
        signature = combined
    reduced = {w: a for w, a in signature.items() if w}
    logarithm, power = {}, {(): Fraction(1)}
    for k in range(1, degree + 1):
        combined = {}
        for u, a in power.items():
            for v, b in reduced.items():
                if len(u) + len(v) <= degree:
                    combined[u + v] = combined.get(u + v, 0) + a * b
        power = combined
        for w, a in power.items():
            logarithm[w] = logarithm.get(w, 0) + Fraction((-1)**(k + 1), k) * a
    words = [w for n in range(1, degree + 1) for w in product(range(2), repeat=n) if all(w < w[k:] for k in range(1, n))]
    expected = [float(logarithm.get(w, 0) * factorial(len(w))) for w in words]
    if mode == 'hall':
        words, duals = hall_basis(2, degree)
        expected = [float(sum(Fraction(c).limit_denominator() * logarithm.get(term, 0) for term, c in duals[w].items()) * factorial(len(w))) for w in words]
    model = dict(words=words, coef=np.eye(len(words)), intercept=np.zeros(len(words)), feature_type=mode,
        degree=degree, dimension=2, time_aug=False, lead_lag=False, end_time=1., normalization='factorial_variation',
        feature_convention='reverse_degree_hall_v1' if mode == 'hall' else mode)
    actual = pysiglib.sig_lasso_predict(np.array([[[0., 0.], [1/3, 0.], [1/3, 2/3]]]), model)
    np.testing.assert_allclose(actual[0], expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('mode', ['log', 'hall'])
def test_resources_and_missing_library(mode, monkeypatch):
    import importlib
    module = importlib.import_module('pysiglib.sig_lasso')
    paths = np.random.default_rng(9).normal(size=(12, 5, 4))
    y = np.arange(12, dtype=np.float64)
    with pytest.raises(MemoryError):
        pysiglib.sig_lasso(paths, y, 4, .01, feature_type=mode, memory_limit=128)
    monkeypatch.setattr(module, 'CPSIG_WORKING_SET', None)
    with pytest.raises(RuntimeError, match='rebuild'):
        pysiglib.sig_lasso(paths, y, 3, .01, feature_type=mode)


@pytest.mark.parametrize('mode', ['log', 'hall'])
def test_columns_and_cuda_rejection(mode, monkeypatch):
    paths = np.random.default_rng(28).normal(size=(16, 5, 3))
    y = paths[:, -1, 0]
    model = pysiglib.sig_lasso(paths, y, 3, .01, feature_type=mode)
    words, design = dictionary(paths, 3, mode)
    for j, word in enumerate(model['words']):
        np.testing.assert_allclose(model['signatures'][:, j], design[:, words.index(word)], atol=1e-12)
    if torch.cuda.is_available():
        import pysiglib.data_handlers as handlers
        monkeypatch.setattr(handlers, 'BUILT_WITH_CUDA', False)
        with pytest.raises(ValueError, match='CPU'):
            pysiglib.sig_lasso(torch.tensor(paths, device='cuda'), torch.tensor(y, device='cuda'), 3, .01, feature_type=mode)
        with pytest.raises(ValueError, match='CPU'):
            pysiglib.sig_lasso(paths, y, 3, torch.tensor(.01, device='cuda'), feature_type=mode)


def test_prediction_rejects_cuda_intercept():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    paths = torch.zeros((4, 2, 1), dtype=torch.float64)
    model = pysiglib.sig_lasso(paths, torch.zeros(4, dtype=torch.float64), 1, .1)
    model['intercept'] = model['intercept'].cuda()
    with pytest.raises(ValueError, match='CPU intercept'):
        pysiglib.sig_lasso_predict(paths, model)


def test_degree_one_agreement():
    rng = np.random.default_rng(87)
    paths = rng.normal(size=(20, 5, 3))
    y = rng.normal(size=20)
    fits = [pysiglib.sig_lasso(paths, y, 1, .01, feature_type=mode, tol=1e-10) for mode in ('sig', 'log', 'hall')]
    for model in fits[1:]:
        assert model['words'] == fits[0]['words']
        np.testing.assert_allclose(model['coef'], fits[0]['coef'], atol=1e-10)


@pytest.mark.parametrize('mode', ['log', 'hall'])
@pytest.mark.parametrize('degree', [4, 170])
def test_one_channel_and_zero_paths(mode, degree):
    paths = np.array([[[0.], [1.]], [[2.], [1.]], [[3.], [3.]], [[0.], [2.]]])
    y = np.array([1., -1., 0., 1.])
    model = pysiglib.sig_lasso(paths, y, degree, .01, feature_type=mode)
    assert model['stats']['dictionary_size'] == 1
    assert model['words'] == [(0,)]
    assert model['converged']
    constant = pysiglib.sig_lasso(np.ones((4, 1, 2)), np.ones(4), degree, .01, feature_type=mode)
    assert not constant['words'] and constant['converged']
    np.testing.assert_array_equal(pysiglib.sig_lasso_predict(np.ones((4, 1, 2)), constant, memory_limit=128), np.ones(4))


@pytest.mark.parametrize('mode', ['log', 'hall'])
def test_cache_eviction_respects_budget(mode):
    rng = np.random.default_rng(11)
    paths = rng.normal(size=(32, 8, 4)).cumsum(1)
    y = rng.normal(size=32)
    options = dict(feature_type=mode, tol=1e-9)
    reference = pysiglib.sig_lasso(paths, y, 4, .01, **options)
    limited = pysiglib.sig_lasso(paths, y, 4, .01, memory_limit=512 * 1024, **options)
    assert limited['converged']
    assert limited['stats']['peak_bytes'] <= 512 * 1024
    np.testing.assert_allclose(limited['signatures'] @ limited['coef'] + limited['intercept'],
                               reference['signatures'] @ reference['coef'] + reference['intercept'], atol=2e-8)
    words, design = dictionary(paths, 4, mode)
    assert max(audit(limited, design, words, y, .01)['gap']) < 2e-8
