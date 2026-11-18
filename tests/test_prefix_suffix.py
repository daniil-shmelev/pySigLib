"""Independent coordinate checks for the default CPU prefix/suffix search."""
from contextlib import nullcontext

import numpy as np
import pytest
import pysiglib
from pysiglib.load_siglib import CPSIG
from benchmarks.benchmark_prefix_suffix import batches, columns, dictionary, normalize


@pytest.fixture(autouse=True)
def default_search():
    # Other experimental-search tests change these process-wide switches.
    assert CPSIG.sig_working_set_experimental_bound(1) == 0
    assert CPSIG.sig_working_set_experimental_pool(256) == 0
    assert CPSIG.sig_working_set_experimental_strategy(0) == 0
    assert CPSIG.sig_working_set_experimental_search(1, 16, 4, 0, 3, 0) == 0
    assert CPSIG.sig_working_set_experimental_summaries(0, 0) == 0
    assert CPSIG.sig_working_set_experimental_features(2039) == 0
    assert CPSIG.sig_working_set_experimental_kernels(16, 39) == 0


@pytest.mark.parametrize('mode,d,degree', [('sig', 5, 5), ('sig', 25, 2), ('log', 4, 6), ('log', 25, 3)])
@pytest.mark.parametrize('jobs,intercept', [(1, False), (4, True)])
def test_all_stationarity_conditions(mode, d, degree, jobs, intercept):
    rng = np.random.default_rng(1729)
    paths = rng.normal(size=(37, 8, d)).cumsum(axis=1)
    paths[:, :, 2:] *= np.geomspace(1., .02, d - 2)
    paths[0] = paths[0, 0]
    x = normalize(paths)
    y = columns(x, [(0,), (0, 1)], mode) @ np.array([[1., -1.], [2., 3.]]) + rng.normal(0., .01, (37, 2))
    alpha = np.array([.002, .003])
    model = pysiglib.sig_lasso(paths, y, degree, alpha, feature_type=mode, fit_intercept=intercept,
                              n_jobs=jobs, tol=1e-9, initial_words=[(0, 1)])
    assert model['converged'].all() and model['stats']['full_transform_calls'] == 0
    selected = columns(x, model['words'], mode)
    np.testing.assert_allclose(model['signatures'], selected, atol=2e-13)
    residual = y - (selected @ model['coef'] + model['intercept'])
    for batch in batches(dictionary(d, degree, mode), 64):
        corr = columns(x, batch, mode).T @ residual / len(x)
        assert np.max(abs(corr) - alpha) <= 1e-9
    active = model['coef'] != 0
    defect = selected.T @ residual / len(x) - np.sign(model['coef']) * alpha
    assert np.max(abs(defect[active])) <= 1e-9


@pytest.mark.parametrize('mode', ['sig', 'log'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_prediction_with_shared_prefixes_and_small_budget(mode, dtype):
    rng = np.random.default_rng(508)
    paths = rng.normal(size=(529, 11, 6)).cumsum(1).astype(dtype)
    model = pysiglib.sig_lasso(paths[:33].copy(), rng.normal(size=33).astype(dtype), 4, .1, feature_type=mode)
    # Deliberately unsorted, including a selected proper prefix and a duplicate.
    words = [(0, 1, 0, 2), (3,), (0, 1, 2), (0, 1), (0, 1), (1, 2, 2, 3)]
    beta = rng.normal(size=(len(words), 2)).astype(dtype)
    model.update(words=words, coef=beta, intercept=np.array([-.5, .3], dtype=dtype))
    expected = columns(normalize(paths.astype(np.float64)), words, mode) @ beta + model['intercept']
    predicted = pysiglib.sig_lasso_predict(paths, model, memory_limit=110_000, n_jobs=3)
    np.testing.assert_allclose(predicted, expected, atol=2e-7 if dtype == np.float32 else 2e-13)


@pytest.mark.parametrize('mode', ['sig', 'log'])
@pytest.mark.parametrize('updates', [1, 3000])
def test_batched_certificate_covers_each_response(mode, updates):
    rng = np.random.default_rng(731)
    paths = rng.normal(size=(67, 9, 5)).cumsum(1)
    paths[::13] = paths[::13, :1]
    x = normalize(paths)
    signal = columns(x, [(0,), (1, 4), (0, 2, 4)], mode)
    y = signal @ rng.normal(size=(3, 11)) + rng.normal(0., .07, size=(len(x), 11))
    # The last response finishes at the intercept. Other responses request
    # different numbers of separations and have different stationarity thresholds.
    y[:, -1] = 1.
    alpha = np.geomspace(.0002, .024, y.shape[1])
    warning = pytest.warns(RuntimeWarning, match="global KKT") if updates == 1 else nullcontext()
    with warning:
        model = pysiglib.sig_lasso(paths, y, 4, alpha, feature_type=mode, max_iter=updates, tol=1e-9, n_jobs=3)
    residual = y - model['signatures'] @ model['coef'] - model['intercept']
    lookup = {w: i for i, w in enumerate(map(tuple, model['words']))}
    violation = np.abs(residual.mean(axis=0))
    max_centered = np.zeros_like(alpha)
    for batch in batches(dictionary(paths.shape[-1], 4, mode), 37):
        design = columns(x, batch, mode)
        corr = design.T @ residual / len(x)
        centered_corr = corr - design.mean(axis=0)[:, None] * residual.mean(axis=0)
        max_centered = np.maximum(max_centered, np.abs(centered_corr).max(axis=0))
        coef = np.array([model['coef'][lookup[w]] if w in lookup else np.zeros_like(alpha) for w in batch])
        defect = np.where(coef != 0., np.abs(corr - alpha * np.sign(coef)), np.maximum(0., np.abs(corr) - alpha))
        violation = np.maximum(violation, defect.max(axis=0))
    assert np.all(violation <= model['kkt_violation'] + 2e-13)
    assert model['stats']['full_transform_calls'] == 0
    theta = (residual - residual.mean(axis=0)) / len(x) * np.minimum(1., alpha / np.maximum(max_centered, 1e-300))
    objective = .5 * np.mean(residual**2, axis=0) + alpha * np.abs(model['coef']).sum(axis=0)
    dual = np.sum(y * theta - .5 * len(x) * theta**2, axis=0)
    np.testing.assert_allclose(model['dual_gap'], objective - dual, atol=1e-11, rtol=1e-9)
    if updates == 1:
        assert not model['converged'].all()
    else:
        assert model['converged'].all() and violation.max() <= 1e-9


@pytest.mark.parametrize('mode', ['sig', 'log'])
def test_disparate_response_thresholds(mode):
    rng = np.random.default_rng(713)
    paths = rng.normal(size=(53, 7, 6)).cumsum(1)
    y = rng.normal(size=(len(paths), 2))
    alpha = [.002, 1e11]
    fit = pysiglib.sig_lasso(paths, y, 3, alpha, feature_type=mode, n_jobs=2)
    single = pysiglib.sig_lasso(paths, y[:, :1].copy(), 3, alpha[0], feature_type=mode, n_jobs=2)
    assert fit['converged'].all() and single['converged'].all()
    pred = fit['signatures'] @ fit['coef'] + fit['intercept']
    expected = (single['signatures'] @ single['coef'] + single['intercept']).ravel()
    np.testing.assert_allclose(pred[:, 0], expected, atol=1e-7)
    np.testing.assert_allclose(pred[:, 1], y[:, 1].mean(), atol=1e-13)


@pytest.mark.parametrize('degree,jobs', [(3, 4), (4, -1)])
def test_parallel_lyndon_search_matches_serial_certificate(degree, jobs):
    rng = np.random.default_rng(871 + degree)
    paths = rng.normal(size=(83, 12, 25)).cumsum(1)
    paths[:, :, 6:] *= np.geomspace(.15, 1e-6, 19)
    paths[::17] = paths[::17, :1]
    paths[1::11, 4:8] = paths[1::11, 3:4]
    x = normalize(paths)
    signal_words = [(0, 1), (1, 3, 4), (0, 0, 1, 3)[:degree]]
    design = columns(x, signal_words, 'log')
    y = (design / design.std(axis=0)) @ rng.normal(size=(3, 4))
    alpha = np.array([.0009, .0018, .003, .0006])
    common = dict(feature_type='log', tol=1e-9, max_iter=3000)
    serial = pysiglib.sig_lasso(paths, y, degree, alpha, n_jobs=1, **common)
    parallel = pysiglib.sig_lasso(paths, y, degree, alpha, n_jobs=jobs, **common)
    for model in (serial, parallel):
        assert model['converged'].all() and model['stats']['full_transform_calls'] == 0
    assert max(map(len, parallel['words'])) == degree
    selected = columns(x, parallel['words'], 'log')
    np.testing.assert_allclose(parallel['signatures'], selected, atol=2e-13)
    pred = selected @ parallel['coef'] + parallel['intercept']
    expected = serial['signatures'] @ serial['coef'] + serial['intercept']
    np.testing.assert_allclose(pred, expected, rtol=1e-9, atol=2e-8)
    np.testing.assert_allclose(parallel['dual_gap'], serial['dual_gap'], atol=1e-10)
    residual = y - pred
    if degree == 3:
        maximum = np.zeros_like(alpha)
        centered = residual - residual.mean(axis=0)
        for group in batches(dictionary(25, degree, 'log'), 64):
            corr = columns(x, group, 'log').T @ centered / len(x)
            maximum = np.maximum(maximum, np.abs(corr).max(axis=0))
        assert np.all(maximum - alpha <= 1e-9)
        theta = (centered / len(x)) * np.minimum(1., alpha / np.maximum(maximum, 1e-300))
        primal = .5 * np.mean(residual**2, axis=0) + alpha * np.abs(parallel['coef']).sum(axis=0)
        dual = np.sum(y * theta - .5 * len(x) * theta**2, axis=0)
        np.testing.assert_allclose(parallel['dual_gap'], primal - dual, atol=1e-10)


@pytest.mark.parametrize('mode', ['sig', 'log'])
def test_streamed_channels_with_subnormal_variation(mode):
    rng = np.random.default_rng(9216)
    paths = rng.integers(-1, 2, size=(53, 9, 27)).cumsum(axis=1).astype(float)
    paths[:, :, 11:19] = 0.
    paths[0] = paths[0, 0]
    x = normalize(paths)
    y = columns(x, [(2,), (1, 7)], mode) @ np.array([[1., -2.], [3., 1.]])
    # Positive variations below 1 / DBL_MAX require division for normalization.
    tiny = np.ldexp(paths, -1040)
    original, scaled = [pysiglib.sig_lasso(p, y, 3, [.0016, .0019], feature_type=mode, n_jobs=4)
                        for p in (paths, tiny)]
    assert original['converged'].all() and scaled['converged'].all()
    assert list(original['words']) == list(scaled['words'])
    np.testing.assert_allclose(scaled['signatures'], columns(x, scaled['words'], mode), atol=1e-13)
    np.testing.assert_allclose(original['coef'], scaled['coef'], atol=2e-8)
    assert scaled['stats']['full_transform_calls'] == 0


def test_restricted_gram_survives_drops_and_insertions():
    rng = np.random.default_rng(3801)
    paths = rng.normal(size=(101, 10, 9)).cumsum(1)
    y = rng.normal(size=(len(paths), 4))
    alpha = [.0007, .001, .002, .0013]
    options = dict(tol=1e-9, n_jobs=3, max_iter=3000)
    packed = pysiglib.sig_lasso(paths, y, 2, alpha, **options)
    assert packed['stats']['drops'] > 0 and len(packed['words']) > 16
    # The 0x02 flag in the default restricted-solver mask enables Gram reuse.
    assert CPSIG.sig_working_set_experimental_kernels(16, 39 & ~2) == 0
    try:
        rebuilt = pysiglib.sig_lasso(paths, y, 2, alpha, **options)
    finally:
        assert CPSIG.sig_working_set_experimental_kernels(16, 39) == 0
    assert rebuilt['converged'].all() and packed['converged'].all()
    np.testing.assert_allclose(packed['objective'], rebuilt['objective'], rtol=1e-10, atol=1e-12)
    score = packed['signatures'] @ packed['coef'] + packed['intercept']
    reference = rebuilt['signatures'] @ rebuilt['coef'] + rebuilt['intercept']
    np.testing.assert_allclose(score, reference, atol=2e-7)
    corr = np.zeros(len(alpha))
    x = normalize(paths)
    for group in batches(dictionary(9, 2, 'sig'), 16):
        values = columns(x, group, 'sig')
        corr = np.maximum(corr, np.abs(values.T @ (y - score)).max(axis=0) / len(y))
    assert np.all(corr - alpha <= 1e-9)


@pytest.mark.parametrize('mode', ['sig', 'log'])
@pytest.mark.parametrize('max_iter', [1, 3000])
def test_degree_continuation_checks_a_new_level_for_each_response(mode, max_iter):
    rng = np.random.default_rng(1593)
    paths = np.zeros((113, 8, 24))
    paths[:, :, :3] = rng.normal(size=(113, 8, 3)).cumsum(axis=1)
    x = normalize(paths[:, :, :3].copy())
    # The first response has no signal on any lower-degree column. A second
    # response can keep adding lower-degree words at the same search pass.
    covariance = np.ones((len(x), len(x)))
    for block in batches(dictionary(3, 3, mode), 7):
        design = columns(x, block, mode)
        covariance += design @ design.T
    eigenvalues, vectors = np.linalg.eigh(covariance)
    span = vectors[:, eigenvalues > eigenvalues[-1] * 1e-13]
    signal = columns(x, [(0, 0, 1, 2)], mode)[:, 0]
    high = signal - span @ (span.T @ signal)
    endpoints = columns(x, [(0,), (1,)], mode)
    y = np.column_stack([high, endpoints @ [1., 2.] + high, np.ones(len(x))])
    alpha = np.array([.15 * (high @ high) / len(x), .004, .01])
    with pytest.warns(RuntimeWarning, match='global KKT') if max_iter == 1 else nullcontext():
        model = pysiglib.sig_lasso(paths, y, 4, alpha, feature_type=mode, n_jobs=3, max_iter=max_iter, tol=1e-9)
    words = list(map(tuple, model['words']))
    assert any(len(w) == 4 and model['coef'][j, 0] != 0 for j, w in enumerate(words))
    assert model['stats']['full_transform_calls'] == 0
    residual = y - model['signatures'] @ model['coef'] - model['intercept']
    max_corr = np.zeros(3)
    # Every word involving any of the padded channels vanishes identically.
    for block in batches(dictionary(3, 4, mode), 17):
        design = columns(x, block, mode)
        max_corr = np.maximum(max_corr, abs((design - design.mean(0)).T @ residual).max(0) / len(x))
    assert np.all(max_corr - alpha <= model['kkt_violation'] + 2e-13)
    theta = (residual - residual.mean(0)) / len(x) * np.minimum(1., alpha / np.maximum(max_corr, 1e-300))
    dual = (theta * (y - .5 * len(x) * theta)).sum(0)
    np.testing.assert_allclose(model['dual_gap'], model['objective'] - dual, atol=2e-11)
    if max_iter > 1:
        assert model['converged'].all()
