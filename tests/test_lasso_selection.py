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

from importlib import import_module
from itertools import product
from math import factorial

import numpy as np
import pytest
import torch

import pysiglib as ps


def dense_features(paths, degree, log=False):
    x = np.asarray(paths, dtype=np.float64)
    x = x - x[:, :1]
    variation = np.abs(np.diff(x, axis=1)).max(axis=2).sum(axis=1)
    x /= np.where(variation, variation, 1.)[:, None, None]
    raw = ps.log_sig(x, degree, method=0) if log else ps.sig(x, degree)
    words = [w for k in range(1, degree + 1) for w in product(range(x.shape[-1]), repeat=k)]
    keep = [i for i, w in enumerate(words) if not log or all(w < w[j:] for j in range(1, len(w)))]
    return [words[i] for i in keep], raw[:, keep] * [factorial(len(words[i])) for i in keep]


def dense_lasso(design, y, penalty, intercept):
    # Independent cyclic soft-thresholding on the complete test dictionary.
    mean_x = design.mean(axis=0) if intercept else np.zeros(design.shape[1])
    mean_y = y.mean(axis=0) if intercept else np.zeros(y.shape[1])
    x = design - mean_x
    residual = y - mean_y
    beta = np.zeros((x.shape[1], y.shape[1]))
    norm = (x * x).mean(axis=0)
    for _ in range(10000):
        previous = beta.copy()
        for j in np.flatnonzero(norm):
            v = beta[j] * norm[j] + x[:, j] @ residual / len(x)
            new = np.sign(v) * np.maximum(np.abs(v) - penalty, 0) / norm[j]
            residual -= x[:, j, None] * (new - beta[j])
            beta[j] = new
        if np.max(np.abs(beta - previous)) < 1e-11:
            return beta, mean_y - mean_x @ beta
    raise AssertionError('dense reference did not converge')


def null_residual(y, loss, intercept):
    if loss == 'huber':
        location = 0.
        if intercept:
            lo, hi = np.median(y) - 1., np.median(y) + 1.
            for _ in range(60):
                location = (lo + hi) / 2
                if np.clip(y - location, -1., 1.).sum() > 0:
                    lo = location
                else:
                    hi = location
        return np.clip(y - location, -1., 1.)
    mean = y.mean() if intercept else (.5 if loss == 'logistic' else 1.)
    return y - mean


def reference_score(y, eta, loss):
    if loss == 'logistic':
        p = 1. / (1. + np.exp(-eta))
        return y - p, -np.where(y == 1, np.log(p), np.log1p(-p)).mean()
    if loss == 'poisson':
        mean = np.exp(eta)
        deviance = mean - y + y * (np.log(np.maximum(1., y)) - eta)
        return y - mean, 2. * deviance.mean()
    error = y - eta
    return np.clip(error, -1., 1.), np.where(abs(error) <= 1., .5 * error**2, abs(error) - .5).mean()


@pytest.mark.parametrize('loss,metric', [('logistic', 'log_loss'), ('poisson', 'poisson_deviance'), ('huber', 'huber')])
@pytest.mark.parametrize('fit,log', [(ps.sig_lasso, False), (ps.log_sig_lasso, True)])
@pytest.mark.parametrize('intercept', [False, True])
def test_all_loss_grid_against_full_dictionary_kkt_and_validation(loss, metric, fit, log, intercept):
    rng = np.random.default_rng(613)
    paths = rng.normal(size=(94, 7, 2)).cumsum(axis=1)
    words, design = dense_features(paths, 2, log)
    signal = 2. * design[:, 0] - design[:, 1] + design[:, words.index((0, 1))]
    if loss == 'logistic':
        y = rng.binomial(1, 1. / (1. + np.exp(-signal))).astype(float)
    elif loss == 'poisson':
        y = rng.poisson(np.exp(.5 + .45 * signal)).astype(float)
    else:
        y = signal + .1 * rng.normal(size=len(paths))
        y[::6] += 9.
    n = 64
    options = dict(loss=loss, fit_intercept=intercept, tol=1e-9)
    model = fit(paths[:n], y[:n], max_degree=2, validation_data=(paths[n:], y[n:]), max_words=3, **options)
    assert model['selection']['scoring'] == metric and model['dual_gap'] is None
    feasible = []
    for m in (1, 2):
        d = sum(len(w) <= m for w in words)
        x = design[:n, :d]
        maximum = np.max(np.abs(x.T @ null_residual(y[:n], loss, intercept))) / n
        rows = [r for r in model['selection']['candidates'] if r['degree'] == m]
        assert len(rows) == 13
        for row, alpha in zip(rows, maximum * np.geomspace(1., .001, 13)):
            assert row['alpha'] == pytest.approx(alpha, rel=2e-12)
            cold = fit(paths[:n], y[:n], m, row['alpha'], **options)
            active = [words.index(w) for w in cold['words']]
            eta = design[:, active] @ cold['coef'] + cold['intercept']
            residual, _ = reference_score(y[:n], eta[:n], loss)
            corr = x.T @ residual / n
            assert np.max(np.abs(corr)) <= alpha + 1e-8 and cold['converged']
            np.testing.assert_allclose(corr[active], alpha * np.sign(cold['coef']), atol=1e-8)
            if intercept:
                assert abs(residual.mean()) < 1e-8
            assert row['words'] == len(active)
            if len(active) > 3:
                assert row['status'] == 'over_cap'
            else:
                _, score = reference_score(y[n:], eta[n:], loss)
                assert row['status'] == 'ok' and row['score'] == pytest.approx(score, abs=3e-8)
                feasible.append(score)
    _, held_score = reference_score(y[n:], ps.sig_lasso_predict(paths[n:], model), loss)
    assert held_score == pytest.approx(min(feasible), abs=3e-8)
    assert len(model['words']) <= 3


@pytest.mark.parametrize('loss', ['logistic', 'poisson', 'huber'])
def test_all_loss_cv_calibrates_on_training_folds(loss):
    rng = np.random.default_rng(121)
    x = rng.normal(size=(48, 5, 2)).cumsum(axis=1)
    _, z = dense_features(x, 1)
    if loss == 'logistic':
        y = (z[:, 0] + .5 * rng.normal(size=len(x)) > 0).astype(float)
    elif loss == 'poisson':
        y = rng.poisson(np.exp(z[:, 0])).astype(float)
    else:
        y = 2 * z[:, 0] + rng.normal(size=len(x))
        y[::5] += 7.
    ratios = np.geomspace(1., .001, 13)
    model = ps.sig_lasso(x, y, 1, loss=loss, cv=3, tol=1e-9)
    order = np.random.default_rng(0).permutation(len(x))
    if loss != 'huber':
        order = order[np.argsort(y[order] > 0, kind='stable')]
    expected = np.zeros(3)
    steps = [0, 4, 8]
    for fold in range(3):
        valid = order[fold::3]
        train = np.setdiff1d(np.arange(len(x)), valid)
        scale = np.max(abs(z[train].T @ null_residual(y[train], loss, True))) / len(train)
        for k, j in enumerate(steps):
            cold = ps.sig_lasso(x[train], y[train], 1, scale * ratios[j], loss=loss, tol=1e-9)
            eta = z[valid][:, [w[0] for w in cold['words']]] @ cold['coef'] + cold['intercept']
            expected[k] += len(valid) * reference_score(y[valid], eta, loss)[1]
    rows = model['selection']['candidates']
    np.testing.assert_allclose([rows[j]['score'] for j in steps], expected / len(x), atol=2e-8)


@pytest.mark.parametrize('log,order,intercept', [(False, 'prefix', True), (False, 'suffix', False),
                                                  (False, 'outside', True), (True, None, True)])
def test_grid_matches_independent_dense_solutions(log, order, intercept):
    rng = np.random.default_rng(725)
    paths = rng.normal(size=(96, 7, 2)).cumsum(axis=1)
    words, design = dense_features(paths, 3, log)
    y = design[:, [words.index((0, 1)), words.index((1,))]] * [2., -1.5] + [.4, -.2]
    y += rng.normal(scale=.025, size=y.shape)
    n, cap = 64, 4
    penalties = [.2, .035, .012, .002]
    fit = ps.log_sig_lasso if log else ps.sig_lasso
    opts = {} if log else dict(tree_order=order)
    model = fit(paths[:n], y[:n], max_degree=3, alpha_grid=penalties[::-1], max_words=cap,
                validation_data=(paths[n:], y[n:]), tol=1e-10, n_jobs=2, fit_intercept=intercept, **opts)
    candidates = {(r['degree'], r['alpha']): r for r in model['selection']['candidates']}
    best = None
    for m in range(1, 4):
        d = sum(len(w) <= m for w in words)
        for alpha in penalties:
            beta, bias = dense_lasso(design[:n, :d], y[:n], alpha, intercept)
            active = np.any(beta != 0., axis=1)
            row = candidates[m, alpha]
            assert row['words'] == active.sum()
            assert row['status'] == ('ok' if active.sum() <= cap else 'over_cap')
            if active.sum() <= cap:
                predicted = design[n:, :d] @ beta + bias
                score = np.mean((predicted - y[n:])**2)
                assert row['score'] == pytest.approx(score, abs=1e-8)
                key = (score, m, -alpha)
                if best is None or key < best[0]:
                    best = key, predicted, [w for w, use in zip(words, active) if use]
    key, predicted, support = best
    assert (model['degree'], model['alpha']) == (key[1], -key[2])
    assert model['words'] == support
    np.testing.assert_allclose(ps.sig_lasso_predict(paths[n:], model), predicted, atol=1e-7)
    assert model['stats']['full_transform_calls'] == 0


@pytest.mark.parametrize('fit,log', [(ps.sig_lasso, False), (ps.log_sig_lasso, True)])
def test_default_grid_calibrates_each_fold_without_heldout_labels(fit, log):
    rng = np.random.default_rng(910)
    x = rng.normal(size=(29, 6, 2)).cumsum(axis=1)
    words, design = dense_features(x, 3, log)
    y = 3. * design[:, words.index((0, 1))] + rng.normal(scale=.03, size=len(x))
    model = fit(x, y, 3, alpha=None, cv=3, scoring='mse', tol=1e-10)
    ratios = np.geomspace(1., .001, 13)
    maximum = np.max(np.abs(design.T @ (y - y.mean()))) / len(y)
    scores = np.zeros(len(ratios))
    order = np.random.default_rng(0).permutation(len(x))
    for fold in range(3):
        held = order[fold::3]
        train = np.ones(len(x), dtype=bool)
        train[held] = False
        ymax = np.max(np.abs(design[train].T @ (y[train] - y[train].mean()))) / train.sum()
        for i, ratio in enumerate(ratios):
            beta, bias = dense_lasso(design[train], y[train, None], ymax * ratio, True)
            scores[i] += np.sum((design[held] @ beta[:, 0] + bias - y[held])**2)
    rows = model['selection']['candidates']
    np.testing.assert_allclose([r['alpha'] for r in rows], ratios * maximum, rtol=1e-11)
    np.testing.assert_allclose([r['score'] for r in rows], scores / len(x), atol=1e-8)
    assert model['alpha'] == pytest.approx(ratios[np.argmin(scores)] * maximum)
    assert model['selection']['folds'] == 3
    assert model['signatures'].shape[0] == len(x)
    joint = fit(x, y, max_degree=3, cv=3, scoring='mse', tol=1e-10)
    assert joint['selection']['native_fits'] < 3 * 4 * len(ratios)
    for degree in range(1, 4):
        separate = model if degree == 3 else fit(x, y, degree, cv=3, scoring='mse', tol=1e-10)
        expected = separate['selection']['candidates']
        actual = [r for r in joint['selection']['candidates'] if r['degree'] == degree]
        for field in ('score', 'alpha', 'words'):
            np.testing.assert_allclose([r[field] for r in actual], [r[field] for r in expected], atol=1e-8)


@pytest.mark.parametrize('fit', [ps.sig_lasso, ps.log_sig_lasso])
def test_default_grid_uses_higher_degrees_and_all_targets(fit):
    square = [[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]]
    orientation = np.tile([-1., 1.], 4)
    paths = np.tile(square, (len(orientation), 1, 1))
    paths[:, :, 1] *= orientation[:, None]
    y = orientation[:, None] * [.5, 2.]
    model = fit(paths, y, max_degree=2, fit_intercept=False, validation_data=(paths, y), scoring='mse')
    # Endpoints vanish. The area column is orientation / 8 after normalization.
    rows = model['selection']['candidates']
    assert rows[0]['alpha'] == 1.
    assert rows[13]['alpha'] == pytest.approx(2. / 8)
    assert model['degree'] == 2 and model['alpha'] == pytest.approx(.001 * 2. / 8)


def test_degree_only_vector_penalty_and_certificate_reuse():
    rng = np.random.default_rng(194)
    x = rng.normal(size=(53, 6, 3)).cumsum(axis=1)
    _, design = dense_features(x, 2)
    y = design[:, :2] * [1.3, -.9]
    penalties = np.array([.03, .012])
    model = ps.sig_lasso(x[:40], y[:40], degree=None, alpha=penalties, max_degree=4,
                         validation_data=(x[40:], y[40:]), tol=1e-10)
    for row in model['selection']['candidates']:
        reference = ps.sig_lasso(x[:40], y[:40], row['degree'], penalties, tol=1e-10)
        if row['degree'] == model['degree']:
            assert model['words'] == reference['words']
        score = np.mean((ps.sig_lasso_predict(x[40:], reference) - y[40:])**2)
        assert row['score'] == pytest.approx(score, abs=1e-9)
    trivial = ps.sig_lasso(x, y, alpha=100., max_degree=4, max_words=0, validation_data=(x, y))
    assert trivial['words'] == [] and trivial['degree'] == 1
    assert trivial['selection']['native_fits'] == 1
    assert len(trivial['selection']['candidates']) == 4
    np.testing.assert_allclose(ps.sig_lasso_predict(x, trivial), np.tile(y.mean(axis=0), (len(x), 1)))


def test_cv_enforces_union_budget_after_full_training_refit():
    # Each feature occurs in just two samples. At the absolute penalty some
    # fold fits use three words, whereas the full training fit uses four.
    endpoints = np.vstack([np.eye(4), -np.eye(4)])
    x = np.stack((np.zeros_like(endpoints), endpoints), axis=1)
    y = endpoints.copy()
    model = ps.sig_lasso(x, y, 1, alpha_grid=[.1, 1.], max_words=3, cv=4, scoring='mse')
    assert model['alpha'] == 1. and not model['words']
    rows = model['selection']['candidates']
    assert rows[1]['status'] == 'over_cap' and rows[1]['words'] == 4
    assert ps.sig_lasso(x, y, 1, .1, max_words=None)['words'] == [(i,) for i in range(4)]
    with pytest.raises(ValueError, match='fixed degree and alpha select 4'):
        ps.sig_lasso(x, y, 1, .1, max_words=3)
    with pytest.raises(ValueError, match='No fitted model'):
        ps.sig_lasso(x, y, 1, alpha_grid=[.1], max_words=3, cv=3)


def test_weaker_penalty_can_become_feasible_after_a_word_drops():
    rng = np.random.default_rng(0)
    endpoints = rng.normal(size=(24, 3)) @ rng.normal(size=(3, 4)) + .12 * rng.normal(size=(24, 4))
    paths = np.stack((np.zeros_like(endpoints), endpoints), axis=1)
    design = endpoints / np.abs(endpoints).max(axis=1, keepdims=True)
    y = design @ rng.normal(size=4) + .2 * rng.normal(size=24)
    grid = [.24, .0064, .0033]
    # Identity validation endpoints expose each coefficient.
    vx = np.stack((np.zeros((4, 4)), np.eye(4)), axis=1)
    beta, intercept = dense_lasso(design, y[:, None], grid[-1], True)
    vy = (beta + intercept).ravel()
    fit = ps.sig_lasso(paths, y, 1, alpha_grid=grid, validation_data=(vx, vy), max_words=3, tol=1e-10)
    rows = fit['selection']['candidates']
    assert [r['words'] for r in rows] == [2, 4, 3]
    assert [r['status'] for r in rows] == ['ok', 'over_cap', 'ok']
    assert fit['alpha'] == grid[-1] and len(fit['words']) == 3
    np.testing.assert_allclose(ps.sig_lasso_predict(vx, fit), vy, atol=1e-7)


@pytest.mark.parametrize('fit', [ps.sig_lasso, ps.log_sig_lasso])
def test_zero_features_and_zero_budget_use_default_degree_ceiling(fit):
    x = np.zeros((6, 3, 1))
    y = np.arange(len(x), dtype=np.float64)
    model = fit(x, y, max_words=0, validation_data=(x, y))
    assert model['degree'] == 1 and model['alpha'] == 1. and model['words'] == []
    assert model['selection']['score'] == np.var(y)
    rows = model['selection']['candidates']
    assert len(rows) == 10 * 13 and rows[-1]['degree'] == 10
    assert model['selection']['native_fits'] == 13
    assert all(row['words'] == 0 and row['status'] == 'ok' for row in rows)


@pytest.mark.parametrize('fit', [ps.sig_lasso, ps.log_sig_lasso])
@pytest.mark.parametrize('loss,scoring', [('squared', 'auto'), ('logistic', 'accuracy'), ('logistic', 'auto')])
def test_torch_transforms_batch_axes_and_multiclass_accuracy(fit, loss, scoring):
    rng = np.random.default_rng(472)
    raw = torch.tensor(rng.normal(size=(4, 12, 5, 4)).cumsum(axis=2), dtype=torch.float32)
    x = raw[..., ::2]
    classes = torch.argmax(torch.stack((x[:, :, -1, 0], -x[:, :, -1, 0], x[:, :, -1, 1]), dim=-1), dim=-1)
    y = torch.nn.functional.one_hot(classes, 3).float()
    model = fit(x[:3], y[:3], max_degree=2, alpha_grid=[.08, .02], validation_data=(x[3:], classes[3:]),
                time_aug=True, lead_lag=True, initial_words=[(0, 1)], n_jobs=2, tol=1e-6, loss=loss, scoring=scoring)
    assert model['signatures'].shape == (3, 12, len(model['words']))
    prediction = ps.sig_lasso_predict(x[3:], model)
    assert tuple(prediction.shape) == (1, 12, 3)
    if model['selection']['scoring'] == 'accuracy':
        assert float((prediction.argmax(dim=-1) == classes[3:]).sum()) / 12 == model['selection']['score']
    else:
        assert model['selection']['scoring'] == 'log_loss'
        expected = torch.nn.functional.binary_cross_entropy_with_logits(prediction.double(), y[3:].double())
        assert model['selection']['score'] == pytest.approx(expected.item())
    assert model['time_aug'] and model['lead_lag'] and model['dimension'] == 2


def test_grid_failure_handling_and_invalid_inputs(monkeypatch):
    endpoints = np.vstack([np.eye(20), -np.eye(20)])
    x = np.stack((np.zeros_like(endpoints), endpoints), axis=1)
    y = endpoints.sum(axis=1)
    with pytest.warns(RuntimeWarning, match='failed certification'):
        result = ps.sig_lasso(x, y, 1, alpha_grid=[1., .025], validation_data=(x, y), max_iter=1)
    assert result['alpha'] == 1.
    assert result['selection']['candidates'][1]['status'] == 'unconverged'
    with pytest.raises(RuntimeError, match='did not converge'):
        ps.sig_lasso(x, y, 1, alpha_grid=[.025], validation_data=(x, y), max_iter=1)
    for kwargs, match in [
        (dict(alpha_grid=[]), 'alpha_grid'), (dict(alpha_grid=[.1, 0]), 'positive'),
        (dict(max_degree=0), 'max_degree'), (dict(max_words=-1), 'max_words'),
        (dict(max_words=True), 'max_words'), (dict(scoring='rmse'), 'scoring'),
        (dict(cv=1), 'cv'), (dict(loss='unknown'), 'loss'), (dict(validation_data=(x,)), 'pair'),
        (dict(degree=1, max_degree=2), 'max_degree'), (dict(alpha=.2, alpha_grid=[.1]), 'alpha_grid'),
        (dict(validation_data=(x, y[:-1])), 'responses'), (dict(initial_words=[(20,)]), 'initial_words'),
    ]:
        with pytest.raises(ValueError, match=match):
            ps.sig_lasso(x, y, **kwargs)
    with pytest.raises(ValueError, match='one sample'):
        ps.sig_lasso(x[0], np.asarray(y[0]))
    with pytest.raises(MemoryError):
        ps.sig_lasso(x, y, 1, alpha_grid=[.025], memory_limit=32, validation_data=(x, y))
    module = import_module('pysiglib.sig_lasso')
    monkeypatch.setattr(module, 'CPSIG_WORKING_SET_GRID', None)
    with pytest.raises(RuntimeError, match='rebuild'):
        ps.sig_lasso(x, y, 1, validation_data=(x, y))
    assert ps.sig_lasso(x, y, 1, 1.)['converged']


def test_grid_shares_the_native_memory_limit_with_validation(monkeypatch):
    rng = np.random.default_rng(734)
    paths = rng.normal(size=(32, 5, 3))
    y = (paths[:, -1, 0] - paths[:, 0, 0]).copy()
    selection = import_module('pysiglib._lasso_selection')
    budgets = []

    def predict(x, model, *, n_jobs, memory_limit):
        budgets.append(memory_limit)
        return ps.sig_lasso_predict(x, model, n_jobs=n_jobs, memory_limit=memory_limit)

    monkeypatch.setattr(selection, 'sig_lasso_predict', predict)
    limit = 4 * 1024**2
    result = ps.sig_lasso(paths, y, 2, alpha_grid=[.5, .02], memory_limit=limit, validation_data=(paths, y))
    assert result['words'] and len(budgets) == 2
    assert all(0 < available < limit for available in budgets)


def test_link_scale_scores_and_validation_with_a_missing_class():
    paths = np.zeros((10, 2, 2))
    y = np.arange(10) < 6
    held = np.ones(10)
    for loss in ('logistic', 'poisson'):
        options = dict(loss=loss, degree=1, alpha_grid=[1.], validation_data=(paths, held))
        for metric, expected in [('accuracy', 1.), ('mse', .4**2)]:
            model = ps.sig_lasso(paths, y.astype(float), scoring=metric, **options)
            assert model['selection']['score'] == pytest.approx(expected)
    logistic = ps.sig_lasso(paths, y.astype(float), 1, loss='logistic', validation_data=(paths, held))
    assert logistic['selection']['score'] == pytest.approx(-np.log(.6))


def test_logistic_certificate_checks_raw_correlations_when_the_centred_bound_is_loose():
    rng = np.random.default_rng(942)
    x = rng.normal(size=(36, 5, 4)).cumsum(axis=1)
    y = (rng.random(36) < .3).astype(float)
    alpha, tol = .001, 1e-5
    model = ps.sig_lasso(x, y, 3, alpha, loss='logistic', tol=tol)
    words, design = dense_features(x, 3)
    active = [words.index(w) for w in model['words']]
    eta = design[:, active] @ model['coef'] + model['intercept']
    residual = y - 1. / (1. + np.exp(-eta))
    corr = design.T @ residual / len(y)
    assert model['converged'] and abs(residual.mean()) <= tol
    assert abs(corr).max() <= alpha + 1e-4 * alpha
    np.testing.assert_allclose(corr[active], alpha * np.sign(model['coef']), rtol=0., atol=1e-4 * alpha)


@pytest.mark.parametrize('loss', ['logistic', 'poisson', 'huber'])
def test_multiple_targets_fixed_penalties_and_degree_selection(loss):
    rng = np.random.default_rng(880)
    x = rng.normal(size=(53, 5, 2))
    y = rng.normal(size=(53, 2))
    if loss == 'logistic':
        y = (y > [-.2, .3]).astype(float)
    elif loss == 'poisson':
        y = rng.poisson([1.4, .8], size=y.shape).astype(float)
    alpha = np.array([.01, .03])
    model = ps.log_sig_lasso(x, y, alpha=alpha, max_degree=3, loss=loss, cv=3)
    cold = ps.log_sig_lasso(x, y, model['degree'], alpha, loss=loss)
    assert np.all(model['converged']) and model['dual_gap'] is None
    np.testing.assert_array_equal(model['alpha'], alpha)
    np.testing.assert_allclose(model['objective'], cold['objective'], atol=1e-9)
    np.testing.assert_allclose(ps.sig_lasso_predict(x, model), ps.sig_lasso_predict(x, cold), atol=1e-6)


@pytest.mark.parametrize('loss', ['logistic', 'poisson'])
def test_grid_rejects_invalid_training_folds_and_targets(loss):
    x = np.zeros((8, 3, 1))
    y = np.zeros(8)
    y[0] = 1.
    with pytest.raises(ValueError, match='CV training fold'):
        ps.sig_lasso(x, y, 1, loss=loss, cv=2)
    with pytest.raises(ValueError, match='Validation'):
        ps.sig_lasso(x, y, 1, loss=loss, validation_data=(x, np.full(8, .5)))
    with pytest.raises(ValueError, match='Training'):
        ps.sig_lasso(x, np.full(8, .5), 1, loss=loss, cv=2)
    model = ps.sig_lasso(x, np.zeros(8), 1, loss=loss, cv=2, fit_intercept=False)
    assert model['words'] == [] and model['intercept'] == 0.


def test_old_native_grid_still_supports_squared_loss(monkeypatch):
    module = import_module('pysiglib.sig_lasso')
    monkeypatch.setattr(module, 'CPSIG_WORKING_SET_GRID_LOSS_MASK', 1)
    x, y = np.zeros((8, 3, 1)), np.arange(8.)
    with pytest.raises(RuntimeError, match='rebuild'):
        ps.sig_lasso(x, y, 1, loss='huber', cv=2)
    assert ps.sig_lasso(x, y, 1, cv=2)['converged']
    assert ps.sig_lasso(x, y, 1, .1, loss='huber')['converged']


def test_huber_null_fit_with_a_large_outlier():
    x = np.zeros((5, 3, 1))
    y = np.array([0., 1., 2., 3., 1e60])
    model = ps.sig_lasso(x, y, 1, loss='huber', max_words=0, validation_data=(x, np.full(5, 2.)))
    assert model['intercept'] == pytest.approx(2.) and not model['words']
    assert model['selection']['score'] < 1e-25


def test_extreme_poisson_validation_scores(monkeypatch):
    selection = import_module('pysiglib._lasso_selection')
    x = np.stack((np.zeros((10, 1)), np.tile([[-1.], [1.]], (5, 1))), axis=1)
    y = np.tile([1., 3.], 5)

    def predict(x, model, **kwargs):
        return np.full(len(x), 1000. if model['words'] else -1000.)

    monkeypatch.setattr(selection, 'sig_lasso_predict', predict)
    with pytest.warns(RuntimeWarning, match='nonfinite validation scores'):
        model = ps.sig_lasso(x, y, 1, loss='poisson', alpha_grid=[2., .1], validation_data=(x, y))
    assert model['alpha'] == 2. and not model['words']
    assert model['selection']['score'] == pytest.approx(2. * np.mean(y * (np.log(y) + 1000.) - y))
    assert model['selection']['candidates'][1]['status'] == 'nonfinite_score'
    with pytest.raises(FloatingPointError, match='finite validation score'):
        ps.sig_lasso(x, y, 1, loss='poisson', alpha_grid=[.1], validation_data=(x, y))
