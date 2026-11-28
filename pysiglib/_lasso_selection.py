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

import warnings

import numpy as np
import torch

from .data_handlers import PathInputHandler
from .param_checks import check_type
from .sig_lasso import _dictionary_size, _fit_lasso, _response_shape, _valid_word, sig_lasso_predict
from .transform_path import transform_path


def _numpy(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('Lasso selection requires CPU inputs')
        return value.detach().numpy()
    return np.asarray(value)


def _class_labels(values, targets):
    if values.shape[1] == 1:
        labels = values[:, 0]
        if np.all((labels >= 0) & (labels < max(2, targets)) & (labels == np.floor(labels))):
            return labels.astype(np.intp)
    elif values.shape[1] == targets and np.all((values == 0) | (values == 1)) and np.all(values.sum(axis=1) == 1):
        return values.argmax(axis=1)
    return None


def _check_targets(values, loss, name, intercept=False):
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} responses must be finite')
    if loss == 'logistic':
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f'{name} logistic responses must be binary')
        if intercept and np.any(values.min(axis=0) == values.max(axis=0)):
            raise ValueError(f'{name} needs both binary classes in each target for a finite logistic intercept')
    elif loss == 'poisson':
        if np.any(values < 0) or not np.all(values == np.floor(values)):
            raise ValueError(f'{name} Poisson responses must be nonnegative counts')
        if intercept and np.any(values.max(axis=0) == 0):
            raise ValueError(f'{name} needs a positive count in each target for a finite Poisson intercept')


def _penalty_grid(alpha, alpha_grid, targets):
    relative = alpha is None and alpha_grid is None
    if alpha is None:
        values = np.geomspace(1., .001, 13) if relative else _numpy(alpha_grid)
        if values.ndim != 1 or not values.size or values.dtype.kind not in 'fiu':
            raise ValueError('alpha_grid must be a nonempty sequence of positive finite scalar penalties')
        values = np.unique(values.astype(np.float64))[::-1]
        grid = np.repeat(values[:, None], targets, axis=1)
    else:
        values = _numpy(alpha)
        if values.ndim == 0:
            values = np.full(targets, values.item())
        if values.shape != (targets,):
            raise ValueError('alpha must be scalar or have one entry per target')
        grid = np.asarray(values, dtype=np.float64).reshape(1, targets)
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError('Lasso penalties must be positive and finite')
    return grid, relative


def _walk_grid(path, y, feature_type, degrees, grid, relative, options, record):
    # A smaller dictionary must retain the support and, for a relative grid,
    # a word attaining the maximum used to scale the penalties.
    covered_to = np.full(len(grid), degrees[-1] + 1)
    for degree in reversed(degrees):
        pending = np.flatnonzero(covered_to > degree)
        if not len(pending):
            continue
        seeds = [w for w in options['initial_words'] if len(w) <= degree]

        def visit(index, model, scale_degree, available):
            j = int(pending[index])
            passed = bool(_numpy(model['converged']).all()) and model['stats']['numerical_failures'] == 0
            lowest = degree
            if passed:
                lowest = max(degrees[0], scale_degree, max(map(len, model['words']), default=1))
            covered_to[j] = lowest
            record(j, model, range(lowest, degree + 1), passed, available)

        _fit_lasso(path, y, degree, grid[pending], feature_type, **dict(options, initial_words=seeds),
                   visit=visit, relative=relative)


def _score_sum(path, truth, labels, model, n_jobs, available, metric):
    if available == 0 and model['words']:
        raise MemoryError('The native memory budget is exhausted before validation prediction')
    prediction = sig_lasso_predict(path, model, n_jobs=n_jobs, memory_limit=available or None)
    prediction = _numpy(prediction).reshape(len(truth), -1)
    loss = model['loss']
    if metric == 'accuracy':
        threshold = {'logistic': 0., 'poisson': np.log(.5)}.get(loss, .5)
        predicted = prediction.argmax(axis=1) if prediction.shape[1] > 1 else prediction[:, 0] >= threshold
        return int(np.count_nonzero(predicted == labels))
    eta = prediction.astype(np.float64, copy=False)
    truth = truth.astype(np.float64, copy=False)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        if metric == 'log_loss':
            values = np.logaddexp(0., np.where(truth == 1., -eta, eta))
        elif metric == 'poisson_deviance':
            positive = truth > 0
            delta = eta[positive] - np.log(truth[positive])
            values = np.empty_like(eta)
            values[positive] = 2. * (truth[positive] * (np.expm1(delta) - delta))
            values[~positive] = 2. * np.exp(eta[~positive])
        elif metric == 'huber':
            error = np.abs(eta - truth)
            clipped = np.minimum(error, 1.)
            values = clipped * (error - .5 * clipped)
        else:
            if loss == 'logistic':
                eta = np.exp(-np.logaddexp(0., -eta))
            elif loss == 'poisson':
                eta = np.exp(eta)
            values = np.square(eta - truth)
        return float(np.sum(values) / eta.shape[1])


def select_lasso(path, y, degree, alpha, feature_type, max_words, max_degree, alpha_grid,
                 validation_data, cv, scoring, options):
    limit = degree if degree is not None else (10 if max_degree is None else max_degree)
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or not 1 <= limit <= 170:
        raise ValueError('degree and max_degree must be integers between 1 and 170')
    degrees = range(1 if degree is None else int(limit), int(limit) + 1)
    loss = options['loss']
    natural = {'squared': 'mse', 'logistic': 'log_loss', 'poisson': 'poisson_deviance', 'huber': 'huber'}[loss]
    metrics = ('auto', 'mse', 'accuracy') if natural == 'mse' else ('auto', 'mse', 'accuracy', natural)
    if not isinstance(scoring, str) or scoring not in metrics:
        raise ValueError(f"Invalid scoring for {loss} loss. Choose from " + ', '.join(metrics))
    if isinstance(cv, bool) or not isinstance(cv, (int, np.integer)) or cv < 2:
        raise ValueError('cv must be an integer of at least two')
    if isinstance(path, torch.Tensor) and path.device.type != 'cpu':
        raise ValueError('Lasso selection requires CPU paths')
    inputs = PathInputHandler(path, False, False, 1., 'path')
    if not inputs.batch_size or not inputs.length:
        raise ValueError('Lasso selection requires nonempty training data')
    single, targets = _response_shape(path, y, inputs)
    n = inputs.batch_size
    truth = _numpy(y).reshape(n, targets)
    check_type(options['fit_intercept'], 'fit_intercept', bool)
    _check_targets(truth, loss, 'Training', options['fit_intercept'])
    use_classes = scoring == 'accuracy' or loss == 'logistic' or (scoring == 'auto' and loss == 'squared')
    labels = _class_labels(truth, targets) if use_classes else None
    if scoring == 'accuracy' and labels is None:
        raise ValueError('Accuracy requires binary or one-hot training responses')
    metric = scoring
    if scoring == 'auto':
        metric = 'accuracy' if loss == 'squared' and labels is not None else natural
    grid, relative = _penalty_grid(alpha, alpha_grid, targets)

    # Transforms are pathwise, so the same transformed paths serve every fold and degree.
    transform = {name: options[name] for name in ('time_aug', 'lead_lag', 'end_time', 'n_jobs')}
    for name in ('time_aug', 'lead_lag'):
        check_type(transform[name], name, bool)
    augment = transform['time_aug'] or transform['lead_lag']
    x = transform_path(inputs.path, **transform) if augment else inputs.path
    options = dict(options, time_aug=False, lead_lag=False, initial_words=tuple(tuple(w) for w in options['initial_words']))
    if any(not _valid_word(word, limit, x.shape[-1]) for word in options['initial_words']):
        raise ValueError('initial_words contains an invalid transformed-channel word')

    if validation_data is not None:
        if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
            raise ValueError('validation_data must be a (path, y) pair')
        vx, vy = validation_data
        if type(vx) is not type(path) or vx.dtype != path.dtype or (isinstance(vx, torch.Tensor) and vx.device != path.device):
            raise ValueError('Validation paths must match the training array type, dtype and device')
        val = PathInputHandler(vx, False, False, 1., 'validation path')
        if not val.length or not val.batch_size or val.dimension != inputs.dimension:
            raise ValueError('Validation paths must be nonempty and match the training channel dimension')
        vx = transform_path(val.path, **transform) if augment else val.path
        vy = _numpy(vy)
        widths = (1, targets) if labels is not None else (targets,)
        valid_shapes = {(*batch, k) for batch in (val.batch_shape, (val.batch_size,)) for k in widths}
        if 1 in widths:
            valid_shapes.update((val.batch_shape, (val.batch_size,)))
        if vy.shape not in valid_shapes or vy.dtype.kind not in 'bfiu':
            raise ValueError('Validation responses must be finite and match the sample and target axes')
        vy = vy.reshape(val.batch_size, -1)
        if vy.shape[1] == 1 and targets > 1:
            indices = _class_labels(vy, targets)
            if indices is None:
                raise ValueError('Validation class labels must be valid class indices')
            vy = indices[:, None] == np.arange(targets)
        _check_targets(vy, loss, 'Validation')
        vlabels = _class_labels(vy, targets) if metric == 'accuracy' else None
        if metric == 'accuracy' and vlabels is None:
            raise ValueError('Validation class labels must be valid class indices or one-hot responses')
        validation_samples = val.batch_size
        folds = 0
    else:
        if n < 2:
            raise ValueError('Automatic selection with one sample needs explicit validation_data')
        folds = min(int(cv), n)
        validation_samples = n

    shape = (len(degrees), len(grid))
    scores = np.zeros(shape)
    certified = np.ones(shape, dtype=bool)
    fits = 0
    if folds:
        xflat = x.reshape(n, *x.shape[-2:])
        yflat = y.reshape(n) if single else y.reshape(n, targets)
        order = np.random.default_rng(0).permutation(n)
        strata = labels
        if strata is None and loss in ('logistic', 'poisson') and options['fit_intercept']:
            _, strata = np.unique(truth > 0, axis=0, return_inverse=True)
            strata = strata.reshape(-1)
        if strata is not None:
            order = order[np.argsort(strata[order], kind='stable')]
        for fold in range(folds):
            held = order[fold::folds]
            train = np.ones(n, dtype=bool)
            train[held] = False
            vx, vy = xflat[held], truth[held]
            vlabels = labels[held] if metric == 'accuracy' else None
            ytrain = yflat[train]
            if loss in ('logistic', 'poisson') and options['fit_intercept']:
                _check_targets(_numpy(ytrain).reshape(-1, targets), loss, f'CV training fold {fold + 1}', True)

            def record_fold(j, model, covered, passed, available):
                nonlocal fits
                fits += 1
                value = _score_sum(vx, vy, vlabels, model, options['n_jobs'], available, metric) if passed else 0.
                rows = slice(covered.start - degrees[0], covered.stop - degrees[0])
                scores[rows, j] += value
                certified[rows, j] &= passed

            _walk_grid(xflat[train], ytrain, feature_type, degrees, grid, relative, options, record_fold)

    best = None
    best_key = None
    records = [[None] * len(grid) for _ in degrees]

    def record_full(j, model, covered, passed, available):
        nonlocal best, best_key, fits
        fits += 1
        words = len(model['words'])
        within_cap = max_words is None or words <= max_words
        if not folds and passed and within_cap:
            value = _score_sum(vx, vy, vlabels, model, options['n_jobs'], available, metric) / validation_samples
        else:
            value = None
        for m in covered:
            row = m - degrees[0]
            valid = passed and bool(certified[row, j])
            score = float(scores[row, j] / validation_samples) if folds and valid else value
            status = 'unconverged' if not valid else ('ok' if within_cap else 'over_cap')
            if score is not None and not np.isfinite(score):
                score = None
                if status == 'ok':
                    status = 'nonfinite_score'
            records[row][j] = dict(degree=m, alpha=model['alpha'], score=score, words=words, status=status)
            if status != 'ok':
                continue
            key = (-score if metric == 'accuracy' else score, m, j)
            if best_key is None or key < best_key:
                best_key = key
                best = dict(model, degree=m, dimension=inputs.dimension, time_aug=transform['time_aug'],
                            lead_lag=transform['lead_lag'], end_time=transform['end_time'])
                if m != model['degree']:
                    best['stats'] = dict(model['stats'], certificate_degree=model['degree'],
                                         dictionary_size=_dictionary_size(feature_type, x.shape[-1], m))

    _walk_grid(x, y, feature_type, degrees, grid, relative, options, record_full)
    records = [candidate for row in records for candidate in row]
    failures = sum(r['status'] == 'unconverged' for r in records)
    nonfinite = sum(r['status'] == 'nonfinite_score' for r in records)
    if best is None:
        if failures:
            raise RuntimeError('No certified candidate satisfies the word budget; some grid fits did not converge')
        if nonfinite:
            raise FloatingPointError('No feasible grid candidate has a finite validation score')
        raise ValueError(f'No fitted model on the requested grid satisfies max_words={max_words}')
    if failures:
        warnings.warn(f'{failures} degree/penalty candidates failed certification and were excluded', RuntimeWarning, stacklevel=3)
    if nonfinite:
        warnings.warn(f'{nonfinite} degree/penalty candidates had nonfinite validation scores and were excluded', RuntimeWarning, stacklevel=3)
    score = -best_key[0] if metric == 'accuracy' else best_key[0]
    best['selection'] = dict(scoring=metric, score=score, folds=folds, candidates=records,
                             native_fits=fits, relative_grid=relative)
    return best
