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

"""Time joint degree/penalty selection against a cold fit at its chosen pair."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

import pysiglib as ps


def synthetic():
    rng = np.random.default_rng(7201)
    increments = rng.normal(size=(512, 12, 4)) / np.sqrt(12)
    paths = np.zeros((512, 13, 4))
    paths[:, 1:] = np.cumsum(increments, axis=1)
    variation = np.abs(increments).max(axis=2).sum(axis=1)
    endpoint = paths[:, -1] / variation[:, None]
    area = (paths[:, :-1, 0] * increments[:, :, 1] - paths[:, :-1, 1] * increments[:, :, 0]).sum(axis=1) / variation**2
    signal = 4. * endpoint[:, 0] - 2.5 * endpoint[:, 1] + 12. * area
    y = signal + .35 * rng.normal(size=len(paths))
    robust = y.copy()
    robust[::10] += 6. * rng.normal(size=len(robust[::10]))
    targets = dict(squared=y, logistic=rng.binomial(1, 1. / (1. + np.exp(.2 - 1.5 * signal))).astype(float),
                   poisson=rng.poisson(np.exp(.3 + .65 * signal)).astype(float), huber=robust)
    return paths[:256].copy(), paths[256:384].copy(), paths[384:].copy(), {
        loss: (values[:256].copy(), values[256:384].copy()) for loss, values in targets.items()}


def training_objective(model, y):
    eta = model['signatures'] @ model['coef'] + model['intercept']
    if model['loss'] == 'logistic':
        values = np.logaddexp(0., (1. - 2. * y) * eta)
    elif model['loss'] == 'poisson':
        values = np.exp(eta) - y * eta
    elif model['loss'] == 'huber':
        err = abs(y - eta)
        values = np.where(err <= 1., .5 * err**2, err - .5)
    else:
        values = .5 * (y - eta)**2
    return values.mean(axis=0) + model['alpha'] * abs(model['coef']).sum(axis=0)


def compare(fit, train, y, val, vy, test, options, selection, repeats):
    validation = None if 'cv' in selection else (val, vy)

    def automatic():
        return fit(train, y, validation_data=validation, **options, **selection)

    warm = automatic()
    degree, alpha = warm['degree'], warm['alpha']

    def fixed():
        return fit(train, y, degree, alpha, **options)

    fixed_words = len(fixed()['words'])
    times = {'automatic': [], 'fixed': []}
    max_error = {'objective': 0., 'test_predictor': 0., 'support_difference': 0}
    for i in range(repeats):
        calls = [('automatic', automatic), ('fixed', fixed)]
        if i % 2:
            calls.reverse()
        models = {}
        for key, call in calls:
            start = perf_counter()
            models[key] = call()
            times[key].append(perf_counter() - start)
        a, b = models['automatic'], models['fixed']
        assert a['degree'] == degree and a['alpha'] == alpha
        for model in (a, b):
            assert np.all(model['converged']) and np.max(model['kkt_violation']) <= options['tol']
            assert model['stats']['full_transform_calls'] == 0 and len(model['words']) <= options['max_words']
            np.testing.assert_allclose(model['objective'], training_objective(model, y), rtol=1e-9, atol=1e-12)
        za, zb = [ps.sig_lasso_predict(test, model, n_jobs=options['n_jobs']) for model in (a, b)]
        max_error['objective'] = max(max_error['objective'], float(np.max(abs(a['objective'] - b['objective']))))
        max_error['test_predictor'] = max(max_error['test_predictor'], float(np.max(abs(za - zb))))
        max_error['support_difference'] = max(max_error['support_difference'], len(set(a['words']) ^ set(b['words'])))
        if options['loss'] == 'logistic':
            pa, pb = (np.exp(-np.logaddexp(0., -z)) for z in (za, zb))
            max_error['probability'] = max(max_error.get('probability', 0.), float(np.max(abs(pa - pb))))
        if options['loss'] == 'logistic' or selection['scoring'] == 'accuracy':
            threshold = 0. if options['loss'] == 'logistic' else .5
            ca, cb = (z.argmax(axis=1) if z.ndim == 2 else z >= threshold for z in (za, zb))
            max_error['test_classes'] = max(max_error.get('test_classes', 0), int(np.sum(ca != cb)))
    scores = warm['selection']
    medians = {key: float(np.median(seconds)) for key, seconds in times.items()}
    return dict(degree=degree, alpha=alpha, words=len(warm['words']), fixed_words=fixed_words,
                scoring=scores['scoring'], score=scores['score'],
                candidates=len(scores['candidates']), native_fits=scores['native_fits'],
                unconverged=sum(row['status'] == 'unconverged' for row in scores['candidates']), seconds=times,
                median_seconds=medians, ratio=medians['automatic'] / medians['fixed'], max_difference=max_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arabic', type=Path, help='Use the saved Spoken Arabic Digits prepared.npz')
    parser.add_argument('--loss', nargs='+', choices=['squared', 'logistic', 'poisson', 'huber'])
    parser.add_argument('--features', nargs='+', choices=['sig', 'log'], default=['sig', 'log'])
    parser.add_argument('--max-degree', type=int, default=4)
    parser.add_argument('--alpha-grid', type=float, nargs='+')
    parser.add_argument('--max-words', type=int)
    parser.add_argument('--cv', type=int, help='Use training-only cross-validation in place of the validation split')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.arabic:
        with np.load(args.arabic) as data:
            train, val, test = (data[k].copy(order='C') for k in ('train', 'validation', 'test'))
            labels = (data['response'].copy(order='C'), data['labels'].copy())
        targets = dict(squared=labels, logistic=labels)
    else:
        train, val, test, targets = synthetic()
    with threadpool_limits(limits=1):
        for loss in args.loss or targets:
            if loss not in targets:
                parser.error(f'No {loss} targets for this dataset')
            for mode in args.features:
                fit = ps.sig_lasso if mode == 'sig' else ps.log_sig_lasso
                options = dict(loss=loss, n_jobs=args.jobs, tol=args.tol, max_iter=3000,
                               max_words=args.max_words if args.max_words is not None else (500 if args.arabic else 100))
                if mode == 'sig' and args.arabic:
                    options['tree_order'] = 'outside'
                selection = dict(max_degree=args.max_degree, alpha_grid=args.alpha_grid,
                                 scoring='accuracy' if args.arabic else 'auto')
                if args.cv is not None:
                    selection['cv'] = args.cv
                y, vy = targets[loss]
                result = compare(fit, train, y, val, vy, test, options, selection, args.repeats)
                print(json.dumps(dict(dataset='arabic' if args.arabic else 'synthetic', features=mode, **options,
                                      selection_options=selection, **result)), flush=True)


if __name__ == '__main__':
    main()
