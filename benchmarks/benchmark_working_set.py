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

"""Time a single fit configuration in a fresh process. Run as a benchmarks module."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("legacy", "native", "prototype"))
    parser.add_argument("case", choices=("gaussian", "anisotropic", "arabic", "commutator", "hall6"))
    parser.add_argument("mode", choices=("sig", "log", "hall"))
    parser.add_argument("loss", choices=("squared", "logistic", "poisson", "huber"))
    parser.add_argument("jobs", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    using_baseline = args.variant in ("legacy", "prototype") and "PYSIGLIB_BASELINE" in os.environ
    if using_baseline:
        sys.path.insert(0, os.environ["PYSIGLIB_BASELINE"])
    import pysiglib

    rng = np.random.default_rng(9217)
    n, d, degree = {"gaussian": (64, 3, 3), "anisotropic": (128, 8, 3),
                    "commutator": (64, 4, 4), "arabic": (128, 14, 2),
                    "hall6": (48, 2, 6)}[args.case]
    increments = rng.normal(size=(n, 10, d))
    if args.case == "anisotropic":
        increments[:, :, 3:] *= .2
    if args.case == "commutator":
        def bracket(a, b):
            return np.concatenate((a, b, -a[:, ::-1], -b[:, ::-1]), axis=1)
        increments = bracket(bracket(increments[:, :1], increments[:, 1:2]),
                             bracket(increments[:, 2:3], increments[:, 3:4]))
    paths = np.concatenate((np.zeros((n, 1, d)), increments.cumsum(axis=1)), axis=1)
    signal = increments[:, 0, 0] if args.case == "commutator" else paths[:, -1, 0]
    signal = (signal - signal.mean()) / signal.std()
    if args.loss == "logistic":
        y = rng.binomial(1, 1 / (1 + np.exp(-signal)))
    elif args.loss == "poisson":
        y = rng.poisson(np.exp(.3 * signal))
    else:
        y = signal + rng.normal(0, .25, n)
    if args.case == "arabic":
        with np.load(Path("sig_branch_and_bound_experiments/spoken_arabic_digit/results/normalized/integral_comparison/prepared.npz")) as saved:
            labels = saved["response"].argmax(axis=1)
            indices = np.sort(np.random.default_rng(9217).choice(len(labels), n, replace=False))
            paths = saved["train"][indices].astype(float)
            y = labels[indices] == 4
            assert 0 < y.sum() < n
    y = np.ascontiguousarray(y, dtype=float)
    alpha = {"commutator": 1e-5, "arabic": .001}.get(args.case, .003)
    options = dict(feature_type=args.mode, n_jobs=args.jobs, tol=1e-7)
    if args.variant == "legacy":
        options.update(bound_method="exact", search_method="grouped_signed")
    elif args.variant == "native":
        options.update(loss=args.loss)
    else:
        from benchmarks.general_loss_working_set import general_loss_fit
        from benchmarks.benchmark_unified_general_loss import make_loss
        callback = make_loss(y, args.loss)
    from benchmarks.benchmark_lasso_features import process_peak_bytes
    fits, predictions = [], []
    for rep in range(args.repeats + 1):
        start = perf_counter()
        if args.variant == "prototype":
            model = general_loss_fit(paths, degree, args.mode, callback, alpha, jobs=args.jobs)
            model.update(dimension=d, degree=degree, feature_type=args.mode, normalization="factorial_variation",
                         time_aug=False, lead_lag=False, end_time=1., feature_convention="reverse_degree_hall_v1" if args.mode=="hall" else args.mode)
        else:
            model = pysiglib.sig_lasso(paths, y, degree, alpha, **options)
        fit_time = perf_counter() - start
        start = perf_counter()
        prediction = pysiglib.sig_lasso_predict(paths, model, n_jobs=args.jobs)
        if rep:
            fits.append(fit_time * 1000)
            predictions.append((perf_counter() - start) * 1000)
        if rep < args.repeats:
            del prediction, model
    rss = process_peak_bytes()
    from benchmarks.lasso_feature_reference import dictionary
    from scipy.special import expit
    words, matrix = dictionary(paths, degree, args.mode)
    columns = [words.index(w) for w in model["words"]]
    eta = model["intercept"] + matrix[:, columns] @ model["coef"]
    if args.loss == "poisson":
        value, gradient = np.exp(eta)-y*eta, np.exp(eta)-y
    elif args.loss == "logistic":
        value, gradient = np.logaddexp(0, eta)-y*eta, expit(eta)-y
    else:
        error = eta-y
        gradient = np.clip(error, -1, 1) if args.loss == "huber" else error
        value = gradient*(error-.5*gradient)
    objective = value.mean()+alpha*np.abs(model["coef"]).sum()
    scores = matrix.T @ gradient / n
    error = np.maximum(np.abs(scores)-alpha, 0.)
    error[columns] = np.abs(scores[columns]+alpha*np.sign(model["coef"]))
    kkt = max(abs(gradient.mean()),error.max())
    prediction_error = float(np.abs(eta-prediction).max())
    stats = model["stats"]
    if not isinstance(stats, dict):
        stats = {name: getattr(stats, name) for name, _ in stats._fields_}
    with open(pysiglib.load_siglib.CPSIG._name, "rb") as library:
        build_id = hashlib.sha256(library.read()).hexdigest()
    print(json.dumps(dict(variant=args.variant, case=args.case, mode=args.mode, loss=args.loss, jobs=args.jobs,
                          workload_version=2, build_id=build_id, prediction_backend="baseline" if using_baseline else "native",
                          fit_ms=np.percentile(fits,[25,50,75]).tolist(), predict_ms=np.percentile(predictions,[25,50,75]).tolist(),
                          fit_samples_ms=fits, predict_samples_ms=predictions, rss=rss, support=len(model["words"]),
                          converged=bool(model.get("converged", True)),
                          objective=objective, kkt=kkt, prediction_error=prediction_error, stats=stats)))


if __name__ == "__main__":
    main()
