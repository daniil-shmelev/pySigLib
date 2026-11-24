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

"""Same-split predictive controls and outer-fold checks on MUTAG."""

import argparse
import gc
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from datasets import dictionary_factory, problem, take
from experiment import FitMemory, fit, prime, score, tune
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from solver import audit, predict
from threadpoolctl import threadpool_limits


def wl_counts(graph, h):
    labels, adj = graph
    codes = [str(a) for a in labels]
    features = Counter(("0:" + v) for v in codes)
    for step in range(1, h + 1):
        codes = [
            v + "(" + ",".join(sorted(codes[i] for i in neighbours)) + ")"
            for v, neighbours in zip(codes, adj)
        ]
        features.update(f"{step}:{v}" for v in codes)
    return features


def wl_kernel(graphs, h):
    # Each feature name is a deterministic rooted neighbourhood. Its coding is independent of the split.
    features = DictVectorizer().fit_transform([wl_counts(g, h) for g in graphs])
    z = normalize(features, axis=1)
    return (z @ z.T).toarray()


class WLSVM:
    def __init__(self, depth, c):
        self.depth = depth
        self.vectorizer = DictVectorizer()
        self.svm = SVC(kernel="precomputed", C=c)

    def fit(self, x, y):
        counts = [wl_counts(g, self.depth) for g in x]
        self.training = normalize(self.vectorizer.fit_transform(counts))
        self.svm.fit((self.training @ self.training.T).toarray(), y)
        return self

    def predict(self, x):
        counts = [wl_counts(g, self.depth) for g in x]
        lengths = np.sqrt([sum(v * v for v in c.values()) for c in counts])
        test = self.vectorizer.transform(counts).multiply(1 / lengths[:, None])
        return self.svm.predict((test @ self.training.T).toarray())


def external(case, parameter):
    if case == "sms":
        vec = CountVectorizer(
            lowercase=False, tokenizer=str.split, token_pattern=None, binary=True, ngram_range=(1, 2)
        )
        return make_pipeline(vec, LinearSVC(C=parameter, dual=True, random_state=603, max_iter=10000))
    if case == "intervals":
        return HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=parameter, early_stopping=False, random_state=603
        )
    if case == "mushroom":
        return make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            DecisionTreeClassifier(random_state=603, max_depth=parameter),
        )
    if case == "tanks":
        return Ridge(alpha=parameter)
    if case == "motifs":
        return WLSVM(*parameter)
    return KNeighborsClassifier(n_neighbors=1, algorithm="brute", n_jobs=1)


def mutag(data):
    p = problem("motifs", data)
    kernels = {h: wl_kernel(p.x, h) for h in (1, 2, 3)}
    labels = p.y[:, 0]

    def accuracy(h, c, train, test):
        k = kernels[h]
        svm = SVC(kernel="precomputed", C=c).fit(k[np.ix_(train, train)], labels[train])
        return np.mean(svm.predict(k[np.ix_(test, train)]) == labels[test])

    result = []
    for fold in range(5):
        p = problem("motifs", data, fold=fold)
        a, v, t = p.train, p.valid, p.test
        base = np.r_[a, v]

        _, h, c = min((-accuracy(h, c, a, v), h, c) for h in kernels for c in (0.1, 1.0, 10.0))
        item = {"fold": fold, "n": len(t), "wl": float(accuracy(h, c, base, t)), "wl_depth": h, "C": c}
        for case in ("motifs", "graph-scattering"):
            config = tune(p, case)
            x, y = take(p.x, base), p.y[base]
            factory = dictionary_factory(case, x)
            train = factory(x)
            model = fit("tree", train, y, config["alpha"])[0]
            matrix, keys = train.explicit()
            err = audit(matrix, keys, y[:, 0], config["alpha"][0], model)
            if err > 1.01e-7:
                raise ArithmeticError("graph outer-fold audit failed")
            pred = predict(factory(take(p.x, t)), [model])
            item[case] = score("binary", p.y[t], pred)["accuracy"]
        result.append(item)
    return result


def tied_columns(case, data, output, arabic):
    p = problem(case, data, arabic)
    config = json.loads((output / case / "validation.json").read_text())
    idx = np.r_[p.train, p.valid] if p.refit else p.train
    x, y = take(p.x, idx), p.y[idx]
    factory = dictionary_factory(case, x)
    d, test = factory(x), factory(take(p.x, p.test))
    results = [
        fit(method, search, y, config["alpha"]) for method, search in (("tree", d), ("full", d.explicit()))
    ]
    train_diff = representative_diff = varying_groups = 0
    for t, f in zip(*results):
        keys = sorted(set(t.keys) | set(f.keys))
        x, xt = d.columns(keys), test.columns(keys)
        positions = {key: i for i, key in enumerate(keys)}
        groups = {}
        for j, col in enumerate(x.T):
            # Complementary indicators differ only by a sign and the free intercept.
            sign = 1 if col[0] == 0 else -1
            code = np.packbits(col != (0 if sign == 1 else 1)).tobytes()
            groups.setdefault(code, []).append((j, sign))
        w = np.zeros((len(keys), 2))
        b = np.array([t.intercept, f.intercept])
        for j, m in enumerate((t, f)):
            w[[positions[k] for k in m.keys], j] = m.coef
        train_diff = max(train_diff, float(np.max(np.abs(np.diff(x @ w + b, axis=1)))))
        canonical_predictions = np.broadcast_to(b, (test.n, 2)).copy()
        for peers in groups.values():
            j0, s0 = peers[0]
            rep = xt[:, j0] if s0 == 1 else 1 - xt[:, j0]
            if any(np.any((xt[:, j] if s == 1 else 1 - xt[:, j]) != rep) for j, s in peers):
                varying_groups += 1
            for j, s in peers:
                canonical_predictions += (rep if s == 1 else 1 - rep)[:, None] * w[j]
        representative_diff = max(
            representative_diff, float(np.abs(np.diff(canonical_predictions, axis=1)).max())
        )
    return {
        "max_training_score_difference": train_diff,
        "max_test_difference_common_representative": representative_diff,
        "training_equivalence_groups_varying_on_test_summed_over_responses": varying_groups,
    }


def controls(data, results):
    def evaluate(case, p, train, test, param):
        x, xt = take(p.x, train), take(p.x, test)
        if case == "sms":
            x, xt = [" ".join(row) for row in x], [" ".join(row) for row in xt]
        model = external(case, param)
        pred = model.fit(x, p.y[train, 0]).predict(xt)[:, None]
        return score(p.kind, p.y[test], pred)

    out = {}
    for case, name in (("gapped", "gunpoint"), ("ecg-gapped", "ecg200"), ("wafer-gapped", "wafer")):
        p = problem(case, data)
        idx = np.r_[p.train, p.valid]
        neighbour = np.concatenate(
            [euclidean_distances(p.x[part], p.x[idx]).argmin(axis=1) for part in np.array_split(p.test, 8)]
        )
        out[name + "_1nn"] = float(np.mean(p.y[p.test] == p.y[idx[neighbour]]))
    for case in ("sms", "intervals", "tanks", "mushroom"):
        p = problem(case, data)

        grid = {
            "sms": (0.1, 1.0, 10.0),
            "intervals": (15, 31),
            "tanks": (0.001, 0.1, 10.0),
            "mushroom": (3, 6, None),
        }[case]
        options = [evaluate(case, p, p.train, p.valid, par) for par in grid]
        i = min(range(len(grid)), key=lambda j: (-options[j].get("accuracy", 0.0), options[j]["mse"]))
        out[case] = dict(parameter=grid[i], **evaluate(case, p, np.r_[p.train, p.valid], p.test, grid[i]))
    out["mutag_folds"] = mutag(data)
    p = problem("walsh", data)
    known = set(map(tuple, p.x[np.r_[p.train, p.valid]]))
    unseen = np.array([tuple(row) not in known for row in p.x[p.test]])
    file = results / "walsh" / "tree-0.npy"
    if file.exists():
        pred = np.load(file)
        out["monk2_unseen"] = dict(
            n=int(unseen.sum()), **score("binary", p.y[p.test][unseen], pred[unseen])
        )
    return out


def time_external(case, data, output):
    cfg = json.loads((output / "controls.json").read_text())
    p = problem(case, data)
    base = np.r_[p.train, p.valid]
    x, xt = take(p.x, base), take(p.x, p.test)
    param = cfg.get(case, {}).get("parameter")
    if case == "motifs":
        param = (cfg["mutag_folds"][0]["wl_depth"], cfg["mutag_folds"][0]["C"])
    prime()
    gc.collect()
    sample = FitMemory()
    begin = perf_counter()
    if case == "sms":
        x = [" ".join(row) for row in x]
    model = external(case, param).fit(x, p.y[base, 0])
    end = perf_counter()
    memory = sample.finish()
    pred_start = perf_counter()
    if case == "sms":
        xt = [" ".join(row) for row in xt]
    pred = model.predict(xt)[:, None]
    return dict(
        fit_s=end - begin,
        predict_s=perf_counter() - pred_start,
        **memory,
        **score(p.kind, p.y[p.test], pred),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arabic", type=Path)
    ap.add_argument("--timed", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--rep", type=int, default=0)
    args = ap.parse_args()
    if args.case:
        with threadpool_limits(limits=1):
            record = time_external(args.case, args.data, args.output)
        (args.output / args.case / f"external-{args.rep}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="ascii"
        )
        sys.exit(0)
    if args.timed:
        cases = ("gapped", "ecg-gapped", "wafer-gapped", "sms", "motifs", "mushroom", "intervals", "tanks")
        for i in range(3):
            for case in cases[i:] + cases[:i]:
                if not (args.output / case / f"external-{i}.json").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--case",
                            case,
                            "--rep",
                            str(i),
                            "--data",
                            str(args.data),
                            "--output",
                            str(args.output),
                        ],
                        check=True,
                        timeout=600,
                    )
        sys.exit(0)
    with threadpool_limits(limits=1):
        results = controls(args.data, args.output)
        results["ties"] = {
            c: tied_columns(c, args.data, args.output, args.arabic)
            for c in ("gapped", "arabic-gapped", "arabic-frame-words")
        }
    (args.output / "controls.json").write_text(json.dumps(results, indent=2) + "\n", encoding="ascii")
