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

"""Train-only dictionary definitions and reproducible evaluation splits."""

import hashlib
import io
import re
from dataclasses import dataclass
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from graphs import Graphs
from intervals import Intervals
from numpy.lib.stride_tricks import sliding_window_view
from products import Products
from scattering import GraphScattering, Haar
from scipy.io import loadmat
from sequences import Sequences
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from subsequences import Subsequences
from templates import Templates
from walsh import Walsh

ARCHIVES = {
    "ECG200": (
        "https://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip",
        "755937ea37849346ea20033666b19d17345bb039971e8fa33458b16b62c09380",
    ),
    "Wafer": (
        "https://www.timeseriesclassification.com/aeon-toolkit/Wafer.zip",
        "06032db83b3f6f3f39ef4b65433061dde9b2593b956caa1ad1ae9ad260a36be4",
    ),
    "arabic": (
        "https://archive.ics.uci.edu/static/public/195/spoken+arabic+digit.zip",
        "3c8c83404289c49631605421830872be337d92d887e61798385aacbe9423010d",
    ),
    "monks": (
        "https://archive.ics.uci.edu/static/public/70/monk+s+problems.zip",
        "7501acef87251b1d427726d7c381a2488c166fbbf70ac4096e09f05951c30a29",
    ),
    "bike": (
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
    ),
    "GunPoint": (
        "https://www.timeseriesclassification.com/aeon-toolkit/GunPoint.zip",
        "d7513cfe222418fabfdb5a6434ffb21ac3de4923e637971e9388ebc857816803",
    ),
    "MUTAG": (
        "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip",
        "c419bdc853c367d2d83da4973c45100954ae15e10f5ae2cddde6ca431f8207f6",
    ),
    "mushroom": (
        "https://archive.ics.uci.edu/static/public/73/mushroom.zip",
        "face32f32647e0d939f6233f36dd30dd5d619ae9f3f9b8e10bea4ac7e1f60b1a",
    ),
    "sms": (
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
    ),
    "tanks": (
        "https://drive.usercontent.google.com/download?id=1HnQf_gu0g_UlggoBqy2s34l9YJiFdN01&export=download&confirm=t",
        "eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51",
    ),
}


def archive(data, name):
    path = data / (name + ".zip")
    url, digest = ARCHIVES[name]
    if not path.exists():
        data.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as stream:
            blob = stream.read()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"download checksum mismatch for {name}")
        path.write_bytes(blob)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"checksum mismatch for {name}")
    return ZipFile(path)


@dataclass
class Problem:
    x: object
    y: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    kind: str
    refit: bool = True


CASES = (
    "mushroom",
    "tanks",
    "legendre",
    "splines",
    "sms",
    "motifs",
    "graph-scattering",
    "haar",
    "arabic",
    "shapelets",
    "landmarks",
    "intervals",
    "gapped",
    "arabic-gapped",
    "walsh",
    "arabic-frames",
    "arabic-frame-words",
    "ecg-gapped",
    "wafer-gapped",
    "shapelet-distance",
)


def arabic_paths(data):
    with archive(data, "arabic") as z:
        raw = []
        for name in ("Train", "Test"):
            text = z.read(name + "_Arabic_Digit.txt").decode("ascii").strip()
            raw.append(re.split(r"\n\s*\n", text))
    rng = np.random.default_rng(311923)
    ids = [rng.permutation(np.arange(660) + j * 660) for j in range(10)]
    order = np.r_[np.concatenate([g[:60] for g in ids]), np.concatenate([g[60:90] for g in ids])]
    grid = np.linspace(0.0, 1.0, 24)

    def resample(block):
        v = np.loadtxt(io.StringIO(block), ndmin=2)
        return np.column_stack([np.interp(grid, np.linspace(0.0, 1.0, len(v)), col) for col in v.T])

    frames = np.stack([resample(raw[0][i]) for i in order] + [resample(b) for b in raw[1]])
    pca = PCA(svd_solver="full").fit(frames[:600].reshape(-1, 13))
    values = (frames - pca.mean_) @ pca.components_.T / (2 * np.sqrt(pca.explained_variance_[0]))
    paths = np.pad(np.cumsum(values / 24, axis=1), ((0, 0), (1, 0), (0, 0)))
    labels = np.r_[order // 660, np.repeat(np.arange(10), 220)]
    return paths, np.eye(10)[labels]


def problem(case, data, arabic=None, fold=0):
    if case == "walsh":
        with archive(data, "monks") as z:
            parts = [
                np.loadtxt(io.BytesIO(z.read("monks-2." + s)), usecols=range(7), dtype=int)
                for s in ("train", "test")
            ]
        a = np.vstack(parts)
        x = np.column_stack(
            [
                ((a[:, j + 1] - 1) >> b) & 1
                for j, size in enumerate([3, 3, 2, 3, 4, 2])
                for b in range((size - 1).bit_length())
            ]
        )
        y = a[:, :1].astype(float)
        n = len(parts[0])
        train, valid = train_test_split(np.arange(n), stratify=y[:n, 0], test_size=0.25, random_state=603)
        return Problem(x, y, train, valid, np.arange(n, len(a)), "binary")
    if case in ("motifs", "graph-scattering"):
        with archive(data, "MUTAG") as z:

            def read(name):
                return np.loadtxt(io.BytesIO(z.read(f"MUTAG/MUTAG_{name}.txt")), dtype=int, delimiter=",")

            ids = read("graph_indicator") - 1
            labels, bonds = read("node_labels"), read("edge_labels") + 1
            edges, target = read("A") - 1, read("graph_labels")
        x = []
        sizes = np.bincount(ids)
        start = np.cumsum(sizes) - sizes
        for i, size in enumerate(sizes):
            x.append((labels[ids == i].tolist(), [{} for _ in range(size)]))
        for (a, b), bond in zip(edges, bonds):
            i = ids[a]
            x[i][1][a - start[i]][b - start[i]] = int(bond)
        y = (target > 0).astype(float)
    elif case in ("haar", "shapelets", "shapelet-distance", "gapped", "ecg-gapped", "wafer-gapped"):
        name = {"ecg-gapped": "ECG200", "wafer-gapped": "Wafer"}.get(case, "GunPoint")
        with archive(data, name) as z:
            parts = [np.loadtxt(io.BytesIO(z.read(f"{name}_{split}.txt"))) for split in ("TRAIN", "TEST")]
        values = np.vstack(parts)
        x = values[:, 1:]
        y = (values[:, 0] == parts[0][:, 0].max()).astype(float)
        n = len(parts[0])
        train, valid = train_test_split(np.arange(n), test_size=0.24, random_state=603, stratify=y[:n])
        return Problem(x, y[:, None], train, valid, np.arange(n, len(y)), "binary")
    elif case == "tanks":
        with archive(data, "tanks") as z:
            values = loadmat(io.BytesIO(z.read("CascadedTanksFiles/dataBenchmark.mat")))
        # An input-only Volterra filter. Test outputs never enter its predictors.
        lags = np.array([0, 1, 2, 3, 5, 7, 10, 14, 19, 25, 32, 40, 50])
        inputs, outputs = [], []
        for split in ("Est", "Val"):
            u, y = values["u" + split].ravel(), values["y" + split].ravel()
            idx = np.arange(50, len(u))
            inputs.append(u[idx[:, None] - lags])
            outputs.append(y[idx])
        n = len(inputs[0])
        return Problem(
            np.vstack(inputs),
            np.concatenate(outputs)[:, None],
            np.arange(0, 730),
            np.arange(730, n),
            np.arange(n, n + len(inputs[1])),
            "regression",
        )
    elif case == "mushroom":
        with archive(data, "mushroom") as z:
            a = np.loadtxt(io.BytesIO(z.read("agaricus-lepiota.data")), dtype="U1", delimiter=",")
        x, y = a[:, 1:], (a[:, 0] == "p").astype(float)
    elif case == "sms":
        with archive(data, "sms") as z:
            lines = z.read("SMSSpamCollection").decode("utf-8").splitlines()
        unique = {}
        for line in lines:
            if "\t" not in line:
                continue
            label, message = line.split("\t", 1)
            words = tuple(re.findall(r"\w+|[^\w\s]", message.casefold()))
            unique.setdefault(words, set()).add(label)
        x = [s for s, labels in unique.items() if len(labels) == 1 and s]
        y = np.array([float("spam" in unique[s]) for s in x])
    elif case == "intervals":
        with archive(data, "bike") as z:
            columns = np.loadtxt(
                io.BytesIO(z.read("hour.csv")),
                delimiter=",",
                skiprows=1,
                usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
            )
        # Chronological split. Use the recorded weather, not future counts.
        n = len(columns)
        return Problem(
            columns[:, :-1],
            (columns[:, -1] / 1000)[:, None],
            np.arange(int(0.6 * n)),
            np.arange(int(0.6 * n), int(0.8 * n)),
            np.arange(int(0.8 * n), n),
            "regression",
        )
    elif case in ("arabic", "arabic-gapped", "arabic-frames", "arabic-frame-words"):
        if arabic is None:
            x, y = arabic_paths(data)
        else:
            with np.load(arabic) as z:
                x = np.concatenate([z["train"], z["validation"], z["test"]])[:, :, :-1]
                y = np.concatenate([z["response"], np.eye(10)[z["labels"]], np.eye(10)[z["test_labels"]]])
        if case in ("arabic-frames", "arabic-frame-words"):
            x = 24 * np.diff(x, axis=1)
        return Problem(
            x, y, np.arange(600), np.arange(600, 900), np.arange(900, len(y)), "multiclass", False
        )
    elif case in ("legendre", "splines", "landmarks"):
        rng = np.random.default_rng(603)
        if case == "legendre":
            x = rng.uniform(-1.0, 1.0, (1792, 8))
            a, b, c = (np.pi * x[:, :3]).T
            y = np.sin(a) * (1.0 + 0.1 * c**4) + 7 * np.sin(b) ** 2
        elif case == "splines":
            x, y = make_friedman1(n_samples=1792, n_features=12, noise=1.0, random_state=603)
        else:
            x = rng.uniform(-1.0, 1.0, (1792, 2))
            c = np.array([[-0.55, -0.4], [-0.35, 0.6], [0.53, 0.4], [0.4, -0.6]])
            y = np.exp(-60 * np.sum((x[:, None, :] - c) ** 2, axis=2)) @ [1.0, -0.8, 1.2, -0.6]
            y += rng.normal(0.0, 0.03, len(y))
        return Problem(
            x, y[:, None], np.arange(512), np.arange(512, 768), np.arange(768, 1792), "regression"
        )
    else:
        raise ValueError(f"unknown case {case}")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=603).split(y, y))
    base, test = folds[fold]
    train, valid = train_test_split(base, test_size=0.25, random_state=604, stratify=y[base])
    return Problem(x, y[:, None], train, valid, test, "binary")


def take(x, idx):
    return x[idx] if isinstance(x, np.ndarray) else [x[i] for i in idx]


def dictionary_factory(case, training):
    if case == "walsh":
        return Walsh
    if case == "intervals":
        cuts = []
        for x in training.T:
            values = np.unique(x)
            if len(values) <= 48:
                edges = (values[:-1] + values[1:]) / 2
            else:
                edges = np.unique(np.quantile(x, np.arange(1, 48) / 48))
                edges = edges[(edges > x.min()) & (edges < x.max())]
            cuts.append(edges)
        return lambda x: Intervals(
            [np.searchsorted(e, v, side="right") for e, v in zip(cuts, x.T)], [len(e) + 1 for e in cuts]
        )
    if case in ("gapped", "ecg-gapped", "wafer-gapped"):

        def slopes(x):
            means = np.column_stack([block.mean(axis=1) for block in np.array_split(x, 30, axis=1)])
            return np.diff(means, axis=1)

        cuts = np.quantile(slopes(training), np.arange(1, 8) / 8)
        return lambda x: Subsequences(np.searchsorted(cuts, slopes(x)), 8, 7)
    if case in ("arabic-gapped", "arabic-frame-words"):
        d = training.shape[-1]
        model = MiniBatchKMeans(n_clusters=8, n_init=3, random_state=603, batch_size=512)
        model.fit(np.ascontiguousarray(training.reshape(-1, d)))

        def factory(x):
            symbols = model.predict(np.ascontiguousarray(x.reshape(-1, d))).reshape(len(x), -1)
            return Subsequences(symbols, 8, 6)

        return factory
    if case == "mushroom":
        levels = [np.unique(v) for v in training.T]
        groups = np.repeat(np.arange(len(levels)), [len(x) for x in levels])

        def factory(x):
            values = np.vstack([(x[:, i] == v) for i, vs in enumerate(levels) for v in vs])
            return Products(values, groups, np.ones(len(groups), int), 3, sparse=True)

        return factory
    if case in ("legendre", "tanks"):
        d = training.shape[1]
        degree = 6 if case == "legendre" else 3
        centre = (training.min() + training.max()) / 2 if case == "tanks" else 0.0
        scale = max(np.max(np.abs(training - centre)), 1e-12) if case == "tanks" else 1.0

        def factory(x):
            v = (x - centre) / scale
            if case == "legendre":
                atoms = np.polynomial.legendre.legvander(v, degree)[:, :, 1:]
            else:
                atoms = v[:, :, None] ** np.arange(1, degree + 1)
            return Products(
                atoms.reshape(len(x), -1).T,
                np.repeat(np.arange(d), degree),
                np.tile(np.arange(1, degree + 1), d),
                degree,
            )

        return factory
    if case == "splines":
        knots = np.linspace(0.0, 1.0, 5)
        groups = np.repeat(np.arange(training.shape[1]), len(knots))

        def factory(x):
            hats = np.maximum(0.0, 1.0 - 4 * np.abs(x[:, :, None] - knots))
            return Products(hats.reshape(len(x), -1).T, groups, np.ones(len(groups), int), 3, sparse=True)

        return factory
    if case == "sms":
        vocab = {w: i for i, w in enumerate(sorted({w for doc in training for w in doc}), 2)}

        def factory(x):
            return Sequences([[vocab.get(t, 1) for t in row] for row in x], 4, len(vocab) + 2)

        return factory
    if case == "motifs":
        return lambda x: Graphs(x, 4)
    if case == "graph-scattering":
        channels = sorted({j for nodes, adj in training for j in nodes})
        return lambda x: GraphScattering(x, channels, 3, 3)
    if case == "haar":
        return lambda x: Haar(x, 7, 4)
    if case in ("arabic", "arabic-frames"):
        return lambda x: Haar(x, 4, 3, increasing=True)
    if case in ("shapelets", "shapelet-distance"):
        length = 16

        def windows(x):
            view = sliding_window_view(x, length, axis=1)
            return (view - view.mean(axis=2, keepdims=True)) / np.maximum(
                view.std(axis=2, keepdims=True), 1e-8
            )

        centres = np.unique(windows(training)[:, ::4].reshape(-1, length), axis=0)
        profile = "distance" if case == "shapelet-distance" else "gaussian"
        return lambda x: Templates(windows(x), centres, 1.0 / length, profile=profile)
    if case == "landmarks":
        grid = np.linspace(-1.0, 1.0, 96)
        a, b = np.meshgrid(grid, grid)
        centres = np.column_stack([a.ravel(), b.ravel()])
        return lambda x: Templates(x[:, None, :], centres, 60.0)
    raise ValueError(case)
