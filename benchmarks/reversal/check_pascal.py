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

"""Exact Pascal inversion checks and floating-point reversal stress cases."""

import argparse
import json
from math import factorial
from pathlib import Path
import sys

import numpy as np
import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plugins/cuda"))
import pysiglib


def exact_checks():
    count = 0
    for m in range(2, 7):
        for n in range(1, 4):
            variables = sp.symbols(f"x:{2 * n}")
            for repeated in [False, True]:
                word = [0 if repeated else j % 2 for j in range(m)]
                coefficients = [sp.Integer(1)] + [sp.Integer(0)] * m
                values = {v: sp.Rational((i * 3 + m) % 5 - 2, 4) for i, v in enumerate(variables)}
                for i in range(n):
                    old = coefficients[:]
                    for j in range(1, m + 1):
                        product = sp.Integer(1)
                        for k in range(1, j + 1):
                            product *= variables[2 * i + word[j - k]]
                            coefficients[j] += old[j - k] * product / factorial(k)
                for all_prefixes in [False, True]:
                    seeds = [sp.Integer(0)] + [sp.Rational(j, 3) if all_prefixes or j == m else sp.Integer(0) for j in range(1, m + 1)]
                    objective = sum(a * b for a, b in zip(seeds, coefficients))
                    expected = [sp.diff(objective, v).subs(values) for v in variables]
                    u = [sp.simplify(c.subs(values) * factorial(j)) for j, c in enumerate(coefficients)]
                    adjoint = [seeds[j] / factorial(j) for j in range(m + 1)]
                    gradient = [sp.Integer(0)] * (2 * n)
                    for i in reversed(range(n)):
                        a = [0] + [values[variables[2 * i + c]] for c in word]
                        q, dq = u[m - 1], a[m] * adjoint[m]
                        for s in reversed(range(1, m)):
                            adjoint[m - 1] += dq
                            for j in range(s, m):
                                gradient[2 * i + word[j - 1]] += u[j - 1] * adjoint[j]
                                if j > 1:
                                    adjoint[j - 1] += a[j] * adjoint[j]
                                u[j] -= a[j] * u[j - 1]
                            q += u[m - 1]
                        adjoint[m - 1] += dq
                        gradient[2 * i + word[m - 1]] += adjoint[m] * q
                    assert gradient == expected
                    assert u[:m] == [1] + [0] * (m - 1)
                    count += 1
    return count


def stress():
    records = []
    for device in ["cpu", "cuda"]:
        for dtype in [torch.float32, torch.float64]:
            for n in [16, 256, 4096]:
                for kind in ["brownian", "cancellation", "zero"]:
                    rng = np.random.default_rng(992 + n)
                    points = rng.normal(scale=.03, size=(3, n + 1, 3))
                    if kind == "brownian":
                        points = points.cumsum(1)
                    elif kind == "zero":
                        points.fill(0)
                    path = torch.tensor(points, dtype=dtype, device=device)
                    words = [(), (0,), (0, 1, 2, 0, 1, 2), (1,) * 12, tuple(i % 3 for i in range(20))]
                    final = pysiglib.sig_coef(path, words, prefixes=True, method="horner")
                    seeds = torch.tensor(rng.normal(size=final.shape), dtype=dtype, device=device)
                    actual = pysiglib.sig_coef_backprop(path, words, final, seeds, method="pascal")
                    horner = pysiglib.sig_coef_backprop(path, words, final, seeds, method="horner")
                    reference64 = pysiglib.sig_coef(path.double(), words, prefixes=True, method="horner")
                    reference = pysiglib.sig_coef_backprop(path.double(), words, reference64, seeds.double(), method="horner")
                    error = actual.double() - reference
                    horner_error = horner.double() - reference
                    tolerance = 3e-5 if dtype == torch.float32 else 2e-11
                    records.append(dict(device=device, dtype=str(dtype), segments=n, kind=kind,
                                        max_absolute=float(error.abs().max()), relative_l2=float(error.norm() / reference.norm()),
                                        pass_tolerance=bool(torch.all(error.abs() <= tolerance * (1 + reference.abs()))),
                                        horner_max_absolute=float(horner_error.abs().max()),
                                        horner_relative_l2=float(horner_error.norm() / reference.norm()),
                                        horner_pass_tolerance=bool(torch.all(horner_error.abs() <= tolerance * (1 + reference.abs())))))
    for dtype in [np.float32, np.float64]:
        for degree in [32, 35, 64, 171]:
            path = np.array([[0.], [.5], [-.25], [1.]], dtype=dtype)
            word = (0,) * degree
            final = pysiglib.sig_coef(path, word, prefixes=True, method="pascal")
            seeds = np.ones_like(final)
            actual = pysiglib.sig_coef_backprop(path, word, final, seeds, method="pascal")
            endpoint = sum(1 / factorial(j) for j in range(degree))
            reference = np.array([[-endpoint], [0], [0], [endpoint]])
            tolerance = 3e-5 if dtype == np.float32 else 2e-11
            records.append(dict(device="cpu", dtype=str(dtype), degree=degree, kind="generic_degree",
                                max_absolute=float(np.max(np.abs(actual - reference))),
                                pass_tolerance=bool(np.allclose(actual, reference, rtol=tolerance, atol=tolerance))))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(exact_checks=exact_checks(), stress=stress())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii", newline="\r\n")
    print(f"{report['exact_checks']} exact checks; {sum(not r['pass_tolerance'] for r in report['stress'])} stress tolerance failures")
