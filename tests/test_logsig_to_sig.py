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
import torch
import pytest

import native_api as pysiglib
from conftest import check_close, DEVICES


# =========================================================================
# Forward: round-trip sig -> log_sig -> logsig_to_sig
# =========================================================================

ROUND_TRIP_CASES = [
    {"dim": 2, "degree": 1},
    {"dim": 2, "degree": 3},
    {"dim": 3, "degree": 3},
    {"dim": 4, "degree": 4},
    {"dim": 5, "degree": 2},
]


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ROUND_TRIP_CASES)
def test_logsig_to_sig_roundtrip_single(method, dtype, case):
    rng = np.random.default_rng(42)
    dim, degree = case["dim"], case["degree"]
    path = rng.uniform(size=(20, dim)).astype(dtype)

    pysiglib.prepare_log_sig(dim, degree, method=2)
    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=method)
    sig_recovered = pysiglib.logsig_to_sig(log_sig, dim, degree, method=method)

    check_close(sig_orig, sig_recovered, double_atol=1e-10)


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ROUND_TRIP_CASES)
def test_logsig_to_sig_roundtrip_batch(method, dtype, case):
    rng = np.random.default_rng(42)
    dim, degree = case["dim"], case["degree"]
    path = rng.uniform(size=(8, 20, dim)).astype(dtype)

    pysiglib.prepare_log_sig(dim, degree, method=2)
    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=method)
    sig_recovered = pysiglib.logsig_to_sig(log_sig, dim, degree, method=method)

    check_close(sig_orig, sig_recovered, double_atol=1e-10)


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_logsig_to_sig_roundtrip_parallel(method, dtype):
    rng = np.random.default_rng(42)
    dim, degree = 3, 3
    path = rng.uniform(size=(16, 20, dim)).astype(dtype)

    pysiglib.prepare_log_sig(dim, degree, method=2)
    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=method)
    sig_recovered = pysiglib.logsig_to_sig(log_sig, dim, degree, method=method, n_jobs=-1)

    check_close(sig_orig, sig_recovered, double_atol=1e-10)


# =========================================================================
# Edge cases
# =========================================================================

def test_logsig_to_sig_zero_gives_identity():
    dim, degree = 3, 3
    sig_len = pysiglib.sig_length(dim, degree)
    log_sig = np.zeros(sig_len, dtype=np.float64)

    sig = pysiglib.logsig_to_sig(log_sig, dim, degree, method=0)

    # scalar_term=False: identity sig is the all-zeros vector (scalar slot is absent).
    expected = np.zeros(sig_len, dtype=np.float64)
    check_close(expected, sig, double_atol=1e-15)


def test_logsig_to_sig_degree_1():
    dim = 4
    log_sig = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    sig = pysiglib.logsig_to_sig(log_sig, dim, 1, method=0)

    expected = np.array([1.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    check_close(expected, sig, double_atol=1e-15)


def test_logsig_to_sig_invalid_method_raises():
    sig_len = pysiglib.sig_length(3, 3)
    log_sig = np.zeros(sig_len, dtype=np.float64)

    with pytest.raises(ValueError):
        pysiglib.logsig_to_sig(log_sig, 3, 3, method=3)

    with pytest.raises(ValueError):
        pysiglib.logsig_to_sig(log_sig, 3, 3, method=-1)


# =========================================================================
# Methods 1 and 2: round-trip
# =========================================================================

METHOD_CASES = [
    {"dim": 2, "degree": 3, "method": 1},
    {"dim": 3, "degree": 3, "method": 1},
    {"dim": 3, "degree": 4, "method": 1},
    {"dim": 2, "degree": 3, "method": 2},
    {"dim": 3, "degree": 3, "method": 2},
    {"dim": 3, "degree": 4, "method": 2},
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", METHOD_CASES)
def test_logsig_to_sig_method_roundtrip(dtype, case):
    rng = np.random.default_rng(42)
    dim, degree, method = case["dim"], case["degree"], case["method"]
    path = rng.uniform(size=(4, 20, dim)).astype(dtype)

    pysiglib.prepare_log_sig(dim, degree, method=2)
    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=method)
    sig_recovered = pysiglib.logsig_to_sig(log_sig, dim, degree, method=method)

    check_close(sig_orig, sig_recovered, double_atol=1e-10)


@pytest.mark.parametrize("case", [
    {"dim": 3, "degree": 3, "method": 1},
    {"dim": 3, "degree": 3, "method": 2},
])
def test_logsig_to_sig_method_backprop(case):
    rng = np.random.default_rng(123)
    dim, degree, method = case["dim"], case["degree"], case["method"]
    pysiglib.prepare_log_sig(dim, degree, method=2)

    path = rng.uniform(size=(10, dim)).astype(np.float64)
    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=method)

    sig_len = pysiglib.sig_length(dim, degree)
    weights = rng.uniform(size=sig_len).astype(np.float64)

    grad_analytic = pysiglib.logsig_to_sig_backprop(log_sig, weights, dim, degree, method=method)

    eps = 1e-7
    grad_numerical = np.zeros_like(log_sig)
    for i in range(len(log_sig)):
        lp = log_sig.copy()
        lp[i] += eps
        sp = pysiglib.logsig_to_sig(lp, dim, degree, method=method)

        lm = log_sig.copy()
        lm[i] -= eps
        sm = pysiglib.logsig_to_sig(lm, dim, degree, method=method)

        grad_numerical[i] = np.dot(weights, (sp - sm) / (2 * eps))

    check_close(grad_analytic, grad_numerical, double_atol=1e-5)


# =========================================================================
# Backward: numerical gradient check
# =========================================================================

@pytest.mark.parametrize("case", [
    {"dim": 2, "degree": 2},
    {"dim": 3, "degree": 3},
    {"dim": 2, "degree": 4},
])
def test_logsig_to_sig_backprop_numerical(case):
    rng = np.random.default_rng(123)
    dim, degree = case["dim"], case["degree"]
    sig_len = pysiglib.sig_length(dim, degree)

    path = rng.uniform(size=(10, dim)).astype(np.float64)
    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=0)

    weights = rng.uniform(size=sig_len).astype(np.float64)

    # Analytic gradient
    grad_analytic = pysiglib.logsig_to_sig_backprop(log_sig, weights, dim, degree, method=0)

    # Numerical gradient
    eps = 1e-7
    grad_numerical = np.zeros_like(log_sig)
    for i in range(sig_len):
        log_sig_p = log_sig.copy()
        log_sig_p[i] += eps
        sig_p = pysiglib.logsig_to_sig(log_sig_p, dim, degree, method=0)

        log_sig_m = log_sig.copy()
        log_sig_m[i] -= eps
        sig_m = pysiglib.logsig_to_sig(log_sig_m, dim, degree, method=0)

        grad_numerical[i] = np.dot(weights, (sig_p - sig_m) / (2 * eps))

    check_close(grad_analytic, grad_numerical, double_atol=1e-5)


@pytest.mark.parametrize("case", [
    {"dim": 2, "degree": 3},
    {"dim": 3, "degree": 2},
])
def test_logsig_to_sig_backprop_batch(case):
    rng = np.random.default_rng(456)
    dim, degree = case["dim"], case["degree"]
    batch = 4

    path = rng.uniform(size=(batch, 15, dim)).astype(np.float64)
    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=0)
    weights = rng.uniform(size=sig_orig.shape).astype(np.float64)

    grad = pysiglib.logsig_to_sig_backprop(log_sig, weights, dim, degree, method=0)
    grad_par = pysiglib.logsig_to_sig_backprop(log_sig, weights, dim, degree, method=0, n_jobs=-1)

    check_close(grad, grad_par, double_atol=1e-12)


# =========================================================================
# Torch tensor support
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_logsig_to_sig_torch_roundtrip(device):
    dim, degree = 3, 3
    path = torch.rand(8, 20, dim, dtype=torch.float64, device=device)

    sig_orig = pysiglib.sig(path, degree)
    log_sig = pysiglib.sig_to_log_sig(sig_orig, dim, degree, method=0)
    sig_recovered = pysiglib.logsig_to_sig(log_sig, dim, degree, method=0)

    check_close(sig_orig, sig_recovered, double_atol=1e-10)
