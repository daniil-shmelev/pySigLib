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

"""Forward prefix-suffix methods, including forced native block boundaries."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api
from conftest import DEVICES, load_fixtures

METHODS = ("ps",)


@pytest.mark.parametrize("device,dtype", product(DEVICES, (torch.float32, torch.float64)))
@pytest.mark.parametrize("time_aug,lead_lag,scalar", product((False, True), repeat=3))
def test_split_and_endpoint_variants(monkeypatch, device, dtype, time_aug, lead_lag, scalar):
    path = torch.tensor(np.random.default_rng(712).normal(scale=0.04, size=(2, 14, 2)),
                        device=device, dtype=dtype)[:, ::2]
    options = dict(time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar, n_jobs=2)
    expected = pysiglib.sig(path, 4, **options)
    seeds = torch.tensor(np.random.default_rng(713).normal(size=expected.shape), device=device, dtype=dtype)
    back_options = dict(time_aug=time_aug, lead_lag=lead_lag, n_jobs=2)
    gradient = pysiglib.sig_backprop(path, expected, seeds, 4, **back_options)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    for a in range(5):
        split = (a, 4 - a) if a else None
        actual = pysiglib.sig(path, 4, method="ps", ps_split=split, **options)
        grad = pysiglib.sig_backprop(path, expected, seeds, 4, method="ps", ps_split=split, **back_options)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(grad, gradient, atol=tolerance, rtol=tolerance)
    monkeypatch.setenv("PYSIGLIB_PS_SPLIT_A", "0")
    for block in (1, 3, 16):
        monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", str(block))
        actual = pysiglib.sig(path, 4, method="ps", **options)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_endpoint_autograd_and_stream(monkeypatch, device):
    monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "3")
    path = torch.randn(2, 6, 2, device=device, dtype=torch.float64) * 0.03
    path.requires_grad_()
    operation = lambda x: torch_api.sig(x, 4, method="ps")
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            assert torch.autograd.gradcheck(operation, (path,))
        stream.synchronize()
    else:
        assert torch.autograd.gradcheck(operation, (path,))


@pytest.mark.parametrize("device", DEVICES)
def test_generic_unbalanced_split(monkeypatch, device):
    monkeypatch.setenv("PYSIGLIB_PS_COEFFICIENT_FACTORS", "0")
    monkeypatch.setenv("PYSIGLIB_PS_SPLIT_A", "5")
    path = torch.randn(2, 8, 3, device=device, dtype=torch.float64) * 0.03
    expected = pysiglib.sig(path, 5)
    actual = pysiglib.sig(path, 5, method="ps")
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("device,dtype", product(DEVICES, (torch.float32, torch.float64)))
@pytest.mark.parametrize("dimension", (8, 16))
def test_automatic_prefix_suffix_dispatch(monkeypatch, device, dtype, dimension):
    path = torch.tensor(np.random.default_rng(714).normal(scale=0.02, size=(32, 257, dimension)),
                        device=device, dtype=dtype)
    monkeypatch.setenv("PYSIGLIB_PS_DISPATCH", "0")
    expected = pysiglib.sig(path, 3, method="ps", n_jobs=20)
    seeds = torch.ones_like(expected)
    gradient = pysiglib.sig_backprop(path, expected, seeds, 3, method="ps", n_jobs=20)
    monkeypatch.delenv("PYSIGLIB_PS_DISPATCH")
    actual = pysiglib.sig(path, 3, method="ps", n_jobs=20)
    grad = pysiglib.sig_backprop(path, actual, seeds, 3, method="ps", n_jobs=20)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(grad, gradient, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_endpoint_workspace_limit(monkeypatch, device):
    monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "3")
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    path = torch.ones(2, 8, 3, device=device)
    with pytest.raises(Exception, match="allocate|workspace"):
        pysiglib.sig(path, 4, method="ps")


@pytest.mark.parametrize("target_name", ("SSE2", "AVX2"))
def test_endpoint_simd_targets(monkeypatch, target_name):
    from pysiglib.load_siglib import CPSIG
    if not hasattr(CPSIG, "cpsig_test_supported_targets"):
        pytest.skip("Target forcing requires a native test build")
    CPSIG.cpsig_test_supported_targets.restype = ctypes.c_int64
    CPSIG.cpsig_test_target_name.argtypes = [ctypes.c_int64]
    CPSIG.cpsig_test_target_name.restype = ctypes.c_char_p
    CPSIG.cpsig_test_set_target.argtypes = [ctypes.c_int64]
    targets = CPSIG.cpsig_test_supported_targets()
    selected = 0
    while targets:
        target = targets & -targets
        targets &= ~target
        if CPSIG.cpsig_test_target_name(target).decode() == target_name:
            selected = target
    if not selected:
        pytest.skip(f"{target_name} is unavailable")
    assert CPSIG.cpsig_test_set_target(selected) == 0
    try:
        path = np.random.default_rng(715).normal(scale=0.03, size=(7, 18, 3))
        expected = pysiglib.sig(path, 5)
        monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "4")
        actual = pysiglib.sig(path, 5, method="ps", n_jobs=3)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    finally:
        assert CPSIG.cpsig_test_set_target(0) == 0


@pytest.mark.parametrize("device,method,dtype", product(DEVICES, METHODS, (torch.float32, torch.float64)))
@pytest.mark.parametrize("dimension,degree,rows", [(2, 8, 1), (3, 8, 5), (5, 6, 7), (11, 4, 3)])
def test_paired_factor_blocks(monkeypatch, device, method, dtype, dimension, degree, rows):
    path = torch.tensor(np.random.default_rng(84).normal(size=(3, 19, dimension)).cumsum(axis=1) * 0.03,
                        device=device, dtype=dtype)
    expected = pysiglib.sig(path, degree, scalar_term=True)
    itemsize = path.element_size()
    width = dimension ** ((degree + 1) // 2) + dimension ** (degree // 2)
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", str(rows * width * itemsize))
    monkeypatch.setenv("PYSIGLIB_PS_FUSION", "0")
    for paired in (0, 1):
        monkeypatch.setenv("PYSIGLIB_PS_PAIR_LEVELS", str(paired))
        actual = pysiglib.sig(path, degree, method=method, scalar_term=True, n_jobs=3)
        tolerance = 1e-4 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device,method,dtype", product(DEVICES, METHODS, (torch.float32, torch.float64)))
def test_fusion_switch(monkeypatch, device, method, dtype):
    path = torch.tensor(np.random.default_rng(91).normal(size=(2, 129, 3)).cumsum(axis=1) * 0.02,
                        device=device, dtype=dtype)
    expected = pysiglib.sig(path, 4)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    monkeypatch.setenv("PYSIGLIB_PS_LOW_DEPTH", "1")
    for fusion in (0, 1):
        monkeypatch.setenv("PYSIGLIB_PS_FUSION", str(fusion))
        actual = pysiglib.sig(path, 4, method=method, n_jobs=1)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dimension,degree", [(2, 4), (2, 6), (3, 4), (3, 6), (4, 4), (8, 6)])
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cpu_small_paired_factors(monkeypatch, dimension, degree, dtype):
    path = np.random.default_rng(106).normal(size=(2, 13, dimension)).cumsum(axis=1).astype(dtype) * 0.03
    expected = pysiglib.sig(path, degree)
    width = 2 * dimension ** (degree // 2)
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", str(5 * width * path.itemsize))
    for fusion, paired in product((0, 1), (0, 1)):
        monkeypatch.setenv("PYSIGLIB_PS_FUSION", str(fusion))
        monkeypatch.setenv("PYSIGLIB_PS_PAIR_LEVELS", str(paired))
        actual = pysiglib.sig(path, degree, method=METHODS[0], n_jobs=3)
        tolerance = 1e-4 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device,method,dtype", product(DEVICES, METHODS, (torch.float32, torch.float64)))
@pytest.mark.parametrize("dimension,degree,length", [(2, 3, 9), (3, 6, 67), (2, 7, 131), (4, 5, 17)])
def test_larger_scan_degree(monkeypatch, device, method, dtype, dimension, degree, length):
    path = torch.tensor(np.random.default_rng(71).normal(size=(2, 1, length, dimension)).cumsum(axis=-2) * 0.02,
                        device=device, dtype=dtype)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_CHUNK", "16")
    for scalar in (False, True):
        expected = pysiglib.sig(path, degree, scalar_term=scalar)
        for m0 in range(degree // 2 + 1, degree + 1):
            monkeypatch.setenv("PYSIGLIB_PS_SCAN_DEGREE", str(m0))
            actual = pysiglib.sig(path, degree, method=method, scalar_term=scalar, n_jobs=2)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device,method", product(DEVICES, METHODS))
def test_scan_degree_transforms_and_errors(monkeypatch, device, method):
    path = torch.tensor([[0., 0.], [0.125, -0.25], [0.375, 0.125], [0., 0.]], device=device)
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_DEGREE", "4")
    actual = pysiglib.sig(path, 6, method=method, time_aug=True, lead_lag=True)
    torch.testing.assert_close(actual, pysiglib.sig(path, 6, time_aug=True, lead_lag=True), atol=1e-4, rtol=1e-4)
    for invalid in ("2", "7", "-1", "x"):
        monkeypatch.setenv("PYSIGLIB_PS_SCAN_DEGREE", invalid)
        with pytest.raises(Exception, match="PYSIGLIB_PS_SCAN_DEGREE"):
            pysiglib.sig(path, 6, method=method)


def exact_signature(increments, degree, return_exact=False):
    d = len(increments[0])
    state = [[Fraction(1)]] + [[Fraction(0)] * d ** k for k in range(1, degree + 1)]
    for increment in increments:
        exp = [[Fraction(1)]]
        for k in range(1, degree + 1):
            exp.append([x * Fraction(v) / k for x in exp[-1] for v in increment])
        state = [[sum(state[k - j][word // d ** j] * exp[j][word % d ** j]
                      for j in range(k + 1)) for word in range(d ** k)] for k in range(degree + 1)]
    values = [value for level in state for value in level]
    return values if return_exact else np.array([float(value) for value in values])


@pytest.mark.parametrize("device,method", product(DEVICES, METHODS))
def test_scan_degree_exact_reference(monkeypatch, device, method):
    increments = [[Fraction(1, 8), Fraction(-1, 4)], [Fraction(1, 4), Fraction(1, 8)],
                  [Fraction(-3, 8), Fraction(1, 8)]]
    path = torch.tensor(np.vstack((np.zeros(2), np.asarray(increments, dtype=float).cumsum(axis=0))),
                        device=device, dtype=torch.float64)
    expected = exact_signature(increments, 6)
    for m0 in (3, 4, 5, 6):
        monkeypatch.setenv("PYSIGLIB_PS_SCAN_DEGREE", str(m0))
        actual = pysiglib.sig(path, 6, method=method, scalar_term=True)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device,method,dtype", product(DEVICES, METHODS, (torch.float32, torch.float64)))
@pytest.mark.parametrize("dimension,degree,length", [(1, 7, 13), (2, 0, 7), (2, 1, 7), (2, 6, 0),
    (2, 5, 1), (3, 5, 2), (2, 5, 12), (3, 4, 14), (3, 6, 130), (4, 5, 259)])
def test_matches_horner(device, method, dtype, dimension, degree, length):
    path = torch.tensor(np.random.default_rng(5).normal(size=(2, 3, length, dimension)).cumsum(axis=-2) * 0.03,
                        device=device, dtype=dtype)
    for scalar in (False, True):
        actual = pysiglib.sig(path, degree, method=method, scalar_term=scalar, n_jobs=2)
        expected = pysiglib.sig(path, degree, scalar_term=scalar, n_jobs=2)
        assert actual.shape == expected.shape
        assert actual.dtype == path.dtype and actual.device == path.device
        torch.testing.assert_close(actual, expected, atol=1e-4 if dtype == torch.float32 else 1e-10,
                                   rtol=1e-4 if dtype == torch.float32 else 1e-10)


@pytest.mark.parametrize("device,method,dtype", product(DEVICES, METHODS, (torch.float32, torch.float64)))
@pytest.mark.parametrize("dimension,degree,length", [(2, 2, 258), (3, 3, 258), (4, 3, 66), (8, 3, 18),
    (16, 2, 66), (5, 3, 18), (2, 4, 514), (3, 5, 258), (4, 5, 66), (4, 6, 18),
    (8, 4, 18), (8, 5, 10), (8, 6, 6)])
def test_low_depth_kernels(monkeypatch, device, method, dtype, dimension, degree, length):
    path = torch.tensor(np.random.default_rng(31).normal(size=(2, 1, length, dimension)).cumsum(axis=-2) / length ** 0.5,
                        dtype=dtype, device=device)
    options = dict(method=method, scalar_term=True, n_jobs=2)
    monkeypatch.setenv("PYSIGLIB_PS_LOW_DEPTH", "0")
    general = pysiglib.sig(path, degree, **options)
    monkeypatch.setenv("PYSIGLIB_PS_LOW_DEPTH", "1")
    actual = pysiglib.sig(path, degree, **options)
    expected = pysiglib.sig(path, degree, scalar_term=True, n_jobs=2)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, general, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dimension,degree", [(2, 2), (3, 3), (4, 2), (8, 3)])
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cpu_direct_preparation(monkeypatch, dimension, degree, dtype):
    path = np.random.default_rng(44).normal(size=(3, 129, dimension)).cumsum(axis=1).astype(dtype) * 0.02
    expected = pysiglib.sig(path, degree, scalar_term=True)
    for strategy in (1, 2):
        monkeypatch.setenv("PYSIGLIB_PS_CPU_STRATEGY", str(strategy))
        actual = pysiglib.sig(path, degree, method=METHODS[0], scalar_term=True, n_jobs=4)
        tolerance = 1e-4 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dimension,degree", [(2, 4), (3, 5), (4, 6), (8, 4)])
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("block_bytes", (16 << 10, 64 << 10, 256 << 10))
def test_cpu_streamed_preparation(monkeypatch, dimension, degree, dtype, block_bytes):
    path = np.random.default_rng(45).normal(size=(2, 130, dimension)).cumsum(axis=1).astype(dtype) * 0.02
    expected = pysiglib.sig(path, degree, scalar_term=True)
    monkeypatch.setenv("PYSIGLIB_PS_CPU_STRATEGY", "3")
    monkeypatch.setenv("PYSIGLIB_PS_CPU_STREAM_BLOCK_BYTES", str(block_bytes))
    for paired in (0, 1):
        monkeypatch.setenv("PYSIGLIB_PS_PAIR_LEVELS", str(paired))
        actual = pysiglib.sig(path, degree, method=METHODS[0], scalar_term=True, n_jobs=4)
        tolerance = 1e-4 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_cpu_preparation_strategy_validation(monkeypatch):
    path = np.zeros((9, 3), dtype=np.float64)
    for strategy in (2, 3, 4):
        monkeypatch.setenv("PYSIGLIB_PS_CPU_STRATEGY", str(strategy))
        with pytest.raises(Exception, match="CPU_STRATEGY|preparation"):
            pysiglib.sig(path, 8, method=METHODS[0])


@pytest.mark.parametrize("batch,dimension,degree,dtype,strategy", [
    (2, 3, 3, np.float32, 2),
    (2, 2, 4, np.float32, 1),
    (2, 2, 6, np.float32, 3),
    (2, 3, 4, np.float32, 1),
    (32, 3, 4, np.float32, 3),
    (2, 4, 4, np.float32, 1),
    (2, 4, 4, np.float64, 3),
    (2, 8, 4, np.float32, 3),
    (2, 8, 4, np.float64, 1),
])
def test_cpu_automatic_preparation(monkeypatch, batch, dimension, degree, dtype, strategy):
    path = np.random.default_rng(47).normal(size=(batch, 17, dimension)).astype(dtype)
    actual = pysiglib.sig(path, degree, method=METHODS[0], n_jobs=4)
    expected = pysiglib.sig(path, degree, n_jobs=4)
    tolerance = 1e-4 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    monkeypatch.setenv("PYSIGLIB_PS_CPU_STRATEGY", str(strategy))
    forced = pysiglib.sig(path, degree, method=METHODS[0], n_jobs=4)
    np.testing.assert_array_equal(actual, forced)


def test_cpu_streamed_one_segment_fallback(monkeypatch):
    path = np.random.default_rng(48).normal(size=(2, 17, 8))
    expected = pysiglib.sig(path, 6)
    monkeypatch.setenv("PYSIGLIB_PS_CPU_STRATEGY", "3")
    monkeypatch.setenv("PYSIGLIB_PS_CPU_STREAM_BLOCK_BYTES", "1")
    actual = pysiglib.sig(path, 6, method=METHODS[0])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    monkeypatch.setenv("PYSIGLIB_PS_CPU_STRATEGY", "1")
    stored = pysiglib.sig(path, 6, method=METHODS[0])
    np.testing.assert_array_equal(actual, stored)


@pytest.mark.parametrize("method,device,degree,ta,ll,dtype",
    product(METHODS, DEVICES, (2, 3), (False, True), (False, True), (torch.float32, torch.float64)))
def test_scan_free_transforms(method, device, degree, ta, ll, dtype):
    increments = np.array([[1, 2], [-1, -2], [2, -1], [-2, 1], [3, 1]], dtype=float) / 16
    path = np.vstack((np.zeros(2), increments.cumsum(axis=0))) + 1024
    path = torch.tensor(path, device=device, dtype=dtype)
    options = dict(scalar_term=True, time_aug=ta, lead_lag=ll, end_time=0.75)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(pysiglib.sig(path, degree, method=method, **options),
                               pysiglib.sig(path, degree, **options), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device,method,dtype", product(DEVICES, METHODS, (torch.float32, torch.float64)))
def test_low_depth_rational_cancellation(device, method, dtype):
    increments = np.tile([[3, -1], [-3, 1], [-2, -3], [2, 3]], (64, 1)) / 16
    increments = np.vstack((increments, [[1 / 1024, -1 / 2048]]))
    path = np.vstack((np.zeros(2), increments.cumsum(axis=0))) + 1024
    exact = exact_signature(increments, 6)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    for degree in (2, 3, 5, 6):
        actual = pysiglib.sig(torch.tensor(path, device=device, dtype=dtype), degree,
                              method=method, scalar_term=True).cpu().numpy()
        for level in range(degree + 1):
            start, end = 2 ** level - 1, 2 ** (level + 1) - 1
            delta = np.abs(actual[start:end].astype(float) - exact[start:end])
            relative = delta.max() / max(np.abs(exact[start:end]).max(), np.finfo(float).tiny)
            assert delta.max() <= tolerance, (degree, level, delta.max(), relative)


@pytest.mark.parametrize("device,method,dtype", product(DEVICES, METHODS, (torch.float32, torch.float64)))
def test_specialized_factor_blocks(monkeypatch, device, method, dtype):
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", "500")
    path = torch.tensor(np.random.default_rng(32).normal(size=(3, 11, 4)) * 0.1, dtype=dtype, device=device)
    actual = pysiglib.sig(path, 6, method=method, scalar_term=True, n_jobs=2)
    expected = pysiglib.sig(path, 6, scalar_term=True)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("method,device", product(METHODS, DEVICES))
def test_exact_reference_and_forced_blocks(monkeypatch, method, device):
    monkeypatch.setenv("PYSIGLIB_PS_FUSION", "0")
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_CHUNK", "3")
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", "1400")
    increments = [[Fraction(x, 16) for x in row] for row in [[1, 2, -1], [0, 0, 0], [-2, 1, 1], [3, -1, 2], [-1, -2, 1], [2, 0, -1], [1, 1, 1]]]
    path = np.vstack((np.zeros(3), np.cumsum(np.array(increments, dtype=float), axis=0)))
    actual = pysiglib.sig(torch.tensor(path, device=device), 5, method=method, scalar_term=True)
    np.testing.assert_allclose(actual.cpu(), exact_signature(increments, 5), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("method,device,ta,ll", product(METHODS, DEVICES, (False, True), (False, True)))
def test_transforms_and_views(method, device, ta, ll):
    path = torch.tensor(np.random.default_rng(7).normal(size=(3, 34, 4)) * 0.05, device=device)[:, ::2, ::2]
    opts = dict(time_aug=ta, lead_lag=ll, end_time=0.7, scalar_term=True)
    torch.testing.assert_close(pysiglib.sig(path, 4, method=method, **opts), pysiglib.sig(path, 4, **opts), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_numpy_fixtures_and_empty(method):
    fixtures = load_fixtures("reference_data.npz")
    for degree in range(2, 6):
        actual = pysiglib.sig(fixtures["path"], degree, method=method)
        np.testing.assert_allclose(actual, fixtures[f"sig__d{degree}"], atol=1e-10, rtol=1e-10)
    for shape in ((0, 5, 2), (2, 0, 5, 2)):
        path = np.empty(shape)
        assert pysiglib.sig(path, 4, method=method).shape == shape[:-2] + (30,)
    path = np.arange(24, dtype=float).reshape(8, 3)[::-1]
    np.testing.assert_allclose(pysiglib.sig(path, 3, method=method), pysiglib.sig(path, 3), atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ("zero", "constant", "alternating", "closed"))
def test_cancellation(method, kind):
    increments = np.random.default_rng(8).normal(size=(12, 2)) * 0.2
    if kind == "zero": increments[:] = 0
    if kind == "constant": increments[:] = increments[0]
    if kind == "alternating": increments[1::2] = -increments[::2]
    if kind == "closed": increments[-1] = -increments[:-1].sum(axis=0)
    path = np.vstack((np.zeros(2), increments.cumsum(axis=0)))
    np.testing.assert_allclose(pysiglib.sig(path, 8, method=method), pysiglib.sig(path, 8), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("api_name", ("base", "torch", "jax"))
@pytest.mark.parametrize("method", (None, *METHODS))
def test_removed_precision_option(api_name, method):
    api = pysiglib if api_name == "base" else torch_api
    if api_name == "jax":
        pytest.importorskip("jax")
        import pysiglib.jax_api as api
    for name in ("sig", "signature"):
        with pytest.raises(TypeError, match="matmul_precision"):
            getattr(api, name)(np.zeros((8, 2)), 3, method=method, matmul_precision="tf32")


@pytest.mark.parametrize("api_name", ("base", "torch", "jax"))
def test_removed_strassen_method(api_name):
    api = pysiglib if api_name == "base" else torch_api
    if api_name == "jax":
        pytest.importorskip("jax")
        import pysiglib.jax_api as api
    for name in ("sig", "signature"):
        with pytest.raises(ValueError, match="method"):
            getattr(api, name)(np.zeros((8, 2)), 3, method="prefix_suffix_strassen")


def test_validation():
    path = np.zeros((8, 2))
    for options in (dict(method="bad"), dict(method=METHODS[0], horner=False),
                    dict(method=METHODS[0], correction=np.zeros(4))):
        with pytest.raises(ValueError): pysiglib.sig(path, 3, **options)
    tensor = torch.zeros((8, 2), requires_grad=True)
    with pytest.raises(NotImplementedError, match="torch_api"): pysiglib.sig(tensor, 3, method=METHODS[0])
    with torch.no_grad(): assert pysiglib.sig(tensor, 3, method=METHODS[0]).shape == (14,)
    assert torch_api.sig(tensor, 3, method=METHODS[0]).requires_grad
    assert pysiglib.sig(path, 3, method=METHODS[0], correction=np.empty(0)).shape == (14,)


@pytest.mark.parametrize("method,split,error", [
    ("ps", (0, 3), ValueError), ("ps", (4, -1), ValueError), ("ps", (1, 1), ValueError),
    ("ps", (1,), TypeError), ("ps", (True, 2), TypeError), ("ps", (1, 2.0), TypeError),
    (None, (2, 1), ValueError), ("auto", (2, 1), ValueError), ("prefix_suffix_gemm", None, ValueError),
])
def test_split_validation(method, split, error):
    path = np.zeros((0, 8, 2))
    signature = pysiglib.sig(path, 3)
    with pytest.raises(error):
        pysiglib.sig(path, 3, method=method, ps_split=split)
    with pytest.raises(error):
        pysiglib.sig_backprop(path, signature, signature, 3, method=method, ps_split=split)


@pytest.mark.parametrize("device", DEVICES)
def test_per_call_split_autograd(monkeypatch, device):
    x = torch.randn(2, 7, 3, device=device, dtype=torch.float64, requires_grad=True)
    expected = torch_api.sig(x, 4)
    grad = torch.autograd.grad(expected.sum(), x)[0]
    monkeypatch.setenv("PYSIGLIB_PS_SPLIT_A", "99")
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(lambda a: torch_api.sig(x, 4, method="ps", ps_split=(a, 4 - a)), (1, 2, 4)))
    for result in results:
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], grad)


@pytest.mark.parametrize("device", DEVICES)
def test_auto_different_forward_and_backward_methods(device, monkeypatch):
    # This shape selects Horner forward and prefix-suffix backward on both backends.
    path = torch.zeros((32, 17, 4), device=device)
    monkeypatch.setenv("PYSIGLIB_PS_SPLIT_A", "99")
    signature = pysiglib.sig(path, 8, method="auto")
    with pytest.raises(Exception, match="split exceeds degree"):
        pysiglib.sig_backprop(path, signature, signature, 8, method="auto")
    path.requires_grad_()
    with pytest.raises(Exception, match="split exceeds degree"):
        torch_api.sig(path, 8, method="auto").sum().backward()


def test_auto_horner_fallback(monkeypatch):
    monkeypatch.setenv("PYSIGLIB_PS_SPLIT_A", "99")
    path = np.zeros((17, 8), dtype=np.float32)
    for options in (dict(correction=np.ones(64, dtype=np.float32) * 0.01), dict(lead_lag=True), dict(time_aug=True)):
        expected = pysiglib.sig(path, 4, **options)
        seeds = np.ones_like(expected)
        np.testing.assert_array_equal(pysiglib.sig(path, 4, method="auto", **options), expected)
        np.testing.assert_array_equal(pysiglib.sig_backprop(path, expected, seeds, 4, method="auto", **options),
                                      pysiglib.sig_backprop(path, expected, seeds, 4, **options))


def test_jax_auto_mixed_passes():
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as api
    path = np.random.default_rng(45).normal(scale=0.02, size=(2, 21, 4)).astype(np.float32)
    reference_path = path.astype(np.float64)
    signature = pysiglib.sig(reference_path, 6)
    seeds = np.random.default_rng(46).normal(size=signature.shape).astype(np.float32)
    expected = pysiglib.sig_backprop(reference_path, signature, seeds.astype(np.float64), 6)
    run = lambda x: (api.sig(x, 6, method="auto") * seeds).sum()
    np.testing.assert_allclose(jax.jit(jax.grad(run))(path), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("method", METHODS)
def test_torch_forward_ad_rejected(method):
    path = torch.zeros((8, 2))
    for api in (pysiglib, torch_api):
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(path, torch.ones_like(path))
            with pytest.raises(NotImplementedError, match="differentiation"):
                api.sig(dual, 3, method=method)
        with pytest.raises(NotImplementedError, match="differentiation"):
            torch.func.jvp(lambda p: api.sig(p, 3, method=method), (path,), (torch.ones_like(path),))


@pytest.mark.parametrize("method", METHODS)
def test_threaded_calls(method):
    path = np.random.default_rng(9).normal(size=(5, 17, 3)) * 0.05
    expected = pysiglib.sig(path, 5)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(lambda _: pysiglib.sig(path, 5, method=method, n_jobs=2), range(6)))
    for result in results: np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA unavailable")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("degree", (2, 3, 6))
def test_cuda_stream(method, degree):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        path = torch.randn((3, 65, 3), device="cuda") * 0.02
        strict = pysiglib.sig(path, degree, method=method)
    stream.synchronize()
    torch.testing.assert_close(strict, pysiglib.sig(path, degree), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA unavailable")
@pytest.mark.parametrize("degree", (2, 3))
def test_scan_free_batch_boundary(degree):
    path = torch.tensor(np.random.default_rng(33).normal(size=(257, 18, 3)) * 0.1, device="cuda")
    torch.testing.assert_close(pysiglib.sig(path, degree, method=METHODS[0], scalar_term=True),
                               pysiglib.sig(path, degree, scalar_term=True), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method,split", [("ps", None), ("ps", (1, 3)), ("ps", (4, 0)), ("auto", None)])
def test_jax_forward_and_reverse(monkeypatch, method, split):
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as api
    jax.config.update("jax_enable_x64", True)
    if split is not None:
        monkeypatch.setenv("PYSIGLIB_PS_SPLIT_A", "99")
    path = np.random.default_rng(10).normal(size=(2, 13, 3)) * 0.05
    run = lambda p: api.sig(p, 4, method=method, ps_split=split)
    expected = pysiglib.sig(path, 4)
    for value in (run(path), jax.jit(run)(path), jax.vmap(run)(path)):
        np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-10)
    expected_grad = pysiglib.sig_backprop(path, expected, np.ones_like(expected), 4)
    np.testing.assert_allclose(jax.grad(lambda p: run(p).sum())(path), expected_grad, atol=1e-10, rtol=1e-10)
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(run, (path,), (np.ones_like(path),))


@pytest.mark.parametrize("method,device", product(METHODS, DEVICES))
def test_workspace_limit_and_batch_chunks(monkeypatch, method, device):
    path = torch.tensor(np.random.default_rng(12).normal(size=(35, 9, 3)) * 0.03, device=device)
    expected = pysiglib.sig(path, 5)
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "65536")
    torch.testing.assert_close(pysiglib.sig(path, 5, method=method, n_jobs=4), expected, atol=1e-10, rtol=1e-10)
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    with pytest.raises(Exception, match="requires.*workspace bytes"):
        pysiglib.sig(path, 5, method=method)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA unavailable")
def test_cuda_concurrent_calls_and_cleanup():
    path = torch.randn((2, 17, 3), device="cuda", dtype=torch.float64) * 0.03
    expected = pysiglib.sig(path, 5)
    torch.cuda.synchronize()
    def run(_):
        with torch.cuda.stream(torch.cuda.Stream()):
            return pysiglib.sig(path, 5, method=METHODS[0])
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, range(6)))
    for result in results: torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
    from pysiglib.load_siglib import CUSIG
    CUSIG.cusig_shutdown()
    torch.testing.assert_close(pysiglib.sig(path, 5, method=METHODS[0]), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_device_changes():
    path = np.random.default_rng(15).normal(size=(13, 3)) * 0.03
    expected = pysiglib.sig(path, 5)
    for device in (0, 1, 0):
        actual = pysiglib.sig(torch.tensor(path, device=f"cuda:{device}"), 5, method=METHODS[0])
        np.testing.assert_allclose(actual.cpu(), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA unavailable")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("coefficients,paired", [(0, 0), (1, 1)])
def test_cuda_global_factor_workspace(monkeypatch, method, coefficients, paired):
    monkeypatch.setenv("PYSIGLIB_PS_COEFFICIENT_FACTORS", str(coefficients))
    monkeypatch.setenv("PYSIGLIB_PS_PAIR_LEVELS", str(paired))
    path = torch.tensor(np.random.default_rng(13).normal(size=(2, 3, 3)) * 0.1, device="cuda")
    torch.testing.assert_close(pysiglib.sig(path, 14, method=method), pysiglib.sig(path, 14), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_cancellation_higher_precision(method):
    increments = np.random.default_rng(14).integers(-4, 5, size=(12, 3)).astype(np.longdouble) / 16
    increments[1::2] = -increments[::2] + np.longdouble(1) / 4096
    path = np.vstack((np.zeros(3), increments.cumsum(axis=0))).astype(np.float64)
    expected = exact_signature([[Fraction(float(x)) for x in row] for row in increments], 8)[1:]
    actual = pysiglib.sig(path, 8, method=method)
    delta = np.abs(actual.astype(np.longdouble) - expected)
    offset = 0
    for k in range(1, 9):
        size = 3 ** k
        absolute = float(delta[offset:offset + size].max())
        relative = absolute / max(float(np.abs(expected[offset:offset + size]).max()), np.finfo(float).tiny)
        assert absolute < 1e-10, (k, absolute, relative)
        offset += size
