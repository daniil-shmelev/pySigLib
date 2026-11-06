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
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api
from conftest import DEVICES, check_sig_methods, force_highway_target, highway_targets, load_fixtures

METHOD = "ps"


@pytest.mark.parametrize("device,dtype", list(product(DEVICES, (torch.float32, torch.float64))))
@pytest.mark.parametrize("time_aug,lead_lag,scalar", list(product((False, True), repeat=3)))
def test_split_and_endpoint_variants(monkeypatch, device, dtype, time_aug, lead_lag, scalar):
    path = torch.tensor(np.random.default_rng(712).normal(scale=0.04, size=(2, 14, 2)),
                        device=device, dtype=dtype)[:, ::2]
    options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=.7, scalar_term=scalar, n_jobs=2)
    variants = [("horner", None), ("ps", None)] + [("ps", a) for a in range(1, 5)]
    check_sig_methods(path, 4, variants, **options)
    for block in (1, 3, 16):
        monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", str(block))
        check_sig_methods(path, 4, [("ps", None)], backward=False, **options)


@pytest.mark.parametrize("device,dtype", list(product(DEVICES, (torch.float32, torch.float64))))
@pytest.mark.parametrize("dimension", (8, 16))
def test_automatic_prefix_suffix_dispatch(device, dtype, dimension):
    path = torch.tensor(np.random.default_rng(714).normal(scale=0.02, size=(32, 257, dimension)),
                        device=device, dtype=dtype)
    check_sig_methods(path, 3, [("ps", None), ("ps", 2)], n_jobs=20)


@pytest.mark.parametrize("device", DEVICES)
def test_endpoint_workspace_limit(monkeypatch, device):
    monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "3")
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    path = torch.ones(2, 8, 3, device=device)
    with pytest.raises(Exception, match="allocate|workspace"):
        pysiglib.sig(path, 4, method="ps")


@pytest.mark.parametrize("target_name", ("SSE2", "AVX2"))
def test_endpoint_simd_targets(monkeypatch, target_name):
    targets = highway_targets()
    if target_name not in targets:
        pytest.skip(f"{target_name} is unavailable")
    with force_highway_target(targets[target_name]):
        path = np.random.default_rng(715).normal(scale=0.03, size=(7, 18, 3))
        expected = pysiglib.sig(path, 5)
        monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "4")
        actual = pysiglib.sig(path, 5, method="ps", n_jobs=3)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("device,dtype", list(product(DEVICES, (torch.float32, torch.float64))))
@pytest.mark.parametrize("dimension,degree,rows", [(2, 8, 1), (3, 8, 5), (5, 6, 7), (11, 4, 3), (4, 6, 1)])
def test_paired_factor_blocks(monkeypatch, device, dtype, dimension, degree, rows):
    path = torch.tensor(np.random.default_rng(84).normal(size=(3, 19, dimension)).cumsum(axis=1) * 0.03,
                        device=device, dtype=dtype)
    expected = pysiglib.sig(path, degree, scalar_term=True)
    itemsize = path.element_size()
    width = dimension ** ((degree + 1) // 2) + dimension ** (degree // 2)
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", str(rows * width * itemsize))
    actual = pysiglib.sig(path, degree, method=METHOD, scalar_term=True, n_jobs=3)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


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


@pytest.mark.parametrize("device", DEVICES)
def test_split_exact_reference(device):
    increments = [[Fraction(1, 8), Fraction(-1, 4)], [Fraction(1, 4), Fraction(1, 8)],
                  [Fraction(-3, 8), Fraction(1, 8)]]
    path = torch.tensor(np.vstack((np.zeros(2), np.asarray(increments, dtype=float).cumsum(axis=0))),
                        device=device, dtype=torch.float64)
    expected = exact_signature(increments, 6)
    for a in range(1, 7):
        actual = pysiglib.sig(path, 6, method=METHOD, scalar_term=True, ps_split=a)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device,dtype", list(product(DEVICES, (torch.float32, torch.float64))))
@pytest.mark.parametrize("dimension,degree,length", [(1, 7, 13), (2, 0, 7), (2, 1, 7), (2, 6, 0),
    (2, 5, 1), (3, 5, 2), (2, 5, 12), (3, 4, 14), (3, 6, 130), (4, 5, 259),
    (2, 2, 258), (3, 3, 258), (4, 3, 66), (8, 3, 18),
    (16, 2, 66), (5, 3, 18), (2, 4, 514), (3, 5, 258), (4, 5, 66), (4, 6, 18),
    (8, 4, 18), (8, 5, 10), (8, 6, 6)])
def test_matches_horner(device, dtype, dimension, degree, length):
    path = torch.tensor(np.random.default_rng(5).normal(size=(2, 3, length, dimension)).cumsum(axis=-2) * 0.03,
                        device=device, dtype=dtype)
    for scalar in (False, True):
        check_sig_methods(path, degree, [("ps", None)], scalar_term=scalar, backward=False, n_jobs=2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch,d,m,block_bytes", [
    (2, 2, 4, None), (2, 2, 6, 16384), (2, 3, 4, None), (32, 3, 4, None),
    (2, 3, 5, 65536), (2, 4, 4, None), (2, 4, 6, 262144), (2, 8, 4, None), (2, 8, 6, 1),
])
def test_cpu_preparation(monkeypatch, dtype, batch, d, m, block_bytes):
    if block_bytes is not None:
        monkeypatch.setenv("PYSIGLIB_PS_CPU_STREAM_BLOCK_BYTES", str(block_bytes))
    path = np.random.default_rng(45).normal(scale=.02, size=(batch, 130, d)).cumsum(axis=1).astype(dtype)
    expected = pysiglib.sig(path, m, scalar_term=True)
    actual = pysiglib.sig(path, m, method=METHOD, scalar_term=True, n_jobs=4)
    tol = 1e-4 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("device,degree,ta,ll,dtype",
    list(product(DEVICES, (2, 3), (False, True), (False, True), (torch.float32, torch.float64))))
def test_scan_free_transforms(device, degree, ta, ll, dtype):
    increments = np.array([[1, 2], [-1, -2], [2, -1], [-2, 1], [3, 1]], dtype=float) / 16
    path = np.vstack((np.zeros(2), increments.cumsum(axis=0))) + 1024
    path = torch.tensor(path, device=device, dtype=dtype)
    options = dict(scalar_term=True, time_aug=ta, lead_lag=ll, end_time=0.75)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(pysiglib.sig(path, degree, method=METHOD, **options),
                               pysiglib.sig(path, degree, **options), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device,dtype", list(product(DEVICES, (torch.float32, torch.float64))))
def test_low_depth_rational_cancellation(device, dtype):
    increments = np.tile([[3, -1], [-3, 1], [-2, -3], [2, 3]], (64, 1)) / 16
    increments = np.vstack((increments, [[1 / 1024, -1 / 2048]]))
    path = np.vstack((np.zeros(2), increments.cumsum(axis=0))) + 1024
    exact = exact_signature(increments, 6)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    for degree in (2, 3, 5, 6):
        actual = pysiglib.sig(torch.tensor(path, device=device, dtype=dtype), degree,
                              method=METHOD, scalar_term=True).cpu().numpy()
        for level in range(degree + 1):
            start, end = 2 ** level - 1, 2 ** (level + 1) - 1
            delta = np.abs(actual[start:end].astype(float) - exact[start:end])
            relative = delta.max() / max(np.abs(exact[start:end]).max(), np.finfo(float).tiny)
            assert delta.max() <= tolerance, (degree, level, delta.max(), relative)


@pytest.mark.parametrize("device", DEVICES)
def test_exact_reference_and_forced_blocks(monkeypatch, device):
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_CHUNK", "3")
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", "1400")
    increments = [[Fraction(x, 16) for x in row] for row in [[1, 2, -1], [0, 0, 0], [-2, 1, 1], [3, -1, 2], [-1, -2, 1], [2, 0, -1], [1, 1, 1]]]
    path = np.vstack((np.zeros(3), np.cumsum(np.array(increments, dtype=float), axis=0)))
    actual = pysiglib.sig(torch.tensor(path, device=device), 5, method=METHOD, scalar_term=True)
    np.testing.assert_allclose(actual.cpu(), exact_signature(increments, 5), rtol=1e-11, atol=1e-12)


def test_numpy_fixtures_and_empty():
    fixtures = load_fixtures("reference_data.npz")
    for degree in range(2, 6):
        actual = pysiglib.sig(fixtures["path"], degree, method=METHOD)
        np.testing.assert_allclose(actual, fixtures[f"sig__d{degree}"], atol=1e-10, rtol=1e-10)
    for shape in ((0, 5, 2), (2, 0, 5, 2)):
        path = np.empty(shape)
        assert pysiglib.sig(path, 4, method=METHOD).shape == shape[:-2] + (30,)
    path = np.arange(24, dtype=float).reshape(8, 3)[::-1]
    np.testing.assert_allclose(pysiglib.sig(path, 3, method=METHOD), pysiglib.sig(path, 3), atol=1e-10)


def test_validation():
    path = np.zeros((8, 2))
    for options in (dict(method="bad"), dict(method=None),
                    dict(method=METHOD, correction=np.zeros(5))):
        with pytest.raises(ValueError): pysiglib.sig(path, 3, **options)
    tensor = torch.zeros((8, 2), requires_grad=True)
    with pytest.raises(NotImplementedError, match="torch_api"): pysiglib.sig(tensor, 3, method=METHOD)
    with torch.no_grad(): assert pysiglib.sig(tensor, 3, method=METHOD).shape == (14,)
    assert torch_api.sig(tensor, 3, method=METHOD).requires_grad
    assert pysiglib.sig(path, 3, method=METHOD, correction=np.empty(0)).shape == (14,)


@pytest.mark.parametrize("method,split,error", [
    ("ps", 0, ValueError), ("ps", 4, ValueError), ("ps", -1, ValueError),
    ("ps", (1, 2), TypeError), ("ps", True, TypeError), ("ps", 1.5, TypeError),
    (None, None, ValueError), ("horner", 2, ValueError), ("auto", 2, ValueError), ("prefix_suffix_gemm", None, ValueError),
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
    monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "invalid")
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(lambda a: torch_api.sig(x, 4, method="ps", ps_split=a), (1, 2, 4)))
    for result in results:
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], grad)


@pytest.mark.parametrize("device", DEVICES)
def test_auto_different_forward_and_backward_methods(device, monkeypatch):
    # This shape selects Horner forward and prefix-suffix backward on both backends.
    path = torch.zeros((32, 17, 4), device=device)
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    signature = pysiglib.sig(path, 8, method="auto")
    with pytest.raises(Exception, match="workspace"):
        pysiglib.sig_backprop(path, signature, signature, 8, method="auto")
    path.requires_grad_()
    with pytest.raises(Exception, match="workspace"):
        torch_api.sig(path, 8, method="auto").sum().backward()


def test_auto_horner_fallback(monkeypatch):
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    path = np.zeros((17, 8), dtype=np.float32)
    for options in (dict(correction=np.ones(64, dtype=np.float32) * 0.01), dict(lead_lag=True), dict(time_aug=True)):
        expected = pysiglib.sig(path, 4, **options)
        seeds = np.ones_like(expected)
        np.testing.assert_array_equal(pysiglib.sig(path, 4, method="auto", **options), expected)
        np.testing.assert_array_equal(pysiglib.sig_backprop(path, expected, seeds, 4, method="auto", **options),
                                      pysiglib.sig_backprop(path, expected, seeds, 4, **options))


@pytest.mark.parametrize("method", ["horner", "ps", "auto"])
def test_torch_forward_ad_rejected(method):
    path = torch.zeros((8, 2))
    for api in (pysiglib, torch_api):
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(path, torch.ones_like(path))
            with pytest.raises(NotImplementedError, match="forward-mode"):
                api.sig(dual, 3, method=method)
        with pytest.raises(NotImplementedError, match="forward-mode"):
            torch.func.jvp(lambda p: api.sig(p, 3, method=method), (path,), (torch.ones_like(path),))


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA unavailable")
@pytest.mark.parametrize("degree", (2, 3, 6))
def test_cuda_stream(degree):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        path = torch.randn((3, 65, 3), device="cuda") * 0.02
        strict = pysiglib.sig(path, degree, method=METHOD)
    stream.synchronize()
    torch.testing.assert_close(strict, pysiglib.sig(path, degree), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA unavailable")
@pytest.mark.parametrize("degree", (2, 3))
def test_scan_free_batch_boundary(degree):
    path = torch.tensor(np.random.default_rng(33).normal(size=(257, 18, 3)) * 0.1, device="cuda")
    torch.testing.assert_close(pysiglib.sig(path, degree, method=METHOD, scalar_term=True),
                               pysiglib.sig(path, degree, scalar_term=True), atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_device_changes():
    path = np.random.default_rng(15).normal(size=(13, 3)) * 0.03
    expected = pysiglib.sig(path, 5)
    for device in (0, 1, 0):
        actual = pysiglib.sig(torch.tensor(path, device=f"cuda:{device}"), 5, method=METHOD)
        np.testing.assert_allclose(actual.cpu(), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA unavailable")
@pytest.mark.parametrize("coefficients", (0, 1))
@pytest.mark.parametrize("degree,points,split", [(5, 8, 5), (14, 3, None)])
def test_cuda_global_factor_workspace(monkeypatch, coefficients, degree, points, split):
    monkeypatch.setenv("PYSIGLIB_PS_COEFFICIENT_FACTORS", str(coefficients))
    path = torch.tensor(np.random.default_rng(13).normal(size=(2, points, 3)) * 0.1, device="cuda")
    check_sig_methods(path, degree, [("ps", split)], backward=False)


def test_cancellation_higher_precision():
    increments = np.random.default_rng(14).integers(-4, 5, size=(12, 3)).astype(np.longdouble) / 16
    increments[1::2] = -increments[::2] + np.longdouble(1) / 4096
    path = np.vstack((np.zeros(3), increments.cumsum(axis=0))).astype(np.float64)
    expected = exact_signature([[Fraction(float(x)) for x in row] for row in increments], 8)[1:]
    actual = pysiglib.sig(path, 8, method=METHOD)
    delta = np.abs(actual.astype(np.longdouble) - expected)
    offset = 0
    for k in range(1, 9):
        size = 3 ** k
        absolute = float(delta[offset:offset + size].max())
        relative = absolute / max(float(np.abs(expected[offset:offset + size]).max()), np.finfo(float).tiny)
        assert absolute < 1e-10, (k, absolute, relative)
        offset += size
