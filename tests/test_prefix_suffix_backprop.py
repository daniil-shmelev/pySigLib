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

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api
from conftest import DEVICES


METHOD = "prefix_suffix_gemm"


def chen_reference(path, degree, time_aug=False, lead_lag=False, end_time=1.):
    if lead_lag:
        repeated = path.repeat_interleave(2, dim=0)
        path = torch.cat((repeated[:-1], repeated[1:]), dim=-1)
    if time_aug:
        time = torch.linspace(0, end_time, path.shape[0], dtype=path.dtype, device=path.device)
        path = torch.cat((path, time[:, None]), dim=-1)
    d = path.shape[-1]
    state = [path.new_ones(1)] + [path.new_zeros(d ** k) for k in range(1, degree + 1)]
    for increment in path[1:] - path[:-1]:
        exponential = [path.new_ones(1)]
        for k in range(1, degree + 1):
            exponential.append(torch.outer(exponential[-1], increment).flatten() / k)
        state = [sum(torch.outer(state[j], exponential[k - j]).flatten() for j in range(k + 1))
                 for k in range(degree + 1)]
    return torch.cat(state)


@pytest.mark.parametrize("device,dtype,scalar", product(DEVICES, (torch.float32, torch.float64), (False, True)))
@pytest.mark.parametrize("dimension,degree", [(1, 12), (2, 2), (3, 3), (4, 4), (3, 5), (2, 8), (8, 6)])
def test_matches_horner(device, dtype, scalar, dimension, degree):
    rng = np.random.default_rng(701)
    path = torch.tensor(rng.normal(size=(2, 3, 11, dimension)).cumsum(axis=-2) * 0.03, device=device, dtype=dtype)
    sig = pysiglib.sig(path, degree, scalar_term=scalar)
    deriv = torch.tensor(rng.normal(size=sig.shape), device=device, dtype=dtype)
    expected = pysiglib.sig_backprop(path, sig, deriv, degree, n_jobs=3)
    actual = pysiglib.sig_backprop(path, sig, deriv, degree, n_jobs=3, method=METHOD)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("time_aug,lead_lag,scalar", product((False, True), repeat=3))
def test_chen_and_torch_autograd(device, time_aug, lead_lag, scalar):
    rng = np.random.default_rng(702)
    path = torch.tensor(rng.normal(size=(7, 2)) * 0.2, dtype=torch.float64, device=device, requires_grad=True)
    options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=0.7)
    reference = chen_reference(path, 5, **options)[0 if scalar else 1:]
    deriv = torch.tensor(rng.normal(size=reference.shape), dtype=path.dtype, device=device)
    expected, = torch.autograd.grad(reference, path, deriv)
    result = torch_api.sig(path, 5, method=METHOD, scalar_term=scalar, n_jobs=2, **options)
    actual, = torch.autograd.grad(result, path, deriv)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length,degree", [(0, 4), (1, 4), (2, 0), (2, 1), (2, 5), (7, 0), (7, 1)])
def test_trivial_paths(device, length, degree):
    path = torch.zeros((2, length, 3), dtype=torch.float64, device=device)
    sig = pysiglib.sig(path, degree, scalar_term=True, lead_lag=True, time_aug=True, method=METHOD)
    expected = (torch.zeros_like(path) if length < 2 else
                pysiglib.sig_backprop(path, sig, torch.ones_like(sig), degree, lead_lag=True, time_aug=True))
    actual = pysiglib.sig_backprop(path, sig, torch.ones_like(sig), degree,
                                  lead_lag=True, time_aug=True, method=METHOD)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_shape", [(0,), (2, 0, 3)])
def test_empty_batch(device, batch_shape):
    path = torch.empty((*batch_shape, 9, 2), device=device)
    sig = pysiglib.sig(path, 4)
    actual = pysiglib.sig_backprop(path, sig, sig, 4, method=METHOD)
    assert actual.shape == path.shape


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,scan_degree", [(1, 3), (5, 4), (7, 6)])
def test_blocks_and_scan_overrides(monkeypatch, device, rows, scan_degree):
    rng = np.random.default_rng(703)
    path = torch.tensor(rng.normal(size=(3, 8, 3)) * 0.05, dtype=torch.float64, device=device)
    sig = pysiglib.sig(path, 6)
    deriv = torch.tensor(rng.normal(size=sig.shape), dtype=path.dtype, device=device)
    expected = pysiglib.sig_backprop(path, sig, deriv, 6)
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", str(rows * 2 * (27 + 27) * 8))
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_DEGREE", str(scan_degree))
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_CHUNK", "3")
    actual = pysiglib.sig_backprop(path, sig, deriv, 6, method=METHOD, n_jobs=3)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_and_workspace(monkeypatch, device):
    rng = np.random.default_rng(704)
    path = torch.tensor(rng.normal(size=(35, 3, 9)) * 0.03, dtype=torch.float64, device=device).transpose(-1, -2)
    sig = pysiglib.sig(path, 5)
    deriv = torch.ones((35, sig.shape[-1] * 2), dtype=path.dtype, device=device)[:, ::2]
    expected = pysiglib.sig_backprop(path, sig, deriv, 5)
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", "4096")
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", str((4 << 20) + 150000 if device == "cuda" else 150000))
    actual = pysiglib.sig_backprop(path, sig, deriv, 5, method=METHOD, n_jobs=-1)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    with pytest.raises(Exception, match="requires.*workspace bytes"):
        pysiglib.sig_backprop(path, sig, deriv, 5, method=METHOD)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("segments,chunk,scan_degree", [(1, 1, 3), (15, 16, 3), (16, 16, 3),
    (17, 16, 3), (33, 1, 3), (65, 3, 3), (129, 64, 3), (33, 3, 7)])
def test_cuda_adjoint_scan_chunks(monkeypatch, dtype, segments, chunk, scan_degree):
    degree = max(6, scan_degree)
    rng = np.random.default_rng(708)
    values = rng.normal(size=(2, segments + 1, 3)).cumsum(axis=1) * (0.1 / np.sqrt(segments))
    path = torch.tensor(values, dtype=dtype, device="cuda")
    reference_path = path.cpu().double().numpy()
    sig = pysiglib.sig(path, degree, scalar_term=True)
    deriv = torch.tensor(rng.normal(size=sig.shape), dtype=dtype, device="cuda")
    reference_sig = pysiglib.sig(reference_path, degree, scalar_term=True)
    expected = pysiglib.sig_backprop(reference_path, reference_sig, deriv.cpu().double().numpy(), degree)
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_CHUNK", str(chunk))
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_DEGREE", str(scan_degree))
    actual = pysiglib.sig_backprop(path, sig, deriv, degree, method=METHOD)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["zero", "repeated", "alternating", "closed", "translated", "cancelling"])
def test_stress(device, dtype, kind):
    rng = np.random.default_rng(705)
    increments = rng.integers(-4, 5, (8, 2)).astype(float) / 8
    if kind == "zero": increments[:] = 0
    if kind == "repeated": increments[:] = increments[0]
    if kind in ("alternating", "cancelling"): increments[1::2] = -increments[::2]
    if kind == "cancelling": increments[1::2] += 1 / 4096
    if kind == "closed": increments[-1] = -increments[:-1].sum(axis=0)
    values = np.vstack((np.zeros(2), increments.cumsum(axis=0)))
    if kind == "translated": values += 128
    path = torch.tensor(values, device=device, dtype=torch.float64, requires_grad=True)
    reference = chen_reference(path, 6)
    deriv = torch.tensor(rng.normal(size=reference.shape), dtype=path.dtype, device=device)
    expected, = torch.autograd.grad(reference, path, deriv)
    actual = pysiglib.sig_backprop(path.to(dtype), reference.detach().to(dtype), deriv.to(dtype), 6, method=METHOD)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)


def test_validation_and_higher_derivatives():
    path = np.zeros((8, 2))
    sig = pysiglib.sig(path, 3)
    for options in (dict(method="bad"), dict(method=METHOD, correction=np.zeros(4)), dict(method=METHOD, n_jobs=0)):
        with pytest.raises(ValueError): pysiglib.sig_backprop(path, sig, sig, 3, **options)
    np.testing.assert_array_equal(pysiglib.sig_backprop(path, sig, sig, 3, correction=np.empty(0), method=METHOD), 0)
    tensor = torch.zeros((8, 2), dtype=torch.float64, requires_grad=True)
    value = torch_api.sig(tensor, 3, method=METHOD).sum()
    with pytest.raises(NotImplementedError, match="higher-order"):
        torch.autograd.grad(value, tensor, create_graph=True)


@pytest.mark.parametrize("device", DEVICES)
def test_concurrent_calls_and_cleanup(device):
    path = torch.tensor(np.random.default_rng(706).normal(size=(3, 65, 3)) * 0.05, device=device, dtype=torch.float64)
    sig = pysiglib.sig(path, 5)
    deriv = torch.ones_like(sig)
    expected = pysiglib.sig_backprop(path, sig, deriv, 5)
    if device == "cuda": torch.cuda.synchronize()
    def run(_):
        if device == "cuda":
            with torch.cuda.stream(torch.cuda.Stream()):
                return pysiglib.sig_backprop(path, sig, deriv, 5, method=METHOD)
        return pysiglib.sig_backprop(path, sig, deriv, 5, method=METHOD, n_jobs=2)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, range(6)))
    for result in results: torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
    if device == "cuda":
        from pysiglib.load_siglib import CUSIG
        CUSIG.cusig_shutdown()
        torch.testing.assert_close(run(0), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("time_aug,lead_lag,scalar", product((False, True), repeat=3))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_grad_jit_vmap(time_aug, lead_lag, scalar, dtype):
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as api
    jax.config.update("jax_enable_x64", True)
    path = (np.random.default_rng(707).normal(size=(2, 7, 2)) * 0.1).astype(dtype)
    options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=0.7, scalar_term=scalar)
    sig = pysiglib.sig(path, 4, **options)
    back_options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=0.7)
    expected = pysiglib.sig_backprop(path, sig, np.ones_like(sig), 4, **back_options)
    grad = jax.grad(lambda p: api.sig(p, 4, method=METHOD, **options).sum())
    tolerance = 1e-4 if dtype == np.float32 else 1e-10
    for actual in (grad(path), jax.jit(grad)(path), jax.vmap(grad)(path), jax.jit(jax.vmap(grad))(path)):
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_jax_empty_batch_and_higher_derivatives():
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as api
    grad = jax.grad(lambda p: api.sig(p, 4, method=METHOD).sum())
    path = np.zeros((2, 0, 7, 3), dtype=np.float32)
    for actual in (grad(path), jax.jit(grad)(path), jax.vmap(grad)(path)):
        assert actual.shape == path.shape
    with pytest.raises((ValueError, TypeError), match="differentiat|JVP"):
        jax.grad(lambda p: grad(p).sum())(np.zeros((7, 3), dtype=np.float32))
