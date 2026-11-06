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
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api
from conftest import DEVICES, check_sig_methods


METHOD = "ps"


@pytest.mark.parametrize("device,dtype", list(product(DEVICES, (torch.float32, torch.float64))))
@pytest.mark.parametrize("dimension,degree,correction_degree", [(1, 6, 2), (2, 2, 2), (2, 5, 2), (2, 6, 3), (3, 4, 4)])
@pytest.mark.parametrize("time_aug,scalar", list(product((False, True), repeat=2)))
def test_corrected_splits(device, dtype, dimension, degree, correction_degree, time_aug, scalar):
    rng = np.random.default_rng(724)
    path = torch.tensor(rng.normal(scale=0.06, size=(2, 2, 11, dimension)), device=device, dtype=dtype)[..., ::2, :]
    width = sum(dimension ** k for k in range(2, correction_degree + 1))
    correction = torch.tensor(rng.normal(scale=0.01, size=(2, 2, 5, width)), device=device, dtype=dtype)
    variants = [("ps", a) for a in range(1, degree + 1)] + [("ps", None), ("auto", None)]
    check_sig_methods(path, degree, variants, scalar_term=scalar, correction=correction, time_aug=time_aug, n_jobs=2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("points", (0, 1, 2, 8))
def test_corrected_layouts_and_empty_paths(monkeypatch, device, points):
    rng = np.random.default_rng(725)
    path = torch.tensor(rng.normal(scale=0.04, size=(2, points, 2)), device=device, dtype=torch.float64)
    value = torch.tensor([0.01, -0.03, 0.02, 0.04], device=device, dtype=torch.float64)
    n = max(0, points - 1)
    corrections = (value, value.expand(n, 4), value.expand(2, n, 4))
    monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "1")
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", "1")
    for correction in corrections:
        check_sig_methods(path, 4, [("ps", s) for s in (None, 1, 4)],
                          correction=correction, scalar_term=True, tol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("corrected", [False, True], ids=["endpoint", "corrected"])
def test_autograd_and_stream(monkeypatch, device, corrected):
    path = (torch.randn(2, 5, 2, device=device, dtype=torch.float64) * 0.03).requires_grad_()
    correction = torch.tensor([0.02, -0.03, 0.04, -0.01], device=device, dtype=torch.float64, requires_grad=True)
    options = dict(correction=correction, ps_split=1) if corrected else {}
    if not corrected:
        monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "3")
    operation = lambda x: torch_api.sig(x, 4, method="ps", **options)
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            assert torch.autograd.gradcheck(operation, (path,))
        stream.synchronize()
    else:
        assert torch.autograd.gradcheck(operation, (path,))
    if corrected:
        grads = torch.autograd.grad(operation(path).sum(), (path, correction), allow_unused=True)
        assert grads[1] is None


def test_corrected_exact_chen():
    d, degree = 2, 4
    path = np.array([[0, 0], [1, -2], [-1, 1]], dtype=np.float64) / 8
    correction = np.array([[(-1) ** i * (i % 3) for i in range(28)]] * 2, dtype=np.float64) / 32

    def multiply(x, y):
        return [sum((np.kron(x[j], y[k - j]) for j in range(k + 1)), np.zeros(d ** k, dtype=object))
                for k in range(degree + 1)]

    identity = [np.array([Fraction(1)])] + [np.full(d ** k, Fraction(0), dtype=object) for k in range(1, degree + 1)]
    signature = identity
    for segment, increment in enumerate(np.diff(path, axis=0)):
        generator = [np.array([Fraction(0)]), np.array([Fraction(x) for x in increment])]
        offset = 0
        for k in range(2, degree + 1):
            generator.append(np.array([Fraction(x) for x in correction[segment, offset:offset + d ** k]]))
            offset += d ** k
        exponential = [x.copy() for x in identity]
        power = identity
        for p in range(1, degree + 1):
            power = [x / p for x in multiply(power, generator)]
            exponential = [x + y for x, y in zip(exponential, power)]
        signature = multiply(signature, exponential)
    expected = np.concatenate(signature).astype(np.float64)
    for a in range(1, degree + 1):
        actual = pysiglib.sig(path, degree, correction=correction, scalar_term=True, method="ps", ps_split=a)
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)


def test_corrected_auto_cpu_region(monkeypatch):
    rng = np.random.default_rng(729)
    path = torch.tensor(rng.normal(scale=.01, size=(1, 17, 4)), dtype=torch.float32)
    correction = torch.tensor(rng.normal(scale=.001, size=16), dtype=path.dtype)
    check_sig_methods(path, 9, [("auto", None)], correction=correction, n_jobs=2)
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    with pytest.raises(Exception, match="workspace"):
        pysiglib.sig(path, 9, correction=correction, method="auto")


@pytest.mark.parametrize("device,dtype", list(product(DEVICES, (torch.float32, torch.float64))))
def test_corrected_scalar_high_degree(device, dtype):
    degree = 20
    path = torch.arange(5, device=device, dtype=dtype).reshape(-1, 1) / 4
    correction = torch.tensor([0.125], device=device, dtype=dtype)
    coefficients = [Fraction(1), Fraction(1)]
    for k in range(2, degree + 1):
        coefficients.append((coefficients[-1] + coefficients[-2]) / k)
    expected = torch.tensor([float(x) for x in coefficients], device=device, dtype=dtype)
    seeds = torch.tensor([(-1.) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    derivative = float(sum((-1) ** k * coefficients[k - 1] for k in range(1, degree + 1)))
    gradient = torch.zeros_like(path)
    gradient[0, 0] = -derivative; gradient[-1, 0] = derivative
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    for split in (None, 1, 20):
        actual = pysiglib.sig(path, degree, correction=correction, method="ps", ps_split=split, scalar_term=True)
        grad = pysiglib.sig_backprop(path, actual, seeds, degree, correction=correction, method="ps", ps_split=split)
        torch.testing.assert_close(actual, expected, atol=1e-15, rtol=tolerance)
        torch.testing.assert_close(grad, gradient, atol=tolerance, rtol=tolerance)


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


@pytest.mark.parametrize("device,dtype,scalar", list(product(DEVICES, (torch.float32, torch.float64), (False, True))))
@pytest.mark.parametrize("dimension,degree", [(1, 12), (2, 2), (3, 3), (4, 4), (3, 5), (2, 8), (8, 6)])
def test_matches_horner(device, dtype, scalar, dimension, degree):
    rng = np.random.default_rng(701)
    path = torch.tensor(rng.normal(size=(2, 3, 11, dimension)).cumsum(axis=-2) * 0.03, device=device, dtype=dtype)
    check_sig_methods(path, degree, [("ps", None)], scalar_term=scalar, n_jobs=3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,degree", [
    ((2, 0, 3), 4), ((2, 1, 3), 4), ((2, 2, 3), 0), ((2, 2, 3), 1), ((2, 2, 3), 5),
    ((2, 7, 3), 0), ((2, 7, 3), 1), ((0, 9, 2), 4), ((2, 0, 3, 9, 2), 4),
])
def test_trivial_paths(device, shape, degree):
    path = torch.zeros(shape, dtype=torch.float64, device=device)
    sig = pysiglib.sig(path, degree, scalar_term=True, lead_lag=True, time_aug=True, method=METHOD)
    expected = (torch.zeros_like(path) if path.shape[-2] < 2 or not path.numel() else
                pysiglib.sig_backprop(path, sig, torch.ones_like(sig), degree, lead_lag=True, time_aug=True))
    actual = pysiglib.sig_backprop(path, sig, torch.ones_like(sig), degree,
                                  lead_lag=True, time_aug=True, method=METHOD)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,a", [(1, 3), (5, 2), (7, 6)])
def test_blocks_and_splits(monkeypatch, device, rows, a):
    rng = np.random.default_rng(703)
    path = torch.tensor(rng.normal(size=(3, 8, 3)) * 0.05, dtype=torch.float64, device=device)
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", str(rows * 2 * (27 + 27) * 8))
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_CHUNK", "3")
    check_sig_methods(path, 6, [("ps", a)], n_jobs=3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("corrected", [False, True])
def test_workspace_and_partial_batches(monkeypatch, device, corrected):
    d, degree = (2, 3) if corrected else (3, 5)
    path = torch.randn(35, d, 9, dtype=torch.float64, device=device).transpose(-1, -2) * .03
    correction = torch.tensor([.01, -.02, .03, .04], dtype=path.dtype, device=device) if corrected else None
    sig = pysiglib.sig(path, degree, correction=correction)
    deriv = torch.ones(35, 2 * sig.shape[-1], dtype=path.dtype, device=device)[:, ::2]
    expected = pysiglib.sig_backprop(path, sig, deriv, degree, correction=correction)
    options = dict(method="ps", n_jobs=-1, correction=correction)
    monkeypatch.setenv("PYSIGLIB_PS_FACTOR_BYTES", "4096")
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", str((4 << 20 if device == "cuda" else 0) + 150000))
    torch.testing.assert_close(pysiglib.sig(path, degree, **options), sig, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(pysiglib.sig_backprop(path, sig, deriv, degree, **options), expected, rtol=1e-10, atol=1e-12)
    monkeypatch.setenv("PYSIGLIB_PS_WORKSPACE_BYTES", "1")
    with pytest.raises(Exception, match="workspace"):
        pysiglib.sig(path, degree, **options)
    with pytest.raises(Exception, match="workspace"):
        pysiglib.sig_backprop(path, sig, deriv, degree, **options)
    if corrected:
        torch.testing.assert_close(pysiglib.sig(path, degree, method="auto", correction=correction), sig)


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("segments,chunk", [(1, 1), (15, 16), (16, 16), (17, 16), (33, 1), (65, 3), (129, 64)])
def test_cuda_adjoint_scan_chunks(monkeypatch, dtype, segments, chunk):
    degree = 6
    rng = np.random.default_rng(708)
    values = rng.normal(size=(2, segments + 1, 3)).cumsum(axis=1) * (0.1 / np.sqrt(segments))
    path = torch.tensor(values, dtype=dtype, device="cuda")
    reference_path = path.cpu().double().numpy()
    sig = pysiglib.sig(path, degree, scalar_term=True)
    deriv = torch.tensor(rng.normal(size=sig.shape), dtype=dtype, device="cuda")
    reference_sig = pysiglib.sig(reference_path, degree, scalar_term=True)
    expected = pysiglib.sig_backprop(reference_path, reference_sig, deriv.cpu().double().numpy(), degree)
    monkeypatch.setenv("PYSIGLIB_PS_SCAN_CHUNK", str(chunk))
    actual = pysiglib.sig_backprop(path, sig, deriv, degree, method=METHOD)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind,ta,ll,scalar", [
    ("random", ta, ll, scalar) for ta, ll, scalar in product((False, True), repeat=3)
] + [(kind, False, False, True) for kind in ("zero", "repeated", "alternating", "closed", "translated", "cancelling")])
def test_chen_and_torch_autograd(device, dtype, kind, ta, ll, scalar):
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
    degree = 4 if kind == "random" else 6
    options = dict(time_aug=ta, lead_lag=ll, end_time=.7)
    reference = chen_reference(path, degree, **options)[0 if scalar else 1:]
    deriv = torch.tensor(rng.normal(size=reference.shape), dtype=path.dtype, device=device)
    expected, = torch.autograd.grad(reference, path, deriv)
    x = path.detach().to(dtype).requires_grad_()
    signature = torch_api.sig(x, degree, method=METHOD, scalar_term=scalar, **options)
    actual, = torch.autograd.grad(signature, x, deriv.to(dtype))
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(signature.double(), reference.detach(), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)


def test_validation_and_higher_derivatives():
    path = np.zeros((8, 2))
    sig = pysiglib.sig(path, 3)
    for options in (dict(method="bad"), dict(method=METHOD, correction=np.zeros(5)), dict(method=METHOD, n_jobs=0)):
        with pytest.raises(ValueError): pysiglib.sig_backprop(path, sig, sig, 3, **options)
    np.testing.assert_array_equal(pysiglib.sig_backprop(path, sig, sig, 3, correction=np.empty(0), method=METHOD), 0)
    tensor = torch.zeros((8, 2), dtype=torch.float64, requires_grad=True)
    value = torch_api.sig(tensor, 3, method=METHOD).sum()
    with pytest.raises(NotImplementedError, match="higher-order"):
        torch.autograd.grad(value, tensor, create_graph=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backward", [False, True])
def test_concurrent_calls_and_cleanup(device, backward):
    path = torch.tensor(np.random.default_rng(706).normal(size=(3, 65, 3)) * 0.05, device=device, dtype=torch.float64)
    sig = pysiglib.sig(path, 5)
    deriv = torch.ones_like(sig)
    expected = pysiglib.sig_backprop(path, sig, deriv, 5) if backward else sig
    if device == "cuda": torch.cuda.synchronize()

    def compute():
        if backward:
            return pysiglib.sig_backprop(path, sig, deriv, 5, method=METHOD, n_jobs=2)
        return pysiglib.sig(path, 5, method=METHOD, n_jobs=2)

    def run(_):
        if device == "cuda":
            with torch.cuda.stream(torch.cuda.Stream()):
                return compute()
        return compute()

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, range(6)))
    for result in results: torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
    if device == "cuda":
        from pysiglib.load_siglib import CUSIG
        CUSIG.cusig_shutdown()
        torch.testing.assert_close(run(0), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree,d,method,split,corrected", [
    (4, 3, "ps", None, False), (4, 3, "ps", 1, False), (4, 3, "ps", 4, False),
    (4, 2, "ps", None, True), (4, 2, "ps", 1, True), (4, 2, "ps", 4, True),
    (6, 4, "auto", None, False),
])
def test_jax_forward_and_reverse(monkeypatch, dtype, degree, d, method, split, corrected):
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as api
    jax.config.update("jax_enable_x64", True)
    if split is not None:
        monkeypatch.setenv("PYSIGLIB_PS_ENDPOINT_BLOCK", "invalid")
    path = np.random.default_rng(726).normal(scale=.02, size=(2, 21, d)).astype(dtype)
    correction = np.array([.02, -.01, .03, .01], dtype=dtype) if corrected else None
    # Exercise augmentation on explicit splits and the original shape for auto dispatch.
    options = dict(correction=correction, time_aug=split == 1, lead_lag=split == 4 and not corrected)
    run = lambda x: api.sig(x, degree, method=method, ps_split=split, scalar_term=corrected, **options)
    expected = pysiglib.sig(path, degree, scalar_term=corrected, **options)
    reference = pysiglib.sig_backprop(path, expected, np.ones_like(expected), degree, **options)
    grad = jax.grad(lambda x: run(x).sum())
    tol = 2e-5 if dtype == np.float32 else 1e-10
    for transform in (lambda f: f, jax.jit, jax.vmap, lambda f: jax.jit(jax.vmap(f))):
        np.testing.assert_allclose(transform(run)(path), expected, rtol=tol, atol=tol)
        np.testing.assert_allclose(transform(grad)(path), reference, rtol=tol, atol=tol)
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(run, (path,), (np.ones_like(path),))


def test_jax_empty_batch_and_higher_derivatives():
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as api
    grad = jax.grad(lambda p: api.sig(p, 4, method=METHOD).sum())
    path = np.zeros((2, 0, 7, 3), dtype=np.float32)
    for actual in (grad(path), jax.jit(grad)(path), jax.vmap(grad)(path)):
        assert actual.shape == path.shape
    with pytest.raises((ValueError, TypeError), match="differentiat|JVP"):
        jax.grad(lambda p: grad(p).sum())(np.zeros((7, 3), dtype=np.float32))
