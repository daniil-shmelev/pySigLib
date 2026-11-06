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

"""Independent forward and reverse checks for coefficient method dispatch."""
from fractions import Fraction
from math import factorial

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api
from conftest import force_highway_target, highway_targets, DEVICES


def pascal_with_gradient(path, words, seeds):
    coefs = pysiglib.sig_coef(path, words, method="pascal", prefixes=True, n_jobs=-1)
    gradient = pysiglib.sig_coef_backprop(path, words, coefs, seeds, method="pascal", n_jobs=-1)
    return coefs, gradient


def chen_prefixes(path, words):
    result = []
    for word in words:
        state = [1] + [0] * len(word)
        for previous, current in zip(path[:-1], path[1:]):
            a = [current[c] - previous[c] for c in word]
            old = state[:]
            for j in range(1, len(word) + 1):
                product = 1
                for k in range(1, j + 1):
                    product *= a[j - k]
                    state[j] += old[j - k] * product / factorial(k)
        result.extend(state[1:] if word else [1])
    return np.asarray(result)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["horner", "pascal", "auto"])
@pytest.mark.parametrize("segments", [1, 7, 63, 64, 65, 129, 256, 257])
def test_coefficient_dispatch(device, dtype, method, segments):
    rng = np.random.default_rng(407 + segments)
    x = torch.tensor(rng.normal(scale=.04, size=(9, segments + 1, 3)).cumsum(axis=1), dtype=dtype, device=device)
    words = [(), (0,), (1, 0, 1), (2, 1, 0, 2), (1, 0, 1)]
    if segments in (1, 7, 65):
        words += [tuple(i % 3 for i in range(m)) for m in range(20, 0, -1)]
    if segments == 1:
        x.zero_()
        x[:, 1, 0] = .125
        words += [(0,) * 16, (0, 1, 0), (0,) * 16]
    y = pysiglib.sig_coef(x, words, method=method, prefixes=True)
    seeds = torch.tensor(rng.normal(size=y.shape), dtype=dtype, device=device)
    derivs = seeds.clone()
    gradient = pysiglib.sig_coef_backprop(x, words, y, derivs, method=method)
    if method != "horner" or device != "cpu":
        torch.testing.assert_close(derivs, seeds, atol=0, rtol=0)
    points = x.cpu().double().numpy()
    direction = rng.normal(size=points.shape)
    # The same Chen sum gives the values and the complex-step directional derivative.
    reference = np.stack([chen_prefixes(p + 1e-25j * v, words) for p, v in zip(points, direction)])
    tol = 3e-5 if dtype == torch.float32 else 2e-11
    np.testing.assert_allclose(y.cpu(), reference.real, atol=tol, rtol=tol)
    np.testing.assert_allclose((gradient.cpu().numpy() * direction).sum(axis=(1, 2)),
                               (seeds.cpu().numpy() * reference.imag).sum(axis=1) / 1e-25, atol=5 * tol, rtol=5 * tol)
    if method == "horner" and segments == 7:
        torch.testing.assert_close(y, pysiglib.sig_coef(x, words, prefixes=True), atol=0, rtol=0)
        last = np.cumsum([max(1, len(w)) for w in words]) - 1
        torch.testing.assert_close(pysiglib.sig_coef(x, words, method=method), y[..., last], atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["horner", "pascal"])
@pytest.mark.parametrize("degree", [2, 3, 8, 16, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pascal_exact_rational(device, method, degree, dtype):
    path = [[Fraction(0), Fraction(0)], [Fraction(1), Fraction(1, 2)],
            [Fraction(-1, 2), Fraction(-1)], [Fraction(1, 32), Fraction(1, 16)]]
    word = tuple(i % 2 for i in range(degree))
    expected = np.asarray(chen_prefixes(path, [word]), dtype=float)
    x = torch.tensor(np.asarray(path, dtype=float), device=device, dtype=dtype)
    actual = pysiglib.sig_coef(x, word, prefixes=True, method=method)
    np.testing.assert_allclose(actual.cpu(), expected, atol=8 * torch.finfo(dtype).eps, rtol=2e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["horner", "pascal"])
def test_pascal_gradcheck(device, method):
    x = torch.randn((5, 2), dtype=torch.float64, device=device, requires_grad=True) * .1
    assert torch.autograd.gradcheck(lambda p: torch_api.sig_coef(p, [(0, 1, 0), (1, 1)],
                                                                prefixes=True, method=method), (x,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pascal_ranges(device, dtype):
    numpy_type = np.float32 if dtype == torch.float32 else np.float64
    tiny = np.nextafter(numpy_type(0), numpy_type(1))
    large = numpy_type(1e30 if dtype == torch.float32 else 1e300)
    x = torch.tensor([[0, 0], [large, 0], [large, 3 * tiny]], dtype=dtype, device=device)
    actual = pysiglib.sig_coef(x, (0, 1, 1), prefixes=True, method="pascal")
    expected = np.asarray(chen_prefixes(x.cpu().numpy().astype(np.longdouble), [(0, 1, 1)]), dtype=numpy_type)
    np.testing.assert_allclose(actual.cpu(), expected, rtol=2e-6 if dtype == torch.float32 else 1e-13, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_pascal_empty_batch_and_invalid_method(device):
    x = torch.empty((2, 0, 5, 2), device=device)
    assert pysiglib.sig_coef(x, [(), (0, 1)], method="pascal").shape == (2, 0, 2)
    with pytest.raises(ValueError, match="method"):
        pysiglib.sig_coef(x, (0,), method="unknown")


def test_pascal_generic_cpu_and_workspace(monkeypatch):
    x = np.array([[0.], [.5], [.75], [1.]], dtype=np.float64)
    word = (0,) * 32
    actual = pysiglib.sig_coef(x, word, method="pascal")
    np.testing.assert_allclose(actual, [1 / factorial(32)], rtol=1e-12, atol=0)
    y = pysiglib.sig_coef(x, word, prefixes=True, method="pascal")
    gradient = pysiglib.sig_coef_backprop(x, word, y, np.ones_like(y), method="pascal")
    endpoint = sum(1 / factorial(j) for j in range(32))
    np.testing.assert_allclose(gradient[:, 0], [-endpoint, 0, 0, endpoint], rtol=1e-12, atol=1e-12)
    monkeypatch.setenv("PYSIGLIB_PASCAL_WORKSPACE_BYTES", "1")
    with pytest.raises(Exception, match="workspace"):
        pysiglib.sig_coef_backprop(x, word, y, np.ones_like(y), method="pascal")


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA required")
@pytest.mark.parametrize("method", [None, "horner", "pascal", "auto"])
@pytest.mark.parametrize("words", [[(0, 1, 2), (2, 2)], [(0, 1, 2), (2, 2, 1)], [(0, 1) * 8, (2,) * 11]])
def test_pascal_nondefault_stream_and_limit(method, words):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn((4, 70, 3), device="cuda", dtype=torch.float64) * .03
        y = torch_api.sig_coef(x.requires_grad_(), words, method=method)
        y.sum().backward()
    stream.synchronize()
    assert torch.isfinite(x.grad).all()
    cpu = x.detach().cpu().requires_grad_()
    torch_api.sig_coef(cpu, words, method="pascal").sum().backward()
    torch.testing.assert_close(x.grad.cpu(), cpu.grad, rtol=1e-10, atol=1e-11)
    with pytest.raises(Exception, match="20"):
        pysiglib.sig_coef(x, (0,) * 21, method="pascal")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pascal_range_restart(device, dtype):
    numpy_type = np.float32 if dtype == torch.float32 else np.float64
    large = 150. if dtype == torch.float32 else 1e16
    words = [(0,) * 20]
    x = torch.tensor([[0.], [large]], dtype=dtype, device=device)
    y = pysiglib.sig_coef(x, words, method="pascal", prefixes=True)
    expected = np.asarray([float(Fraction.from_float(large) ** j / factorial(j)) for j in range(1, 21)])
    np.testing.assert_allclose(y.cpu(), expected, rtol=5e-6 if dtype == torch.float32 else 2e-14)
    x = torch.tensor([[0.], [16.]], dtype=dtype, device=device)
    seeds = torch.zeros(20, dtype=dtype, device=device)
    seeds[-1] = 1e-35 if dtype == torch.float32 else 1e-310
    _, gradient = pascal_with_gradient(x, words, seeds)
    expected_grad = float(Fraction.from_float(seeds[-1].item()) * 16 ** 19 / factorial(19))
    np.testing.assert_allclose(gradient.cpu(), [[-expected_grad], [expected_grad]], rtol=1e-5, atol=0)
    large = numpy_type(np.finfo(numpy_type).max * .6)
    x = torch.tensor([[0., 0.], [large, 0.], [large, .25]], dtype=dtype, device=device)
    y, gradient = pascal_with_gradient(x, [(0, 1)], torch.tensor([0., 1.], dtype=dtype, device=device))
    assert torch.isfinite(y).all() and torch.isfinite(gradient).all()
    np.testing.assert_allclose(y.cpu(), [large, large * .25], rtol=2e-6 if dtype == torch.float32 else 1e-14)
    np.testing.assert_allclose(gradient.cpu().numpy()[:, 1] / large, [-.5, -.5, 1.], rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [1, 3, 7, 9])
def test_pascal_partial_chunks_and_word_workers(device, batch, monkeypatch):
    torch.manual_seed(81)
    x = torch.randn((batch, 66, 3), dtype=torch.float64, device=device) * .03
    words = [(0, 1, 0), (2, 1), (), (1,) * 16] * 4
    y = pysiglib.sig_coef(x, words, prefixes=True, method="pascal", n_jobs=1)
    seeds = torch.randn_like(y)
    serial = pysiglib.sig_coef_backprop(x, words, y, seeds, method="pascal", n_jobs=1)
    parallel = pysiglib.sig_coef_backprop(x, words, y, seeds, method="pascal", n_jobs=-1)
    torch.testing.assert_close(parallel, serial, atol=1e-12, rtol=1e-11)
    monkeypatch.setenv("PYSIGLIB_PASCAL_WORKSPACE_BYTES", "65536")
    limited = pysiglib.sig_coef_backprop(x, words, y, seeds, method="pascal", n_jobs=-1)
    torch.testing.assert_close(limited, serial, atol=1e-12, rtol=1e-11)
    monkeypatch.setenv("PYSIGLIB_PASCAL_WORKSPACE_BYTES", "1")
    with pytest.raises(Exception, match="workspace"):
        pysiglib.sig_coef_backprop(x, words, y, seeds, method="pascal")


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA required")
@pytest.mark.parametrize("degree,batch,segments", [(8, 3, 7), (15, 3, 65), (16, 3, 65), (20, 32, 65), (20, 33, 65)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("constrained", [False, True])
def test_pascal_warp_groups(degree, batch, segments, dtype, constrained, monkeypatch):
    torch.manual_seed(923 + degree)
    x = torch.randn((batch, segments + 1, 3), dtype=dtype) * .1
    word = tuple(j % 3 for j in range(degree))
    words = [word, tuple(reversed(word)), (0,) * degree, word, ()] if degree < 20 else [word] * 8
    coefs = pysiglib.sig_coef(x.double(), words, prefixes=True, method="pascal")
    seeds = torch.randn_like(coefs).to(dtype)
    reference = pysiglib.sig_coef_backprop(x.double(), words, coefs, seeds.double(), method="pascal")
    monkeypatch.setenv("PYSIGLIB_PASCAL_WORKSPACE_BYTES", str(1024 * x.element_size() if constrained else 256 * 1024 * 1024))
    actual, gradient = pascal_with_gradient(x.cuda(), words, seeds.cuda())
    torch.testing.assert_close(actual.cpu().double(), coefs, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(gradient.cpu().double(), reference, rtol=3e-5, atol=3e-6)


def test_pascal_highway_targets():
    rng = np.random.default_rng(581)
    x = torch.tensor(rng.normal(size=(9, 5, 2)) * .1, dtype=torch.float64)
    words = [(0, 1) * 8, (1,) * 5]
    reference = np.asarray([chen_prefixes(p, words) for p in x.numpy()])
    for target in highway_targets().values():
        with force_highway_target(target):
            y = pysiglib.sig_coef(x, words, method="pascal", prefixes=True)
            np.testing.assert_allclose(y, reference, rtol=1e-12, atol=1e-13)
            dx = pysiglib.sig_coef_backprop(x, words, y, torch.ones_like(y), method="pascal")
            np.testing.assert_allclose(dx.sum(axis=1), 0., atol=1e-12)


@pytest.mark.parametrize("method", [None, "horner", "pascal", "auto"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("time_aug,lead_lag", [(False, False), (True, False), (False, True), (True, True)])
def test_pascal_jax_shapes_and_transforms(method, dtype, time_aug, lead_lag):
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as jax_api
    jax.config.update("jax_enable_x64", True)
    words = [(), (0, 1), (1, 1), (0, 1)]
    options = dict(prefixes=True, method=method, time_aug=time_aug, lead_lag=lead_lag, end_time=2.)
    for shape in [(2, 0, 2), (2, 1, 2), (2, 3, 8, 2), (0, 8, 2)]:
        x = np.random.default_rng(31).normal(scale=.1, size=shape).astype(dtype)
        tol = 2e-5 if dtype == np.float32 else 1e-11
        f = lambda p: jax_api.sig_coef(p, words, **options)
        expected = pysiglib.sig_coef(x, words, **options)
        for run in (f, jax.jit(f), jax.jit(jax.vmap(f))):
            np.testing.assert_allclose(run(x), expected, atol=tol)
        seeds = np.random.default_rng(32).normal(size=expected.shape).astype(dtype)
        gradient = jax.jit(jax.grad(lambda p: (f(p) * seeds).sum()))(x)
        reference = pysiglib.sig_coef_backprop(x, words, expected, seeds,
                                              **{k: v for k, v in options.items() if k != "prefixes"})
        np.testing.assert_allclose(gradient, reference, atol=tol)


@pytest.mark.parametrize("backend", DEVICES + ["jax"])
@pytest.mark.parametrize("method", [None, "horner", "pascal", "auto"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_single_segment(backend, method, dtype):
    path = np.zeros((2, 32), dtype=dtype)
    path[1] = .125
    word = tuple(range(16))
    if backend == "jax":
        jax = pytest.importorskip("jax")
        import pysiglib.jax_api as api
        jax.config.update("jax_enable_x64", True)
        value, gradient = jax.jit(jax.value_and_grad(lambda x: api.sig_coef(x, word, method=method).sum()))(path)
    else:
        x = torch.tensor(path, device=backend, requires_grad=True)
        y = torch_api.sig_coef(x, word, method=method)
        gradient, = torch.autograd.grad(y.sum(), x)
        value, gradient = y.detach().cpu(), gradient.cpu()
    expected = .125 ** 16 / factorial(16)
    expected_gradient = np.zeros_like(path)
    expected_gradient[1, :16] = expected / .125
    expected_gradient[0] = -expected_gradient[1]
    tol = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(value, expected, rtol=tol, atol=0)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=tol, atol=0)
