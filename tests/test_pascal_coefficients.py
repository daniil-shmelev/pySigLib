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
import ctypes

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api
from conftest import DEVICES


def native_variant(path, words, method, seeds=None):
    from pysiglib import load_siglib as lib
    dtype = str(path.dtype).split(".")[-1]
    scalar = ctypes.c_float if dtype == "float32" else ctypes.c_double
    pointer = ctypes.POINTER(scalar)
    integer_pointer = ctypes.POINTER(ctypes.c_uint64)
    indices = np.asarray([c for word in words for c in word], dtype=np.uint64)
    degrees = np.asarray([len(word) for word in words], dtype=np.uint64)
    batch = int(np.prod(path.shape[:-2]))
    length, dimension = path.shape[-2:]
    coefs = torch.empty((*path.shape[:-2], sum(max(1, len(w)) for w in words)), dtype=path.dtype, device=path.device)
    gradient = torch.empty_like(path)
    address = lambda x: ctypes.cast(x.data_ptr(), pointer)
    args = (indices.ctypes.data_as(integer_pointer), len(words), degrees.ctypes.data_as(integer_pointer),
            batch, dimension, length)
    if path.device.type == "cpu":
        status = lib.CPSIG_SIG_COEF[dtype](address(path), address(coefs), *args, False, False, 1., True, -1, method)
        if seeds is not None and not status:
            status = lib.CPSIG_SIG_COEF_BACKPROP[dtype](address(path), address(gradient), address(coefs), address(seeds), *args, False, False, 1., -1, method)
    else:
        stream = torch.cuda.current_stream(path.device).cuda_stream
        status = lib.CUSIG_SIG_COEF[dtype](address(path), address(coefs), *args, True, False, False, 1., method, stream)
        if seeds is not None and not status:
            status = lib.CUSIG_SIG_COEF_BACKPROP[dtype](address(path), address(gradient), address(coefs), address(seeds), *args, False, False, 1., method, stream)
    assert status == 0
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
@pytest.mark.parametrize("method", ["pascal", "auto"])
@pytest.mark.parametrize("segments", [1, 7, 63, 64, 65, 129])
def test_pascal_prefixes_and_reverse(device, dtype, method, segments):
    rng = np.random.default_rng(407 + segments)
    array = np.cumsum(rng.normal(size=(3, segments + 1, 3)) * .04, axis=1)
    words = [(), (0,), (1, 0, 1), (2, 1, 0, 2), (1, 0, 1)]
    x = torch.tensor(array, dtype=dtype, device=device)
    actual = pysiglib.sig_coef(x, words, method=method, prefixes=True)
    expected = np.asarray([chen_prefixes(p, words) for p in x.cpu().numpy()])
    tolerance = 3e-5 if dtype == torch.float32 else 2e-11
    np.testing.assert_allclose(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
    seeds = rng.normal(size=expected.shape)
    direction = rng.normal(size=array.shape)
    derivs = torch.tensor(seeds, dtype=dtype, device=device)
    gradient = pysiglib.sig_coef_backprop(x, words, actual, derivs, method=method)
    reference = np.asarray([np.dot(chen_prefixes(p.astype(complex) + 1e-25j * v, words).imag / 1e-25, s)
                            for p, v, s in zip(x.cpu().numpy(), direction, derivs.cpu().numpy())])
    np.testing.assert_allclose(np.sum(gradient.cpu().numpy() * direction, axis=(1, 2)), reference,
                               rtol=5 * tolerance, atol=5 * tolerance)
    np.testing.assert_array_equal(derivs.cpu(), seeds.astype(derivs.cpu().numpy().dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("degree", [2, 3, 8, 16, 20])
def test_pascal_exact_rational(device, degree):
    path = [[Fraction(0), Fraction(0)], [Fraction(1, 8), Fraction(-1, 4)],
            [Fraction(1, 8), Fraction(1, 8)], [Fraction(-1, 4), Fraction(3, 8)]]
    word = tuple(i % 2 for i in range(degree))
    expected = np.asarray(chen_prefixes(path, [word]), dtype=float)
    actual = pysiglib.sig_coef(torch.tensor(np.asarray(path, dtype=float), device=device), word,
                              prefixes=True, method="pascal")
    np.testing.assert_allclose(actual.cpu(), expected, atol=1e-15, rtol=2e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", [None, "pascal", "auto"])
@pytest.mark.parametrize("length", [0, 1])
def test_pascal_identity(device, method, length):
    x = torch.zeros((2, length, 2), dtype=torch.float64, device=device)
    words = [(), (0,), (0, 1), ()]
    y = pysiglib.sig_coef(x, words, prefixes=True, method=method)
    np.testing.assert_array_equal(y.cpu(), [[1, 0, 0, 0, 1]] * 2)
    gradient = pysiglib.sig_coef_backprop(x, words, y, torch.ones_like(y), method=method)
    assert gradient.shape == x.shape
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("time_aug,lead_lag", [(True, False), (False, True), (True, True)])
def test_pascal_transforms_and_views(device, time_aug, lead_lag):
    torch.manual_seed(9)
    x = (torch.randn((2, 3, 10, 4), device=device, dtype=torch.float64) * .1)[..., ::2]
    x.requires_grad_()
    d = (4 if lead_lag else 2) + int(time_aug)
    words = [(), (d - 1, 0, 1), (1, 1)]
    result = torch_api.sig_coef(x, words, prefixes=True, method="pascal", time_aug=time_aug, lead_lag=lead_lag, end_time=2.)
    transformed = torch_api.transform_path(x, time_aug=time_aug, lead_lag=lead_lag, end_time=2.)
    expected = torch_api.sig_coef(transformed, words, prefixes=True, method="pascal")
    torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-11)
    weights = torch.randn_like(result)
    a, = torch.autograd.grad((result * weights).sum(), x, retain_graph=True)
    b, = torch.autograd.grad((expected * weights).sum(), x)
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_pascal_gradcheck(device):
    x = torch.randn((5, 2), dtype=torch.float64, device=device, requires_grad=True) * .1
    assert torch.autograd.gradcheck(lambda p: torch_api.sig_coef(p, [(0, 1, 0), (1, 1)],
                                                                prefixes=True, method="pascal"), (x,))


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
@pytest.mark.parametrize("method", ["pascal", "auto"])
@pytest.mark.parametrize("words", [[(0, 1, 2), (2, 2)], [(0, 1, 2), (2, 2, 1)]])
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


@pytest.mark.parametrize("method", ["pascal", "auto"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pascal_jax(method, dtype):
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as jax_api
    jax.config.update("jax_enable_x64", True)
    x = (np.random.default_rng(12).normal(size=(2, 7, 3)) * .1).astype(dtype)
    words = [(), (0, 1, 0), (2, 1)]
    f = lambda p: jax_api.sig_coef(p, words, prefixes=True, method=method)
    expected = pysiglib.sig_coef(x, words, prefixes=True, method=method)
    tolerance = 2e-6 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(jax.jit(f)(x), expected, atol=tolerance)
    np.testing.assert_allclose(jax.jit(jax.vmap(f))(x), expected, atol=tolerance)
    gradient = jax.jit(jax.grad(lambda p: f(p).sum()))(x)
    reference = pysiglib.sig_coef_backprop(x, words, expected, np.ones_like(expected), method=method)
    np.testing.assert_allclose(gradient, reference, atol=tolerance)


def test_default_jax_empty_word_gradient():
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as jax_api
    path = np.array([[0.], [1.], [2.]], dtype=np.float32)
    function = lambda x: (jax_api.sig_coef(x, [(), (0,)], prefixes=True) * np.array([7., 3.])).sum()
    value, gradient = jax.jit(jax.value_and_grad(function))(path)
    np.testing.assert_allclose(value, 13.)
    np.testing.assert_allclose(gradient, [[-3.], [0.], [3.]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", [3, 4, 5])
def test_pascal_native_range_restart(device, dtype, variant):
    numpy_type = np.float32 if dtype == torch.float32 else np.float64
    large = 150. if dtype == torch.float32 else 1e16
    words = [(0,) * 20]
    x = torch.tensor([[0.], [large]], dtype=dtype, device=device)
    y, _ = native_variant(x, words, variant)
    expected = np.asarray([float(Fraction.from_float(large) ** j / factorial(j)) for j in range(1, 21)])
    np.testing.assert_allclose(y.cpu(), expected, rtol=5e-6 if dtype == torch.float32 else 2e-14)
    x = torch.tensor([[0.], [16.]], dtype=dtype, device=device)
    seeds = torch.zeros(20, dtype=dtype, device=device)
    seeds[-1] = 1e-35 if dtype == torch.float32 else 1e-310
    _, gradient = native_variant(x, words, variant, seeds)
    expected_grad = float(Fraction.from_float(seeds[-1].item()) * 16 ** 19 / factorial(19))
    np.testing.assert_allclose(gradient.cpu(), [[-expected_grad], [expected_grad]], rtol=1e-5, atol=0)
    large = numpy_type(np.finfo(numpy_type).max * .6)
    x = torch.tensor([[0., 0.], [large, 0.], [large, .25]], dtype=dtype, device=device)
    y, gradient = native_variant(x, [(0, 1)], variant, torch.tensor([0., 1.], dtype=dtype, device=device))
    assert torch.isfinite(y).all() and torch.isfinite(gradient).all()
    np.testing.assert_allclose(y.cpu(), [large, large * .25], rtol=2e-6 if dtype == torch.float32 else 1e-14)
    np.testing.assert_allclose(gradient.cpu().numpy()[:, 1] / large, [-.5, -.5, 1.], rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", [3, 4])
def test_pascal_native_cancellation(device, dtype, variant):
    rational = [[Fraction(0), Fraction(0)], [Fraction(1), Fraction(1, 2)],
                [Fraction(-1, 2), Fraction(-1)], [Fraction(1, 32), Fraction(1, 16)]]
    word = (0, 1) * 8
    x = torch.tensor(np.asarray(rational, dtype=float), dtype=dtype, device=device)
    seeds = torch.linspace(-1., 1., 16, dtype=dtype, device=device)
    actual, gradient = native_variant(x, [word], variant, seeds)
    expected = np.asarray(chen_prefixes(rational, [word]), dtype=float)
    eps = torch.finfo(dtype).eps
    np.testing.assert_allclose(actual.cpu(), expected, atol=128 * eps, rtol=128 * eps)
    direction = np.random.default_rng(173).normal(size=x.shape)
    reference = np.dot(chen_prefixes(x.cpu().numpy().astype(complex) + 1e-25j * direction, [word]).imag / 1e-25, seeds.cpu().numpy())
    np.testing.assert_allclose(np.sum(gradient.cpu().numpy() * direction), reference, atol=512 * eps, rtol=512 * eps)


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pascal_all_specializations(device, dtype, monkeypatch):
    rng = np.random.default_rng(914)
    for degree in range(2, 21):
        x = torch.tensor(rng.normal(size=(7, 6, 3)) * .15, dtype=dtype, device=device)
        words = [tuple(map(int, rng.integers(3, size=degree)))]
        direction = rng.normal(size=x.shape)
        seeds = torch.tensor(rng.normal(size=(7, degree)), dtype=dtype, device=device)
        reference = np.asarray([chen_prefixes(p, words) for p in x.cpu().numpy()])
        derivative = np.asarray([np.dot(chen_prefixes(p.astype(complex) + 1e-25j * v, words).imag / 1e-25, s)
                                 for p, v, s in zip(x.cpu().numpy(), direction, seeds.cpu().numpy())])
        for variant in [3, 4]:
            for warp in ([0, 1] if device == "cuda" else [0]):
                monkeypatch.setenv("PYSIGLIB_PASCAL_BENCH_WARP", str(warp))
                y, dx = native_variant(x, words, variant, seeds)
                tolerance = 2e-5 if dtype == torch.float32 else 2e-12
                np.testing.assert_allclose(y.cpu(), reference, rtol=tolerance, atol=tolerance)
                np.testing.assert_allclose(np.sum(dx.cpu().numpy() * direction, axis=(1, 2)), derivative,
                                           rtol=tolerance, atol=tolerance)


def test_pascal_highway_targets():
    from pysiglib.load_siglib import CPSIG
    if not hasattr(CPSIG, "cpsig_test_supported_targets"):
        pytest.skip("Requires a native test build")
    CPSIG.cpsig_test_supported_targets.restype = ctypes.c_int64
    CPSIG.cpsig_test_target_name.argtypes = [ctypes.c_int64]
    CPSIG.cpsig_test_target_name.restype = ctypes.c_char_p
    CPSIG.cpsig_test_set_target.argtypes = [ctypes.c_int64]
    remaining = CPSIG.cpsig_test_supported_targets()
    rng = np.random.default_rng(581)
    x = torch.tensor(rng.normal(size=(9, 5, 2)) * .1, dtype=torch.float64)
    words = [(0, 1) * 8, (1,) * 5]
    reference = np.asarray([chen_prefixes(p, words) for p in x.numpy()])
    tested = []
    try:
        while remaining:
            target = remaining & -remaining
            remaining &= ~target
            tested.append(CPSIG.cpsig_test_target_name(target).decode())
            assert CPSIG.cpsig_test_set_target(target) == 0
            y = pysiglib.sig_coef(x, words, method="pascal", prefixes=True)
            np.testing.assert_allclose(y, reference, rtol=1e-12, atol=1e-13)
            dx = pysiglib.sig_coef_backprop(x, words, y, torch.ones_like(y), method="pascal")
            np.testing.assert_allclose(dx.sum(axis=1), 0., atol=1e-12)
    finally:
        assert CPSIG.cpsig_test_set_target(0) == 0
    assert tested


@pytest.mark.parametrize("method", ["pascal", "auto"])
@pytest.mark.parametrize("time_aug,lead_lag", [(False, False), (True, False), (False, True), (True, True)])
def test_pascal_jax_shapes_and_transforms(method, time_aug, lead_lag):
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as jax_api
    jax.config.update("jax_enable_x64", True)
    words = [(), (0, 1), (1, 1), (0, 1)]
    options = dict(prefixes=True, method=method, time_aug=time_aug, lead_lag=lead_lag, end_time=2.)
    for shape in [(2, 0, 2), (2, 1, 2), (2, 3, 8, 2), (0, 8, 2)]:
        x = np.random.default_rng(31).normal(size=shape) * .1
        f = lambda p: jax_api.sig_coef(p, words, **options)
        expected = pysiglib.sig_coef(x, words, **options)
        np.testing.assert_allclose(f(x), expected, atol=1e-12)
        np.testing.assert_allclose(jax.jit(f)(x), expected, atol=1e-12)
        gradient = jax.jit(jax.grad(lambda p: f(p).sum()))(x)
        reference = pysiglib.sig_coef_backprop(x, words, expected, np.ones_like(expected),
                                              **{k: v for k, v in options.items() if k != "prefixes"})
        np.testing.assert_allclose(gradient, reference, atol=1e-11)


@pytest.mark.parametrize("method", ["pascal", "auto"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pascal_jax_sparse_large_signature(method, dtype):
    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as jax_api
    jax.config.update("jax_enable_x64", True)
    path = np.zeros((2, 32), dtype=dtype)
    path[1] = .125
    word = tuple(range(16))
    function = lambda x: jax_api.sig_coef(x, word, method=method).sum()
    value, gradient = jax.jit(jax.value_and_grad(function))(path)
    expected = .125 ** 16 / factorial(16)
    expected_gradient = np.zeros_like(path)
    expected_gradient[1, :16] = expected / .125
    expected_gradient[0] = -expected_gradient[1]
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(value, expected, rtol=tolerance, atol=0)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=tolerance, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_auto_single_segment_all_prefix_seeds(device, dtype):
    path = torch.tensor([[0., 0.], [.125, 0.]], dtype=dtype, device=device)
    words = [(0,) * 16, (0, 1, 0), (0,) * 16]
    coefs = pysiglib.sig_coef(path, words, prefixes=True, method="auto")
    seeds = torch.arange(1, 36, dtype=dtype, device=device) / 32
    gradient = pysiglib.sig_coef_backprop(path, words, coefs, seeds, method="auto")
    expected = np.zeros((2, 2))
    for i in range(1, 17):
        expected[1, 0] += (seeds[i - 1].item() + seeds[18 + i].item()) * .125 ** (i - 1) / factorial(i - 1)
    expected[1, 0] += seeds[16].item()
    expected[1, 1] = seeds[17].item() * .125 / 2 + seeds[18].item() * .125 ** 2 / 6
    expected[0] = -expected[1]
    np.testing.assert_allclose(gradient.cpu(), expected, rtol=2e-6 if dtype == torch.float32 else 1e-13, atol=0)
