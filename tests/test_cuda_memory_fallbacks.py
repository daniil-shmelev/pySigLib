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
import pytest
import torch

import native_api as pysiglib
from conftest import check_close, skip_no_cuda


def _tolerance(dtype):
    return 5e-4 if dtype == torch.float32 else 2e-10


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_signature_cuda_global_fallback_degree_one(dtype, scalar_term):
    generator = torch.Generator().manual_seed(20260529)
    base = torch.randn(3, 200000, dtype=dtype, generator=generator)
    path_cpu = base[:, ::2]
    assert not path_cpu.is_contiguous()
    path_cuda = path_cpu.cuda().requires_grad_()

    for horner in (True, False):
        expected = pysiglib.sig(
            path_cpu, 1, horner=horner, scalar_term=scalar_term)
        actual = pysiglib.torch_api.signature(
            path_cuda, 1, horner=horner, scalar_term=scalar_term)
        weights = torch.linspace(
            -0.25, 0.5, actual.numel(), dtype=dtype, device="cuda"
        ).reshape(actual.shape)
        grad = torch.autograd.grad(
            (actual * weights).sum(), path_cuda, retain_graph=True)[0]

        torch.testing.assert_close(
            actual.cpu(), expected, rtol=_tolerance(dtype),
            atol=_tolerance(dtype))
        torch.testing.assert_close(
            grad[0], -weights[..., -100000:], rtol=0, atol=0)
        torch.testing.assert_close(grad[1], torch.zeros_like(grad[1]))
        torch.testing.assert_close(
            grad[2], weights[..., -100000:], rtol=0, atol=0)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signature_cuda_global_fallback_lead_lag(dtype):
    generator = torch.Generator().manual_seed(20260530)
    path_cpu = torch.randn(
        3, 50000, dtype=dtype, generator=generator).requires_grad_()
    path_cuda = path_cpu.cuda().requires_grad_()

    expected = pysiglib.torch_api.signature(path_cpu, 1, lead_lag=True)
    actual = pysiglib.torch_api.signature(path_cuda, 1, lead_lag=True)
    expected_grad = torch.autograd.grad(expected.sum(), path_cpu)[0]
    actual_grad = torch.autograd.grad(actual.sum(), path_cuda)[0]

    torch.testing.assert_close(
        actual.cpu(), expected, rtol=_tolerance(dtype), atol=_tolerance(dtype))
    torch.testing.assert_close(
        actual_grad.cpu(), expected_grad,
        rtol=_tolerance(dtype), atol=_tolerance(dtype))


@skip_no_cuda
@pytest.mark.parametrize(
    "dimension,dtype",
    [(34, torch.float32), (27, torch.float64)],
)
def test_method_three_log_sig_backward_global_operands(dimension, dtype):
    pysiglib.prepare_log_sig(dimension, 3, 3)
    generator = torch.Generator().manual_seed(20260531)
    path_cpu = (
        0.01 * torch.randn(2, 4, dimension, dtype=dtype, generator=generator)
    ).requires_grad_()
    path_cuda = path_cpu.detach().cuda().requires_grad_()

    expected = pysiglib.torch_api.log_sig(path_cpu, 3, method=3)
    actual = pysiglib.torch_api.log_sig(path_cuda, 3, method=3)
    weights = torch.randn(expected.shape, dtype=dtype, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), path_cpu)[0]
    actual_grad = torch.autograd.grad(
        (actual * weights.cuda()).sum(), path_cuda)[0]

    torch.testing.assert_close(
        actual.cpu(), expected, rtol=_tolerance(dtype), atol=_tolerance(dtype))
    torch.testing.assert_close(
        actual_grad.cpu(), expected_grad,
        rtol=_tolerance(dtype), atol=_tolerance(dtype))


@skip_no_cuda
@pytest.mark.parametrize(
    "dimension,degree,planar,dtype",
    [
        (1, 7, True, torch.float32),
        (1, 7, True, torch.float64),
        (4, 4, False, torch.float64),
    ],
)
def test_branched_dense_cuda_global_fallback(
        dimension, degree, planar, dtype):
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
    generator = torch.Generator().manual_seed(20260601)
    path_cpu = (
        0.05 * torch.randn(2, 5, dimension, dtype=dtype, generator=generator)
    ).requires_grad_()
    path_cuda = path_cpu.detach().cuda().requires_grad_()

    expected = pysiglib.torch_api.branched_sig(
        path_cpu, degree, planar=planar)
    actual = pysiglib.torch_api.branched_sig(
        path_cuda, degree, planar=planar)
    weights = torch.randn(expected.shape, dtype=dtype, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), path_cpu)[0]
    actual_grad = torch.autograd.grad(
        (actual * weights.cuda()).sum(), path_cuda)[0]

    torch.testing.assert_close(
        actual.cpu(), expected, rtol=_tolerance(dtype), atol=_tolerance(dtype))
    torch.testing.assert_close(
        actual_grad.cpu(), expected_grad,
        rtol=_tolerance(dtype), atol=_tolerance(dtype))


@skip_no_cuda
@pytest.mark.parametrize(
    "dimension,degree,planar",
    [(1, 8, True), (5, 4, False)],
)
def test_branched_more_than_1024_trees_all_paths(
        dimension, degree, planar):
    dtype = torch.float32
    basis = list(pysiglib.trees(dimension, degree, planar=planar)[1:])
    assert len(basis) > 1024
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
    pysiglib.prepare_branched_sig_coef(
        dimension, basis, planar=planar)
    generator = torch.Generator().manual_seed(20260602)
    path1 = 0.03 * torch.randn(
        1, 3, dimension, dtype=dtype, generator=generator)
    path2 = 0.03 * torch.randn(
        1, 3, dimension, dtype=dtype, generator=generator)
    cuda_path1 = path1.cuda()
    cuda_path2 = path2.cuda()

    cpu1 = pysiglib.branched_sig(path1, degree, planar=planar)
    cpu2 = pysiglib.branched_sig(path2, degree, planar=planar)
    cuda1 = pysiglib.branched_sig(cuda_path1, degree, planar=planar)
    cuda2 = pysiglib.branched_sig(cuda_path2, degree, planar=planar)
    weights = torch.randn(cpu1.shape, dtype=dtype, generator=generator)
    cpu_grad = pysiglib.branched_sig_backprop(
        path1, cpu1, weights, degree, planar=planar)
    cuda_grad = pysiglib.branched_sig_backprop(
        cuda_path1, cuda1, weights.cuda(), degree, planar=planar)
    cpu_combine = pysiglib.branched_sig_combine(
        cpu1, cpu2, dimension, degree, planar=planar)
    cuda_combine = pysiglib.branched_sig_combine(
        cuda1, cuda2, dimension, degree, planar=planar)
    cpu_d1, cpu_d2 = pysiglib.branched_sig_combine_backprop(
        weights, cpu1, cpu2, dimension, degree, planar=planar)
    cuda_d1, cuda_d2 = pysiglib.branched_sig_combine_backprop(
        weights.cuda(), cuda1, cuda2, dimension, degree, planar=planar)
    cpu_coef = pysiglib.branched_sig_coef(path1, basis, planar=planar)
    cuda_coef = pysiglib.branched_sig_coef(
        cuda_path1, basis, planar=planar)
    cpu_coef_grad = pysiglib.branched_sig_coef_backprop(
        path1, basis, cpu_coef, weights, planar=planar)
    cuda_coef_grad = pysiglib.branched_sig_coef_backprop(
        cuda_path1, basis, cuda_coef, weights.cuda(), planar=planar)

    for actual, expected in (
        (cuda1, cpu1), (cuda_grad, cpu_grad),
        (cuda_combine, cpu_combine), (cuda_d1, cpu_d1), (cuda_d2, cpu_d2),
        (cuda_coef, cpu_coef), (cuda_coef_grad, cpu_coef_grad),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-4, atol=5e-4)


@skip_no_cuda
def test_branched_correction_rolling_reverse_matches_finite_difference():
    dimension, degree = 1, 4
    planar = True
    basis = list(pysiglib.trees(dimension, degree, planar=planar)[1:])
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar)
    pysiglib.prepare_branched_sig_coef(dimension, basis, planar=planar)
    rng = np.random.default_rng(20260603)
    path = rng.normal(scale=0.05, size=(3, 1))
    correction = rng.normal(scale=0.01, size=(2, 3))
    weights = rng.normal(size=pysiglib.branched_sig_length(
        dimension, degree, planar=planar))
    epsilon = 1e-6

    cuda_path = torch.tensor(path, dtype=torch.float64, device="cuda")
    cuda_correction = torch.tensor(
        correction, dtype=torch.float64, device="cuda")
    cuda_sig = pysiglib.branched_sig(
        cuda_path, degree, planar=planar, correction=cuda_correction)
    cuda_grad = pysiglib.branched_sig_backprop(
        cuda_path, cuda_sig,
        torch.tensor(weights, dtype=torch.float64, device="cuda"),
        degree, planar=planar, correction=cuda_correction).cpu().numpy()

    finite_diff = np.zeros_like(path)
    for index in np.ndindex(path.shape):
        plus = path.copy()
        minus = path.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        value_plus = np.dot(
            pysiglib.branched_sig(
                plus, degree, planar=planar, correction=correction),
            weights)
        value_minus = np.dot(
            pysiglib.branched_sig(
                minus, degree, planar=planar, correction=correction),
            weights)
        finite_diff[index] = (value_plus - value_minus) / (2 * epsilon)
    np.testing.assert_allclose(cuda_grad, finite_diff, rtol=1e-7, atol=1e-8)

    cuda_coef = pysiglib.branched_sig_coef(
        cuda_path, basis, planar=planar, correction=cuda_correction)
    cuda_coef_grad = pysiglib.branched_sig_coef_backprop(
        cuda_path, basis, cuda_coef,
        torch.tensor(weights, dtype=torch.float64, device="cuda"),
        planar=planar, correction=cuda_correction)
    np.testing.assert_allclose(
        cuda_coef_grad.cpu().numpy(), finite_diff, rtol=1e-7, atol=1e-8)


@skip_no_cuda
@pytest.mark.parametrize(
    "dimension,dtype",
    [
        (8, torch.float64),
        (14, torch.float32),
        (11, torch.float64),
        (20, torch.float64),
    ],
)
def test_branched_conversion_method_zero_global_fallback(dimension, dtype):
    degree = 3
    pysiglib.prepare_branched_log_sig(dimension, degree, 0)
    generator = torch.Generator().manual_seed(20260604)
    path = 0.02 * torch.randn(
        1, 4, dimension, dtype=dtype, generator=generator)
    bsig = pysiglib.branched_sig(path, degree).requires_grad_()
    cuda_bsig = bsig.cuda().requires_grad_()

    expected = pysiglib.torch_api.branched_sig_to_log_sig(
        bsig, dimension, degree, method=0)
    actual = pysiglib.torch_api.branched_sig_to_log_sig(
        cuda_bsig, dimension, degree, method=0)
    weights = torch.randn(expected.shape, dtype=dtype, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), bsig)[0]
    actual_grad = torch.autograd.grad(
        (actual * weights.cuda()).sum(), cuda_bsig)[0]

    torch.testing.assert_close(
        actual.cpu(), expected, rtol=_tolerance(dtype), atol=_tolerance(dtype))
    torch.testing.assert_close(
        actual_grad.cpu(), expected_grad,
        rtol=_tolerance(dtype), atol=_tolerance(dtype))


@skip_no_cuda
@pytest.mark.parametrize("method", [0, 1, 2])
def test_planar_branched_conversion_global_fallback_all_methods(method):
    dimension, degree = 14, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True)
    generator = torch.Generator().manual_seed(20260605)
    path = 0.01 * torch.randn(
        1, 3, dimension, dtype=torch.float32, generator=generator)
    bsig = pysiglib.branched_sig(path, degree, planar=True).requires_grad_()
    cuda_bsig = bsig.detach().cuda().requires_grad_()
    expected = pysiglib.torch_api.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=method)
    actual = pysiglib.torch_api.branched_sig_to_log_sig(
        cuda_bsig, dimension, degree, planar=True, method=method)
    weights = torch.randn(expected.shape, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), bsig)[0]
    actual_grad = torch.autograd.grad(
        (actual * weights.cuda()).sum(), cuda_bsig)[0]
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(
        actual_grad.cpu(), expected_grad, rtol=5e-4, atol=5e-4)


@skip_no_cuda
def test_full_branched_log_sig_global_fallback_torch_backward():
    dimension, degree = 14, 3
    pysiglib.prepare_branched_log_sig(dimension, degree, 0)
    generator = torch.Generator().manual_seed(20260606)
    path_cpu = (
        0.01 * torch.randn(
            2, 4, dimension, dtype=torch.float32, generator=generator)
    ).requires_grad_()
    path_cuda = path_cpu.detach().cuda().requires_grad_()
    expected = pysiglib.torch_api.branched_log_sig(path_cpu, degree, method=0)
    actual = pysiglib.torch_api.branched_log_sig(path_cuda, degree, method=0)
    weights = torch.randn(expected.shape, generator=generator)
    expected_grad = torch.autograd.grad((expected * weights).sum(), path_cpu)[0]
    actual_grad = torch.autograd.grad(
        (actual * weights.cuda()).sum(), path_cuda)[0]
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(
        actual_grad.cpu(), expected_grad, rtol=5e-4, atol=5e-4)


@skip_no_cuda
def test_cuda_global_fallback_empty_batch_and_batch_shape():
    pysiglib.prepare_branched_sig(1, 7, planar=True)
    empty = torch.empty((2, 0, 3, 1), dtype=torch.float64, device="cuda")
    empty_out = pysiglib.branched_sig(empty, 7, planar=True)
    assert empty_out.shape == (2, 0, 625)

    base = torch.randn(2, 1, 4, 2, dtype=torch.float64)
    path = base[..., ::2]
    assert not path.is_contiguous()
    expected = pysiglib.branched_sig(path, 7, planar=True)
    actual = pysiglib.branched_sig(path.cuda(), 7, planar=True)
    check_close(actual, expected, double_atol=2e-10)


@skip_no_cuda
def test_jax_signature_global_fallback_vjp():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as jax_api

    try:
        gpu = jax.devices("gpu")[0]
    except (IndexError, RuntimeError):
        pytest.skip("JAX CUDA device is unavailable")
    path_np = np.random.default_rng(20260607).normal(
        scale=0.01, size=(3, 30000)).astype(np.float32)
    weights_np = np.random.default_rng(20260608).normal(
        size=(30000,)).astype(np.float32)
    path = jax.device_put(jnp.asarray(path_np), gpu)
    weights = jax.device_put(jnp.asarray(weights_np), gpu)

    value, pullback = jax.vjp(lambda x: jax_api.sig(x, 1), path)
    grad = pullback(weights)[0]

    np.testing.assert_allclose(
        np.asarray(value), path_np[-1] - path_np[0], rtol=1e-6, atol=1e-7)
    expected_grad = np.zeros_like(path_np)
    expected_grad[0] = -weights_np
    expected_grad[-1] = weights_np
    np.testing.assert_allclose(
        np.asarray(grad), expected_grad, rtol=1e-6, atol=1e-7)
