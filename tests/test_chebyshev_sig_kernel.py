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

import math

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as torch_api


METHOD = "chebyshev"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("end_y", [0.8, -0.7, -1.4457964907366968])
def test_chebyshev_one_dimensional_reference(dtype, end_y):
    x = np.linspace(0., 1., 17, dtype=dtype).reshape(-1, 1)
    y = np.linspace(0., end_y, 20, dtype=dtype).reshape(-1, 1)
    product = float(x[-1, 0] - x[0, 0]) * float(y[-1, 0] - y[0, 0])
    expected = math.fsum(product ** k / math.factorial(k) ** 2 for k in range(40))
    result = pysiglib.sig_kernel(x, y, method=METHOD, order=6)
    tolerance = 3e-6 if dtype == np.float32 else 3e-13
    assert isinstance(result, np.ndarray)
    assert result.dtype == dtype
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("order", [(1, 0), (31, 0), (1, 2), (3, 2), 3])
def test_chebyshev_first_cell_neumann_order(order):
    x = np.array([[0.], [1.]], dtype=np.float64)
    y = np.array([[0.], [2.]], dtype=np.float64)
    neumann_order = order if isinstance(order, int) else order[1]
    expected = math.fsum(2. ** k / math.factorial(k) ** 2 for k in range(neumann_order + 1))
    grid = pysiglib.sig_kernel(x, y, method=METHOD, order=order, return_grid=True)
    np.testing.assert_allclose(grid, [[1., 1.], [1., expected]], rtol=2e-15, atol=2e-15)


def test_chebyshev_mixed_path_reference_and_subdivision():
    x = np.array([[0., 0.], [.3, -.2], [.2, .2], [.45, .35]], dtype=np.float64)
    y = np.array([[0., 0.], [.2, .1], [-.15, .15]], dtype=np.float64)
    actual = pysiglib.sig_kernel(x, y, method=METHOD, order=8)
    np.testing.assert_allclose(actual, 0.9868046051301949, rtol=3e-14, atol=3e-14)
    grid = pysiglib.sig_kernel(x, y, method=METHOD, order=(8, 8), return_grid=True)
    np.testing.assert_array_equal(grid[-1, -1], actual)
    x_split = np.empty((2 * len(x) - 1, 2), dtype=x.dtype)
    x_split[::2] = x
    x_split[1::2] = (x[:-1] + x[1:]) / 2
    refined = pysiglib.sig_kernel(x_split, y, method=METHOD, order=8)
    swapped = pysiglib.sig_kernel(y, x, method=METHOD, order=8)
    np.testing.assert_allclose([refined, swapped], actual, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("order", [2, (2, 3), 3, 4])
def test_chebyshev_float32_subdivision_roundoff(device, order):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    dx = torch.tensor([[5, 4, -8], [0, 1, -3], [-6, 3, 4], [2, 1, -8]], dtype=torch.float32) / 16
    dy = torch.tensor([[-8, 3, -4], [2, 1, 3], [-6, 0, -4], [1, 3, -6]], dtype=torch.float32) / 16
    x = torch.cat((torch.zeros((1, 3), dtype=torch.float32), (dx / 64).repeat_interleave(64, dim=0).cumsum(dim=0))).to(device)
    y = torch.cat((torch.zeros((1, 3), dtype=torch.float32), (dy / 64).repeat_interleave(64, dim=0).cumsum(dim=0))).to(device)
    # 80-digit signature-level reference, with tail bound below 5e-41.
    expected = 2.0104290618226504348673452833253600467585
    actual = pysiglib.sig_kernel(x, y, method=METHOD, order=order)
    assert actual.dtype == torch.float32
    assert abs(actual.item() - expected) < 5e-7


@pytest.mark.parametrize("return_grid", [False, True])
@pytest.mark.parametrize("order", [(1, 2), (1, 3), (2, 3), 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chebyshev_cached_state_and_discrete_gradient(return_grid, order, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.tensor([[0., 0.], [.2, -.1], [.35, .15]], dtype=torch.float64, device=device)
    y = torch.tensor([[0., 0.], [-.1, .25], [.2, .3]], dtype=torch.float64, device=device)
    result, state = pysiglib.sig_kernel(
        x, y, method=METHOD, order=order, return_grid=return_grid, _return_state=True)
    degree = order[0] if isinstance(order, tuple) else order
    assert state.shape == (1, 2, 2, 2 * degree + 8)
    weights = torch.arange(1, result.numel() + 1, dtype=x.dtype, device=device).reshape(result.shape) / 10
    cached = pysiglib.sig_kernel_backprop(
        weights, x, y, method=METHOD, order=order, return_grid=return_grid,
        left_deriv=True, right_deriv=True, _state=state)
    recomputed = pysiglib.sig_kernel_backprop(
        weights, x, y, method=METHOD, order=order, return_grid=return_grid,
        left_deriv=True, right_deriv=True)
    for actual, expected in zip(cached, recomputed):
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    x.requires_grad_(True)
    y.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda left, right: torch_api.sig_kernel(
            left, right, method=METHOD, order=order, return_grid=return_grid),
        (x, y), eps=1e-6, atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("order", [2, (2, 3), 3])
@pytest.mark.parametrize("end_y", [1., -1.4457964907366968])
def test_chebyshev_long_linear_reference(device, dtype, order, end_y):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.linspace(0., 1., 1025, dtype=dtype, device=device).reshape(-1, 1)
    y = torch.linspace(0., end_y, 1025, dtype=dtype, device=device).reshape(-1, 1)
    product = y[-1, 0].item()
    reference = math.fsum(product ** k / math.factorial(k) ** 2 for k in range(50))
    actual = pysiglib.sig_kernel(x, y, method=METHOD, order=order).item()
    tolerance = 2e-7 if dtype == torch.float32 else (5e-13 if order == 2 else 2e-14)
    assert abs(actual - reference) < tolerance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("return_grid", [False, True])
@pytest.mark.parametrize("segments, order", [
    ((4, 3), (1, 0)), ((17, 23), (3, 5)), ((33, 37), (4, 4)),
    ((33, 37), (1, 1)), ((33, 37), (2, 2)), ((33, 37), (3, 3)),
    ((33, 37), (1, 2)), ((33, 37), (1, 3)), ((33, 37), (2, 3)),
    ((129, 135), (2, 3)),
    ((5, 7), (5, 5)), ((5, 7), (7, 7)), ((5, 7), (9, 9)),
    ((5, 7), (10, 10)), ((5, 7), (11, 11)),
    ((5, 7), (8, 8)), ((7, 5), (12, 12)),
])
def test_chebyshev_cuda_matches_cpu(dtype, return_grid, segments, order):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    generator = torch.Generator().manual_seed(314)
    x = (torch.randn((3, 3, 2 * segments[0] + 1, 4), generator=generator, dtype=dtype) * .08)[:, :, ::2, ::2]
    y = (torch.randn((3, 3, 2 * segments[1] + 1, 4), generator=generator, dtype=dtype) * .08)[:, :, ::2, ::2]
    x.requires_grad_(True)
    y.requires_grad_(True)
    expected = torch_api.sig_kernel(x, y, method=METHOD, order=order, return_grid=return_grid)
    weights = torch.randn(expected.shape, generator=generator, dtype=dtype)
    if return_grid:
        # Compare gradients of a mean-weighted grid objective across grid sizes.
        weights /= expected.shape[-1] * expected.shape[-2]
    expected_grads = torch.autograd.grad((weights * expected).sum(), (x, y))
    x_cuda = x.detach().cuda().requires_grad_(True)
    y_cuda = y.detach().cuda().requires_grad_(True)
    actual = torch_api.sig_kernel(x_cuda, y_cuda, method=METHOD, order=order, return_grid=return_grid)
    actual_grads = torch.autograd.grad((weights.cuda() * actual).sum(), (x_cuda, y_cuda))
    tolerance = 4e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
    recomputed = pysiglib.sig_kernel_backprop(
        weights.cuda(), x_cuda, y_cuda, method=METHOD, order=order, return_grid=return_grid,
        left_deriv=True, right_deriv=True)
    if dtype == torch.float32 and return_grid:
        x_reference = x.detach().double().requires_grad_(True)
        y_reference = y.detach().double().requires_grad_(True)
        reference = torch_api.sig_kernel(
            x_reference, y_reference, method=METHOD, order=order, return_grid=True)
        reference_grads = torch.autograd.grad(
            (weights.double() * reference).sum(), (x_reference, y_reference))
        for grads in (expected_grads, actual_grads, recomputed):
            for actual_grad, reference_grad in zip(grads, reference_grads):
                torch.testing.assert_close(
                    actual_grad.cpu().double(), reference_grad, rtol=4e-6, atol=1e-8)
        for cached_grad, recomputed_grad in zip(actual_grads, recomputed):
            torch.testing.assert_close(cached_grad, recomputed_grad, rtol=0, atol=0)
    else:
        for grads in (actual_grads, recomputed):
            for actual_grad, expected_grad in zip(grads, expected_grads):
                torch.testing.assert_close(actual_grad.cpu(), expected_grad, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("return_grid", [False, True])
def test_chebyshev_large_batch_forward(dtype, return_grid):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    generator = torch.Generator().manual_seed(1257)
    x = torch.randn((129, 257, 2), generator=generator, dtype=dtype) * .03
    y = torch.randn((129, 257, 2), generator=generator, dtype=dtype) * .03
    expected = pysiglib.sig_kernel(x, y, method=METHOD, order=(2, 3), return_grid=return_grid, n_jobs=1)
    actual = pysiglib.sig_kernel(x.cuda(), y.cuda(), method=METHOD, order=(2, 3), return_grid=return_grid)
    tolerance = 1e-6 if dtype == torch.float32 else 3e-13
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("transform_kwargs", [{}, {"time_aug": True}, {"lead_lag": True}])
def test_chebyshev_transforms_static_kernel_and_gram(transform_kwargs):
    generator = torch.Generator().manual_seed(62)
    x = torch.randn((2, 2, 4, 2), generator=generator, dtype=torch.float64) * .1
    y = torch.randn((3, 5, 2), generator=generator, dtype=torch.float64) * .1
    options = dict(method=METHOD, order=(5, 4), static_kernel=pysiglib.RBFKernel(sigma=1.), **transform_kwargs)
    actual = pysiglib.sig_kernel_gram(x, y, max_batch=1, **options)
    expected = torch.stack([
        pysiglib.sig_kernel(left, right, **options)
        for left in x.reshape(-1, 4, 2) for right in y
    ]).reshape(2, 2, 3)
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)
    normalized = torch_api.sig_kernel_gram(x, x, normalize=True, max_batch=1, **options)
    torch.testing.assert_close(normalized.reshape(4, 4).diagonal(), torch.ones(4, dtype=x.dtype))


@pytest.mark.parametrize("return_grid", [False, True])
def test_chebyshev_symmetric_gram_reverse(return_grid):
    x = torch.tensor([[[0.], [.2], [-.1]], [[0.], [-.1], [.3]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: torch_api.sig_kernel_gram(
            value, value, method=METHOD, order=(3, 2), max_batch=1, return_grid=return_grid),
        (x,), eps=1e-6, atol=3e-8, rtol=3e-6)


def test_chebyshev_transformed_rbf_reverse():
    x = torch.tensor([[0.], [.2], [-.1]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0.], [.15]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda left, right: torch_api.sig_kernel(
            left, right, method=METHOD, order=(4, 3), static_kernel=pysiglib.RBFKernel(sigma=.8),
            time_aug=True, lead_lag=True, end_time=.5, normalize=True),
        (x, y), eps=1e-6, atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize("return_grid", [False, True])
def test_chebyshev_empty_batches_and_constant_paths(return_grid):
    x = torch.empty((2, 0, 3, 2), dtype=torch.float64, requires_grad=True)
    result = torch_api.sig_kernel(x, x, method=METHOD, order=4, return_grid=return_grid)
    assert result.shape == ((2, 0, 3, 3) if return_grid else (2, 0))
    result.sum().backward()
    assert x.grad.shape == x.shape
    gram = torch_api.sig_kernel_gram(x, x, method=METHOD, order=4, return_grid=return_grid)
    assert gram.shape == ((2, 0, 2, 0, 3, 3) if return_grid else (2, 0, 2, 0))
    gram.sum().backward()
    constant = torch.zeros((2, 3, 1, 2), dtype=torch.float64, requires_grad=True)
    value = torch_api.sig_kernel(constant, constant, method=METHOD, order=4, return_grid=return_grid)
    torch.testing.assert_close(value, torch.ones_like(value))
    value.sum().backward()
    torch.testing.assert_close(constant.grad, torch.zeros_like(constant))


@pytest.mark.parametrize("order", [None, 0, -1, 16, True, 2.5, (0, 1), (1, -1), (31, 1), (True, 2), (1, 2, 3), [2, 2]])
def test_chebyshev_invalid_order(order):
    x = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        pysiglib.sig_kernel(x, x, method=METHOD, order=order)


def test_chebyshev_rejects_dyadic_order_and_branched_use():
    x = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="dyadic_order"):
        pysiglib.sig_kernel(x, x, method=METHOD, order=4, dyadic_order=0)
    with pytest.raises(ValueError, match="chebyshev"):
        pysiglib.branched_sig_kernel(x, x, depth=2, method=METHOD, order=4)


@pytest.mark.parametrize("return_grid", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chebyshev_jax_jit_and_reverse(return_grid, dtype):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    api = pytest.importorskip("pysiglib.jax_api")
    x = np.array([[[0., 0.], [.2, -.1], [.35, .15]], [[0., 0.], [-.1, .2], [.1, .3]]], dtype=dtype)
    y = np.array([[[0., 0.], [-.1, .25], [.2, .3]]], dtype=dtype)
    weights = np.array([[.2], [-.3]], dtype=dtype)
    if return_grid:
        weights = weights[..., None, None] * np.arange(9, dtype=dtype).reshape(3, 3) / 10

    def objective(left, right):
        return jnp.sum(api.sig_kernel_gram(
            left, right, method=METHOD, order=(4, 3), max_batch=1, return_grid=return_grid) * jnp.asarray(weights))

    with jax.enable_x64(dtype == np.float64):
        actual, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(jnp.asarray(x), jnp.asarray(y))
    expected = np.sum(pysiglib.sig_kernel_gram(
        x, y, method=METHOD, order=(4, 3), max_batch=1, return_grid=return_grid) * weights)
    expected_grads = pysiglib.sig_kernel_gram_backprop(
        weights, x, y, method=METHOD, order=(4, 3), max_batch=1, return_grid=return_grid,
        left_deriv=True, right_deriv=True)
    tolerance = 3e-6 if dtype == np.float32 else 3e-12
    assert np.asarray(actual).dtype == dtype
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=tolerance, atol=tolerance)
    for gradient, reference in zip(gradients, expected_grads):
        assert np.asarray(gradient).dtype == dtype
        np.testing.assert_allclose(np.asarray(gradient), reference, rtol=7 * tolerance, atol=tolerance)


@pytest.mark.parametrize("api_name", ["base", "torch_api", "jax_api"])
def test_chebyshev_metrics_forward_orders(api_name):
    x = np.array([[[0.], [.2], [-.1]], [[0.], [-.1], [.3]], [[0.], [.4], [.2]]], dtype=np.float32)
    y = np.array([[[0.], [-.2], [.1]], [[0.], [.1], [.4]]], dtype=np.float32)
    options = dict(method=METHOD, order=(4, 3), max_batch=1)
    xx = pysiglib.sig_kernel_gram(x, x, **options)
    xy = pysiglib.sig_kernel_gram(x, y, **options)
    yy = pysiglib.sig_kernel_gram(y, y, **options)
    xx_mean = (np.sum(xx) - np.trace(xx)) / 6
    yy_mean = (np.sum(yy) - np.trace(yy)) / 2
    expected_score = xx_mean - 2 * np.mean(xy, axis=0)
    if api_name == "base":
        api, left, right = pysiglib, x, y
    elif api_name == "torch_api":
        api, left, right = torch_api, torch.from_numpy(x), torch.from_numpy(y)
    else:
        jnp = pytest.importorskip("jax.numpy")
        api = pytest.importorskip("pysiglib.jax_api")
        left, right = jnp.asarray(x), jnp.asarray(y)
    np.testing.assert_allclose(np.asarray(api.sig_score(left, right, **options)), expected_score, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(np.asarray(api.expected_sig_score(left, right, **options)), np.mean(expected_score), atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(np.asarray(api.sig_mmd(left, right, **options)), xx_mean - 2 * np.mean(xy) + yy_mean, atol=3e-6, rtol=3e-6)
