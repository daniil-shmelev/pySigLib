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

import pysiglib


METHOD = "polynomial"


class DoubleLinearKernel(pysiglib.StaticKernel):
    def __call__(self, ctx, x, y):
        return 2 * torch.bmm(torch.diff(x, dim=1), torch.diff(y, dim=1).transpose(1, 2))

    def grad_x(self, ctx, derivs):
        raise NotImplementedError

    def grad_y(self, ctx, derivs):
        raise NotImplementedError


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polynomial_sig_kernel_golden(dtype):
    x = np.array([[0., 0.], [.3, -.2], [.2, .2], [.45, .35]], dtype=dtype)
    y = np.array([[0., 0.], [.2, .1], [-.15, .15]], dtype=dtype)
    expected = 0.9868046051301949

    result = pysiglib.sig_kernel(x, y, method=METHOD, order=7)

    assert isinstance(result, np.ndarray)
    assert result.dtype == dtype
    assert np.allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_polynomial_sig_kernel_order_is_highest_degree():
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    y = torch.tensor([[0.], [2.]], dtype=torch.float64)

    result = pysiglib.sig_kernel(x, y, method=METHOD, order=2)

    assert torch.equal(result, torch.tensor(4., dtype=torch.float64))


def test_polynomial_sig_kernel_custom_static_kernel():
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    y = torch.tensor([[0.], [1.]], dtype=torch.float64)

    result = pysiglib.sig_kernel(
        x, y, method=METHOD, order=2, static_kernel=DoubleLinearKernel())

    assert torch.equal(result, torch.tensor(4., dtype=torch.float64))


def test_polynomial_sig_kernel_normalize():
    x = torch.tensor([[0., 0.], [.2, .1], [.3, -.1]], dtype=torch.float64)
    y = torch.tensor([[0., 0.], [-.1, .2], [.15, .25]], dtype=torch.float64)
    kxy = pysiglib.sig_kernel(x, y, method=METHOD, order=7)
    kxx = pysiglib.sig_kernel(x, x, method=METHOD, order=7)
    kyy = pysiglib.sig_kernel(y, y, method=METHOD, order=7)

    normalized = pysiglib.sig_kernel(x, y, method=METHOD, order=7, normalize=True)

    assert torch.allclose(normalized, kxy / torch.sqrt(kxx * kyy), rtol=1e-13, atol=1e-13)


def test_polynomial_sig_kernel_gram_and_batching():
    x = torch.tensor([
        [[0., 0.], [.2, .1], [.3, -.1]],
        [[0., 0.], [-.1, .3], [.2, .2]],
    ], dtype=torch.float64)
    y = torch.tensor([
        [[0., 0.], [.1, -.2], [.2, .1]],
        [[0., 0.], [.3, .1], [.1, .2]],
        [[0., 0.], [-.2, .1], [.2, -.1]],
    ], dtype=torch.float64)

    result = pysiglib.sig_kernel_gram(x, y, method=METHOD, order=7, max_batch=1, n_jobs=2)
    expected = torch.stack([
        torch.stack([pysiglib.sig_kernel(xi, yj, method=METHOD, order=7) for yj in y])
        for xi in x
    ])

    assert torch.allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("transform_kwargs", [
    {"time_aug": True},
    {"lead_lag": True},
    {"time_aug": True, "lead_lag": True},
])
def test_polynomial_sig_kernel_path_transforms(transform_kwargs):
    x = torch.tensor([[0., 0.], [.2, .1], [.3, -.1]], dtype=torch.float64)
    y = torch.tensor([[0., 0.], [-.1, .2], [.15, .25]], dtype=torch.float64)
    tx = pysiglib.transform_path(x, **transform_kwargs)
    ty = pysiglib.transform_path(y, **transform_kwargs)

    transformed = pysiglib.sig_kernel(x, y, method=METHOD, order=7, **transform_kwargs)
    explicit = pysiglib.sig_kernel(tx, ty, method=METHOD, order=7)

    assert torch.allclose(transformed, explicit, rtol=1e-13, atol=1e-13)


def test_polynomial_sig_kernel_gram_normalize():
    x = torch.tensor([
        [[0., 0.], [.2, .1], [.3, -.1]],
        [[0., 0.], [-.1, .3], [.2, .2]],
    ], dtype=torch.float64)

    result = pysiglib.sig_kernel_gram(x, x, method=METHOD, order=7, normalize=True)

    assert torch.allclose(torch.diagonal(result), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(result, result.T, rtol=1e-13, atol=1e-13)


def test_polynomial_sig_kernel_grid_ends_at_scalar():
    x = np.array([[0., 0.], [.3, -.2], [.2, .2], [.45, .35]], dtype=np.float64)
    y = np.array([[0., 0.], [.2, .1], [-.15, .15]], dtype=np.float64)
    grid = pysiglib.sig_kernel(x, y, method=METHOD, order=7, return_grid=True)
    scalar = pysiglib.sig_kernel(x, y, method=METHOD, order=7)

    assert grid.shape == (4, 3)
    assert np.all(grid[0] == 1)
    assert np.all(grid[:, 0] == 1)
    assert grid[-1, -1] == scalar


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polynomial_sig_kernel_cuda_matches_cpu(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    generator = torch.Generator().manual_seed(25022025)
    x = torch.randn((2, 35, 3), generator=generator, dtype=dtype) * 0.03
    y = torch.randn((2, 67, 3), generator=generator, dtype=dtype) * 0.03

    cpu_scalar = pysiglib.sig_kernel(x, y, method=METHOD, order=7)
    cuda_scalar = pysiglib.sig_kernel(
        x.cuda(), y.cuda(), method=METHOD, order=7).cpu()
    cpu_grid = pysiglib.sig_kernel(x, y, method=METHOD, order=7, return_grid=True)
    cuda_grid = pysiglib.sig_kernel(
        x.cuda(), y.cuda(), method=METHOD, order=7, return_grid=True).cpu()

    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    assert torch.allclose(cuda_scalar, cpu_scalar, rtol=tolerance, atol=tolerance)
    assert torch.allclose(cuda_grid, cpu_grid, rtol=tolerance, atol=tolerance)
    assert torch.equal(cuda_grid[:, 0], torch.ones_like(cuda_grid[:, 0]))
    assert torch.equal(cuda_grid[:, :, 0], torch.ones_like(cuda_grid[:, :, 0]))
    assert torch.equal(cuda_scalar, cuda_grid[:, -1, -1])


def test_polynomial_sig_kernel_cuda_composition():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.tensor([
        [[0., 0.], [.2, .1], [.3, -.1]],
        [[0., 0.], [-.1, .3], [.2, .2]],
    ], dtype=torch.float64)
    y = torch.tensor([
        [[0., 0.], [.1, -.2], [.2, .1]],
        [[0., 0.], [.3, .1], [.1, .2]],
        [[0., 0.], [-.2, .1], [.2, -.1]],
    ], dtype=torch.float64)
    kwargs = {
        "method": METHOD,
        "order": 7,
        "static_kernel": DoubleLinearKernel(),
        "time_aug": True,
        "lead_lag": True,
    }

    cpu = pysiglib.sig_kernel_gram(x, y, max_batch=1, **kwargs)
    cuda = pysiglib.sig_kernel_gram(x.cuda(), y.cuda(), max_batch=1, **kwargs).cpu()
    cpu_normalized = pysiglib.sig_kernel_gram(x, y, max_batch=1, normalize=True, **kwargs)
    cuda_normalized = pysiglib.sig_kernel_gram(
        x.cuda(), y.cuda(), max_batch=1, normalize=True, **kwargs).cpu()

    assert torch.allclose(cuda, cpu, rtol=2e-12, atol=2e-12)
    assert torch.allclose(cuda_normalized, cpu_normalized, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("return_grid", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polynomial_sig_kernel_cuda_backprop_matches_cpu(dtype, return_grid):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    generator = torch.Generator().manual_seed(25022025)
    x = torch.randn((2, 6, 3), generator=generator, dtype=dtype) * 0.04
    y = torch.randn((2, 8, 3), generator=generator, dtype=dtype) * 0.04
    if return_grid:
        derivs = torch.randn((2, 6, 8), generator=generator, dtype=dtype)
    else:
        derivs = torch.randn((2,), generator=generator, dtype=dtype)

    expected = pysiglib.sig_kernel_backprop(
        derivs, x, y, method=METHOD, order=7,
        left_deriv=True, right_deriv=True, return_grid=return_grid)
    actual = pysiglib.sig_kernel_backprop(
        derivs.cuda(), x.cuda(), y.cuda(), method=METHOD, order=7,
        left_deriv=True, right_deriv=True, return_grid=return_grid)

    tolerance = 5e-5 if dtype == torch.float32 else 4e-12
    assert torch.allclose(actual[0].cpu(), expected[0], rtol=tolerance, atol=tolerance)
    assert torch.allclose(actual[1].cpu(), expected[1], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("return_grid", [False, True])
def test_torch_polynomial_sig_kernel_cuda_autograd(return_grid):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x_cpu = torch.tensor(
        [[0., 0.], [.2, -.1], [.35, .15]], dtype=torch.float64,
        requires_grad=True)
    y_cpu = torch.tensor(
        [[0., 0.], [-.1, .25], [.2, .3], [.3, .1]], dtype=torch.float64,
        requires_grad=True)
    x_cuda = x_cpu.detach().cuda().requires_grad_()
    y_cuda = y_cpu.detach().cuda().requires_grad_()
    if return_grid:
        weights_cpu = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10
    else:
        weights_cpu = torch.tensor(0.7, dtype=torch.float64)

    expected = pysiglib.torch_api.sig_kernel(
        x_cpu, y_cpu, method=METHOD, order=7, return_grid=return_grid)
    expected_derivs = torch.autograd.grad(
        torch.sum(expected * weights_cpu), (x_cpu, y_cpu))
    actual = pysiglib.torch_api.sig_kernel(
        x_cuda, y_cuda, method=METHOD, order=7, return_grid=return_grid)
    actual_derivs = torch.autograd.grad(
        torch.sum(actual * weights_cpu.cuda()), (x_cuda, y_cuda))

    assert torch.allclose(actual.cpu(), expected, rtol=3e-12, atol=3e-12)
    assert torch.allclose(
        actual_derivs[0].cpu(), expected_derivs[0], rtol=5e-12, atol=5e-12)
    assert torch.allclose(
        actual_derivs[1].cpu(), expected_derivs[1], rtol=5e-12, atol=5e-12)


def test_polynomial_sig_kernel_cuda_backprop_generic_order():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    generator = torch.Generator().manual_seed(25022025)
    x = torch.randn((1, 4, 2), generator=generator, dtype=torch.float64) * 0.01
    y = torch.randn((1, 5, 2), generator=generator, dtype=torch.float64) * 0.01
    derivs = torch.tensor([0.75], dtype=torch.float64)

    expected = pysiglib.sig_kernel_backprop(
        derivs, x, y, method=METHOD, order=64,
        left_deriv=True, right_deriv=True)
    actual = pysiglib.sig_kernel_backprop(
        derivs.cuda(), x.cuda(), y.cuda(), method=METHOD, order=64,
        left_deriv=True, right_deriv=True)

    assert torch.allclose(actual[0].cpu(), expected[0], rtol=2e-11, atol=2e-11)
    assert torch.allclose(actual[1].cpu(), expected[1], rtol=2e-11, atol=2e-11)


def test_polynomial_sig_kernel_converges_to_direct_signature():
    rng = np.random.default_rng(25022025)
    increments = rng.normal(scale=0.04, size=(7, 2))
    x = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
    y = np.vstack([np.zeros(2), np.cumsum(increments[::-1], axis=0)])
    sig_x = pysiglib.sig(x, 12, scalar_term=True)
    sig_y = pysiglib.sig(y, 12, scalar_term=True)
    reference = float(np.dot(sig_x, sig_y))

    errors = [
        abs(float(pysiglib.sig_kernel(x, y, method=METHOD, order=order)) - reference)
        for order in (2, 5, 8, 12)
    ]

    assert errors[-1] < errors[0]
    assert all(next_error <= error + 1e-14 for error, next_error in zip(errors, errors[1:]))


@pytest.mark.parametrize("kwargs", [
    {},
    {"method": "invalid", "dyadic_order": 0},
    {"method": "finite_difference", "dyadic_order": 0, "order": 2},
    {"method": "polynomial", "order": 1},
    {"method": "polynomial", "order": 65},
])
def test_sig_kernel_method_validation(kwargs):
    x = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        pysiglib.sig_kernel(x, x, **kwargs)


def test_polynomial_sig_kernel_explicit_grid_backprop():
    x = np.array([[0., 0.], [.2, -.1], [.35, .15]], dtype=np.float64)
    y = np.array([[0., 0.], [-.1, .25], [.2, .3]], dtype=np.float64)
    weights = np.arange(9, dtype=np.float64).reshape(3, 3) / 10
    dx, dy = pysiglib.sig_kernel_backprop(
        weights, x, y, method=METHOD, order=7,
        left_deriv=True, right_deriv=True, return_grid=True)

    epsilon = 1e-6
    expected_dx = np.empty_like(x)
    expected_dy = np.empty_like(y)
    for path, expected in ((x, expected_dx), (y, expected_dy)):
        for index in np.ndindex(path.shape):
            plus = path.copy()
            minus = path.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            if path is x:
                value_plus = pysiglib.sig_kernel(
                    plus, y, method=METHOD, order=7, return_grid=True)
                value_minus = pysiglib.sig_kernel(
                    minus, y, method=METHOD, order=7, return_grid=True)
            else:
                value_plus = pysiglib.sig_kernel(
                    x, plus, method=METHOD, order=7, return_grid=True)
                value_minus = pysiglib.sig_kernel(
                    x, minus, method=METHOD, order=7, return_grid=True)
            expected[index] = np.sum(weights * (value_plus - value_minus)) / (2 * epsilon)

    assert np.allclose(dx, expected_dx, rtol=2e-8, atol=2e-9)
    assert np.allclose(dy, expected_dy, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("api_name", ["torch_api", "jax_api"])
def test_reverse_apis_support_polynomial_grid(api_name):
    api = pytest.importorskip("pysiglib." + api_name)
    x_np = np.array([[0., 0.], [.2, -.1], [.35, .15]], dtype=np.float32)
    y_np = np.array([[0., 0.], [-.1, .25], [.2, .3]], dtype=np.float32)
    weights_np = np.arange(9, dtype=np.float32).reshape(3, 3) / 10

    if api_name == "torch_api":
        x = torch.tensor(x_np, requires_grad=True)
        y = torch.tensor(y_np)
        weights = torch.tensor(weights_np)
        grid = api.sig_kernel(
            x, y, method=METHOD, order=7, return_grid=True)
        grad, = torch.autograd.grad(torch.sum(grid * weights), (x,))
        grid_np = grid.detach().numpy()
        grad_np = grad.detach().numpy()
    else:
        import jax
        import jax.numpy as jnp
        x = jnp.asarray(x_np)
        y = jnp.asarray(y_np)
        weights = jnp.asarray(weights_np)

        def objective(value):
            return jnp.sum(api.sig_kernel(
                value, y, method=METHOD, order=7, return_grid=True) * weights)

        grid_np = np.asarray(api.sig_kernel(
            x, y, method=METHOD, order=7, return_grid=True))
        grad_np = np.asarray(jax.grad(objective)(x))

    expected_grid = pysiglib.sig_kernel(
        x_np, y_np, method=METHOD, order=7, return_grid=True)
    expected_grad, _ = pysiglib.sig_kernel_backprop(
        weights_np, x_np, y_np, method=METHOD, order=7,
        left_deriv=True, right_deriv=False, return_grid=True)
    assert np.allclose(grid_np, expected_grid, rtol=2e-5, atol=2e-6)
    assert np.allclose(grad_np, expected_grad, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("api_name", ["torch_api", "jax_api"])
def test_reverse_apis_support_polynomial_gram(api_name):
    api = pytest.importorskip("pysiglib." + api_name)
    x_np = np.array([
        [[0., 0.], [.2, -.1], [.35, .15]],
        [[0., 0.], [-.1, .2], [.1, .3]],
    ], dtype=np.float32)
    y_np = np.array([
        [[0., 0.], [-.1, .25], [.2, .3]],
        [[0., 0.], [.15, .1], [.25, -.2]],
    ], dtype=np.float32)
    weights_np = np.array([[.2, -.3], [.5, .7]], dtype=np.float32)

    if api_name == "torch_api":
        x = torch.tensor(x_np, requires_grad=True)
        y = torch.tensor(y_np, requires_grad=True)
        gram = api.sig_kernel_gram(
            x, y, method=METHOD, order=7, max_batch=1)
        dx, dy = torch.autograd.grad(
            torch.sum(gram * torch.tensor(weights_np)), (x, y))
        dx_np = dx.detach().numpy()
        dy_np = dy.detach().numpy()
    else:
        import jax
        import jax.numpy as jnp
        x = jnp.asarray(x_np)
        y = jnp.asarray(y_np)
        weights = jnp.asarray(weights_np)

        def objective(left, right):
            return jnp.sum(api.sig_kernel_gram(
                left, right, method=METHOD, order=7, max_batch=1) * weights)

        dx, dy = jax.grad(objective, argnums=(0, 1))(x, y)
        dx_np = np.asarray(dx)
        dy_np = np.asarray(dy)

    expected_dx, expected_dy = pysiglib.sig_kernel_gram_backprop(
        weights_np, x_np, y_np, method=METHOD, order=7,
        left_deriv=True, right_deriv=True, max_batch=1)
    assert np.allclose(dx_np, expected_dx, rtol=3e-4, atol=3e-5)
    assert np.allclose(dy_np, expected_dy, rtol=3e-4, atol=3e-5)
