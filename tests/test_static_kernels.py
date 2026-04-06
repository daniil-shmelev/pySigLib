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

from copy import deepcopy
import pytest
import numpy as np
import torch

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, atol=1e-1)

# =========================================================================
# Finite difference helpers
# =========================================================================

def finite_difference(x1, x2, dyadic_order, kernel):
    """Numerically differentiate sig_kernel w.r.t. x1 using finite differences."""
    x1 = x1.to(device="cpu", dtype=torch.double)
    x2 = x2.to(device="cpu", dtype=torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-10
    k = pysiglib.sig_kernel(x1, x2, dyadic_order, static_kernel=kernel)
    out = np.empty(shape=(batch_size, length, dim))

    for i in range(length):
        for d in range(dim):
            x1_d = deepcopy(x1)
            x1_d[:, i, d] += eps
            k_d = pysiglib.sig_kernel(x1_d, x2, dyadic_order, static_kernel=kernel)
            out[:, i, d] = (k_d - k) / eps
    return out


def finite_difference_gram(x1, x2, dyadic_order, kernel):
    """Numerically differentiate sig_kernel_gram w.r.t. x1 using finite differences."""
    x1 = x1.to(device="cpu", dtype=torch.double)
    x2 = x2.to(device="cpu", dtype=torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-10
    k = pysiglib.sig_kernel_gram(x1, x2, dyadic_order, static_kernel=kernel)
    out = np.empty(shape=(batch_size, length, dim))

    for i in range(length):
        for d in range(dim):
            x1_d = deepcopy(x1)
            x1_d[:, i, d] += eps
            k_d = pysiglib.sig_kernel_gram(x1_d, x2, dyadic_order, static_kernel=kernel)
            out[:, i, d] = ((k_d - k) / eps).sum(1)
    return out

# =========================================================================
# PolynomialKernel
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_polynomial_kernel_forward_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.PolynomialKernel(degree=2., gamma=1., scale=1.)
    k = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(k, device)
    assert k.shape == (batch,)
    assert torch.all(torch.isfinite(k))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_polynomial_kernel_gram_batch(device, dyadic_order):
    batch1, batch2, len1, len2, dim = 4, 3, 10, 10, 3
    X = torch.rand(size=(batch1, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch2, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.PolynomialKernel(degree=2., gamma=1., scale=1.)
    gram = pysiglib.sig_kernel_gram(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(gram, device)
    assert gram.shape == (batch1, batch2)
    assert torch.all(torch.isfinite(gram))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_polynomial_kernel_backprop_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.PolynomialKernel(degree=2., gamma=1., scale=1.)

    d1 = finite_difference(X, Y, dyadic_order, kernel=kernel)
    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    d3, d4 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=True, right_deriv=True, static_kernel=kernel
    )
    assert_device(d3, device)
    assert_device(d4, device)

    check_close(d1, d3)
    check_close(d2, d4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_polynomial_kernel_backprop_grad_y_without_grad_x(device, dyadic_order):
    """Verify grad_y is correct even when grad_x is not requested (independent code path)."""
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.PolynomialKernel(degree=2., gamma=1., scale=1.)

    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    _, d5 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=False, right_deriv=True, static_kernel=kernel
    )
    assert_device(d5, device)
    check_close(d2, d5)


# =========================================================================
# Matern12Kernel
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern12_kernel_forward_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.Matern12Kernel(sigma=1.0)
    k = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(k, device)
    assert k.shape == (batch,)
    assert torch.all(torch.isfinite(k))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern12_kernel_gram_batch(device, dyadic_order):
    batch1, batch2, len1, len2, dim = 4, 3, 10, 10, 3
    X = torch.rand(size=(batch1, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch2, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.Matern12Kernel(sigma=1.0)
    gram = pysiglib.sig_kernel_gram(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(gram, device)
    assert gram.shape == (batch1, batch2)
    assert torch.all(torch.isfinite(gram))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern12_kernel_backprop_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.Matern12Kernel(sigma=1.0)

    d1 = finite_difference(X, Y, dyadic_order, kernel=kernel)
    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    d3, d4 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=True, right_deriv=True, static_kernel=kernel
    )
    assert_device(d3, device)
    assert_device(d4, device)

    check_close(d1, d3)
    check_close(d2, d4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern12_kernel_backprop_grad_y_without_grad_x(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.Matern12Kernel(sigma=1.0)

    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    _, d5 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=False, right_deriv=True, static_kernel=kernel
    )
    assert_device(d5, device)
    check_close(d2, d5)


# =========================================================================
# Matern32Kernel
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern32_kernel_forward_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.Matern32Kernel(sigma=1.0)
    k = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(k, device)
    assert k.shape == (batch,)
    assert torch.all(torch.isfinite(k))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern32_kernel_gram_batch(device, dyadic_order):
    batch1, batch2, len1, len2, dim = 4, 3, 10, 10, 3
    X = torch.rand(size=(batch1, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch2, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.Matern32Kernel(sigma=1.0)
    gram = pysiglib.sig_kernel_gram(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(gram, device)
    assert gram.shape == (batch1, batch2)
    assert torch.all(torch.isfinite(gram))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern32_kernel_backprop_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.Matern32Kernel(sigma=1.0)

    d1 = finite_difference(X, Y, dyadic_order, kernel=kernel)
    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    d3, d4 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=True, right_deriv=True, static_kernel=kernel
    )
    assert_device(d3, device)
    assert_device(d4, device)

    check_close(d1, d3)
    check_close(d2, d4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern32_kernel_backprop_grad_y_without_grad_x(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.Matern32Kernel(sigma=1.0)

    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    _, d5 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=False, right_deriv=True, static_kernel=kernel
    )
    assert_device(d5, device)
    check_close(d2, d5)


# =========================================================================
# Matern52Kernel
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern52_kernel_forward_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.Matern52Kernel(sigma=1.0)
    k = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(k, device)
    assert k.shape == (batch,)
    assert torch.all(torch.isfinite(k))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern52_kernel_gram_batch(device, dyadic_order):
    batch1, batch2, len1, len2, dim = 4, 3, 10, 10, 3
    X = torch.rand(size=(batch1, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch2, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.Matern52Kernel(sigma=1.0)
    gram = pysiglib.sig_kernel_gram(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(gram, device)
    assert gram.shape == (batch1, batch2)
    assert torch.all(torch.isfinite(gram))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern52_kernel_backprop_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.Matern52Kernel(sigma=1.0)

    d1 = finite_difference(X, Y, dyadic_order, kernel=kernel)
    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    d3, d4 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=True, right_deriv=True, static_kernel=kernel
    )
    assert_device(d3, device)
    assert_device(d4, device)

    check_close(d1, d3)
    check_close(d2, d4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_matern52_kernel_backprop_grad_y_without_grad_x(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.Matern52Kernel(sigma=1.0)

    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    _, d5 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=False, right_deriv=True, static_kernel=kernel
    )
    assert_device(d5, device)
    check_close(d2, d5)


# =========================================================================
# RationalQuadraticKernel
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_rational_quadratic_kernel_forward_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.RationalQuadraticKernel(sigma=1.0, alpha=1.0)
    k = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(k, device)
    assert k.shape == (batch,)
    assert torch.all(torch.isfinite(k))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_rational_quadratic_kernel_gram_batch(device, dyadic_order):
    batch1, batch2, len1, len2, dim = 4, 3, 10, 10, 3
    X = torch.rand(size=(batch1, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch2, len2, dim), device=device, dtype=torch.double) / 2

    kernel = pysiglib.RationalQuadraticKernel(sigma=1.0, alpha=1.0)
    gram = pysiglib.sig_kernel_gram(X, Y, dyadic_order, static_kernel=kernel)
    assert_device(gram, device)
    assert gram.shape == (batch1, batch2)
    assert torch.all(torch.isfinite(gram))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_rational_quadratic_kernel_backprop_batch(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.RationalQuadraticKernel(sigma=1.0, alpha=1.0)

    d1 = finite_difference(X, Y, dyadic_order, kernel=kernel)
    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    d3, d4 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=True, right_deriv=True, static_kernel=kernel
    )
    assert_device(d3, device)
    assert_device(d4, device)

    check_close(d1, d3)
    check_close(d2, d4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_rational_quadratic_kernel_backprop_grad_y_without_grad_x(device, dyadic_order):
    batch, len1, len2, dim = 4, 10, 10, 3
    X = torch.rand(size=(batch, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones(batch, device=device, dtype=torch.double)

    kernel = pysiglib.RationalQuadraticKernel(sigma=1.0, alpha=1.0)

    d2 = finite_difference(Y, X, dyadic_order, kernel=kernel)
    _, d5 = pysiglib.sig_kernel_backprop(
        derivs, X, Y, dyadic_order, left_deriv=False, right_deriv=True, static_kernel=kernel
    )
    assert_device(d5, device)
    check_close(d2, d5)


# =========================================================================
# Cross-kernel: sig_kernel(X, X) should be positive for all static kernels
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel_factory", [
    lambda: pysiglib.PolynomialKernel(degree=2., gamma=1., scale=1.),
    lambda: pysiglib.Matern12Kernel(sigma=1.0),
    lambda: pysiglib.Matern32Kernel(sigma=1.0),
    lambda: pysiglib.Matern52Kernel(sigma=1.0),
    lambda: pysiglib.RationalQuadraticKernel(sigma=1.0, alpha=1.0),
], ids=["polynomial", "matern12", "matern32", "matern52", "rational_quadratic"])
def test_sig_kernel_self_positive(device, kernel_factory):
    """sig_kernel(X, X) should be positive for positive-definite static kernels."""
    batch, length, dim = 4, 10, 3
    X = torch.rand(size=(batch, length, dim), device=device, dtype=torch.double) / 2

    kernel = kernel_factory()
    k = pysiglib.sig_kernel(X, X, 0, static_kernel=kernel)
    assert_device(k, device)
    assert torch.all(k > 0), f"Expected all positive, got {k}"


# =========================================================================
# Cross-kernel: gram backprop finite-difference check
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel_factory,kernel_name", [
    (lambda: pysiglib.PolynomialKernel(degree=2., gamma=1., scale=1.), "polynomial"),
    (lambda: pysiglib.Matern12Kernel(sigma=1.0), "matern12"),
    (lambda: pysiglib.Matern32Kernel(sigma=1.0), "matern32"),
    (lambda: pysiglib.Matern52Kernel(sigma=1.0), "matern52"),
    (lambda: pysiglib.RationalQuadraticKernel(sigma=1.0, alpha=1.0), "rational_quadratic"),
])
def test_sig_kernel_gram_backprop_static_kernels(device, kernel_factory, kernel_name):
    """Finite-difference check on sig_kernel_gram_backprop for each static kernel."""
    batch1, batch2, len1, len2, dim = 4, 3, 10, 10, 3
    X = torch.rand(size=(batch1, len1, dim), device=device, dtype=torch.double) / 2
    Y = torch.rand(size=(batch2, len2, dim), device=device, dtype=torch.double) / 2
    derivs = torch.ones((batch1, batch2), device=device, dtype=torch.double)

    kernel = kernel_factory()

    d1 = finite_difference_gram(X, Y, 0, kernel=kernel)
    d2 = finite_difference_gram(Y, X, 0, kernel=kernel)
    d3, d4 = pysiglib.sig_kernel_gram_backprop(
        derivs, X, Y, 0, left_deriv=True, right_deriv=True, static_kernel=kernel
    )
    assert_device(d3, device)
    assert_device(d4, device)

    check_close(d1, d3)
    check_close(d2, d4)
