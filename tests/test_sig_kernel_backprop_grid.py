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

SINGLE_EPSILON = 1e-2
DOUBLE_EPSILON = 1e-2

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    EPSILON = SINGLE_EPSILON if a_.dtype == np.float32 else DOUBLE_EPSILON
    assert not np.any(np.abs(a_ - b_) > EPSILON)


################################################
## Consistency with return_grid=False
################################################

@pytest.mark.parametrize(("len1", "len2"), [(10, 10), (10, 5), (5, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_grid_backprop_consistency_with_scalar(len1, len2, dyadic_order):
    """
    When derivs_grid has 1.0 only at [-1,-1] (and 0 elsewhere), return_grid=True backprop
    should produce the same result as return_grid=False with deriv=1.0.
    """
    X = torch.rand(size=(len1, 3), dtype=torch.float64)
    Y = torch.rand(size=(len2, 3), dtype=torch.float64)

    # Scalar backprop
    derivs_scalar = torch.ones(1, dtype=torch.float64)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order, left_deriv=True, right_deriv=True
    )

    # Grid backprop: construct derivs_grid with 1.0 at [-1,-1]
    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid)
    derivs_grid[-1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)


@pytest.mark.parametrize("dyadic_order", range(3))
def test_grid_backprop_consistency_batch(dyadic_order):
    X = torch.rand(size=(8, 10, 3), dtype=torch.float64)
    Y = torch.rand(size=(8, 10, 3), dtype=torch.float64)

    derivs_scalar = torch.ones(8, dtype=torch.float64)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order, left_deriv=True, right_deriv=True
    )

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid)
    derivs_grid[:, -1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)


################################################
## Zero and linearity tests
################################################

def test_grid_backprop_zero_derivs():
    """Grid backprop with zero derivs should produce zero path derivatives."""
    X = torch.rand(size=(5, 3), dtype=torch.float64)
    Y = torch.rand(size=(5, 3), dtype=torch.float64)

    k_grid = pysiglib.sig_kernel(X, Y, 0, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid)
    d1, d2 = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, 0, left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1, torch.zeros_like(d1))
    check_close(d2, torch.zeros_like(d2))


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_linearity(dyadic_order):
    """backprop(a*d1 + b*d2) == a*backprop(d1) + b*backprop(d2)."""
    X = torch.rand(size=(5, 3), dtype=torch.float64)
    Y = torch.rand(size=(7, 3), dtype=torch.float64)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, return_grid=True)

    derivs1 = torch.rand_like(k_grid)
    derivs2 = torch.rand_like(k_grid)
    a, b = 2.5, -1.3

    d1_a, d2_a = pysiglib.sig_kernel_backprop(
        derivs1, X, Y, dyadic_order, left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )
    d1_b, d2_b = pysiglib.sig_kernel_backprop(
        derivs2, X, Y, dyadic_order, left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )
    d1_c, d2_c = pysiglib.sig_kernel_backprop(
        a * derivs1 + b * derivs2, X, Y, dyadic_order,
        left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1_c, a * d1_a + b * d1_b)
    check_close(d2_c, a * d2_a + b * d2_b)


################################################
## Finite difference tests
################################################

def _finite_difference_grid(x1, x2, derivs_grid, dyadic_order, time_aug=False, lead_lag=False, static_kernel=None):
    """
    Numerically compute d/dx1 [ sum_{i,j} derivs_grid[i,j] * k_grid[i,j](x1, x2) ]
    using finite differences.
    """
    x1 = x1.to(device="cpu", dtype=torch.double)
    x2 = x2.to(device="cpu", dtype=torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
        derivs_grid = derivs_grid[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-7
    k_grid = pysiglib.sig_kernel(x1, x2, dyadic_order, time_aug=time_aug, lead_lag=lead_lag, static_kernel=static_kernel, return_grid=True)
    F0 = (derivs_grid * k_grid).sum(dim=(-2, -1))  # (batch,)

    out = torch.empty(batch_size, length, dim, dtype=torch.double)

    for i in range(length):
        for d in range(dim):
            x1_d = deepcopy(x1)
            x1_d[:, i, d] += eps
            k_grid_d = pysiglib.sig_kernel(x1_d, x2, dyadic_order, time_aug=time_aug, lead_lag=lead_lag, static_kernel=static_kernel, return_grid=True)
            F1 = (derivs_grid * k_grid_d).sum(dim=(-2, -1))
            out[:, i, d] = (F1 - F0) / eps
    return out


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_finite_diff(dyadic_order):
    X = torch.rand(size=(5, 3), dtype=torch.float64)
    Y = torch.rand(size=(7, 3), dtype=torch.float64)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, return_grid=True)
    derivs_grid = torch.rand_like(k_grid)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order)
    check_close(d1, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_finite_diff_batch(dyadic_order):
    X = torch.rand(size=(4, 5, 3), dtype=torch.float64)
    Y = torch.rand(size=(4, 7, 3), dtype=torch.float64)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, return_grid=True)
    derivs_grid = torch.rand_like(k_grid)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order)
    check_close(d1, d1_fd)


################################################
## Torch api tests
################################################

@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_torch_api_sum(dyadic_order):
    X = torch.rand(size=(5, 3), dtype=torch.float64, requires_grad=True)
    Y = torch.rand(size=(7, 3), dtype=torch.float64, requires_grad=True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order, return_grid=True)
    loss = k_grid.sum()
    loss.backward()

    d1_torch = X.grad.clone()

    # Finite-difference reference: derivs_grid = all ones
    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    derivs_grid = torch.ones_like(k_grid.detach())
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, derivs_grid, dyadic_order)

    check_close(d1_torch, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_torch_api_weighted(dyadic_order):
    X = torch.rand(size=(5, 3), dtype=torch.float64, requires_grad=True)
    Y = torch.rand(size=(7, 3), dtype=torch.float64, requires_grad=True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order, return_grid=True)
    weights = torch.rand_like(k_grid.detach())
    loss = (k_grid * weights).sum()
    loss.backward()

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, weights, dyadic_order)

    check_close(d1_torch, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_torch_api_batch(dyadic_order):
    X = torch.rand(size=(4, 5, 3), dtype=torch.float64, requires_grad=True)
    Y = torch.rand(size=(4, 7, 3), dtype=torch.float64, requires_grad=True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order, return_grid=True)
    weights = torch.rand_like(k_grid.detach())
    loss = (k_grid * weights).sum()
    loss.backward()

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, weights, dyadic_order)

    check_close(d1_torch, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_torch_api_consistency_last_element(dyadic_order):
    X = torch.rand(size=(4, 5, 3), dtype=torch.float64)
    Y = torch.rand(size=(4, 7, 3), dtype=torch.float64)

    # Scalar path: return_grid=False
    X1 = X.clone().requires_grad_()
    Y1 = Y.clone().requires_grad_()
    k_scalar = pysiglib.torch_api.sig_kernel(X1, Y1, dyadic_order)
    loss_scalar = k_scalar.sum()
    loss_scalar.backward()

    # Grid path: return_grid=True, sum only [-1,-1]
    X2 = X.clone().requires_grad_()
    Y2 = Y.clone().requires_grad_()
    k_grid = pysiglib.torch_api.sig_kernel(X2, Y2, dyadic_order, return_grid=True)
    loss_grid = k_grid[:, -1, -1].sum()
    loss_grid.backward()

    check_close(X1.grad, X2.grad)
    check_close(Y1.grad, Y2.grad)


################################################
## time_aug and lead_lag tests
################################################

@pytest.mark.parametrize("dyadic_order", range(3))
def test_grid_backprop_consistency_time_aug(dyadic_order):
    """Consistency with scalar backprop when time_aug=True."""
    X = torch.rand(size=(10, 5), dtype=torch.float64)
    Y = torch.rand(size=(10, 5), dtype=torch.float64)

    derivs_scalar = torch.ones(1, dtype=torch.float64)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order, time_aug=True,
        left_deriv=True, right_deriv=True
    )

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, time_aug=True, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid)
    derivs_grid[-1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, time_aug=True,
        left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)


@pytest.mark.parametrize("dyadic_order", range(3))
def test_grid_backprop_consistency_lead_lag(dyadic_order):
    """Consistency with scalar backprop when lead_lag=True."""
    X = torch.rand(size=(5, 2), dtype=torch.float64)
    Y = torch.rand(size=(10, 2), dtype=torch.float64)

    derivs_scalar = torch.ones(1, dtype=torch.float64)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order, lead_lag=True,
        left_deriv=True, right_deriv=True
    )

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, lead_lag=True, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid)
    derivs_grid[-1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, lead_lag=True,
        left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)


@pytest.mark.parametrize("dyadic_order", range(3))
def test_grid_backprop_consistency_time_aug_lead_lag(dyadic_order):
    """Consistency with scalar backprop when both time_aug=True and lead_lag=True."""
    X = torch.rand(size=(5, 2), dtype=torch.float64) / 2
    Y = torch.rand(size=(10, 2), dtype=torch.float64) / 2

    derivs_scalar = torch.ones(1, dtype=torch.float64)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order, time_aug=True, lead_lag=True,
        left_deriv=True, right_deriv=True
    )

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, time_aug=True, lead_lag=True, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid)
    derivs_grid[-1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, time_aug=True, lead_lag=True,
        left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_finite_diff_time_aug(dyadic_order):
    """Finite-difference check with time_aug=True."""
    X = torch.rand(size=(10, 5), dtype=torch.float64)
    Y = torch.rand(size=(10, 5), dtype=torch.float64)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, time_aug=True, return_grid=True)
    derivs_grid = torch.rand_like(k_grid)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, time_aug=True,
        left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order, time_aug=True)
    check_close(d1, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_finite_diff_lead_lag(dyadic_order):
    """Finite-difference check with lead_lag=True."""
    X = torch.rand(size=(5, 2), dtype=torch.float64)
    Y = torch.rand(size=(10, 2), dtype=torch.float64)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, lead_lag=True, return_grid=True)
    derivs_grid = torch.rand_like(k_grid)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, lead_lag=True,
        left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order, lead_lag=True)
    check_close(d1, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_finite_diff_time_aug_lead_lag(dyadic_order):
    """Finite-difference check with both time_aug=True and lead_lag=True."""
    X = torch.rand(size=(5, 2), dtype=torch.float64) / 2
    Y = torch.rand(size=(10, 2), dtype=torch.float64) / 2

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, time_aug=True, lead_lag=True, return_grid=True)
    derivs_grid = torch.rand_like(k_grid)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, time_aug=True, lead_lag=True,
        left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order, time_aug=True, lead_lag=True)
    check_close(d1, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_torch_api_time_aug(dyadic_order):
    """Torch autograd test with time_aug=True and return_grid=True."""
    X = torch.rand(size=(10, 5), dtype=torch.float64, requires_grad=True)
    Y = torch.rand(size=(10, 5), dtype=torch.float64, requires_grad=True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order, time_aug=True, return_grid=True)
    weights = torch.rand_like(k_grid.detach())
    loss = (k_grid * weights).sum()
    loss.backward()

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, weights, dyadic_order, time_aug=True)
    check_close(d1_torch, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_torch_api_lead_lag(dyadic_order):
    """Torch autograd test with lead_lag=True and return_grid=True."""
    X = torch.rand(size=(5, 2), dtype=torch.float64, requires_grad=True)
    Y = torch.rand(size=(10, 2), dtype=torch.float64, requires_grad=True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order, lead_lag=True, return_grid=True)
    weights = torch.rand_like(k_grid.detach())
    loss = (k_grid * weights).sum()
    loss.backward()

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, weights, dyadic_order, lead_lag=True)
    check_close(d1_torch, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_torch_api_time_aug_lead_lag(dyadic_order):
    """Torch autograd test with both time_aug=True and lead_lag=True, return_grid=True."""
    X = (torch.rand(size=(5, 2), dtype=torch.float64) / 2).requires_grad_(True)
    Y = (torch.rand(size=(10, 2), dtype=torch.float64) / 2).requires_grad_(True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order, time_aug=True, lead_lag=True, return_grid=True)
    weights = torch.rand_like(k_grid.detach())
    loss = (k_grid * weights).sum()
    loss.backward()

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, weights, dyadic_order, time_aug=True, lead_lag=True)
    check_close(d1_torch, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_batch_time_aug_lead_lag(dyadic_order):
    """Batch consistency test with time_aug=True and lead_lag=True."""
    X = torch.rand(size=(8, 5, 2), dtype=torch.float64) / 2
    Y = torch.rand(size=(8, 10, 2), dtype=torch.float64) / 2

    derivs_scalar = torch.ones(8, dtype=torch.float64)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order, time_aug=True, lead_lag=True,
        left_deriv=True, right_deriv=True
    )

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, time_aug=True, lead_lag=True, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid)
    derivs_grid[:, -1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, time_aug=True, lead_lag=True,
        left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)


################################################
## Static kernel tests
################################################

@pytest.mark.parametrize("dyadic_order", range(2))
def test_grid_backprop_rbf_kernel(dyadic_order):
    X = torch.rand(size=(5, 3), dtype=torch.float64)
    Y = torch.rand(size=(7, 3), dtype=torch.float64)
    kernel = pysiglib.RBFKernel(0.5)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=kernel, return_grid=True)
    derivs_grid = torch.rand_like(k_grid)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order, static_kernel=kernel,
        left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )

    # Finite difference
    X_ = X.to(dtype=torch.double)
    Y_ = Y.to(dtype=torch.double)
    eps = 1e-7
    k0 = pysiglib.sig_kernel(X_[None], Y_[None], dyadic_order, static_kernel=kernel, return_grid=True)
    F0 = (derivs_grid[None] * k0).sum(dim=(-2, -1))
    out = torch.empty_like(X_[None])
    for i in range(X_.shape[0]):
        for d_ in range(X_.shape[1]):
            Xp = deepcopy(X_[None])
            Xp[:, i, d_] += eps
            kp = pysiglib.sig_kernel(Xp, Y_[None], dyadic_order, static_kernel=kernel, return_grid=True)
            Fp = (derivs_grid[None] * kp).sum(dim=(-2, -1))
            out[:, i, d_] = (Fp - F0) / eps

    check_close(d1, out)
