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

import native_api as pysiglib

np.random.seed(42)
torch.manual_seed(42)

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, single_atol=1e-2, double_atol=1e-4)

################################################
## Consistency with return_grid=False
################################################

@pytest.mark.parametrize(("len1", "len2"), [(10, 10), (10, 5), (5, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_consistency_with_scalar(len1, len2, dyadic_order, device):
    """
    When derivs_grid has 1.0 only at [-1,-1] (and 0 elsewhere), return_grid=True backprop
    should produce the same result as return_grid=False with deriv=1.0.
    """
    X = torch.rand(size=(len1, 3), dtype=torch.float64, device=device)
    Y = torch.rand(size=(len2, 3), dtype=torch.float64, device=device)

    # Scalar backprop
    derivs_scalar = torch.ones(1, dtype=torch.float64, device=device)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=True
    )
    assert_device(d1_scalar, device)
    assert_device(d2_scalar, device)

    # Grid backprop: construct derivs_grid with 1.0 at [-1,-1]
    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid, device=device)
    derivs_grid[-1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )
    assert_device(d1_grid, device)
    assert_device(d2_grid, device)

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)


@pytest.mark.parametrize("dyadic_order", range(3))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_consistency_batch(dyadic_order, device):
    X = torch.rand(size=(32, 10, 5), dtype=torch.float64, device=device)
    Y = torch.rand(size=(32, 10, 5), dtype=torch.float64, device=device)

    derivs_scalar = torch.ones(32, dtype=torch.float64, device=device)
    d1_scalar, d2_scalar = pysiglib.sig_kernel_backprop(
        derivs_scalar, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=True
    )
    assert_device(d1_scalar, device)
    assert_device(d2_scalar, device)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order, return_grid=True)
    derivs_grid = torch.zeros_like(k_grid, device=device)
    derivs_grid[:, -1, -1] = 1.0
    d1_grid, d2_grid = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=True,
        k_grid=k_grid, return_grid=True
    )
    assert_device(d1_grid, device)
    assert_device(d2_grid, device)

    check_close(d1_scalar, d1_grid)
    check_close(d2_scalar, d2_grid)

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
    derivs_grid = derivs_grid.to(device="cpu", dtype=torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
        derivs_grid = derivs_grid[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-7
    k_grid = pysiglib.sig_kernel(x1, x2, dyadic_order=dyadic_order, time_aug=time_aug, lead_lag=lead_lag, static_kernel=static_kernel, return_grid=True)
    F0 = (derivs_grid * k_grid).sum(dim=(-2, -1))  # (batch,)

    out = torch.empty(batch_size, length, dim, dtype=torch.double)

    for i in range(length):
        for d in range(dim):
            x1_d = deepcopy(x1)
            x1_d[:, i, d] += eps
            k_grid_d = pysiglib.sig_kernel(x1_d, x2, dyadic_order=dyadic_order, time_aug=time_aug, lead_lag=lead_lag, static_kernel=static_kernel, return_grid=True)
            F1 = (derivs_grid * k_grid_d).sum(dim=(-2, -1))
            out[:, i, d] = (F1 - F0) / eps
    return out


@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_finite_diff(dyadic_order, device):
    X = torch.rand(size=(5, 5), dtype=torch.float64, device=device)
    Y = torch.rand(size=(7, 5), dtype=torch.float64, device=device)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order, return_grid=True)
    derivs_grid = torch.rand_like(k_grid, device=device)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )
    assert_device(d1, device)

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order)
    check_close(d1, d1_fd)


@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_finite_diff_batch(dyadic_order, device):
    X = torch.rand(size=(32, 5, 5), dtype=torch.float64, device=device)
    Y = torch.rand(size=(32, 7, 5), dtype=torch.float64, device=device)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order, return_grid=True)
    derivs_grid = torch.rand_like(k_grid, device=device)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True
    )
    assert_device(d1, device)

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order)
    check_close(d1, d1_fd)

@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_finite_diff_time_aug_batch(dyadic_order, device):
    X = torch.rand(size=(32, 5, 5), dtype=torch.float64, device=device)
    Y = torch.rand(size=(32, 7, 5), dtype=torch.float64, device=device)

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order, time_aug=True, return_grid=True)
    derivs_grid = torch.rand_like(k_grid, device=device)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True, time_aug=True,
    )
    assert_device(d1, device)

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order, time_aug=True)
    check_close(d1, d1_fd)

@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_finite_diff_lead_lag_batch(dyadic_order, device):
    X = torch.rand(size=(32, 5, 5), dtype=torch.float64, device=device) / 200
    Y = torch.rand(size=(32, 10, 5), dtype=torch.float64, device=device) / 200

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order, return_grid=True, lead_lag=True)
    derivs_grid = torch.rand_like(k_grid, device=device)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True, lead_lag=True
    )
    assert_device(d1, device)

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order, lead_lag=True)
    check_close(d1, d1_fd)

@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_finite_diff_time_aug_lead_lag_batch(dyadic_order, device):
    X = torch.rand(size=(32, 5, 5), dtype=torch.float64, device=device) / 200
    Y = torch.rand(size=(32, 10, 5), dtype=torch.float64, device=device) / 200

    k_grid = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order, return_grid=True, time_aug=True, lead_lag=True)
    derivs_grid = torch.rand_like(k_grid, device=device)

    d1, _ = pysiglib.sig_kernel_backprop(
        derivs_grid, X, Y, dyadic_order=dyadic_order, left_deriv=True, right_deriv=False,
        k_grid=k_grid, return_grid=True, time_aug=True, lead_lag=True
    )
    assert_device(d1, device)

    d1_fd = _finite_difference_grid(X, Y, derivs_grid, dyadic_order, time_aug=True, lead_lag=True)
    check_close(d1, d1_fd)


################################################
## Torch api tests
################################################

@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_torch_api_batch(dyadic_order, device):
    X = torch.rand(size=(32, 5, 5), dtype=torch.float64, device=device, requires_grad=True)
    Y = torch.rand(size=(32, 7, 5), dtype=torch.float64, device=device, requires_grad=True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order=dyadic_order, return_grid=True)
    assert_device(k_grid, device)
    derivs = torch.rand(k_grid.shape, device=device)
    k_grid.backward(derivs)

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, derivs, dyadic_order)

    check_close(d1_torch, d1_fd)

@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_torch_api_time_aug_batch(dyadic_order, device):
    X = torch.rand(size=(32, 5, 4), dtype=torch.float64, device=device, requires_grad=True)
    Y = torch.rand(size=(32, 7, 4), dtype=torch.float64, device=device, requires_grad=True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order=dyadic_order, time_aug=True, return_grid=True)
    assert_device(k_grid, device)
    derivs = torch.rand(k_grid.shape, device=device)
    k_grid.backward(derivs)

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, derivs, dyadic_order, time_aug=True)

    check_close(d1_torch, d1_fd)

@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_torch_api_lead_lag_batch(dyadic_order, device):
    X = (torch.rand(size=(32, 5, 2)) / 200).to(dtype=torch.float64, device=device).requires_grad_(True)
    Y = (torch.rand(size=(32, 10, 2)) / 200).to(dtype=torch.float64, device=device).requires_grad_(True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order=dyadic_order, lead_lag=True, return_grid=True)
    assert_device(k_grid, device)
    derivs = torch.rand(k_grid.shape, device=device)
    k_grid.backward(derivs)

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, derivs, dyadic_order, lead_lag=True)

    check_close(d1_torch, d1_fd)

@pytest.mark.parametrize("dyadic_order", range(2))
@pytest.mark.parametrize("device", DEVICES)
def test_grid_backprop_torch_api_time_aug_lead_lag_batch(dyadic_order, device):
    X = (torch.rand(size=(32, 5, 2)) / 200).to(dtype=torch.float64, device=device).requires_grad_(True)
    Y = (torch.rand(size=(32, 10, 2)) / 200).to(dtype=torch.float64, device=device).requires_grad_(True)

    k_grid = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order=dyadic_order, time_aug=True, lead_lag=True, return_grid=True)
    assert_device(k_grid, device)
    derivs = torch.rand(k_grid.shape, device=device)
    k_grid.backward(derivs)

    d1_torch = X.grad.clone()

    X_nograd = X.detach().clone()
    Y_nograd = Y.detach().clone()
    d1_fd = _finite_difference_grid(X_nograd, Y_nograd, derivs, dyadic_order, time_aug=True, lead_lag=True)

    check_close(d1_torch, d1_fd)
