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

SINGLE_EPSILON = 1e-1
DOUBLE_EPSILON = 1e-1

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    EPSILON = SINGLE_EPSILON if a_.dtype == np.float32 else DOUBLE_EPSILON
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def finite_difference(x1, x2, dyadic_order, time_aug = False, lead_lag = False, kernel = None):
    x1 = x1.to(device = "cpu", dtype = torch.double)
    x2 = x2.to(device = "cpu", dtype = torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-10
    k = pysiglib.sig_kernel(x1, x2, dyadic_order, time_aug = time_aug, lead_lag = lead_lag, static_kernel= kernel)
    out = np.empty(shape = (batch_size, length, dim))

    for i in range(length):
        for d in range(dim):
            x1_d = deepcopy(x1)
            x1_d[:,i,d] += eps
            k_d = pysiglib.sig_kernel(x1_d, x2, dyadic_order, time_aug = time_aug, lead_lag = lead_lag, static_kernel= kernel)
            out[:,i,d] = (k_d - k) / eps
    return out

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

@skip_no_cuda
@pytest.mark.parametrize(("len1", "len2"), [(100, 100), (100, 10), (10, 100)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_cuda(len1, len2, dyadic_order):
    X = torch.rand(size=(len1, 5), device = "cuda")
    Y = torch.rand(size=(len2, 5), device = "cuda")
    derivs = torch.ones(1, device = "cuda")

    d1 = finite_difference(X, Y, dyadic_order)
    d2 = finite_difference(Y, X, dyadic_order)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = True, right_deriv = True)
    assert d3.device.type == "cuda"
    assert d4.device.type == "cuda"

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_cuda(dyadic_order):
    X = torch.rand(size=(32, 10, 5), device = "cuda")
    Y = torch.rand(size=(32, 100, 5), device = "cuda")
    derivs = torch.ones(32, device = "cuda")

    d1 = finite_difference(X, Y, dyadic_order)
    d2 = finite_difference(Y, X, dyadic_order)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = True, right_deriv = True)
    assert d3.device.type == "cuda"
    assert d4.device.type == "cuda"

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_scaled_linear_backprop_batch_cuda(dyadic_order):
    X = torch.rand(size=(32, 10, 5), device = "cuda")
    Y = torch.rand(size=(32, 100, 5), device = "cuda")
    derivs = torch.ones(32, device = "cuda")

    kernel = pysiglib.ScaledLinearKernel(0.5)

    d1 = finite_difference(X, Y, dyadic_order, kernel = kernel)
    d2 = finite_difference(Y, X, dyadic_order, kernel = kernel)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = True, right_deriv = True, static_kernel= kernel)
    assert d3.device.type == "cuda"
    assert d4.device.type == "cuda"

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_rbf_backprop_batch_cuda(dyadic_order):
    X = torch.rand(size=(32, 10, 5), device = "cuda")
    Y = torch.rand(size=(32, 100, 5), device = "cuda")
    derivs = torch.ones(32, device = "cuda")

    kernel = pysiglib.RBFKernel(0.5)

    d1 = finite_difference(X, Y, dyadic_order, kernel = kernel)
    d2 = finite_difference(Y, X, dyadic_order, kernel = kernel)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = True, right_deriv = True, static_kernel= kernel)
    _, d5 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, left_deriv = False, right_deriv = True, static_kernel= kernel)
    assert d3.device.type == "cuda"
    assert d4.device.type == "cuda"
    assert d5.device.type == "cuda"

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())
    check_close(d2, d5.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_cuda_time_aug(dyadic_order):
    X = torch.rand(size=(32, 5, 5), device = "cuda")
    Y = torch.rand(size=(32, 10, 5), device = "cuda")
    derivs = torch.ones(32, device = "cuda")

    d1 = finite_difference(X, Y, dyadic_order, time_aug = True)
    d2 = finite_difference(Y, X, dyadic_order, time_aug=True)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, time_aug = True, left_deriv = True, right_deriv = True)
    assert d3.device.type == "cuda"
    assert d4.device.type == "cuda"

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_cuda_lead_lag(dyadic_order):
    X = torch.rand(size=(32, 5, 2), device = "cuda")
    Y = torch.rand(size=(32, 10, 2), device = "cuda")
    derivs = torch.ones(32, device = "cuda")

    d1 = finite_difference(X, Y, dyadic_order, lead_lag = True)
    d2 = finite_difference(Y, X, dyadic_order, lead_lag=True)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, lead_lag = True, left_deriv = True, right_deriv = True)
    assert d3.device.type == "cuda"
    assert d4.device.type == "cuda"

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_cuda_time_aug_lead_lag(dyadic_order):
    X = torch.rand(size=(32, 5, 2), device = "cuda") / 2
    Y = torch.rand(size=(32, 10, 2), device = "cuda") / 2
    derivs = torch.ones(32, device = "cuda")

    d1 = finite_difference(X, Y, dyadic_order, time_aug = True, lead_lag = True)
    d2 = finite_difference(Y, X, dyadic_order, time_aug=True, lead_lag=True)
    d3, d4 = pysiglib.sig_kernel_backprop(derivs, X, Y, dyadic_order, time_aug = True, lead_lag = True, left_deriv = True, right_deriv = True)

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_backprop_batch_cuda_torch_api(dyadic_order):
    X = torch.rand(size=(32, 10, 5), device = "cuda")
    Y = torch.rand(size=(32, 100, 5), device = "cuda")
    derivs = torch.ones(32, device = "cuda")

    d1 = finite_difference(X, Y, dyadic_order)
    d2 = finite_difference(Y, X, dyadic_order)
    X.requires_grad_()
    Y.requires_grad_()
    K = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order)
    K.backward(derivs)
    d3, d4 = X.grad, Y.grad

    check_close(d1, d3.cpu())
    check_close(d2, d4.cpu())
