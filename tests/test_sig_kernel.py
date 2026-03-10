# Copyright 2025 Daniil Shmelev
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

import pytest
import numpy as np
import torch
import sigkernel

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, single_atol=1e-3, double_atol=1e-5)

def lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:-1], repeats=2, dim=0)
    lag = torch.cat((lag, x[-1:]))
    lead = torch.repeat_interleave(x[1:], repeats=2, dim=0)
    lead = torch.cat((x[0:1], lead))
    path = torch.cat((lag, lead), dim=-1)
    return path

def batch_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    return path

def time_aug_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:-1], repeats=2, dim=0)
    lag = torch.cat((lag, x[-1:]))
    lead = torch.repeat_interleave(x[1:], repeats=2, dim=0)
    lead = torch.cat((x[0:1], lead))
    path = torch.cat((lag, lead), dim=-1)
    t = torch.linspace(0, path.shape[0] - 1, path.shape[0]).unsqueeze(1)
    path = torch.cat((path, t), dim =  1)
    return path

def batch_time_aug_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    t = torch.linspace(0, path.shape[1] - 1, path.shape[1]).unsqueeze(0)
    t = torch.tile(t, (path.shape[0], 1)).unsqueeze(2)
    path = torch.cat((path, t), dim=2)
    return path

def sig_kernel_full_grid(X1, X2, len1, len2, batch):
    result = np.ones(shape = (batch, len1, len2))
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, 0)
    for b in range(batch):
        for i in range(1, len1):
            for j in range(1, len2):
                XX1 = torch.tensor(X1[b, :i + 1, :][np.newaxis, :, :])
                XX2 = torch.tensor(X2[b, :j + 1, :][np.newaxis, :, :])
                result[b][i][j] = float(signature_kernel.compute_kernel(XX1, XX2, 100)[0])
    return result


@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_trivial(device):
    X = torch.tensor([[0.]], device=device)
    k = pysiglib.sig_kernel(X, X, 0)
    assert_device(k, device)
    check_close(torch.tensor([1.]), k)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_numpy(device):
    x = torch.tensor([[0., 1.], [3., 2.]], device=device)
    k = pysiglib.sig_kernel(x, x, 0)
    assert_device(k, device)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_kernel_dtypes(device, dtype):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = (torch.rand(size=(batch, len1, dim), device=device) * 3 - 1.5).to(dtype=dtype) / 100
    Y = (torch.rand(size=(batch, len2, dim), device=device) * 3 - 1.5).to(dtype=dtype) / 100

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, 0)
    kernel1 = signature_kernel.compute_kernel(X.double(), Y.double(), 100)
    kernel2 = pysiglib.sig_kernel(X, Y, 0)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_random(device, dyadic_order):
    batch, len1, len2, dim = 32, 100, 100, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_scaled_linear(device, dyadic_order):
    batch, len1, len2, dim = 32, 100, 100, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel(0.5)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)

    static_kernel = pysiglib.ScaledLinearKernel(0.5)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel= static_kernel)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_rbf(device, dyadic_order):
    batch, len1, len2, dim = 32, 100, 100, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.RBFKernel(0.5)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)

    static_kernel = pysiglib.RBFKernel(0.5)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel= static_kernel)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(10, 100), (100, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_random_non_square(device, len1, len2, dyadic_order):
    batch, dim = 32, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", [(1,0), (2,0), (2,1)])
def test_sig_kernel_different_dyadics(device, dyadic_order):
    batch, len1, len2, dim = 32, 10, 100, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    kernel1 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert_device(kernel1, device)
    kernel2 = pysiglib.sig_kernel(Y, X, dyadic_order[::-1])

    check_close(kernel1, kernel2)


@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_non_contiguous(device):
    # Make sure sig_kernel works with any form of array
    dim, length, batch = 10, 100, 32

    rand_data = torch.rand(size=(batch, length), dtype=torch.float64, device=device)[:, :, None]
    X_non_cont = rand_data.expand(-1, -1, dim)
    X = X_non_cont.clone()

    res1 = pysiglib.sig_kernel(X, X, 0)
    res2 = pysiglib.sig_kernel(X_non_cont, X_non_cont, 0)
    assert_device(res1, device)
    assert_device(res2, device)
    check_close(res1, res2)

    rand_data = np.random.normal(size=(batch, length))[:, :, None]
    X_non_cont = np.broadcast_to(rand_data, (batch, length, dim))
    X = np.array(X_non_cont)

    res1 = pysiglib.sig_kernel(X, X, 0)
    res2 = pysiglib.sig_kernel(X_non_cont, X_non_cont, 0)
    check_close(res1, res2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_lead_lag(device, dyadic_order):
    X = torch.rand(size=(32, 50, 5), dtype = torch.double, device=device) / 100
    Y = torch.rand(size=(32, 100, 5), dtype = torch.double, device=device) / 100

    X_ll = batch_lead_lag(X).double()
    Y_ll = batch_lead_lag(Y).double()

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X_ll, Y_ll, 100)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, lead_lag = True)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(10, 10), (10, 5), (5, 10)])
def test_sig_kernel_full_grid(device, len1, len2):
    batch, dim = 2, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    kernel1 = sig_kernel_full_grid(X, Y, len1, len2, batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, return_grid=True)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_full_grid_time_aug(device):
    batch, len1, len2, dim = 10, 5, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    X_t = pysiglib.transform_path(X, time_aug = True)
    Y_t = pysiglib.transform_path(Y, time_aug = True)

    kernel1 = sig_kernel_full_grid(X_t, Y_t, len1, len2, batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, time_aug = True, return_grid=True)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_full_grid_lead_lag(device):
    batch, len1, len2, dim = 10, 5, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    X_ll = pysiglib.transform_path(X, lead_lag = True)
    Y_ll = pysiglib.transform_path(Y, lead_lag = True)

    kernel1 = sig_kernel_full_grid(X_ll, Y_ll, X_ll.shape[1], Y_ll.shape[1], batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, lead_lag = True, return_grid=True)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_full_grid_time_aug_lead_lag(device):
    batch, len1, len2, dim = 10, 5, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    X_ll = pysiglib.transform_path(X, time_aug = True, lead_lag=True)
    Y_ll = pysiglib.transform_path(Y, time_aug = True, lead_lag=True)

    kernel1 = sig_kernel_full_grid(X_ll, Y_ll, X_ll.shape[1], Y_ll.shape[1], batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, lead_lag = True, time_aug = True, return_grid=True)
    assert_device(kernel2, device)

    check_close(kernel1, kernel2)
