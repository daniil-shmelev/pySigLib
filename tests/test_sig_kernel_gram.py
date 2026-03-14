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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_kernel_gram_dtypes(device, dtype):
    batch1, batch2, len1, len2, dim = 8, 4, 10, 10, 5
    X = torch.rand(size=(batch1, len1, dim), device=device).to(dtype=dtype)
    Y = torch.rand(size=(batch2, len2, dim), device=device).to(dtype=dtype)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, 0)
    kernel1 = signature_kernel.compute_Gram(X.double(), Y.double(), False, 100)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, 0)
    assert_device(kernel2, device)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, 0, max_batch = 2)

    check_close(kernel1, kernel2)
    check_close(kernel1, kernel3)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_gram_random(device, dyadic_order):
    batch1, batch2, len1, len2, dim = 32, 16, 100, 100, 5
    X = torch.rand(size=(batch1, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch2, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_Gram(X, Y, False, 100)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, dyadic_order)
    assert_device(kernel2, device)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, max_batch=2)

    check_close(kernel1, kernel2)
    check_close(kernel1, kernel3)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_gram_lead_lag(device, dyadic_order):
    X = torch.rand(size=(4, 50, 5), dtype = torch.double, device=device) / 100
    Y = torch.rand(size=(4, 100, 5), dtype = torch.double, device=device) / 100

    X_ll = batch_lead_lag(X).double()
    Y_ll = batch_lead_lag(Y).double()

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_Gram(X_ll, Y_ll, False, 100)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, lead_lag = True)
    assert_device(kernel2, device)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, lead_lag = True, max_batch=2)

    check_close(kernel1, kernel2)
    check_close(kernel1, kernel3)
