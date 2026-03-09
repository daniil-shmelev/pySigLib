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

import pytest
import numpy as np
import torch
import sigkernel

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

SINGLE_EPSILON = 1e-3
DOUBLE_EPSILON = 1e-5

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    EPSILON = SINGLE_EPSILON if a_.dtype == np.float32 else DOUBLE_EPSILON
    assert not np.any(np.abs(a_ - b_) > EPSILON)

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_gram_random_cuda(dyadic_order):
    batch1, batch2, len1, len2, dim = 32, 16, 100, 100, 5
    X = torch.rand(size=(batch1, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch2, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_Gram(X, Y, False, 100)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, dyadic_order)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, max_batch = 2)
    assert kernel2.device.type == "cuda"
    assert kernel3.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())
    check_close(kernel1.cpu(), kernel3.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_kernel_gram_dtypes_cuda(dtype):
    batch1, batch2, len1, len2, dim = 8, 4, 10, 10, 5
    X = torch.rand(size=(batch1, len1, dim), device="cuda").to(dtype=dtype)
    Y = torch.rand(size=(batch2, len2, dim), device="cuda").to(dtype=dtype)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, 0)
    kernel1 = signature_kernel.compute_Gram(X.double(), Y.double(), False, 100)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, 0)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, 0, max_batch = 2)
    assert kernel2.device.type == "cuda"
    assert kernel3.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())
    check_close(kernel1.cpu(), kernel3.cpu())

def batch_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    return path

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_gram_lead_lag_cuda(dyadic_order):
    X = torch.rand(size=(4, 50, 5), device="cuda", dtype = torch.double) / 100
    Y = torch.rand(size=(4, 100, 5), device="cuda", dtype = torch.double) / 100

    X_ll = batch_lead_lag(X).double()
    Y_ll = batch_lead_lag(Y).double()

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_Gram(X_ll, Y_ll, False, 100)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, lead_lag = True)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, lead_lag = True, max_batch = 2)
    assert kernel2.device.type == "cuda"
    assert kernel3.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())
    check_close(kernel1.cpu(), kernel3.cpu())

@skip_no_cuda
def test_sig_kernel_gram_time_aug_cuda():
    batch1, batch2, length, dim = 16, 16, 10, 3
    X = torch.rand(size=(batch1, length, dim), device="cuda", dtype=torch.double)
    Y = torch.rand(size=(batch2, length, dim), device="cuda", dtype=torch.double)

    # CPU reference
    X_cpu = X.cpu()
    Y_cpu = Y.cpu()
    kernel1 = pysiglib.sig_kernel_gram(X_cpu, Y_cpu, 0, time_aug=True)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, 0, time_aug=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_gram_time_aug_lead_lag_cuda():
    batch1, batch2, length, dim = 16, 16, 10, 3
    X = torch.rand(size=(batch1, length, dim), device="cuda", dtype=torch.double)
    Y = torch.rand(size=(batch2, length, dim), device="cuda", dtype=torch.double)

    X_cpu = X.cpu()
    Y_cpu = Y.cpu()
    kernel1 = pysiglib.sig_kernel_gram(X_cpu, Y_cpu, 0, time_aug=True, lead_lag=True)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, 0, time_aug=True, lead_lag=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_gram_max_batch_cuda():
    # Verify that using a small max_batch produces the same result as using a large one.
    batch1, batch2, length, dim = 16, 16, 10, 3
    X = torch.rand(size=(batch1, length, dim), device="cuda", dtype=torch.double)
    Y = torch.rand(size=(batch2, length, dim), device="cuda", dtype=torch.double)

    kernel1 = pysiglib.sig_kernel_gram(X, Y, 0, max_batch=4)
    kernel2 = pysiglib.sig_kernel_gram(X, Y, 0, max_batch=100)
    assert kernel1.device.type == "cuda"
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())
