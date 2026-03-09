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

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_kernel_dtypes_cuda(dtype):
    batch, len1, len2, dim = 32, 10, 10, 5
    # arr * 3 - 1.5 below gives us non-zero values for int dtypes
    X = (torch.rand(size=(batch, len1, dim), device="cuda") * 3 - 1.5).to(dtype=dtype) / 100
    Y = (torch.rand(size=(batch, len2, dim), device="cuda") * 3 - 1.5).to(dtype=dtype) / 100

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, 0)
    kernel1 = signature_kernel.compute_kernel(X.double(), Y.double(), 100)
    kernel2 = pysiglib.sig_kernel(X, Y, 0)
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_random_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 100, 100, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())

@skip_no_cuda
@pytest.mark.parametrize(("len1", "len2"), [(10, 100), (100, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_random_non_square_cuda(len1, len2, dyadic_order):
    batch, dim = 32, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", [(1,0), (2,0), (2,1)])
def test_sig_kernel_different_dyadics_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 10, 100, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    kernel1 = pysiglib.sig_kernel(X, Y, dyadic_order)
    kernel2 = pysiglib.sig_kernel(Y, X, dyadic_order[::-1])
    assert kernel1.device.type == "cuda"
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())

@skip_no_cuda
@pytest.mark.parametrize(("len1", "len2"), [(10, 10), (10, 100), (100, 10)])
def test_sig_kernel_full_grid_cuda(len1, len2):
    batch, dim = 2, 5
    X = torch.rand(size=(batch, len1, dim), device = "cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device = "cuda", dtype = torch.double)

    kernel1 = sig_kernel_full_grid(X, Y, len1, len2, batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, return_grid=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_full_grid_time_aug_cuda():
    batch, len1, len2, dim = 10, 5, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    X_t = pysiglib.transform_path(X, time_aug = True)
    Y_t = pysiglib.transform_path(Y, time_aug = True)

    kernel1 = sig_kernel_full_grid(X_t, Y_t, len1, len2, batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, time_aug = True, return_grid=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_full_grid_lead_lag_cuda():
    batch, len1, len2, dim = 10, 5, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    X_ll = pysiglib.transform_path(X, lead_lag = True)
    Y_ll = pysiglib.transform_path(Y, lead_lag = True)

    kernel1 = sig_kernel_full_grid(X_ll, Y_ll, X_ll.shape[1], Y_ll.shape[1], batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, lead_lag = True, return_grid=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_full_grid_time_aug_lead_lag_cuda():
    batch, len1, len2, dim = 10, 5, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    X_ll = pysiglib.transform_path(X, time_aug = True, lead_lag=True)
    Y_ll = pysiglib.transform_path(Y, time_aug = True, lead_lag=True)

    kernel1 = sig_kernel_full_grid(X_ll, Y_ll, X_ll.shape[1], Y_ll.shape[1], batch)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, lead_lag = True, time_aug = True, return_grid=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

def batch_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    return path

@skip_no_cuda
def test_sig_kernel_trivial_cuda():
    X = torch.tensor([[0.]], device="cuda")
    k = pysiglib.sig_kernel(X, X, 0)
    assert k.device.type == "cuda"
    check_close(torch.tensor([1.]), k.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_scaled_linear_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 100, 100, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel(0.5)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)

    static_kernel = pysiglib.ScaledLinearKernel(0.5)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel= static_kernel)
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_rbf_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 100, 100, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.RBFKernel(0.5)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X, Y, 100)

    static_kernel = pysiglib.RBFKernel(0.5)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel= static_kernel)
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_non_contiguous_cuda():
    dim, length, batch = 10, 100, 32

    rand_data = torch.rand(size=(batch, length), dtype=torch.float64, device="cuda")[:, :, None]
    X_non_cont = rand_data.expand(-1, -1, dim)
    X = X_non_cont.clone()

    res1 = pysiglib.sig_kernel(X, X, 0)
    res2 = pysiglib.sig_kernel(X_non_cont, X_non_cont, 0)
    assert res1.device.type == "cuda"
    assert res2.device.type == "cuda"

    check_close(res1.cpu(), res2.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_lead_lag_cuda(dyadic_order):
    X = torch.rand(size=(32, 50, 5), dtype = torch.double, device="cuda") / 100
    Y = torch.rand(size=(32, 100, 5), dtype = torch.double, device="cuda") / 100

    X_ll = batch_lead_lag(X).double()
    Y_ll = batch_lead_lag(Y).double()

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel1 = signature_kernel.compute_kernel(X_ll, Y_ll, 100)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, lead_lag = True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1.cpu(), kernel2.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_sig_kernel_time_aug_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype=torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype=torch.double)

    # CPU reference: manually apply time-aug then run sig_kernel on CPU
    X_cpu = X.cpu()
    Y_cpu = Y.cpu()
    kernel1 = pysiglib.sig_kernel(X_cpu, Y_cpu, dyadic_order, time_aug=True)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, time_aug=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_sig_kernel_time_aug_lead_lag_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype=torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype=torch.double)

    X_cpu = X.cpu()
    Y_cpu = Y.cpu()
    kernel1 = pysiglib.sig_kernel(X_cpu, Y_cpu, dyadic_order, time_aug=True, lead_lag=True)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, time_aug=True, lead_lag=True)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_end_time_cuda():
    batch, len1, len2, dim = 32, 10, 10, 5
    end_time = 2.0
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype=torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype=torch.double)

    X_cpu = X.cpu()
    Y_cpu = Y.cpu()
    kernel1 = pysiglib.sig_kernel(X_cpu, Y_cpu, 0, time_aug=True, end_time=end_time)
    kernel2 = pysiglib.sig_kernel(X, Y, 0, time_aug=True, end_time=end_time)
    assert kernel2.device.type == "cuda"

    check_close(kernel1, kernel2.cpu())

@skip_no_cuda
def test_sig_kernel_single_point_cuda():
    # A path with a single time-step has a trivial signature of 1; verify
    # the kernel is well-defined and equals 1 for two identical single-point paths.
    dim = 5
    X = torch.rand(size=(1, dim), device="cuda", dtype=torch.double)
    k = pysiglib.sig_kernel(X, X, 0)
    assert k.device.type == "cuda"
    check_close(torch.tensor([1.0], dtype=torch.double), k.cpu())
