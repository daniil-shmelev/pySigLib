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
import iisignature
import pysiglib

np.random.seed(42)
torch.manual_seed(42)

SINGLE_EPSILON = 1e-3
DOUBLE_EPSILON = 1e-10

def check_close(a, b, dtype):
    a_ = np.array(a)
    b_ = np.array(b)
    eps = SINGLE_EPSILON if dtype == torch.float32 else DOUBLE_EPSILON
    assert not np.any(np.abs(a_ - b_) > eps), f"Max diff: {np.max(np.abs(a_ - b_))}"

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_sig_combine_backprop_cuda_vs_cpu(deg, dtype):
    dim = 5
    sig_len = pysiglib.sig_length(dim, deg)

    sig1_cpu = torch.rand(sig_len, dtype=dtype)
    sig2_cpu = torch.rand(sig_len, dtype=dtype)
    deriv_cpu = torch.rand(sig_len, dtype=dtype)

    d1_cpu, d2_cpu = pysiglib.sig_combine_backprop(deriv_cpu, sig1_cpu, sig2_cpu, dim, deg)

    sig1_cuda = sig1_cpu.cuda()
    sig2_cuda = sig2_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    d1_cuda, d2_cuda = pysiglib.sig_combine_backprop(deriv_cuda, sig1_cuda, sig2_cuda, dim, deg)
    assert d1_cuda.device.type == "cuda"
    assert d2_cuda.device.type == "cuda"

    check_close(d1_cpu, d1_cuda.cpu(), dtype)
    check_close(d2_cpu, d2_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_batch_sig_combine_backprop_cuda_vs_cpu(deg, dtype):
    dim, batch = 5, 10
    sig_len = pysiglib.sig_length(dim, deg)

    sig1_cpu = torch.rand(batch, sig_len, dtype=dtype)
    sig2_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, sig_len, dtype=dtype)

    d1_cpu, d2_cpu = pysiglib.sig_combine_backprop(deriv_cpu, sig1_cpu, sig2_cpu, dim, deg)

    sig1_cuda = sig1_cpu.cuda()
    sig2_cuda = sig2_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    d1_cuda, d2_cuda = pysiglib.sig_combine_backprop(deriv_cuda, sig1_cuda, sig2_cuda, dim, deg)
    assert d1_cuda.device.type == "cuda"
    assert d2_cuda.device.type == "cuda"

    check_close(d1_cpu, d1_cuda.cpu(), dtype)
    check_close(d2_cpu, d2_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sig_combine_backprop_cuda_float32(deg, dtype):
    dim = 5
    sig_len = pysiglib.sig_length(dim, deg)

    sig1_cpu = torch.rand(sig_len, dtype=dtype)
    sig2_cpu = torch.rand(sig_len, dtype=dtype)
    deriv_cpu = torch.rand(sig_len, dtype=dtype)

    d1_cpu, d2_cpu = pysiglib.sig_combine_backprop(deriv_cpu, sig1_cpu, sig2_cpu, dim, deg)

    sig1_cuda = sig1_cpu.cuda()
    sig2_cuda = sig2_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    d1_cuda, d2_cuda = pysiglib.sig_combine_backprop(deriv_cuda, sig1_cuda, sig2_cuda, dim, deg)
    assert d1_cuda.device.type == "cuda"
    assert d2_cuda.device.type == "cuda"

    check_close(d1_cpu, d1_cuda.cpu(), dtype)
    check_close(d2_cpu, d2_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float32])
def test_batch_sig_combine_backprop_cuda_float32(deg, dtype):
    dim, batch = 5, 10
    sig_len = pysiglib.sig_length(dim, deg)

    sig1_cpu = torch.rand(batch, sig_len, dtype=dtype)
    sig2_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, sig_len, dtype=dtype)

    d1_cpu, d2_cpu = pysiglib.sig_combine_backprop(deriv_cpu, sig1_cpu, sig2_cpu, dim, deg)

    sig1_cuda = sig1_cpu.cuda()
    sig2_cuda = sig2_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    d1_cuda, d2_cuda = pysiglib.sig_combine_backprop(deriv_cuda, sig1_cuda, sig2_cuda, dim, deg)
    assert d1_cuda.device.type == "cuda"
    assert d2_cuda.device.type == "cuda"

    check_close(d1_cpu, d1_cuda.cpu(), dtype)
    check_close(d2_cpu, d2_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_sig_combine_backprop_random_cuda(deg):
    dimension = 5
    sig_len = pysiglib.sig_length(dimension, deg)

    sig1 = torch.rand(size = (sig_len,), device = "cuda", dtype = torch.float64)
    sig2 = torch.rand(size = (sig_len,), device = "cuda", dtype = torch.float64)
    derivs = torch.rand(size = (sig_len,), device = "cuda", dtype = torch.float64)

    sig1_deriv, sig2_deriv = pysiglib.sig_combine_backprop(derivs, sig1, sig2, dimension, deg)
    assert sig1_deriv.device.type == "cuda"
    assert sig2_deriv.device.type == "cuda"
    iisig1_deriv, iisig2_deriv = iisignature.sigcombinebackprop(derivs[1:].cpu(), sig1[1:].cpu(), sig2[1:].cpu(), dimension, deg)
    # iisignature returns float32, so use single precision tolerance
    check_close(sig1_deriv[1:].cpu(), iisig1_deriv, torch.float32)
    check_close(sig2_deriv[1:].cpu(), iisig2_deriv, torch.float32)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_batch_sig_combine_backprop_random_cuda(deg):
    dimension, batch_size = 5, 10
    sig_len = pysiglib.sig_length(dimension, deg)

    sig1 = torch.rand(size=(batch_size, sig_len), device="cuda", dtype=torch.float64)
    sig2 = torch.rand(size=(batch_size, sig_len), device="cuda", dtype=torch.float64)
    derivs = torch.rand(size=(batch_size, sig_len), device="cuda", dtype=torch.float64)

    sig1_deriv, sig2_deriv = pysiglib.sig_combine_backprop(derivs, sig1, sig2, dimension, deg)
    assert sig1_deriv.device.type == "cuda"
    assert sig2_deriv.device.type == "cuda"

    iisig1_deriv, iisig2_deriv = iisignature.sigcombinebackprop(
        derivs[:, 1:].cpu().numpy(), sig1[:, 1:].cpu().numpy(), sig2[:, 1:].cpu().numpy(), dimension, deg
    )
    # iisignature returns float32, so use single precision tolerance
    check_close(sig1_deriv[:, 1:].cpu(), iisig1_deriv, torch.float32)
    check_close(sig2_deriv[:, 1:].cpu(), iisig2_deriv, torch.float32)
