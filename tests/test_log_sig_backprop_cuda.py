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
import pysiglib

np.random.seed(42)
torch.manual_seed(42)

SINGLE_EPSILON = 1e-4
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
@pytest.mark.parametrize("deg", range(2, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_log_sig_backprop_expanded_cuda_vs_cpu(deg, dtype):
    dim = 3
    sig_len = pysiglib.sig_length(dim, deg)
    sig_cpu = torch.rand(sig_len, dtype=dtype)
    deriv_cpu = torch.rand(sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=0)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=0)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_batch_log_sig_backprop_expanded_cuda_vs_cpu(deg, dtype):
    dim, batch = 3, 10
    sig_len = pysiglib.sig_length(dim, deg)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=0)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=0)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_log_sig_backprop_lyndon_words_cuda_vs_cpu(deg, dtype):
    dim = 3
    pysiglib.prepare_log_sig(dim, deg, method=1)
    sig_len = pysiglib.sig_length(dim, deg)
    log_sig_len = pysiglib.log_sig_length(dim, deg)
    sig_cpu = torch.rand(sig_len, dtype=dtype)
    deriv_cpu = torch.rand(log_sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=1)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=1)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_batch_log_sig_backprop_lyndon_words_cuda_vs_cpu(deg, dtype):
    dim, batch = 3, 10
    pysiglib.prepare_log_sig(dim, deg, method=1)
    sig_len = pysiglib.sig_length(dim, deg)
    log_sig_len = pysiglib.log_sig_length(dim, deg)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, log_sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=1)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=1)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_log_sig_backprop_lyndon_basis_cuda_vs_cpu(deg, dtype):
    dim = 3
    pysiglib.prepare_log_sig(dim, deg, method=2)
    sig_len = pysiglib.sig_length(dim, deg)
    log_sig_len = pysiglib.log_sig_length(dim, deg)
    sig_cpu = torch.rand(sig_len, dtype=dtype)
    deriv_cpu = torch.rand(log_sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=2)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=2)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 6))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_batch_log_sig_backprop_lyndon_basis_cuda_vs_cpu(deg, dtype):
    dim, batch = 3, 10
    pysiglib.prepare_log_sig(dim, deg, method=2)
    sig_len = pysiglib.sig_length(dim, deg)
    log_sig_len = pysiglib.log_sig_length(dim, deg)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, log_sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=2)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=2)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_to_log_sig_backprop_expanded_random_cuda(deg, dtype):
    dim = 1
    sig_len = pysiglib.sig_length(dim, deg)
    sig_cpu = torch.rand(1, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(1, sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=0)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=0)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_sig_backprop_expanded_time_aug_cuda_vs_cpu(deg, dtype):
    dim, batch = 2, 32
    sig_len = pysiglib.sig_length(dim, deg, time_aug=True)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, time_aug=True, method=0)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, time_aug=True, method=0)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 5))
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sig_to_log_sig_backprop_float32_cuda(deg, dtype):
    # Verify that float32 backprop on CUDA matches the CPU result within SINGLE_EPSILON.
    dim, batch = 5, 10
    sig_len = pysiglib.sig_length(dim, deg)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(sig_cpu, deriv_cpu, dim, deg, method=0)

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(sig_cuda, deriv_cuda, dim, deg, method=0)
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 5))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_sig_to_log_sig_backprop_time_aug_lead_lag_cuda(deg, dtype):
    # Test method=0 backprop with both time_aug and lead_lag enabled.
    # The augmented dimension is 2*dim + 1; sig/deriv lengths account for this.
    dim, batch = 5, 10
    sig_len = pysiglib.sig_length(dim, deg, time_aug=True, lead_lag=True)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(
        sig_cpu, deriv_cpu, dim, deg, time_aug=True, lead_lag=True, method=0
    )

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(
        sig_cuda, deriv_cuda, dim, deg, time_aug=True, lead_lag=True, method=0
    )
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 5))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_sig_to_log_sig_backprop_m1_time_aug_cuda(deg, dtype):
    # Test method=1 (Lyndon words) backprop with time_aug=True.
    # prepare_log_sig must be called with matching flags before computing.
    dim, batch = 5, 10
    pysiglib.prepare_log_sig(dim, deg, method=1, time_aug=True)
    sig_len = pysiglib.sig_length(dim, deg, time_aug=True)
    log_sig_len = pysiglib.log_sig_length(dim, deg, time_aug=True)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, log_sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(
        sig_cpu, deriv_cpu, dim, deg, time_aug=True, method=1
    )

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(
        sig_cuda, deriv_cuda, dim, deg, time_aug=True, method=1
    )
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)
    pysiglib.clear_cache()

@skip_no_cuda
@pytest.mark.parametrize("deg", range(2, 5))
@pytest.mark.parametrize("dtype", [torch.float64])
def test_sig_to_log_sig_backprop_m2_time_aug_cuda(deg, dtype):
    # Test method=2 (Lyndon basis) backprop with time_aug=True.
    # prepare_log_sig must be called with matching flags before computing.
    dim, batch = 5, 10
    pysiglib.prepare_log_sig(dim, deg, method=2, time_aug=True)
    sig_len = pysiglib.sig_length(dim, deg, time_aug=True)
    log_sig_len = pysiglib.log_sig_length(dim, deg, time_aug=True)
    sig_cpu = torch.rand(batch, sig_len, dtype=dtype)
    deriv_cpu = torch.rand(batch, log_sig_len, dtype=dtype)

    result_cpu = pysiglib.sig_to_log_sig_backprop(
        sig_cpu, deriv_cpu, dim, deg, time_aug=True, method=2
    )

    sig_cuda = sig_cpu.cuda()
    deriv_cuda = deriv_cpu.cuda()
    result_cuda = pysiglib.sig_to_log_sig_backprop(
        sig_cuda, deriv_cuda, dim, deg, time_aug=True, method=2
    )
    assert result_cuda.device.type == "cuda"

    check_close(result_cpu, result_cuda.cpu(), dtype)
    pysiglib.clear_cache()
