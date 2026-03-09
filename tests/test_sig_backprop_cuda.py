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

try:
    import iisignature
    HAS_IISIGNATURE = True
except ImportError:
    HAS_IISIGNATURE = False

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

SINGLE_EPSILON = 1e-4
DOUBLE_EPSILON = 1e-5

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    EPSILON = SINGLE_EPSILON if a_.dtype == np.float32 else DOUBLE_EPSILON
    assert not np.max(np.abs(a_ - b_)) > EPSILON

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

@pytest.mark.skipif(not HAS_IISIGNATURE, reason="iisignature not available")
@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_sig_backprop_random_cuda(deg):
    X = torch.rand(size=(100, 5), device = "cuda")
    sig_derivs = torch.rand(size = (pysiglib.sig_length(5, deg),), device = "cuda")

    sig = pysiglib.sig(X, deg)

    sig_back1 = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg)
    assert sig_back1.device.type == "cuda"
    sig_back2 = iisignature.sigbackprop(sig_derivs[1:].clone().cpu(), X.clone().cpu(), deg)
    check_close(sig_back1.cpu(), sig_back2)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_backprop_random_cuda_vs_cpu(deg, dtype):
    X = torch.rand(size=(100, 5), dtype=dtype)
    sig_derivs = torch.rand(size=(pysiglib.sig_length(5, deg),), dtype=dtype)
    sig = pysiglib.sig(X, deg)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_batch_sig_backprop_random_cuda(deg):
    X = torch.rand(size=(100, 3, 2), dtype=torch.float64)
    sig_derivs = torch.rand(size=(100, pysiglib.sig_length(2, deg)), dtype=torch.float64)
    sig = pysiglib.sig(X.clone(), deg)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_sig_backprop_time_aug_random_cuda(deg):
    length, dimension = 100, 5
    X = torch.rand(size=(length, dimension), dtype=torch.float64)
    sig_derivs = torch.rand(size=(pysiglib.sig_length(dimension + 1, deg),), dtype=torch.float64)
    sig = pysiglib.sig(X, deg, time_aug=True)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, time_aug=True)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg, time_aug=True)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_time_aug_random_cuda(deg):
    batch_size, length, dimension = 10, 100, 5
    X = torch.rand(size=(batch_size, length, dimension), dtype=torch.float64)
    sig_derivs = torch.rand(size=(batch_size, pysiglib.sig_length(dimension + 1, deg)), dtype=torch.float64)
    sig = pysiglib.sig(X.clone(), deg, time_aug=True)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, time_aug=True)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg, time_aug=True)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_sig_backprop_lead_lag_random_cuda(deg):
    length, dimension = 100, 5
    X = torch.rand(size=(length, dimension), dtype=torch.float64)
    sig = pysiglib.sig(X, deg, lead_lag=True)
    sig_derivs = torch.rand(size=(pysiglib.sig_length(dimension * 2, deg),), dtype=torch.float64)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, lead_lag=True)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg, lead_lag=True)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_lead_lag_random_cuda(deg):
    batch_size, length, dimension = 10, 100, 5
    X = torch.rand(size=(batch_size, length, dimension), dtype=torch.float64)
    sig = pysiglib.sig(X.clone(), deg, lead_lag=True)
    sig_derivs = torch.rand(size=(batch_size, pysiglib.sig_length(dimension * 2, deg)), dtype=torch.float64)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, lead_lag=True)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg, lead_lag=True)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_sig_backprop_time_aug_lead_lag_random_cuda(deg):
    length, dimension = 100, 5
    X = torch.rand(size=(length, dimension), dtype=torch.float64)
    sig = pysiglib.sig(X, deg, time_aug=True, lead_lag=True)
    sig_derivs = torch.rand(size=(pysiglib.sig_length(dimension * 2 + 1, deg),), dtype=torch.float64)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, time_aug=True, lead_lag=True)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg, time_aug=True, lead_lag=True)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_time_aug_lead_lag_random_cuda(deg):
    batch_size, length, dimension = 10, 100, 5
    X = torch.rand(size=(batch_size, length, dimension), dtype=torch.float64)
    sig = pysiglib.sig(X.clone(), deg, time_aug=True, lead_lag=True)
    sig_derivs = torch.rand(size=(batch_size, pysiglib.sig_length(dimension * 2 + 1, deg)), dtype=torch.float64)

    sig_back_cpu = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, time_aug=True, lead_lag=True)
    sig_back_cuda = pysiglib.sig_backprop(X.clone().cuda(), sig.clone().cuda(), sig_derivs.clone().cuda(), deg, time_aug=True, lead_lag=True)
    assert sig_back_cuda.device.type == "cuda"
    check_close(sig_back_cuda.cpu(), sig_back_cpu)
