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

SINGLE_EPSILON = 1e-4
DOUBLE_EPSILON = 1e-10

def lead_lag(X):
    lag = []
    lead = []

    for val_lag, val_lead in zip(X[:-1], X[1:]):
        lag.append(val_lag)
        lead.append(val_lag)

        lag.append(val_lag)
        lead.append(val_lead)

    lag.append(X[-1])
    lead.append(X[-1])

    return np.c_[lag, lead]

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
@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_random_cuda(deg):
    X = np.random.uniform(size=(100, 5))
    iisig = iisignature.sig(X, deg)
    X = torch.tensor(X, device="cuda")
    sig = pysiglib.sig(X, deg)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu()[1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_signature_random_cuda_dtypes(deg, dtype):
    X = np.random.uniform(size=(100, 5))
    iisig = iisignature.sig(X, deg)
    X_cuda = torch.tensor(X, device="cuda", dtype=dtype)
    sig = pysiglib.sig(X_cuda, deg)
    assert sig.device.type == "cuda"
    sig = sig.cpu().numpy()
    if dtype == torch.float32:
        check_close(iisig.astype(np.float32), sig[1:].astype(np.float32))
    else:
        check_close(iisig, sig[1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("horner", [True, False])
def test_signature_random_cuda_horner(deg, horner):
    X = np.random.uniform(size=(100, 5))
    iisig = iisignature.sig(X, deg)
    X_cuda = torch.tensor(X, device="cuda")
    sig = pysiglib.sig(X_cuda, deg, horner=horner)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu()[1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_random_batch_cuda(deg):
    X = np.random.uniform(size=(32, 100, 5))
    iisig = iisignature.sig(X, deg)
    X_cuda = torch.tensor(X, device="cuda")
    sig = pysiglib.sig(X_cuda, deg)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu()[:, 1:])

@skip_no_cuda
def test_signature_trivial_cuda():
    X = torch.tensor([[0., 0.], [1., 1.]], device="cuda")
    sig0 = pysiglib.sig(X, 0)
    assert sig0.device.type == "cuda"
    check_close(sig0.cpu(), [1.])
    sig1 = pysiglib.sig(X, 1)
    assert sig1.device.type == "cuda"
    check_close(sig1.cpu(), [1., 1., 1.])
    X2 = torch.tensor([[0., 0.]], device="cuda")
    sig2 = pysiglib.sig(X2, 1)
    assert sig2.device.type == "cuda"
    check_close(sig2.cpu(), [1., 0., 0.])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_time_aug_cuda(deg):
    X = np.random.uniform(size=(10, 4))
    t = np.linspace(0, 1, 10)[:, np.newaxis]
    X_aug = np.concatenate([X, t], axis=1)
    iisig = iisignature.sig(X_aug, deg)
    X_cuda = torch.tensor(X, device="cuda")
    sig = pysiglib.sig(X_cuda, deg, time_aug=True)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu()[1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_lead_lag_cuda(deg):
    X = np.random.uniform(size=(10, 2))
    X_aug = lead_lag(X)
    iisig = iisignature.sig(X_aug, deg)
    X_cuda = torch.tensor(X, device="cuda")
    sig = pysiglib.sig(X_cuda, deg, lead_lag=True)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu()[1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_signature_time_aug_lead_lag_cuda(deg, dtype):
    X = np.random.uniform(size=(10, 2))
    X_aug = lead_lag(X)
    t = np.linspace(0, 1, 19)[:, np.newaxis]
    X_aug = np.concatenate([X_aug, t], axis=1)
    iisig = iisignature.sig(X_aug, deg)
    X_cuda = torch.tensor(X, device="cuda", dtype=dtype)
    sig = pysiglib.sig(X_cuda, deg, lead_lag=True, time_aug=True)
    assert sig.device.type == "cuda"
    sig = sig.cpu().numpy()
    if dtype == torch.float32:
        check_close(iisig.astype(np.float32), sig[1:].astype(np.float32))
    else:
        check_close(iisig, sig[1:])

@skip_no_cuda
def test_signature_non_contiguous_cuda():
    dim, degree, length, batch = 10, 3, 100, 32

    rand_data = torch.rand((batch, length), dtype=torch.float64, device="cuda")[:, :, None]
    X_non_cont = rand_data.expand(-1, -1, dim)
    X = X_non_cont.clone()

    res1 = pysiglib.sig(X, degree)
    res2 = pysiglib.sig(X_non_cont, degree)
    assert res1.device.type == "cuda"
    assert res2.device.type == "cuda"
    check_close(res1.cpu(), res2.cpu())
