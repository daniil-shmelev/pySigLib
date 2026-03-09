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

@skip_no_cuda
@pytest.mark.parametrize("end_time", [2.0, 0.5])
def test_sig_end_time_cuda(end_time):
    # Verify that the CUDA signature with time_aug and non-default end_time
    # matches the CPU result for the same end_time value.
    batch_size, length, dimension = 10, 50, 3
    X = np.random.uniform(size=(batch_size, length, dimension))
    X_cpu = torch.tensor(X, dtype=torch.float64)
    X_cuda = torch.tensor(X, dtype=torch.float64, device="cuda")

    sig_cpu = pysiglib.sig(X_cpu, 3, time_aug=True, end_time=end_time)
    sig_cuda = pysiglib.sig(X_cuda, 3, time_aug=True, end_time=end_time)

    assert sig_cuda.device.type == "cuda"
    check_close(sig_cpu, sig_cuda.cpu())

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 4))
def test_sig_single_point_cuda(deg):
    # The signature of a single-point path has no increments, so all terms
    # beyond degree 0 are zero; the degree-0 term is 1 (the leading scalar).
    dimension = 3
    sig_len = pysiglib.sig_length(dimension, deg)

    X_single = torch.zeros((1, dimension), dtype=torch.float64, device="cuda")
    sig = pysiglib.sig(X_single, deg)

    assert sig.device.type == "cuda"
    expected = torch.zeros(sig_len, dtype=torch.float64)
    expected[0] = 1.0
    check_close(expected, sig.cpu())

    # Batch version: each path consists of a single point
    batch_size = 5
    X_batch = torch.zeros((batch_size, 1, dimension), dtype=torch.float64, device="cuda")
    sig_batch = pysiglib.sig(X_batch, deg)

    assert sig_batch.device.type == "cuda"
    expected_batch = torch.zeros((batch_size, sig_len), dtype=torch.float64)
    expected_batch[:, 0] = 1.0
    check_close(expected_batch, sig_batch.cpu())

@skip_no_cuda
@pytest.mark.parametrize("deg", [6, 7])
def test_sig_high_degree_cuda(deg):
    # Verify correctness of CUDA signature at high truncation levels by
    # comparing against the CPU result on a low-dimensional path.
    length, dimension = 50, 2
    X = np.random.uniform(size=(length, dimension))
    X_cpu = torch.tensor(X, dtype=torch.float64)
    X_cuda = torch.tensor(X, dtype=torch.float64, device="cuda")

    sig_cpu = pysiglib.sig(X_cpu, deg)
    sig_cuda = pysiglib.sig(X_cuda, deg)

    assert sig_cuda.device.type == "cuda"
    check_close(sig_cpu, sig_cuda.cpu())

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
def test_sig_float32_cuda_vs_cpu(deg):
    # Float32 CUDA and CPU results should agree within SINGLE_EPSILON.
    length, dimension = 50, 4
    X = np.random.uniform(size=(length, dimension)).astype(np.float32)
    X_cpu = torch.tensor(X, dtype=torch.float32)
    X_cuda = torch.tensor(X, dtype=torch.float32, device="cuda")

    sig_cpu = pysiglib.sig(X_cpu, deg)
    sig_cuda = pysiglib.sig(X_cuda, deg)

    assert sig_cuda.device.type == "cuda"
    check_close(
        sig_cpu.numpy().astype(np.float32),
        sig_cuda.cpu().numpy().astype(np.float32),
    )

@skip_no_cuda
def test_batch_sig_end_time_lead_lag_cuda():
    # Verify CUDA vs CPU agreement for a batch signature with both time
    # augmentation and lead-lag enabled, using a non-default end_time.
    batch_size, length, dimension, deg = 5, 50, 3, 3
    end_time = 3.0
    X = np.random.uniform(size=(batch_size, length, dimension))
    X_cpu = torch.tensor(X, dtype=torch.float64)
    X_cuda = torch.tensor(X, dtype=torch.float64, device="cuda")

    sig_cpu = pysiglib.sig(X_cpu, deg, time_aug=True, lead_lag=True, end_time=end_time)
    sig_cuda = pysiglib.sig(X_cuda, deg, time_aug=True, lead_lag=True, end_time=end_time)

    assert sig_cuda.device.type == "cuda"
    check_close(sig_cpu, sig_cuda.cpu())
