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

signatory = None
try:
    import signatory
except Exception:
    signatory = None

import pysiglib.torch_api as pysiglib

np.random.seed(42)
torch.manual_seed(42)

SINGLE_EPSILON = 1e-4
DOUBLE_EPSILON = 1e-10

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
def test_prepare_memory_cuda():
    X = torch.rand(size=(100, 5), device="cuda", dtype=torch.float64)
    pysiglib.clear_cache(True)
    pysiglib.prepare_log_sig(5, 2, 1)

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=2)

    pysiglib.clear_cache()

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=1)

    pysiglib.prepare_log_sig(5, 2, 2)
    result = pysiglib.log_sig(X, 2, method=1)
    assert result.device.type == "cuda"
    pysiglib.clear_cache()

@skip_no_cuda
def test_prepare_disk_cuda():
    X = torch.rand(size=(100, 5), device="cuda", dtype=torch.float64)
    pysiglib.clear_cache(True)
    pysiglib.prepare_log_sig(5, 2, 1, use_disk=True)
    pysiglib.clear_cache(False)

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=2)

    pysiglib.clear_cache(True)

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=1)

    pysiglib.prepare_log_sig(5, 2, 2, use_disk=True)
    pysiglib.clear_cache(False)
    result = pysiglib.log_sig(X, 2, method=1)
    assert result.device.type == "cuda"
    pysiglib.clear_cache(True)

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_expanded_random_cuda(deg, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_np = np.random.uniform(size=(100, 5)).astype(np_dtype)

    s = iisignature.prepare(5, deg, "x")
    iisig = iisignature.logsig(X_np, s, "x").astype(np_dtype)

    X = torch.tensor(X_np, device="cuda", dtype=dtype)
    sig = pysiglib.log_sig(X, deg, method=0)

    assert sig.device.type == "cuda"
    check_close(iisig, sig[1:].cpu().numpy())

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_expanded_random_cuda(deg, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_np = np.random.uniform(size=(32, 100, 5)).astype(np_dtype)

    s = iisignature.prepare(5, deg, "x")
    iisig = iisignature.logsig(X_np, s, "x").astype(np_dtype)

    X = torch.tensor(X_np, device="cuda", dtype=dtype)
    sig = pysiglib.log_sig(X, deg, method=0)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu().numpy()[:, 1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_expanded_time_aug_random_cuda(deg, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_np = np.random.uniform(size=(32, 100, 5)).astype(np_dtype)

    s = iisignature.prepare(6, deg, "x")
    iisig = iisignature.logsig(pysiglib.transform_path(X_np, time_aug=True), s, "x").astype(np_dtype)

    X = torch.tensor(X_np, device="cuda", dtype=dtype)
    sig = pysiglib.log_sig(X, deg, time_aug=True, method=0)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu().numpy()[:, 1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_expanded_lead_lag_random_cuda(deg, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_np = np.random.uniform(size=(32, 100, 5)).astype(np_dtype)

    s = iisignature.prepare(10, deg, "x")
    iisig = iisignature.logsig(pysiglib.transform_path(X_np, lead_lag=True), s, "x").astype(np_dtype)

    X = torch.tensor(X_np, device="cuda", dtype=dtype)
    sig = pysiglib.log_sig(X, deg, lead_lag=True, method=0)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu().numpy()[:, 1:])

@skip_no_cuda
@pytest.mark.skipif(signatory is None, reason="signatory not available")
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_lyndon_words_random_cuda(deg, dtype):
    X = torch.rand(size=(1, 100, 5), dtype=dtype)

    ls = signatory.logsignature(X, deg, mode="words")[0]
    pysiglib.prepare_log_sig(5, deg, 1)
    X_cuda = X[0].to("cuda")
    sig = pysiglib.log_sig(X_cuda, deg, method=1)
    assert sig.device.type == "cuda"
    check_close(ls, sig.cpu())
    pysiglib.clear_cache()

@skip_no_cuda
@pytest.mark.skipif(signatory is None, reason="signatory not available")
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_lyndon_words_random_cuda(deg, dtype):
    X = torch.rand(size=(32, 100, 5), dtype=dtype)

    ls = signatory.logsignature(X, deg, mode="words")
    pysiglib.prepare_log_sig(5, deg, 1)
    X_cuda = X.to("cuda")
    sig = pysiglib.log_sig(X_cuda, deg, method=1)
    assert sig.device.type == "cuda"
    check_close(ls, sig.cpu())
    pysiglib.clear_cache()

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_lyndon_basis_random_cuda(deg, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_np = np.random.uniform(size=(100, 5)).astype(np_dtype)

    s = iisignature.prepare(5, deg, "s")
    iisig = iisignature.logsig(X_np, s, "s").astype(np_dtype)

    pysiglib.prepare_log_sig(5, deg, 2)
    X = torch.tensor(X_np, device="cuda", dtype=dtype)
    sig = pysiglib.log_sig(X, deg, method=2)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu().numpy())
    pysiglib.clear_cache()

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_lyndon_basis_random_cuda(deg, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_np = np.random.uniform(size=(32, 100, 5)).astype(np_dtype)

    s = iisignature.prepare(5, deg, "s")
    iisig = iisignature.logsig(X_np, s, "s").astype(np_dtype)

    pysiglib.prepare_log_sig(5, deg, 2)
    X = torch.tensor(X_np, device="cuda", dtype=dtype)
    sig = pysiglib.log_sig(X, deg, method=2)
    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu().numpy())
    pysiglib.clear_cache()

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_log_signature_expanded_end_time_cuda(deg):
    # Test method=0 with time_aug=True and a non-default end_time on a single path.
    # iisignature receives the pre-transformed path so that end_time is reflected.
    X_np = np.random.uniform(size=(100, 5)).astype(np.float64)

    s = iisignature.prepare(6, deg, "x")
    iisig = iisignature.logsig(
        pysiglib.transform_path(X_np, time_aug=True, end_time=2.0), s, "x"
    ).astype(np.float64)

    X = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    sig = pysiglib.log_sig(X, deg, time_aug=True, end_time=2.0, method=0)

    assert sig.device.type == "cuda"
    check_close(iisig, sig[1:].cpu().numpy())

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_log_signature_expanded_end_time_cuda(deg):
    # Same as test_log_signature_expanded_end_time_cuda but for a batch of paths.
    X_np = np.random.uniform(size=(32, 100, 5)).astype(np.float64)

    s = iisignature.prepare(6, deg, "x")
    iisig = iisignature.logsig(
        pysiglib.transform_path(X_np, time_aug=True, end_time=2.0), s, "x"
    ).astype(np.float64)

    X = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    sig = pysiglib.log_sig(X, deg, time_aug=True, end_time=2.0, method=0)

    assert sig.device.type == "cuda"
    check_close(iisig, sig.cpu().numpy()[:, 1:])

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 5))
def test_log_signature_expanded_time_aug_lead_lag_cuda(deg):
    # Test method=0 with both time_aug=True and lead_lag=True on a single path.
    # lead_lag doubles the dimension (10), time_aug adds one channel (11 total).
    X_np = np.random.uniform(size=(100, 5)).astype(np.float64)

    s = iisignature.prepare(11, deg, "x")
    iisig = iisignature.logsig(
        pysiglib.transform_path(X_np, time_aug=True, lead_lag=True), s, "x"
    ).astype(np.float64)

    X = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    sig = pysiglib.log_sig(X, deg, time_aug=True, lead_lag=True, method=0)

    assert sig.device.type == "cuda"
    check_close(iisig, sig[1:].cpu().numpy())

@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 4))
def test_log_signature_single_point_cuda(deg):
    # The log signature of a path with a single point has no increments and must be all zeros.
    X_np = np.random.uniform(size=(1, 5)).astype(np.float64)

    X = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    sig = pysiglib.log_sig(X, deg, method=0)

    assert sig.device.type == "cuda"
    assert np.allclose(sig.cpu().numpy(), 0.0)
