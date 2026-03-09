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

EPSILON = 1e-10
SINGLE_EPSILON = 1e-4

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON), f"Max diff: {np.max(np.abs(a_ - b_))}"

def check_close_single(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > SINGLE_EPSILON), f"Max diff: {np.max(np.abs(a_ - b_))}"

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

def get_true_sig_coefs(multi_indices, X, *args, **kwargs):
    dim = X.shape[-1]
    sig = pysiglib.signature(X, *args, **kwargs)
    res = []
    for idx in multi_indices:
        flat_idx = 0
        for i in idx:
            flat_idx *= dim
            flat_idx += i + 1
        res.append(sig[..., flat_idx])
    return np.array(res).T

# Mirrors test_sig_coef_trivial
@skip_no_cuda
def test_sig_coef_trivial_cuda():
    X = torch.tensor([[0., 0.], [1., 1.]], device="cuda", dtype=torch.float64)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert result.device.type == "cuda"
    check_close(result.cpu(), [1., 1.])

    X = torch.tensor([[0., 0.]], device="cuda", dtype=torch.float64)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert result.device.type == "cuda"
    check_close(result.cpu(), [0., 0.])

# Mirrors test_batch_sig_coef_trivial
@skip_no_cuda
def test_batch_sig_coef_trivial_cuda():
    X = torch.tensor([[[0., 0.], [1., 1.]]], device="cuda", dtype=torch.float64)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert result.device.type == "cuda"
    check_close(result.cpu(), [1., 1.])

    X = torch.tensor([[[0., 0.]]], device="cuda", dtype=torch.float64)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert result.device.type == "cuda"
    check_close(result.cpu(), [0., 0.])

# Mirrors test_sig_coef
@skip_no_cuda
def test_sig_coef_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]

    true_coeffs = get_true_sig_coefs(multi_indices, X_np, 5)
    coeff = pysiglib.sig_coef(X_cuda, multi_indices)
    assert coeff.device.type == "cuda"
    check_close(true_coeffs, coeff.cpu())

# Mirrors test_sig_coef_prefixes
@skip_no_cuda
def test_sig_coef_prefixes_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]
    grid_idx = [(0,), (0, 1), (2,), (2, 1), (2, 1, 0), (1,)]

    true_coeffs = get_true_sig_coefs(grid_idx, X_np, 5)
    coeff = pysiglib.sig_coef(X_cuda, multi_indices, prefixes=True)
    assert coeff.device.type == "cuda"
    check_close(true_coeffs, coeff.cpu())

# Mirrors test_batch_sig_coef_prefixes
@skip_no_cuda
def test_batch_sig_coef_prefixes_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(10, 100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]
    grid_idx = [(0,), (0, 1), (2,), (2, 1), (2, 1, 0), (1,)]

    true_coeffs = get_true_sig_coefs(grid_idx, X_np, 5)
    coeff = pysiglib.sig_coef(X_cuda, multi_indices, prefixes=True)
    assert coeff.device.type == "cuda"
    check_close(true_coeffs, coeff.cpu())

# Mirrors test_sig_coef_full
@skip_no_cuda
def test_sig_coef_full_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = pysiglib.words(3, 5)[1:]

    coeff = pysiglib.sig_coef(X_cuda, multi_indices)
    sig = pysiglib.signature(X_np, 5)
    assert coeff.device.type == "cuda"
    check_close(sig[1:], coeff.cpu())

# Mirrors test_batch_sig_coef_full
@skip_no_cuda
def test_batch_sig_coef_full_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(10, 100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = pysiglib.words(3, 5)[1:]

    coeff = pysiglib.sig_coef(X_cuda, multi_indices)
    sig = pysiglib.signature(X_np, 5)
    assert coeff.device.type == "cuda"
    check_close(sig[:, 1:], coeff.cpu())

# CUDA vs CPU comparison: verify CUDA output matches CPU output exactly
@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 8))
def test_sig_coef_cuda_vs_cpu(deg):
    np.random.seed(42)
    X_np = np.random.uniform(size=(50, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = [(0,) * deg, (1,) * deg, (2, 0) * (deg // 2 + 1)]
    multi_indices = [idx[:deg] for idx in multi_indices]

    coeff_cpu = pysiglib.sig_coef(X_np, multi_indices)
    coeff_cuda = pysiglib.sig_coef(X_cuda, multi_indices)
    assert coeff_cuda.device.type == "cuda"
    check_close(coeff_cpu, coeff_cuda.cpu())

# Batch CUDA vs CPU comparison
@skip_no_cuda
@pytest.mark.parametrize("deg", range(1, 8))
def test_batch_sig_coef_cuda_vs_cpu(deg):
    np.random.seed(42)
    X_np = np.random.uniform(size=(5, 50, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = [(0,) * deg, (1,) * deg, (2, 0) * (deg // 2 + 1)]
    multi_indices = [idx[:deg] for idx in multi_indices]

    coeff_cpu = pysiglib.sig_coef(X_np, multi_indices)
    coeff_cuda = pysiglib.sig_coef(X_cuda, multi_indices)
    assert coeff_cuda.device.type == "cuda"
    check_close(coeff_cpu, coeff_cuda.cpu())

# Mirrors test_extract_sig_coef_all
@skip_no_cuda
def test_extract_sig_coef_all_cuda():
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension), device="cuda", dtype=torch.float64)
    sig = pysiglib.sig(x, degree)
    assert sig.device.type == "cuda"
    words = pysiglib.words(dimension, degree)
    coefs = pysiglib.extract_sig_coef(sig, words, dimension)
    assert coefs.device.type == "cuda"
    check_close(sig.cpu(), coefs.cpu())

# Mirrors test_extract_sig_coef_lyndon
@skip_no_cuda
def test_extract_sig_coef_lyndon_cuda():
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension), device="cuda", dtype=torch.float64)
    pysiglib.prepare_log_sig(dimension, degree, method=1)
    log_sig_full = pysiglib.log_sig(x, degree, method=0)
    log_sig = pysiglib.log_sig(x, degree, method=1)
    assert log_sig_full.device.type == "cuda"
    assert log_sig.device.type == "cuda"
    words = pysiglib.lyndon_words(dimension, degree)
    coefs = pysiglib.extract_sig_coef(log_sig_full, words, dimension)
    assert coefs.device.type == "cuda"
    check_close(log_sig.cpu(), coefs.cpu())

# Mirrors test_batch_sig_coef_full_time_aug
@skip_no_cuda
def test_batch_sig_coef_full_time_aug_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(10, 100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = pysiglib.words(4, 5)[1:]

    coeff = pysiglib.sig_coef(X_cuda, multi_indices, time_aug=True)
    sig = pysiglib.signature(X_np, 5, time_aug=True)
    assert coeff.device.type == "cuda"
    check_close(sig[:, 1:], coeff.cpu())

# Mirrors test_batch_sig_coef_full_lead_lag
@skip_no_cuda
def test_batch_sig_coef_full_lead_lag_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(10, 100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = pysiglib.words(6, 5)[1:]

    coeff = pysiglib.sig_coef(X_cuda, multi_indices, lead_lag=True)
    sig = pysiglib.signature(X_np, 5, lead_lag=True)
    assert coeff.device.type == "cuda"
    check_close(sig[:, 1:], coeff.cpu())

# Mirrors test_batch_sig_coef_full_time_aug_lead_lag
@skip_no_cuda
def test_batch_sig_coef_full_time_aug_lead_lag_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(10, 100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = pysiglib.words(7, 5)[1:]

    coeff = pysiglib.sig_coef(X_cuda, multi_indices, time_aug=True, lead_lag=True)
    sig = pysiglib.signature(X_np, 5, time_aug=True, lead_lag=True)
    assert coeff.device.type == "cuda"
    check_close(sig[:, 1:], coeff.cpu())

# float32 path (single): verify CUDA output matches reference signature coefficients
@skip_no_cuda
def test_sig_coef_float32_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float32)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]

    true_coeffs = get_true_sig_coefs(multi_indices, X_np, 5)
    coeff = pysiglib.sig_coef(X_cuda, multi_indices)
    assert coeff.device.type == "cuda"
    check_close_single(true_coeffs, coeff.cpu())

# float32 batch: verify CUDA output matches reference signature coefficients
@skip_no_cuda
def test_batch_sig_coef_float32_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(10, 100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float32)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]

    true_coeffs = get_true_sig_coefs(multi_indices, X_np, 5)
    coeff = pysiglib.sig_coef(X_cuda, multi_indices)
    assert coeff.device.type == "cuda"
    check_close_single(true_coeffs, coeff.cpu())

# Test sig_coef with time_aug=True and a non-default end_time
# Words index into the augmented path dimension (dim=3+1=4 for time_aug)
@skip_no_cuda
def test_sig_coef_end_time_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(10, 100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    # Augmented dimension is 4 (time channel prepended to dim=3)
    # Use words(4, 5) to get the full set for augmented dimension, then compare
    # against pysiglib.signature with the same params
    multi_indices = pysiglib.words(4, 5)[1:]

    coeff = pysiglib.sig_coef(X_cuda, multi_indices, time_aug=True, end_time=2.0)
    sig = pysiglib.signature(X_np, 5, time_aug=True, end_time=2.0)
    assert coeff.device.type == "cuda"
    check_close(sig[:, 1:], coeff.cpu())

# Test sig_coef with a single word passed as a tuple (not a list)
@skip_no_cuda
def test_sig_coef_single_word_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    word = (0, 1, 2)

    true_coeffs = get_true_sig_coefs([word], X_np, 5)
    coeff = pysiglib.sig_coef(X_cuda, word)
    assert coeff.device.type == "cuda"
    check_close(true_coeffs.squeeze(), coeff.cpu())

# Test sig_coef with a high-degree word (degree 7): verifies CUDA vs CPU consistency
@skip_no_cuda
def test_sig_coef_high_degree_word_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(50, 2))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float64)
    multi_indices = [(0, 1, 0, 1, 0, 1, 0)]

    coeff_cpu = pysiglib.sig_coef(X_np, multi_indices)
    coeff_cuda = pysiglib.sig_coef(X_cuda, multi_indices)
    assert coeff_cuda.device.type == "cuda"
    check_close(coeff_cpu, coeff_cuda.cpu())

# float32 batch with prefixes=True: verify CUDA output matches reference
@skip_no_cuda
def test_sig_coef_prefixes_float32_cuda():
    np.random.seed(42)
    X_np = np.random.uniform(size=(100, 3))
    X_cuda = torch.tensor(X_np, device="cuda", dtype=torch.float32)
    multi_indices = [(0, 1), (2, 1, 0)]
    grid_idx = [(0,), (0, 1), (2,), (2, 1), (2, 1, 0)]

    true_coeffs = get_true_sig_coefs(grid_idx, X_np, 5)
    coeff = pysiglib.sig_coef(X_cuda, multi_indices, prefixes=True)
    assert coeff.device.type == "cuda"
    check_close_single(true_coeffs, coeff.cpu())

