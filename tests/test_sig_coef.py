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

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def test_extract_sig_coef_all():
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension))
    sig = pysiglib.sig(x, degree)
    words = pysiglib.words(dimension, degree)
    coefs = pysiglib.extract_sig_coef(sig, words, dimension)
    check_close(sig, coefs)

def test_extract_sig_coef_lyndon():
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension))
    pysiglib.prepare_log_sig(dimension, degree, method=1)
    log_sig_full = pysiglib.log_sig(x, degree, method=0)
    log_sig = pysiglib.log_sig(x, degree, method=1)
    words = pysiglib.lyndon_words(dimension, degree)
    coefs = pysiglib.extract_sig_coef(log_sig_full, words, dimension)
    check_close(log_sig, coefs)

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

def test_sig_coef_trivial():
    X = np.array([[0., 0.], [1., 1.]])
    check_close(pysiglib.sig_coef(X, [(0,), (1,)]), [1., 1.])

    X = np.array([[0., 0.]])
    check_close(pysiglib.sig_coef(X, [(0,), (1,)]), [0., 0.])

def test_batch_sig_coef_trivial():
    X = np.array([[[0., 0.], [1., 1.]]])
    check_close(pysiglib.sig_coef(X, [(0,), (1,)]), [1., 1.])

    X = np.array([[[0., 0.]]])
    check_close(pysiglib.sig_coef(X, [(0,), (1,)]), [0., 0.])

def test_sig_coef():
    X = np.random.uniform(size=(100, 3))
    multi_indices = [(0, 1), (2, 1, 0), (1,)]

    true_coeffs = get_true_sig_coefs(multi_indices, X, 5)
    coeff = pysiglib.sig_coef(X, multi_indices)
    check_close(true_coeffs, coeff)

def test_sig_coef_prefixes():
    X = np.random.uniform(size=(100, 3))
    multi_indices = [(0, 1), (2, 1, 0), (1,)]
    grid_idx = [(0,), (0,1), (2,), (2,1), (2,1,0), (1,)]

    true_coeffs = get_true_sig_coefs(grid_idx, X, 5)
    coeff = pysiglib.sig_coef(X, multi_indices, prefixes = True)
    check_close(true_coeffs, coeff)

def test_batch_sig_coef_prefixes():
    X = np.random.uniform(size=(10, 100, 3))
    multi_indices = [(0, 1), (2, 1, 0), (1,)]
    grid_idx = [(0,), (0,1), (2,), (2,1), (2,1,0), (1,)]

    true_coeffs = get_true_sig_coefs(grid_idx, X, 5)
    coeff = pysiglib.sig_coef(X, multi_indices, prefixes = True)
    check_close(true_coeffs, coeff)

def test_sig_coef_full():
    X = np.random.uniform(size=(100, 3))
    multi_indices = pysiglib.words(3, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices)
    sig = pysiglib.signature(X, 5)
    check_close(sig[1:], coeff)

def test_batch_sig_coef_full():
    X = np.random.uniform(size=(10, 100, 3))
    multi_indices = pysiglib.words(3, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices)
    sig = pysiglib.signature(X, 5)
    check_close(sig[:, 1:], coeff)

def test_batch_sig_coef_full_time_aug():
    X = np.random.uniform(size=(10, 100, 3))
    multi_indices = pysiglib.words(4, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices, time_aug = True)
    sig = pysiglib.signature(X, 5, time_aug = True)
    check_close(sig[:, 1:], coeff)

def test_batch_sig_coef_full_lead_lag():
    X = np.random.uniform(size=(10, 100, 3))
    multi_indices = pysiglib.words(6, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices, lead_lag = True)
    sig = pysiglib.signature(X, 5, lead_lag = True)
    check_close(sig[:, 1:], coeff)

def test_batch_sig_coef_full_time_aug_lead_lag():
    X = np.random.uniform(size=(10, 100, 3))
    multi_indices = pysiglib.words(7, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices, time_aug = True, lead_lag = True)
    sig = pysiglib.signature(X, 5, time_aug = True, lead_lag = True)
    check_close(sig[:, 1:], coeff)

@pytest.mark.skipif(not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()), reason="CUDA not available or disabled")
def test_batch_sig_coef_full_cuda():
    X = torch.rand(size=(10, 100, 3), device="cuda", dtype=torch.float64)
    multi_indices = pysiglib.words(3, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices)
    sig = pysiglib.signature(X, 5)

    assert coeff.device.type == "cuda"
    check_close(sig[:, 1:].cpu(), coeff.cpu())
