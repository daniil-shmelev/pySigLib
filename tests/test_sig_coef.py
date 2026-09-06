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
import native_api as pysiglib

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, atol=1e-10)

@pytest.mark.parametrize("device", DEVICES)
def test_extract_sig_coef_all(device):
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension), device=device)
    sig = pysiglib.sig(x, degree)
    words = pysiglib.words(dimension, degree)[1:]  # exclude empty word
    coefs = pysiglib.extract_sig_coef(sig, words, dimension)
    assert_device(coefs, device)
    check_close(sig, coefs)

@pytest.mark.parametrize("device", DEVICES)
def test_extract_sig_coef_lyndon(device):
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension), device=device)
    pysiglib.prepare_log_sig(dimension, degree, method=1)
    log_sig_full = pysiglib.log_sig(x, degree, method=0)
    log_sig = pysiglib.log_sig(x, degree, method=1)
    words = pysiglib.lyndon_words(dimension, degree)
    coefs = pysiglib.extract_sig_coef(log_sig_full, words, dimension)
    assert_device(coefs, device)
    check_close(log_sig, coefs)

def get_true_sig_coefs(multi_indices, X, *args, **kwargs):
    dim = X.shape[-1]
    sig = pysiglib.signature(X, *args, **kwargs)
    res = []
    for idx in multi_indices:
        # word_to_idx with scalar_term=False maps the word to its position in a
        # scalar_term=False sig, which matches pysiglib.signature's default output.
        flat_idx = pysiglib.word_to_idx(tuple(idx), dim)
        res.append(sig[..., flat_idx])
    return np.array(res).T

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_trivial(device):
    X = torch.tensor([[0., 0.], [1., 1.]], device=device)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert_device(result, device)
    check_close(result, [1., 1.])

    X = torch.tensor([[0., 0.]], device=device)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert_device(result, device)
    check_close(result, [0., 0.])

@pytest.mark.parametrize("device", DEVICES)
def test_batch_sig_coef_trivial(device):
    X = torch.tensor([[[0., 0.], [1., 1.]]], device=device)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert_device(result, device)
    check_close(result, [1., 1.])

    X = torch.tensor([[[0., 0.]]], device=device)
    result = pysiglib.sig_coef(X, [(0,), (1,)])
    assert_device(result, device)
    check_close(result, [0., 0.])

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef(device):
    X = torch.rand(size=(100, 3), dtype=torch.float64, device=device)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]

    true_coeffs = get_true_sig_coefs(multi_indices, X.cpu(), 5)
    coeff = pysiglib.sig_coef(X, multi_indices)
    assert_device(coeff, device)
    check_close(true_coeffs, coeff)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_prefixes(device):
    X = torch.rand(size=(100, 3), dtype=torch.float64, device=device)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]
    grid_idx = [(0,), (0,1), (2,), (2,1), (2,1,0), (1,)]

    true_coeffs = get_true_sig_coefs(grid_idx, X.cpu(), 5)
    coeff = pysiglib.sig_coef(X, multi_indices, prefixes = True)
    assert_device(coeff, device)
    check_close(true_coeffs, coeff)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_sig_coef_prefixes(device):
    X = torch.rand(size=(10, 100, 3), dtype=torch.float64, device=device)
    multi_indices = [(0, 1), (2, 1, 0), (1,)]
    grid_idx = [(0,), (0,1), (2,), (2,1), (2,1,0), (1,)]

    true_coeffs = get_true_sig_coefs(grid_idx, X.cpu(), 5)
    coeff = pysiglib.sig_coef(X, multi_indices, prefixes = True)
    assert_device(coeff, device)
    check_close(true_coeffs, coeff)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_full(device):
    X = torch.rand(size=(100, 3), dtype=torch.float64, device=device)
    multi_indices = pysiglib.words(3, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices)
    assert_device(coeff, device)
    sig = pysiglib.signature(X, 5)
    check_close(sig, coeff)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_sig_coef_full(device):
    X = torch.rand(size=(10, 100, 3), device=device, dtype=torch.float64)
    multi_indices = pysiglib.words(3, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices)
    assert_device(coeff, device)
    sig = pysiglib.signature(X, 5)
    check_close(sig, coeff)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_sig_coef_full_time_aug(device):
    X = torch.rand(size=(10, 100, 3), device=device, dtype=torch.float64)
    multi_indices = pysiglib.words(4, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices, time_aug = True)
    assert_device(coeff, device)
    sig = pysiglib.signature(X, 5, time_aug = True)
    check_close(sig, coeff)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_sig_coef_full_lead_lag(device):
    X = torch.rand(size=(10, 100, 3), device=device, dtype=torch.float64)
    multi_indices = pysiglib.words(6, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices, lead_lag = True)
    assert_device(coeff, device)
    sig = pysiglib.signature(X, 5, lead_lag = True)
    check_close(sig, coeff)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_sig_coef_full_time_aug_lead_lag(device):
    X = torch.rand(size=(10, 100, 3), device=device, dtype=torch.float64)
    multi_indices = pysiglib.words(7, 5)[1:]

    coeff = pysiglib.sig_coef(X, multi_indices, time_aug = True, lead_lag = True)
    assert_device(coeff, device)
    sig = pysiglib.signature(X, 5, time_aug = True, lead_lag = True)
    check_close(sig, coeff)
