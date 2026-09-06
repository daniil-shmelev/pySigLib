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
check_close = partial(_check_close, atol=1e-5)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop(device):
    X = torch.rand(size=(100, 3), dtype=torch.float64, device=device)
    word = (0,1,2)

    sig = pysiglib.sig(X, 3)
    derivs = torch.zeros_like(sig)
    derivs[pysiglib.word_to_idx(word, 3)] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, word, prefixes=True)
    derivs = torch.tensor([0.,0.,1.], dtype=torch.float64, device=device)
    d2 = pysiglib.sig_coef_backprop(X, word, coef, derivs)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_batch(device):
    X = torch.rand(size=(32, 100, 3), dtype=torch.float64, device=device)
    words = [(0,1,2), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = torch.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = torch.tensor([[0.,0.,1., 0., 0., 1.]]*32, dtype=torch.float64, device=device)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_time_aug_batch(device):
    X = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device)
    words = [(0,1,2), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3, time_aug = True)
    derivs = torch.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, time_aug = True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug = True)
    derivs = torch.tensor([[0.,0.,1., 0., 0., 1.]]*32, dtype=torch.float64, device=device)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug = True)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_lead_lag_batch(device):
    X = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device)
    words = [(0,1,3), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 4) for w in words]

    sig = pysiglib.sig(X, 3, lead_lag = True)
    derivs = torch.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, lead_lag = True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, lead_lag = True)
    derivs = torch.tensor([[0.,0.,1., 0., 0., 1.]]*32, dtype=torch.float64, device=device)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, lead_lag = True)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_time_aug_lead_lag_batch(device):
    X = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device)
    words = [(0,1,2,3), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 5) for w in words]

    sig = pysiglib.sig(X, 4, time_aug=True, lead_lag=True)
    derivs = torch.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 4, time_aug=True, lead_lag=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs = torch.tensor([[0.,0.,0.,1., 0., 0., 1.]]*32, dtype=torch.float64, device=device)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug=True, lead_lag=True)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_batch_2(device):
    X = torch.rand(size=(32, 100, 3), dtype=torch.float64, device=device)
    words = [(2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = torch.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = torch.tensor([[0., 0., 1.]]*32, dtype=torch.float64, device=device)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_batch_3(device):
    X = torch.rand(size=(32, 100, 3), dtype=torch.float64, device=device)
    words = [tuple(), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words if w]

    sig = pysiglib.sig(X, 3)
    derivs = torch.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = torch.tensor([[1., 0., 0., 1.]]*32, dtype=torch.float64, device=device)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_full_batch(device):
    X = torch.rand(size=(32, 100, 3), dtype=torch.float64, device=device)
    words = pysiglib.words(3, 3)

    sig = pysiglib.sig(X, 3)
    derivs = torch.ones_like(sig)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = torch.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_time_aug_full_batch(device):
    X = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device)
    words = pysiglib.words(3, 3)

    sig = pysiglib.sig(X, 3, time_aug=True)
    derivs = torch.ones_like(sig)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, time_aug=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug=True)
    derivs = torch.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug=True)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_lead_lag_full_batch(device):
    X = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device)
    words = pysiglib.words(4, 3)

    sig = pysiglib.sig(X, 3, lead_lag=True)
    derivs = torch.ones_like(sig)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, lead_lag=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, lead_lag=True)
    derivs = torch.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, lead_lag=True)
    assert_device(d2, device)

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop_time_aug_lead_lag_full_batch(device):
    X = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device)
    words = pysiglib.words(5, 3)

    sig = pysiglib.sig(X, 3, time_aug=True, lead_lag=True)
    derivs = torch.ones_like(sig)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, time_aug=True, lead_lag=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs = torch.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug=True, lead_lag=True)
    assert_device(d2, device)

    check_close(d1, d2)

###########################################################
## torch_api
###########################################################

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_torch_api_full(device):
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, device=device, requires_grad = True)
    X2 = X1.detach().clone().requires_grad_(True)
    words = pysiglib.words(3, 3)[1:]

    sig = pysiglib.torch_api.sig(X1, 3)
    assert_device(sig, device)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device=device)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words)
    assert_device(coef, device)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_time_aug_torch_api_full(device):
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, device=device, requires_grad = True)
    X2 = X1.detach().clone().requires_grad_(True)
    words = pysiglib.words(4, 3)[1:]

    sig = pysiglib.torch_api.sig(X1, 3, time_aug=True)
    assert_device(sig, device)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device=device)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, n_jobs = -1, time_aug=True)
    assert_device(coef, device)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_lead_lag_torch_api_full(device):
    X1 = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device, requires_grad = True)
    X2 = X1.detach().clone().requires_grad_(True)
    words = pysiglib.words(4, 3)[1:]

    sig = pysiglib.torch_api.sig(X1, 3, lead_lag=True)
    assert_device(sig, device)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device=device)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, lead_lag=True, n_jobs = -1)
    assert_device(coef, device)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_time_aug_lead_lag_torch_api_full(device):
    X1 = torch.rand(size=(32, 100, 2), dtype=torch.float64, device=device, requires_grad = True)
    X2 = X1.detach().clone().requires_grad_(True)
    words = pysiglib.words(5, 3)[1:]

    sig = pysiglib.torch_api.sig(X1, 3, time_aug=True, lead_lag=True)
    assert_device(sig, device)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device=device)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, time_aug=True, lead_lag=True, n_jobs = -1)
    assert_device(coef, device)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
def test_extract_sig_coef_torch_api_full(device):
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, device=device, requires_grad = True)
    X2 = X1.detach().clone().requires_grad_(True)
    words = pysiglib.words(3, 2)[1:]

    sig = pysiglib.torch_api.sig(X1, 3)
    assert_device(sig, device)
    derivs1 = torch.zeros(sig.shape, device=device)
    derivs1[:, :pysiglib.sig_length(3,2)] = 1.
    sig.backward(derivs1)
    d1 = X1.grad

    sig2 = pysiglib.torch_api.sig(X2, 3)
    assert_device(sig2, device)
    coef = pysiglib.torch_api.extract_sig_coef(sig2, words, dimension = 3)
    assert_device(coef, device)
    coef.backward(torch.ones_like(coef))
    d2 = X2.grad

    check_close(d1, d2)
