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

EPSILON = 1e-5

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def test_sig_coef_backprop():
    X = np.random.uniform(size=(100, 3))
    word = (0,1,2)

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[pysiglib.word_to_idx(word, 3)] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, word, prefixes=True)
    derivs = np.array([0.,0.,1.])
    d2 = pysiglib.sig_coef_backprop(X, word, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_batch():
    X = np.random.uniform(size=(32, 100, 3))
    words = [(0,1,2), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.array([[0.,0.,1., 0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_time_aug_batch():
    X = np.random.uniform(size=(32, 100, 2))
    words = [(0,1,2), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3, time_aug = True)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, time_aug = True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug = True)
    derivs = np.array([[0.,0.,1., 0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug = True)

    check_close(d1, d2)

def test_sig_coef_backprop_lead_lag_batch():
    X = np.random.uniform(size=(32, 100, 2))
    words = [(0,1,3), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 4) for w in words]

    sig = pysiglib.sig(X, 3, lead_lag = True)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, lead_lag = True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, lead_lag = True)
    derivs = np.array([[0.,0.,1., 0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, lead_lag = True)

    check_close(d1, d2)

def test_sig_coef_backprop_time_aug_lead_lag_batch():
    X = np.random.uniform(size=(32, 100, 2))
    words = [(0,1,2,3), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 5) for w in words]

    sig = pysiglib.sig(X, 4, time_aug=True, lead_lag=True)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 4, time_aug=True, lead_lag=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs = np.array([[0.,0.,0.,1., 0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug=True, lead_lag=True)

    check_close(d1, d2)

def test_sig_coef_backprop_batch_2():
    X = np.random.uniform(size=(32, 100, 3))
    words = [(2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.array([[0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_batch_3():
    X = np.random.uniform(size=(32, 100, 3))
    words = [tuple(), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.array([[1., 0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_full_batch():
    X = np.random.uniform(size=(32, 100, 3))
    words = pysiglib.words(3, 3)

    sig = pysiglib.sig(X, 3)
    derivs = np.ones(sig.shape)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_time_aug_full_batch():
    X = np.random.uniform(size=(32, 100, 2))
    words = pysiglib.words(3, 3)

    sig = pysiglib.sig(X, 3, time_aug=True)
    derivs = np.ones(sig.shape)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, time_aug=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug=True)
    derivs = np.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug=True)

    check_close(d1, d2)

def test_sig_coef_backprop_lead_lag_full_batch():
    X = np.random.uniform(size=(32, 100, 2))
    words = pysiglib.words(4, 3)

    sig = pysiglib.sig(X, 3, lead_lag=True)
    derivs = np.ones(sig.shape)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, lead_lag=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, lead_lag=True)
    derivs = np.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, lead_lag=True)

    check_close(d1, d2)

def test_sig_coef_backprop_time_aug_lead_lag_full_batch():
    X = np.random.uniform(size=(32, 100, 2))
    words = pysiglib.words(5, 3)

    sig = pysiglib.sig(X, 3, time_aug=True, lead_lag=True)
    derivs = np.ones(sig.shape)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3, time_aug=True, lead_lag=True)

    coef = pysiglib.sig_coef(X, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs = np.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs, time_aug=True, lead_lag=True)

    check_close(d1, d2)

###########################################################
## torch_api
###########################################################

def test_sig_coef_torch_api_full():
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(3, 3)

    sig = pysiglib.torch_api.sig(X1, 3)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, n_jobs = -1)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

def test_sig_coef_time_aug_torch_api_full():
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(4, 3)

    sig = pysiglib.torch_api.sig(X1, 3, time_aug=True)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, n_jobs = -1, time_aug=True)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

def test_sig_coef_lead_lag_torch_api_full():
    X1 = torch.rand(size=(32, 100, 2), dtype=torch.float64, requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(4, 3)

    sig = pysiglib.torch_api.sig(X1, 3, lead_lag=True)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, lead_lag=True, n_jobs = -1)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

def test_sig_coef_time_aug_lead_lag_torch_api_full():
    X1 = torch.rand(size=(32, 100, 2), dtype=torch.float64, requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(5, 3)

    sig = pysiglib.torch_api.sig(X1, 3, time_aug=True, lead_lag=True)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape)
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, time_aug=True, lead_lag=True, n_jobs = -1)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    check_close(s1, s2)
    check_close(d1, d2)

@pytest.mark.skipif(not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()), reason="CUDA not available or disabled")
def test_sig_coef_torch_api_full_cuda():
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, device="cuda", requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(3, 3)

    sig = pysiglib.torch_api.sig(X1, 3)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device="cuda")
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, n_jobs = -1)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    assert coef.device.type == "cuda"
    assert d2.device.type == "cuda"

    check_close(s1.cpu(), s2.cpu())
    check_close(d1.cpu(), d2.cpu())
