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

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

###########################################################
## direct backprop tests
###########################################################

@skip_no_cuda
def test_sig_coef_backprop_cuda():
    X_np = np.random.uniform(size=(100, 3))
    word = (0,1,2)

    # CPU reference
    sig = pysiglib.sig(X_np, 3)
    derivs = np.zeros_like(sig)
    derivs[pysiglib.word_to_idx(word, 3)] = 1.
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3)

    # CUDA
    X_cuda = torch.tensor(X_np, device="cuda")
    sig_cuda = pysiglib.sig(X_cuda, 3)
    derivs_cuda = torch.zeros_like(sig_cuda)
    derivs_cuda[pysiglib.word_to_idx(word, 3)] = 1.
    d_cuda = pysiglib.sig_backprop(X_cuda, sig_cuda, derivs_cuda, 3)

    coef_cpu = pysiglib.sig_coef(X_np, word, prefixes=True)
    derivs_coef = np.array([0.,0.,1.])
    d2_cpu = pysiglib.sig_coef_backprop(X_np, word, coef_cpu, derivs_coef)

    coef_cuda = pysiglib.sig_coef(X_cuda, word, prefixes=True)
    derivs_coef_cuda = torch.tensor([0.,0.,1.], device="cuda", dtype=torch.float64)
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, word, coef_cuda, derivs_coef_cuda)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 3))
    words = [(0,1,2), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    # CPU reference
    sig = pysiglib.sig(X_np, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True)
    derivs_coef = np.array([[0.,0.,1., 0., 0., 1.]]*32)
    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef)

    # CUDA
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True)
    derivs_coef_cuda = torch.tensor([[0.,0.,1., 0., 0., 1.]]*32, device="cuda", dtype=torch.float64)
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_time_aug_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 2))
    words = [(0,1,2), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    # CPU reference
    sig = pysiglib.sig(X_np, 3, time_aug=True)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3, time_aug=True)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True, time_aug=True)
    derivs_coef = np.array([[0.,0.,1., 0., 0., 1.]]*32)
    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef, time_aug=True)

    # CUDA
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True, time_aug=True)
    derivs_coef_cuda = torch.tensor([[0.,0.,1., 0., 0., 1.]]*32, device="cuda", dtype=torch.float64)
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda, time_aug=True)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_lead_lag_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 2))
    words = [(0,1,3), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 4) for w in words]

    # CPU reference
    sig = pysiglib.sig(X_np, 3, lead_lag=True)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3, lead_lag=True)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True, lead_lag=True)
    derivs_coef = np.array([[0.,0.,1., 0., 0., 1.]]*32)
    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef, lead_lag=True)

    # CUDA
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True, lead_lag=True)
    derivs_coef_cuda = torch.tensor([[0.,0.,1., 0., 0., 1.]]*32, device="cuda", dtype=torch.float64)
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda, lead_lag=True)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_time_aug_lead_lag_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 2))
    words = [(0,1,2,3), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 5) for w in words]

    # CPU reference
    sig = pysiglib.sig(X_np, 4, time_aug=True, lead_lag=True)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 4, time_aug=True, lead_lag=True)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs_coef = np.array([[0.,0.,0.,1., 0., 0., 1.]]*32)
    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef, time_aug=True, lead_lag=True)

    # CUDA
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs_coef_cuda = torch.tensor([[0.,0.,0.,1., 0., 0., 1.]]*32, device="cuda", dtype=torch.float64)
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda, time_aug=True, lead_lag=True)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_batch_2_cuda():
    X_np = np.random.uniform(size=(32, 100, 3))
    words = [(2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    # CPU reference
    sig = pysiglib.sig(X_np, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True)
    derivs_coef = np.array([[0., 0., 1.]]*32)
    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef)

    # CUDA
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True)
    derivs_coef_cuda = torch.tensor([[0., 0., 1.]]*32, device="cuda", dtype=torch.float64)
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_batch_3_cuda():
    X_np = np.random.uniform(size=(32, 100, 3))
    words = [tuple(), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    # CPU reference
    sig = pysiglib.sig(X_np, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True)
    derivs_coef = np.array([[1., 0., 0., 1.]]*32)
    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef)

    # CUDA
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True)
    derivs_coef_cuda = torch.tensor([[1., 0., 0., 1.]]*32, device="cuda", dtype=torch.float64)
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_full_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 3))
    words = pysiglib.words(3, 3)

    # CPU reference
    sig = pysiglib.sig(X_np, 3)
    derivs = np.ones(sig.shape)
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True)
    derivs_coef = np.zeros_like(coef_cpu)
    i = 0
    for w in words:
        i += len(w)
        derivs_coef[:, i] = 1.

    # Create CUDA tensor before CPU call (CPU backprop mutates derivs in-place)
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True)
    derivs_coef_cuda = torch.tensor(derivs_coef, device="cuda")

    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef)

    # CUDA
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_time_aug_full_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 2))
    words = pysiglib.words(3, 3)

    # CPU reference
    sig = pysiglib.sig(X_np, 3, time_aug=True)
    derivs = np.ones(sig.shape)
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3, time_aug=True)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True, time_aug=True)
    derivs_coef = np.zeros_like(coef_cpu)
    i = 0
    for w in words:
        i += len(w)
        derivs_coef[:, i] = 1.

    # Create CUDA tensor before CPU call (CPU backprop mutates derivs in-place)
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True, time_aug=True)
    derivs_coef_cuda = torch.tensor(derivs_coef, device="cuda")

    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef, time_aug=True)

    # CUDA
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda, time_aug=True)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_lead_lag_full_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 2))
    words = pysiglib.words(4, 3)

    # CPU reference
    sig = pysiglib.sig(X_np, 3, lead_lag=True)
    derivs = np.ones(sig.shape)
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3, lead_lag=True)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True, lead_lag=True)
    derivs_coef = np.zeros_like(coef_cpu)
    i = 0
    for w in words:
        i += len(w)
        derivs_coef[:, i] = 1.

    # Create CUDA tensor before CPU call (CPU backprop mutates derivs in-place)
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True, lead_lag=True)
    derivs_coef_cuda = torch.tensor(derivs_coef, device="cuda")

    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef, lead_lag=True)

    # CUDA
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda, lead_lag=True)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

@skip_no_cuda
def test_sig_coef_backprop_time_aug_lead_lag_full_batch_cuda():
    X_np = np.random.uniform(size=(32, 100, 2))
    words = pysiglib.words(5, 3)

    # CPU reference
    sig = pysiglib.sig(X_np, 3, time_aug=True, lead_lag=True)
    derivs = np.ones(sig.shape)
    d_cpu = pysiglib.sig_backprop(X_np, sig, derivs, 3, time_aug=True, lead_lag=True)

    coef_cpu = pysiglib.sig_coef(X_np, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs_coef = np.zeros_like(coef_cpu)
    i = 0
    for w in words:
        i += len(w)
        derivs_coef[:, i] = 1.

    # Create CUDA tensor before CPU call (CPU backprop mutates derivs in-place)
    X_cuda = torch.tensor(X_np, device="cuda")
    coef_cuda = pysiglib.sig_coef(X_cuda, words, prefixes=True, time_aug=True, lead_lag=True)
    derivs_coef_cuda = torch.tensor(derivs_coef, device="cuda")

    d2_cpu = pysiglib.sig_coef_backprop(X_np, words, coef_cpu, derivs_coef, time_aug=True, lead_lag=True)

    # CUDA
    d2_cuda = pysiglib.sig_coef_backprop(X_cuda, words, coef_cuda, derivs_coef_cuda, time_aug=True, lead_lag=True)

    assert d2_cuda.device.type == "cuda"
    check_close(d_cpu, d2_cpu)
    check_close(d_cpu, d2_cuda.cpu().numpy())

###########################################################
## torch_api
###########################################################

@skip_no_cuda
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

@skip_no_cuda
def test_sig_coef_time_aug_torch_api_full_cuda():
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, device="cuda", requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(4, 3)

    sig = pysiglib.torch_api.sig(X1, 3, time_aug=True)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device="cuda")
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, n_jobs = -1, time_aug=True)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    assert coef.device.type == "cuda"
    assert d2.device.type == "cuda"

    check_close(s1.cpu(), s2.cpu())
    check_close(d1.cpu(), d2.cpu())

@skip_no_cuda
def test_sig_coef_lead_lag_torch_api_full_cuda():
    X1 = torch.rand(size=(32, 100, 2), dtype=torch.float64, device="cuda", requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(4, 3)

    sig = pysiglib.torch_api.sig(X1, 3, lead_lag=True)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device="cuda")
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, lead_lag=True, n_jobs = -1)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    assert coef.device.type == "cuda"
    assert d2.device.type == "cuda"

    check_close(s1.cpu(), s2.cpu())
    check_close(d1.cpu(), d2.cpu())

@skip_no_cuda
def test_sig_coef_time_aug_lead_lag_torch_api_full_cuda():
    X1 = torch.rand(size=(32, 100, 2), dtype=torch.float64, device="cuda", requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(5, 3)

    sig = pysiglib.torch_api.sig(X1, 3, time_aug=True, lead_lag=True)
    s1 = sig.clone().detach()
    derivs1 = torch.ones(sig.shape, device="cuda")
    derivs2 = derivs1.clone()
    sig.backward(derivs1)
    d1 = X1.grad

    coef = pysiglib.torch_api.sig_coef(X2, words, time_aug=True, lead_lag=True, n_jobs = -1)
    s2 = coef.clone().detach()
    coef.backward(derivs2)
    d2 = X2.grad

    assert coef.device.type == "cuda"
    assert d2.device.type == "cuda"

    check_close(s1.cpu(), s2.cpu())
    check_close(d1.cpu(), d2.cpu())

@skip_no_cuda
def test_extract_sig_coef_torch_api_full_cuda():
    X1 = torch.rand(size=(32, 100, 3), dtype=torch.float64, device="cuda", requires_grad = True)
    X2 = torch.tensor(X1.clone().detach(), requires_grad = True)
    words = pysiglib.words(3, 2)

    sig = pysiglib.torch_api.sig(X1, 3)
    derivs1 = torch.zeros(sig.shape, device="cuda")
    derivs1[:, :pysiglib.sig_length(3,2)] = 1.
    sig.backward(derivs1)
    d1 = X1.grad

    sig2 = pysiglib.torch_api.sig(X2, 3)
    coef = pysiglib.torch_api.extract_sig_coef(sig2, words, dimension = 3)
    coef.backward(torch.ones_like(coef))
    d2 = X2.grad

    assert coef.device.type == "cuda"
    assert d2.device.type == "cuda"

    check_close(d1.cpu(), d2.cpu())
