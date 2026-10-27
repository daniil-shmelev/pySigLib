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

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, atol=1e-5)

def _sig_coef_reference(path, words, time_aug=False, lead_lag=False, end_time=1.):
    if lead_lag:
        lag = torch.repeat_interleave(path, 2, dim=-2)[..., :-1, :]
        lead = torch.repeat_interleave(path, 2, dim=-2)[..., 1:, :]
        path = torch.cat((lag, lead), dim=-1)
    if time_aug:
        times = torch.linspace(0., end_time, path.shape[-2], dtype=path.dtype, device=path.device)
        times = times.expand(*path.shape[:-1]).unsqueeze(-1)
        path = torch.cat((path, times), dim=-1)
    increments = path[..., 1:, :] - path[..., :-1, :]
    one = torch.ones_like(path[..., 0, 0])
    result = []
    for word in words:
        coefs = [one] + [torch.zeros_like(one) for _ in word]
        for t in range(increments.shape[-2]):
            updated = [one]
            for i in range(1, len(word) + 1):
                product = one
                value = coefs[i]
                for j in range(i - 1, -1, -1):
                    product = product * increments[..., t, word[j]] / (i - j)
                    value = value + coefs[j] * product
                updated.append(value)
            coefs = updated
        result.extend(coefs[1:] if word else [one])
    return torch.stack(result, dim=-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degree", [1, 2, 5, 12, 20, 21])
@pytest.mark.parametrize("mixed", [False, True])
def test_sig_coef_backprop_horner_reference(device, dtype, degree, mixed):
    if device == "cuda" and degree > 20:
        pytest.skip("CUDA coefficient degree limit is 20")
    path = torch.tensor(
        [[0., 0.], [0.7, -0.4], [0.7, 0.3], [-0.2, 0.3], [0.1, -0.1]],
        dtype=dtype, device=device, requires_grad=True)
    word = tuple(i % 2 for i in range(degree))
    words = [word, (1, 0, 1), (), word] if mixed else [word, word]
    reference = _sig_coef_reference(path, words)
    weights = torch.linspace(-0.7, 1.1, reference.shape[-1], dtype=dtype, device=device)
    expected, = torch.autograd.grad(reference, path, weights)
    coefs = pysiglib.sig_coef(path, words, prefixes=True)
    actual = pysiglib.sig_coef_backprop(path, words, coefs, weights.clone())
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(coefs, reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert_device(actual, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("time_aug,lead_lag", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("prefixes", [False, True])
def test_sig_coef_backprop_horner_batch_views(device, dtype, time_aug, lead_lag, prefixes):
    rng = np.random.default_rng(721)
    storage = torch.tensor(rng.normal(scale=0.2, size=(2, 3, 5, 4)), dtype=dtype, device=device)
    storage[..., 2, :] = storage[..., 1, :]
    path = storage[..., ::2].detach().requires_grad_(True)
    dimension = (4 if lead_lag else 2) + int(time_aug)
    words = [tuple(i % dimension for i in range(8)), (1, 1, 0), ()]
    options = dict(time_aug=time_aug, lead_lag=lead_lag, end_time=1.7)
    reference = _sig_coef_reference(path, words, **options)
    weights_storage = torch.tensor(
        rng.normal(size=(*reference.shape[:-1], 2 * reference.shape[-1])), dtype=dtype, device=device)
    weights = weights_storage[..., ::2]
    expected, = torch.autograd.grad(reference, path, weights, retain_graph=True)
    coefs = pysiglib.sig_coef(path, words, prefixes=True, **options)
    actual = pysiglib.sig_coef_backprop(path, words, coefs, weights, n_jobs=-1, **options)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    if not prefixes:
        indices = np.cumsum([len(word) if word else 1 for word in words]) - 1
        reference = reference[..., indices]
    weights = torch.ones_like(reference)
    expected, = torch.autograd.grad(reference, path, weights)
    output = pysiglib.torch_api.sig_coef(path, words, prefixes=prefixes, **options)
    actual, = torch.autograd.grad(output, path, weights)
    torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.shape == path.shape
    assert_device(actual, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 0, 5, 2), (2, 3, 0, 2), (2, 3, 1, 2)])
def test_sig_coef_backprop_horner_empty(device, dtype, shape):
    path = torch.zeros(shape, dtype=dtype, device=device)
    words = [(0, 1) * 4, (1,)]
    coefs = pysiglib.sig_coef(path, words, prefixes=True)
    actual = pysiglib.sig_coef_backprop(path, words, coefs, torch.ones_like(coefs))
    torch.testing.assert_close(actual, torch.zeros_like(path))
    assert actual.shape == path.shape
    assert_device(actual, device)


@pytest.mark.parametrize("device", DEVICES)
def test_sig_coef_backprop(device):
    X = torch.rand(size=(100, 3), dtype=torch.float64, device=device)
    word = (0,1,2)

    sig = pysiglib.sig(X, 3)
    derivs = torch.zeros_like(sig)
    derivs[pysiglib.word_to_idx(word, 3, len(word))] = 1.
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
    words_idx = [pysiglib.word_to_idx(w, 3, len(w)) for w in words]

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
    words_idx = [pysiglib.word_to_idx(w, 3, len(w)) for w in words]

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
    words_idx = [pysiglib.word_to_idx(w, 4, len(w)) for w in words]

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
    words_idx = [pysiglib.word_to_idx(w, 5, len(w)) for w in words]

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
    words_idx = [pysiglib.word_to_idx(w, 3, len(w)) for w in words]

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
    words_idx = [pysiglib.word_to_idx(w, 3, len(w)) for w in words if w]

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
    words = pysiglib.words(3, 3, scalar_term=True)

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
    words = pysiglib.words(3, 3, scalar_term=True)

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
    words = pysiglib.words(4, 3, scalar_term=True)

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
    words = pysiglib.words(5, 3, scalar_term=True)

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
    words = pysiglib.words(3, 3)

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
    words = pysiglib.words(4, 3)

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
    words = pysiglib.words(4, 3)

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
    words = pysiglib.words(5, 3)

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
    words = pysiglib.words(3, 2)

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
