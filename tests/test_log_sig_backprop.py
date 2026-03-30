# Copyright 2025 Daniil Shmelev
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

try:
    import signatory
except:
    signatory = None

import pysiglib.torch_api as pysiglib
from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, double_atol=1e-5)

np.random.seed(42)
torch.manual_seed(42)

@pytest.mark.skipif(signatory is None, reason="signatory not available")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_to_log_sig_backprop_expanded_random(device, deg, dtype):
    X = torch.rand(size=(1, pysiglib.sig_length(1, deg)), requires_grad=True, dtype=dtype, device=device)
    ls = pysiglib.sig_to_log_sig(X, 1, deg, method=0)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, dtype=dtype, device=device)
    ls.backward(derivs)
    d1 = X.grad[:, 1:]

    X = X.clone().detach()[:, 1:].cpu()
    X = torch.tensor(X, requires_grad=True)
    derivs = derivs[:, 1:].cpu()
    ls = signatory.signature_to_logsignature(X, 1, deg, mode="expand")
    ls.backward(derivs)
    d2 = X.grad
    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_backprop_expanded_random(device, deg, dtype):
    X = torch.rand(size=(100, 5), requires_grad=True, dtype=dtype, device=device)
    ls = pysiglib.log_sig(X, deg, method=0)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, dtype=dtype, device=device)
    ls.backward(derivs)
    d1 = X.grad

    X = np.array(X.detach().cpu())
    derivs = np.array(derivs.cpu())[1:]
    s = iisignature.prepare(5, deg, "x")
    d2 = iisignature.logsigbackprop(derivs, X, s, "x")

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_backprop_expanded_random(device, deg, dtype):
    X = torch.rand(size=(32, 100, 5), requires_grad=True, dtype=dtype, device=device)
    ls = pysiglib.log_sig(X, deg, method=0)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, dtype=dtype, device=device)
    ls.backward(derivs)
    d1 = X.grad

    X = np.array(X.detach().cpu())
    derivs = np.array(derivs.cpu())[:, 1:]
    s = iisignature.prepare(5, deg, "x")
    d2 = iisignature.logsigbackprop(derivs, X, s, "x")

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_backprop_expanded_time_aug_random(device, deg, dtype):
    X = torch.rand(size=(32, 100, 2), requires_grad=True, dtype=dtype, device=device)
    ls = pysiglib.log_sig(X, deg, time_aug=True, method=0)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, dtype=dtype, device=device)
    ls.backward(derivs)
    d1 = X.grad

    X = np.array(X.detach().cpu())
    X = pysiglib.transform_path(X, time_aug=True)
    derivs = np.array(derivs.cpu())[:, 1:]
    s = iisignature.prepare(3, deg, "x")
    d2 = iisignature.logsigbackprop(derivs, X, s, "x")[:, :, :-1]

    check_close(d1, d2)

@pytest.mark.skipif(signatory is None, reason="signatory not available")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_lyndon_words_random(device, deg, dtype):
    X = torch.rand(size=(1, 100, 5), dtype=dtype, device=device, requires_grad=True)
    pysiglib.prepare_log_sig(5, deg, 1)
    ls = pysiglib.log_sig(X[0], deg, method=1)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, device=device)
    ls.backward(derivs)
    d2 = X.grad

    X = torch.tensor(X.clone().detach().cpu(), requires_grad=True)
    ls = signatory.logsignature(X, deg, mode="words")[0]
    ls.backward(derivs.cpu())
    d1 = X.grad

    check_close(d1, d2)

@pytest.mark.skipif(signatory is None, reason="signatory not available")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_lyndon_words_random(device, deg, dtype):
    X = torch.rand(size=(32, 100, 5), dtype=dtype, device=device, requires_grad=True)
    pysiglib.prepare_log_sig(5, deg, 1)
    ls = pysiglib.log_sig(X[0], deg, method=1)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, device=device)
    ls.backward(derivs)
    d2 = X.grad

    X = torch.tensor(X.clone().detach().cpu(), requires_grad=True)
    ls = signatory.logsignature(X, deg, mode="words")[0]
    ls.backward(derivs.cpu())
    d1 = X.grad

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("method", [2, 3])
def test_log_signature_backprop_lyndon_basis_random(device, deg, dtype, method):
    X = torch.rand(size=(100, 5), requires_grad=True, dtype=dtype, device=device)
    pysiglib.prepare_log_sig(5, deg, 2)
    ls = pysiglib.log_sig(X, deg, method=method)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, dtype=dtype, device=device)
    ls.backward(derivs)
    d1 = X.grad

    X = np.array(X.detach().cpu())
    derivs = np.array(derivs.cpu())
    s = iisignature.prepare(5, deg, "s")
    d2 = iisignature.logsigbackprop(derivs, X, s, "s")

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("method", [2, 3])
def test_batch_log_signature_backprop_lyndon_basis_random(device, deg, dtype, method):
    X = torch.rand(size=(32, 100, 5), requires_grad=True, dtype=dtype, device=device)
    pysiglib.prepare_log_sig(5, deg, 2)
    ls = pysiglib.log_sig(X, deg, method=method)
    assert_device(ls, device)
    derivs = torch.rand(size=ls.shape, dtype=dtype, device=device)
    ls.backward(derivs)
    d1 = X.grad

    X = np.array(X.detach().cpu())
    derivs = np.array(derivs.cpu())
    s = iisignature.prepare(5, deg, "s")
    d2 = iisignature.logsigbackprop(derivs, X, s, "s")

    check_close(d1, d2, single_atol=1e-3)
