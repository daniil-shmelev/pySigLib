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

try:
    import iisignature
    HAS_IISIGNATURE = True
except ImportError:
    HAS_IISIGNATURE = False

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

SINGLE_EPSILON = 1e-4
DOUBLE_EPSILON = 1e-5

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    EPSILON = SINGLE_EPSILON if a_.dtype == np.float32 else DOUBLE_EPSILON
    assert not np.max(np.abs(a_ - b_)) > EPSILON

def lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:-1], repeats=2, dim=0)
    lag = torch.cat((lag, x[-1:]))
    lead = torch.repeat_interleave(x[1:], repeats=2, dim=0)
    lead = torch.cat((x[0:1], lead))
    path = torch.cat((lag, lead), dim=-1)
    return path

def batch_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    return path

def time_aug_lead_lag(x, end_time = 1.):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:-1], repeats=2, dim=0)
    lag = torch.cat((lag, x[-1:]))
    lead = torch.repeat_interleave(x[1:], repeats=2, dim=0)
    lead = torch.cat((x[0:1], lead))
    path = torch.cat((lag, lead), dim=-1)
    t = torch.linspace(0, end_time, path.shape[0]).unsqueeze(1)
    path = torch.cat((path, t), dim =  1)
    return path

def batch_time_aug_lead_lag(x, end_time = 1.):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    t = torch.linspace(0, end_time, path.shape[1]).unsqueeze(0)
    t = torch.tile(t, (path.shape[0], 1)).unsqueeze(2)
    path = torch.cat((path, t), dim=2)
    return path

@pytest.mark.skipif(not HAS_IISIGNATURE, reason="iisignature not available")
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sig_backprop_random(deg, dtype):
    X = np.random.uniform(size=(100, 5)).astype(dtype)
    sig_derivs = np.random.uniform(size = pysiglib.sig_length(5, deg)).astype(dtype)

    sig = pysiglib.sig(X, deg)

    sig_back1 = pysiglib.sig_backprop(X.copy(), sig.copy(), sig_derivs.copy(), deg)
    sig_back2 = iisignature.sigbackprop(sig_derivs[1:].copy(), X.copy(), deg)
    check_close(sig_back1, sig_back2)

@pytest.mark.skipif(not HAS_IISIGNATURE, reason="iisignature not available")
@pytest.mark.parametrize("deg", range(1, 6))
def test_batch_sig_backprop_random(deg):
    X = np.random.uniform(size=(100, 3, 2)).astype("double")
    sig_derivs = np.random.uniform(size = (100, pysiglib.sig_length(2, deg))).astype("double")

    sig = pysiglib.sig(X.copy(), deg)

    sig_back1 = pysiglib.sig_backprop(X.copy(), sig.copy(), sig_derivs.copy(), deg)
    sig_back2 = iisignature.sigbackprop(sig_derivs[:, 1:].copy(), X.copy(), deg)
    check_close(sig_back1, sig_back2)

@pytest.mark.parametrize("deg", range(1, 5))
def test_sig_backprop_time_aug_random(deg):
    length, dimension = 100, 5
    X = np.random.uniform(size=(length, dimension))
    t = np.linspace(0, 1, length)[:, np.newaxis]
    X_time_aug = np.concatenate([X, t], axis = 1)
    sig_derivs = np.random.uniform(size=pysiglib.sig_length(dimension + 1, deg))

    sig = pysiglib.sig(X, deg, time_aug = True)

    sig_back1 = pysiglib.sig_backprop(X.copy(), sig.copy(), sig_derivs.copy(), deg, time_aug = True)
    sig_back2 = pysiglib.sig_backprop(X_time_aug.copy(), sig.copy(), sig_derivs.copy(), deg)[:, :-1]
    check_close(sig_back1, sig_back2)

@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_time_aug_random(deg):
    batch_size, length, dimension = 10, 100, 5
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    t = np.linspace(0, 1, length)[np.newaxis, :, np.newaxis]
    t = np.tile(t, (batch_size, 1, 1))
    X_time_aug = np.concatenate([X, t], axis=2)
    sig_derivs = np.random.uniform(size = (batch_size, pysiglib.sig_length(dimension + 1, deg)))

    sig = pysiglib.sig(X.copy(), deg, time_aug = True)

    sig_back1 = pysiglib.sig_backprop(X.copy(), sig.copy(), sig_derivs.copy(), deg, time_aug = True)
    sig_back2 = pysiglib.sig_backprop(X_time_aug.copy(), sig.copy(), sig_derivs.copy(), deg)[:, :, :-1]
    check_close(sig_back1, sig_back2)

@pytest.mark.parametrize("deg", range(1, 5))
def test_sig_backprop_lead_lag_random(deg):
    length, dimension = 100, 5
    X = np.random.uniform(size=(length, dimension))
    X = torch.tensor(X, dtype = torch.float64, requires_grad = True)
    X_ll = lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = np.random.uniform(size=pysiglib.sig_length(dimension * 2, deg))
    sig_derivs = torch.tensor(sig_derivs)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, lead_lag = True)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)

@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_lead_lag_random(deg):
    batch_size, length, dimension = 10, 100, 5
    X = np.random.uniform(size=(batch_size, length, dimension))
    X = torch.tensor(X, dtype = torch.float64, requires_grad = True)
    X_ll = batch_lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = np.random.uniform(size=(batch_size, pysiglib.sig_length(dimension * 2, deg)))
    sig_derivs = torch.tensor(sig_derivs)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, lead_lag = True)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)

@pytest.mark.parametrize("deg", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sig_backprop_time_aug_lead_lag_random(deg, dtype):
    length, dimension = 100, 5
    X = np.random.uniform(size=(length, dimension)).astype(dtype)
    X = torch.tensor(X, dtype = torch.float64, requires_grad = True)
    X_ll = time_aug_lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = np.random.uniform(size=pysiglib.sig_length(dimension * 2 + 1, deg))
    sig_derivs = torch.tensor(sig_derivs)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, time_aug = True, lead_lag = True)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)

@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_time_aug_lead_lag_random(deg):
    batch_size, length, dimension = 10, 100, 5
    X = np.random.uniform(size=(batch_size, length, dimension))
    X = torch.tensor(X, dtype = torch.float64, requires_grad = True)
    X_ll = batch_time_aug_lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = np.random.uniform(size=(batch_size, pysiglib.sig_length(dimension * 2 + 1, deg)))
    sig_derivs = torch.tensor(sig_derivs)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, time_aug = True, lead_lag = True)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)
