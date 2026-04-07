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

import pysiglib

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device, load_fixtures
check_close = partial(_check_close, double_atol=1e-5)

FIXTURES = load_fixtures("reference_data.npz")

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
    t = torch.linspace(0, end_time, path.shape[0], device=x.device).unsqueeze(1)
    path = torch.cat((path, t), dim =  1)
    return path

def batch_time_aug_lead_lag(x, end_time = 1.):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    t = torch.linspace(0, end_time, path.shape[1], device=x.device).unsqueeze(0)
    t = torch.tile(t, (path.shape[0], 1)).unsqueeze(2)
    path = torch.cat((path, t), dim=2)
    return path

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_backprop_random(device, deg, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X = torch.tensor(FIXTURES["path"][0].astype(np_dtype), device=device, dtype=dtype)
    sig_derivs = torch.tensor(FIXTURES[f"sig_bp_derivs__d{deg}"][0], device=device, dtype=dtype)
    expected = FIXTURES[f"sig_bp_expected__d{deg}"][0]

    sig = pysiglib.sig(X, deg)

    sig_back = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg)
    assert_device(sig_back, device)
    check_close(sig_back, expected)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_batch_sig_backprop_random(device, deg):
    X = torch.tensor(FIXTURES["path"], device=device, dtype=torch.float64)
    sig_derivs = torch.tensor(FIXTURES[f"sig_bp_derivs__d{deg}"], device=device, dtype=torch.float64)
    expected = FIXTURES[f"sig_bp_expected__d{deg}"]

    sig = pysiglib.sig(X, deg)

    sig_back_serial = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, n_jobs=1)
    sig_back_parallel = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, n_jobs=-1)
    assert_device(sig_back_serial, device)
    assert_device(sig_back_parallel, device)
    check_close(sig_back_serial, expected)
    check_close(sig_back_parallel, expected)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 5))
def test_sig_backprop_time_aug_random(device, deg):
    length, dimension = 100, 5
    X = torch.rand(size=(length, dimension), dtype=torch.float64, device=device)
    t = torch.linspace(0, 1, length, device=device).unsqueeze(1)
    X_time_aug = torch.cat([X, t], dim=1)
    sig_derivs = torch.rand(size=(pysiglib.sig_length(dimension + 1, deg),), dtype=torch.float64, device=device)

    sig = pysiglib.sig(X, deg, time_aug = True)

    sig_back1 = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, time_aug = True)
    assert_device(sig_back1, device)
    sig_back2 = pysiglib.sig_backprop(X_time_aug.clone(), sig.clone(), sig_derivs.clone(), deg)[:, :-1]
    check_close(sig_back1, sig_back2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_time_aug_random(device, deg):
    batch_size, length, dimension = 10, 100, 5
    X = torch.rand(size=(batch_size, length, dimension), dtype=torch.float64, device=device)
    t = torch.linspace(0, 1, length, device=device).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1)
    X_time_aug = torch.cat([X, t], dim=2)
    sig_derivs = torch.rand(size=(batch_size, pysiglib.sig_length(dimension + 1, deg)), dtype=torch.float64, device=device)

    sig = pysiglib.sig(X, deg, time_aug = True)

    sig_back1 = pysiglib.sig_backprop(X.clone(), sig.clone(), sig_derivs.clone(), deg, time_aug = True)
    assert_device(sig_back1, device)
    sig_back2 = pysiglib.sig_backprop(X_time_aug.clone(), sig.clone(), sig_derivs.clone(), deg)[:, :, :-1]
    check_close(sig_back1, sig_back2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 5))
def test_sig_backprop_lead_lag_random(device, deg):
    length, dimension = 100, 5
    X = torch.rand(size=(length, dimension), dtype=torch.float64, device=device, requires_grad=True)
    X_ll = lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = torch.rand(size=(pysiglib.sig_length(dimension * 2, deg),), dtype=torch.float64, device=device)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, lead_lag = True)
    assert_device(sig_back2, device)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_lead_lag_random(device, deg):
    batch_size, length, dimension = 10, 100, 5
    X = torch.rand(size=(batch_size, length, dimension), dtype=torch.float64, device=device, requires_grad=True)
    X_ll = batch_lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = torch.rand(size=(batch_size, pysiglib.sig_length(dimension * 2, deg)), dtype=torch.float64, device=device)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, lead_lag = True)
    assert_device(sig_back2, device)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sig_backprop_time_aug_lead_lag_random(device, deg, dtype):
    length, dimension = 100, 5
    X = torch.rand(size=(length, dimension), dtype=torch.float64, device=device, requires_grad=True)
    X_ll = time_aug_lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = torch.rand(size=(pysiglib.sig_length(dimension * 2 + 1, deg),), dtype=torch.float64, device=device)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, time_aug = True, lead_lag = True)
    assert_device(sig_back2, device)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 5))
def test_batch_sig_backprop_time_aug_lead_lag_random(device, deg):
    batch_size, length, dimension = 10, 100, 5
    X = torch.rand(size=(batch_size, length, dimension), dtype=torch.float64, device=device, requires_grad=True)
    X_ll = batch_time_aug_lead_lag(X)
    sig = pysiglib.sig(X_ll, deg)
    sig_derivs = torch.rand(size=(batch_size, pysiglib.sig_length(dimension * 2 + 1, deg)), dtype=torch.float64, device=device)

    sig_back1 = pysiglib.sig_backprop(X_ll, sig, sig_derivs, deg)
    sig_back2 = pysiglib.sig_backprop(X, sig, sig_derivs, deg, time_aug = True, lead_lag = True)
    assert_device(sig_back2, device)

    grad_input1, = torch.autograd.grad(X_ll, X, sig_back1, False, True)

    check_close(grad_input1, sig_back2)
