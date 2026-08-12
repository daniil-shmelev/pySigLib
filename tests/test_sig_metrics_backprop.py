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

from copy import deepcopy

import pytest
import numpy as np
import torch

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, single_atol=1e-3, double_atol=1e-3)

def sig_score_fd(x1, x2, dyadic_order, time_aug = False, lead_lag = False):
    x1 = x1.to(device = "cpu", dtype = torch.double)
    x2 = x2.to(device = "cpu", dtype = torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-10
    k = pysiglib.sig_score(x1, x2, dyadic_order=dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
    out = np.empty(shape = (batch_size, length, dim))

    for b in range(batch_size):
        for i in range(length):
            for d in range(dim):
                x1_d = deepcopy(x1)
                x1_d[b,i,d] += eps
                k_d = pysiglib.sig_score(x1_d, x2, dyadic_order=dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
                out[b,i,d] = (k_d - k) / eps
    return out

def expected_sig_score_fd(x1, x2, dyadic_order, time_aug = False, lead_lag = False):
    x1 = x1.to(device = "cpu", dtype = torch.double)
    x2 = x2.to(device = "cpu", dtype = torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-10
    k = pysiglib.expected_sig_score(x1, x2, dyadic_order=dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
    out = np.empty(shape = (batch_size, length, dim))

    for b in range(batch_size):
        for i in range(length):
            for d in range(dim):
                x1_d = deepcopy(x1)
                x1_d[b,i,d] += eps
                k_d = pysiglib.expected_sig_score(x1_d, x2, dyadic_order=dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
                out[b,i,d] = (k_d - k) / eps
    return out

def sig_mmd_fd(x1, x2, dyadic_order, time_aug = False, lead_lag = False):
    x1 = x1.to(device = "cpu", dtype = torch.double)
    x2 = x2.to(device = "cpu", dtype = torch.double)
    if len(x1.shape) == 2:
        x1 = x1[None, :, :]
        x2 = x2[None, :, :]
    batch_size = x1.shape[0]
    length = x1.shape[1]
    dim = x1.shape[2]

    eps = 1e-10
    k = pysiglib.sig_mmd(x1, x2, dyadic_order=dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
    out = np.empty(shape = (batch_size, length, dim))

    for b in range(batch_size):
        for i in range(length):
            for d in range(dim):
                x1_d = deepcopy(x1)
                x1_d[b,i,d] += eps
                k_d = pysiglib.sig_mmd(x1_d, x2, dyadic_order=dyadic_order, time_aug = time_aug, lead_lag = lead_lag)
                out[b,i,d] = (k_d - k) / eps
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_score_backprop_random(device, dyadic_order):
    batch, len1, len2, dim = 8, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double, requires_grad = True)
    Y = torch.rand(size=(1, len2, dim), device=device, dtype = torch.double)

    d1 = sig_score_fd(X.detach(), Y, dyadic_order)
    k = pysiglib.torch_api.sig_score(X, Y, dyadic_order=dyadic_order)
    assert_device(k, device)
    k.backward()
    d2 = X.grad

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_score_backprop_random_batch(device, dyadic_order):
    batch, len1, len2, dim = 8, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double, requires_grad = True)
    Y = torch.rand(size=(4, len2, dim), device=device, dtype = torch.double)

    d1 = torch.from_numpy(np.array([sig_score_fd(X.detach(), Y[i], dyadic_order) for i in range(4)])).sum(0)
    k = pysiglib.torch_api.sig_score(X, Y, dyadic_order=dyadic_order)
    assert_device(k, device)
    k.backward(torch.ones(4, device=device))
    d2 = X.grad

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_backprop_random(device, dyadic_order):
    batch, len1, len2, dim = 8, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double, requires_grad = True)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    d1 = expected_sig_score_fd(X.detach(), Y, dyadic_order)
    k = pysiglib.torch_api.expected_sig_score(X, Y, dyadic_order=dyadic_order)
    assert_device(k, device)
    k.backward()
    d2 = X.grad

    check_close(d1, d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_backprop_random(device, dyadic_order):
    batch, len1, len2, dim = 8, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double, requires_grad = True)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    d1 = sig_mmd_fd(X.detach(), Y, dyadic_order)
    k = pysiglib.torch_api.sig_mmd(X, Y, dyadic_order=dyadic_order)
    assert_device(k, device)
    k.backward()
    d2 = X.grad

    check_close(d1, d2)
