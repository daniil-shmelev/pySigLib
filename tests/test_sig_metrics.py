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
import sigkernel

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

EPSILON = 1e-5

from functools import partial
from conftest import DEVICES, check_close as _check_close
check_close = partial(_check_close, atol=EPSILON)

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

def time_aug_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:-1], repeats=2, dim=0)
    lag = torch.cat((lag, x[-1:]))
    lead = torch.repeat_interleave(x[1:], repeats=2, dim=0)
    lead = torch.cat((x[0:1], lead))
    path = torch.cat((lag, lead), dim=-1)
    t = torch.linspace(0, path.shape[0] - 1, path.shape[0]).unsqueeze(1)
    path = torch.cat((path, t), dim =  1)
    return path

def batch_time_aug_lead_lag(x):
    # A backpropagatable version of lead-lag
    lag = torch.repeat_interleave(x[:, :-1], repeats=2, dim=1)
    lag = torch.cat((lag, x[:, -1:]), dim=1)
    lead = torch.repeat_interleave(x[:, 1:], repeats=2, dim=1)
    lead = torch.cat((x[:, 0:1], lead), axis=1)
    path = torch.cat((lag, lead), dim=2)
    t = torch.linspace(0, path.shape[1] - 1, path.shape[1]).unsqueeze(0)
    t = torch.tile(t, (path.shape[0], 1)).unsqueeze(2)
    path = torch.cat((path, t), dim=2)
    return path

def sig_kernel_full_grid(X1, X2, len1, len2, batch):
    result = np.ones(shape = (batch, len1, len2))
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, 0)
    for b in range(batch):
        for i in range(1, len1):
            for j in range(1, len2):
                XX1 = torch.tensor(X1[b, :i + 1, :][np.newaxis, :, :])
                XX2 = torch.tensor(X2[b, :j + 1, :][np.newaxis, :, :])
                result[b][i][j] = float(signature_kernel.compute_kernel(XX1, XX2, 100)[0])
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random(device, dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    d1 = float(signature_kernel.compute_expected_scoring_rule(X, Y, 100))
    d2 = float(pysiglib.expected_sig_score(X, Y, dyadic_order))

    assert not abs(d1 - d2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random_rbf(device, dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.RBFKernel(2.)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    d1 = float(signature_kernel.compute_expected_scoring_rule(X, Y, 100))
    d2 = float(pysiglib.expected_sig_score(X, Y, dyadic_order, static_kernel= pysiglib.RBFKernel(2.)))

    assert not abs(d1 - d2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(10, 50), (50, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random_non_square(device, len1, len2, dyadic_order):
    batch, dim = 32, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    d1 = float(signature_kernel.compute_expected_scoring_rule(X, Y, 100))
    d2 = float(pysiglib.expected_sig_score(X, Y, dyadic_order))

    assert not abs(d1 - d2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random(device, dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    mmd1 = float(signature_kernel.compute_mmd(X, Y, 100))
    mmd2 = float(pysiglib.sig_mmd(X, Y, dyadic_order))

    assert not abs(mmd1 - mmd2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random_rbf(device, dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.RBFKernel(2.)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    mmd1 = float(signature_kernel.compute_mmd(X, Y, 100))
    mmd2 = float(pysiglib.sig_mmd(X, Y, dyadic_order, static_kernel= pysiglib.RBFKernel(2.)))

    assert not abs(mmd1 - mmd2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(10, 50), (50, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random_non_square(device, len1, len2, dyadic_order):
    batch, dim = 32, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    mmd1 = float(signature_kernel.compute_mmd(X, Y, 100))
    mmd2 = float(pysiglib.sig_mmd(X, Y, dyadic_order))

    assert not abs(mmd1 - mmd2) > EPSILON
