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

from conftest import DEVICES, check_close, assert_device, load_fixtures

FIXTURES = load_fixtures("reference_data.npz")

@pytest.mark.parametrize("device", DEVICES)
def test_signature_trivial(device):
    check_close(pysiglib.sig(torch.tensor([[0., 0.], [1., 1.]], device=device), 1), [1., 1.])
    check_close(pysiglib.sig(torch.tensor([[0., 0.]], device=device), 1), [0., 0.])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_signature_random(device, deg, dtype):
    X = FIXTURES["path"][0].astype(dtype)
    expected = FIXTURES[f"sig__d{deg}"][0]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.sig(X_dev, deg)
    assert_device(sig, device)
    check_close(expected, sig)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_random_batch(device, deg):
    X = FIXTURES["path"]
    expected = FIXTURES[f"sig__d{deg}"]
    X_dev = torch.tensor(X, device=device)
    sig_serial = pysiglib.sig(X_dev, deg, n_jobs=1)
    sig_parallel = pysiglib.sig(X_dev, deg, n_jobs=-1)
    assert_device(sig_serial, device)
    assert_device(sig_parallel, device)
    check_close(expected, sig_serial)
    check_close(expected, sig_parallel)


@pytest.mark.parametrize("device", DEVICES)
def test_signature_non_contiguous(device):
    # Make sure signature works with any form of array
    dim, degree, length, batch = 10, 3, 100, 32

    rand_data = torch.rand((batch, length), dtype=torch.float64, device=device)[:, :, None]
    X_non_cont = rand_data.expand(-1, -1, dim)
    X = X_non_cont.clone()

    res1 = pysiglib.sig(X, degree)
    res2 = pysiglib.sig(X_non_cont, degree)
    assert_device(res1, device)
    assert_device(res2, device)
    check_close(res1, res2)

    rand_data = np.random.normal(size=(batch, length))[:, :, None]
    X_non_cont = np.broadcast_to(rand_data, (batch, length, dim))
    X = np.array(X_non_cont)

    res1 = pysiglib.sig(X, degree)
    res2 = pysiglib.sig(X_non_cont, degree)
    check_close(res1, res2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_time_aug(device, deg):
    X = FIXTURES["path"][0]
    expected = FIXTURES[f"sig_ta__d{deg}"][0]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.sig(X_dev, deg, time_aug=True)
    assert_device(sig, device)
    check_close(expected, sig)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_signature_lead_lag(device, deg):
    X = FIXTURES["path_dim2"][0]
    expected = FIXTURES[f"sig_ll__d{deg}"][0]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.sig(X_dev, deg, lead_lag=True)
    assert_device(sig, device)
    check_close(expected, sig)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_signature_time_aug_lead_lag(device, deg, dtype):
    X = FIXTURES["path_dim2"][0].astype(dtype)
    expected = FIXTURES[f"sig_ta_ll__d{deg}"][0]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.sig(X_dev, deg, lead_lag=True, time_aug=True)
    assert_device(sig, device)
    check_close(expected, sig)
