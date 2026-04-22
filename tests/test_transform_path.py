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

np.random.seed(42)
torch.manual_seed(42)
from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device
check_close = partial(_check_close, atol=1e-5)

def time_aug(X, end_time = 1., is_batch = False):
    length = X.shape[1] if is_batch else X.shape[0]
    batch_size = X.shape[0] if is_batch else None
    t = np.linspace(0, end_time, length)
    t = np.tile(t[np.newaxis, :, np.newaxis], (batch_size, 1, 1)) if is_batch else t[:, np.newaxis]
    return np.concatenate((X, t), axis = 2 if is_batch else 1)

def lead_lag(X, is_batch = False):
    if not is_batch:
        lag = []
        lead = []

        for val_lag, val_lead in zip(X[:-1], X[1:]):
            lag.append(val_lag)
            lead.append(val_lag)

            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(X[-1])
        lead.append(X[-1])

        return np.c_[lag, lead]
    else:
        return np.concatenate([lead_lag(X_)[np.newaxis, :, :] for X_ in X], axis = 0)

@pytest.mark.parametrize("device", DEVICES)
def test_transform_path_time_aug(device):
    X = np.random.uniform(size=(100, 5))
    X1 = time_aug(X)
    X_dev = torch.tensor(X, device=device)
    X2 = pysiglib.transform_path(X_dev, time_aug = True)
    assert_device(X2, device)
    check_close(X1, X2)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_transform_path_time_aug(device):
    X = np.random.uniform(size=(10, 100, 5))
    X1 = time_aug(X, is_batch = True)
    X_dev = torch.tensor(X, device=device)
    X2 = pysiglib.transform_path(X_dev, time_aug = True)
    assert_device(X2, device)
    check_close(X1, X2)

@pytest.mark.parametrize("device", DEVICES)
def test_transform_path_lead_lag(device):
    X = np.random.uniform(size=(100, 5))
    X1 = lead_lag(X)
    X_dev = torch.tensor(X, device=device)
    X2 = pysiglib.transform_path(X_dev, lead_lag = True)
    assert_device(X2, device)
    check_close(X1, X2)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_transform_path_lead_lag(device):
    X = np.random.uniform(size=(10, 100, 5))
    X1 = lead_lag(X, True)
    X_dev = torch.tensor(X, device=device)
    X2 = pysiglib.transform_path(X_dev, lead_lag = True)
    assert_device(X2, device)
    check_close(X1, X2)

@pytest.mark.parametrize("device", DEVICES)
def test_transform_path_backprop_lead_lag(device):
    X = torch.rand(size=(100, 5), dtype = torch.double, device=device)

    X_ll = pysiglib.transform_path(X, lead_lag = True)
    deriv = torch.ones(X_ll.shape, dtype = torch.double, device=device)
    X1 = pysiglib.transform_path_backprop(deriv, lead_lag = True)
    assert_device(X1, device)

    X2 = torch.ones((100,5), dtype = torch.double, device=device) * 4.
    X2[0, :] = 3.
    X2[-1, :] = 3.

    check_close(X1, X2)

@pytest.mark.parametrize("device", DEVICES)
def test_transform_path_time_aug_lead_lag(device):
    X = np.random.uniform(size=(100, 5))
    X1 = time_aug(lead_lag(X))
    X_dev = torch.tensor(X, device=device)
    X2 = pysiglib.transform_path(X_dev, time_aug=True, lead_lag=True)
    assert_device(X2, device)
    check_close(X1, X2)

@pytest.mark.parametrize("device", DEVICES)
def test_transform_path_custom_end_time(device):
    X = np.random.uniform(size=(100, 5))
    X1 = time_aug(X, end_time=2.0)
    X_dev = torch.tensor(X, device=device)
    X2 = pysiglib.transform_path(X_dev, time_aug=True, end_time=2.0)
    assert_device(X2, device)
    check_close(X1, X2)

@pytest.mark.parametrize("device", DEVICES)
def test_transform_path_backprop_time_aug(device):
    X = torch.rand(size=(100, 5), dtype=torch.double, device=device)
    X_ta = pysiglib.transform_path(X, time_aug=True)
    deriv = torch.ones(X_ta.shape, dtype=torch.double, device=device)
    grad = pysiglib.transform_path_backprop(deriv, time_aug=True)
    assert_device(grad, device)
    expected = torch.ones((100, 5), dtype=torch.double, device=device)
    check_close(grad, expected)

@pytest.mark.parametrize("device", DEVICES)
def test_batch_transform_path_backprop_lead_lag(device):
    X = torch.rand(size=(10, 100, 5), dtype=torch.double, device=device)
    X_ll = pysiglib.transform_path(X, lead_lag=True)
    deriv = torch.ones(X_ll.shape, dtype=torch.double, device=device)
    grad = pysiglib.transform_path_backprop(deriv, lead_lag=True)
    assert_device(grad, device)
    expected = torch.ones((10, 100, 5), dtype=torch.double, device=device) * 4.
    expected[:, 0, :] = 3.
    expected[:, -1, :] = 3.
    check_close(grad, expected)

@pytest.mark.parametrize("device", DEVICES)
def test_transform_path_backprop_custom_end_time(device):
    X = torch.rand(size=(100, 5), dtype=torch.double, device=device)
    X_ta = pysiglib.transform_path(X, time_aug=True, end_time=2.0)
    deriv = torch.ones(X_ta.shape, dtype=torch.double, device=device)
    grad = pysiglib.transform_path_backprop(deriv, time_aug=True, end_time=2.0)
    assert_device(grad, device)
    expected = torch.ones((100, 5), dtype=torch.double, device=device)
    check_close(grad, expected)
