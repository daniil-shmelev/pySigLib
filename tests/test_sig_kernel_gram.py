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
import torch

import pysiglib

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device, load_fixtures
check_close = partial(_check_close, single_atol=1e-3, double_atol=1e-5)

FIXTURES = load_fixtures("reference_data.npz")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_kernel_gram_dtypes(device, dtype):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=dtype)
    Y = torch.tensor(FIXTURES["gram_Y"], device=device, dtype=dtype)
    expected = FIXTURES["kernel_gram__do0"]

    kernel2 = pysiglib.sig_kernel_gram(X, Y, 0)
    assert_device(kernel2, device)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, 0, max_batch=2)

    check_close(expected, kernel2)
    check_close(expected, kernel3)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_gram_random(device, dyadic_order):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["gram_Y"], device=device, dtype=torch.double)
    expected = FIXTURES[f"kernel_gram__do{dyadic_order}"]

    kernel2 = pysiglib.sig_kernel_gram(X, Y, dyadic_order)
    assert_device(kernel2, device)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, max_batch=2)

    check_close(expected, kernel2)
    check_close(expected, kernel3)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_gram_lead_lag(device, dyadic_order):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["gram_Y"], device=device, dtype=torch.double)
    expected = FIXTURES[f"kernel_gram_ll__do{dyadic_order}"]

    kernel2 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, lead_lag=True)
    assert_device(kernel2, device)
    kernel3 = pysiglib.sig_kernel_gram(X, Y, dyadic_order, lead_lag=True, max_batch=2)

    check_close(expected, kernel2)
    check_close(expected, kernel3)
