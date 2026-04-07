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
check_close = partial(_check_close, single_atol=1e-3, double_atol=1e-5)

FIXTURES = load_fixtures("reference_data.npz")


@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_trivial(device):
    X = torch.tensor([[0.]], device=device)
    k = pysiglib.sig_kernel(X, X, 0)
    assert_device(k, device)
    check_close(torch.tensor([1.]), k)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_numpy(device):
    x = torch.tensor([[0., 1.], [3., 2.]], device=device)
    k = pysiglib.sig_kernel(x, x, 0)
    assert_device(k, device)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_kernel_dtypes(device, dtype):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=dtype)
    Y = torch.tensor(FIXTURES["kern_Y"], device=device, dtype=dtype)
    expected = FIXTURES["kernel_linear__do0"]

    kernel2 = pysiglib.sig_kernel(X, Y, 0)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_random(device, dyadic_order):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["kern_Y"], device=device, dtype=torch.double)
    expected = FIXTURES[f"kernel_linear__do{dyadic_order}"]

    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_scaled_linear(device, dyadic_order):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["kern_Y"], device=device, dtype=torch.double)
    expected = FIXTURES[f"kernel_scaled_linear__do{dyadic_order}"]

    static_kernel = pysiglib.ScaledLinearKernel(0.5)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=static_kernel)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_rbf(device, dyadic_order):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["kern_Y"], device=device, dtype=torch.double)
    expected = FIXTURES[f"kernel_rbf__do{dyadic_order}"]

    static_kernel = pysiglib.RBFKernel(0.5)
    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, static_kernel=static_kernel)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(20, 5), (5, 20)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_random_non_square(device, len1, len2, dyadic_order):
    if len1 == 20:
        X = torch.tensor(FIXTURES["kern_X2"], device=device, dtype=torch.double)
        Y = torch.tensor(FIXTURES["kern_Y2"], device=device, dtype=torch.double)
        expected = FIXTURES[f"kernel_nonsq_long_short__do{dyadic_order}"]
    else:
        X = torch.tensor(FIXTURES["kern_Y2"], device=device, dtype=torch.double)
        Y = torch.tensor(FIXTURES["kern_X2"], device=device, dtype=torch.double)
        expected = FIXTURES[f"kernel_nonsq_short_long__do{dyadic_order}"]

    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", [(1,0), (2,0), (2,1)])
def test_sig_kernel_different_dyadics(device, dyadic_order):
    batch, len1, len2, dim = 32, 10, 100, 5
    X = torch.rand(size=(batch, len1, dim), device=device, dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device=device, dtype = torch.double)

    kernel1 = pysiglib.sig_kernel(X, Y, dyadic_order)
    assert_device(kernel1, device)
    kernel2 = pysiglib.sig_kernel(Y, X, dyadic_order[::-1])

    check_close(kernel1, kernel2)


@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_non_contiguous(device):
    # Make sure sig_kernel works with any form of array
    dim, length, batch = 10, 100, 32

    rand_data = torch.rand(size=(batch, length), dtype=torch.float64, device=device)[:, :, None]
    X_non_cont = rand_data.expand(-1, -1, dim)
    X = X_non_cont.clone()

    res1 = pysiglib.sig_kernel(X, X, 0)
    res2 = pysiglib.sig_kernel(X_non_cont, X_non_cont, 0)
    assert_device(res1, device)
    assert_device(res2, device)
    check_close(res1, res2)

    rand_data = (np.random.normal(size=(batch, length)) / 100)[:, :, None]
    X_non_cont = np.broadcast_to(rand_data, (batch, length, dim))
    X = np.array(X_non_cont)

    res1 = pysiglib.sig_kernel(X, X, 0)
    res2 = pysiglib.sig_kernel(X_non_cont, X_non_cont, 0)
    check_close(res1, res2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_kernel_lead_lag(device, dyadic_order):
    X = torch.tensor(FIXTURES["kern_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["kern_Y"], device=device, dtype=torch.double)
    expected = FIXTURES[f"kernel_lead_lag__do{dyadic_order}"]

    kernel2 = pysiglib.sig_kernel(X, Y, dyadic_order, lead_lag=True)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(10, 10), (10, 5), (5, 10)])
def test_sig_kernel_full_grid(device, len1, len2):
    X = torch.tensor(FIXTURES[f"kernel_grid__{len1}x{len2}__X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES[f"kernel_grid__{len1}x{len2}__Y"], device=device, dtype=torch.double)
    expected = FIXTURES[f"kernel_grid__{len1}x{len2}__expected"]

    kernel2 = pysiglib.sig_kernel(X, Y, 0, return_grid=True)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_full_grid_time_aug(device):
    X = torch.tensor(FIXTURES["kernel_grid_ta__X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["kernel_grid_ta__Y"], device=device, dtype=torch.double)
    expected = FIXTURES["kernel_grid_ta__expected"]

    kernel2 = pysiglib.sig_kernel(X, Y, 0, time_aug=True, return_grid=True)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_full_grid_lead_lag(device):
    X = torch.tensor(FIXTURES["kernel_grid_ll__X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["kernel_grid_ll__Y"], device=device, dtype=torch.double)
    expected = FIXTURES["kernel_grid_ll__expected"]

    kernel2 = pysiglib.sig_kernel(X, Y, 0, lead_lag=True, return_grid=True)
    assert_device(kernel2, device)

    check_close(expected, kernel2)

@pytest.mark.parametrize("device", DEVICES)
def test_sig_kernel_full_grid_time_aug_lead_lag(device):
    X = torch.tensor(FIXTURES["kernel_grid_ta_ll__X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["kernel_grid_ta_ll__Y"], device=device, dtype=torch.double)
    expected = FIXTURES["kernel_grid_ta_ll__expected"]

    kernel2 = pysiglib.sig_kernel(X, Y, 0, lead_lag=True, time_aug=True, return_grid=True)
    assert_device(kernel2, device)

    check_close(expected, kernel2)
