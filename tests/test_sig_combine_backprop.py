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

import native_api as pysiglib

from functools import partial
from conftest import DEVICES, check_close as _check_close, assert_device, load_fixtures
check_close = partial(_check_close, single_atol=1e-3, double_atol=1e-5)

FIXTURES = load_fixtures("reference_data.npz")
DIM = 3

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_sig_combine_backprop_random(device, deg):
    sig1 = torch.tensor(FIXTURES[f"sig_comb_bp_sig1__d{deg}"][0], device=device, dtype=torch.float64)
    sig2 = torch.tensor(FIXTURES[f"sig_comb_bp_sig2__d{deg}"][0], device=device, dtype=torch.float64)
    derivs = torch.tensor(FIXTURES[f"sig_comb_bp_derivs__d{deg}"][0], device=device, dtype=torch.float64)
    expected_d1 = FIXTURES[f"sig_comb_bp_d1__d{deg}"][0]
    expected_d2 = FIXTURES[f"sig_comb_bp_d2__d{deg}"][0]

    sig1_deriv, sig2_deriv = pysiglib.sig_combine_backprop(derivs, sig1, sig2, DIM, deg)
    assert_device(sig1_deriv, device)
    assert_device(sig2_deriv, device)
    check_close(sig1_deriv[1:], expected_d1)
    check_close(sig2_deriv[1:], expected_d2)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_batch_sig_backprop_random(device, deg):
    sig1 = torch.tensor(FIXTURES[f"sig_comb_bp_sig1__d{deg}"], device=device, dtype=torch.float64)
    sig2 = torch.tensor(FIXTURES[f"sig_comb_bp_sig2__d{deg}"], device=device, dtype=torch.float64)
    derivs = torch.tensor(FIXTURES[f"sig_comb_bp_derivs__d{deg}"], device=device, dtype=torch.float64)
    expected_d1 = FIXTURES[f"sig_comb_bp_d1__d{deg}"]
    expected_d2 = FIXTURES[f"sig_comb_bp_d2__d{deg}"]

    sig1_deriv, sig2_deriv = pysiglib.sig_combine_backprop(derivs, sig1, sig2, DIM, deg, n_jobs=1)
    assert_device(sig1_deriv, device)
    assert_device(sig2_deriv, device)
    check_close(sig1_deriv[:, 1:], expected_d1)
    check_close(sig2_deriv[:, 1:], expected_d2)

    sig1_deriv, sig2_deriv = pysiglib.sig_combine_backprop(derivs, sig1, sig2, DIM, deg, n_jobs=-1)
    assert_device(sig1_deriv, device)
    assert_device(sig2_deriv, device)
    check_close(sig1_deriv[:, 1:], expected_d1)
    check_close(sig2_deriv[:, 1:], expected_d2)
