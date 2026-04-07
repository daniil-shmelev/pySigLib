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

import pysiglib.torch_api as pysiglib
from conftest import DEVICES, check_close as _check_close, assert_device, load_fixtures
from functools import partial

check_close = partial(_check_close, double_atol=1e-5)

FIXTURES = load_fixtures("reference_data.npz")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sig_to_log_sig_backprop_expanded_random(device, deg, dtype):
    key_in = f"sig_to_logsig_bp_input__d{deg}"
    key_exp = f"sig_to_logsig_bp_expected__d{deg}"
    if key_in not in FIXTURES:
        pytest.skip("sig_to_logsig fixture not available (signatory needed)")

    X = torch.tensor(FIXTURES[key_in], dtype=dtype, device=device, requires_grad=True)
    ls = pysiglib.sig_to_log_sig(X, 1, deg, method=0)
    assert_device(ls, device)
    ls.backward(torch.ones_like(ls))

    expected = FIXTURES[key_exp]
    # Fixture stores grad w.r.t. non-scalar part; pysiglib grad includes scalar term
    check_close(expected, X.grad[:, 1:])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_backprop_expanded_random(device, deg, dtype):
    X = torch.tensor(FIXTURES["path"][0], dtype=dtype, device=device, requires_grad=True)
    ls = pysiglib.log_sig(X, deg, method=0)
    assert_device(ls, device)

    derivs = torch.tensor(
        FIXTURES[f"logsig_bp_exp_derivs__d{deg}"][0], dtype=dtype, device=device
    )
    full_derivs = torch.zeros_like(ls)
    full_derivs[1:] = derivs
    ls.backward(full_derivs)

    expected = FIXTURES[f"logsig_bp_exp_expected__d{deg}"][0]
    check_close(expected, X.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_backprop_expanded_random(device, deg, dtype):
    X = torch.tensor(FIXTURES["path"], dtype=dtype, device=device, requires_grad=True)
    ls = pysiglib.log_sig(X, deg, method=0)
    assert_device(ls, device)

    derivs = torch.tensor(
        FIXTURES[f"logsig_bp_exp_derivs__d{deg}"], dtype=dtype, device=device
    )
    full_derivs = torch.zeros_like(ls)
    full_derivs[:, 1:] = derivs
    ls.backward(full_derivs)

    expected = FIXTURES[f"logsig_bp_exp_expected__d{deg}"]
    check_close(expected, X.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_backprop_expanded_time_aug_random(device, deg, dtype):
    X = torch.tensor(FIXTURES["path"], dtype=dtype, device=device, requires_grad=True)
    ls = pysiglib.log_sig(X, deg, time_aug=True, method=0)
    assert_device(ls, device)

    derivs = torch.tensor(
        FIXTURES[f"logsig_bp_exp_ta_derivs__d{deg}"], dtype=dtype, device=device
    )
    full_derivs = torch.zeros_like(ls)
    full_derivs[:, 1:] = derivs
    ls.backward(full_derivs)

    expected = FIXTURES[f"logsig_bp_exp_ta_expected__d{deg}"]
    check_close(expected, X.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_log_signature_lyndon_words_random(device, deg, dtype):
    key_derivs = f"logsig_bp_words_derivs__d{deg}"
    key_exp = f"logsig_bp_words_expected__d{deg}"
    if key_derivs not in FIXTURES:
        pytest.skip("Lyndon words backprop fixture not available (signatory needed)")

    X = torch.tensor(FIXTURES["path"][:1], dtype=dtype, device=device, requires_grad=True)
    pysiglib.prepare_log_sig(3, deg, 1)
    ls = pysiglib.log_sig(X, deg, method=1)
    assert_device(ls, device)

    derivs = torch.tensor(
        FIXTURES[key_derivs][:1], dtype=dtype, device=device
    )
    ls.backward(derivs)

    expected = FIXTURES[key_exp][:1]
    check_close(expected, X.grad)
    pysiglib.clear_cache()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batch_log_signature_lyndon_words_random(device, deg, dtype):
    key_derivs = f"logsig_bp_words_derivs__d{deg}"
    key_exp = f"logsig_bp_words_expected__d{deg}"
    if key_derivs not in FIXTURES:
        pytest.skip("Lyndon words backprop fixture not available (signatory needed)")

    X = torch.tensor(FIXTURES["path"], dtype=dtype, device=device, requires_grad=True)
    pysiglib.prepare_log_sig(3, deg, 1)
    ls = pysiglib.log_sig(X, deg, method=1)
    assert_device(ls, device)

    derivs = torch.tensor(
        FIXTURES[key_derivs], dtype=dtype, device=device
    )
    ls.backward(derivs)

    expected = FIXTURES[key_exp]
    check_close(expected, X.grad)
    pysiglib.clear_cache()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("method", [2, 3])
def test_log_signature_backprop_lyndon_basis_random(device, deg, dtype, method):
    X = torch.tensor(FIXTURES["path"][0], dtype=dtype, device=device, requires_grad=True)
    pysiglib.prepare_log_sig(3, deg, 2)
    ls = pysiglib.log_sig(X, deg, method=method)
    assert_device(ls, device)

    derivs = torch.tensor(
        FIXTURES[f"logsig_bp_basis_derivs__d{deg}"][0], dtype=dtype, device=device
    )
    ls.backward(derivs)

    expected = FIXTURES[f"logsig_bp_basis_expected__d{deg}"][0]
    check_close(expected, X.grad)
    pysiglib.clear_cache()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("method", [2, 3])
def test_batch_log_signature_backprop_lyndon_basis_random(device, deg, dtype, method):
    X = torch.tensor(FIXTURES["path"], dtype=dtype, device=device, requires_grad=True)
    pysiglib.prepare_log_sig(3, deg, 2)
    ls = pysiglib.log_sig(X, deg, method=method)
    assert_device(ls, device)

    derivs = torch.tensor(
        FIXTURES[f"logsig_bp_basis_derivs__d{deg}"], dtype=dtype, device=device
    )
    ls.backward(derivs)

    expected = FIXTURES[f"logsig_bp_basis_expected__d{deg}"]
    check_close(expected, X.grad, single_atol=1e-3)
    pysiglib.clear_cache()
