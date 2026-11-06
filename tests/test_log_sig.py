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

import pysiglib.torch_api as pysiglib
from conftest import DEVICES, check_close, assert_device, load_fixtures

FIXTURES = load_fixtures("reference_data.npz")
DIM = 3


def test_prepare_memory():
    X = np.random.uniform(size=(100, 5))
    pysiglib.clear_cache(use_disk=True)
    pysiglib.prepare_log_sig(5, 2, 1)

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=2)

    pysiglib.clear_cache()

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=1)

    pysiglib.prepare_log_sig(5, 2, 2)
    pysiglib.log_sig(X, 2, method=1)
    pysiglib.clear_cache()


@pytest.mark.parametrize("device", DEVICES)
def test_bch_requires_preparation(device):
    X = torch.zeros((3, 2), dtype=torch.float64, device=device)
    pysiglib.clear_cache()

    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.log_sig(X, 3, method=3)

    pysiglib.prepare_log_sig(2, 3, 2, device=device)
    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.log_sig(X, 3, method=3)

    pysiglib.prepare_log_sig(2, 3, 3, device=device)
    pysiglib.log_sig(X, 3, method=3)
    pysiglib.clear_cache()

    with pytest.raises(Exception, match="Could not find prepared cache"):
        pysiglib.log_sig(X, 3, method=3)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_log_sig_cache_is_per_device():
    pysiglib.clear_cache()
    with torch.cuda.device(0):
        pysiglib.prepare_log_sig(2, 3, 3, device="cuda")

    with torch.cuda.device(1):
        X = torch.zeros((3, 2), dtype=torch.float64, device="cuda:1")
        with pytest.raises(Exception, match="Could not find prepared cache"):
            pysiglib.log_sig(X, 3, method=3)
        pysiglib.prepare_log_sig(2, 3, 3, device="cuda")
        pysiglib.log_sig(X, 3, method=3)


def test_prepare_disk():
    X = np.random.uniform(size=(100, 5))
    pysiglib.clear_cache(use_disk=True)
    pysiglib.prepare_log_sig(5, 2, 1, use_disk=True)
    pysiglib.clear_cache(use_disk=False)

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=2)

    pysiglib.clear_cache(use_disk=True)

    with pytest.raises(Exception):
        pysiglib.log_sig(X, 2, method=1)

    pysiglib.prepare_log_sig(5, 2, 2, use_disk=True)
    pysiglib.clear_cache(use_disk=False)
    pysiglib.log_sig(X, 2, method=1)
    pysiglib.clear_cache(use_disk=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_log_signature_expanded_random(device, deg, dtype):
    X = FIXTURES["path"][0].astype(dtype)
    expected = FIXTURES[f"logsig_exp__d{deg}"][0]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.log_sig(X_dev, deg, method=0)
    assert_device(sig, device)
    check_close(expected, sig)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batch_log_signature_expanded_random(device, deg, dtype):
    X = FIXTURES["path"].astype(dtype)
    expected = FIXTURES[f"logsig_exp__d{deg}"]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.log_sig(X_dev, deg, method=0)
    assert_device(sig, device)
    check_close(expected, sig)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batch_log_signature_expanded_time_aug_random(device, deg, dtype):
    X = FIXTURES["path"].astype(dtype)
    expected = FIXTURES[f"logsig_exp_ta__d{deg}"]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.log_sig(X_dev, deg, time_aug=True, method=0)
    assert_device(sig, device)
    check_close(expected, sig)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batch_log_signature_expanded_lead_lag_random(device, deg, dtype):
    X = FIXTURES["path"][:2].astype(dtype)
    expected = FIXTURES[f"logsig_exp_ll__d{deg}"]
    X_dev = torch.tensor(X, device=device)
    sig = pysiglib.log_sig(X_dev, deg, lead_lag=True, method=0)
    assert_device(sig, device)
    check_close(expected, sig)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_log_signature_lyndon_words_random(device, deg, dtype):
    key = f"logsig_words__d{deg}"
    if key not in FIXTURES:
        pytest.skip("signatory fixture not available")
    X = FIXTURES["path"][:1].astype(dtype)
    expected = FIXTURES[key][:1]
    X_dev = torch.tensor(X, device=device)
    pysiglib.prepare_log_sig(DIM, deg, 1)
    sig = pysiglib.log_sig(X_dev[0], deg, method=1)
    assert_device(sig, device)
    check_close(expected, sig)
    pysiglib.clear_cache()

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batch_log_signature_lyndon_words_random(device, deg, dtype):
    key = f"logsig_words__d{deg}"
    if key not in FIXTURES:
        pytest.skip("signatory fixture not available")
    X = FIXTURES["path"].astype(dtype)
    expected = FIXTURES[key]
    X_dev = torch.tensor(X, device=device)
    pysiglib.prepare_log_sig(DIM, deg, 1)
    sig = pysiglib.log_sig(X_dev, deg, method=1)
    assert_device(sig, device)
    check_close(expected, sig)
    pysiglib.clear_cache()

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("method", [2, 3])
def test_log_signature_lyndon_basis_random(device, deg, dtype, method):
    X = FIXTURES["path"][0].astype(dtype)
    expected = FIXTURES[f"logsig_basis__d{deg}"][0]
    X_dev = torch.tensor(X, device=device)
    pysiglib.prepare_log_sig(DIM, deg, method)
    sig = pysiglib.log_sig(X_dev, deg, method=method)
    assert_device(sig, device)
    check_close(expected, sig)
    pysiglib.clear_cache()

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("method", [2, 3])
def test_batch_log_signature_lyndon_basis_random(device, deg, dtype, method):
    X = FIXTURES["path"].astype(dtype)
    expected = FIXTURES[f"logsig_basis__d{deg}"]
    X_dev = torch.tensor(X, device=device)
    pysiglib.prepare_log_sig(DIM, deg, method)
    sig = pysiglib.log_sig(X_dev, deg, method=method)
    assert_device(sig, device)
    check_close(expected, sig)
    pysiglib.clear_cache()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("dimension", "degree"), [(2, 3), (2, 4), (2, 5), (2, 8), (3, 8)])
def test_method3_pruned_plan_forward_and_backward(device, dimension, degree):
    path = torch.randn(6, dimension, dtype=torch.float64, device=device, requires_grad=True)
    pysiglib.prepare_log_sig(dimension, degree, 2, device=device)
    pysiglib.prepare_log_sig(dimension, degree, 3, device=device)
    expected = pysiglib.log_sig(path, degree, method=2)
    actual = pysiglib.log_sig(path, degree, method=3)
    check_close(expected, actual)
    d_out = torch.randn_like(actual)
    expected_grad, = torch.autograd.grad(expected, path, d_out, retain_graph=True)
    actual_grad, = torch.autograd.grad(actual, path, d_out)
    check_close(expected_grad, actual_grad)


@pytest.mark.parametrize("device", DEVICES)
def test_log_signature_method0_correction_matches_sig_to_log_sig(device):
    X = torch.tensor([[0.0], [3.0]], dtype=torch.float64, device=device)
    correction = torch.tensor([[2.0]], dtype=torch.float64, device=device)

    sig = pysiglib.sig(X, 4, scalar_term=True, correction=correction)
    expected = pysiglib.sig_to_log_sig(sig, 1, 4, method=0)
    actual = pysiglib.log_sig(X, 4, method=0, scalar_term=True, correction=correction)

    assert_device(actual, device)
    check_close(expected, actual)


def test_log_signature_method3_rejects_correction():
    X = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    correction = torch.tensor([[1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="method=3"):
        pysiglib.log_sig(X, 2, method=3, correction=correction)


@pytest.mark.parametrize("device", DEVICES)
def test_log_signature_method3_empty_correction_match_default(device):
    X = torch.tensor([[0.0, 0.1], [0.5, 0.4], [1.0, -0.2]], dtype=torch.float64, device=device)
    correction = torch.empty((0,), dtype=torch.float64, device=device)

    pysiglib.prepare_log_sig(2, 3, 3)

    expected = pysiglib.log_sig(X, 3, method=3)
    actual = pysiglib.log_sig(X, 3, method=3, correction=correction)

    assert_device(actual, device)
    check_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("sig_method,split", [("ps", None), ("ps", 1), ("ps", 3), ("auto", None)])
@pytest.mark.parametrize("time_aug,lead_lag,scalar,corrected", [
    (False, False, False, False), (True, False, True, True),
    (False, True, True, False), (True, True, False, False),
])
def test_log_signature_ps_forward_and_backward(device, dtype, method, sig_method, split,
                                               time_aug, lead_lag, scalar, corrected):
    import pysiglib as base_api

    path = torch.tensor(np.random.default_rng(24).normal(scale=0.05, size=(2, 1, 12, 2)),
                        device=device, dtype=dtype)[..., ::2, :].detach().requires_grad_()
    correction = None
    if corrected:
        correction = torch.tensor([0.0, 0.001, -0.001, 0.0], device=device, dtype=dtype)
    options = dict(method=method, time_aug=time_aug, lead_lag=lead_lag,
                   scalar_term=scalar, correction=correction, n_jobs=2)
    pysiglib.prepare_log_sig(2, 3, method, time_aug=time_aug, lead_lag=lead_lag, device=device)
    expected = pysiglib.log_sig(path, 3, **options)
    actual = pysiglib.log_sig(path, 3, sig_method=sig_method, ps_split=split, **options)
    seeds = torch.linspace(-0.5, 0.5, actual.numel(), device=device, dtype=dtype).reshape(actual.shape)
    expected_grad, = torch.autograd.grad(expected, path, seeds)
    actual_grad, = torch.autograd.grad(actual, path, seeds)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)
    if device == "cpu":
        options["correction"] = correction.detach().numpy() if corrected else None
        result = base_api.log_sig(path.detach().numpy(), 3, sig_method=sig_method, ps_split=split, **options)
        np.testing.assert_allclose(result, expected.detach().numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("sig_method,split", [("ps", None), ("auto", None), (None, 2)])
def test_log_signature_method3_rejects_signature_options(sig_method, split):
    import pysiglib as base_api

    path = torch.zeros((4, 2), dtype=torch.float64)
    for api in (base_api, pysiglib):
        with pytest.raises(ValueError, match="require log_sig method"):
            api.log_sig(path, 3, method=3, sig_method=sig_method, ps_split=split)
