# Copyright 2026 Daniil Shmelev
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
check_close = partial(_check_close, single_atol=1e-3, double_atol=1e-4)


def _finite_diff_check(ls1, ls2, d_out, dimension, degree, eps=1e-6, n_jobs=1, device="cpu"):
    """Verify backprop via directional finite differences.

    For a random direction v, checks:
        d_ls1 @ v  \approx  d_out @ (F(ls1+eps*v, ls2) - F(ls1-eps*v, ls2)) / (2*eps)
    and similarly for d_ls2.
    """
    d_ls1, d_ls2 = pysiglib.log_sig_combine_backprop(
        d_out, ls1, ls2, dimension, degree, n_jobs=n_jobs
    )
    assert_device(d_ls1, device)
    assert_device(d_ls2, device)

    is_batch = ls1.dim() == 2

    # --- check d_ls1 ---
    v1 = torch.randn_like(ls1)
    fwd_plus = pysiglib.log_sig_combine(ls1 + eps * v1, ls2, dimension, degree, n_jobs=n_jobs)
    fwd_minus = pysiglib.log_sig_combine(ls1 - eps * v1, ls2, dimension, degree, n_jobs=n_jobs)
    if is_batch:
        numerical1 = torch.sum(d_out * (fwd_plus - fwd_minus) / (2 * eps), dim=-1)
        analytic1 = torch.sum(d_ls1 * v1, dim=-1)
    else:
        numerical1 = torch.dot(d_out, (fwd_plus - fwd_minus) / (2 * eps))
        analytic1 = torch.dot(d_ls1, v1)
    check_close(numerical1, analytic1)

    # --- check d_ls2 ---
    v2 = torch.randn_like(ls2)
    fwd_plus = pysiglib.log_sig_combine(ls1, ls2 + eps * v2, dimension, degree, n_jobs=n_jobs)
    fwd_minus = pysiglib.log_sig_combine(ls1, ls2 - eps * v2, dimension, degree, n_jobs=n_jobs)
    if is_batch:
        numerical2 = torch.sum(d_out * (fwd_plus - fwd_minus) / (2 * eps), dim=-1)
        analytic2 = torch.sum(d_ls2 * v2, dim=-1)
    else:
        numerical2 = torch.dot(d_out, (fwd_plus - fwd_minus) / (2 * eps))
        analytic2 = torch.dot(d_ls2, v2)
    check_close(numerical2, analytic2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_backprop_single(device, deg):
    dim = 5
    ls_len = pysiglib.log_sig_length(dim, deg)

    ls1 = torch.randn(ls_len, dtype=torch.float64, device=device)
    ls2 = torch.randn(ls_len, dtype=torch.float64, device=device)
    d_out = torch.randn(ls_len, dtype=torch.float64, device=device)

    _finite_diff_check(ls1, ls2, d_out, dim, deg, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_backprop_batch(device, deg):
    dim = 5
    batch = 8
    ls_len = pysiglib.log_sig_length(dim, deg)

    ls1 = torch.randn(batch, ls_len, dtype=torch.float64, device=device)
    ls2 = torch.randn(batch, ls_len, dtype=torch.float64, device=device)
    d_out = torch.randn(batch, ls_len, dtype=torch.float64, device=device)

    _finite_diff_check(ls1, ls2, d_out, dim, deg, device=device)


@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_backprop_multithreaded(deg):
    dim = 5
    batch = 8
    ls_len = pysiglib.log_sig_length(dim, deg)

    ls1 = torch.randn(batch, ls_len, dtype=torch.float64)
    ls2 = torch.randn(batch, ls_len, dtype=torch.float64)
    d_out = torch.randn(batch, ls_len, dtype=torch.float64)

    d_ls1_j1, d_ls2_j1 = pysiglib.log_sig_combine_backprop(d_out, ls1, ls2, dim, deg, n_jobs=1)
    d_ls1_jn, d_ls2_jn = pysiglib.log_sig_combine_backprop(d_out, ls1, ls2, dim, deg, n_jobs=-1)
    check_close(d_ls1_j1, d_ls1_jn)
    check_close(d_ls2_j1, d_ls2_jn)

    _finite_diff_check(ls1, ls2, d_out, dim, deg, n_jobs=-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_backprop_zero_deriv(device, deg):
    """Backward with zero upstream gradient should give zero derivatives."""
    dim = 5
    ls_len = pysiglib.log_sig_length(dim, deg)

    ls1 = torch.randn(ls_len, dtype=torch.float64, device=device)
    ls2 = torch.randn(ls_len, dtype=torch.float64, device=device)
    d_out = torch.zeros(ls_len, dtype=torch.float64, device=device)

    d_ls1, d_ls2 = pysiglib.log_sig_combine_backprop(d_out, ls1, ls2, dim, deg)
    assert_device(d_ls1, device)
    assert_device(d_ls2, device)
    check_close(d_ls1, torch.zeros_like(d_ls1))
    check_close(d_ls2, torch.zeros_like(d_ls2))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_backprop_numpy(device, deg):
    """Test that numpy arrays work as inputs."""
    if device != "cpu":
        pytest.skip("numpy only on cpu")
    dim = 5
    batch = 4
    ls_len = pysiglib.log_sig_length(dim, deg)

    ls1 = np.random.randn(batch, ls_len)
    ls2 = np.random.randn(batch, ls_len)
    d_out = np.random.randn(batch, ls_len)

    d_ls1, d_ls2 = pysiglib.log_sig_combine_backprop(d_out, ls1, ls2, dim, deg)
    assert isinstance(d_ls1, np.ndarray)
    assert isinstance(d_ls2, np.ndarray)
    assert d_ls1.shape == ls1.shape
    assert d_ls2.shape == ls2.shape
