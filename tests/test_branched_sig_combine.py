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
from conftest import check_close, skip_no_cuda


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_concatenation(d, N):
    """Chen's identity: bsig(X*Y) == branched_sig_combine(bsig(X), bsig(Y))."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(99)
    L1, L2 = 5, 6
    path1 = np.cumsum(np.random.randn(L1, d) * 0.1, axis=0)
    path2_increments = np.random.randn(L2 - 1, d) * 0.1
    path2 = np.vstack([path1[-1:], path1[-1] + np.cumsum(path2_increments, axis=0)])

    bsig1 = pysiglib.branched_sig(path1, N)
    bsig2 = pysiglib.branched_sig(path2, N)
    combined = pysiglib.branched_sig_combine(bsig1, bsig2, d, N)

    full_path = np.vstack([path1, path2[1:]])
    bsig_full = pysiglib.branched_sig(full_path, N)

    np.testing.assert_allclose(combined, bsig_full, atol=1e-10)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_lead_lag_concatenation(d, N):
    """Chen's identity should hold with lead_lag enabled."""
    aug_dim = 2 * d
    pysiglib.prepare_branched_sig(aug_dim, N)
    np.random.seed(204)
    L1, L2 = 5, 6
    path1 = np.cumsum(np.random.randn(L1, d) * 0.1, axis=0)
    path2_inc = np.random.randn(L2 - 1, d) * 0.1
    path2 = np.vstack([path1[-1:], path1[-1] + np.cumsum(path2_inc, axis=0)])

    bsig1 = pysiglib.branched_sig(path1, N, lead_lag=True)
    bsig2 = pysiglib.branched_sig(path2, N, lead_lag=True)
    combined = pysiglib.branched_sig_combine(bsig1, bsig2, aug_dim, N)

    full_path = np.vstack([path1, path2[1:]])
    bsig_full = pysiglib.branched_sig(full_path, N, lead_lag=True)

    np.testing.assert_allclose(combined, bsig_full, atol=1e-10)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_combine_cuda_matches_cpu(d, N):
    """CUDA combine matches CPU."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(601)
    bsig_len = pysiglib.branched_sig_length(d, N)
    bsig1 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)
    bsig2 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)

    cpu = pysiglib.branched_sig_combine(bsig1, bsig2, d, N)
    cuda = pysiglib.branched_sig_combine(bsig1.cuda(), bsig2.cuda(), d, N)
    check_close(cpu, cuda, double_atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_concatenation(d, N):
    """Chen's identity: bsig(X*Y) == branched_sig_combine(bsig(X), bsig(Y))."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1002)
    L1, L2 = 5, 6
    path1 = np.cumsum(np.random.randn(L1, d) * 0.1, axis=0)
    path2_increments = np.random.randn(L2 - 1, d) * 0.1
    path2 = np.vstack([path1[-1:], path1[-1] + np.cumsum(path2_increments, axis=0)])

    bsig1 = pysiglib.branched_sig(path1, N, planar=True)
    bsig2 = pysiglib.branched_sig(path2, N, planar=True)
    combined = pysiglib.branched_sig_combine(bsig1, bsig2, d, N, planar=True)

    full_path = np.vstack([path1, path2[1:]])
    bsig_full = pysiglib.branched_sig(full_path, N, planar=True)

    np.testing.assert_allclose(combined, bsig_full, atol=1e-10)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_lead_lag_concatenation(d, N):
    """Chen's identity should hold with lead_lag enabled for planar sigs."""
    aug_dim = 2 * d
    pysiglib.prepare_branched_sig(aug_dim, N, planar=True)
    np.random.seed(1104)
    L1, L2 = 5, 6
    path1 = np.cumsum(np.random.randn(L1, d) * 0.1, axis=0)
    path2_inc = np.random.randn(L2 - 1, d) * 0.1
    path2 = np.vstack([path1[-1:], path1[-1] + np.cumsum(path2_inc, axis=0)])

    bsig1 = pysiglib.branched_sig(path1, N, lead_lag=True, planar=True)
    bsig2 = pysiglib.branched_sig(path2, N, lead_lag=True, planar=True)
    combined = pysiglib.branched_sig_combine(bsig1, bsig2, aug_dim, N, planar=True)

    full_path = np.vstack([path1, path2[1:]])
    bsig_full = pysiglib.branched_sig(full_path, N, lead_lag=True, planar=True)

    np.testing.assert_allclose(combined, bsig_full, atol=1e-10)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_combine_cuda_matches_cpu(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1501)
    bsig_len = pysiglib.branched_sig_length(d, N, planar=True)
    bsig1 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)
    bsig2 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)

    cpu = pysiglib.branched_sig_combine(bsig1, bsig2, d, N, planar=True)
    cuda = pysiglib.branched_sig_combine(bsig1.cuda(), bsig2.cuda(), d, N, planar=True)
    check_close(cpu, cuda, double_atol=1e-12)
