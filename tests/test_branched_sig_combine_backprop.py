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


@skip_no_cuda
def test_branched_sig_combine_torch_api(d=2, N=3):
    """torch_api branched_sig_combine backward works."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(503)
    bsig1 = torch.tensor(np.random.randn(pysiglib.branched_sig_length(d, N)),
                         dtype=torch.float64, requires_grad=True)
    bsig2 = torch.tensor(np.random.randn(pysiglib.branched_sig_length(d, N)),
                         dtype=torch.float64, requires_grad=True)

    combined = pysiglib.torch_api.branched_sig_combine(bsig1, bsig2, d, N)
    combined.sum().backward()

    assert bsig1.grad is not None
    assert bsig2.grad is not None
    assert bsig1.grad.shape == bsig1.shape
    assert bsig2.grad.shape == bsig2.shape


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_combine_torch_api(d, N):
    """torch_api planar branched_sig_combine backward matches manual backprop."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(5031)
    bsig_len = pysiglib.branched_sig_length(d, N, planar=True)
    bsig1 = torch.tensor(np.random.randn(bsig_len),
                         dtype=torch.float64, requires_grad=True)
    bsig2 = torch.tensor(np.random.randn(bsig_len),
                         dtype=torch.float64, requires_grad=True)

    combined = pysiglib.torch_api.branched_sig_combine(bsig1, bsig2, d, N, planar=True)
    combined.sum().backward()

    d1_manual, d2_manual = pysiglib.branched_sig_combine_backprop(
        torch.ones_like(combined), bsig1.detach(), bsig2.detach(), d, N, planar=True)

    check_close(bsig1.grad, d1_manual, double_atol=1e-12)
    check_close(bsig2.grad, d2_manual, double_atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_combine_backprop_finite_diff(d, N):
    """Combine backprop matches finite differences."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(600)
    bsig_len = pysiglib.branched_sig_length(d, N)
    bsig1 = np.random.randn(bsig_len)
    bsig2 = np.random.randn(bsig_len)
    derivs = np.random.randn(bsig_len)

    d1, d2 = pysiglib.branched_sig_combine_backprop(derivs, bsig1, bsig2, d, N)

    eps = 1e-8
    fd1 = np.zeros(bsig_len)
    for i in range(bsig_len):
        b1p = bsig1.copy()
        b1p[i] += eps
        cp = np.array(pysiglib.branched_sig_combine(b1p, bsig2, d, N))
        cm = np.array(pysiglib.branched_sig_combine(bsig1, bsig2, d, N))
        fd1[i] = np.dot(derivs, cp - cm) / eps
    np.testing.assert_allclose(d1, fd1, atol=1e-4)

    fd2 = np.zeros(bsig_len)
    for i in range(bsig_len):
        b2p = bsig2.copy()
        b2p[i] += eps
        cp = np.array(pysiglib.branched_sig_combine(bsig1, b2p, d, N))
        cm = np.array(pysiglib.branched_sig_combine(bsig1, bsig2, d, N))
        fd2[i] = np.dot(derivs, cp - cm) / eps
    np.testing.assert_allclose(d2, fd2, atol=1e-4)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_combine_backprop_cuda_matches_cpu(d, N):
    """CUDA combine backprop matches CPU."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(602)
    bsig_len = pysiglib.branched_sig_length(d, N)
    bsig1 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)
    bsig2 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)
    derivs = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)

    d1_cpu, d2_cpu = pysiglib.branched_sig_combine_backprop(derivs, bsig1, bsig2, d, N)
    d1_cuda, d2_cuda = pysiglib.branched_sig_combine_backprop(
        derivs.cuda(), bsig1.cuda(), bsig2.cuda(), d, N)
    check_close(d1_cpu, d1_cuda, double_atol=1e-10)
    check_close(d2_cpu, d2_cuda, double_atol=1e-10)


@skip_no_cuda
def test_branched_sig_combine_torch_api_cuda(d=2, N=3):
    """torch_api branched_sig_combine backward on CUDA."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(603)
    bsig_len = pysiglib.branched_sig_length(d, N)
    bsig1 = torch.tensor(np.random.randn(bsig_len),
                         dtype=torch.float64, device="cuda", requires_grad=True)
    bsig2 = torch.tensor(np.random.randn(bsig_len),
                         dtype=torch.float64, device="cuda", requires_grad=True)

    combined = pysiglib.torch_api.branched_sig_combine(bsig1, bsig2, d, N)
    combined.sum().backward()

    assert bsig1.grad is not None
    assert bsig2.grad is not None

    b1_cpu = bsig1.detach().cpu().requires_grad_(True)
    b2_cpu = bsig2.detach().cpu().requires_grad_(True)
    pysiglib.torch_api.branched_sig_combine(b1_cpu, b2_cpu, d, N).sum().backward()
    check_close(b1_cpu.grad, bsig1.grad, double_atol=1e-10)
    check_close(b2_cpu.grad, bsig2.grad, double_atol=1e-10)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_combine_backprop_finite_diff(d, N):
    """Planar combine backprop matches finite differences."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1500)
    bsig_len = pysiglib.branched_sig_length(d, N, planar=True)
    bsig1 = np.random.randn(bsig_len)
    bsig2 = np.random.randn(bsig_len)
    derivs = np.random.randn(bsig_len)

    d1, d2 = pysiglib.branched_sig_combine_backprop(derivs, bsig1, bsig2, d, N, planar=True)

    eps = 1e-8
    fd1 = np.zeros(bsig_len)
    for i in range(bsig_len):
        b1p = bsig1.copy()
        b1p[i] += eps
        cp = np.array(pysiglib.branched_sig_combine(b1p, bsig2, d, N, planar=True))
        cm = np.array(pysiglib.branched_sig_combine(bsig1, bsig2, d, N, planar=True))
        fd1[i] = np.dot(derivs, cp - cm) / eps
    np.testing.assert_allclose(d1, fd1, atol=1e-4)

    fd2 = np.zeros(bsig_len)
    for i in range(bsig_len):
        b2p = bsig2.copy()
        b2p[i] += eps
        cp = np.array(pysiglib.branched_sig_combine(bsig1, b2p, d, N, planar=True))
        cm = np.array(pysiglib.branched_sig_combine(bsig1, bsig2, d, N, planar=True))
        fd2[i] = np.dot(derivs, cp - cm) / eps
    np.testing.assert_allclose(d2, fd2, atol=1e-4)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_combine_backprop_cuda_matches_cpu(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1502)
    bsig_len = pysiglib.branched_sig_length(d, N, planar=True)
    bsig1 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)
    bsig2 = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)
    derivs = torch.tensor(np.random.randn(bsig_len), dtype=torch.float64)

    d1_cpu, d2_cpu = pysiglib.branched_sig_combine_backprop(derivs, bsig1, bsig2, d, N, planar=True)
    d1_cuda, d2_cuda = pysiglib.branched_sig_combine_backprop(
        derivs.cuda(), bsig1.cuda(), bsig2.cuda(), d, N, planar=True)
    check_close(d1_cpu, d1_cuda, double_atol=1e-10)
    check_close(d2_cpu, d2_cuda, double_atol=1e-10)
