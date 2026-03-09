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
import sigkernel

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

EPSILON = 1e-5

skip_no_cuda = pytest.mark.skipif(
    not (pysiglib.BUILT_WITH_CUDA and torch.cuda.is_available()),
    reason="CUDA not available or disabled"
)

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    d1 = float(signature_kernel.compute_expected_scoring_rule(X, Y, 100).cpu())
    d2 = pysiglib.expected_sig_score(X, Y, dyadic_order)

    assert not abs(d1 - d2) > EPSILON

@skip_no_cuda
@pytest.mark.parametrize(("len1", "len2"), [(10, 50), (50, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random_non_square_cuda(len1, len2, dyadic_order):
    batch, dim = 32, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    d1 = float(signature_kernel.compute_expected_scoring_rule(X, Y, 100).cpu())
    d2 = pysiglib.expected_sig_score(X, Y, dyadic_order)

    assert not abs(d1 - d2) > EPSILON

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random_cuda_rbf(dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.RBFKernel(2.)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    d1 = float(signature_kernel.compute_expected_scoring_rule(X, Y, 100).cpu())
    d2 = pysiglib.expected_sig_score(X, Y, dyadic_order, static_kernel= pysiglib.RBFKernel(2.))

    assert not abs(d1 - d2) > EPSILON

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random_cuda(dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    mmd1 = float(signature_kernel.compute_mmd(X, Y, 100).cpu())
    mmd2 = pysiglib.sig_mmd(X, Y, dyadic_order)

    assert not abs(mmd1 - mmd2) > EPSILON

@skip_no_cuda
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random_cuda_rbf(dyadic_order):
    batch, len1, len2, dim = 32, 10, 10, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.RBFKernel(2.)
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    mmd1 = float(signature_kernel.compute_mmd(X, Y, 100).cpu())
    mmd2 = pysiglib.sig_mmd(X, Y, dyadic_order, static_kernel= pysiglib.RBFKernel(2.))

    assert not abs(mmd1 - mmd2) > EPSILON

@skip_no_cuda
@pytest.mark.parametrize(("len1", "len2"), [(10, 50), (50, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random_non_square_cuda(len1, len2, dyadic_order):
    batch, dim = 32, 5
    X = torch.rand(size=(batch, len1, dim), device="cuda", dtype = torch.double)
    Y = torch.rand(size=(batch, len2, dim), device="cuda", dtype = torch.double)

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    mmd1 = float(signature_kernel.compute_mmd(X, Y, 100).cpu())
    mmd2 = pysiglib.sig_mmd(X, Y, dyadic_order)

    assert not abs(mmd1 - mmd2) > EPSILON
