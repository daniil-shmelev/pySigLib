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

from conftest import check_close


@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_random(deg):
    dim = 5
    X1 = torch.rand(size=(100, dim), dtype=torch.float64)
    X2 = torch.rand(size=(100, dim), dtype=torch.float64)
    X = torch.cat((X1, X2), dim=0)
    X2 = torch.cat((X1[[-1], :], X2), dim=0)

    pysiglib.prepare_log_sig(dim, deg, 2)
    ls1 = pysiglib.log_sig(X1, deg, method=2)
    ls2 = pysiglib.log_sig(X2, deg, method=2)
    ls_expected = pysiglib.log_sig(X, deg, method=2)

    ls_combined = pysiglib.log_sig_combine(ls1, ls2, dim, deg)
    check_close(ls_expected, ls_combined)


@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_random_batch(deg):
    dim = 5
    X1 = torch.rand(size=(32, 100, dim), dtype=torch.float64)
    X2 = torch.rand(size=(32, 100, dim), dtype=torch.float64)
    X = torch.cat((X1, X2), dim=1)
    X2 = torch.cat((X1[:, [-1], :], X2), dim=1)

    pysiglib.prepare_log_sig(dim, deg, 2)
    ls1 = pysiglib.log_sig(X1, deg, method=2)
    ls2 = pysiglib.log_sig(X2, deg, method=2)
    ls_expected = pysiglib.log_sig(X, deg, method=2)

    ls_combined = pysiglib.log_sig_combine(ls1, ls2, dim, deg)
    check_close(ls_expected, ls_combined)


def test_log_sig_combine_non_contiguous():
    dim, degree, batch = 3, 3, 16
    ls_length = pysiglib.log_sig_length(dim, degree)

    rand_data = torch.rand(size=(batch,), dtype=torch.float64)[:, None]
    X_non_cont = rand_data.expand(-1, ls_length)
    X = X_non_cont.clone()

    res1 = pysiglib.log_sig_combine(X, X, dim, degree)
    res2 = pysiglib.log_sig_combine(X_non_cont, X_non_cont, dim, degree)
    check_close(res1, res2)

    rand_data = np.random.normal(size=batch)[:, None]
    X_non_cont = np.broadcast_to(rand_data, (batch, ls_length))
    X = np.array(X_non_cont)

    res1 = pysiglib.log_sig_combine(X, X, dim, degree)
    res2 = pysiglib.log_sig_combine(X_non_cont, X_non_cont, dim, degree)
    check_close(res1, res2)


@pytest.mark.parametrize("deg", range(1, 6))
def test_log_sig_combine_identity(deg):
    dim = 5
    X = torch.rand(size=(50, dim), dtype=torch.float64)
    pysiglib.prepare_log_sig(dim, deg, 2)
    ls = pysiglib.log_sig(X, deg, method=2)

    ls_length = pysiglib.log_sig_length(dim, deg)
    zero = torch.zeros(ls_length, dtype=torch.float64)

    ls_combined = pysiglib.log_sig_combine(ls, zero, dim, deg)
    check_close(ls, ls_combined)


@pytest.mark.parametrize("deg", range(2, 6))
def test_log_sig_combine_associative(deg):
    dim = 3
    X1 = torch.rand(size=(30, dim), dtype=torch.float64)
    X2 = torch.rand(size=(30, dim), dtype=torch.float64)
    X3 = torch.rand(size=(30, dim), dtype=torch.float64)

    X2 = torch.cat((X1[[-1], :], X2), dim=0)
    X3 = torch.cat((torch.cat((X1, X2[1:]), dim=0)[[-1], :], X3), dim=0)

    pysiglib.prepare_log_sig(dim, deg, 2)
    ls1 = pysiglib.log_sig(X1, deg, method=2)
    ls2 = pysiglib.log_sig(X2, deg, method=2)
    ls3 = pysiglib.log_sig(X3, deg, method=2)

    # (L1 * L2) * L3
    ls12 = pysiglib.log_sig_combine(ls1, ls2, dim, deg)
    ls_left = pysiglib.log_sig_combine(ls12, ls3, dim, deg)

    # L1 * (L2 * L3)
    ls23 = pysiglib.log_sig_combine(ls2, ls3, dim, deg)
    ls_right = pysiglib.log_sig_combine(ls1, ls23, dim, deg)

    check_close(ls_left, ls_right)
