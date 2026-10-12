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

import numpy as np
import pytest
import torch

import pysiglib
from conftest import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("dimension,degree", [(1, 3), (1, 4), (2, 4)])
def test_zero_correction_matches_default(device, planar, dimension, degree):
    pysiglib.prepare_branched_sig(dimension, degree, planar=planar, device=device)
    pysiglib.prepare_branched_log_sig(dimension, degree, 0, planar=planar, device=device)
    path = torch.tensor(np.random.default_rng(77).normal(size=(4, dimension)),
                        dtype=torch.float64, device=device)
    correction = torch.zeros((3, dimension ** 2), dtype=path.dtype, device=device)
    options = dict(planar=planar, scalar_term=True)
    expected = pysiglib.branched_sig(path, degree, **options)
    actual = pysiglib.branched_sig(path, degree, correction=correction, **options)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    weights = torch.arange(actual.numel(), dtype=path.dtype, device=device)
    expected_grad = pysiglib.branched_sig_backprop(path, expected, weights, degree, planar=planar)
    actual_grad = pysiglib.branched_sig_backprop(
        path, actual, weights, degree, correction=correction, planar=planar)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)

    expected_log = pysiglib.branched_log_sig(path, degree, method=0, planar=planar)
    actual_log = pysiglib.branched_log_sig(path, degree, method=0, planar=planar, correction=correction)
    torch.testing.assert_close(actual_log, expected_log, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("subdivide", [False, True])
def test_zero_correction_fork_coefficient_and_gradient(device, dtype, subdivide):
    fork = ((0,), (0,), 0)
    degree = 3
    pysiglib.prepare_branched_sig(1, degree, device=device)
    pysiglib.prepare_branched_sig_coef(1, fork, device=device)
    path = torch.tensor([[0.0], [0.5], [1.0]] if subdivide else [[0.0], [1.0]],
                        dtype=dtype, device=device)
    correction = torch.zeros((len(path) - 1, 1), dtype=dtype, device=device)
    full = pysiglib.branched_sig(path, degree, correction=correction)
    index = pysiglib.tree_to_idx(fork, 1, degree)
    selected = pysiglib.branched_sig_coef(path, fork, correction=correction)
    torch.testing.assert_close(full[index], path.new_tensor(1 / 3))
    torch.testing.assert_close(selected, path.new_full(selected.shape, 1 / 3))
    weights = torch.zeros_like(full)
    weights[index] = 1
    grad = pysiglib.branched_sig_backprop(
        path, full, weights, degree, correction=correction)
    selected_grad = pysiglib.branched_sig_coef_backprop(
        path, fork, selected, torch.ones_like(selected), correction=correction)
    expected = torch.zeros_like(path)
    expected[0] = -1
    expected[-1] = 1
    torch.testing.assert_close(grad, expected)
    torch.testing.assert_close(selected_grad, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_correction_fork(device):
    # exp_*(x leaf + c2 chain2 + c3 chain3)(fork) = x^3 / 3 + x c2.
    fork = ((0,), (0,), 0)
    pysiglib.prepare_branched_sig(1, 4, device=device)
    pysiglib.prepare_branched_sig_coef(1, fork, device=device)
    path = torch.tensor([[0.0], [0.7]], dtype=torch.float64, device=device)
    correction = torch.tensor([[0.4, -0.15]], dtype=path.dtype, device=device)
    full = pysiglib.branched_sig(path, 4, correction=correction)
    index = pysiglib.tree_to_idx(fork, 1, 4)
    selected = pysiglib.branched_sig_coef(path, fork, correction=correction)
    expected = 0.7 ** 3 / 3 + 0.7 * 0.4
    torch.testing.assert_close(full[index], path.new_tensor(expected))
    torch.testing.assert_close(selected, path.new_full(selected.shape, expected))
    weights = torch.zeros_like(full)
    weights[index] = 1
    grad = pysiglib.branched_sig_backprop(path, full, weights, 4, correction=correction)
    selected_grad = pysiglib.branched_sig_coef_backprop(
        path, fork, selected, torch.ones_like(selected), correction=correction)
    expected_grad = path.new_tensor([[-(0.7 ** 2 + 0.4)], [0.7 ** 2 + 0.4]])
    torch.testing.assert_close(grad, expected_grad)
    torch.testing.assert_close(selected_grad, expected_grad)
