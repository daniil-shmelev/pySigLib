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

from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import torch

import pysiglib
import pysiglib.torch_api as pysiglib_torch
from conftest import DEVICES, assert_device, check_close as _check_close

check_close = partial(_check_close, single_atol=5e-3, double_atol=1e-5)
check_grad_close = partial(_check_close, single_atol=5e-2, double_atol=5e-4)


def _path(shape, seed):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.normal(size=shape)).astype(np.float64)


def _dyadic_pair(dyadic_order):
    if isinstance(dyadic_order, tuple):
        return dyadic_order
    return dyadic_order, dyadic_order


def _linear_gram(path1, path2):
    dx = np.diff(path1, axis=-2)
    dy = np.diff(path2, axis=-2)
    return dx @ np.swapaxes(dy, -1, -2)


def _reference_grid_from_gram(gram, depth, dyadic_order):
    do1, do2 = _dyadic_pair(dyadic_order)
    factor1 = 1 << do1
    factor2 = 1 << do2
    refined = np.repeat(np.repeat(gram, factor1, axis=-2), factor2, axis=-1)
    dl1 = refined.shape[-2] + 1
    dl2 = refined.shape[-1] + 1
    prev = np.ones(gram.shape[:-2] + (dl1, dl2), dtype=gram.dtype)
    scale = 0.25 / (1 << (do1 + do2))

    for _ in range(depth):
        cell = scale * refined * (
            prev[..., :-1, :-1] + prev[..., 1:, :-1] +
            prev[..., :-1, 1:] + prev[..., 1:, 1:]
        )
        accum = np.cumsum(np.cumsum(cell, axis=-2), axis=-1)
        curr = np.ones_like(prev)
        curr[..., 1:, 1:] = np.exp(accum)
        prev = curr

    return prev


def _reference_grid(path1, path2, depth, dyadic_order):
    return _reference_grid_from_gram(_linear_gram(path1, path2), depth, dyadic_order)


def _as_torch(path, device):
    return torch.as_tensor(path, dtype=torch.float64, device=device)


def _truncated_branched_sig_inner_product(path1, path2, degree):
    pysiglib.prepare_branched_sig(path1.shape[-1], degree)
    sig1 = pysiglib.branched_sig(path1, degree, scalar_term=True)
    sig2 = pysiglib.branched_sig(path2, degree, scalar_term=True)
    return float(np.dot(sig1, sig2))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("dyadic_order", [0, (1, 0)])
def test_branched_sig_kernel_matches_reference(device, depth, dyadic_order):
    path1 = _path((3, 5, 2), 1)
    path2 = _path((3, 4, 2), 2)
    expected = _reference_grid(path1, path2, depth, dyadic_order)[..., -1, -1]

    actual = pysiglib.branched_sig_kernel(
        _as_torch(path1, device), _as_torch(path2, device), depth, dyadic_order)

    assert_device(actual, device)
    check_close(expected, actual)


def test_branched_sig_kernel_numpy_matches_reference():
    path1 = _path((5, 2), 3)
    path2 = _path((4, 2), 4)
    expected = _reference_grid(path1, path2, 2, 1)[-1, -1]

    actual = pysiglib.branched_sig_kernel(path1, path2, 2, 1)

    assert isinstance(actual, np.ndarray)
    check_close(expected, actual)


def test_branched_sig_kernel_matches_truncated_branched_sig_inner_product():
    path1 = np.array([
        [0.00, 0.00],
        [0.03, -0.01],
        [0.02, 0.04],
        [0.06, 0.05],
    ])
    path2 = np.array([
        [0.00, 0.00],
        [-0.02, 0.03],
        [0.01, 0.02],
        [0.04, 0.07],
    ])

    actual = pysiglib.branched_sig_kernel(path1, path2, 2, 0)
    expected = _truncated_branched_sig_inner_product(path1, path2, 6)

    assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_branched_sig_kernel_grid_matches_reference(device):
    path1 = _path((5, 2), 5)
    path2 = _path((4, 2), 6)
    expected = _reference_grid(path1, path2, 2, (1, 0))

    actual = pysiglib.branched_sig_kernel(
        _as_torch(path1, device), _as_torch(path2, device), 2, (1, 0),
        return_grid=True)

    assert_device(actual, device)
    check_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_batch", [-1, 1])
def test_branched_sig_kernel_gram_matches_reference(device, max_batch):
    path1 = _path((2, 5, 2), 7)
    path2 = _path((3, 4, 2), 8)
    expected = np.empty((2, 3), dtype=np.float64)
    for i in range(2):
        for j in range(3):
            expected[i, j] = _reference_grid(path1[i], path2[j], 2, 0)[-1, -1]

    actual = pysiglib.branched_sig_kernel_gram(
        _as_torch(path1, device), _as_torch(path2, device), 2, 0,
        max_batch=max_batch)

    assert_device(actual, device)
    check_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
def test_branched_sig_kernel_gram_symmetric_grid(device):
    path = _as_torch(_path((3, 5, 2), 9), device)

    grid = pysiglib.branched_sig_kernel_gram(path, path, 2, 0, return_grid=True)

    assert_device(grid, device)
    check_close(grid[0, 1], grid[1, 0].transpose(-2, -1))


@pytest.mark.parametrize("device", DEVICES)
def test_branched_sig_kernel_normalize_self(device):
    path = _as_torch(_path((3, 5, 2), 10), device)

    actual = pysiglib.branched_sig_kernel(path, path, 2, 0, normalize=True)

    assert_device(actual, device)
    check_close(torch.ones_like(actual), actual)


def test_branched_sig_kernel_normalize_return_grid_raises():
    path = _path((5, 2), 11)

    with pytest.raises(ValueError, match="normalize.*return_grid"):
        pysiglib.branched_sig_kernel(path, path, 2, 0, normalize=True, return_grid=True)

    with pytest.raises(ValueError, match="normalize.*return_grid"):
        pysiglib.branched_sig_kernel_gram(path, path, 2, 0, normalize=True, return_grid=True)


def _weighted_kernel_value(path1, path2, depth, dyadic_order, weights, return_grid):
    value = pysiglib.branched_sig_kernel(
        torch.as_tensor(path1, dtype=torch.float64),
        torch.as_tensor(path2, dtype=torch.float64),
        depth,
        dyadic_order,
        return_grid=return_grid,
    )
    return float((value * torch.as_tensor(weights, dtype=torch.float64)).sum())


def _finite_difference(path1, path2, depth, dyadic_order, weights, return_grid, arg):
    eps = 1e-6
    base = _weighted_kernel_value(path1, path2, depth, dyadic_order, weights, return_grid)
    target = path1 if arg == "left" else path2
    out = np.empty_like(target)

    for idx in np.ndindex(target.shape):
        path1_step = deepcopy(path1)
        path2_step = deepcopy(path2)
        step_target = path1_step if arg == "left" else path2_step
        step_target[idx] += eps
        value = _weighted_kernel_value(path1_step, path2_step, depth, dyadic_order, weights, return_grid)
        out[idx] = (value - base) / eps

    return out


@pytest.mark.parametrize("device", DEVICES)
def test_branched_sig_kernel_backprop_matches_finite_difference(device):
    path1 = _path((2, 4, 2), 12)
    path2 = _path((2, 5, 2), 13)
    weights = _path((2,), 14)

    ld, rd = pysiglib.branched_sig_kernel_backprop(
        _as_torch(weights, device), _as_torch(path1, device), _as_torch(path2, device),
        2, 0, left_deriv=True, right_deriv=True)

    assert_device(ld, device)
    assert_device(rd, device)
    check_grad_close(_finite_difference(path1, path2, 2, 0, weights, False, "left"), ld)
    check_grad_close(_finite_difference(path1, path2, 2, 0, weights, False, "right"), rd)


@pytest.mark.parametrize("device", DEVICES)
def test_branched_sig_kernel_grid_backprop_endpoint_matches_scalar(device):
    path1 = _as_torch(_path((2, 4, 2), 15), device)
    path2 = _as_torch(_path((2, 5, 2), 16), device)
    weights = _as_torch(_path((2,), 17), device)

    ld_scalar, rd_scalar = pysiglib.branched_sig_kernel_backprop(
        weights, path1, path2, 2, 1, left_deriv=True, right_deriv=True)

    grid = pysiglib.branched_sig_kernel(path1, path2, 2, 1, return_grid=True)
    derivs_grid = torch.zeros_like(grid)
    derivs_grid[:, -1, -1] = weights
    ld_grid, rd_grid = pysiglib.branched_sig_kernel_backprop(
        derivs_grid, path1, path2, 2, 1, left_deriv=True, right_deriv=True,
        return_grid=True)

    check_grad_close(ld_scalar, ld_grid)
    check_grad_close(rd_scalar, rd_grid)


@pytest.mark.parametrize("device", DEVICES)
def test_branched_sig_kernel_gram_backprop_max_batch(device):
    path1 = _as_torch(_path((2, 4, 2), 18), device)
    path2 = _as_torch(_path((3, 5, 2), 19), device)
    derivs = _as_torch(_path((2, 3), 20), device)

    ld, rd = pysiglib.branched_sig_kernel_gram_backprop(
        derivs, path1, path2, 2, 0, left_deriv=True, right_deriv=True)
    ld_mb, rd_mb = pysiglib.branched_sig_kernel_gram_backprop(
        derivs, path1, path2, 2, 0, left_deriv=True, right_deriv=True,
        max_batch=1)

    assert_device(ld, device)
    assert_device(rd, device)
    check_grad_close(ld, ld_mb)
    check_grad_close(rd, rd_mb)


def test_branched_sig_kernel_leading_batch_dims_match_flattened():
    path1 = _path((2, 3, 4, 2), 21)
    path2 = _path((2, 3, 5, 2), 22)

    actual = pysiglib.branched_sig_kernel(path1, path2, 2, 0)
    expected = pysiglib.branched_sig_kernel(
        path1.reshape(-1, 4, 2), path2.reshape(-1, 5, 2), 2, 0)

    assert actual.shape == (2, 3)
    check_close(actual.reshape(-1), expected)


def test_branched_sig_kernel_gram_leading_batch_dims_match_flattened():
    path1 = _path((2, 3, 4, 2), 23)
    path2 = _path((4, 5, 2), 24)

    actual = pysiglib.branched_sig_kernel_gram(path1, path2, 2, 0)
    expected = pysiglib.branched_sig_kernel_gram(path1.reshape(-1, 4, 2), path2, 2, 0)

    assert actual.shape == (2, 3, 4)
    check_close(actual.reshape(6, 4), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_torch_api_branched_sig_kernel_backward_matches_explicit(device):
    path1 = _as_torch(_path((2, 4, 2), 25), device).requires_grad_(True)
    path2 = _as_torch(_path((2, 5, 2), 26), device).requires_grad_(True)
    weights = _as_torch(_path((2,), 27), device)

    value = pysiglib_torch.branched_sig_kernel(path1, path2, 2, 0)
    value.backward(weights)
    ld, rd = pysiglib.branched_sig_kernel_backprop(
        weights, path1.detach(), path2.detach(), 2, 0,
        left_deriv=True, right_deriv=True)

    check_grad_close(path1.grad, ld)
    check_grad_close(path2.grad, rd)


@pytest.mark.parametrize("device", DEVICES)
def test_torch_api_branched_sig_kernel_gram_backward_matches_explicit(device):
    path1 = _as_torch(_path((2, 4, 2), 28), device).requires_grad_(True)
    path2 = _as_torch(_path((3, 5, 2), 29), device).requires_grad_(True)
    weights = _as_torch(_path((2, 3), 30), device)

    value = pysiglib_torch.branched_sig_kernel_gram(path1, path2, 2, 0)
    value.backward(weights)
    ld, rd = pysiglib.branched_sig_kernel_gram_backprop(
        weights, path1.detach(), path2.detach(), 2, 0,
        left_deriv=True, right_deriv=True)

    check_grad_close(path1.grad, ld)
    check_grad_close(path2.grad, rd)
