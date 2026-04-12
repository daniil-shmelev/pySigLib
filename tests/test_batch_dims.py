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

"""Tests for arbitrary leading batch dimensions.

For each function, we verify that a multi-dimensional batch input
``(*batch_shape, ...)`` produces an output whose leading dims match
``batch_shape``, and whose values match the equivalent flat-batch call.
"""

from math import prod

import numpy as np
import torch
import pytest
import pysiglib

from conftest import check_close

LENGTH = 10
DIM = 3
DEGREE = 3

def setup_module(_module):
    for m in (1, 2):
        pysiglib.prepare_log_sig(DIM, DEGREE, method=m)
        pysiglib.prepare_log_sig(2, 3, method=m)

BATCH_SHAPES = [
    (),          # single (unbatched)
    (4,),        # standard batch
    (2, 3),      # two leading dims
    (2, 3, 2),   # three leading dims
]


def _random_path(batch_shape, length=LENGTH, dim=DIM, dtype=np.float64):
    return np.random.default_rng(42).standard_normal((*batch_shape, length, dim)).astype(dtype)


def _flat_path(path):
    """Flatten all leading dims into one batch dim."""
    if path.ndim == 2:
        return path
    return path.reshape(-1, path.shape[-2], path.shape[-1])


def _flat_sig(sig):
    """Flatten all leading dims into one batch dim."""
    if sig.ndim == 1:
        return sig
    return sig.reshape(-1, sig.shape[-1])


# ---------------------------------------------------------------------------
# Unary path -> sig functions
# ---------------------------------------------------------------------------

class TestSig:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path = _random_path(batch_shape)
        result = pysiglib.sig(path, DEGREE)

        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, sig_len)

        flat_result = pysiglib.sig(_flat_path(path), DEGREE)
        check_close(result.reshape(-1, sig_len), flat_result.reshape(-1, sig_len))

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_torch(self, batch_shape):
        path = torch.from_numpy(_random_path(batch_shape))
        result = pysiglib.sig(path, DEGREE)

        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, sig_len)
        assert isinstance(result, torch.Tensor)


class TestLogSig:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    @pytest.mark.parametrize("method", [1, 2])
    def test_shape_and_values(self, batch_shape, method):
        path = _random_path(batch_shape)
        result = pysiglib.log_sig(path, DEGREE, method=method)

        ls_len = pysiglib.log_sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, ls_len)

        flat_result = pysiglib.log_sig(_flat_path(path), DEGREE, method=method)
        check_close(result.reshape(-1, ls_len), flat_result.reshape(-1, ls_len))


class TestTransformPath:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_time_aug(self, batch_shape):
        path = _random_path(batch_shape)
        result = pysiglib.transform_path(path, time_aug=True)

        assert result.shape == (*batch_shape, LENGTH, DIM + 1)


class TestBranchedSig:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path = _random_path(batch_shape, dim=2)
        N = 3
        pysiglib.prepare_branched_sig(2, N)
        result = pysiglib.branched_sig(path, N)

        bsig_len = pysiglib.branched_sig_length(2, N)
        assert result.shape == (*batch_shape, bsig_len)


# ---------------------------------------------------------------------------
# Unary sig -> sig functions
# ---------------------------------------------------------------------------

class TestSigToLogSig:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path = _random_path(batch_shape)
        sig_ = pysiglib.sig(path, DEGREE)
        result = pysiglib.sig_to_log_sig(sig_, DIM, DEGREE)

        ls_len = pysiglib.log_sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, ls_len)


class TestLogSigToSig:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path = _random_path(batch_shape)
        ls = pysiglib.log_sig(path, DEGREE)
        result = pysiglib.logsig_to_sig(ls, DIM, DEGREE)

        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, sig_len)


# ---------------------------------------------------------------------------
# Binary sig functions
# ---------------------------------------------------------------------------

class TestSigCombine:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path1 = _random_path(batch_shape, length=5)
        path2 = _random_path(batch_shape, length=5)
        sig1 = pysiglib.sig(path1, DEGREE)
        sig2 = pysiglib.sig(path2, DEGREE)
        result = pysiglib.sig_combine(sig1, sig2, DIM, DEGREE)

        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, sig_len)


class TestLogSigCombine:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path1 = _random_path(batch_shape, length=5)
        path2 = _random_path(batch_shape, length=5)
        ls1 = pysiglib.log_sig(path1, DEGREE)
        ls2 = pysiglib.log_sig(path2, DEGREE)
        result = pysiglib.log_sig_combine(ls1, ls2, DIM, DEGREE)

        ls_len = pysiglib.log_sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, ls_len)


class TestSigJoin:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path = _random_path(batch_shape, length=5)
        sig_ = pysiglib.sig(path, DEGREE)
        disp = np.random.default_rng(99).standard_normal((*batch_shape, DIM))
        result = pysiglib.sig_join(sig_, disp, DIM, DEGREE)

        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, sig_len)


class TestLogSigJoin:
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_shape_and_values(self, batch_shape):
        path = _random_path(batch_shape, length=5)
        ls = pysiglib.log_sig(path, DEGREE)
        disp = np.random.default_rng(99).standard_normal((*batch_shape, DIM))
        result = pysiglib.log_sig_join(ls, disp, DIM, DEGREE)

        ls_len = pysiglib.log_sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, ls_len)


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

class TestSigKernel:
    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_shape(self, batch_shape):
        path1 = _random_path(batch_shape, length=5, dim=2)
        path2 = _random_path(batch_shape, length=5, dim=2)
        result = pysiglib.sig_kernel(path1, path2, 0)

        assert result.shape == batch_shape


# ---------------------------------------------------------------------------
# PyTorch autograd with multi-dim batches
# ---------------------------------------------------------------------------

class TestTorchAutograd:
    @pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
    def test_sig_grad(self, batch_shape):
        path = torch.from_numpy(_random_path(batch_shape)).requires_grad_(True)
        result = pysiglib.torch_api.sig(path, DEGREE)

        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, sig_len)

        loss = result.sum()
        loss.backward()
        assert path.grad is not None
        assert path.grad.shape == path.shape

    @pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
    def test_log_sig_grad(self, batch_shape):
        path = torch.from_numpy(_random_path(batch_shape)).requires_grad_(True)
        result = pysiglib.torch_api.log_sig(path, DEGREE)

        ls_len = pysiglib.log_sig_length(DIM, DEGREE)
        assert result.shape == (*batch_shape, ls_len)

        loss = result.sum()
        loss.backward()
        assert path.grad is not None
        assert path.grad.shape == path.shape

    @pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
    def test_sig_combine_grad(self, batch_shape):
        path1 = torch.from_numpy(_random_path(batch_shape, length=5)).requires_grad_(True)
        path2 = torch.from_numpy(_random_path(batch_shape, length=5)).requires_grad_(True)
        sig1 = pysiglib.torch_api.sig(path1, DEGREE)
        sig2 = pysiglib.torch_api.sig(path2, DEGREE)
        result = pysiglib.torch_api.sig_combine(sig1, sig2, DIM, DEGREE)
        loss = result.sum()
        loss.backward()
        assert path1.grad is not None
        assert path1.grad.shape == path1.shape


# ---------------------------------------------------------------------------
# Empty batch (batch dim = 0)
# ---------------------------------------------------------------------------

class TestEmptyBatch:
    def test_sig_empty(self):
        path = np.empty((0, LENGTH, DIM), dtype=np.float64)
        result = pysiglib.sig(path, DEGREE)
        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (0, sig_len)

    def test_sig_multi_dim_empty(self):
        path = np.empty((3, 0, LENGTH, DIM), dtype=np.float64)
        result = pysiglib.sig(path, DEGREE)
        sig_len = pysiglib.sig_length(DIM, DEGREE)
        assert result.shape == (3, 0, sig_len)

    def test_log_sig_empty(self):
        path = np.empty((0, LENGTH, DIM), dtype=np.float64)
        result = pysiglib.log_sig(path, DEGREE, method=2)
        ls_len = pysiglib.log_sig_length(DIM, DEGREE)
        assert result.shape == (0, ls_len)

    def test_sig_combine_empty(self):
        sig_len = pysiglib.sig_length(DIM, DEGREE)
        s1 = np.empty((0, sig_len), dtype=np.float64)
        s2 = np.empty((0, sig_len), dtype=np.float64)
        result = pysiglib.sig_combine(s1, s2, DIM, DEGREE)
        assert result.shape == (0, sig_len)

    def test_transform_path_empty(self):
        path = np.empty((0, LENGTH, DIM), dtype=np.float64)
        result = pysiglib.transform_path(path, time_aug=True)
        assert result.shape == (0, LENGTH, DIM + 1)

    def test_sig_join_empty(self):
        sig_len = pysiglib.sig_length(DIM, DEGREE)
        s = np.empty((0, sig_len), dtype=np.float64)
        d = np.empty((0, DIM), dtype=np.float64)
        result = pysiglib.sig_join(s, d, DIM, DEGREE)
        assert result.shape == (0, sig_len)
