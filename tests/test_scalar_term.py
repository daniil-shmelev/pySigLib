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

"""Tests for the scalar_term flag."""

import warnings
import numpy as np
import torch
import pytest
import pysiglib


DIM = 3
DEGREE = 3
LENGTH = 15


def setup_module(_module):
    for m in (1, 2):
        pysiglib.prepare_log_sig(DIM, DEGREE, method=m)
    pysiglib.prepare_branched_sig(DIM, DEGREE)


def _path(dtype=np.float64):
    return np.random.default_rng(42).standard_normal((LENGTH, DIM)).astype(dtype)


def _batch_path(dtype=np.float64):
    return np.random.default_rng(42).standard_normal((5, LENGTH, DIM)).astype(dtype)


class TestSig:
    def test_shape(self):
        path = _path()
        s_full = pysiglib.sig(path, DEGREE, scalar_term=True)
        s_no = pysiglib.sig(path, DEGREE, scalar_term=False)
        assert s_full.shape[-1] == pysiglib.sig_length(DIM, DEGREE, scalar_term=True)
        assert s_no.shape[-1] == pysiglib.sig_length(DIM, DEGREE, scalar_term=False)
        assert s_full.shape[-1] == s_no.shape[-1] + 1

    def test_values_match(self):
        path = _path()
        s_full = pysiglib.sig(path, DEGREE, scalar_term=True)
        s_no = pysiglib.sig(path, DEGREE, scalar_term=False)
        np.testing.assert_allclose(s_full[1:], s_no, atol=1e-12)

    def test_batch(self):
        path = _batch_path()
        s_full = pysiglib.sig(path, DEGREE, scalar_term=True)
        s_no = pysiglib.sig(path, DEGREE, scalar_term=False)
        np.testing.assert_allclose(s_full[..., 1:], s_no, atol=1e-12)


class TestSigCombine:
    def test_strip_output(self):
        p1 = _path()[:8]
        p2 = _path()[7:]
        s1 = pysiglib.sig(p1, DEGREE, scalar_term=True)
        s2 = pysiglib.sig(p2, DEGREE, scalar_term=True)
        combined_full = pysiglib.sig_combine(s1, s2, DIM, DEGREE, scalar_term=True)
        combined_no = pysiglib.sig_combine(s1, s2, DIM, DEGREE, scalar_term=False)

        np.testing.assert_allclose(combined_full[1:], combined_no, atol=1e-10)


class TestLogSig:
    @pytest.mark.parametrize("method", [1, 2])
    def test_log_sig_unaffected(self, method):
        path = _path()
        ls_true = pysiglib.log_sig(path, DEGREE, scalar_term=True, method=method)
        ls_false = pysiglib.log_sig(path, DEGREE, scalar_term=False, method=method)
        np.testing.assert_allclose(ls_true, ls_false, atol=1e-12)

    def test_log_sig_method0_strips(self):
        path = _path()
        ls_true = pysiglib.log_sig(path, DEGREE, scalar_term=True, method=0)
        ls_false = pysiglib.log_sig(path, DEGREE, scalar_term=False, method=0)
        assert ls_true.shape[-1] == ls_false.shape[-1] + 1
        np.testing.assert_allclose(ls_true[1:], ls_false, atol=1e-12)


class TestBranchedSig:
    def test_shape(self):
        path = _path()
        bs_full = pysiglib.branched_sig(path, DEGREE, scalar_term=True)
        bs_no = pysiglib.branched_sig(path, DEGREE, scalar_term=False)
        assert bs_full.shape[-1] == bs_no.shape[-1] + 1

    def test_values_match(self):
        path = _path()
        bs_full = pysiglib.branched_sig(path, DEGREE, scalar_term=True)
        bs_no = pysiglib.branched_sig(path, DEGREE, scalar_term=False)
        np.testing.assert_allclose(bs_full[1:], bs_no, atol=1e-12)


class TestLinearSig:
    def test_shape(self):
        disp = np.random.default_rng(42).standard_normal(DIM)
        ls_full = pysiglib.linear_sig(disp, DIM, DEGREE, scalar_term=True)
        ls_no = pysiglib.linear_sig(disp, DIM, DEGREE, scalar_term=False)
        assert ls_full.shape[-1] == ls_no.shape[-1] + 1
        np.testing.assert_allclose(ls_full[1:], ls_no, atol=1e-12)


class TestTorchAutograd:
    def test_sig_grad(self):
        path = torch.from_numpy(_path()).requires_grad_(True)
        result = pysiglib.torch_api.sig(path, DEGREE, scalar_term=False)
        assert result.shape[-1] == pysiglib.sig_length(DIM, DEGREE, scalar_term=False)
        loss = result.sum()
        loss.backward()
        assert path.grad is not None
        assert path.grad.shape == path.shape

    def test_sig_combine_grad(self):
        p1 = torch.from_numpy(_path()[:8]).requires_grad_(True)
        p2 = torch.from_numpy(_path()[7:]).requires_grad_(True)
        s1 = pysiglib.torch_api.sig(p1, DEGREE, scalar_term=True)
        s2 = pysiglib.torch_api.sig(p2, DEGREE, scalar_term=True)
        combined = pysiglib.torch_api.sig_combine(s1, s2, DIM, DEGREE, scalar_term=False)
        assert combined.shape[-1] == pysiglib.sig_length(DIM, DEGREE, scalar_term=False)
        loss = combined.sum()
        loss.backward()
        assert p1.grad is not None


class TestFutureWarning:
    def test_warning_emitted_when_unset(self):
        path = _path()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pysiglib.sig(path, DEGREE)
            assert any(issubclass(x.category, FutureWarning) for x in w)

    def test_no_warning_when_explicit(self):
        path = _path()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pysiglib.sig(path, DEGREE, scalar_term=True)
            future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warnings) == 0


class TestSigLength:
    def test_scalar_term_true(self):
        assert pysiglib.sig_length(3, 4, scalar_term=True) == 121

    def test_scalar_term_false(self):
        assert pysiglib.sig_length(3, 4, scalar_term=False) == 120


class TestBranchedSigLength:
    def test_scalar_term_default(self):
        full = pysiglib.branched_sig_length(DIM, DEGREE)
        no_scalar = pysiglib.branched_sig_length(DIM, DEGREE, scalar_term=False)
        assert full == no_scalar + 1
