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
import torch
import pytest
import pysiglib


@pytest.fixture(params=[3, 5])
def dimension(request):
    return request.param

@pytest.fixture(params=[2, 3, 4])
def degree(request):
    return request.param


class TestLogSigJoinBackpropNumpy:
    """Test log_sig_join_backprop with numpy arrays."""

    def test_basic(self, dimension, degree):
        """Basic forward + backward should not crash."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        ls_len = pysiglib.log_sig_length(dimension, degree)
        ls = np.random.randn(ls_len)
        disp = np.random.randn(dimension)
        result = pysiglib.log_sig_join(ls, disp, dimension, degree)
        d_out = np.ones_like(result)
        d_ls, d_disp = pysiglib.log_sig_join_backprop(d_out, ls, disp, dimension, degree)
        assert d_ls.shape == ls.shape
        assert d_disp.shape == disp.shape

    def test_zero_logsig(self, dimension, degree):
        """Backward from zero log-sig (identity) should not crash."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        ls_len = pysiglib.log_sig_length(dimension, degree)
        ls = np.zeros(ls_len)
        disp = np.random.randn(dimension)
        result = pysiglib.log_sig_join(ls, disp, dimension, degree)
        d_out = np.ones_like(result)
        d_ls, d_disp = pysiglib.log_sig_join_backprop(d_out, ls, disp, dimension, degree)
        assert d_ls.shape == ls.shape

    def test_chained(self, dimension, degree):
        """Chaining multiple log_sig_join + backprop should work."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        ls_len = pysiglib.log_sig_length(dimension, degree)
        ls = np.zeros(ls_len)
        for _ in range(5):
            disp = np.random.randn(dimension)
            ls = pysiglib.log_sig_join(ls, disp, dimension, degree)
        d_out = np.ones_like(ls)
        # Backprop through the last join only
        d_ls, d_disp = pysiglib.log_sig_join_backprop(d_out, ls, disp, dimension, degree)
        assert d_ls.shape == ls.shape


class TestLogSigJoinBackpropTorch:
    """Test log_sig_join_backprop via torch autograd."""

    def test_single_join_backward(self, dimension, degree):
        """Single log_sig_join backward via autograd."""
        from pysiglib.torch_api import log_sig_join
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        ls_len = pysiglib.log_sig_length(dimension, degree)
        ls = torch.zeros(ls_len, dtype=torch.float64, requires_grad=True)
        disp = torch.randn(dimension, dtype=torch.float64, requires_grad=True)
        result = log_sig_join(ls, disp, dimension, degree)
        result.sum().backward()
        assert disp.grad is not None

    def test_chained_backward(self, dimension, degree):
        """Chained log_sig_join backward via autograd."""
        from pysiglib.torch_api import log_sig_join
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        ls_len = pysiglib.log_sig_length(dimension, degree)
        ls = torch.zeros(ls_len, dtype=torch.float64, requires_grad=True)
        for i in range(5):
            disp = torch.randn(dimension, dtype=torch.float64)
            ls = log_sig_join(ls, disp, dimension, degree)
        ls.sum().backward()
        # Can't check grad on initial ls since it's been overwritten
        # Just verify no crash

    def test_chained_backward_with_grad(self, dimension, degree):
        """Chained backward should propagate gradients to displacement."""
        from pysiglib.torch_api import log_sig_join
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        ls_len = pysiglib.log_sig_length(dimension, degree)
        ls = torch.zeros(ls_len, dtype=torch.float64)
        disp = torch.randn(dimension, dtype=torch.float64, requires_grad=True)
        result = log_sig_join(ls, disp, dimension, degree)
        result.sum().backward()
        assert disp.grad is not None
        assert not torch.all(disp.grad == 0)

    def test_long_chain_backward(self, dimension, degree):
        """14-step chain like LogSigStream uses."""
        from pysiglib.torch_api import log_sig_join
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        ls_len = pysiglib.log_sig_length(dimension, degree)
        path = torch.randn(15, dimension, dtype=torch.float64, requires_grad=True)
        ls = torch.zeros(ls_len, dtype=torch.float64)
        for i in range(14):
            disp = path[i + 1] - path[i]
            ls = log_sig_join(ls, disp, dimension, degree)
        ls.sum().backward()
        assert path.grad is not None
