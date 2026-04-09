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


# ---- Fixtures ----

@pytest.fixture(params=[3, 5])
def dimension(request):
    return request.param

@pytest.fixture(params=[2, 4])
def degree(request):
    return request.param


# ---- SigStream tests (numpy) ----

class TestSigStream:
    def test_full_path_matches_sig(self, dimension, degree):
        """push_batch endpoint should match pysiglib.sig of the full path."""
        path = np.random.randn(20, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        stream.push_batch(path)
        # push_batch stores [identity, endpoint] at indices [0, 1]
        stream_sig = stream.sig(0, 1)
        direct_sig = pysiglib.sig(path, degree)
        np.testing.assert_allclose(stream_sig, direct_sig, rtol=1e-3)

    def test_interval_via_push(self, dimension, degree):
        """Per-point push allows querying arbitrary sub-intervals."""
        path = np.random.randn(30, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        for i in range(len(path)):
            stream.push(path[i])
        for start, end in [(5, 15), (0, 10), (10, 29)]:
            stream_sig = stream.sig(start, end)
            direct_sig = pysiglib.sig(path[start:end + 1], degree)
            np.testing.assert_allclose(stream_sig, direct_sig, rtol=1e-3)

    def test_multiple_batches(self, dimension, degree):
        """Multiple push_batch calls accumulate correctly."""
        path = np.random.randn(30, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        stream.push_batch(path[:10])
        stream.push_batch(path[10:20])
        stream.push_batch(path[20:])
        # 4 cumulative sigs: identity + 3 batch endpoints
        assert stream.size == 4
        full_sig = stream.sig(0, 3)
        direct = pysiglib.sig(path, degree)
        np.testing.assert_allclose(full_sig, direct, rtol=1e-3)

    def test_sig_all(self, dimension, degree):
        """sig_all should return cumulative signatures at batch boundaries."""
        path = np.random.randn(10, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        stream.push_batch(path)
        all_sigs = stream.sig_all()
        assert all_sigs.shape[0] == 2
        np.testing.assert_allclose(all_sigs[0, 0], 1.0)
        np.testing.assert_allclose(all_sigs[0, 1:], 0.0, atol=1e-15)

    def test_pop_front(self, dimension, degree):
        """After pop_front, earlier indices are invalid but later queries still work."""
        np.random.seed(0)
        path = np.random.randn(20, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        for i in range(len(path)):
            stream.push(path[i])
        stream.pop_front()
        stream.pop_front()
        assert stream.start_index == 2
        assert stream.size == 18
        sig_5_15 = stream.sig(5, 15)
        direct = pysiglib.sig(path[5:16], degree)
        np.testing.assert_allclose(sig_5_15, direct, rtol=1e-3)

    def test_push_incremental_matches_batch(self, dimension, degree):
        """Per-point push and push_batch should give same endpoint signature."""
        path = np.random.randn(15, dimension)
        stream1 = pysiglib.SigStream(dimension, degree)
        stream1.push_batch(path)
        stream2 = pysiglib.SigStream(dimension, degree)
        for i in range(len(path)):
            stream2.push(path[i])
        # Compare full-path signature: batch stores at index 1, per-point at index 14
        np.testing.assert_allclose(stream1.sig(0, 1), stream2.sig(0, 14), rtol=1e-3)


# ---- LogSigStream tests (numpy) ----

class TestLogSigStream:
    def test_full_path_matches_log_sig(self, dimension, degree):
        """push_batch endpoint should match pysiglib.log_sig of the full path."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        path = np.random.randn(20, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        stream.push_batch(path)
        stream_ls = stream.sig(0, 1)
        direct_ls = pysiglib.log_sig(path, degree, method=2)
        np.testing.assert_allclose(stream_ls, direct_ls, rtol=1e-3)

    def test_interval_via_push(self, dimension, degree):
        """Per-point push allows querying arbitrary sub-intervals."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        path = np.random.randn(30, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        for i in range(len(path)):
            stream.push(path[i])
        for start, end in [(5, 15), (0, 10), (10, 29)]:
            stream_ls = stream.sig(start, end)
            direct_ls = pysiglib.log_sig(path[start:end + 1], degree, method=2)
            np.testing.assert_allclose(stream_ls, direct_ls, rtol=1e-3)

    def test_sig_all(self, dimension, degree):
        """sig_all should return cumulative log-signatures at batch boundaries."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        path = np.random.randn(10, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        stream.push_batch(path)
        all_ls = stream.sig_all()
        assert all_ls.shape[0] == 2
        np.testing.assert_allclose(all_ls[0], 0.0, atol=1e-15)

    def test_pop_front(self, dimension, degree):
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        path = np.random.randn(20, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        for i in range(len(path)):
            stream.push(path[i])
        stream.pop_front()
        stream.pop_front()
        ls_5_15 = stream.sig(5, 15)
        direct = pysiglib.log_sig(path[5:16], degree, method=2)
        np.testing.assert_allclose(ls_5_15, direct, rtol=1e-3)


# ---- SigWindowStream tests ----

class TestSigWindowStream:
    def test_windows_match_direct(self, dimension, degree):
        """Each emitted window should match direct sig on the sub-path."""
        path = np.random.randn(50, dimension)
        window_size = 10
        stride = 5
        ws = pysiglib.SigWindowStream(dimension, degree, window_size=window_size, stride=stride)
        ws.push_batch(path)
        window_sigs = ws.sig()
        expected_n = (len(path) - window_size) // stride + 1
        assert ws.num_windows == expected_n
        for i in range(ws.num_windows):
            w_start = i * stride
            w_end = w_start + window_size
            direct = pysiglib.sig(path[w_start:w_end], degree)
            np.testing.assert_allclose(window_sigs[i], direct, rtol=1e-3)

    def test_stride_1(self, dimension, degree):
        """stride=1 should produce a window starting at every point."""
        path = np.random.randn(15, dimension)
        window_size = 5
        ws = pysiglib.SigWindowStream(dimension, degree, window_size=window_size, stride=1)
        ws.push_batch(path)
        assert ws.num_windows == len(path) - window_size + 1


# ---- LogSigWindowStream tests ----

class TestLogSigWindowStream:
    def test_windows_match_direct(self, dimension, degree):
        """Each emitted window should match direct log_sig on the sub-path."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        path = np.random.randn(50, dimension)
        window_size = 10
        stride = 5
        ws = pysiglib.LogSigWindowStream(dimension, degree, window_size=window_size, stride=stride)
        ws.push_batch(path)
        window_ls = ws.sig()
        expected_n = (len(path) - window_size) // stride + 1
        assert ws.num_windows == expected_n
        for i in range(ws.num_windows):
            w_start = i * stride
            w_end = w_start + window_size
            direct = pysiglib.log_sig(path[w_start:w_end], degree, method=3)
            np.testing.assert_allclose(window_ls[i], direct, rtol=1e-8)


# ---- Torch tensor tests (no autograd — base API is forward-only) ----

class TestTorchTensors:
    def test_sig_stream_torch(self):
        """SigStream should work with torch tensors (forward-only)."""
        dim, deg = 3, 3
        path = torch.randn(15, dim, dtype=torch.float64)
        stream = pysiglib.SigStream(dim, deg)
        stream.push_batch(path)
        result = stream.sig(0, 1)
        assert isinstance(result, torch.Tensor)

    def test_log_sig_stream_torch(self):
        """LogSigStream should work with torch tensors (forward-only)."""
        dim, deg = 3, 3
        pysiglib.prepare_log_sig(dim, deg, method=2)
        path = torch.randn(15, dim, dtype=torch.float64)
        stream = pysiglib.LogSigStream(dim, deg)
        stream.push_batch(path)
        result = stream.sig(0, 1)
        assert isinstance(result, torch.Tensor)


# ---- Torch autograd tests (via torch_api) ----

class TestTorchAutograd:
    def test_sig_stream_backward(self):
        """Gradients should flow through torch_api.SigStream (per-point push)."""
        from pysiglib.torch_api import SigStream
        dim, deg = 3, 3
        path = torch.randn(15, dim, dtype=torch.float64, requires_grad=True)
        stream = SigStream(dim, deg)
        for i in range(path.shape[0]):
            stream.push(path[i])
        result = stream.sig(0, 14)
        loss = result.sum()
        loss.backward()
        assert path.grad is not None
        assert not torch.all(path.grad == 0)

    def test_sig_stream_interval_backward(self):
        """Gradients should flow through interval queries."""
        from pysiglib.torch_api import SigStream
        dim, deg = 3, 3
        path = torch.randn(20, dim, dtype=torch.float64, requires_grad=True)
        stream = SigStream(dim, deg)
        for i in range(path.shape[0]):
            stream.push(path[i])
        result = stream.sig(5, 15)
        loss = result.sum()
        loss.backward()
        assert path.grad is not None

    def test_log_sig_stream_backward(self):
        """Gradients should flow through torch_api.LogSigStream (per-point push)."""
        from pysiglib.torch_api import LogSigStream
        dim, deg = 3, 3
        pysiglib.prepare_log_sig(dim, deg, method=2)
        path = torch.randn(15, dim, dtype=torch.float64, requires_grad=True)
        stream = LogSigStream(dim, deg)
        for i in range(path.shape[0]):
            stream.push(path[i])
        result = stream.sig(0, 14)
        loss = result.sum()
        loss.backward()
        assert path.grad is not None
        assert not torch.all(path.grad == 0)
