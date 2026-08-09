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

@pytest.fixture(params=[(), (4,), (2, 3)], ids=["single", "flat4", "2x3"])
def batch_shape(request):
    return request.param


# ---- SigStream tests (numpy) ----

class TestSigStream:
    def test_full_path_matches_sig(self, dimension, degree, batch_shape):
        """push_batch endpoint should match pysiglib.sig of the full path."""
        path = np.random.randn(*batch_shape, 20, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        stream.push_batch(path)
        # push_batch stores [identity, endpoint] at indices [0, 1]
        stream_sig = stream.sig(0, 1)
        direct_sig = pysiglib.sig(path, degree)
        np.testing.assert_allclose(stream_sig, direct_sig, rtol=1e-3, atol=1e-10)

    def test_interval_via_push(self, dimension, degree, batch_shape):
        """Per-point push allows querying arbitrary sub-intervals."""
        path = np.random.randn(*batch_shape, 30, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        for i in range(path.shape[-2]):
            stream.push(path[..., i, :])
        for start, end in [(5, 15), (0, 10), (10, 29)]:
            stream_sig = stream.sig(start, end)
            direct_sig = pysiglib.sig(path[..., start:end + 1, :], degree)
            np.testing.assert_allclose(stream_sig, direct_sig, rtol=1e-3, atol=1e-10)

    def test_multiple_batches(self, dimension, degree, batch_shape):
        """Multiple push_batch calls accumulate correctly."""
        path = np.random.randn(*batch_shape, 30, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        stream.push_batch(path[..., :10, :])
        stream.push_batch(path[..., 10:20, :])
        stream.push_batch(path[..., 20:, :])
        # 4 cumulative sigs: identity + 3 batch endpoints
        assert stream.size == 4
        full_sig = stream.sig(0, 3)
        direct = pysiglib.sig(path, degree)
        np.testing.assert_allclose(full_sig, direct, rtol=1e-3, atol=1e-10)

    def test_sig_all(self, dimension, degree, batch_shape):
        """sig_all should return cumulative signatures at batch boundaries."""
        path = np.random.randn(*batch_shape, 10, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        stream.push_batch(path)
        all_sigs = stream.sig_all()
        assert all_sigs.shape[0] == 2
        # First entry corresponds to a zero-length path: identity sig is all zeros
        # (scalar_term=False default; no leading 1 to check).
        np.testing.assert_allclose(all_sigs[0], 0.0, atol=1e-15)

    def test_pop_front(self, dimension, degree, batch_shape):
        """After pop_front, earlier indices are invalid but later queries still work."""
        np.random.seed(0)
        path = np.random.randn(*batch_shape, 20, dimension)
        stream = pysiglib.SigStream(dimension, degree)
        for i in range(path.shape[-2]):
            stream.push(path[..., i, :])
        stream.pop_front()
        stream.pop_front()
        assert stream.start_index == 2
        assert stream.size == 18
        sig_5_15 = stream.sig(5, 15)
        direct = pysiglib.sig(path[..., 5:16, :], degree)
        np.testing.assert_allclose(sig_5_15, direct, rtol=1e-3, atol=1e-10)

    def test_push_incremental_matches_batch(self, dimension, degree, batch_shape):
        """Per-point push and push_batch should give same endpoint signature."""
        path = np.random.randn(*batch_shape, 15, dimension)
        stream1 = pysiglib.SigStream(dimension, degree)
        stream1.push_batch(path)
        stream2 = pysiglib.SigStream(dimension, degree)
        for i in range(path.shape[-2]):
            stream2.push(path[..., i, :])
        # Compare full-path signature: batch stores at index 1, per-point at index 14
        np.testing.assert_allclose(stream1.sig(0, 1), stream2.sig(0, 14), rtol=1e-3, atol=1e-10)


# ---- LogSigStream tests (numpy) ----

class TestLogSigStream:
    def test_full_path_matches_log_sig(self, dimension, degree, batch_shape):
        """push_batch endpoint should match pysiglib.log_sig of the full path."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        pysiglib.prepare_log_sig(dimension, degree, method=3)
        path = np.random.randn(*batch_shape, 20, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        stream.push_batch(path)
        stream_ls = stream.sig(0, 1)
        direct_ls = pysiglib.log_sig(path, degree, method=2)
        np.testing.assert_allclose(stream_ls, direct_ls, rtol=1e-3, atol=1e-10)

    def test_interval_via_push(self, dimension, degree, batch_shape):
        """Per-point push allows querying arbitrary sub-intervals."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        pysiglib.prepare_log_sig(dimension, degree, method=3)
        path = np.random.randn(*batch_shape, 30, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        for i in range(path.shape[-2]):
            stream.push(path[..., i, :])
        for start, end in [(5, 15), (0, 10), (10, 29)]:
            stream_ls = stream.sig(start, end)
            direct_ls = pysiglib.log_sig(path[..., start:end + 1, :], degree, method=2)
            np.testing.assert_allclose(stream_ls, direct_ls, rtol=1e-3, atol=1e-10)

    def test_sig_all(self, dimension, degree, batch_shape):
        """sig_all should return cumulative log-signatures at batch boundaries."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        pysiglib.prepare_log_sig(dimension, degree, method=3)
        path = np.random.randn(*batch_shape, 10, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        stream.push_batch(path)
        all_ls = stream.sig_all()
        assert all_ls.shape[0] == 2
        np.testing.assert_allclose(all_ls[0], 0.0, atol=1e-15)

    def test_pop_front(self, dimension, degree, batch_shape):
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        pysiglib.prepare_log_sig(dimension, degree, method=3)
        path = np.random.randn(*batch_shape, 20, dimension)
        stream = pysiglib.LogSigStream(dimension, degree)
        for i in range(path.shape[-2]):
            stream.push(path[..., i, :])
        stream.pop_front()
        stream.pop_front()
        ls_5_15 = stream.sig(5, 15)
        direct = pysiglib.log_sig(path[..., 5:16, :], degree, method=2)
        np.testing.assert_allclose(ls_5_15, direct, rtol=1e-3, atol=1e-10)


# ---- SigWindowStream tests ----

class TestSigWindowStream:
    def test_windows_match_direct(self, dimension, degree, batch_shape):
        """Each emitted window should match direct sig on the sub-path."""
        path = np.random.randn(*batch_shape, 50, dimension)
        window_size = 10
        stride = 5
        ws = pysiglib.SigWindowStream(dimension, degree, window_size=window_size, stride=stride)
        ws.push_batch(path)
        window_sigs = ws.sig()
        path_len = path.shape[-2]
        expected_n = (path_len - window_size) // stride + 1
        assert ws.num_windows == expected_n
        for i in range(ws.num_windows):
            w_start = i * stride
            w_end = w_start + window_size
            direct = pysiglib.sig(path[..., w_start:w_end, :], degree)
            np.testing.assert_allclose(window_sigs[i], direct, rtol=1e-3, atol=1e-10)

    def test_stride_1(self, dimension, degree, batch_shape):
        """stride=1 should produce a window starting at every point."""
        path = np.random.randn(*batch_shape, 15, dimension)
        window_size = 5
        ws = pysiglib.SigWindowStream(dimension, degree, window_size=window_size, stride=1)
        ws.push_batch(path)
        assert ws.num_windows == path.shape[-2] - window_size + 1


# ---- LogSigWindowStream tests ----

class TestLogSigWindowStream:
    def test_windows_match_direct(self, dimension, degree, batch_shape):
        """Each emitted window should match direct log_sig on the sub-path."""
        pysiglib.prepare_log_sig(dimension, degree, method=2)
        pysiglib.prepare_log_sig(dimension, degree, method=3)
        path = np.random.randn(*batch_shape, 50, dimension)
        window_size = 10
        stride = 5
        ws = pysiglib.LogSigWindowStream(dimension, degree, window_size=window_size, stride=stride)
        ws.push_batch(path)
        window_ls = ws.sig()
        path_len = path.shape[-2]
        expected_n = (path_len - window_size) // stride + 1
        assert ws.num_windows == expected_n
        for i in range(ws.num_windows):
            w_start = i * stride
            w_end = w_start + window_size
            direct = pysiglib.log_sig(path[..., w_start:w_end, :], degree, method=3)
            np.testing.assert_allclose(window_ls[i], direct, rtol=1e-8)


# ---- Torch tensor tests (no autograd - base API is forward-only) ----

class TestTorchTensors:
    def test_sig_stream_torch(self, batch_shape):
        """SigStream should work with torch tensors (forward-only)."""
        dim, deg = 3, 3
        path = torch.randn(*batch_shape, 15, dim, dtype=torch.float64)
        stream = pysiglib.SigStream(dim, deg)
        stream.push_batch(path)
        result = stream.sig(0, 1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (*batch_shape, pysiglib.sig_length(dim, deg))

    def test_log_sig_stream_torch(self, batch_shape):
        """LogSigStream should work with torch tensors (forward-only)."""
        dim, deg = 3, 3
        pysiglib.prepare_log_sig(dim, deg, method=2)
        pysiglib.prepare_log_sig(dim, deg, method=3)
        path = torch.randn(*batch_shape, 15, dim, dtype=torch.float64)
        stream = pysiglib.LogSigStream(dim, deg)
        stream.push_batch(path)
        result = stream.sig(0, 1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (*batch_shape, pysiglib.log_sig_length(dim, deg))


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
        pysiglib.prepare_log_sig(dim, deg, method=3)
        path = torch.randn(15, dim, dtype=torch.float64, requires_grad=True)
        stream = LogSigStream(dim, deg)
        for i in range(path.shape[0]):
            stream.push(path[i])
        result = stream.sig(0, 14)
        loss = result.sum()
        loss.backward()
        assert path.grad is not None
        assert not torch.all(path.grad == 0)


# ---- Batched-only behaviour (not covered by the parametrised tests) ----

class TestBatchShapeLocking:
    def test_push_then_mismatched_push(self):
        stream = pysiglib.SigStream(dimension=3, degree=3)
        stream.push(np.random.randn(4, 3))
        assert stream.batch_shape == (4,)
        with pytest.raises(ValueError, match="locked in"):
            stream.push(np.random.randn(5, 3))

    def test_push_batch_then_mismatched_push(self):
        stream = pysiglib.SigStream(dimension=3, degree=3)
        stream.push_batch(np.random.randn(3, 5, 3))
        assert stream.batch_shape == (3,)
        stream.push(np.random.randn(3, 3))
        with pytest.raises(ValueError, match="locked in"):
            stream.push(np.random.randn(4, 3))
        with pytest.raises(ValueError, match="locked in"):
            stream.push_batch(np.random.randn(4, 5, 3))

    def test_window_stream_shape_locking(self):
        ws = pysiglib.SigWindowStream(dimension=3, degree=3, window_size=5, stride=1)
        ws.push_batch(np.random.randn(2, 10, 3))
        assert ws.batch_shape == (2,)
        with pytest.raises(ValueError, match="locked in"):
            ws.push(np.random.randn(3, 3))

    def test_wrong_dimension_rejected(self):
        stream = pysiglib.SigStream(dimension=3, degree=3)
        with pytest.raises(ValueError, match="push expects"):
            stream.push(np.random.randn(4))  # 4 != dimension=3
        with pytest.raises(ValueError, match="push_batch expects"):
            stream.push_batch(np.random.randn(5, 4))  # trailing axis 4 != dimension=3


class TestBatchedIndependence:
    """A batched stream of K paths must produce the same results as K separate
    single-path streams, proving the parallel paths stay independent."""

    def test_sig_stream_batched_matches_single(self):
        np.random.seed(0)
        paths = np.random.randn(4, 20, 3)
        batched = pysiglib.SigStream(3, 3)
        batched.push_batch(paths)
        batched_sig = batched.sig(0, 1)
        assert batched_sig.shape == (4, pysiglib.sig_length(3, 3))
        for k in range(4):
            single_sig = pysiglib.sig(paths[k], 3)
            np.testing.assert_allclose(batched_sig[k], single_sig, rtol=1e-5)

    def test_sig_stream_batched_per_point_matches_single(self):
        np.random.seed(1)
        paths = np.random.randn(4, 15, 3)
        batched = pysiglib.SigStream(3, 3)
        for i in range(paths.shape[-2]):
            batched.push(paths[..., i, :])
        batched_sig = batched.sig(0, 14)
        for k in range(4):
            single_sig = pysiglib.sig(paths[k], 3)
            np.testing.assert_allclose(batched_sig[k], single_sig, rtol=1e-5)

    def test_log_sig_stream_batched_matches_single(self):
        pysiglib.prepare_log_sig(3, 3, method=2)
        pysiglib.prepare_log_sig(3, 3, method=3)
        np.random.seed(2)
        paths = np.random.randn(4, 15, 3)
        batched = pysiglib.LogSigStream(3, 3)
        batched.push_batch(paths)
        batched_ls = batched.sig(0, 1)
        for k in range(4):
            single_ls = pysiglib.log_sig(paths[k], 3, method=2)
            np.testing.assert_allclose(batched_ls[k], single_ls, rtol=1e-5)


class TestBatchedWindowStream:
    def test_sig_window_stream_batched_matches_single(self):
        np.random.seed(3)
        paths = np.random.randn(4, 50, 3)
        ws = pysiglib.SigWindowStream(3, 3, window_size=10, stride=5)
        ws.push_batch(paths)
        window_sigs = ws.sig()  # (num_windows, 4, sig_len)
        num_windows = (50 - 10) // 5 + 1
        assert window_sigs.shape[0] == num_windows
        assert window_sigs.shape[1] == 4
        for i in range(num_windows):
            for k in range(4):
                direct = pysiglib.sig(paths[k, i * 5:i * 5 + 10, :], 3)
                np.testing.assert_allclose(window_sigs[i, k], direct, rtol=1e-3, atol=1e-10)


class TestEmptyPushBatch:
    def test_empty_push_batch_does_not_lock_shape(self):
        stream = pysiglib.SigStream(3, 3)
        stream.push_batch(np.zeros((4, 0, 3)))
        assert stream.batch_shape is None
        # Next push can then lock a different batch shape
        stream.push_batch(np.random.randn(2, 5, 3))
        assert stream.batch_shape == (2,)

    def test_empty_push_batch_noop_single(self):
        stream = pysiglib.SigStream(3, 3)
        stream.push_batch(np.zeros((0, 3)))
        assert stream.batch_shape is None
        assert stream.size == 0


class TestSinglePathBackwardCompat:
    """Sanity check that the single-path (batch_shape == ()) mode still produces
    1-D outputs, not something accidentally promoted to 2-D."""

    def test_sig_stream_single_path_1d_outputs(self):
        sig_len = pysiglib.sig_length(3, 3)
        stream = pysiglib.SigStream(3, 3)
        stream.push_batch(np.random.randn(10, 3))
        assert stream.sig(0, 1).shape == (sig_len,)
        assert stream.sig_all().shape == (2, sig_len)
        assert stream.batch_shape == ()

    def test_log_sig_stream_single_path_1d_outputs(self):
        pysiglib.prepare_log_sig(3, 3, method=2)
        pysiglib.prepare_log_sig(3, 3, method=3)
        ls_len = pysiglib.log_sig_length(3, 3)
        stream = pysiglib.LogSigStream(3, 3)
        stream.push_batch(np.random.randn(10, 3))
        assert stream.sig(0, 1).shape == (ls_len,)
        assert stream.sig_all().shape == (2, ls_len)


class TestBatchedGradients:
    """Per-batch gradients must not leak into other batch entries."""

    def test_sig_stream_grad_isolated_per_batch(self):
        from pysiglib.torch_api import SigStream
        torch.manual_seed(0)
        paths = torch.randn(4, 10, 3, dtype=torch.float64, requires_grad=True)
        stream = SigStream(3, 3)
        for i in range(paths.shape[-2]):
            stream.push(paths[..., i, :])
        result = stream.sig(0, 9)
        loss = result[0].sum()
        loss.backward()
        assert paths.grad is not None
        assert not torch.all(paths.grad[0] == 0)
        for k in range(1, 4):
            assert torch.all(paths.grad[k] == 0), f"gradient leaked to batch element {k}"

    def test_sig_stream_batched_push_batch_grad(self):
        from pysiglib.torch_api import SigStream
        torch.manual_seed(1)
        paths = torch.randn(3, 12, 3, dtype=torch.float64, requires_grad=True)
        stream = SigStream(3, 3)
        stream.push_batch(paths)
        result = stream.sig(0, 1)
        loss = result.sum()
        loss.backward()
        assert paths.grad is not None
        # Every batch element was summed into the loss, so every batch element's grad is non-zero.
        for k in range(3):
            assert not torch.all(paths.grad[k] == 0)


# ---- scalar_term flag tests ----

class TestScalarTermFlag:
    def test_sig_stream_scalar_term_false_push_batch(self):
        dim, deg = 3, 3
        path = np.random.randn(20, dim)
        stream_true = pysiglib.SigStream(dim, deg, scalar_term=True)
        stream_false = pysiglib.SigStream(dim, deg, scalar_term=False)
        stream_true.push_batch(path)
        stream_false.push_batch(path)
        full_true = stream_true.sig(0, 1)
        full_false = stream_false.sig(0, 1)
        assert full_true.shape[-1] == full_false.shape[-1] + 1
        np.testing.assert_allclose(full_true[..., 1:], full_false, rtol=1e-10)
        direct = pysiglib.sig(path, deg, scalar_term=False)
        np.testing.assert_allclose(full_false, direct, rtol=1e-3, atol=1e-10)

    def test_sig_stream_scalar_term_false_push(self):
        dim, deg = 3, 3
        path = np.random.randn(15, dim)
        stream = pysiglib.SigStream(dim, deg, scalar_term=False)
        for i in range(len(path)):
            stream.push(path[i])
        result = stream.sig(0, len(path) - 1)
        direct = pysiglib.sig(path, deg, scalar_term=False)
        assert result.shape == direct.shape
        np.testing.assert_allclose(result, direct, rtol=1e-3, atol=1e-10)

    def test_sig_stream_scalar_term_false_interval(self):
        dim, deg = 3, 3
        path = np.random.randn(30, dim)
        stream = pysiglib.SigStream(dim, deg, scalar_term=False)
        for i in range(len(path)):
            stream.push(path[i])
        result = stream.sig(5, 15)
        direct = pysiglib.sig(path[5:16], deg, scalar_term=False)
        np.testing.assert_allclose(result, direct, rtol=1e-3, atol=1e-10)

    def test_sig_window_stream_scalar_term_false(self):
        dim, deg = 3, 3
        path = np.random.randn(50, dim)
        ws = pysiglib.SigWindowStream(dim, deg, window_size=10, stride=5,
                                      scalar_term=False)
        ws.push_batch(path)
        windows = ws.sig()
        for i in range(ws.num_windows):
            direct = pysiglib.sig(path[i * 5:i * 5 + 10], deg, scalar_term=False)
            np.testing.assert_allclose(windows[i], direct, rtol=1e-3, atol=1e-10)


# ---- n_jobs flag tests ----

class TestNJobsFlag:
    """n_jobs is a performance knob; correctness is identical across values.
    Verify the flag is accepted and produces matching numerical results."""

    def test_sig_stream_n_jobs_matches_serial(self):
        path = np.random.randn(30, 3)
        stream_serial = pysiglib.SigStream(3, 3, n_jobs=1)
        stream_parallel = pysiglib.SigStream(3, 3, n_jobs=-1)
        stream_serial.push_batch(path)
        stream_parallel.push_batch(path)
        np.testing.assert_allclose(stream_serial.sig(0, 1),
                                   stream_parallel.sig(0, 1),
                                   rtol=1e-10, atol=1e-12)

    def test_log_sig_stream_n_jobs_accepted(self):
        pysiglib.prepare_log_sig(3, 3, method=2)
        pysiglib.prepare_log_sig(3, 3, method=3)
        path = np.random.randn(20, 3)
        stream = pysiglib.LogSigStream(3, 3, n_jobs=-1)
        stream.push_batch(path)
        result = stream.sig(0, 1)
        direct = pysiglib.log_sig(path, 3, method=2)
        np.testing.assert_allclose(result, direct, rtol=1e-3, atol=1e-10)

    def test_sig_window_stream_n_jobs_accepted(self):
        path = np.random.randn(30, 3)
        ws = pysiglib.SigWindowStream(3, 3, window_size=10, stride=5, n_jobs=-1)
        ws.push_batch(path)
        assert ws.num_windows == 5

    def test_log_sig_window_stream_n_jobs_accepted(self):
        pysiglib.prepare_log_sig(3, 3, method=2)
        pysiglib.prepare_log_sig(3, 3, method=3)
        path = np.random.randn(30, 3)
        ws = pysiglib.LogSigWindowStream(3, 3, window_size=10, stride=5, n_jobs=-1)
        ws.push_batch(path)
        assert ws.num_windows == 5

    @pytest.mark.parametrize("factory", [
        lambda: pysiglib.SigStream(3, 3, n_jobs=0),
        lambda: pysiglib.LogSigStream(3, 3, n_jobs=0),
        lambda: pysiglib.SigWindowStream(3, 3, window_size=5, n_jobs=0),
        lambda: pysiglib.LogSigWindowStream(3, 3, window_size=5, n_jobs=0),
    ], ids=["SigStream", "LogSigStream", "SigWindowStream", "LogSigWindowStream"])
    def test_n_jobs_zero_rejected(self, factory):
        with pytest.raises(ValueError, match="n_jobs cannot be 0"):
            factory()
