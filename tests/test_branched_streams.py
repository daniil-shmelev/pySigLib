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
import pysiglib.torch_api as torch_api
from conftest import DEVICES


@pytest.fixture(params=[False, True])
def planar(request):
    pysiglib.prepare_branched_sig(2, 3, planar=request.param)
    return request.param


@pytest.fixture(params=[False, True])
def scalar_term(request):
    return request.param


@pytest.fixture(params=[(), (2, 3), (0, 2)], ids=["single", "batch", "empty_batch"])
def batch_shape(request):
    return request.param


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_branched_stream_checkpoints(planar, scalar_term, batch_shape, dtype):
    rng = np.random.default_rng(42)
    path = rng.normal(0, 0.2, (*batch_shape, 12, 4)).astype(dtype)[..., ::2]
    stream = pysiglib.BranchedSigStream(2, 3, planar=planar, scalar_term=scalar_term)
    stream.push(path[..., 0, :])
    stream.push_batch(path[..., 1:5, :])
    stream.push(path[..., 5, :])
    stream.push_batch(path[..., 6:, :])
    checkpoints = [0, 4, 5, 11]
    assert stream.size == 4
    assert stream.start_index == 0
    assert stream.end_index == 3
    assert stream.batch_shape == batch_shape
    all_sigs = stream.sig_all()
    assert all_sigs.dtype == dtype
    assert all_sigs.shape == (4, *batch_shape, pysiglib.branched_sig_length(
        2, 3, planar=planar, scalar_term=scalar_term))
    for start, end in [(0, 3), (1, 2), (1, 3), (2, 2)]:
        expected = pysiglib.branched_sig(
            path[..., checkpoints[start]:checkpoints[end] + 1, :], 3,
            planar=planar, scalar_term=scalar_term)
        np.testing.assert_allclose(stream.sig(start, end), expected, rtol=1e-4, atol=2e-6)
    for i, end in enumerate(checkpoints):
        expected = pysiglib.branched_sig(
            path[..., :end + 1, :], 3, planar=planar, scalar_term=scalar_term)
        np.testing.assert_allclose(all_sigs[i], expected, rtol=1e-4, atol=2e-6)
    intervals = [(0, 1), (1, 3)]
    np.testing.assert_allclose(stream.sig_batch(intervals), np.stack([
        stream.sig(start, end) for start, end in intervals]), rtol=1e-4, atol=2e-6)
    expected = stream.sig(1, 3)
    stream.pop_front()
    assert stream.start_index == 1
    assert stream.end_index == 3
    np.testing.assert_allclose(stream.sig(1, 3), expected, rtol=1e-4, atol=2e-6)
    with pytest.raises(IndexError):
        stream.sig(0, 3)


def test_branched_stream_push_matches_batch(planar, scalar_term):
    path = np.random.default_rng(12).normal(0, 0.2, (9, 2))
    stream = pysiglib.BranchedSigStream(2, 3, planar=planar, scalar_term=scalar_term)
    batched = pysiglib.BranchedSigStream(2, 3, planar=planar, scalar_term=scalar_term, n_jobs=-1)
    for point in path:
        stream.push(point)
    batched.push_batch(path)
    assert batched.size == 2
    np.testing.assert_allclose(stream.sig(0, 8), batched.sig(0, 1), atol=1e-12)
    result = batched.sig(0, 1)
    result[...] = 100
    np.testing.assert_allclose(batched.sig(0, 1), stream.sig(0, 8), atol=1e-12)


@pytest.mark.parametrize("window_size,stride", [(1, 1), (4, 1), (4, 4), (3, 6)])
@pytest.mark.parametrize("mode", ["push", "batch", "mixed"])
def test_branched_windows(planar, scalar_term, batch_shape, window_size, stride, mode):
    path = np.random.default_rng(8).normal(0, 0.2, (*batch_shape, 17, 4))[..., ::2]
    stream = pysiglib.BranchedSigWindowStream(
        2, 3, window_size, stride=stride, planar=planar, scalar_term=scalar_term, n_jobs=-1)
    with pytest.raises(ValueError, match="No complete windows"):
        stream.sig()
    if mode == "push":
        for i in range(path.shape[-2]):
            stream.push(path[..., i, :])
    elif mode == "batch":
        stream.push_batch(path)
    else:
        stream.push(path[..., 0, :])
        stream.push_batch(path[..., 1:5, :])
        stream.push(path[..., 5, :])
        stream.push_batch(path[..., 6:8, :])
        stream.push_batch(path[..., 8:, :])
    expected = np.stack([
        pysiglib.branched_sig(path[..., start:start + window_size, :], 3,
                             planar=planar, scalar_term=scalar_term)
        for start in range(0, 18 - window_size, stride)
    ])
    assert stream.num_windows == expected.shape[0]
    assert stream.batch_shape == batch_shape
    np.testing.assert_allclose(stream.sig(), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("windowed", [False, True])
def test_branched_stream_validation(windowed):
    factory = pysiglib.BranchedSigWindowStream if windowed else pysiglib.BranchedSigStream
    kwargs = {"window_size": 4} if windowed else {}
    for invalid in [{"n_jobs": 0}, {"dimension": -1}, {"degree": -1}, {"planar": 1}]:
        with pytest.raises((TypeError, ValueError)):
            factory(**({"dimension": 2, "degree": 3, **kwargs, **invalid}))
    stream = factory(2, 3, **kwargs)
    stream.push_batch(np.zeros((4, 0, 2)))
    assert stream.batch_shape is None
    with pytest.raises(ValueError, match="push expects"):
        stream.push(np.zeros(3))
    with pytest.raises(ValueError, match="push_batch expects"):
        stream.push_batch(np.zeros((3, 3)))
    stream.push_batch(np.zeros((2, 1, 2)))
    assert stream.batch_shape == (2,)
    with pytest.raises(ValueError, match="locked in"):
        stream.push(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="locked in"):
        stream.push_batch(np.zeros((3, 2, 2)))
    if not windowed:
        assert stream.size == 1
        with pytest.raises(ValueError, match="Cannot pop_front"):
            stream.pop_front()


@pytest.mark.parametrize("kwargs", [
    {"window_size": 0}, {"window_size": 1.5},
    {"window_size": 3, "stride": 0}, {"window_size": 3, "stride": 1.5},
])
def test_branched_window_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        pysiglib.BranchedSigWindowStream(2, 3, **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["push", "batch", "window"])
def test_branched_stream_torch_gradients(planar, scalar_term, device, dtype, mode):
    torch.manual_seed(42)
    path = (0.2 * torch.randn(2, 3, 9, 2, device=device, dtype=dtype)).requires_grad_()
    if mode == "window":
        stream = torch_api.BranchedSigWindowStream(
            2, 3, 4, stride=2, planar=planar, scalar_term=scalar_term)
        stream.push(path[..., 0, :])
        stream.push_batch(path[..., 1:5, :])
        for i in range(5, 9):
            stream.push(path[..., i, :])
        result = stream.sig()[1]
        interval = slice(2, 6)
    else:
        stream = torch_api.BranchedSigStream(2, 3, planar=planar, scalar_term=scalar_term)
        if mode == "push":
            for i in range(9):
                stream.push(path[..., i, :])
            result = stream.sig(2, 7)
            interval = slice(2, 8)
        else:
            stream.push_batch(path[..., :3, :])
            stream.push_batch(path[..., 3:8, :])
            stream.push(path[..., 8, :])
            result = stream.sig(1, 2)
            interval = slice(2, 8)
    direct = torch_api.branched_sig(path[..., interval, :], 3, planar=planar, scalar_term=scalar_term)
    assert result.dtype == dtype
    assert result.device == path.device
    torch.testing.assert_close(result, direct, rtol=1e-4, atol=2e-6)
    weights = torch.linspace(0.1, 1, result.shape[-1], dtype=dtype, device=device)
    actual_grad, = torch.autograd.grad((result[0, 1] * weights).sum(), path)
    expected_grad, = torch.autograd.grad((direct[0, 1] * weights).sum(), path)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=2e-6)
    assert torch.count_nonzero(actual_grad[1]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("windowed", [False, True])
def test_branched_stream_base_torch(planar, scalar_term, device, dtype, windowed):
    path = torch.zeros(0, 2, 5, 2, dtype=dtype, device=device)
    if windowed:
        stream = pysiglib.BranchedSigWindowStream(2, 3, 5, planar=planar, scalar_term=scalar_term)
    else:
        stream = pysiglib.BranchedSigStream(2, 3, planar=planar, scalar_term=scalar_term)
    stream.push_batch(path)
    result = stream.sig()[0] if windowed else stream.sig(0, 1)
    assert result.dtype == dtype
    assert result.device == path.device
    assert result.shape == (0, 2, pysiglib.branched_sig_length(2, 3, planar=planar, scalar_term=scalar_term))


@pytest.mark.skipif(not pysiglib.BUILT_WITH_JAX_FFI, reason="JAX FFI not built")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["push", "batch", "window"])
def test_branched_stream_jax_gradients(planar, scalar_term, dtype, mode):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax_api = pytest.importorskip("pysiglib.jax_api")
    jax.config.update("jax_enable_x64", True)
    path = jnp.asarray(np.random.default_rng(42).normal(0, 0.2, (2, 3, 8, 2)).astype(dtype))

    def streamed(x):
        if mode == "window":
            stream = jax_api.BranchedSigWindowStream(
                2, 3, 4, stride=2, planar=planar, scalar_term=scalar_term)
            stream.push_batch(x[..., :2, :])
            for i in range(2, 8):
                stream.push(x[..., i, :])
            return stream.sig()[1]
        stream = jax_api.BranchedSigStream(2, 3, planar=planar, scalar_term=scalar_term)
        if mode == "push":
            for i in range(8):
                stream.push(x[..., i, :])
            return stream.sig(2, 5)
        stream.push_batch(x[..., :3, :])
        stream.push_batch(x[..., 3:6, :])
        return stream.sig(1, 2)

    def direct(x):
        return jax_api.branched_sig(x[..., 2:6, :], 3, planar=planar, scalar_term=scalar_term)

    actual = jax.jit(streamed)(path)
    assert isinstance(actual, jax.Array)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, direct(path), rtol=1e-4, atol=2e-6)
    weights = jnp.linspace(0.1, 1, actual.shape[-1], dtype=path.dtype)
    actual_grad = jax.jit(jax.grad(lambda x: jnp.sum(streamed(x)[0, 1] * weights)))(path)
    expected_grad = jax.grad(lambda x: jnp.sum(direct(x)[0, 1] * weights))(path)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-4, atol=2e-6)
    np.testing.assert_array_equal(actual_grad[1], 0)
