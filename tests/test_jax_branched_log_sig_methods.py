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

import pysiglib
from pysiglib.branched_log_sig_backprop import (
    _branched_log_sig_from_path_backprop,
)


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
jax_api = pytest.importorskip("pysiglib.jax_api")


pytestmark = pytest.mark.skipif(
    not pysiglib.BUILT_WITH_JAX_FFI,
    reason="JAX FFI not built",
)


def _planar_bsig(dimension, degree, *, batch=False, scalar_term=False):
    rng = np.random.default_rng(20260815)
    shape = (3, 7, dimension) if batch else (7, dimension)
    path = np.cumsum(rng.normal(scale=0.2, size=shape), axis=-2)
    return pysiglib.branched_sig(
        path, degree, planar=True, scalar_term=scalar_term)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("scalar_term", [False, True])
@pytest.mark.parametrize("jitted", [False, True])
def test_jax_compressed_branched_log_sig_matches_base(
        method, scalar_term, jitted):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu", use_disk=False)
    bsig = _planar_bsig(
        dimension, degree, batch=True, scalar_term=scalar_term)
    expected = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=method)

    def convert(value):
        return jax_api.branched_sig_to_log_sig(
            value, dimension, degree, planar=True, method=method)

    fn = jax.jit(convert) if jitted else convert
    actual = fn(jnp.asarray(bsig, dtype=jnp.float64))

    assert actual.shape == expected.shape
    assert actual.shape[-1] == pysiglib.branched_log_sig_length(
        dimension, degree, planar=True)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("scalar_term", [False, True])
@pytest.mark.parametrize("jitted", [False, True])
def test_jax_compressed_branched_log_sig_grad_matches_base(
        method, scalar_term, jitted):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu", use_disk=False)
    bsig = _planar_bsig(
        dimension, degree, batch=True, scalar_term=scalar_term)
    out = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=method)
    weights = np.random.default_rng(12345).normal(size=out.shape)
    expected = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, weights, dimension, degree, planar=True, method=method)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64)

    def loss(value):
        result = jax_api.branched_sig_to_log_sig(
            value, dimension, degree, planar=True, method=method)
        return jnp.sum(result * weights_jax)

    grad_fn = jax.jit(jax.grad(loss)) if jitted else jax.grad(loss)
    actual = grad_fn(jnp.asarray(bsig, dtype=jnp.float64))

    assert actual.shape == bsig.shape
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("method", [1, 2])
def test_jax_direct_compressed_branched_log_sig_is_scalar_free(method):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu", use_disk=False)
    path = np.array(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]], dtype=np.float64)

    without_scalar = jax_api.branched_log_sig(
        jnp.asarray(path), degree, planar=True, method=method,
        scalar_term=False)
    with_scalar = jax_api.branched_log_sig(
        jnp.asarray(path), degree, planar=True, method=method,
        scalar_term=True)

    expected_length = jax_api.branched_log_sig_length(
        dimension, degree, planar=True)
    assert without_scalar.shape == (expected_length,)
    assert with_scalar.shape == (expected_length,)
    np.testing.assert_allclose(
        np.asarray(without_scalar), np.asarray(with_scalar), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("jitted", [False, True])
def test_jax_direct_compressed_branched_log_sig_grad_matches_base(
        method, jitted):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu", use_disk=False)
    path = np.array(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]], dtype=np.float64)
    bsig = pysiglib.branched_sig(
        path, degree, planar=True, scalar_term=True)
    blog_sig = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=method)
    weights = np.random.default_rng(6789).normal(size=blog_sig.shape)
    bsig_grad = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, weights, dimension, degree, planar=True, method=method)
    expected = pysiglib.branched_sig_backprop(
        path, bsig, bsig_grad, degree, planar=True)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64)

    def loss(value):
        result = jax_api.branched_log_sig(
            value, degree, planar=True, method=method, scalar_term=True)
        return jnp.sum(result * weights_jax)

    grad_fn = jax.jit(jax.grad(loss)) if jitted else jax.grad(loss)
    actual = grad_fn(jnp.asarray(path, dtype=jnp.float64))

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-9, rtol=1e-9)


def test_jax_planar_default_is_method_1():
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 1, planar=True, device="cpu", use_disk=False)
    bsig = _planar_bsig(dimension, degree)
    bsig_jax = jnp.asarray(bsig, dtype=jnp.float64)

    default = jax_api.branched_sig_to_log_sig(
        bsig_jax, dimension, degree, planar=True)
    explicit = jax_api.branched_sig_to_log_sig(
        bsig_jax, dimension, degree, planar=True, method=1)

    np.testing.assert_allclose(
        np.asarray(default), np.asarray(explicit), atol=1e-12, rtol=1e-12)


def test_jax_branched_log_sig_method_validation():
    dimension, degree = 1, 1
    bsig = jnp.ones(
        pysiglib.branched_sig_length(
            dimension, degree, planar=True, scalar_term=True),
        dtype=jnp.float64,
    )

    with pytest.raises(ValueError, match="not supported"):
        jax_api.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=3)
    with pytest.raises(ValueError, match="require planar=True"):
        jax_api.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=False, method=1)
    with pytest.raises(ValueError, match="method must be"):
        jax_api.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=-1)


def test_jax_compressed_branched_log_sig_rejects_gpu_array():
    gpu_devices = [
        device for device in jax.devices()
        if device.platform in ("cuda", "gpu")
    ]
    if not gpu_devices:
        pytest.skip("JAX GPU device unavailable")

    dimension, degree = 1, 1
    bsig = jax.device_put(
        jnp.ones(
            pysiglib.branched_sig_length(
                dimension, degree, planar=True, scalar_term=True),
            dtype=jnp.float64,
        ),
        gpu_devices[0],
    )
    with pytest.raises(NotImplementedError, match="only implemented on CPU"):
        jax_api.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=1)


@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_jax_method_three_matches_method_two(jitted, batch):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, device="cpu", use_disk=False)
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cpu", use_disk=False)
    rng = np.random.default_rng(9124)
    shape = (3, 5, dimension) if batch else (5, dimension)
    path = np.cumsum(rng.normal(scale=0.2, size=shape), axis=-2)
    expected = pysiglib.branched_log_sig(
        path, degree, planar=True, method=2)

    def compute(value):
        return jax_api.branched_log_sig(
            value, degree, planar=True, method=3, scalar_term=True)

    function = jax.jit(compute) if jitted else compute
    actual = function(jnp.asarray(path, dtype=jnp.float64))
    np.testing.assert_allclose(
        np.asarray(actual), expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("jitted", [False, True])
def test_jax_method_three_gradient_matches_explicit_backward(jitted):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cpu", use_disk=False)
    path = np.array(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]], dtype=np.float64)
    weights = np.random.default_rng(31415).normal(
        size=pysiglib.branched_log_sig_length(
            dimension, degree, planar=True))
    expected = _branched_log_sig_from_path_backprop(
        weights, path, degree)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64)

    def loss(value):
        result = jax_api.branched_log_sig(
            value, degree, planar=True, method=3)
        return jnp.sum(result * weights_jax)

    grad_function = jax.jit(jax.grad(loss)) if jitted else jax.grad(loss)
    actual = grad_function(jnp.asarray(path, dtype=jnp.float64))
    np.testing.assert_allclose(
        np.asarray(actual), expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize(
    "time_aug,lead_lag", [(True, False), (False, True), (True, True)])
def test_jax_method_three_augmentation_matches_method_two(
        time_aug, lead_lag):
    dimension, degree = 2, 2
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, time_aug=time_aug,
        lead_lag=lead_lag, device="cpu", use_disk=False)
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, time_aug=time_aug,
        lead_lag=lead_lag, device="cpu", use_disk=False)
    path = np.array(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]], dtype=np.float64)
    expected = pysiglib.branched_log_sig(
        path, degree, planar=True, method=2,
        time_aug=time_aug, lead_lag=lead_lag)
    actual = jax.jit(lambda value: jax_api.branched_log_sig(
        value, degree, planar=True, method=3,
        time_aug=time_aug, lead_lag=lead_lag))(
            jnp.asarray(path, dtype=jnp.float64))
    np.testing.assert_allclose(
        np.asarray(actual), expected, atol=2e-10, rtol=2e-10)


def test_jax_method_three_rejects_correction():
    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]], dtype=jnp.float64)
    correction = jnp.zeros((path.shape[0] - 1, path.shape[-1] ** 2))
    with pytest.raises(ValueError, match="correction is not supported"):
        jax_api.branched_log_sig(
            path, 2, planar=True, method=3, correction=correction)
