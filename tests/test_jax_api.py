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

from conftest import check_close

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
jax_api = pytest.importorskip("pysiglib.jax_api")


pytestmark = pytest.mark.skipif(
    not pysiglib.BUILT_WITH_JAX_FFI,
    reason="JAX FFI not built",
)


def _jax_devices():
    platforms = {d.platform for d in jax.devices()}
    devices = []
    if "cpu" in platforms:
        devices.append("cpu")
    if pysiglib.BUILT_WITH_CUDA and platforms & {"gpu", "cuda"}:
        devices.append("cuda")
    return devices


def _jax_device(platform: str):
    wanted = {"cpu": {"cpu"}, "cuda": {"gpu", "cuda"}}[platform]
    for device in jax.devices():
        if device.platform in wanted:
            return device
    raise RuntimeError(f"No JAX device found for platform {platform}")


def _as_jax_array(x, device: str, dtype):
    arr = jnp.asarray(x, dtype=dtype)
    return jax.device_put(arr, _jax_device(device))


FORWARD_CASES = [
    {"shape": (10, 3), "degree": 1, "time_aug": False, "lead_lag": False},
    {"shape": (12, 2), "degree": 3, "time_aug": True, "lead_lag": False},
    {"shape": (9, 2), "degree": 3, "time_aug": False, "lead_lag": True},
    {"shape": (8, 2), "degree": 2, "time_aug": True, "lead_lag": True},
    {"shape": (4, 10, 3), "degree": 2, "time_aug": False, "lead_lag": False},
]

BACKWARD_CASES = [
    {"shape": (12, 3), "degree": 1, "time_aug": False, "lead_lag": False},
    {"shape": (11, 2), "degree": 3, "time_aug": True, "lead_lag": False},
    {"shape": (10, 2), "degree": 2, "time_aug": False, "lead_lag": True},
    {"shape": (9, 2), "degree": 2, "time_aug": True, "lead_lag": True},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", FORWARD_CASES)
def test_jax_sig_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    expected = pysiglib.sig(
        x,
        case["degree"],
        time_aug=case["time_aug"],
        lead_lag=case["lead_lag"],
    )

    x_jax = _as_jax_array(x, device, dtype)

    def fn(path):
        return jax_api.sig(
            path,
            case["degree"],
            time_aug=case["time_aug"],
            lead_lag=case["lead_lag"],
        )

    actual = jax.jit(fn)(x_jax) if jitted else fn(x_jax)
    check_close(expected, actual, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BACKWARD_CASES)
def test_jax_sig_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    sig_ref = pysiglib.sig(
        x,
        case["degree"],
        time_aug=case["time_aug"],
        lead_lag=case["lead_lag"],
    )
    weights = rng.uniform(size=sig_ref.shape).astype(dtype)
    grad_ref = pysiglib.sig_backprop(
        x,
        sig_ref,
        weights,
        case["degree"],
        time_aug=case["time_aug"],
        lead_lag=case["lead_lag"],
    )

    x_jax = _as_jax_array(x, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(path):
        sig = jax_api.sig(
            path,
            case["degree"],
            time_aug=case["time_aug"],
            lead_lag=case["lead_lag"],
        )
        return jnp.sum(sig * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad = grad_fn(x_jax)
    check_close(grad_ref, grad, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_sig_correction_matches_pysiglib(device, jitted, dtype):
    x = np.array([[0.0], [3.0]], dtype=dtype)
    correction = np.array([[2.0]], dtype=dtype)
    expected = pysiglib.sig(x, 4, scalar_term=True, correction=correction)

    x_jax = _as_jax_array(x, device, dtype)
    correction_jax = _as_jax_array(correction, device, dtype)

    def fn(path, prim):
        return jax_api.sig(path, 4, scalar_term=True, correction=prim)

    actual = jax.jit(fn)(x_jax, correction_jax) if jitted else fn(x_jax, correction_jax)
    check_close(expected, actual, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_sig_correction_grad_matches_pysiglib(device, jitted, dtype):
    x = np.array([[0.1, -0.2], [0.4, 0.3], [0.0, 0.7]], dtype=dtype)
    correction = np.array([[0.2, -0.1, 0.05, 0.3]] * (x.shape[0] - 1), dtype=dtype)
    sig_ref = pysiglib.sig(x, 3, scalar_term=True, correction=correction)
    weights = np.linspace(-0.5, 0.7, sig_ref.shape[-1]).astype(dtype)
    grad_ref = pysiglib.sig_backprop(x, sig_ref, weights, 3, correction=correction)

    x_jax = _as_jax_array(x, device, dtype)
    correction_jax = _as_jax_array(correction, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(path):
        sig = jax_api.sig(path, 3, scalar_term=True, correction=correction_jax)
        return jnp.sum(sig * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad = grad_fn(x_jax)
    check_close(grad_ref, grad, double_atol=1e-8)


def test_jax_sig_accepts_rank1_correction():
    path = np.array([[0.0], [3.0]], dtype=np.float64)
    correction = np.array([2.0], dtype=np.float64)

    expected = pysiglib.sig(path, 4, scalar_term=True, correction=correction)
    actual = jax_api.sig(jnp.asarray(path), 4, scalar_term=True, correction=jnp.asarray(correction))

    check_close(expected, actual, double_atol=1e-8)


def test_jax_sig_rejects_bad_correction_shape():
    path = jnp.zeros((2, 2), dtype=jnp.float64)
    correction = jnp.zeros((1, 1, 4), dtype=jnp.float64)

    with pytest.raises(ValueError, match="correction shape"):
        jax_api.sig(path, 2, correction=correction)


# =========================================================================
# sig_combine - forward
# =========================================================================

SIG_COMBINE_CASES = [
    {"dim": 3, "degree": 2, "time_aug": False, "lead_lag": False},
    {"dim": 2, "degree": 3, "time_aug": True, "lead_lag": False},
    {"dim": 2, "degree": 2, "time_aug": False, "lead_lag": True},
    {"dim": 2, "degree": 2, "time_aug": True, "lead_lag": True},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_COMBINE_CASES)
def test_jax_sig_combine_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]

    x1 = rng.uniform(size=(10, dim)).astype(dtype)
    x2 = rng.uniform(size=(8, dim)).astype(dtype)

    sig1 = pysiglib.sig(x1, deg, time_aug=ta, lead_lag=ll)
    sig2 = pysiglib.sig(x2, deg, time_aug=ta, lead_lag=ll)
    expected = pysiglib.sig_combine(sig1, sig2, dim, deg, time_aug=ta, lead_lag=ll)

    sig1_jax = _as_jax_array(sig1, device, dtype)
    sig2_jax = _as_jax_array(sig2, device, dtype)

    def fn(s1, s2):
        return jax_api.sig_combine(s1, s2, dim, deg, time_aug=ta, lead_lag=ll)

    actual = jax.jit(fn)(sig1_jax, sig2_jax) if jitted else fn(sig1_jax, sig2_jax)
    check_close(expected, actual, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_COMBINE_CASES)
def test_jax_sig_combine_batch_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]
    batch = 4

    x1 = rng.uniform(size=(batch, 10, dim)).astype(dtype)
    x2 = rng.uniform(size=(batch, 8, dim)).astype(dtype)

    sig1 = pysiglib.sig(x1, deg, time_aug=ta, lead_lag=ll)
    sig2 = pysiglib.sig(x2, deg, time_aug=ta, lead_lag=ll)
    expected = pysiglib.sig_combine(sig1, sig2, dim, deg, time_aug=ta, lead_lag=ll)

    sig1_jax = _as_jax_array(sig1, device, dtype)
    sig2_jax = _as_jax_array(sig2, device, dtype)

    def fn(s1, s2):
        return jax_api.sig_combine(s1, s2, dim, deg, time_aug=ta, lead_lag=ll)

    actual = jax.jit(fn)(sig1_jax, sig2_jax) if jitted else fn(sig1_jax, sig2_jax)
    check_close(expected, actual, double_atol=1e-8)


# =========================================================================
# sig_combine - backward
# =========================================================================

SIG_COMBINE_BACKWARD_CASES = [
    {"dim": 3, "degree": 2, "time_aug": False, "lead_lag": False},
    {"dim": 2, "degree": 3, "time_aug": True, "lead_lag": False},
    {"dim": 2, "degree": 2, "time_aug": False, "lead_lag": True},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_COMBINE_BACKWARD_CASES)
def test_jax_sig_combine_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    dim, deg = case["dim"], case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]

    x1 = rng.uniform(size=(10, dim)).astype(dtype)
    x2 = rng.uniform(size=(8, dim)).astype(dtype)

    sig1_np = pysiglib.sig(x1, deg, time_aug=ta, lead_lag=ll)
    sig2_np = pysiglib.sig(x2, deg, time_aug=ta, lead_lag=ll)
    combined = pysiglib.sig_combine(sig1_np, sig2_np, dim, deg, time_aug=ta, lead_lag=ll)

    weights = rng.uniform(size=combined.shape).astype(dtype)
    grad_sig1_ref, grad_sig2_ref = pysiglib.sig_combine_backprop(
        weights, sig1_np, sig2_np, dim, deg, time_aug=ta, lead_lag=ll
    )

    sig1_jax = _as_jax_array(sig1_np, device, dtype)
    sig2_jax = _as_jax_array(sig2_np, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(s1, s2):
        comb = jax_api.sig_combine(s1, s2, dim, deg, time_aug=ta, lead_lag=ll)
        return jnp.sum(comb * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1))) if jitted else jax.grad(loss_fn, argnums=(0, 1))
    grad_sig1, grad_sig2 = grad_fn(sig1_jax, sig2_jax)

    check_close(grad_sig1_ref, grad_sig1, double_atol=1e-8)
    check_close(grad_sig2_ref, grad_sig2, double_atol=1e-8)


# =========================================================================
# transform_path - forward
# =========================================================================

TRANSFORM_PATH_CASES = [
    {"shape": (10, 3), "time_aug": True, "lead_lag": False, "end_time": 1.0},
    {"shape": (9, 2), "time_aug": False, "lead_lag": True, "end_time": 1.0},
    {"shape": (8, 2), "time_aug": True, "lead_lag": True, "end_time": 2.0},
    {"shape": (4, 10, 3), "time_aug": True, "lead_lag": False, "end_time": 1.0},
    {"shape": (3, 9, 2), "time_aug": False, "lead_lag": True, "end_time": 1.0},
    {"shape": (3, 8, 2), "time_aug": True, "lead_lag": True, "end_time": 2.0},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", TRANSFORM_PATH_CASES)
def test_jax_transform_path_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    ta, ll, et = case["time_aug"], case["lead_lag"], case["end_time"]

    expected = pysiglib.transform_path(x, time_aug=ta, lead_lag=ll, end_time=et)

    x_jax = _as_jax_array(x, device, dtype)

    def fn(path):
        return jax_api.transform_path(path, time_aug=ta, lead_lag=ll, end_time=et)

    actual = jax.jit(fn)(x_jax) if jitted else fn(x_jax)
    check_close(expected, actual, double_atol=1e-8)


# =========================================================================
# transform_path - backward
# =========================================================================

TRANSFORM_PATH_BACKWARD_CASES = [
    {"shape": (10, 3), "time_aug": True, "lead_lag": False, "end_time": 1.0},
    {"shape": (9, 2), "time_aug": False, "lead_lag": True, "end_time": 1.0},
    {"shape": (8, 2), "time_aug": True, "lead_lag": True, "end_time": 2.0},
    {"shape": (4, 10, 3), "time_aug": True, "lead_lag": False, "end_time": 1.0},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", TRANSFORM_PATH_BACKWARD_CASES)
def test_jax_transform_path_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    ta, ll, et = case["time_aug"], case["lead_lag"], case["end_time"]

    transformed_ref = pysiglib.transform_path(x, time_aug=ta, lead_lag=ll, end_time=et)
    weights = rng.uniform(size=transformed_ref.shape).astype(dtype)
    grad_ref = pysiglib.transform_path_backprop(weights, time_aug=ta, lead_lag=ll, end_time=et)

    x_jax = _as_jax_array(x, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(path):
        t = jax_api.transform_path(path, time_aug=ta, lead_lag=ll, end_time=et)
        return jnp.sum(t * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad = grad_fn(x_jax)
    check_close(grad_ref, grad, double_atol=1e-8)


# =========================================================================
# sig_to_log_sig - forward
# =========================================================================

SIG_TO_LOG_SIG_CASES = [
    {"dim": 3, "degree": 2, "method": 1, "time_aug": False, "lead_lag": False},
    {"dim": 2, "degree": 3, "method": 2, "time_aug": False, "lead_lag": False},
    {"dim": 2, "degree": 2, "method": 1, "time_aug": True, "lead_lag": False},
    {"dim": 2, "degree": 2, "method": 2, "time_aug": False, "lead_lag": True},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_TO_LOG_SIG_CASES)
def test_jax_sig_to_log_sig_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]
    method = case["method"]
    ta, ll = case["time_aug"], case["lead_lag"]

    x = rng.uniform(size=(10, dim)).astype(dtype)
    sig_np = pysiglib.sig(x, deg, time_aug=ta, lead_lag=ll)

    pysiglib.prepare_log_sig(dim, deg, method, time_aug=ta, lead_lag=ll)
    expected = pysiglib.sig_to_log_sig(sig_np, dim, deg, time_aug=ta, lead_lag=ll, method=method)

    sig_jax = _as_jax_array(sig_np, device, dtype)

    def fn(s):
        return jax_api.sig_to_log_sig(s, dim, deg, time_aug=ta, lead_lag=ll, method=method)

    actual = jax.jit(fn)(sig_jax) if jitted else fn(sig_jax)
    check_close(expected, actual, double_atol=1e-8)
    pysiglib.clear_cache()


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_TO_LOG_SIG_CASES)
def test_jax_sig_to_log_sig_batch_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]
    method = case["method"]
    ta, ll = case["time_aug"], case["lead_lag"]
    batch = 4

    x = rng.uniform(size=(batch, 10, dim)).astype(dtype)
    sig_np = pysiglib.sig(x, deg, time_aug=ta, lead_lag=ll)

    pysiglib.prepare_log_sig(dim, deg, method, time_aug=ta, lead_lag=ll)
    expected = pysiglib.sig_to_log_sig(sig_np, dim, deg, time_aug=ta, lead_lag=ll, method=method)

    sig_jax = _as_jax_array(sig_np, device, dtype)

    def fn(s):
        return jax_api.sig_to_log_sig(s, dim, deg, time_aug=ta, lead_lag=ll, method=method)

    actual = jax.jit(fn)(sig_jax) if jitted else fn(sig_jax)
    check_close(expected, actual, double_atol=1e-8)
    pysiglib.clear_cache()


# =========================================================================
# sig_to_log_sig - backward
# =========================================================================

SIG_TO_LOG_SIG_BACKWARD_CASES = [
    {"dim": 3, "degree": 2, "method": 1, "time_aug": False, "lead_lag": False},
    {"dim": 2, "degree": 3, "method": 2, "time_aug": False, "lead_lag": False},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_TO_LOG_SIG_BACKWARD_CASES)
def test_jax_sig_to_log_sig_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    dim, deg = case["dim"], case["degree"]
    method = case["method"]
    ta, ll = case["time_aug"], case["lead_lag"]

    x = rng.uniform(size=(10, dim)).astype(dtype)
    sig_np = pysiglib.sig(x, deg, time_aug=ta, lead_lag=ll)

    pysiglib.prepare_log_sig(dim, deg, method, time_aug=ta, lead_lag=ll)
    log_sig_np = pysiglib.sig_to_log_sig(sig_np, dim, deg, time_aug=ta, lead_lag=ll, method=method)

    weights = rng.uniform(size=log_sig_np.shape).astype(dtype)
    grad_ref = pysiglib.sig_to_log_sig_backprop(
        sig_np, weights, dim, deg, time_aug=ta, lead_lag=ll, method=method
    )

    sig_jax = _as_jax_array(sig_np, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(s):
        ls = jax_api.sig_to_log_sig(s, dim, deg, time_aug=ta, lead_lag=ll, method=method)
        return jnp.sum(ls * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad = grad_fn(sig_jax)
    check_close(grad_ref, grad, double_atol=1e-8)
    pysiglib.clear_cache()


# =========================================================================
# log_sig - forward (composition of sig + sig_to_log_sig)
# =========================================================================

LOG_SIG_CASES = [
    {"shape": (10, 3), "degree": 2, "method": 1, "time_aug": False, "lead_lag": False},
    {"shape": (12, 2), "degree": 3, "method": 2, "time_aug": False, "lead_lag": False},
    {"shape": (10, 2), "degree": 2, "method": 1, "time_aug": True, "lead_lag": False},
    {"shape": (9, 2), "degree": 2, "method": 2, "time_aug": False, "lead_lag": True},
    {"shape": (4, 10, 3), "degree": 2, "method": 1, "time_aug": False, "lead_lag": False},
    {"shape": (3, 12, 2), "degree": 3, "method": 2, "time_aug": False, "lead_lag": False},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", LOG_SIG_CASES)
def test_jax_log_sig_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    deg = case["degree"]
    method = case["method"]
    ta, ll = case["time_aug"], case["lead_lag"]
    dim = case["shape"][-1]

    pysiglib.prepare_log_sig(dim, deg, method, time_aug=ta, lead_lag=ll)
    expected = pysiglib.log_sig(x, deg, time_aug=ta, lead_lag=ll, method=method)

    x_jax = _as_jax_array(x, device, dtype)

    def fn(path):
        return jax_api.log_sig(path, deg, time_aug=ta, lead_lag=ll, method=method)

    actual = jax.jit(fn)(x_jax) if jitted else fn(x_jax)
    check_close(expected, actual, double_atol=1e-8)
    pysiglib.clear_cache()


# =========================================================================
# log_sig_combine - forward
# =========================================================================

LOG_SIG_COMBINE_CASES = [
    {"dim": 3, "degree": 2, "time_aug": False, "lead_lag": False},
    {"dim": 2, "degree": 3, "time_aug": True, "lead_lag": False},
    {"dim": 2, "degree": 2, "time_aug": False, "lead_lag": True},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", LOG_SIG_COMBINE_CASES)
def test_jax_log_sig_combine_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]

    x1 = rng.uniform(size=(10, dim)).astype(dtype)
    x2 = rng.uniform(size=(8, dim)).astype(dtype)

    pysiglib.prepare_log_sig(dim, deg, 2, time_aug=ta, lead_lag=ll)

    ls1 = pysiglib.log_sig(x1, deg, time_aug=ta, lead_lag=ll, method=1)
    ls2 = pysiglib.log_sig(x2, deg, time_aug=ta, lead_lag=ll, method=1)
    expected = pysiglib.log_sig_combine(ls1, ls2, dim, deg, time_aug=ta, lead_lag=ll)

    ls1_jax = _as_jax_array(ls1, device, dtype)
    ls2_jax = _as_jax_array(ls2, device, dtype)

    def fn(l1, l2):
        return jax_api.log_sig_combine(l1, l2, dim, deg, time_aug=ta, lead_lag=ll)

    actual = jax.jit(fn)(ls1_jax, ls2_jax) if jitted else fn(ls1_jax, ls2_jax)
    check_close(expected, actual, double_atol=1e-8)
    pysiglib.clear_cache()


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", LOG_SIG_COMBINE_CASES)
def test_jax_log_sig_combine_batch_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]
    batch = 4

    x1 = rng.uniform(size=(batch, 10, dim)).astype(dtype)
    x2 = rng.uniform(size=(batch, 8, dim)).astype(dtype)

    pysiglib.prepare_log_sig(dim, deg, 2, time_aug=ta, lead_lag=ll)

    ls1 = pysiglib.log_sig(x1, deg, time_aug=ta, lead_lag=ll, method=1)
    ls2 = pysiglib.log_sig(x2, deg, time_aug=ta, lead_lag=ll, method=1)
    expected = pysiglib.log_sig_combine(ls1, ls2, dim, deg, time_aug=ta, lead_lag=ll)

    ls1_jax = _as_jax_array(ls1, device, dtype)
    ls2_jax = _as_jax_array(ls2, device, dtype)

    def fn(l1, l2):
        return jax_api.log_sig_combine(l1, l2, dim, deg, time_aug=ta, lead_lag=ll)

    actual = jax.jit(fn)(ls1_jax, ls2_jax) if jitted else fn(ls1_jax, ls2_jax)
    check_close(expected, actual, double_atol=1e-8)
    pysiglib.clear_cache()


# =========================================================================
# log_sig_combine - backward
# =========================================================================

LOG_SIG_COMBINE_BACKWARD_CASES = [
    {"dim": 3, "degree": 2, "time_aug": False, "lead_lag": False},
    {"dim": 2, "degree": 3, "time_aug": True, "lead_lag": False},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", LOG_SIG_COMBINE_BACKWARD_CASES)
def test_jax_log_sig_combine_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    dim, deg = case["dim"], case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]

    x1 = rng.uniform(size=(10, dim)).astype(dtype)
    x2 = rng.uniform(size=(8, dim)).astype(dtype)

    pysiglib.prepare_log_sig(dim, deg, 2, time_aug=ta, lead_lag=ll)

    ls1_np = pysiglib.log_sig(x1, deg, time_aug=ta, lead_lag=ll, method=1)
    ls2_np = pysiglib.log_sig(x2, deg, time_aug=ta, lead_lag=ll, method=1)
    combined = pysiglib.log_sig_combine(ls1_np, ls2_np, dim, deg, time_aug=ta, lead_lag=ll)

    weights = rng.uniform(size=combined.shape).astype(dtype)
    grad_ls1_ref, grad_ls2_ref = pysiglib.log_sig_combine_backprop(
        weights, ls1_np, ls2_np, dim, deg, time_aug=ta, lead_lag=ll
    )

    ls1_jax = _as_jax_array(ls1_np, device, dtype)
    ls2_jax = _as_jax_array(ls2_np, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(l1, l2):
        comb = jax_api.log_sig_combine(l1, l2, dim, deg, time_aug=ta, lead_lag=ll)
        return jnp.sum(comb * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1))) if jitted else jax.grad(loss_fn, argnums=(0, 1))
    grad_ls1, grad_ls2 = grad_fn(ls1_jax, ls2_jax)

    check_close(grad_ls1_ref, grad_ls1, double_atol=1e-8)
    check_close(grad_ls2_ref, grad_ls2, double_atol=1e-8)
    pysiglib.clear_cache()


# =========================================================================
# sig_kernel - forward
# =========================================================================

SIG_KERNEL_CASES = [
    {"length1": 8, "length2": 8, "dim": 2, "dyadic_order": 0, "batch": None},
    {"length1": 6, "length2": 7, "dim": 3, "dyadic_order": 1, "batch": None},
    {"length1": 6, "length2": 6, "dim": 2, "dyadic_order": 0, "batch": 3},
    {"length1": 5, "length2": 6, "dim": 2, "dyadic_order": 1, "batch": 2},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_KERNEL_CASES)
def test_jax_sig_kernel_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    l1, l2, dim = case["length1"], case["length2"], case["dim"]
    do = case["dyadic_order"]
    batch = case["batch"]

    if batch is not None:
        shape1 = (batch, l1, dim)
        shape2 = (batch, l2, dim)
    else:
        shape1 = (l1, dim)
        shape2 = (l2, dim)

    p1 = (rng.uniform(size=shape1) * 0.1).astype(dtype)
    p2 = (rng.uniform(size=shape2) * 0.1).astype(dtype)

    expected = pysiglib.sig_kernel(p1, p2, do)

    p1_jax = _as_jax_array(p1, device, dtype)
    p2_jax = _as_jax_array(p2, device, dtype)

    def fn(a, b):
        return jax_api.sig_kernel(a, b, do)

    actual = jax.jit(fn)(p1_jax, p2_jax) if jitted else fn(p1_jax, p2_jax)
    check_close(np.asarray(expected), actual, double_atol=1e-8)


# =========================================================================
# sig_kernel - backward
# =========================================================================

SIG_KERNEL_BACKWARD_CASES = [
    {"length1": 8, "length2": 8, "dim": 2, "dyadic_order": 0},
    {"length1": 6, "length2": 7, "dim": 2, "dyadic_order": 1},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("case", SIG_KERNEL_BACKWARD_CASES)
def test_jax_sig_kernel_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    l1, l2, dim = case["length1"], case["length2"], case["dim"]
    do = case["dyadic_order"]
    batch = 2

    p1 = (rng.uniform(size=(batch, l1, dim)) * 0.1).astype(dtype)
    p2 = (rng.uniform(size=(batch, l2, dim)) * 0.1).astype(dtype)

    k_ref = pysiglib.sig_kernel(p1, p2, do)
    weights = rng.uniform(size=k_ref.shape).astype(dtype)

    grad_p1_ref, _ = pysiglib.sig_kernel_backprop(
        weights, p1, p2, do, left_deriv=True, right_deriv=False
    )

    p1_jax = _as_jax_array(p1, device, dtype)
    p2_jax = _as_jax_array(p2, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(path1):
        k = jax_api.sig_kernel(path1, p2_jax, do)
        return jnp.sum(k * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad_p1 = grad_fn(p1_jax)
    check_close(grad_p1_ref, grad_p1, double_atol=1e-5)


# =========================================================================
# sig_kernel_gram - forward
# =========================================================================

SIG_KERNEL_GRAM_CASES = [
    {"length1": 6, "length2": 6, "dim": 2, "dyadic_order": 0, "batch1": 2, "batch2": 3},
    {"length1": 5, "length2": 7, "dim": 2, "dyadic_order": 1, "batch1": 2, "batch2": 2},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("case", SIG_KERNEL_GRAM_CASES)
def test_jax_sig_kernel_gram_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    l1, l2, dim = case["length1"], case["length2"], case["dim"]
    do = case["dyadic_order"]
    b1, b2 = case["batch1"], case["batch2"]

    p1 = (rng.uniform(size=(b1, l1, dim)) * 0.1).astype(dtype)
    p2 = (rng.uniform(size=(b2, l2, dim)) * 0.1).astype(dtype)

    expected = pysiglib.sig_kernel_gram(p1, p2, do)

    p1_jax = _as_jax_array(p1, device, dtype)
    p2_jax = _as_jax_array(p2, device, dtype)

    def fn(a, b):
        return jax_api.sig_kernel_gram(a, b, do)

    actual = jax.jit(fn)(p1_jax, p2_jax) if jitted else fn(p1_jax, p2_jax)
    check_close(np.asarray(expected), actual, double_atol=1e-5)


# =========================================================================
# sig_coef - forward (without and with prefixes)
# =========================================================================

SIG_COEF_CASES = [
    {"shape": (10, 3), "words": [(0,), (1, 2), (2, 0, 1)], "prefixes": False},
    {"shape": (10, 3), "words": [(0,), (1, 2)], "prefixes": True},
    {"shape": (4, 10, 3), "words": [(0,), (2, 1)], "prefixes": False},
    {"shape": (4, 10, 3), "words": [(1, 0), (2, 1, 0)], "prefixes": True},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", SIG_COEF_CASES)
def test_jax_sig_coef_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    words = case["words"]
    prefixes = case["prefixes"]

    expected = pysiglib.sig_coef(x, words, prefixes=prefixes)

    x_jax = _as_jax_array(x, device, dtype)

    def fn(path):
        return jax_api.sig_coef(path, words, prefixes=prefixes)

    actual = jax.jit(fn)(x_jax) if jitted else fn(x_jax)
    check_close(np.asarray(expected), actual, double_atol=1e-8)


# =========================================================================
# sig_mmd - forward
# =========================================================================

SIG_MMD_CASES = [
    {"length": 6, "dim": 2, "dyadic_order": 0, "batch1": 3, "batch2": 3},
    {"length": 5, "dim": 2, "dyadic_order": 1, "batch1": 2, "batch2": 3},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("case", SIG_MMD_CASES)
def test_jax_sig_mmd_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    length, dim = case["length"], case["dim"]
    do = case["dyadic_order"]
    b1, b2 = case["batch1"], case["batch2"]

    s1 = (rng.uniform(size=(b1, length, dim)) * 0.1).astype(dtype)
    s2 = (rng.uniform(size=(b2, length, dim)) * 0.1).astype(dtype)

    expected = pysiglib.sig_mmd(s1, s2, do)

    s1_jax = _as_jax_array(s1, device, dtype)
    s2_jax = _as_jax_array(s2, device, dtype)

    def fn(a, b):
        return jax_api.sig_mmd(a, b, do)

    actual = jax.jit(fn)(s1_jax, s2_jax) if jitted else fn(s1_jax, s2_jax)
    check_close(np.asarray(expected), actual, double_atol=1e-5)


# =========================================================================
# branched_sig - forward
# =========================================================================

BRANCHED_SIG_CASES = [
    {"shape": (10, 3), "degree": 1, "time_aug": False, "lead_lag": False},
    {"shape": (8, 2), "degree": 2, "time_aug": True, "lead_lag": False},
    {"shape": (10, 2), "degree": 2, "time_aug": False, "lead_lag": True},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BRANCHED_SIG_CASES)
def test_jax_branched_sig_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    deg = case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]
    dim = case["shape"][-1]

    pysiglib.prepare_branched_sig(dim, deg, time_aug=ta, lead_lag=ll)
    expected = pysiglib.branched_sig(x, deg, time_aug=ta, lead_lag=ll)

    x_jax = _as_jax_array(x, device, dtype)

    def fn(path):
        return jax_api.branched_sig(path, deg, time_aug=ta, lead_lag=ll)

    actual = jax.jit(fn)(x_jax) if jitted else fn(x_jax)
    check_close(expected, actual, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BRANCHED_SIG_CASES)
def test_jax_branched_sig_batch_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    batch = 4
    x = rng.uniform(size=(batch,) + tuple(case["shape"])).astype(dtype)
    deg = case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]
    dim = case["shape"][-1]

    pysiglib.prepare_branched_sig(dim, deg, time_aug=ta, lead_lag=ll)
    expected = pysiglib.branched_sig(x, deg, time_aug=ta, lead_lag=ll)

    x_jax = _as_jax_array(x, device, dtype)

    def fn(path):
        return jax_api.branched_sig(path, deg, time_aug=ta, lead_lag=ll)

    actual = jax.jit(fn)(x_jax) if jitted else fn(x_jax)
    check_close(expected, actual, double_atol=1e-8)


# =========================================================================
# branched_sig - backward
# =========================================================================

BRANCHED_SIG_BACKWARD_CASES = [
    {"shape": (10, 3), "degree": 1, "time_aug": False, "lead_lag": False},
    {"shape": (8, 2), "degree": 2, "time_aug": True, "lead_lag": False},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BRANCHED_SIG_BACKWARD_CASES)
def test_jax_branched_sig_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    x = rng.uniform(size=case["shape"]).astype(dtype)
    deg = case["degree"]
    ta, ll = case["time_aug"], case["lead_lag"]
    dim = case["shape"][-1]

    pysiglib.prepare_branched_sig(dim, deg, time_aug=ta, lead_lag=ll)
    bsig_ref = pysiglib.branched_sig(x, deg, time_aug=ta, lead_lag=ll)

    weights = rng.uniform(size=bsig_ref.shape).astype(dtype)
    grad_ref = pysiglib.branched_sig_backprop(
        x, bsig_ref, weights, deg, time_aug=ta, lead_lag=ll
    )

    x_jax = _as_jax_array(x, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(path):
        bsig = jax_api.branched_sig(path, deg, time_aug=ta, lead_lag=ll)
        return jnp.sum(bsig * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad = grad_fn(x_jax)
    check_close(grad_ref, grad, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
def test_jax_branched_sig_planar_matches_pysiglib(device, jitted):
    rng = np.random.default_rng(4242)
    x = rng.uniform(size=(8, 2)).astype(np.float64)
    deg = 2

    pysiglib.prepare_branched_sig(2, deg, planar=True)
    expected = pysiglib.branched_sig(x, deg, planar=True)

    x_jax = _as_jax_array(x, device, np.float64)

    def fn(path):
        return jax_api.branched_sig(path, deg, planar=True)

    actual = jax.jit(fn)(x_jax) if jitted else fn(x_jax)
    check_close(expected, actual, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
def test_jax_branched_sig_planar_grad_matches_pysiglib(device, jitted):
    rng = np.random.default_rng(4344)
    x = rng.uniform(size=(7, 2)).astype(np.float64)
    deg = 2

    pysiglib.prepare_branched_sig(2, deg, planar=True)
    bsig_ref = pysiglib.branched_sig(x, deg, planar=True)
    weights = rng.uniform(size=bsig_ref.shape).astype(np.float64)
    grad_ref = pysiglib.branched_sig_backprop(x, bsig_ref, weights, deg, planar=True)

    x_jax = _as_jax_array(x, device, np.float64)
    weights_jax = _as_jax_array(weights, device, np.float64)

    def loss_fn(path):
        bsig = jax_api.branched_sig(path, deg, planar=True)
        return jnp.sum(bsig * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad = grad_fn(x_jax)
    check_close(grad_ref, grad, double_atol=1e-8)


# =========================================================================
# branched_sig_combine - forward
# =========================================================================

BRANCHED_SIG_COMBINE_CASES = [
    {"dim": 3, "degree": 1},
    {"dim": 2, "degree": 2},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BRANCHED_SIG_COMBINE_CASES)
def test_jax_branched_sig_combine_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]

    pysiglib.prepare_branched_sig(dim, deg)

    x1 = rng.uniform(size=(10, dim)).astype(dtype)
    x2 = rng.uniform(size=(8, dim)).astype(dtype)

    bsig1 = pysiglib.branched_sig(x1, deg)
    bsig2 = pysiglib.branched_sig(x2, deg)
    expected = pysiglib.branched_sig_combine(bsig1, bsig2, dim, deg)

    bsig1_jax = _as_jax_array(bsig1, device, dtype)
    bsig2_jax = _as_jax_array(bsig2, device, dtype)

    def fn(s1, s2):
        return jax_api.branched_sig_combine(s1, s2, dim, deg)

    actual = jax.jit(fn)(bsig1_jax, bsig2_jax) if jitted else fn(bsig1_jax, bsig2_jax)
    check_close(expected, actual, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BRANCHED_SIG_COMBINE_CASES)
def test_jax_branched_sig_combine_batch_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    dim, deg = case["dim"], case["degree"]
    batch = 4

    pysiglib.prepare_branched_sig(dim, deg)

    x1 = rng.uniform(size=(batch, 10, dim)).astype(dtype)
    x2 = rng.uniform(size=(batch, 8, dim)).astype(dtype)

    bsig1 = pysiglib.branched_sig(x1, deg)
    bsig2 = pysiglib.branched_sig(x2, deg)
    expected = pysiglib.branched_sig_combine(bsig1, bsig2, dim, deg)

    bsig1_jax = _as_jax_array(bsig1, device, dtype)
    bsig2_jax = _as_jax_array(bsig2, device, dtype)

    def fn(s1, s2):
        return jax_api.branched_sig_combine(s1, s2, dim, deg)

    actual = jax.jit(fn)(bsig1_jax, bsig2_jax) if jitted else fn(bsig1_jax, bsig2_jax)
    check_close(expected, actual, double_atol=1e-8)


# =========================================================================
# branched_sig_combine - backward
# =========================================================================

BRANCHED_SIG_COMBINE_BACKWARD_CASES = [
    {"dim": 3, "degree": 1},
    {"dim": 2, "degree": 2},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BRANCHED_SIG_COMBINE_BACKWARD_CASES)
def test_jax_branched_sig_combine_grad_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(123)
    dim, deg = case["dim"], case["degree"]

    pysiglib.prepare_branched_sig(dim, deg)

    x1 = rng.uniform(size=(10, dim)).astype(dtype)
    x2 = rng.uniform(size=(8, dim)).astype(dtype)

    bsig1_np = pysiglib.branched_sig(x1, deg)
    bsig2_np = pysiglib.branched_sig(x2, deg)
    combined = pysiglib.branched_sig_combine(bsig1_np, bsig2_np, dim, deg)

    weights = rng.uniform(size=combined.shape).astype(dtype)
    grad_bsig1_ref, grad_bsig2_ref = pysiglib.branched_sig_combine_backprop(
        weights, bsig1_np, bsig2_np, dim, deg
    )

    bsig1_jax = _as_jax_array(bsig1_np, device, dtype)
    bsig2_jax = _as_jax_array(bsig2_np, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(s1, s2):
        comb = jax_api.branched_sig_combine(s1, s2, dim, deg)
        return jnp.sum(comb * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1))) if jitted else jax.grad(loss_fn, argnums=(0, 1))
    grad_bsig1, grad_bsig2 = grad_fn(bsig1_jax, bsig2_jax)

    check_close(grad_bsig1_ref, grad_bsig1, double_atol=1e-8)
    check_close(grad_bsig2_ref, grad_bsig2, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
def test_jax_branched_sig_combine_planar_matches_pysiglib(device, jitted):
    rng = np.random.default_rng(4444)
    dim, deg = 2, 2

    pysiglib.prepare_branched_sig(dim, deg, planar=True)

    x1 = rng.uniform(size=(9, dim)).astype(np.float64)
    x2 = rng.uniform(size=(6, dim)).astype(np.float64)

    bsig1 = pysiglib.branched_sig(x1, deg, planar=True)
    bsig2 = pysiglib.branched_sig(x2, deg, planar=True)
    expected = pysiglib.branched_sig_combine(bsig1, bsig2, dim, deg, planar=True)

    bsig1_jax = _as_jax_array(bsig1, device, np.float64)
    bsig2_jax = _as_jax_array(bsig2, device, np.float64)

    def fn(s1, s2):
        return jax_api.branched_sig_combine(s1, s2, dim, deg, planar=True)

    actual = jax.jit(fn)(bsig1_jax, bsig2_jax) if jitted else fn(bsig1_jax, bsig2_jax)
    check_close(expected, actual, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
def test_jax_branched_sig_combine_planar_grad_matches_pysiglib(device, jitted):
    rng = np.random.default_rng(4546)
    dim, deg = 2, 2

    pysiglib.prepare_branched_sig(dim, deg, planar=True)

    x1 = rng.uniform(size=(8, dim)).astype(np.float64)
    x2 = rng.uniform(size=(7, dim)).astype(np.float64)

    bsig1_np = pysiglib.branched_sig(x1, deg, planar=True)
    bsig2_np = pysiglib.branched_sig(x2, deg, planar=True)
    combined = pysiglib.branched_sig_combine(bsig1_np, bsig2_np, dim, deg, planar=True)

    weights = rng.uniform(size=combined.shape).astype(np.float64)
    grad_bsig1_ref, grad_bsig2_ref = pysiglib.branched_sig_combine_backprop(
        weights, bsig1_np, bsig2_np, dim, deg, planar=True)

    bsig1_jax = _as_jax_array(bsig1_np, device, np.float64)
    bsig2_jax = _as_jax_array(bsig2_np, device, np.float64)
    weights_jax = _as_jax_array(weights, device, np.float64)

    def loss_fn(s1, s2):
        comb = jax_api.branched_sig_combine(s1, s2, dim, deg, planar=True)
        return jnp.sum(comb * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1))) if jitted else jax.grad(loss_fn, argnums=(0, 1))
    grad_bsig1, grad_bsig2 = grad_fn(bsig1_jax, bsig2_jax)

    check_close(grad_bsig1_ref, grad_bsig1, double_atol=1e-8)
    check_close(grad_bsig2_ref, grad_bsig2, double_atol=1e-8)
