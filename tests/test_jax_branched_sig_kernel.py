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

import native_api as pysiglib
from conftest import check_close

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
jax_api = pytest.importorskip("pysiglib.jax_api")


def _jax_devices():
    platforms = {d.platform for d in jax.devices()}
    devices = []
    if "cpu" in platforms:
        devices.append("cpu")
    if pysiglib.BUILT_WITH_CUDA and platforms & {"gpu", "cuda"}:
        devices.append("cuda")
    return devices


def _jax_device(platform):
    wanted = {"cpu": {"cpu"}, "cuda": {"gpu", "cuda"}}[platform]
    for device in jax.devices():
        if device.platform in wanted:
            return device
    raise RuntimeError(f"No JAX device found for platform {platform}")


def _as_jax_array(x, device, dtype):
    arr = jnp.asarray(x, dtype=dtype)
    return jax.device_put(arr, _jax_device(device))


BRANCHED_KERNEL_CASES = [
    {"length1": 5, "length2": 5, "dim": 2, "depth": 1, "dyadic_order": 0, "batch": None},
    {"length1": 5, "length2": 6, "dim": 2, "depth": 2, "dyadic_order": 1, "batch": 2},
]


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", BRANCHED_KERNEL_CASES)
def test_jax_branched_sig_kernel_matches_pysiglib(device, jitted, dtype, case):
    rng = np.random.default_rng(42)
    batch = case["batch"]
    shape1 = (case["length1"], case["dim"]) if batch is None else (batch, case["length1"], case["dim"])
    shape2 = (case["length2"], case["dim"]) if batch is None else (batch, case["length2"], case["dim"])
    path1 = (0.1 * rng.normal(size=shape1)).astype(dtype)
    path2 = (0.1 * rng.normal(size=shape2)).astype(dtype)

    expected = pysiglib.branched_sig_kernel(
        path1, path2, case["depth"], case["dyadic_order"])
    path1_jax = _as_jax_array(path1, device, dtype)
    path2_jax = _as_jax_array(path2, device, dtype)

    def fn(a, b):
        return jax_api.branched_sig_kernel(
            a, b, case["depth"], case["dyadic_order"])

    actual = jax.jit(fn)(path1_jax, path2_jax) if jitted else fn(path1_jax, path2_jax)
    check_close(expected, actual, single_atol=5e-3, double_atol=1e-8)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float64])
def test_jax_branched_sig_kernel_grad_matches_pysiglib(device, jitted, dtype):
    rng = np.random.default_rng(123)
    path1 = (0.1 * rng.normal(size=(2, 5, 2))).astype(dtype)
    path2 = (0.1 * rng.normal(size=(2, 6, 2))).astype(dtype)
    depth = 2
    dyadic_order = 0

    ref = pysiglib.branched_sig_kernel(path1, path2, depth, dyadic_order)
    weights = rng.normal(size=ref.shape).astype(dtype)
    grad_ref, _ = pysiglib.branched_sig_kernel_backprop(
        weights, path1, path2, depth, dyadic_order,
        left_deriv=True, right_deriv=False)

    path1_jax = _as_jax_array(path1, device, dtype)
    path2_jax = _as_jax_array(path2, device, dtype)
    weights_jax = _as_jax_array(weights, device, dtype)

    def loss_fn(path):
        value = jax_api.branched_sig_kernel(path, path2_jax, depth, dyadic_order)
        return jnp.sum(value * weights_jax)

    grad_fn = jax.jit(jax.grad(loss_fn)) if jitted else jax.grad(loss_fn)
    grad = grad_fn(path1_jax)
    check_close(grad_ref, grad, single_atol=5e-3, double_atol=5e-5)


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("dtype", [np.float64])
def test_jax_branched_sig_kernel_gram_matches_pysiglib(device, jitted, dtype):
    rng = np.random.default_rng(456)
    path1 = (0.1 * rng.normal(size=(2, 5, 2))).astype(dtype)
    path2 = (0.1 * rng.normal(size=(3, 6, 2))).astype(dtype)
    depth = 2
    dyadic_order = 0

    expected = pysiglib.branched_sig_kernel_gram(path1, path2, depth, dyadic_order)
    path1_jax = _as_jax_array(path1, device, dtype)
    path2_jax = _as_jax_array(path2, device, dtype)

    def fn(a, b):
        return jax_api.branched_sig_kernel_gram(a, b, depth, dyadic_order)

    actual = jax.jit(fn)(path1_jax, path2_jax) if jitted else fn(path1_jax, path2_jax)
    check_close(expected, actual, single_atol=5e-3, double_atol=1e-6)
