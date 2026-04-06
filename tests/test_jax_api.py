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
    devices = ["cpu"]
    if pysiglib.BUILT_WITH_CUDA and any(d.platform in {"gpu", "cuda"} for d in jax.devices()):
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
