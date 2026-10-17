# Copyright 2025 Daniil Shmelev
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

import pytest
import numpy as np
import torch

import pysiglib
from conftest import DEVICES, skip_no_cuda
from pysiglib.load_siglib import CPSIG, CUSIG

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-10


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("suffix", ["f", "d"])
@pytest.mark.parametrize("operation", [
    "sig_to_log_sig", "sig_to_log_sig_backprop",
    "logsig_to_sig", "logsig_to_sig_backprop",
])
@pytest.mark.parametrize("method", [-1, 3, 4])
def test_native_log_conversion_rejects_invalid_method(device, suffix, operation, method):
    library = CPSIG if device == "cpu" else CUSIG
    name = operation + ("_cuda" if device == "cuda" else "") + "_" + suffix
    pointers = [None] * (3 if operation.endswith("backprop") else 2)
    args = pointers + [0, 2, 3]
    if device == "cpu":
        args += [False, False]
    args += [method, True]
    if device == "cpu":
        args += [1]
    assert getattr(library, name)(*args) == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", [-1, 4])
def test_native_log_preparation_rejects_invalid_method(device, method):
    prepare = CPSIG.prepare_log_sig if device == "cpu" else CUSIG.prepare_log_sig_cuda
    assert prepare(2, 3, method, False) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_native_expanded_log_preparation(device):
    prepare = CPSIG.prepare_log_sig if device == "cpu" else CUSIG.prepare_log_sig_cuda
    assert prepare(2, 3, 0, False) == 0


@pytest.mark.parametrize("dim, deg", [(-1, 2), (1, -2)])
def test_sig_length_value_error(dim, deg):
    with pytest.raises(ValueError):
        pysiglib.sig_length(dim, deg)


@pytest.mark.parametrize("args", [
    ('a', 2, False, False, False, False),
    (np.array(['a', 'b']), 2, False, False, False, False),
    (np.array([[0.], [1.]]), 'a', False, False, False, False),
    (np.array([[0.], [1.]]), 2, 'a', False, False, False),
    (np.array([[0.], [1.]]), 2, False, 'a', False, False),
    (np.array([[0.], [1.]]), 2, False, False, 'a', False),
    (np.array([[[0.], [1.]]]), 2, False, False, False, 'a'),
])
def test_signature_type_error(args):
    with pytest.raises(TypeError):
        pysiglib.sig(*args)


@pytest.mark.parametrize("X, deg", [
    (np.array([0., 1.]), 2),
    (np.array([[0.], [1.]]), -1),
])
def test_signature_value_error(X, deg):
    with pytest.raises(ValueError):
        pysiglib.sig(X, deg)


@pytest.mark.parametrize("x, y, d", [
    ('a', np.array([[0.], [1.]]), 2),
    (np.array([[0.], [1.]]), 'a', 2),
    (np.array([[0.], [1.]]), np.array([[0.], [1.]]), 'a'),
])
def test_sig_kernel_type_error(x, y, d):
    with pytest.raises(TypeError):
        pysiglib.sig_kernel(x, y, dyadic_order=d)


@pytest.mark.parametrize("x, y, d", [
    (np.array([0., 1.]), np.array([[0.], [1.]]), 2),
    (np.array([[0.], [1.]]), np.array([0., 1.]), 2),
    (np.array([[[[0.]]], [[[1.]]]]), np.array([[0.], [1.]]), 2),
    (np.array([[0.], [1.]]), np.array([[[[0.]]], [[[1.]]]]), 2),
    (np.array([[0.], [1.]]), np.array([[0.], [1.]]), -2),
])
def test_sig_kernel_value_error(x, y, d):
    with pytest.raises(ValueError):
        pysiglib.sig_kernel(x, y, dyadic_order=d)

def test_signature_n_jobs_zero():
    with pytest.raises(ValueError):
        pysiglib.sig(np.array([[[0.], [1.]]]), 2, n_jobs = 0)

def test_sig_combine_n_jobs_zero():
    sig = pysiglib.sig(np.array([[0.], [1.]]), 2)
    with pytest.raises(ValueError):
        pysiglib.sig_combine(sig, sig, 1, 2, n_jobs = 0)

def test_sig_kernel_n_jobs_zero():
    with pytest.raises(ValueError):
        pysiglib.sig_kernel(np.array([[0.], [1.]]), np.array([[0.], [1.]]), dyadic_order=0, n_jobs = 0)


def test_cpu_native_error_detail_is_propagated_without_stderr(capsys):
    pysiglib.clear_cache(device="cpu")
    path = np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(Exception) as exc_info:
        pysiglib.branched_sig(path, 3)

    message = str(exc_info.value)
    assert "Could not find prepared cache" in message
    assert "call prepare_branched_sig first" in message
    assert capsys.readouterr().err == ""


@skip_no_cuda
def test_cuda_native_error_detail_is_propagated_without_stderr(capsys):
    pysiglib.clear_cache(device="cuda")
    path = torch.zeros((3, 2), dtype=torch.float64, device="cuda")

    with pytest.raises(Exception) as exc_info:
        pysiglib.branched_sig(path, 3)

    message = str(exc_info.value)
    assert "Could not find prepared cache" in message
    assert "device='cuda'" in message
    assert capsys.readouterr().err == ""


@pytest.mark.skipif(
    not pysiglib.BUILT_WITH_JAX_FFI,
    reason="pysiglib was built without JAX FFI",
)
def test_jax_native_error_detail_is_propagated():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as jax_api

    cuda_backend = jax.default_backend() == "gpu"
    device = "cuda" if cuda_backend else "cpu"
    expected = (
        "call prepare_branched_sig with device='cuda' first"
        if cuda_backend else "call prepare_branched_sig first")
    pysiglib.clear_cache(device=device)
    path = jnp.zeros((3, 2), dtype=jnp.float32)

    with pytest.raises(Exception, match=expected):
        jax_api.branched_sig(path, 3).block_until_ready()
