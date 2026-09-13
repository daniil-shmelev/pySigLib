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

import pysiglib as ps
import pysiglib.torch_api as pt


LINEAR_CASES = [
    ("linear_log_sig", "log_sig", {"method": method, "scalar_term": scalar})
    for method in range(4) for scalar in (False, True)
] + [
    ("linear_branched_sig", "branched_sig", {"planar": planar, "scalar_term": scalar})
    for planar in (False, True) for scalar in (False, True)
] + [
    ("linear_branched_log_sig", "branched_log_sig",
     {"planar": planar, "method": method, "scalar_term": scalar})
    for planar, method in [(False, 0), (True, 0), (True, 1), (True, 2), (True, 3)]
    for scalar in (False, True)
]

DEVICES = ["cpu"] + (["cuda"] if ps.BUILT_WITH_CUDA and torch.cuda.is_available() else [])


@pytest.fixture(scope="module", autouse=True)
def prepare():
    for degree in (0, 1, 3, 4):
        for method in range(4):
            ps.prepare_log_sig(2, degree, method)
            ps.prepare_branched_log_sig(2, degree, method, planar=True)
        ps.prepare_branched_log_sig(2, degree, 0, planar=False)


@pytest.mark.parametrize("name,path_name,kwargs", LINEAR_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_linear_forward_and_backprop(name, path_name, kwargs, dtype, device):
    torch.manual_seed(9)
    displacement = (torch.randn(2, 3, 4, dtype=dtype, device=device)[..., ::2] * 0.2).requires_grad_()
    path = torch.stack([torch.zeros_like(displacement), displacement], dim=-2)
    expected = getattr(pt, path_name)(path, 3, **kwargs)
    actual = getattr(pt, name)(displacement, 2, 3, **kwargs)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    expected_grad, = torch.autograd.grad(expected, displacement, weights)
    actual_grad, = torch.autograd.grad(actual, displacement, weights)
    torch.testing.assert_close(actual_grad, expected_grad)
    explicit = getattr(ps, name + "_backprop")(weights, displacement.detach(), 2, 3, **kwargs)
    torch.testing.assert_close(explicit, expected_grad)
    if device == "cpu":
        value = displacement.detach().numpy()
        np.testing.assert_allclose(getattr(ps, name)(value, 2, 3, **kwargs), actual.detach().numpy(), rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(getattr(ps, name + "_backprop")(weights.numpy(), value, 2, 3, **kwargs), expected_grad.numpy(), rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("name,path_name,kwargs", LINEAR_CASES)
@pytest.mark.parametrize("shape", [(2,), (2, 0, 2)])
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_linear_shapes(name, path_name, kwargs, shape, degree):
    displacement = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    result = getattr(pt, name)(displacement, 2, degree, **kwargs)
    assert result.shape[:-1] == shape[:-1]
    grad, = torch.autograd.grad(result.sum(), displacement)
    assert grad.shape == displacement.shape


@pytest.mark.parametrize("planar,scalar", [(False, False), (False, True), (True, False), (True, True), (None, False)])
@pytest.mark.parametrize("prepend", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_join_matches_path_and_gradients(planar, scalar, prepend, dtype, device):
    torch.manual_seed(11)
    path = torch.randn(2, 3, 4, 2, dtype=dtype, device=device) * 0.1
    displacement = (torch.randn(2, 3, 2, dtype=dtype, device=device) * 0.2).requires_grad_()
    is_log = planar is None
    name = "branched_log_sig_join" if is_log else "branched_sig_join"
    path_fn = pt.branched_log_sig if is_log else pt.branched_sig
    path_kwargs = {"planar": True, "method": 3} if is_log else {"planar": planar, "scalar_term": scalar}
    kwargs = {"prepend": prepend} if is_log else {"planar": planar, "prepend": prepend}
    signature = path_fn(path, 4, **path_kwargs).detach().requires_grad_()
    actual = getattr(pt, name)(signature, displacement, 2, 4, **kwargs)
    if prepend:
        extended = torch.cat([path[..., :1, :] - displacement.unsqueeze(-2), path], dim=-2)
    else:
        extended = torch.cat([path, path[..., -1:, :] + displacement.unsqueeze(-2)], dim=-2)
    expected = path_fn(extended, 4, **path_kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
    weights = torch.randn_like(actual)
    expected_grad, = torch.autograd.grad(expected, displacement, weights)
    d_signature, d_displacement = torch.autograd.grad(actual, (signature, displacement), weights)
    torch.testing.assert_close(d_displacement, expected_grad, rtol=2e-4, atol=2e-6)
    explicit = getattr(ps, name + "_backprop")(weights, signature.detach(), displacement.detach(), 2, 4, **kwargs)
    torch.testing.assert_close(explicit[0], d_signature)
    torch.testing.assert_close(explicit[1], d_displacement)
    if device == "cpu":
        np.testing.assert_allclose(getattr(ps, name)(signature.detach().numpy(), displacement.detach().numpy(), 2, 4, **kwargs), actual.detach().numpy(), rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("prepend", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_log_join_gradcheck(prepend, device):
    torch.manual_seed(12)
    length = ps.branched_log_sig_length(2, 4, planar=True)
    logsig = (torch.randn(length, dtype=torch.float64, device=device) * 0.1).requires_grad_()
    displacement = torch.tensor([0.0, 0.2], dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: pt.branched_log_sig_join(a, b, 2, 4, prepend=prepend), (logsig, displacement))


@pytest.mark.parametrize("degree", [0, 1, 3])
@pytest.mark.parametrize("shape", [(2,), (2, 0, 2)])
@pytest.mark.parametrize("device", DEVICES)
def test_join_empty_and_zero(degree, shape, device):
    displacement = torch.zeros(shape, dtype=torch.float64, device=device, requires_grad=True)
    for is_log in (False, True):
        linear = pt.linear_branched_log_sig if is_log else pt.linear_branched_sig
        join = pt.branched_log_sig_join if is_log else pt.branched_sig_join
        kwargs = {"planar": True, "method": 3} if is_log else {"planar": True}
        signature = linear(displacement, 2, degree, **kwargs).detach().requires_grad_()
        result = join(signature, displacement, 2, degree, **({} if is_log else {"planar": True}))
        grad_sig, grad_disp = torch.autograd.grad(result.sum(), (signature, displacement))
        assert grad_sig.shape == signature.shape
        assert grad_disp.shape == displacement.shape
        torch.testing.assert_close(grad_disp, torch.zeros_like(grad_disp) if degree == 0 else torch.ones_like(grad_disp))


@pytest.mark.parametrize("name", ["linear_log_sig", "linear_branched_sig", "linear_branched_log_sig"])
@pytest.mark.parametrize("value", [np.array(1.0), np.zeros(3), np.zeros(2, dtype=np.int64)])
def test_linear_invalid_displacement(name, value):
    with pytest.raises((ValueError, TypeError)):
        getattr(ps, name)(value, 2, 3)


def test_log_join_invalid_inputs():
    length = ps.branched_log_sig_length(2, 3, planar=True)
    log = np.zeros((2, length))
    for displacement in (np.zeros(3), np.zeros((3, 2)), np.zeros((2, 2), dtype=np.float32)):
        with pytest.raises(ValueError):
            ps.branched_log_sig_join(log, displacement, 2, 3)
    with pytest.raises(ValueError):
        ps.branched_log_sig_join(np.zeros(length + 1), np.zeros(2), 2, 3)


@pytest.mark.parametrize("name,path_name,kwargs", LINEAR_CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_linear(name, path_name, kwargs, dtype):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as pj
    jax.config.update("jax_enable_x64", True)
    displacement = np.array([[0.1, -0.3], [0.0, 0.2]], dtype=dtype)
    function = lambda v: getattr(pj, name)(v, 2, 3, **kwargs)
    result = jax.jit(jax.vmap(function))(jnp.asarray(displacement))
    expected = getattr(ps, name)(displacement, 2, 3, **kwargs)
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-7)
    grad = jax.jit(jax.grad(lambda v: function(v).sum()))(jnp.asarray(displacement))
    expected_grad = getattr(ps, name + "_backprop")(np.ones_like(expected), displacement, 2, 3, **kwargs)
    np.testing.assert_allclose(grad, expected_grad, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("is_log", [False, True])
@pytest.mark.parametrize("prepend", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_join(is_log, prepend, dtype):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as pj
    jax.config.update("jax_enable_x64", True)
    linear = ps.linear_branched_log_sig if is_log else ps.linear_branched_sig
    name = "branched_log_sig_join" if is_log else "branched_sig_join"
    kwargs = {"prepend": prepend} if is_log else {"planar": True, "prepend": prepend}
    displacement = np.array([[0.1, -0.3], [0.0, 0.2]], dtype=dtype)
    signature = linear(displacement, 2, 3, **({"planar": True, "method": 3} if is_log else {"planar": True}))
    function = lambda a, b: getattr(pj, name)(a, b, 2, 3, **kwargs)
    result = jax.jit(jax.vmap(function))(jnp.asarray(signature), jnp.asarray(displacement))
    expected = getattr(ps, name)(signature, displacement, 2, 3, **kwargs)
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-7)
    grads = jax.jit(jax.grad(lambda a, b: function(a, b).sum(), argnums=(0, 1)))(jnp.asarray(signature), jnp.asarray(displacement))
    expected_grads = getattr(ps, name + "_backprop")(np.ones_like(expected), signature, displacement, 2, 3, **kwargs)
    for actual, expected in zip(grads, expected_grads):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_jax_empty_join(degree):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    import pysiglib.jax_api as pj
    displacement = jnp.zeros((2, 0, 2), dtype=jnp.float32)
    for is_log in (False, True):
        linear = pj.linear_branched_log_sig if is_log else pj.linear_branched_sig
        join = pj.branched_log_sig_join if is_log else pj.branched_sig_join
        signature = linear(displacement, 2, degree, **({"planar": True, "method": 3} if is_log else {"planar": True}))
        function = lambda a, b: join(a, b, 2, degree, **({} if is_log else {"planar": True}))
        result = jax.jit(function)(signature, displacement)
        assert result.shape == signature.shape
        grads = jax.jit(jax.grad(lambda a, b: function(a, b).sum(), argnums=(0, 1)))(signature, displacement)
        assert grads[0].shape == signature.shape
        assert grads[1].shape == displacement.shape


@pytest.mark.parametrize("prepend", [False, True])
def test_join_threading_and_views(prepend):
    rng = np.random.default_rng(17)
    displacement = rng.normal(size=(3, 2, 4))[..., ::2] * 0.1
    for name in ("branched_sig", "branched_log_sig"):
        kwargs = {"planar": True, "method": 3} if name == "branched_log_sig" else {"planar": True}
        signature = getattr(ps, "linear_" + name)(displacement, 2, 3, **kwargs)
        signature = np.repeat(signature, 2, axis=-1)[..., ::2]
        join_kwargs = {"prepend": prepend} if name == "branched_log_sig" else {"planar": True, "prepend": prepend}
        join = getattr(ps, name + "_join")
        expected = join(signature, displacement, 2, 3, **join_kwargs)
        actual = join(signature, displacement, 2, 3, n_jobs=2, **join_kwargs)
        np.testing.assert_allclose(actual, expected)
        backprop = getattr(ps, name + "_join_backprop")
        weights = rng.normal(size=expected.shape)
        expected_grads = backprop(weights, signature, displacement, 2, 3, **join_kwargs)
        actual_grads = backprop(weights, signature, displacement, 2, 3, n_jobs=2, **join_kwargs)
        for actual, expected in zip(actual_grads, expected_grads):
            np.testing.assert_allclose(actual, expected)
