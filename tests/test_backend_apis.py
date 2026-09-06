"""Contracts of the three public APIs, including installations without extras."""

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import pysiglib


@pytest.mark.parametrize(
    "backend,blocked",
    [
        ("numpy", ["torch", "jax", "jaxlib"]),
        ("torch", ["jax", "jaxlib"]),
        ("jax", ["torch"]),
    ],
)
def test_backend_import_and_execution_without_other_frameworks(backend, blocked):
    if backend != "numpy":
        pytest.importorskip(backend)
    script = """
import importlib.abc
import sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockFrameworks())
import numpy as np
import pysiglib
x = np.random.default_rng(7).normal(size=(2, 5, 2)) * 0.1
if BACKEND == 'numpy':
    api = pysiglib
    array_type = np.ndarray
elif BACKEND == 'torch':
    import torch
    import pysiglib.torch_api as api
    x = torch.tensor(x, requires_grad=True)
    array_type = torch.Tensor
else:
    import jax
    import jax.numpy as jnp
    import pysiglib.jax_api as api
    x = jnp.asarray(x)
    array_type = jax.Array
s = api.sig(x, 3)
assert isinstance(s, array_type)
assert isinstance(api.extract_sig_coef(s, (0,), 2), array_type)
assert isinstance(api.sig_coef(x, [(0,), (1, 0)]), array_type)
api.prepare_log_sig(2, 3, method=2)
assert isinstance(api.log_sig(x, 3), array_type)
for kernel in (api.LinearKernel(), api.RBFKernel(1.0), api.PolynomialKernel()):
    for method in ({'dyadic_order': 1}, {'method': 'polynomial', 'order': 3}):
        assert isinstance(api.sig_kernel(x, x, static_kernel=kernel, **method), array_type)
        assert isinstance(api.sig_kernel_gram(x, x, static_kernel=kernel, **method), array_type)
assert isinstance(api.sig_mmd(x, x, dyadic_order=0), array_type)
api.prepare_branched_sig(2, 2)
assert isinstance(api.branched_sig(x, 2), array_type)
assert isinstance(api.branched_sig_kernel(x, x, 2, 0), array_type)
stream = api.SigStream(2, 3)
stream.push_batch(x)
assert isinstance(stream.sig_all(), array_type)
window = api.SigWindowStream(2, 3, 3)
window.push_batch(x)
assert isinstance(window.sig(), array_type)
if BACKEND == 'torch':
    s.sum().backward()
    assert x.grad is not None
elif BACKEND == 'jax':
    assert isinstance(jax.jit(lambda a: api.sig(a, 3))(x), array_type)
    assert isinstance(jax.grad(lambda a: api.sig(a, 3).sum())(x), array_type)
    assert isinstance(jax.vmap(lambda a: api.sig(a, 3))(x), array_type)
assert not any(name in sys.modules for name in BLOCKED)
"""
    script = f"BACKEND = {backend!r}\nBLOCKED = {blocked!r}\n" + textwrap.dedent(script)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_apis_reject_foreign_arrays():
    import torch
    import pysiglib.torch_api as torch_api

    jax = pytest.importorskip("jax")
    import pysiglib.jax_api as jax_api

    values = [np.ones((4, 2)), torch.ones((4, 2)), jax.numpy.ones((4, 2))]
    for i, api in enumerate([pysiglib, torch_api, jax_api]):
        for j, value in enumerate(values):
            if i == j:
                continue
            with pytest.raises(TypeError, match="matching pysiglib backend API"):
                api.sig(value, 2)
            with pytest.raises(TypeError, match="matching pysiglib backend API"):
                api.SigStream(2, 2).push_batch(value)


def test_backend_docstrings_describe_their_arrays():
    import pysiglib.torch_api as torch_api

    jax_api = pytest.importorskip("pysiglib.jax_api")
    for api, name in [
        (pysiglib, "numpy.ndarray"),
        (torch_api, "torch.Tensor"),
        (jax_api, "jax.Array"),
    ]:
        doc = inspect.getdoc(api.sig)
        assert f":type path: {name}" in doc
        assert f":rtype: {name}" in doc


def test_numpy_kernel_backprop_matches_finite_differences():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(2, 1, 4, 2)) * 0.1
    for method in ({"dyadic_order": 1}, {"method": "polynomial", "order": 3}):
        kernel = pysiglib.RBFKernel(1.0)
        value = pysiglib.sig_kernel(x, y, static_kernel=kernel, **method)
        dx, dy = pysiglib.sig_kernel_backprop(
            np.ones_like(value), x, y, static_kernel=kernel, right_deriv=True, **method
        )
        for array, grad in [(x, dx), (y, dy)]:
            for idx in np.ndindex(array.shape):
                old = array[idx]
                array[idx] = old + 1e-5
                plus = pysiglib.sig_kernel(x, y, static_kernel=kernel, **method).sum()
                array[idx] = old - 1e-5
                minus = pysiglib.sig_kernel(x, y, static_kernel=kernel, **method).sum()
                array[idx] = old
                np.testing.assert_allclose(
                    grad[idx], (plus - minus) / 2e-5, rtol=1e-5, atol=1e-7
                )
