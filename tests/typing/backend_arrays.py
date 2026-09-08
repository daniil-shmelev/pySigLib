"""Static assertions: run with pyright or basedpyright."""

from typing_extensions import assert_type
import numpy as np
import torch
import jax
import jax.numpy as jnp
import pysiglib as numpy_api
import pysiglib.torch_api as torch_api
import pysiglib.jax_api as jax_api

xn = np.ones((4, 2), dtype=np.float64)
xt = torch.ones((4, 2), dtype=torch.float64)
xj = jnp.ones((4, 2), dtype=jnp.float32)

assert_type(numpy_api.sig(xn, 2), np.ndarray)
assert_type(torch_api.sig(xt, 2), torch.Tensor)
assert_type(jax_api.sig(xj, 2), jax.Array)
assert_type(numpy_api.log_sig(xn, 2), np.ndarray)
assert_type(torch_api.log_sig(xt, 2), torch.Tensor)
assert_type(jax_api.log_sig(xj, 2), jax.Array)
assert_type(numpy_api.sig_kernel(xn, xn, dyadic_order=0), np.ndarray)
assert_type(torch_api.sig_kernel(xt, xt, dyadic_order=0), torch.Tensor)
assert_type(jax_api.sig_kernel(xj, xj, dyadic_order=0), jax.Array)
assert_type(numpy_api.extract_sig_coef(numpy_api.sig(xn, 2), (0,), 2), np.ndarray)
assert_type(torch_api.extract_sig_coef(torch_api.sig(xt, 2), (0,), 2), torch.Tensor)
assert_type(jax_api.extract_sig_coef(jax_api.sig(xj, 2), (0,), 2), jax.Array)

ns = numpy_api.SigStream(2, 2)
ts = torch_api.SigStream(2, 2)
js = jax_api.SigStream(2, 2)
ns.push_batch(xn)
ts.push_batch(xt)
js.push_batch(xj)
assert_type(ns.sig_all(), np.ndarray)
assert_type(ts.sig_all(), torch.Tensor)
assert_type(js.sig_all(), jax.Array)
assert_type(numpy_api.SigWindowStream(2, 2, 3).sig(), np.ndarray)
assert_type(torch_api.SigWindowStream(2, 2, 3).sig(), torch.Tensor)
assert_type(jax_api.SigWindowStream(2, 2, 3).sig(), jax.Array)
assert_type(
    numpy_api.LinearKernel()(numpy_api.Context(), xn[None], xn[None]), np.ndarray
)
assert_type(
    torch_api.LinearKernel()(torch_api.Context(), xt[None], xt[None]), torch.Tensor
)
assert_type(jax_api.LinearKernel()(xj[None], xj[None]), jax.Array)

# These ignores must remain necessary: foreign array types are rejected statically.
numpy_api.sig(xt, 2)  # pyright: ignore[reportArgumentType]
torch_api.sig(xn, 2)  # pyright: ignore[reportArgumentType]
jax_api.sig(xt, 2)  # pyright: ignore[reportArgumentType]
ns.push_batch(xt)  # pyright: ignore[reportArgumentType]
ts.push_batch(xj)  # pyright: ignore[reportArgumentType]
js.push_batch(xn)  # pyright: ignore[reportArgumentType]
