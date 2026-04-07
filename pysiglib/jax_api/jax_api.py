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

from __future__ import annotations

from functools import partial

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:
    raise ImportError(
        "pysiglib.jax_api requires JAX. Install jax/jaxlib before importing this module."
    ) from exc

from ..sig import sig as sig_forward
from ..sig import sig_combine as sig_combine_forward
from ..transform_path import transform_path as transform_path_forward
from ..log_sig import sig_to_log_sig as sig_to_log_sig_forward
from ..logsig_to_sig import logsig_to_sig as logsig_to_sig_forward
from ..log_sig import log_sig as log_sig_forward
from ..log_sig_combine import log_sig_combine as log_sig_combine_forward
from ..branched_sig import branched_sig as branched_sig_forward
from ..branched_sig import branched_sig_combine as branched_sig_combine_forward
from ..words import word_to_idx
from ..param_checks import check_type, check_non_neg, check_word_or_word_list, parse_dyadic_order
from ._ffi import (
    _augmented_dim,
    ensure_registered,
    sig_backprop_ffi_call,
    sig_ffi_call,
    sig_combine_ffi_call,
    sig_combine_backprop_ffi_call,
    transform_path_ffi_call,
    transform_path_backprop_ffi_call,
    sig_to_log_sig_ffi_call,
    sig_to_log_sig_backprop_ffi_call,
    log_sig_combine_ffi_call,
    log_sig_combine_backprop_ffi_call,
    sig_kernel_pde_ffi_call,
    sig_kernel_pde_backprop_ffi_call,
    logsig_to_sig_ffi_call,
    logsig_to_sig_backprop_ffi_call,
    log_sig_from_path_ffi_call,
    log_sig_from_path_backprop_ffi_call,
    branched_sig_ffi_call,
    branched_sig_backprop_ffi_call,
    branched_sig_combine_ffi_call,
    branched_sig_combine_backprop_ffi_call,
)


def _validate_shape(path) -> None:
    if path.ndim not in (2, 3):
        raise ValueError(f"path.shape must have length 2 or 3, got {path.ndim}.")
    if path.shape[-1] == 0:
        raise ValueError("path must have at least one channel.")


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def _sig(path, degree, time_aug, lead_lag, end_time, horner, n_jobs):
    return sig_ffi_call(path, degree, time_aug, lead_lag, end_time, horner, n_jobs)


def _sig_fwd(path, degree, time_aug, lead_lag, end_time, horner, n_jobs):
    sig_ = sig_ffi_call(path, degree, time_aug, lead_lag, end_time, horner, n_jobs)
    return sig_, (path, sig_)


def _sig_bwd(degree, time_aug, lead_lag, end_time, horner, n_jobs, residual, cotangent):
    del horner
    path, sig_ = residual
    grad = sig_backprop_ffi_call(path, sig_, cotangent, degree, time_aug, lead_lag, end_time, n_jobs)
    return (grad,)


_sig.defvjp(_sig_fwd, _sig_bwd)


def sig(
    path,
    degree: int,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    horner: bool = True,
    n_jobs: int = 1,
):
    ensure_registered()

    path = jnp.asarray(path)
    _validate_shape(path)

    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(horner, "horner", bool)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    return _sig(path, degree, time_aug, lead_lag, end_time, horner, n_jobs)


sig.__doc__ = sig_forward.__doc__


# ---------------------------------------------------------------------------
# sig_combine
# ---------------------------------------------------------------------------

def _validate_sig_shape(arr, name="signature"):
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name}.shape must have length 1 or 2, got {arr.ndim}.")


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _sig_combine(sig1, sig2, dimension, degree, n_jobs):
    return sig_combine_ffi_call(sig1, sig2, dimension, degree, n_jobs)


def _sig_combine_fwd(sig1, sig2, dimension, degree, n_jobs):
    result = sig_combine_ffi_call(sig1, sig2, dimension, degree, n_jobs)
    return result, (sig1, sig2)


def _sig_combine_bwd(dimension, degree, n_jobs, residual, cotangent):
    sig1, sig2 = residual
    grad_sig1, grad_sig2 = sig_combine_backprop_ffi_call(
        cotangent, sig1, sig2, dimension, degree, n_jobs
    )
    return (grad_sig1, grad_sig2)


_sig_combine.defvjp(_sig_combine_fwd, _sig_combine_bwd)


def sig_combine(
    sig1,
    sig2,
    dimension: int,
    degree: int,
    time_aug: bool = False,
    lead_lag: bool = False,
    n_jobs: int = 1,
):
    ensure_registered()

    sig1 = jnp.asarray(sig1)
    sig2 = jnp.asarray(sig2)
    _validate_sig_shape(sig1, "sig1")
    _validate_sig_shape(sig2, "sig2")

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    return _sig_combine(sig1, sig2, _augmented_dim(dimension, time_aug, lead_lag), degree, n_jobs)


sig_combine.__doc__ = sig_combine_forward.__doc__


# ---------------------------------------------------------------------------
# transform_path
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _transform_path(path, time_aug, lead_lag, end_time, n_jobs):
    return transform_path_ffi_call(path, time_aug, lead_lag, end_time, n_jobs)


def _transform_path_fwd(path, time_aug, lead_lag, end_time, n_jobs):
    result = transform_path_ffi_call(path, time_aug, lead_lag, end_time, n_jobs)
    # Save original dimension and length for backprop (not the path itself)
    return result, (path.shape[-2], path.shape[-1])


def _transform_path_bwd(time_aug, lead_lag, end_time, n_jobs, residual, cotangent):
    orig_length, orig_dimension = residual
    grad = transform_path_backprop_ffi_call(
        cotangent, orig_dimension, orig_length, time_aug, lead_lag, end_time, n_jobs
    )
    return (grad,)


_transform_path.defvjp(_transform_path_fwd, _transform_path_bwd)


def transform_path(
    path,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
):
    ensure_registered()

    path = jnp.asarray(path)
    _validate_shape(path)

    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    return _transform_path(path, time_aug, lead_lag, end_time, n_jobs)


transform_path.__doc__ = transform_path_forward.__doc__


# ---------------------------------------------------------------------------
# sig_to_log_sig
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _sig_to_log_sig(sig_arr, dimension, degree, method, n_jobs):
    return sig_to_log_sig_ffi_call(sig_arr, dimension, degree, method, n_jobs)


def _sig_to_log_sig_fwd(sig_arr, dimension, degree, method, n_jobs):
    result = sig_to_log_sig_ffi_call(sig_arr, dimension, degree, method, n_jobs)
    return result, (sig_arr,)


def _sig_to_log_sig_bwd(dimension, degree, method, n_jobs, residual, cotangent):
    sig_arr, = residual
    grad = sig_to_log_sig_backprop_ffi_call(sig_arr, cotangent, dimension, degree, method, n_jobs)
    return (grad,)


_sig_to_log_sig.defvjp(_sig_to_log_sig_fwd, _sig_to_log_sig_bwd)


def sig_to_log_sig(
    sig_arr,
    dimension: int,
    degree: int,
    time_aug: bool = False,
    lead_lag: bool = False,
    method: int = 1,
    n_jobs: int = 1,
):
    ensure_registered()

    sig_arr = jnp.asarray(sig_arr)
    _validate_sig_shape(sig_arr, "sig")

    if method not in (0, 1, 2, 3):
        raise ValueError(f"method must be 0, 1, 2, or 3, got {method}")
    if method == 3:
        raise NotImplementedError(
            "method=3 is not supported in the JAX API. "
            "Use method=1 or method=2 instead."
        )

    aug_dim = _augmented_dim(dimension, time_aug, lead_lag)
    return _sig_to_log_sig(sig_arr, aug_dim, degree, method, n_jobs)


sig_to_log_sig.__doc__ = sig_to_log_sig_forward.__doc__


# ---------------------------------------------------------------------------
# logsig_to_sig (tensor exponential)
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _logsig_to_sig(log_sig_arr, dimension, degree, method, n_jobs):
    return logsig_to_sig_ffi_call(log_sig_arr, dimension, degree, method, n_jobs)


def _logsig_to_sig_fwd(log_sig_arr, dimension, degree, method, n_jobs):
    result = logsig_to_sig_ffi_call(log_sig_arr, dimension, degree, method, n_jobs)
    return result, (log_sig_arr,)


def _logsig_to_sig_bwd(dimension, degree, method, n_jobs, residual, cotangent):
    log_sig_arr, = residual
    grad = logsig_to_sig_backprop_ffi_call(log_sig_arr, cotangent, dimension, degree, method, n_jobs)
    return (grad,)


_logsig_to_sig.defvjp(_logsig_to_sig_fwd, _logsig_to_sig_bwd)


def logsig_to_sig(
    log_sig_arr,
    dimension: int,
    degree: int,
    time_aug: bool = False,
    lead_lag: bool = False,
    method: int = 1,
    n_jobs: int = 1,
):
    ensure_registered()

    log_sig_arr = jnp.asarray(log_sig_arr)
    _validate_sig_shape(log_sig_arr, "log_sig")

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(method, "method", int)
    if method not in (0, 1, 2):
        raise ValueError("method must be 0, 1, or 2")

    aug_dim = _augmented_dim(dimension, time_aug, lead_lag)
    return _logsig_to_sig(log_sig_arr, aug_dim, degree, method, n_jobs)


logsig_to_sig.__doc__ = logsig_to_sig_forward.__doc__


# ---------------------------------------------------------------------------
# log_sig
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _log_sig_from_path(path, dimension, degree, n_jobs):
    return log_sig_from_path_ffi_call(path, dimension, degree, n_jobs)


def _log_sig_from_path_fwd(path, dimension, degree, n_jobs):
    result = log_sig_from_path_ffi_call(path, dimension, degree, n_jobs)
    return result, (path,)


def _log_sig_from_path_bwd(dimension, degree, n_jobs, residual, cotangent):
    path, = residual
    grad = log_sig_from_path_backprop_ffi_call(cotangent, path, dimension, degree, n_jobs)
    return (grad,)


_log_sig_from_path.defvjp(_log_sig_from_path_fwd, _log_sig_from_path_bwd)


def log_sig(
    path,
    degree: int,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    method: int = 1,
    n_jobs: int = 1,
):
    ensure_registered()

    path = jnp.asarray(path)
    _validate_shape(path)

    if method == 3:
        if time_aug or lead_lag:
            path = transform_path(path, time_aug, lead_lag, end_time, n_jobs)
        aug_dim = path.shape[-1]
        return _log_sig_from_path(path, aug_dim, degree, n_jobs)

    dimension = path.shape[-1]
    sig_ = sig(path, degree, time_aug, lead_lag, end_time, True, n_jobs)
    return sig_to_log_sig(sig_, dimension, degree, time_aug, lead_lag, method, n_jobs)


log_sig.__doc__ = log_sig_forward.__doc__


# ---------------------------------------------------------------------------
# log_sig_combine
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _log_sig_combine(ls1, ls2, dimension, degree, n_jobs):
    return log_sig_combine_ffi_call(ls1, ls2, dimension, degree, n_jobs)


def _log_sig_combine_fwd(ls1, ls2, dimension, degree, n_jobs):
    result = log_sig_combine_ffi_call(ls1, ls2, dimension, degree, n_jobs)
    return result, (ls1, ls2)


def _log_sig_combine_bwd(dimension, degree, n_jobs, residual, cotangent):
    ls1, ls2 = residual
    grad1, grad2 = log_sig_combine_backprop_ffi_call(cotangent, ls1, ls2, dimension, degree, n_jobs)
    return (grad1, grad2)


_log_sig_combine.defvjp(_log_sig_combine_fwd, _log_sig_combine_bwd)


def log_sig_combine(
    log_sig1,
    log_sig2,
    dimension: int,
    degree: int,
    time_aug: bool = False,
    lead_lag: bool = False,
    n_jobs: int = 1,
):
    ensure_registered()

    log_sig1 = jnp.asarray(log_sig1)
    log_sig2 = jnp.asarray(log_sig2)
    _validate_sig_shape(log_sig1, "log_sig1")
    _validate_sig_shape(log_sig2, "log_sig2")

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    aug_dim = _augmented_dim(dimension, time_aug, lead_lag)
    return _log_sig_combine(log_sig1, log_sig2, aug_dim, degree, n_jobs)


log_sig_combine.__doc__ = log_sig_combine_forward.__doc__


# ---------------------------------------------------------------------------
# sig_kernel PDE solver (FFI)
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def _sig_kernel_pde(gram, dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs):
    return sig_kernel_pde_ffi_call(gram, dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs)


def _sig_kernel_pde_fwd(gram, dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs):
    # Always compute grid for backward
    k_grid = sig_kernel_pde_ffi_call(gram, dimension, dyadic_order_1, dyadic_order_2, True, n_jobs)
    if return_grid:
        result = k_grid
    else:
        if k_grid.ndim == 2:
            result = k_grid[-1, -1]
        else:
            result = k_grid[:, -1, -1]
    return result, (gram, k_grid)


def _sig_kernel_pde_bwd(dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs, residual, cotangent):
    gram, k_grid = residual
    grad_gram = sig_kernel_pde_backprop_ffi_call(
        gram, cotangent, k_grid, dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs
    )
    return (grad_gram,)


_sig_kernel_pde.defvjp(_sig_kernel_pde_fwd, _sig_kernel_pde_bwd)


# ---------------------------------------------------------------------------
# sig_kernel (public API composing static kernel + PDE solve)
# ---------------------------------------------------------------------------

def sig_kernel(
    path1,
    path2,
    dyadic_order,
    static_kernel=None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    return_grid: bool = False,
    normalize: bool = False,
):
    """Compute signature kernel between paired paths using JAX.

    This composes the static kernel evaluation (pure JAX) with the
    PDE solver (C++ FFI). Fully differentiable via JAX autodiff.
    """
    ensure_registered()

    path1 = jnp.asarray(path1)
    path2 = jnp.asarray(path2)

    if path1.ndim not in (2, 3) or path2.ndim not in (2, 3):
        raise ValueError("path1 and path2 must have ndim 2 or 3.")
    if path1.ndim != path2.ndim:
        raise ValueError("path1 and path2 must have the same ndim.")

    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")

    if time_aug or lead_lag:
        path1 = transform_path(path1, time_aug, lead_lag, end_time, n_jobs)
        path2 = transform_path(path2, time_aug, lead_lag, end_time, n_jobs)

    if path1.ndim == 2:
        path1 = path1[None, :, :]
        path2 = path2[None, :, :]
        squeeze = True
    else:
        squeeze = False

    if static_kernel is None:
        from .static_kernels_jax import LinearKernel as _DefaultKernel
        gram = _DefaultKernel()(path1, path2)
    else:
        gram = static_kernel(path1, path2)

    dimension = path1.shape[-1]
    do1, do2 = parse_dyadic_order(dyadic_order)

    if return_grid:
        result = _sig_kernel_pde(gram, dimension, do1, do2, True, n_jobs)
    else:
        k_grid = _sig_kernel_pde(gram, dimension, do1, do2, True, n_jobs)
        result = k_grid[:, -1, -1]

    if normalize:
        k1 = sig_kernel(path1, path1, dyadic_order, static_kernel, n_jobs=n_jobs)
        k2 = sig_kernel(path2, path2, dyadic_order, static_kernel, n_jobs=n_jobs)
        result = result / jnp.sqrt(jnp.clip(k1 * k2, 1e-30))

    if squeeze:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# sig_kernel_gram (pure Python composition)
# ---------------------------------------------------------------------------

def sig_kernel_gram(
    path1,
    path2,
    dyadic_order,
    static_kernel=None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    max_batch: int = -1,
    return_grid: bool = False,
    normalize: bool = False,
):
    """Compute Gram matrix of signature kernels using JAX."""
    ensure_registered()

    path1 = jnp.asarray(path1)
    path2 = jnp.asarray(path2)

    if path1.ndim != 3 or path2.ndim != 3:
        raise ValueError("path1 and path2 must be 3D arrays (batch, length, dimension)")

    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(max_batch, "max_batch", int)
    if max_batch == 0 or max_batch < -1:
        raise ValueError("max_batch must be a positive integer or -1")
    if normalize and return_grid:
        raise ValueError("normalize=True cannot be used with return_grid=True")

    batch1 = path1.shape[0]
    batch2 = path2.shape[0]

    if time_aug or lead_lag:
        path1 = transform_path(path1, time_aug, lead_lag, end_time, n_jobs)
        path2 = transform_path(path2, time_aug, lead_lag, end_time, n_jobs)

    def _row(p1_single):
        p1_batch = jnp.broadcast_to(p1_single[None], (batch2,) + p1_single.shape)
        return sig_kernel(p1_batch, path2, dyadic_order, static_kernel,
                          n_jobs=n_jobs, return_grid=return_grid)

    res = jax.lax.map(_row, path1)

    if normalize:
        d1 = sig_kernel(path1, path1, dyadic_order, static_kernel, n_jobs=n_jobs)
        d2 = sig_kernel(path2, path2, dyadic_order, static_kernel, n_jobs=n_jobs)
        res = res / jnp.sqrt(jnp.clip(d1[:, None] * d2[None, :], 1e-30))

    return res


# ---------------------------------------------------------------------------
# Signature kernel metrics (pure Python)
# ---------------------------------------------------------------------------

def sig_score(
    sample,
    y,
    dyadic_order,
    lam: float = 1.0,
    static_kernel=None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    max_batch: int = -1,
):
    """Compute signature kernel score using JAX."""
    sample = jnp.asarray(sample)
    y = jnp.asarray(y)
    if y.ndim == 2:
        y = y[None, :, :]

    B = sample.shape[0]
    if B < 2:
        raise ValueError(f"sig_score requires at least 2 sample paths (got {B}).")

    if time_aug or lead_lag:
        sample = transform_path(sample, time_aug, lead_lag, end_time, n_jobs)
        y = transform_path(y, time_aug, lead_lag, end_time, n_jobs)

    xx = sig_kernel_gram(sample, sample, dyadic_order, static_kernel, n_jobs=n_jobs, max_batch=max_batch)
    xy = sig_kernel_gram(sample, y, dyadic_order, static_kernel, n_jobs=n_jobs, max_batch=max_batch)

    xx_sum = (jnp.sum(xx) - jnp.trace(xx)) / (B * (B - 1.))
    xy_sum = jnp.sum(xy, axis=0) * (2. / B)

    return lam * xx_sum - xy_sum


def expected_sig_score(
    sample1,
    sample2,
    dyadic_order,
    lam: float = 1.0,
    static_kernel=None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    max_batch: int = -1,
):
    """Compute expected signature kernel score using JAX."""
    res = sig_score(sample1, sample2, dyadic_order, lam, static_kernel, time_aug, lead_lag, end_time, n_jobs, max_batch)
    return jnp.mean(res, axis=0, keepdims=True)


def sig_mmd(
    sample1,
    sample2,
    dyadic_order,
    static_kernel=None,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    n_jobs: int = 1,
    max_batch: int = -1,
):
    """Compute signature kernel MMD using JAX."""
    sample1 = jnp.asarray(sample1)
    sample2 = jnp.asarray(sample2)

    m = sample1.shape[0]
    n = sample2.shape[0]
    if m < 2:
        raise ValueError(f"sig_mmd requires at least 2 paths in sample1 (got {m}).")
    if n < 2:
        raise ValueError(f"sig_mmd requires at least 2 paths in sample2 (got {n}).")

    if time_aug or lead_lag:
        sample1 = transform_path(sample1, time_aug, lead_lag, end_time, n_jobs)
        sample2 = transform_path(sample2, time_aug, lead_lag, end_time, n_jobs)

    xx = sig_kernel_gram(sample1, sample1, dyadic_order, static_kernel, n_jobs=n_jobs, max_batch=max_batch)
    xy = sig_kernel_gram(sample1, sample2, dyadic_order, static_kernel, n_jobs=n_jobs, max_batch=max_batch)
    yy = sig_kernel_gram(sample2, sample2, dyadic_order, static_kernel, n_jobs=n_jobs, max_batch=max_batch)

    xx_sum = (jnp.sum(xx) - jnp.trace(xx)) / (m * (m - 1))
    xy_sum = 2. * jnp.mean(xy)
    yy_sum = (jnp.sum(yy) - jnp.trace(yy)) / (n * (n - 1))

    return xx_sum - xy_sum + yy_sum


# ---------------------------------------------------------------------------
# sig_coef (pure JAX composition — compute sig then index)
# ---------------------------------------------------------------------------

def sig_coef(
    path,
    words,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    prefixes: bool = False,
    n_jobs: int = 1,
):
    """Compute signature coefficients at specific words using JAX.

    Computes the full signature via ``sig()`` then extracts the requested
    coefficients. This is fully differentiable through JAX autodiff.

    :param path: Path or batch of paths, shape ``(length, dim)`` or ``(batch, length, dim)``.
    :param words: Word or list of words, e.g. ``(1, 2)`` or ``[(0,), (1, 0)]``.
    :param time_aug: Whether to apply time augmentation.
    :param lead_lag: Whether to apply lead-lag transformation.
    :param end_time: End time for time augmentation.
    :param prefixes: If True, also return coefficients at all prefixes of each word.
    :param n_jobs: Number of threads for CPU computation.
    :return: Signature coefficients at the requested words.
    """
    ensure_registered()

    path = jnp.asarray(path)
    _validate_shape(path)

    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(prefixes, "prefixes", bool)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    dimension = path.shape[-1]
    aug_dim = _augmented_dim(dimension, time_aug, lead_lag)

    words = check_word_or_word_list(words, aug_dim, "words")

    degree = max(len(w) for w in words)

    sig_ = sig(path, degree, time_aug, lead_lag, end_time, True, n_jobs)

    idx = []
    for w in words:
        if prefixes:
            for k in range(1, len(w) + 1):
                idx.append(word_to_idx(w[:k], aug_dim))
        else:
            idx.append(word_to_idx(w, aug_dim))
    return sig_[..., jnp.array(idx)]


# ---------------------------------------------------------------------------
# branched_sig
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def _branched_sig(path, max_nodes, time_aug, lead_lag, end_time, n_jobs):
    return branched_sig_ffi_call(path, max_nodes, time_aug, lead_lag, end_time, n_jobs)


def _branched_sig_fwd(path, max_nodes, time_aug, lead_lag, end_time, n_jobs):
    bsig = branched_sig_ffi_call(path, max_nodes, time_aug, lead_lag, end_time, n_jobs)
    return bsig, (path, bsig)


def _branched_sig_bwd(max_nodes, time_aug, lead_lag, end_time, n_jobs, residual, cotangent):
    path, bsig = residual
    grad = branched_sig_backprop_ffi_call(path, bsig, cotangent, max_nodes, time_aug, lead_lag, end_time, n_jobs)
    return (grad,)


_branched_sig.defvjp(_branched_sig_fwd, _branched_sig_bwd)


def branched_sig(
    path,
    degree: int,
    n_jobs: int = 1,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
):
    ensure_registered()

    path = jnp.asarray(path)
    _validate_shape(path)

    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    return _branched_sig(path, degree, time_aug, lead_lag, end_time, n_jobs)


branched_sig.__doc__ = branched_sig_forward.__doc__


# ---------------------------------------------------------------------------
# branched_sig_combine
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _branched_sig_combine(bsig1, bsig2, dimension, max_nodes, n_jobs):
    return branched_sig_combine_ffi_call(bsig1, bsig2, dimension, max_nodes, n_jobs)


def _branched_sig_combine_fwd(bsig1, bsig2, dimension, max_nodes, n_jobs):
    result = branched_sig_combine_ffi_call(bsig1, bsig2, dimension, max_nodes, n_jobs)
    return result, (bsig1, bsig2)


def _branched_sig_combine_bwd(dimension, max_nodes, n_jobs, residual, cotangent):
    bsig1, bsig2 = residual
    grad1, grad2 = branched_sig_combine_backprop_ffi_call(
        cotangent, bsig1, bsig2, dimension, max_nodes, n_jobs
    )
    return (grad1, grad2)


_branched_sig_combine.defvjp(_branched_sig_combine_fwd, _branched_sig_combine_bwd)


def branched_sig_combine(
    bsig1,
    bsig2,
    dimension: int,
    degree: int,
    n_jobs: int = 1,
):
    ensure_registered()

    bsig1 = jnp.asarray(bsig1)
    bsig2 = jnp.asarray(bsig2)
    _validate_sig_shape(bsig1, "bsig1")
    _validate_sig_shape(bsig2, "bsig2")

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    return _branched_sig_combine(bsig1, bsig2, dimension, degree, n_jobs)


branched_sig_combine.__doc__ = branched_sig_combine_forward.__doc__
