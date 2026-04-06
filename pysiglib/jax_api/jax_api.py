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
from ._ffi import (
    ensure_registered,
    sig_backprop_ffi_call,
    sig_ffi_call,
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

    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if not isinstance(time_aug, bool):
        raise TypeError("time_aug must be a bool")
    if not isinstance(lead_lag, bool):
        raise TypeError("lead_lag must be a bool")
    if not isinstance(horner, bool):
        raise TypeError("horner must be a bool")
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an int")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    return _sig(path, degree, time_aug, lead_lag, float(end_time), horner, n_jobs)


sig.__doc__ = sig_forward.__doc__
