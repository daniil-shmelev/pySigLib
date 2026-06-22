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

from ctypes import POINTER, byref, c_uint64
from typing import Union

import numpy as np
import torch

from .data_handlers import PathInputHandler, SigOutputHandler
from .dtypes import (
    CPSIG_FREE_VOLTERRA_SIG,
    CPSIG_PREPARE_VOLTERRA_SIG,
    CPSIG_VOLTERRA_SIG,
    DTYPES,
)
from .error_codes import err_msg
from .param_checks import check_type, check_non_neg, check_n_jobs
from .sig_length import sig_length


def _numpy_array(value, name, dtype, ndim):
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(name + " must have rank " + str(ndim) + ", got rank " + str(arr.ndim))
    return np.ascontiguousarray(arr)


def _numpy_ptr(arr, dtype_name):
    return arr.ctypes.data_as(POINTER(DTYPES[dtype_name]))


def _rank(value):
    if hasattr(value, "ndim"):
        return value.ndim
    return np.ndim(value)


def _is_finite(arr):
    return bool(np.isfinite(arr).all())


def _value_dtype(value):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.float32:
            return np.dtype(np.float32)
        if value.dtype == torch.float64:
            return np.dtype(np.float64)
        return None
    arr = np.asarray(value)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr.dtype
    return None


def _resolve_prepare_dtype(dtype, kernel):
    if dtype is None:
        dtypes = [
            item for item in (
                _value_dtype(kernel.Lambda),
                _value_dtype(kernel.A),
                _value_dtype(kernel.b),
            ) if item is not None
        ]
        dtype = np.result_type(*dtypes) if dtypes else np.float64
    elif dtype == torch.float32:
        dtype = np.float32
    elif dtype == torch.float64:
        dtype = np.float64

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return dtype, "float32"
    if dtype == np.dtype(np.float64):
        return dtype, "float64"
    raise ValueError("dtype must be float32 or float64")


class VolterraKernel:
    """Base class for Volterra kernels."""

    def __init__(self):
        self._prepared_volterra_sig = {}

    def prepare(
            self,
            degree: int,
            *,
            dt: float,
            readout_lag: float = 0.,
            scheme: str = "exact",
            dtype=None,
    ):
        key = self._prepared_key(degree, dt, readout_lag, scheme, dtype)
        cache = self._prepared_volterra_sig
        if key in cache:
            return None

        cache[key] = _prepare_volterra_sig_impl(
            degree,
            self,
            dt=dt,
            readout_lag=readout_lag,
            scheme=scheme,
            dtype=dtype,
        )
        return None

    def clear_cache(self):
        cache = self._prepared_volterra_sig
        for entry in cache.values():
            self._free_prepared_entry(entry)
        cache.clear()

    def _prepared_key(self, degree, dt, readout_lag, scheme, dtype):
        check_type(degree, "degree", int)
        check_non_neg(degree, "degree")
        check_type(dt, "dt", float)
        check_type(readout_lag, "readout_lag", float)
        check_type(scheme, "scheme", str)
        if scheme != "exact":
            raise ValueError("scheme must be 'exact'")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if readout_lag < 0:
            raise ValueError("readout_lag must be non-negative")
        _, dtype_name = _resolve_prepare_dtype(dtype, self)
        return degree, dt, readout_lag, scheme, dtype_name

    def _get_prepared(self, degree, dt, readout_lag, scheme, dtype_name):
        cache = self._prepared_volterra_sig
        prepared = cache.get((degree, dt, readout_lag, scheme, dtype_name))
        if prepared is not None:
            return prepared
        for cache_key in cache:
            if cache_key[:4] == (degree, dt, readout_lag, scheme):
                raise ValueError("path dtype must match the prepared Volterra signature dtype")
        return None

    def _free_prepared_entry(self, entry):
        handle = entry["handle"]
        if handle == 0:
            return
        err_code = CPSIG_FREE_VOLTERRA_SIG[entry["dtype"]](handle)
        entry["handle"] = 0
        if err_code:
            raise Exception("Error in pysiglib.VolterraKernel.clear_cache: " + err_msg(err_code))

    def _volterra_sig(self, data, prepared, scalar_term, n_jobs):
        if data.data_dimension != prepared["dimension"]:
            raise ValueError("path.shape[-1] must match the prepared Volterra signature path dimension")

        sig_len = sig_length(prepared["target_dimension"], prepared["degree"], scalar_term=scalar_term)
        result = SigOutputHandler(data, sig_len)
        if prepared["degree"] == 0:
            if scalar_term:
                result.data[...] = 1
            return result.data
        if data.batch_size == 0:
            return result.data

        err_code = CPSIG_VOLTERRA_SIG[data.dtype](
            data.data_ptr, result.data_ptr, prepared["handle"],
            data.batch_size, data.data_dimension, data.data_length,
            scalar_term, n_jobs)
        if err_code:
            raise Exception("Error in pysiglib.volterra_sig: " + err_msg(err_code))
        return result.data

    def _prepare_exact_numpy(self, dtype):
        raise NotImplementedError

    def __del__(self):
        try:
            self.clear_cache()
        except Exception:
            pass


class VolterraFSSK(VolterraKernel):
    """
    Finite state-space Volterra kernel with diagonal ``Lambda``.

    The represented kernel is

    .. math::

        K(t,s) = \\sum_p \\left(1^T e^{-\\Lambda(t-s)} b_p\\right) A_p.

    ``Lambda`` must currently be a diagonal realization, supplied either as
    a vector of diagonal entries with shape ``(R,)`` or as a diagonal matrix
    with shape ``(R, R)``. ``A`` has shape ``(q, m, d)`` and ``b`` has shape
    ``(q, R)``.
    """

    def __init__(self, Lambda, A, b, *, quad_order: int = 32):
        super().__init__()
        check_type(quad_order, "quad_order", int)
        if quad_order <= 0:
            raise ValueError("quad_order must be positive")

        self.Lambda = Lambda
        self.A = A
        self.b = b
        self.quad_order = quad_order

    @classmethod
    def diagonal(cls, Lambda, A, b, *, quad_order: int = 32):
        return cls(Lambda, A, b, quad_order=quad_order)

    def _prepare_exact_numpy(self, dtype):
        lambda_diag = self._prepare_diagonal_Lambda_numpy(dtype)

        b_rank = _rank(self.b)
        if b_rank not in (1, 2):
            raise ValueError("kernel.b must have rank 1 or 2, got rank " + str(b_rank))
        b = _numpy_array(self.b, "kernel.b", dtype, b_rank)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        if b.shape[1] != lambda_diag.shape[0]:
            raise ValueError("kernel.b.shape[-1] must match the state dimension R of kernel.Lambda")

        num_components = b.shape[0]
        A_rank = _rank(self.A)
        if A_rank not in (2, 3):
            raise ValueError("kernel.A must have rank 2 or 3, got rank " + str(A_rank))
        A = _numpy_array(self.A, "kernel.A", dtype, A_rank)
        if A.ndim == 2:
            A = A.reshape(1, *A.shape)

        if A.shape[0] != num_components:
            raise ValueError("kernel.A.shape[0] must match the number of kernel components in kernel.b")
        if not (_is_finite(lambda_diag) and _is_finite(b) and _is_finite(A)):
            raise ValueError("kernel.Lambda, kernel.b, and kernel.A must contain only finite values")

        return (
            lambda_diag,
            A,
            b,
            num_components,
            A.shape[1],
            lambda_diag.shape[0],
            A.shape[2],
        )

    def _prepare_diagonal_Lambda_numpy(self, dtype):
        Lambda_rank = _rank(self.Lambda)
        if Lambda_rank not in (1, 2):
            raise ValueError("kernel.Lambda must have rank 1 or 2, got rank " + str(Lambda_rank))
        Lambda = _numpy_array(self.Lambda, "kernel.Lambda", dtype, Lambda_rank)
        if Lambda.ndim == 1:
            return Lambda
        if Lambda.shape[0] != Lambda.shape[1]:
            raise ValueError("kernel.Lambda must have shape (R,) or (R, R)")
        off_diag = Lambda.copy()
        np.fill_diagonal(off_diag, 0)
        if np.any(off_diag != 0):
            raise ValueError("scheme='exact' currently requires diagonal kernel.Lambda")
        return np.ascontiguousarray(np.diag(Lambda))


def _prepare_volterra_sig_impl(
        degree: int,
        kernel: VolterraKernel,
        *,
        dt: float,
        readout_lag: float,
        scheme: str = "exact",
        dtype=None,
) -> dict:
    """
    Prepares CPU data for repeated exact Volterra signature computations.

    :param degree: Truncation level.
    :param kernel: Volterra kernel object.
    :param dt: Uniform time step between consecutive path samples.
    :param readout_lag: Non-negative readout lag after the final path sample.
        The kernel must have been prepared with the same value.
    :param scheme: Evaluation scheme. Currently only ``"exact"`` is supported.
    :param dtype: Floating dtype for the prepared native data. If omitted, this
        is inferred from the kernel arrays, defaulting to ``float64``.
    :return: Prepared Volterra signature cache entry.
    """
    np_dtype, dtype_name = _resolve_prepare_dtype(dtype, kernel)
    (
        lambda_diag,
        A,
        b,
        num_components,
        target_dimension,
        state_dimension,
        dimension,
    ) = kernel._prepare_exact_numpy(np_dtype)

    handle = c_uint64(0)
    err_code = CPSIG_PREPARE_VOLTERRA_SIG[dtype_name](
        _numpy_ptr(lambda_diag, dtype_name),
        _numpy_ptr(A, dtype_name),
        _numpy_ptr(b, dtype_name),
        dimension,
        num_components,
        target_dimension,
        state_dimension,
        degree,
        dt,
        readout_lag,
        kernel.quad_order,
        byref(handle),
    )
    if err_code:
        raise Exception("Error in pysiglib.prepare_volterra_sig: " + err_msg(err_code))

    return {
        "handle": handle.value,
        "dtype": dtype_name,
        "degree": degree,
        "dt": dt,
        "readout_lag": readout_lag,
        "dimension": dimension,
        "num_components": num_components,
        "target_dimension": target_dimension,
        "state_dimension": state_dimension,
        "quad_order": kernel.quad_order,
    }


def prepare_volterra_sig(
        degree: int,
        kernel: VolterraKernel,
        *,
        dt: float,
        readout_lag: float = 0.,
        scheme: str = "exact",
        dtype=None,
) -> None:
    """
    Prepares CPU data for repeated exact Volterra signature computations.

    This is equivalent to
    ``kernel.prepare(degree, dt=dt, readout_lag=readout_lag, scheme=scheme, dtype=dtype)``.
    """
    check_type(kernel, "kernel", VolterraKernel)
    return kernel.prepare(degree, dt=dt, readout_lag=readout_lag, scheme=scheme, dtype=dtype)


def volterra_sig(
        path: Union[np.ndarray, torch.Tensor],
        degree: int,
        kernel: VolterraKernel,
        *,
        dt: float,
        readout_lag: float = 0.,
        scheme: str = "exact",
        scalar_term: bool = False,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the truncated Volterra signature.

    :param path: Path or batch of paths with shape ``(..., length, dimension)``.
    :param degree: Truncation level.
    :param kernel: Volterra kernel object.
    :param dt: Uniform time step between consecutive path samples.
    :param readout_lag: Non-negative readout lag after the final path sample.
    :param scheme: Evaluation scheme. Currently only ``"exact"`` is supported.
    :param scalar_term: If True, include the leading scalar term.
    :param n_jobs: Number of CPU threads.
    :return: Volterra signature array.
    """
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(kernel, "kernel", VolterraKernel)
    check_type(dt, "dt", float)
    check_type(readout_lag, "readout_lag", float)
    check_type(scheme, "scheme", str)
    check_type(scalar_term, "scalar_term", bool)
    check_n_jobs(n_jobs)
    if scheme != "exact":
        raise ValueError("scheme must be 'exact'")
    if readout_lag < 0:
        raise ValueError("readout_lag must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")

    data = PathInputHandler(path, False, False, 1., "path")
    if data.device != "cpu":
        raise RuntimeError("volterra_sig currently supports CPU inputs only")

    prepared = kernel._get_prepared(degree, dt, readout_lag, scheme, data.dtype)
    if prepared is not None:
        return kernel._volterra_sig(
            data,
            prepared,
            scalar_term=scalar_term,
            n_jobs=n_jobs,
        )

    raise RuntimeError(
        "Volterra kernel has not been prepared for this degree, dt, readout_lag, scheme, and dtype; "
        "call kernel.prepare(degree, dt=dt, readout_lag=readout_lag, scheme=scheme, dtype=path.dtype) first")
