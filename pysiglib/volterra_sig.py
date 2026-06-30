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
    CPSIG_FREE_VOLTERRA_SIG_GENERAL,
    CPSIG_PREPARE_VOLTERRA_SIG,
    CPSIG_PREPARE_VOLTERRA_SIG_GENERAL,
    CPSIG_VOLTERRA_CONV_SIG,
    CPSIG_VOLTERRA_SIG,
    CPSIG_VOLTERRA_SIG_GENERAL,
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
            item for item in (_value_dtype(a) for a in kernel._dtype_source_arrays())
            if item is not None
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


def _resolve_volterra_scheme(kernel, scheme, readout_lag):
    """Return the effective evaluation scheme for a kernel.

    Convolution kernels always use the general convolution scheme (and require
    ``readout_lag == 0``); every other kernel uses the exact FSSK scheme.
    """
    if isinstance(kernel, VolterraConvolutionKernel):
        if readout_lag != 0:
            raise ValueError("the convolution scheme requires readout_lag=0")
        return "convolution"
    if scheme != "exact":
        raise ValueError("scheme must be 'exact'")
    return scheme


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
        scheme = _resolve_volterra_scheme(self, scheme, readout_lag)
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
        free_map = CPSIG_FREE_VOLTERRA_SIG_GENERAL if entry.get("general") else CPSIG_FREE_VOLTERRA_SIG
        err_code = free_map[entry["dtype"]](handle)
        entry["handle"] = 0
        if err_code:
            raise Exception("Error in pysiglib.VolterraKernel.clear_cache: " + err_msg(err_code))

    def _volterra_sig(self, data, prepared, scalar_term, n_jobs):
        if data.data_dimension != prepared["dimension"]:
            raise ValueError("path.shape[-1] must match the prepared Volterra signature path dimension")

        if prepared.get("convolution"):
            return self._volterra_conv_sig(data, prepared, scalar_term, n_jobs)

        sig_len = sig_length(prepared["target_dimension"], prepared["degree"], scalar_term=scalar_term)
        result = SigOutputHandler(data, sig_len)
        if prepared["degree"] == 0:
            if scalar_term:
                result.data[...] = 1
            return result.data
        if data.batch_size == 0:
            return result.data

        sig_map = CPSIG_VOLTERRA_SIG_GENERAL if prepared.get("general") else CPSIG_VOLTERRA_SIG
        err_code = sig_map[data.dtype](
            data.data_ptr, result.data_ptr, prepared["handle"],
            data.batch_size, data.data_dimension, data.data_length,
            scalar_term, n_jobs)
        if err_code:
            raise Exception("Error in pysiglib.volterra_sig: " + err_msg(err_code))
        return result.data

    def _volterra_conv_sig(self, data, prepared, scalar_term, n_jobs):
        from ._volterra_conv import convolution_lag_coefficients

        degree = prepared["degree"]
        sig_len = sig_length(prepared["target_dimension"], degree, scalar_term=scalar_term)
        result = SigOutputHandler(data, sig_len)
        if degree == 0:
            if scalar_term:
                result.data[...] = 1
            return result.data
        if data.batch_size == 0:
            return result.data

        # Lag coefficients depend only on (kind, params, dt, degree, dtype, S);
        # everything but S is fixed for this prepared entry, so cache by S to
        # avoid rebuilding the scipy coefficients on every call.
        S = data.data_length - 1
        cache = prepared.setdefault("_alpha_cache", {})
        cached = cache.get(S)
        if cached is None:
            np_dtype = np.float32 if prepared["dtype"] == "float32" else np.float64
            cached = convolution_lag_coefficients(
                prepared["conv_kind"], prepared["conv_params"],
                dt=prepared["dt"], degree=degree, S=S, dtype=np_dtype)
            cache[S] = cached
        alpha_lag, M = cached
        A = prepared["A"]

        err_code = CPSIG_VOLTERRA_CONV_SIG[data.dtype](
            data.data_ptr, result.data_ptr,
            _numpy_ptr(A, data.dtype), _numpy_ptr(alpha_lag, data.dtype),
            data.batch_size, data.data_dimension, data.data_length,
            prepared["num_components"], prepared["target_dimension"], degree, M,
            scalar_term, n_jobs)
        if err_code:
            raise Exception("Error in pysiglib.volterra_sig: " + err_msg(err_code))
        return result.data

    def _prepare_exact_numpy(self, dtype):
        raise NotImplementedError

    def _dtype_source_arrays(self):
        """Arrays whose dtype seeds the default prepared dtype (float64 if none)."""
        return ()

    def __del__(self):
        try:
            self.clear_cache()
        except Exception:
            pass


def _as_int_tuple(sizes):
    arr = np.atleast_1d(np.asarray(sizes, dtype=np.int64))
    out = tuple(int(s) for s in arr.tolist()) if arr.size else ()
    if any(s <= 0 for s in out):
        raise ValueError("block sizes must be strictly positive")
    return out


def _build_jordan_lambda_matrix(real_rates, real_sizes, osc_decays, osc_freqs, osc_sizes):
    """Assemble the dense block-diagonal Jordan realization of Lambda.

    Real block of rate ``r`` and size ``n``: ``r`` on the diagonal, ``-1`` on
    the first superdiagonal. Oscillatory block of decay ``a``, frequency ``w``
    and size ``n``: ``2n x 2n`` with ``[[a, -w], [w, a]]`` on the diagonal and
    ``-I2`` on the super-block diagonal. Real blocks precede oscillatory blocks.
    """
    real_rates = np.atleast_1d(np.asarray(real_rates, dtype=np.float64)) if np.size(real_rates) else np.zeros(0)
    osc_decays = np.atleast_1d(np.asarray(osc_decays, dtype=np.float64)) if np.size(osc_decays) else np.zeros(0)
    osc_freqs = np.atleast_1d(np.asarray(osc_freqs, dtype=np.float64)) if np.size(osc_freqs) else np.zeros(0)
    real_sizes = _as_int_tuple(real_sizes)
    osc_sizes = _as_int_tuple(osc_sizes)
    if real_rates.shape[0] != len(real_sizes):
        raise ValueError("real_rates and real_sizes must have the same length")
    if not (osc_decays.shape[0] == osc_freqs.shape[0] == len(osc_sizes)):
        raise ValueError("osc_decays, osc_freqs and osc_sizes must have the same length")

    R = int(sum(real_sizes) + 2 * sum(osc_sizes))
    Lambda = np.zeros((R, R), dtype=np.float64)
    off = 0
    for rate, n in zip(real_rates, real_sizes):
        for k in range(n):
            Lambda[off + k, off + k] = rate
            if k + 1 < n:
                Lambda[off + k, off + k + 1] = -1.0
        off += n
    for a, w, n in zip(osc_decays, osc_freqs, osc_sizes):
        for k in range(n):
            base = off + 2 * k
            Lambda[base, base] = a
            Lambda[base, base + 1] = -w
            Lambda[base + 1, base] = w
            Lambda[base + 1, base + 1] = a
            if k + 1 < n:
                Lambda[base, base + 2] = -1.0
                Lambda[base + 1, base + 3] = -1.0
        off += 2 * n
    return Lambda


def _b_from_prony(real_sizes, osc_sizes, alpha, beta, delta):
    """Assemble state vectors ``b`` (q, R) from Prony coefficients.

    Real block: ``b_k = alpha_k - alpha_{k+1}`` (with ``alpha_n = 0``).
    Oscillatory block: ``b_{2k}   = 1/2 (d_beta_k - d_delta_k)`` and
    ``b_{2k+1} = 1/2 (d_beta_k + d_delta_k)`` where ``d_x_k = x_k - x_{k+1}``.
    """
    real_sizes = _as_int_tuple(real_sizes)
    osc_sizes = _as_int_tuple(osc_sizes)
    n_real = sum(real_sizes)
    n_osc = sum(osc_sizes)

    def _rows(arr, name, width):
        if width == 0:
            return None
        if arr is None:
            raise ValueError(name + " is required for the specified blocks")
        m = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        if m.shape[1] != width:
            raise ValueError(name + " must have shape (q, " + str(width) + ")")
        return m

    a = _rows(alpha, "alpha", n_real)
    bt = _rows(beta, "beta", n_osc)
    dl = _rows(delta, "delta", n_osc)
    q = next((m.shape[0] for m in (a, bt, dl) if m is not None), 1)
    R = int(n_real + 2 * n_osc)
    b = np.zeros((q, R), dtype=np.float64)

    out = 0
    pos = 0
    for n in real_sizes:
        block = a[:, pos:pos + n]
        ext = np.concatenate([block, np.zeros((q, 1))], axis=1)
        b[:, out:out + n] = ext[:, :n] - ext[:, 1:n + 1]
        out += n
        pos += n
    pos = 0
    for n in osc_sizes:
        bb = np.concatenate([bt[:, pos:pos + n], np.zeros((q, 1))], axis=1)
        dd = np.concatenate([dl[:, pos:pos + n], np.zeros((q, 1))], axis=1)
        d_beta = bb[:, :n] - bb[:, 1:n + 1]
        d_delta = dd[:, :n] - dd[:, 1:n + 1]
        b[:, out:out + 2 * n:2] = 0.5 * (d_beta - d_delta)
        b[:, out + 1:out + 2 * n:2] = 0.5 * (d_beta + d_delta)
        out += 2 * n
        pos += n
    return b


class VolterraFSSK(VolterraKernel):
    """
    Finite state-space Volterra kernel with diagonal ``Lambda``.

    The represented kernel is

    .. math::

        K(t,s) = \\sum_p \\left(1^T e^{-\\Lambda(t-s)} b_p\\right) A_p.

    ``Lambda`` may be supplied as a vector of diagonal entries with shape
    ``(R,)``, a diagonal matrix with shape ``(R, R)``, or any dense matrix with
    shape ``(R, R)``. A diagonal ``Lambda`` uses the fast per-rate path; a dense
    ``Lambda`` uses the general matrix path, which keeps ``Lambda`` real and
    handles diagonalizable matrices (real or oscillatory/complex spectrum) and
    genuine (defective) Jordan blocks alike. ``A`` has shape ``(q, m, d)`` and
    ``b`` has shape ``(q, R)``.
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

    @classmethod
    def from_jordan(cls, *, A, b, real_rates=(), real_sizes=(), osc_decays=(),
                    osc_freqs=(), osc_sizes=(), quad_order: int = 32):
        """
        Build a kernel from real and oscillatory Jordan blocks with explicit
        state vectors ``b`` (in the Jordan basis).

        Real scalar poles are given by ``real_rates`` (rates) and ``real_sizes``
        (block sizes); oscillatory pole pairs by ``osc_decays``, ``osc_freqs``
        and ``osc_sizes``. ``b`` has shape ``(q, R)`` (or ``(R,)`` for a single
        component), ordered as all real blocks followed by all oscillatory
        blocks. All block sizes are supported, including genuine Jordan blocks
        of size > 1 (repeated poles), which are evaluated by the general matrix
        recursion.
        """
        Lambda = _build_jordan_lambda_matrix(
            real_rates, real_sizes, osc_decays, osc_freqs, osc_sizes)
        return cls(Lambda, A, b, quad_order=quad_order)

    @classmethod
    def from_prony(cls, *, A, real_rates=(), real_sizes=(), osc_decays=(),
                   osc_freqs=(), osc_sizes=(), alpha=None, beta=None, delta=None,
                   quad_order: int = 32):
        """
        Build a kernel from Jordan blocks and Prony coefficients.

        Same block specification as :meth:`from_jordan`, but the state vectors
        ``b`` are assembled from Prony coefficients: ``alpha`` (shape
        ``(q, sum(real_sizes))``) for the real blocks and ``beta``/``delta``
        (shape ``(q, sum(osc_sizes))``) for the oscillatory blocks, following
        the differenced-coefficient convention
        ``b_real = (alpha_k - alpha_{k+1})`` and
        ``b_osc = (1/2(d_beta - d_delta), 1/2(d_beta + d_delta))``.
        """
        Lambda = _build_jordan_lambda_matrix(
            real_rates, real_sizes, osc_decays, osc_freqs, osc_sizes)
        b = _b_from_prony(real_sizes, osc_sizes, alpha, beta, delta)
        return cls(Lambda, A, b, quad_order=quad_order)

    def _dtype_source_arrays(self):
        return (self.Lambda, self.A, self.b)

    def _prepare_exact_numpy(self, dtype):
        b_rank = _rank(self.b)
        if b_rank not in (1, 2):
            raise ValueError("kernel.b must have rank 1 or 2, got rank " + str(b_rank))
        b = _numpy_array(self.b, "kernel.b", dtype, b_rank)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        mode, lambda_data = self._prepare_Lambda_realization(dtype)
        state_dimension = lambda_data.shape[0]
        if b.shape[1] != state_dimension:
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
        if not (_is_finite(lambda_data) and _is_finite(b) and _is_finite(A)):
            raise ValueError("kernel.Lambda, kernel.b, and kernel.A must contain only finite values")

        realization = {
            "mode": mode,
            "A": A,
            "b": b,
            "num_components": num_components,
            "target_dimension": A.shape[1],
            "state_dimension": state_dimension,
            "dimension": A.shape[2],
        }
        if mode == "diagonal":
            realization["lambda_diag"] = lambda_data
        else:
            realization["Lambda"] = lambda_data
        return realization

    def _prepare_Lambda_realization(self, dtype):
        """Return ``(mode, lambda_data)`` describing the state realization.

        A diagonal ``Lambda`` (vector or diagonal matrix) takes the fast
        ``"diagonal"`` path with per-rate coefficients. Any dense ``Lambda``
        takes the ``"general"`` path, which keeps ``Lambda`` as a real matrix
        and evaluates the FSSK coefficients from the matrix resolvent and a
        matrix exponential. The general path handles diagonalizable matrices
        (real or oscillatory/complex spectrum) and genuine (defective) Jordan
        blocks uniformly.
        """
        Lambda_rank = _rank(self.Lambda)
        if Lambda_rank not in (1, 2):
            raise ValueError("kernel.Lambda must have rank 1 or 2, got rank " + str(Lambda_rank))
        Lambda = _numpy_array(self.Lambda, "kernel.Lambda", dtype, Lambda_rank)
        if Lambda.ndim == 1:
            return "diagonal", Lambda
        if Lambda.shape[0] != Lambda.shape[1]:
            raise ValueError("kernel.Lambda must have shape (R,) or (R, R)")

        off_diag = Lambda.copy()
        np.fill_diagonal(off_diag, 0)
        if not np.any(off_diag != 0):
            return "diagonal", np.ascontiguousarray(np.diag(Lambda))
        return "general", np.ascontiguousarray(Lambda)


class VolterraFractionalKernel(VolterraKernel):
    """
    Fractional (rough) Volterra kernel.

    The represented kernel is

    .. math::

        K(t,s) = \\sum_p k(t-s) A_p,
        \\qquad k(u) = \\frac{u^{\\beta - 1}}{\\Gamma(\\beta)},

    with fractional exponent ``beta`` in ``(1/2, 1)`` (the rough/singular
    regime, Hurst ``H = beta - 1/2``). The scalar kernel ``k`` is approximated
    by a sum of ``R`` exponentials, ``k(u) ~= sum_r b_r exp(-x_r u)``, using the
    BL2 positive-Hurst quadrature rule, which yields a diagonal finite
    state-space realization that is evaluated by the exact Volterra recursion.

    ``A`` has shape ``(q, m, d)`` (or ``(m, d)`` for a single component).
    ``R`` is the number of exponentials and equals the state dimension. ``T`` is
    the approximation horizon over which the exponential fit is optimized.

    This requires ``scipy`` for the node optimization (an optional dependency,
    imported only when the kernel is prepared).
    """

    def __init__(self, A, *, beta: float, R: int, T: float = 1.0, quad_order: int = 32):
        super().__init__()
        check_type(beta, "beta", float)
        check_type(R, "R", int)
        check_type(T, "T", float)
        check_type(quad_order, "quad_order", int)
        if not (0.5 < beta < 1.0):
            raise ValueError("beta must lie in (1/2, 1) for the fractional kernel")
        if R <= 0:
            raise ValueError("R must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        if quad_order <= 0:
            raise ValueError("quad_order must be positive")

        self.A = A
        self.beta = beta
        self.R = R
        self.T = T
        self.quad_order = quad_order

    def _dtype_source_arrays(self):
        return (self.A,)

    def _prepare_exact_numpy(self, dtype):
        from ._rough_kernel import bl2_quadrature_rule

        A_rank = _rank(self.A)
        if A_rank not in (2, 3):
            raise ValueError("kernel.A must have rank 2 or 3, got rank " + str(A_rank))
        A = _numpy_array(self.A, "kernel.A", dtype, A_rank)
        if A.ndim == 2:
            A = A.reshape(1, *A.shape)
        if not _is_finite(A):
            raise ValueError("kernel.A must contain only finite values")

        num_components = A.shape[0]

        nodes, weights = bl2_quadrature_rule(self.beta, self.R, self.T)
        lambda_diag = np.ascontiguousarray(nodes.astype(dtype))
        b = np.ascontiguousarray(np.broadcast_to(weights.astype(dtype), (num_components, self.R)))

        return {
            "mode": "diagonal",
            "lambda_diag": lambda_diag,
            "A": A,
            "b": b,
            "num_components": num_components,
            "target_dimension": A.shape[1],
            "state_dimension": self.R,
            "dimension": A.shape[2],
        }


def _conv_projection(A, dtype):
    """Validate ``A`` and pack it into a contiguous ``(q, m, d)`` array.

    Enforces ``q == 1`` (the scalar case implemented by the convolution scheme).
    """
    A_rank = _rank(A)
    if A_rank not in (2, 3):
        raise ValueError("kernel.A must have rank 2 or 3, got rank " + str(A_rank))
    A = _numpy_array(A, "kernel.A", dtype, A_rank)
    if A.ndim == 2:
        A = A.reshape(1, *A.shape)
    if not _is_finite(A):
        raise ValueError("kernel.A must contain only finite values")
    q, m, d = A.shape
    if q != 1:
        raise NotImplementedError(
            "the general convolution scheme currently supports q=1 kernels "
            "(a single kernel component); multivariate q>1 is not yet implemented")
    return np.ascontiguousarray(A), q, m, d


class VolterraConvolutionKernel(VolterraKernel):
    r"""
    Base class for general-convolution-scheme Volterra kernels.

    These kernels represent

    .. math::

        K(t,s) = \\sum_p k_p(t-s)\\, A_p,

    where the scalar parts :math:`k_p` are general convolution kernels that are
    not finite state-space (no exponential-sum form). The truncated signature is
    computed by the quadratic Volterra-Chen recursion (the general convolution
    scheme) rather than the exact FSSK state recursion. Such kernels always use
    the convolution scheme and require ``readout_lag=0``.

    This release implements the scalar (``q=1``) case.
    """

    def _conv_realization(self, dtype):
        raise NotImplementedError

    def _dtype_source_arrays(self):
        return (self.A,)


class VolterraConvFractionalKernel(VolterraConvolutionKernel):
    r"""
    Fractional kernel for the general convolution scheme.

    The represented kernel is

    .. math::

        K(t,s) = k(t-s)\\, A, \\qquad k(u) = \\frac{u^{\\beta - 1}}{\\Gamma(\\beta)},

    with ``beta > 0``. Unlike :class:`VolterraFractionalKernel`, which
    approximates ``k`` by a sum of exponentials and uses the exact FSSK scheme,
    this evaluates the interval coefficients of ``k`` exactly (closed-form
    regularized incomplete beta) and uses the quadratic convolution recursion.

    ``A`` has shape ``(m, d)`` (or ``(1, m, d)``).
    """

    def __init__(self, A, *, beta: float):
        super().__init__()
        check_type(beta, "beta", float)
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.A = A
        self.beta = beta

    def _conv_realization(self, dtype):
        A, q, m, d = _conv_projection(self.A, dtype)
        return {"A": A, "q": q, "m": m, "d": d, "kind": "fractional",
                "params": {"beta": float(self.beta)}}


class VolterraConvGammaKernel(VolterraConvolutionKernel):
    r"""
    Gamma kernel for the general convolution scheme.

    The represented kernel is

    .. math::

        K(t,s) = k(t-s)\\, A, \\qquad
        k(u) = \\mathrm{scale}\\, e^{-\\mathrm{rate}\\, u}
               \\frac{u^{\\beta - 1}}{\\Gamma(\\beta)},

    with ``beta > 0``, ``scale > 0`` and ``rate >= 0``. The interval
    coefficients are built by Gauss-Legendre quadrature and the signature is
    evaluated by the quadratic convolution recursion.

    ``A`` has shape ``(m, d)`` (or ``(1, m, d)``).
    """

    def __init__(self, A, *, beta: float, scale: float = 1.0, rate: float = 0.0,
                 quad_order: int = 32):
        super().__init__()
        check_type(beta, "beta", float)
        check_type(scale, "scale", float)
        check_type(rate, "rate", float)
        check_type(quad_order, "quad_order", int)
        if beta <= 0:
            raise ValueError("beta must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if rate < 0:
            raise ValueError("rate must be non-negative")
        if quad_order <= 0:
            raise ValueError("quad_order must be positive")
        self.A = A
        self.beta = beta
        self.scale = scale
        self.rate = rate
        self.quad_order = quad_order

    def _conv_realization(self, dtype):
        A, q, m, d = _conv_projection(self.A, dtype)
        return {"A": A, "q": q, "m": m, "d": d, "kind": "gamma",
                "params": {"beta": float(self.beta), "scale": float(self.scale),
                           "rate": float(self.rate), "quad_order": int(self.quad_order)}}


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
    Prepares CPU data for repeated Volterra signature computations.

    :param degree: Truncation level.
    :param kernel: Volterra kernel object.
    :param dt: Uniform time step between consecutive path samples.
    :param readout_lag: Non-negative readout lag after the final path sample.
        The kernel must have been prepared with the same value.
    :param scheme: Evaluation scheme for FSSK kernels (only ``"exact"``).
        Convolution kernels (:class:`VolterraConvolutionKernel`) ignore this and
        always use the general convolution scheme.
    :param dtype: Floating dtype for the prepared native data. If omitted, this
        is inferred from the kernel arrays, defaulting to ``float64``.
    :return: Prepared Volterra signature cache entry.
    """
    np_dtype, dtype_name = _resolve_prepare_dtype(dtype, kernel)
    if isinstance(kernel, VolterraConvolutionKernel):
        spec = kernel._conv_realization(np_dtype)
        return {
            "convolution": True,
            "handle": 0,
            "dtype": dtype_name,
            "degree": degree,
            "dt": dt,
            "readout_lag": readout_lag,
            "dimension": spec["d"],
            "num_components": spec["q"],
            "target_dimension": spec["m"],
            "A": spec["A"],
            "conv_kind": spec["kind"],
            "conv_params": spec["params"],
        }
    realization = kernel._prepare_exact_numpy(np_dtype)
    mode = realization["mode"]
    A = realization["A"]
    b = realization["b"]
    num_components = realization["num_components"]
    target_dimension = realization["target_dimension"]
    state_dimension = realization["state_dimension"]
    dimension = realization["dimension"]

    handle = c_uint64(0)
    if mode == "general":
        from ._general_fssk import general_coefficients
        coef = general_coefficients(
            realization["Lambda"], b, dt=dt, readout_lag=readout_lag,
            quad_order=kernel.quad_order, degree=degree, dtype=np_dtype)
        err_code = CPSIG_PREPARE_VOLTERRA_SIG_GENERAL[dtype_name](
            _numpy_ptr(coef["E"], dtype_name),
            _numpy_ptr(coef["psi"], dtype_name),
            _numpy_ptr(coef["phi"], dtype_name),
            _numpy_ptr(coef["readout_weights"], dtype_name),
            _numpy_ptr(A, dtype_name),
            dimension,
            num_components,
            target_dimension,
            state_dimension,
            degree,
            byref(handle),
        )
    else:
        err_code = CPSIG_PREPARE_VOLTERRA_SIG[dtype_name](
            _numpy_ptr(realization["lambda_diag"], dtype_name),
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
        "general": mode == "general",
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
    Prepares CPU data for repeated Volterra signature computations.

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
    :param scheme: Evaluation scheme for FSSK kernels (only ``"exact"``).
        Convolution kernels (:class:`VolterraConvolutionKernel`) ignore this and
        always use the general convolution scheme.
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
    scheme = _resolve_volterra_scheme(kernel, scheme, readout_lag)
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
