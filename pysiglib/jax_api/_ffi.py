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

import ctypes
import os
import sys

import numpy as np

from ..load_siglib import (
    BUILT_WITH_CUDA,
    BUILT_WITH_JAX_FFI,
    PYSIGLIB_CUDA_DIR,
    SYSTEM,
    native_lib_filename,
)
from ..sig_length import sig_length, log_sig_length
from ..branched_sig import branched_sig_length

import jax

# jax>=0.9.1 is the first release whose XLA FFI framework version is 0.3,
# matching what our compiled handlers declare. Older jax versions raise a
# cryptic XlaRuntimeError at handler registration; fail early with a clear
# message instead.
_MIN_JAX = (0, 9, 1)


def _parse_version(v):
    out = []
    for part in v.split("."):
        num = ""
        for c in part:
            if c.isdigit():
                num += c
            else:
                break
        if num:
            out.append(int(num))
    return tuple(out)


if _parse_version(jax.__version__)[:3] < _MIN_JAX:
    raise ImportError(
        "pysiglib.jax_api requires jax>=0.9.1 (XLA FFI API 0.3); "
        "found jax==" + jax.__version__ + ". "
        "Install a compatible jax via: pip install 'pysiglib[jax]' "
        "or pip install 'jax>=0.9.1'."
    )

_jax_ffi = jax.ffi


_CPU_LIB = None
_CUDA_LIB = None
_REGISTERED = False

_TARGETS = {
    "sig": {
        "cpu": ("pysiglib_sig_cpu", "PySigLibSigCpu"),
        "cuda": ("pysiglib_sig_cuda", "PySigLibSigCuda"),
    },
    "sig_backprop": {
        "cpu": ("pysiglib_sig_backprop_cpu", "PySigLibSigBackpropCpu"),
        "cuda": ("pysiglib_sig_backprop_cuda", "PySigLibSigBackpropCuda"),
    },
    "sig_combine": {
        "cpu": ("pysiglib_sig_combine_cpu", "PySigLibSigCombineCpu"),
        "cuda": ("pysiglib_sig_combine_cuda", "PySigLibSigCombineCuda"),
    },
    "sig_combine_backprop": {
        "cpu": ("pysiglib_sig_combine_backprop_cpu", "PySigLibSigCombineBackpropCpu"),
        "cuda": ("pysiglib_sig_combine_backprop_cuda", "PySigLibSigCombineBackpropCuda"),
    },
    "transform_path": {
        "cpu": ("pysiglib_transform_path_cpu", "PySigLibTransformPathCpu"),
        "cuda": ("pysiglib_transform_path_cuda", "PySigLibTransformPathCuda"),
    },
    "transform_path_backprop": {
        "cpu": ("pysiglib_transform_path_backprop_cpu", "PySigLibTransformPathBackpropCpu"),
        "cuda": ("pysiglib_transform_path_backprop_cuda", "PySigLibTransformPathBackpropCuda"),
    },
    "sig_to_log_sig": {
        "cpu": ("pysiglib_sig_to_log_sig_cpu", "PySigLibSigToLogSigCpu"),
        "cuda": ("pysiglib_sig_to_log_sig_cuda", "PySigLibSigToLogSigCuda"),
    },
    "sig_to_log_sig_backprop": {
        "cpu": ("pysiglib_sig_to_log_sig_backprop_cpu", "PySigLibSigToLogSigBackpropCpu"),
        "cuda": ("pysiglib_sig_to_log_sig_backprop_cuda", "PySigLibSigToLogSigBackpropCuda"),
    },
    "log_sig_combine": {
        "cpu": ("pysiglib_log_sig_combine_cpu", "PySigLibLogSigCombineCpu"),
        "cuda": ("pysiglib_log_sig_combine_cuda", "PySigLibLogSigCombineCuda"),
    },
    "log_sig_combine_backprop": {
        "cpu": ("pysiglib_log_sig_combine_backprop_cpu", "PySigLibLogSigCombineBackpropCpu"),
        "cuda": ("pysiglib_log_sig_combine_backprop_cuda", "PySigLibLogSigCombineBackpropCuda"),
    },
    "sig_kernel_pde": {
        "cpu": ("pysiglib_sig_kernel_pde_cpu", "PySigLibSigKernelPdeCpu"),
        "cuda": ("pysiglib_sig_kernel_pde_cuda", "PySigLibSigKernelPdeCuda"),
    },
    "sig_kernel_pde_backprop": {
        "cpu": ("pysiglib_sig_kernel_pde_backprop_cpu", "PySigLibSigKernelPdeBackpropCpu"),
        "cuda": ("pysiglib_sig_kernel_pde_backprop_cuda", "PySigLibSigKernelPdeBackpropCuda"),
    },
    "logsig_to_sig": {
        "cpu": ("pysiglib_logsig_to_sig_cpu", "PySigLibLogSigToSigCpu"),
        "cuda": ("pysiglib_logsig_to_sig_cuda", "PySigLibLogSigToSigCuda"),
    },
    "logsig_to_sig_backprop": {
        "cpu": ("pysiglib_logsig_to_sig_backprop_cpu", "PySigLibLogSigToSigBackpropCpu"),
        "cuda": ("pysiglib_logsig_to_sig_backprop_cuda", "PySigLibLogSigToSigBackpropCuda"),
    },
    "log_sig_from_path": {
        "cpu": ("pysiglib_log_sig_from_path_cpu", "PySigLibLogSigFromPathCpu"),
        "cuda": ("pysiglib_log_sig_from_path_cuda", "PySigLibLogSigFromPathCuda"),
    },
    "log_sig_from_path_backprop": {
        "cpu": ("pysiglib_log_sig_from_path_backprop_cpu", "PySigLibLogSigFromPathBackpropCpu"),
        "cuda": ("pysiglib_log_sig_from_path_backprop_cuda", "PySigLibLogSigFromPathBackpropCuda"),
    },
    "branched_sig": {
        "cpu": ("pysiglib_branched_sig_cpu", "PySigLibBranchedSigCpu"),
        "cuda": ("pysiglib_branched_sig_cuda", "PySigLibBranchedSigCuda"),
    },
    "branched_sig_backprop": {
        "cpu": ("pysiglib_branched_sig_backprop_cpu", "PySigLibBranchedSigBackpropCpu"),
        "cuda": ("pysiglib_branched_sig_backprop_cuda", "PySigLibBranchedSigBackpropCuda"),
    },
    "branched_sig_combine": {
        "cpu": ("pysiglib_branched_sig_combine_cpu", "PySigLibBranchedSigCombineCpu"),
        "cuda": ("pysiglib_branched_sig_combine_cuda", "PySigLibBranchedSigCombineCuda"),
    },
    "branched_sig_combine_backprop": {
        "cpu": ("pysiglib_branched_sig_combine_backprop_cpu", "PySigLibBranchedSigCombineBackpropCpu"),
        "cuda": ("pysiglib_branched_sig_combine_backprop_cuda", "PySigLibBranchedSigCombineBackpropCuda"),
    },
    "branched_sig_to_log_sig": {
        "cpu": ("pysiglib_branched_sig_to_log_sig_cpu", "PySigLibBranchedSigToLogSigCpu"),
        "cuda": ("pysiglib_branched_sig_to_log_sig_cuda", "PySigLibBranchedSigToLogSigCuda"),
    },
    "branched_sig_to_log_sig_backprop": {
        "cpu": ("pysiglib_branched_sig_to_log_sig_backprop_cpu", "PySigLibBranchedSigToLogSigBackpropCpu"),
        "cuda": ("pysiglib_branched_sig_to_log_sig_backprop_cuda", "PySigLibBranchedSigToLogSigBackpropCuda"),
    },
}


def _package_dirs():
    dirs = []
    pkg = sys.modules["pysiglib"]

    for path in getattr(pkg, "__path__", ()):
        if path not in dirs:
            dirs.append(path)

    pkg_file = getattr(pkg, "__file__", None)
    if pkg_file is not None:
        pkg_dir = os.path.dirname(pkg_file)
        if pkg_dir not in dirs:
            dirs.append(pkg_dir)

    return dirs


def _find_native_lib(filename: str) -> str:
    for directory in _package_dirs():
        lib_path = os.path.join(directory, filename)
        if os.path.exists(lib_path):
            return lib_path

    searched = ", ".join(_package_dirs())
    raise OSError(
        f"Could not find native library '{filename}' in pysiglib package paths: {searched}"
    )


def _open_cdll(path: str) -> ctypes.CDLL:
    """ctypes.CDLL wrapper that applies winmode=0 on Windows."""
    if SYSTEM == "Windows":
        return ctypes.CDLL(path, winmode=0)
    return ctypes.CDLL(path)


def _preload_windows_dll(path: str) -> None:
    """winmode=0 disables LOAD_LIBRARY_SEARCH_USER_DIRS, so a freshly-loaded
    DLL cannot resolve its imports from its own install directory. Pre-loading
    the sibling puts its handle in the process table, which satisfies the
    loader when it later resolves imports of a dependent DLL.
    """
    if SYSTEM != "Windows":
        return
    try:
        ctypes.CDLL(path, winmode=0)
    except OSError:
        pass


def _load_cpu_ffi_library() -> ctypes.CDLL:
    """Load the base pysiglib wheel's CPU-only JAX FFI library."""
    global _CPU_LIB
    if _CPU_LIB is not None:
        return _CPU_LIB

    if not BUILT_WITH_JAX_FFI:
        raise RuntimeError(
            "pySigLib was built without JAX FFI support. Rebuild after installing jaxlib "
            "so the XLA FFI headers are available at build time."
        )

    if SYSTEM == "Windows":
        try:
            _preload_windows_dll(_find_native_lib(native_lib_filename("cpsig")))
        except OSError:
            pass

    _CPU_LIB = _open_cdll(_find_native_lib(native_lib_filename("pysiglib_jax_ffi_cpu")))
    return _CPU_LIB


def _plugin_has_jax_ffi() -> bool:
    """True only if the pysiglib-cuda plugin shipped a _config.py declaring
    BUILT_WITH_JAX_FFI=True (i.e. the plugin build found jaxlib headers).
    """
    if PYSIGLIB_CUDA_DIR is None:
        return False
    try:
        from pysiglib_cuda import _config as _plugin_config
    except ImportError:
        return False
    return bool(getattr(_plugin_config, "BUILT_WITH_JAX_FFI", False))


def _load_cuda_ffi_library():
    """Load the pysiglib-cuda plugin's CUDA JAX FFI library, or return None
    if the plugin is not installed, was built without JAX FFI, or is missing
    the expected DLL. `ensure_registered` promotes None to a hard error when
    BUILT_WITH_CUDA is True.
    """
    global _CUDA_LIB
    if _CUDA_LIB is not None:
        return _CUDA_LIB

    if not _plugin_has_jax_ffi():
        return None

    lib_path = os.path.join(PYSIGLIB_CUDA_DIR, native_lib_filename("pysiglib_jax_ffi_cuda"))
    if not os.path.exists(lib_path):
        return None

    if SYSTEM == "Windows":
        try:
            _preload_windows_dll(_find_native_lib(native_lib_filename("cpsig")))
        except OSError:
            pass
        cusig_path = os.path.join(PYSIGLIB_CUDA_DIR, native_lib_filename("cusig"))
        if os.path.exists(cusig_path):
            _preload_windows_dll(cusig_path)

    _CUDA_LIB = _open_cdll(lib_path)
    return _CUDA_LIB


def _augmented_dim(dimension, time_aug, lead_lag):
    return (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)


def _normalize_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise TypeError(f"Only float32 and float64 are supported, got {dtype}.")
    return dtype


def _check_same_dtype(*arrays) -> np.dtype:
    """Validate that every array has a supported float dtype and that all
    arrays share the same dtype. Returns the common dtype."""
    if not arrays:
        raise ValueError("at least one array required")
    dt = _normalize_dtype(arrays[0].dtype)
    for a in arrays[1:]:
        other = _normalize_dtype(a.dtype)
        if other != dt:
            raise TypeError(f"dtype mismatch: {dt} vs {other}")
    return dt


def _target_name(op: str, platform: str) -> str:
    return _TARGETS[op][platform][0]


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    cpu_lib = _load_cpu_ffi_library()
    for op_targets in _TARGETS.values():
        target_name, symbol_name = op_targets["cpu"]
        _jax_ffi.register_ffi_target(
            target_name,
            _jax_ffi.pycapsule(getattr(cpu_lib, symbol_name)),
            platform="cpu",
        )

    if BUILT_WITH_CUDA:
        cuda_lib = _load_cuda_ffi_library()
        if cuda_lib is None:
            raise RuntimeError(
                "pysiglib-cuda is installed but its JAX FFI CUDA library is missing. "
                "The plugin likely pre-dates the JAX FFI split, or its build could not "
                "locate jaxlib headers. Install jaxlib and reinstall the plugin:\n"
                "    pip install 'jaxlib>=0.9.1'\n"
                "    pip install --no-cache-dir --force-reinstall 'pysiglib-cuda'"
            )
        for op_targets in _TARGETS.values():
            target_name, symbol_name = op_targets["cuda"]
            _jax_ffi.register_ffi_target(
                target_name,
                _jax_ffi.pycapsule(getattr(cuda_lib, symbol_name)),
                platform="CUDA",
            )

    _REGISTERED = True


def _sig_shape(path_shape, degree: int, time_aug: bool, lead_lag: bool) -> tuple[int, ...]:
    dimension = path_shape[-1]
    out_len = sig_length(_augmented_dim(dimension, time_aug, lead_lag), degree, scalar_term=True)
    if out_len == 0:
        raise ValueError("Signature length overflow.")
    return (*path_shape[:-2], out_len)


def _make_ffi_call(op_name, inputs, out_type, call_kwargs):
    cpu_call = _jax_ffi.ffi_call(_target_name(op_name, "cpu"), out_type, vmap_method="broadcast_all")
    if BUILT_WITH_CUDA:
        cuda_call = _jax_ffi.ffi_call(_target_name(op_name, "cuda"), out_type, vmap_method="broadcast_all")
        return jax.lax.platform_dependent(
            *inputs,
            cpu=lambda *args: cpu_call(*args, **call_kwargs),
            cuda=lambda *args: cuda_call(*args, **call_kwargs),
        )
    return cpu_call(*inputs, **call_kwargs)


# ---------------------------------------------------------------------------
# sig
# ---------------------------------------------------------------------------

def sig_ffi_call(path, degree, time_aug, lead_lag, end_time, horner, n_jobs):
    _normalize_dtype(path.dtype)
    out_type = jax.ShapeDtypeStruct(_sig_shape(path.shape, degree, time_aug, lead_lag), path.dtype)
    call_kwargs = dict(degree=np.int64(degree), time_aug=np.bool_(time_aug), lead_lag=np.bool_(lead_lag),
                       end_time=np.float64(end_time), horner=np.bool_(horner), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig", (path,), out_type, call_kwargs)


def sig_backprop_ffi_call(path, sig_, cotangent, degree, time_aug, lead_lag, end_time, n_jobs):
    _check_same_dtype(path, sig_, cotangent)
    out_type = jax.ShapeDtypeStruct(path.shape, path.dtype)
    call_kwargs = dict(degree=np.int64(degree), time_aug=np.bool_(time_aug), lead_lag=np.bool_(lead_lag),
                       end_time=np.float64(end_time), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig_backprop", (path, sig_, cotangent), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# sig_combine
# ---------------------------------------------------------------------------

def sig_combine_ffi_call(sig1, sig2, dimension, degree, n_jobs):
    _check_same_dtype(sig1, sig2)
    out_type = jax.ShapeDtypeStruct(sig1.shape, sig1.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig_combine", (sig1, sig2), out_type, call_kwargs)


def sig_combine_backprop_ffi_call(cotangent, sig1, sig2, dimension, degree, n_jobs):
    _check_same_dtype(cotangent, sig1, sig2)
    grad_type = jax.ShapeDtypeStruct(sig1.shape, sig1.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig_combine_backprop", (cotangent, sig1, sig2), (grad_type, grad_type), call_kwargs)


# ---------------------------------------------------------------------------
# transform_path
# ---------------------------------------------------------------------------

def _transform_path_out_shape(path_shape, time_aug, lead_lag):
    length = path_shape[-2]
    dimension = path_shape[-1]
    out_length = (2 * length - 1) if lead_lag else length
    return (*path_shape[:-2], out_length, _augmented_dim(dimension, time_aug, lead_lag))


def transform_path_ffi_call(path, time_aug, lead_lag, end_time, n_jobs):
    _normalize_dtype(path.dtype)
    out_type = jax.ShapeDtypeStruct(_transform_path_out_shape(path.shape, time_aug, lead_lag), path.dtype)
    call_kwargs = dict(time_aug=np.bool_(time_aug), lead_lag=np.bool_(lead_lag),
                       end_time=np.float64(end_time), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("transform_path", (path,), out_type, call_kwargs)


def transform_path_backprop_ffi_call(cotangent, orig_dimension, orig_length, time_aug, lead_lag, end_time, n_jobs):
    _normalize_dtype(cotangent.dtype)
    out_shape = (*cotangent.shape[:-2], orig_length, orig_dimension)
    out_type = jax.ShapeDtypeStruct(out_shape, cotangent.dtype)
    call_kwargs = dict(orig_dimension=np.int64(orig_dimension), orig_length=np.int64(orig_length),
                       time_aug=np.bool_(time_aug), lead_lag=np.bool_(lead_lag),
                       end_time=np.float64(end_time), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("transform_path_backprop", (cotangent,), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# sig_to_log_sig
# ---------------------------------------------------------------------------

def sig_to_log_sig_ffi_call(sig_arr, dimension, degree, method, n_jobs):
    _normalize_dtype(sig_arr.dtype)
    if method == 0:
        out_len = sig_length(dimension, degree, scalar_term=True)
    else:
        out_len = log_sig_length(dimension, degree)
    out_type = jax.ShapeDtypeStruct((*sig_arr.shape[:-1], out_len), sig_arr.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree),
                       method=np.int64(method), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig_to_log_sig", (sig_arr,), out_type, call_kwargs)


def sig_to_log_sig_backprop_ffi_call(sig_arr, cotangent, dimension, degree, method, n_jobs):
    _check_same_dtype(sig_arr, cotangent)
    out_type = jax.ShapeDtypeStruct(sig_arr.shape, sig_arr.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree),
                       method=np.int64(method), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig_to_log_sig_backprop", (sig_arr, cotangent), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# log_sig_combine
# ---------------------------------------------------------------------------

def log_sig_combine_ffi_call(ls1, ls2, dimension, degree, n_jobs):
    _check_same_dtype(ls1, ls2)
    out_type = jax.ShapeDtypeStruct(ls1.shape, ls1.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("log_sig_combine", (ls1, ls2), out_type, call_kwargs)


def log_sig_combine_backprop_ffi_call(cotangent, ls1, ls2, dimension, degree, n_jobs):
    _check_same_dtype(cotangent, ls1, ls2)
    grad_type = jax.ShapeDtypeStruct(ls1.shape, ls1.dtype)
    out_type = (grad_type, grad_type)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("log_sig_combine_backprop", (cotangent, ls1, ls2), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# sig_kernel PDE solver
# ---------------------------------------------------------------------------

def sig_kernel_pde_ffi_call(gram, dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs):
    _normalize_dtype(gram.dtype)
    if return_grid:
        L1m1, L2m1 = gram.shape[-2], gram.shape[-1]
        gl1 = ((L1m1) << dyadic_order_1) + 1
        gl2 = ((L2m1) << dyadic_order_2) + 1
        out_shape = (*gram.shape[:-2], gl1, gl2)
    else:
        out_shape = gram.shape[:-2]
    out_type = jax.ShapeDtypeStruct(out_shape, gram.dtype)
    call_kwargs = dict(dimension=np.int64(dimension),
                       dyadic_order_1=np.int64(dyadic_order_1),
                       dyadic_order_2=np.int64(dyadic_order_2),
                       return_grid=np.bool_(return_grid), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig_kernel_pde", (gram,), out_type, call_kwargs)


def sig_kernel_pde_backprop_ffi_call(gram, derivs, k_grid, dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs):
    _check_same_dtype(gram, derivs, k_grid)
    out_type = jax.ShapeDtypeStruct(gram.shape, gram.dtype)
    call_kwargs = dict(dimension=np.int64(dimension),
                       dyadic_order_1=np.int64(dyadic_order_1),
                       dyadic_order_2=np.int64(dyadic_order_2),
                       return_grid=np.bool_(return_grid), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("sig_kernel_pde_backprop", (gram, derivs, k_grid), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# logsig_to_sig (tensor exponential)
# ---------------------------------------------------------------------------

def logsig_to_sig_ffi_call(log_sig_arr, dimension, degree, method, n_jobs):
    _normalize_dtype(log_sig_arr.dtype)
    # Output is sig-shaped for every method; for methods 1 and 2 the input
    # is log-sig-shaped (shorter), so `log_sig_arr.shape` can't be reused.
    out_len = sig_length(dimension, degree, scalar_term=True)
    out_type = jax.ShapeDtypeStruct((*log_sig_arr.shape[:-1], out_len), log_sig_arr.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree),
                       method=np.int64(method), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("logsig_to_sig", (log_sig_arr,), out_type, call_kwargs)


def logsig_to_sig_backprop_ffi_call(log_sig_arr, cotangent, dimension, degree, method, n_jobs):
    _check_same_dtype(log_sig_arr, cotangent)
    out_type = jax.ShapeDtypeStruct(log_sig_arr.shape, log_sig_arr.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree),
                       method=np.int64(method), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("logsig_to_sig_backprop", (log_sig_arr, cotangent), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# log_sig_from_path (method=3)
# ---------------------------------------------------------------------------

def log_sig_from_path_ffi_call(path, dimension, degree, n_jobs):
    _normalize_dtype(path.dtype)
    out_len = log_sig_length(dimension, degree)
    out_type = jax.ShapeDtypeStruct((*path.shape[:-2], out_len), path.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("log_sig_from_path", (path,), out_type, call_kwargs)


def log_sig_from_path_backprop_ffi_call(cotangent, path, dimension, degree, n_jobs):
    _check_same_dtype(cotangent, path)
    out_type = jax.ShapeDtypeStruct(path.shape, path.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), degree=np.int64(degree), n_jobs=np.int64(n_jobs))
    return _make_ffi_call("log_sig_from_path_backprop", (cotangent, path), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# branched_sig
# ---------------------------------------------------------------------------

def _branched_sig_shape(path_shape, dimension, max_nodes, time_aug, lead_lag, planar):
    aug_dim = _augmented_dim(dimension, time_aug, lead_lag)
    out_len = branched_sig_length(aug_dim, max_nodes, scalar_term=True, planar=planar)
    if out_len == 0:
        raise ValueError("Branched signature length overflow.")
    return (*path_shape[:-2], out_len)


def branched_sig_ffi_call(path, max_nodes, time_aug, lead_lag, end_time, n_jobs, planar):
    _normalize_dtype(path.dtype)
    dimension = path.shape[-1]
    out_type = jax.ShapeDtypeStruct(
        _branched_sig_shape(path.shape, dimension, max_nodes, time_aug, lead_lag, planar), path.dtype)
    call_kwargs = dict(max_nodes=np.int64(max_nodes), time_aug=np.bool_(time_aug), lead_lag=np.bool_(lead_lag),
                       end_time=np.float64(end_time), n_jobs=np.int64(n_jobs), planar=np.bool_(planar))
    return _make_ffi_call("branched_sig", (path,), out_type, call_kwargs)


def branched_sig_backprop_ffi_call(path, bsig, cotangent, max_nodes, time_aug, lead_lag, end_time, n_jobs, planar):
    _check_same_dtype(path, bsig, cotangent)
    out_type = jax.ShapeDtypeStruct(path.shape, path.dtype)
    call_kwargs = dict(max_nodes=np.int64(max_nodes), time_aug=np.bool_(time_aug), lead_lag=np.bool_(lead_lag),
                       end_time=np.float64(end_time), n_jobs=np.int64(n_jobs), planar=np.bool_(planar))
    return _make_ffi_call("branched_sig_backprop", (path, bsig, cotangent), out_type, call_kwargs)


# ---------------------------------------------------------------------------
# branched_sig_combine
# ---------------------------------------------------------------------------

def branched_sig_combine_ffi_call(bsig1, bsig2, dimension, max_nodes, n_jobs, planar):
    _check_same_dtype(bsig1, bsig2)
    out_type = jax.ShapeDtypeStruct(bsig1.shape, bsig1.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), max_nodes=np.int64(max_nodes),
                       n_jobs=np.int64(n_jobs), planar=np.bool_(planar))
    return _make_ffi_call("branched_sig_combine", (bsig1, bsig2), out_type, call_kwargs)


def branched_sig_combine_backprop_ffi_call(cotangent, bsig1, bsig2, dimension, max_nodes, n_jobs, planar):
    _check_same_dtype(cotangent, bsig1, bsig2)
    grad_type = jax.ShapeDtypeStruct(bsig1.shape, bsig1.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), max_nodes=np.int64(max_nodes),
                       n_jobs=np.int64(n_jobs), planar=np.bool_(planar))
    return _make_ffi_call("branched_sig_combine_backprop", (cotangent, bsig1, bsig2), (grad_type, grad_type), call_kwargs)


# ---------------------------------------------------------------------------
# branched_sig_to_log_sig
# ---------------------------------------------------------------------------

def branched_sig_to_log_sig_ffi_call(bsig, dimension, max_nodes, n_jobs, planar):
    _normalize_dtype(bsig.dtype)
    out_type = jax.ShapeDtypeStruct(bsig.shape, bsig.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), max_nodes=np.int64(max_nodes),
                       n_jobs=np.int64(n_jobs), planar=np.bool_(planar))
    return _make_ffi_call("branched_sig_to_log_sig", (bsig,), out_type, call_kwargs)


def branched_sig_to_log_sig_backprop_ffi_call(bsig, cotangent, dimension, max_nodes, n_jobs, planar):
    _check_same_dtype(bsig, cotangent)
    out_type = jax.ShapeDtypeStruct(bsig.shape, bsig.dtype)
    call_kwargs = dict(dimension=np.int64(dimension), max_nodes=np.int64(max_nodes),
                       n_jobs=np.int64(n_jobs), planar=np.bool_(planar))
    return _make_ffi_call("branched_sig_to_log_sig_backprop", (bsig, cotangent), out_type, call_kwargs)
