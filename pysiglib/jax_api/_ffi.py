from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

from ..load_siglib import BUILT_WITH_CUDA, BUILT_WITH_JAX_FFI, SYSTEM
from ..sig_length import sig_length

import jax


_FFI_LIB = None
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


def _ffi_library_filename() -> str:
    if SYSTEM == "Windows":
        return "pysiglib_jax_ffi.dll"
    if SYSTEM == "Linux":
        return "libpysiglib_jax_ffi.so"
    if SYSTEM == "Darwin":
        return "libpysiglib_jax_ffi.dylib"
    raise RuntimeError(f"Unsupported platform: {SYSTEM}")


def _load_ffi_library():
    global _FFI_LIB

    if _FFI_LIB is not None:
        return _FFI_LIB

    if not BUILT_WITH_JAX_FFI:
        raise RuntimeError(
            "pySigLib was built without JAX FFI support. Rebuild after installing jaxlib "
            "so the XLA FFI headers are available at build time."
        )

    lib_path = _find_native_lib(_ffi_library_filename())
    if SYSTEM in {"Windows", "Linux"}:
        _FFI_LIB = ctypes.CDLL(lib_path, winmode=0)
    else:
        _FFI_LIB = ctypes.CDLL(lib_path)
    return _FFI_LIB


def _normalize_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise TypeError(f"Only float32 and float64 are supported, got {dtype}.")
    return dtype


def _target_name(op: str, platform: str) -> str:
    return _TARGETS[op][platform][0]


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    lib = _load_ffi_library()

    for op_targets in _TARGETS.values():
        target_name, symbol_name = op_targets["cpu"]
        jax.ffi.register_ffi_target(
            target_name,
            jax.ffi.pycapsule(getattr(lib, symbol_name)),
            platform="cpu",
        )

    if BUILT_WITH_CUDA:
        for op_targets in _TARGETS.values():
            target_name, symbol_name = op_targets["cuda"]
            jax.ffi.register_ffi_target(
                target_name,
                jax.ffi.pycapsule(getattr(lib, symbol_name)),
                platform="CUDA",
            )

    _REGISTERED = True


def _sig_shape(path_shape, degree: int, time_aug: bool, lead_lag: bool) -> tuple[int, ...]:
    dimension = path_shape[-1]
    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)
    out_len = sig_length(aug_dimension, degree)
    if out_len == 0:
        raise ValueError("Signature length overflow.")
    return (*path_shape[:-2], out_len)


def sig_ffi_call(path, degree, time_aug, lead_lag, end_time, horner, n_jobs):
    _normalize_dtype(path.dtype)
    out_type = jax.ShapeDtypeStruct(
        _sig_shape(path.shape, degree, time_aug, lead_lag),
        path.dtype,
    )

    call_kwargs = dict(
        degree=np.int64(degree),
        time_aug=np.bool_(time_aug),
        lead_lag=np.bool_(lead_lag),
        end_time=np.float64(end_time),
        horner=np.bool_(horner),
        n_jobs=np.int64(n_jobs),
    )

    cpu_call = jax.ffi.ffi_call(
        _target_name("sig", "cpu"),
        out_type,
        vmap_method="sequential",
    )
    if BUILT_WITH_CUDA:
        cuda_call = jax.ffi.ffi_call(
            _target_name("sig", "cuda"),
            out_type,
            vmap_method="sequential",
        )
        return jax.lax.platform_dependent(
            path,
            cpu=lambda x: cpu_call(x, **call_kwargs),
            cuda=lambda x: cuda_call(x, **call_kwargs),
        )

    return cpu_call(path, **call_kwargs)


def sig_backprop_ffi_call(path, sig_, cotangent, degree, time_aug, lead_lag, end_time, n_jobs):
    _normalize_dtype(path.dtype)
    out_type = jax.ShapeDtypeStruct(path.shape, path.dtype)

    call_kwargs = dict(
        degree=np.int64(degree),
        time_aug=np.bool_(time_aug),
        lead_lag=np.bool_(lead_lag),
        end_time=np.float64(end_time),
        n_jobs=np.int64(n_jobs),
    )

    cpu_call = jax.ffi.ffi_call(
        _target_name("sig_backprop", "cpu"),
        out_type,
        vmap_method="sequential",
    )
    if BUILT_WITH_CUDA:
        cuda_call = jax.ffi.ffi_call(
            _target_name("sig_backprop", "cuda"),
            out_type,
            vmap_method="sequential",
        )
        return jax.lax.platform_dependent(
            path,
            sig_,
            cotangent,
            cpu=lambda x, y, z: cpu_call(x, y, z, **call_kwargs),
            cuda=lambda x, y, z: cuda_call(x, y, z, **call_kwargs),
        )

    return cpu_call(path, sig_, cotangent, **call_kwargs)
