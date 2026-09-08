"""NumPy/Torch storage used by the ctypes native calls. JAX uses FFI."""

from __future__ import annotations
from ctypes import c_float, c_double, c_uint64, POINTER, cast

import numpy as np
from array_api_compat import array_namespace, device, is_torch_array
from ._array import NativeArray, NativeArrayT, dtype_name


def check_native_array(arr: object, name: str) -> None:
    if not isinstance(arr, np.ndarray) and not is_torch_array(arr):
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")


def pointer(arr: NativeArray, ctype=None):
    if ctype is None:
        ctype = {"float32": c_float, "float64": c_double, "uint64": c_uint64}[dtype_name(arr)]
    address = arr.ctypes.data if isinstance(arr, np.ndarray) else arr.data_ptr()
    return cast(address, POINTER(ctype))


def owns_contiguous(arr: NativeArray) -> bool:
    if isinstance(arr, np.ndarray):
        return arr.base is None and arr.flags.c_contiguous
    return arr._base is None and arr.is_contiguous() and all(s != 0 for s in arr.stride())


def contiguous(arr: NativeArrayT) -> NativeArrayT:
    return np.ascontiguousarray(arr) if isinstance(arr, np.ndarray) else arr.contiguous()


def pair_indices(n: int, m: int, symmetric: bool, like: NativeArrayT) -> tuple[NativeArrayT, NativeArrayT]:
    if isinstance(like, np.ndarray):
        return np.triu_indices(n) if symmetric else (np.repeat(np.arange(n), m), np.tile(np.arange(m), n))
    xp = array_namespace(like)
    if symmetric:
        i, j = xp.triu_indices(n, n, device=device(like))
        return i, j
    return xp.arange(n, device=device(like)).repeat_interleave(m), xp.arange(m, device=device(like)).repeat(n)


def add_at(out: NativeArrayT, indices: NativeArrayT, values: NativeArrayT) -> None:
    if isinstance(out, np.ndarray):
        np.add.at(out, indices, values)
    else:
        out.index_add_(0, indices, array_namespace(out).astype(values, out.dtype))
