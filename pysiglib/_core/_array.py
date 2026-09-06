"""Shared array types and operations outside the Array API standard."""

from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np
from array_api_compat import array_namespace, device, is_jax_array, is_torch_array

if TYPE_CHECKING:
    from torch import Tensor
    from jax import Array as JaxArray

ArrayT = TypeVar("ArrayT", np.ndarray, "Tensor", "JaxArray")
NativeArray = Union[np.ndarray, "Tensor"]
NativeArrayT = TypeVar("NativeArrayT", np.ndarray, "Tensor")


def dtype_name(arr: Union[NativeArray, JaxArray]) -> str:
    return str(arr.dtype).removeprefix("torch.")


def copy_array(arr: ArrayT) -> ArrayT:
    # torch.asarray(copy=True) detaches gradients; clone preserves autograd.
    if is_torch_array(arr):
        return arr.clone()
    return arr if is_jax_array(arr) else arr.copy()


def set_first(arr: ArrayT, value: float) -> ArrayT:
    if is_jax_array(arr):
        return arr.at[..., 0].set(value)
    arr[..., 0] = value
    return arr


def empty_like_shape(arr: ArrayT, shape: tuple[int, ...]) -> ArrayT:
    return array_namespace(arr).empty(shape, dtype=arr.dtype, device=device(arr))


def require_array(arr: object, array_type: type, name: str) -> None:
    if not isinstance(arr, array_type):
        raise TypeError(
            f"{name} must be a {array_type.__module__}.{array_type.__name__}; use the matching pysiglib backend API"
        )
