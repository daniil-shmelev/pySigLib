# Copyright 2025 Daniil Shmelev
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

import inspect
import warnings

import numpy as np
import torch

from .dtypes import SUPPORTED_DTYPES, SUPPORTED_DTYPES_STR

def get_type_str(type_):
    try:
        if hasattr(type_, '__name__'):
            return type_.__name__
        return str(type_)
    except:
        return "UNKNOWN"

def check_type(param, param_name, type_):
    if not isinstance(param, type_):
        raise TypeError(param_name + " must be of type " + get_type_str(type_) + ", got " + get_type_str(type(param)) + " instead")

def check_type_multiple(param, param_name, type_tuple):
    if not isinstance(param, type_tuple):
        type_tuple_str = ' or '.join(get_type_str(type_) for type_ in type_tuple)
        raise TypeError(param_name + " must be of type " + type_tuple_str + ", got " + get_type_str(type(param)) + " instead")

def check_non_neg(param, param_name):
    if param < 0:
        raise ValueError(param_name + " must be a non-negative integer, got " + param_name + " = " + str(param))

def check_pos(param, param_name):
    if param <= 0:
        raise ValueError(param_name + " must be a positive integer, got " + param_name + " = " + str(param))

def check_n_jobs(n_jobs):
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

def check_dtype(arr, arr_name):
    if arr.dtype not in SUPPORTED_DTYPES:
        raise TypeError(arr_name + ".dtype must be " + SUPPORTED_DTYPES_STR + ", got " + str(arr.dtype) + " instead")

def warn(msg):
    frame = inspect.currentframe().f_back
    depth = 2
    while frame and frame.f_globals.get("__name__", "").startswith("pysiglib"):
        frame = frame.f_back
        depth += 1
    warnings.warn(msg, stacklevel=depth)

WARNED_ONCE_ABOUT_MEMORY = False
MEMORY_WARNING = "Detected a non-contiguous or view-based array. Such arrays will be cloned to ensure safe access. To avoid this overhead, pass arrays that are both contiguous and own their data. This warning will only appear once per session."

def ensure_own_contiguous_storage(arr):
    global WARNED_ONCE_ABOUT_MEMORY

    if isinstance(arr, torch.Tensor):
        is_view = arr._base is not None
        has_stride_0 = any(s == 0 for s in arr.stride())
        is_contiguous = arr.is_contiguous()
        if is_view or not is_contiguous or has_stride_0:
            if not WARNED_ONCE_ABOUT_MEMORY:
                warn(MEMORY_WARNING)
                WARNED_ONCE_ABOUT_MEMORY = True
            return arr.clone().contiguous()
        return arr

    if isinstance(arr, np.ndarray):
        owns_data = arr.base is None
        is_contiguous = arr.flags['C_CONTIGUOUS']
        if not owns_data or not is_contiguous:
            if not WARNED_ONCE_ABOUT_MEMORY:
                warn(MEMORY_WARNING)
                WARNED_ONCE_ABOUT_MEMORY = True
            return np.ascontiguousarray(arr.copy())
        return arr

    raise TypeError("Unexpected error in ensure_own_contiguous_storage: arr must be of type torch.Tensor or numpy.ndarray")

def parse_dyadic_order(dyadic_order):
    if isinstance(dyadic_order, tuple) and len(dyadic_order) == 2:
        do1, do2 = dyadic_order
    elif isinstance(dyadic_order, int):
        do1 = do2 = dyadic_order
    else:
        raise TypeError("dyadic_order must be an integer or a tuple of length 2")
    if do1 < 0 or do2 < 0:
        raise ValueError("dyadic_order must be a non-negative integer or tuple of non-negative integers")
    return do1, do2

def dyadic_grid_length(path_length, dyadic_order):
    return ((path_length - 1) << dyadic_order) + 1

def check_log_sig_method(method):
    if method < 0 or method > 3:
        raise ValueError("method must be one of 0, 1, 2 or 3. Got " + str(method) + " instead.")

def check_word(word, max_val, word_name):
    if not isinstance(word, tuple):
        raise TypeError(word_name + " must be a tuple of integers.")

    for i in word:
        if not isinstance(i, int):
            raise TypeError(
                word_name + " must be a tuple of integers. Found element of type " + str(type(i)) + " instead.")
        if not 0 <= i < max_val:
            raise ValueError(word_name + " must only contain letters in the range [0, " + str(max_val) + ")")

def check_word_list(arr, max_val, arr_name):
    if not isinstance(arr, list):
        raise TypeError(arr_name + " must be a list of tuples of integers.")

    for word in arr:
        check_word(word, max_val, "Every word in " + arr_name)

def check_word_or_word_list(param, max_val, param_name):
    try:
        if isinstance(param, tuple):
            check_word(param, max_val, param_name)
            return [param]
        if isinstance(param, list):
            check_word_list(param, max_val, param_name)
            return param
    except Exception as e:
        raise type(e)("Error processing words: " + str(e)) from e

    raise TypeError(param_name + " must be a tuple of integers or a list of tuples of integers.")


def prepend_scalar(arr, value):
    """Prepend a scalar value (0 or 1) along the last axis. Works with numpy and torch."""
    if isinstance(arr, np.ndarray):
        fill = np.full((*arr.shape[:-1], 1), value, dtype=arr.dtype)
        return np.concatenate([fill, arr], axis=-1)
    fill = torch.full((*arr.shape[:-1], 1), value, dtype=arr.dtype, device=arr.device)
    return torch.cat([fill, arr], dim=-1)


def resolve_scalar_term(scalar_term):
    """Resolve scalar_term: None (unset) emits FutureWarning and defaults to True."""
    if scalar_term is None:
        warnings.warn(
            "scalar_term will default to False in pySigLib v4.0. "
            "Pass scalar_term=True to keep the current behavior, "
            "or scalar_term=False to opt in to the new default.",
            FutureWarning, stacklevel=3
        )
        return True
    return scalar_term
