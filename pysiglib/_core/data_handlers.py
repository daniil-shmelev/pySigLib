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

from __future__ import annotations
from array_api_compat import array_namespace, device, size
from ._array import dtype_name
from ._storage import check_native_array, pointer

from math import prod

import numpy as np

from .param_checks import (
    check_dtype,
    ensure_own_contiguous_storage,
)
from ..load_siglib import BUILT_WITH_CUDA


def names_str(name_list):
    return ", ".join(name_list)


def _check_cuda_available(device):
    if device != "cpu" and not BUILT_WITH_CUDA:
        raise RuntimeError(
            "pySigLib was built without CUDA support but received a GPU tensor. "
            "Either move your tensor to CPU with .cpu(), or reinstall pySigLib "
            "with the full CUDA toolkit available (ensure CUDA_PATH is set and nvcc is on PATH)."
        )


def make_output(obj, data, shape):
    full_shape = (*obj.batch_shape, *shape)
    xp = data.xp
    obj.xp = xp
    obj.device = data.device
    obj.data = xp.empty(full_shape, dtype=data.array_dtype, device=data.allocation_device)
    obj.data_ptr = pointer(obj.data)


class SigInputHandler:
    """
    Handle input which is (shaped like) a signature or a batch of signatures
    """

    def __init__(self, sig_, sig_len, param_name):
        check_native_array(sig_, param_name)
        self.sig = ensure_own_contiguous_storage(sig_)
        check_dtype(self.sig, param_name)

        if len(self.sig.shape) < 1:
            raise ValueError(param_name + " must have at least rank 1, got rank 0.")

        self.batch_shape = tuple(self.sig.shape[:-1])
        self.batch_size = prod(self.batch_shape) if self.batch_shape else 1
        length = self.sig.shape[-1]

        if length == sig_len + 1:
            self.sig = ensure_own_contiguous_storage(self.sig[..., 1:])
            length = sig_len

        if length != sig_len:
            raise ValueError(param_name + " is of incorrect length. Expected " + str(sig_len) + ", got " + str(length))

        _set_storage(self, self.sig)


class MultipleSigInputHandler:
    """
    Handle multiple inputs which are (shaped like) signatures or batches of signatures
    """

    def __init__(self, sig_list, sig_len, sig_name_list):
        self.data = [SigInputHandler(sig_, sig_len, sig_name) for sig_, sig_name in zip(sig_list, sig_name_list)]
        self.sig = [d.sig for d in self.data]

        if not all(d.type_ == self.data[0].type_ for d in self.data):
            raise ValueError(names_str(sig_name_list) + " must all be numpy arrays or both torch arrays")

        if not all(d.dtype == self.data[0].dtype for d in self.data):
            raise ValueError(names_str(sig_name_list) + " must have the same dtype")

        if not all(d.batch_shape == self.data[0].batch_shape for d in self.data):
            raise ValueError(names_str(sig_name_list) + " have different batch shapes")

        if not all(d.device == self.data[0].device for d in self.data):
            raise ValueError(names_str(sig_name_list) + " must be on the same device")

        self.dtype = self.data[0].dtype
        self.xp = self.data[0].xp
        self.array_dtype = self.data[0].array_dtype
        self.allocation_device = self.data[0].allocation_device
        self.batch_shape = self.data[0].batch_shape
        self.batch_size = self.data[0].batch_size
        self.type_ = self.data[0].type_
        self.sig_ptr = [d.data_ptr for d in self.data]

        self.device = self.data[0].device


class SigOutputHandler:
    """
    Handle output which is (shaped like) a signature or a batch of signatures
    """
    def __init__(self, data, sig_len):
        self.batch_shape = data.batch_shape
        self.batch_size = data.batch_size
        self.type_ = data.type_
        self.dtype = data.dtype
        make_output(self, data, (sig_len,))


class PathInputHandler:
    """
    Handle input which is (shaped like) a path or a batch of paths
    """

    def __init__(self, path_, time_aug, lead_lag, end_time, param_name):
        self.param_name = param_name
        check_native_array(path_, param_name)
        self.path = ensure_own_contiguous_storage(path_)
        check_dtype(self.path, param_name)

        self.time_aug = time_aug
        self.lead_lag = lead_lag
        self.end_time = end_time

        if len(self.path.shape) < 2:
            raise ValueError(
                self.param_name + " must have at least rank 2, got rank " + str(len(self.path.shape)) + ".")

        self.batch_shape = tuple(self.path.shape[:-2])
        self.batch_size = prod(self.batch_shape) if self.batch_shape else 1
        self.data_length = self.path.shape[-2]
        self.data_dimension = self.path.shape[-1]

        if self.data_dimension == 0:
            raise ValueError(
                self.param_name + " has 0 channels (dimension). Path dimension must be at least 1.")

        _set_storage(self, self.path)

        self.length, self.dimension = self.transformed_dims()
        _check_cuda_available(self.device)

    def transformed_dims(self):
        length_ = self.data_length
        dimension_ = self.data_dimension
        if self.lead_lag:
            length_ = 2 * length_ - 1
            dimension_ *= 2
        if self.time_aug:
            dimension_ += 1
        return length_, dimension_


class CorrectionInputHandler:
    """
    Handle correction levels for signature APIs.
    """

    def __init__(self, correction, path_data, degree):
        self.correction = None
        self.length = 0
        self.data_ptr = None
        self.batch_stride = 0
        self.segment_stride = 0

        if correction is None:
            return

        check_native_array(correction, 'correction')
        if isinstance(correction, np.ndarray) != (path_data.type_ == "numpy"):
            raise ValueError("correction must have the same array type as path")

        self.correction = ensure_own_contiguous_storage(correction)
        check_dtype(self.correction, "correction")

        _set_storage(self, self.correction)

        if self.dtype != path_data.dtype:
            raise ValueError("correction and path must have the same dtype")
        if self.device != path_data.device:
            raise ValueError("correction and path must be on the same device")

        if size(self.correction) == 0:
            return

        if path_data.lead_lag:
            raise ValueError("correction cannot be used with lead_lag=True")

        corr_shape = tuple(self.correction.shape)
        if len(corr_shape) == 0:
            raise ValueError(
                "correction shape must be (C,), (path.shape[-2] - 1, C), or "
                "path.shape[:-2] + (path.shape[-2] - 1, C)"
            )

        segments = max(path_data.data_length - 1, 0)
        if len(corr_shape) == 1:
            self.length = corr_shape[0]
            self.batch_stride = 0
            self.segment_stride = 0
        elif len(corr_shape) == 2 and corr_shape[0] == segments:
            self.length = corr_shape[1]
            self.batch_stride = 0
            self.segment_stride = self.length
        else:
            expected_shape = (*path_data.batch_shape, segments, corr_shape[-1])
            if corr_shape != expected_shape:
                raise ValueError(
                    "correction shape must be (C,), (path.shape[-2] - 1, C), or " +
                    str(expected_shape) + " for path shape " + str(tuple(path_data.path.shape))
                )
            self.length = corr_shape[-1]
            self.batch_stride = segments * self.length
            self.segment_stride = self.length

        _infer_correction_degree(path_data.data_dimension, degree, self.length)


def _infer_correction_degree(dimension, degree, length):
    if length == 0:
        return 1
    if degree < 2:
        raise ValueError("correction must be empty when degree < 2")

    offset = 0
    level_size = dimension
    for level in range(2, degree + 1):
        level_size *= dimension
        offset += level_size
        if offset == length:
            return level
        if offset > length:
            break
    raise ValueError("correction length must be a prefix of tensor levels 2..degree")


class MultiplePathInputHandler:
    """
    Handle multiple inputs which are (shaped like) paths or a batch of paths
    """

    def __init__(
        self, path_list, time_aug, lead_lag, end_time, path_names, check_batch=True
    ):
        self.data = [PathInputHandler(p, time_aug, lead_lag, end_time, n) for p,n in zip(path_list, path_names)]
        self.path = [d.path for d in self.data]
        self.length = [d.length for d in self.data]

        if not all(d.type_ == self.data[0].type_ for d in self.data):
            raise ValueError(names_str(path_names) + " must all be numpy arrays or both torch arrays")

        if not all(d.dtype == self.data[0].dtype for d in self.data):
            raise ValueError(names_str(path_names) + " must have the same dtype")

        if check_batch:
            if not all(d.batch_shape == self.data[0].batch_shape for d in self.data):
                raise ValueError(names_str(path_names) + " have different batch shapes")

        if not all(d.data_dimension == self.data[0].data_dimension for d in self.data):
            raise ValueError(names_str(path_names) + " have different dimensions")

        if not all(d.device == self.data[0].device for d in self.data):
            raise ValueError(names_str(path_names) + " must be on the same device")

        self.dtype = self.data[0].dtype
        self.xp = self.data[0].xp
        self.array_dtype = self.data[0].array_dtype
        self.allocation_device = self.data[0].allocation_device
        self.type_ = self.data[0].type_
        self.device = self.data[0].device
        self.data_dimension = self.data[0].data_dimension
        self.dimension = self.data[0].dimension

        if check_batch:
            self.batch_shape = self.data[0].batch_shape
            self.batch_size = self.data[0].batch_size


class ScalarInputHandler:
    """
    Handle input which is (shaped like) a scalar or a batch of scalars
    """

    def __init__(self, data_, is_batch=False, data_name="scalars"):
        self.data_name = data_name
        check_native_array(data_, data_name)
        self.data = ensure_own_contiguous_storage(data_)
        check_dtype(self.data, data_name)

        self.batch_shape = tuple(self.data.shape) if is_batch else ()
        self.batch_size = prod(self.batch_shape) if self.batch_shape else 1

        _set_storage(self, self.data)


class ScalarOutputHandler:
    """
    Handle output which is (shaped like) a scalar or a batch of scalars
    """
    def __init__(self, data):
        self.dtype = data.dtype
        self.type_ = data.type_
        self.batch_shape = data.batch_shape
        self.batch_size = data.batch_size
        make_output(self, data, tuple())


class GridOutputHandler:
    """
    Handle output which is (shaped like) a grid or a batch of grids
    """

    def __init__(self, x_size, y_size, data):
        self.x_size = x_size
        self.y_size = y_size
        self.batch_shape = data.batch_shape
        self.batch_size = data.batch_size
        self.type_ = data.type_
        self.dtype = data.dtype
        make_output(self, data, (self.x_size, self.y_size))

    def transpose(self):
        self.data = array_namespace(self.data).matrix_transpose(self.data)


class PathOutputHandler(GridOutputHandler):
    """
    Handle output which is (shaped like) a path or a batch of paths
    """
    def __init__(self, length, dimension, data):
        super().__init__(length, dimension, data)
        self.length = length
        self.dimension = dimension


def _set_storage(obj, arr):
    obj.xp = array_namespace(arr)
    obj.type_ = "numpy" if isinstance(arr, np.ndarray) else "torch"
    obj.dtype = dtype_name(arr)
    obj.array_dtype = arr.dtype
    obj.allocation_device = device(arr)
    obj.device = getattr(obj.allocation_device, "type", obj.allocation_device)
    _check_cuda_available(obj.device)
    obj.data_ptr = pointer(arr)
