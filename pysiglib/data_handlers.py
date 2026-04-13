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

from ctypes import POINTER, cast
from math import prod

import numpy as np
import torch

from .param_checks import check_type_multiple, check_dtype, ensure_own_contiguous_storage
from .dtypes import DTYPES
from .load_siglib import BUILT_WITH_CUDA

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
    if obj.type_ == "numpy":
        dtype_ = np.float32 if data.dtype == "float32" else np.float64
        obj.device = "cpu"
        obj.data = np.empty(shape=full_shape, dtype=dtype_)
        obj.data_ptr = obj.data.ctypes.data_as(POINTER(DTYPES[obj.dtype]))
    else:
        dtype_ = torch.float32 if data.dtype == "float32" else torch.float64
        obj.device = data.device if hasattr(data, "device") else "cpu"
        obj.data = torch.empty(size=full_shape, dtype=dtype_, device=obj.device)
        obj.data_ptr = cast(obj.data.data_ptr(), POINTER(DTYPES[obj.dtype]))

class SigInputHandler:
    """
    Handle input which is (shaped like) a signature or a batch of signatures
    """
    def __init__(self, sig_, sig_len, param_name):
        check_type_multiple(sig_, param_name, (np.ndarray, torch.Tensor))
        self.sig = ensure_own_contiguous_storage(sig_)
        check_dtype(self.sig, param_name)

        if len(self.sig.shape) < 1:
            raise ValueError(param_name + " must have at least rank 1, got rank 0.")

        self.batch_shape = tuple(self.sig.shape[:-1])
        self.batch_size = prod(self.batch_shape) if self.batch_shape else 1
        length = self.sig.shape[-1]

        if length != sig_len:
            hint = ""
            if length == sig_len - 1:
                hint = (
                    " (Hint: the input appears to be 1 element short -- was it computed with "
                    "scalar_term=False? Input signatures must always include the leading scalar "
                    "term. Use scalar_term=True when computing input signatures, and only use "
                    "scalar_term=False to control the output format.)"
                )
            raise ValueError(param_name + " is of incorrect length. Expected " + str(sig_len) + ", got " + str(length) + hint)

        if isinstance(self.sig, np.ndarray):
            self.type_ = "numpy"
            self.dtype = str(self.sig.dtype)
            self.device = "cpu"
            self.data_ptr = self.sig.ctypes.data_as(POINTER(DTYPES[self.dtype]))
        else:
            self.type_ = "torch"
            self.dtype = str(self.sig.dtype)[6:]
            self.device = self.sig.device.type
            _check_cuda_available(self.device)
            self.data_ptr = cast(self.sig.data_ptr(), POINTER(DTYPES[self.dtype]))

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
        self.batch_shape = self.data[0].batch_shape
        self.batch_size = self.data[0].batch_size
        self.type_ = self.data[0].type_
        self.sig_ptr = [d.data_ptr for d in self.data]

        # Propagate device info so SigOutputHandler can allocate on the correct device
        if self.type_ == "torch":
            self.device = self.data[0].sig.device.type
        else:
            self.device = "cpu"

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
        check_type_multiple(path_, param_name, (np.ndarray, torch.Tensor))
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

        if isinstance(self.path, np.ndarray):
            self.type_ = "numpy"
            self.dtype = str(self.path.dtype)
            self.data_ptr = self.path.ctypes.data_as(POINTER(DTYPES[self.dtype]))
        elif isinstance(self.path, torch.Tensor):
            self.type_ = "torch"
            self.dtype = str(self.path.dtype)[6:]
            self.data_ptr = cast(self.path.data_ptr(), POINTER(DTYPES[self.dtype]))

        self.length, self.dimension = self.transformed_dims()
        self.device = self.path.device.type if self.type_ == "torch" else "cpu"
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

class MultiplePathInputHandler:
    """
    Handle multiple inputs which are (shaped like) paths or a batch of paths
    """
    def __init__(self, path_list, time_aug, lead_lag, end_time, path_names,
                 check_batch=True):
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
        self.type_ = self.data[0].type_
        self.device = self.path[0].device.type if self.type_ == "torch" else "cpu"
        self.data_dimension = self.data[0].data_dimension
        self.dimension = self.data[0].dimension

        if check_batch:
            self.batch_shape = self.data[0].batch_shape
            self.batch_size = self.data[0].batch_size

class ScalarInputHandler:
    """
    Handle input which is (shaped like) a scalar or a batch of scalars
    """
    def __init__(self, data_, is_batch = False, data_name = "scalars"):
        self.data_name = data_name
        check_type_multiple(data_, data_name, (np.ndarray, torch.Tensor))
        self.data = ensure_own_contiguous_storage(data_)
        check_dtype(self.data, data_name)

        self.batch_shape = tuple(self.data.shape) if is_batch else ()
        self.batch_size = prod(self.batch_shape) if self.batch_shape else 1

        if isinstance(self.data, np.ndarray):
            self.type_ = "numpy"
            self.dtype = str(self.data.dtype)
            self.data_ptr = self.data.ctypes.data_as(POINTER(DTYPES[self.dtype]))
        elif isinstance(self.data, torch.Tensor):
            self.type_ = "torch"
            self.dtype = str(self.data.dtype)[6:]
            self.data_ptr = cast(self.data.data_ptr(), POINTER(DTYPES[self.dtype]))

        self.device = self.data.device.type if self.type_ == "torch" else "cpu"

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
        if self.type_ == "numpy":
            self.data = np.swapaxes(self.data, -2, -1)
        else:
            self.data = torch.transpose(self.data, -2, -1)

class PathOutputHandler(GridOutputHandler):
    """
    Handle output which is (shaped like) a path or a batch of paths
    """
    def __init__(self, length, dimension, data):
        super().__init__(length, dimension, data)
        self.length = length
        self.dimension = dimension

class DeviceToHost:
    """
    If data is on GPU, move to CPU
    """
    def __init__(self, data, names):
        self.type = type(data[0])
        self.device = data[0].device if isinstance(data[0], torch.Tensor) else None

        for i in range(1, len(data)):
            d_type = type(data[i])
            d_device = data[i].device if isinstance(data[i], torch.Tensor) else None

            if d_type != self.type:
                msg = ", ".join(names) + " must all be torch tensors or all be numpy arrays."
                raise ValueError(msg)

            if d_device != self.device:
                msg = ", ".join(names) + " must all be on the same device."
                raise ValueError(msg)

        if self.device is not None:
            self.data = [d.cpu() for d in data]
        else:
            self.data = data
        self.names = names
