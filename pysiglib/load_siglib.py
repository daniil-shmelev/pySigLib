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

import os
import sys
import platform
import ctypes
from ctypes import c_char_p, c_float, c_double, c_int, c_uint64, c_bool, POINTER

######################################################
# Figure out how pysiglib was built, in particular
# whether CUDA is being used
######################################################

try:
    from ._config import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX
except ImportError as exc:
    raise RuntimeError("Could not import configuration properties from _config.py - package may not have been built correctly.") from exc

if SYSTEM != platform.system():
    raise RuntimeError("System on which pySigLib was built does not match the current system - package may not have been built correctly.")

######################################################
# Load the cpsig and cusig libraries
######################################################

CPSIG, CUSIG = None, None

# winmode = 0 is necessary here
# https://github.com/NVIDIA/warp/issues/24

DIR_ = os.path.dirname(sys.modules['pysiglib'].__file__)

if SYSTEM == 'Windows':
    CPSIG_PATH = os.path.join(DIR_, 'cpsig.dll')
    CPSIG = ctypes.CDLL(CPSIG_PATH, winmode = 0)

    if BUILT_WITH_CUDA:
        CUSIG_PATH = os.path.join(DIR_, 'cusig.dll')
        CUSIG = ctypes.CDLL(CUSIG_PATH, winmode=0)
elif SYSTEM == "Linux":
    CPSIG_PATH = os.path.join(DIR_, 'libcpsig.so')
    CPSIG = ctypes.CDLL(CPSIG_PATH, winmode=0)

    if BUILT_WITH_CUDA:
        CUSIG_PATH = os.path.join(DIR_, 'libcusig.so')
        CUSIG = ctypes.CDLL(CUSIG_PATH, winmode=0)
elif SYSTEM == 'Darwin':
    CPSIG_PATH = os.path.join(DIR_, 'libcpsig.dylib')
    CPSIG = ctypes.CDLL(CPSIG_PATH)
else:
    raise Exception("Unsupported OS during pysiglib.py")

######################################################
# Set argtypes and restypes for all imported functions
######################################################

######################################################
# transform_path
######################################################

CPSIG.transform_path_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float
)
CPSIG.transform_path_f.restype = c_int

CPSIG.transform_path_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double
)
CPSIG.transform_path_d.restype = c_int

######################################################
# batch_transform_path
######################################################

CPSIG.batch_transform_path_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_int
)
CPSIG.batch_transform_path_f.restype = c_int

CPSIG.batch_transform_path_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_int
)
CPSIG.batch_transform_path_d.restype = c_int

######################################################
# transform_path_backprop
######################################################

CPSIG.transform_path_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float
)
CPSIG.transform_path_backprop_f.restype = c_int

CPSIG.transform_path_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double
)
CPSIG.transform_path_backprop_d.restype = c_int

######################################################
# batch_transform_path_backprop
######################################################

CPSIG.batch_transform_path_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_int
)
CPSIG.batch_transform_path_backprop_f.restype = c_int

CPSIG.batch_transform_path_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_int
)
CPSIG.batch_transform_path_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # transform_path_cuda
    ######################################################

    CUSIG.transform_path_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float
    )
    CUSIG.transform_path_cuda_f.restype = c_int

    CUSIG.transform_path_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double
    )
    CUSIG.transform_path_cuda_d.restype = c_int

    ######################################################
    # batch_transform_path_cuda
    ######################################################

    CUSIG.batch_transform_path_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float
    )
    CUSIG.batch_transform_path_cuda_f.restype = c_int

    CUSIG.batch_transform_path_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double
    )
    CUSIG.batch_transform_path_cuda_d.restype = c_int

    ######################################################
    # transform_path_backprop_cuda
    ######################################################

    CUSIG.transform_path_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float
    )
    CUSIG.transform_path_backprop_cuda_f.restype = c_int

    CUSIG.transform_path_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double
    )
    CUSIG.transform_path_backprop_cuda_d.restype = c_int

    ######################################################
    # batch_transform_path_backprop_cuda
    ######################################################

    CUSIG.batch_transform_path_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float
    )
    CUSIG.batch_transform_path_backprop_cuda_f.restype = c_int

    CUSIG.batch_transform_path_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double
    )
    CUSIG.batch_transform_path_backprop_cuda_d.restype = c_int

######################################################
# sig_length
######################################################

CPSIG.sig_length.argtypes = (
    c_uint64,
    c_uint64
)
CPSIG.sig_length.restype = c_uint64

######################################################
# sig_combine
######################################################

CPSIG.sig_combine_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64
)
CPSIG.sig_combine_f.restype = c_int

CPSIG.sig_combine_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64
)
CPSIG.sig_combine_d.restype = c_int

######################################################
# batch_sig_combine
######################################################

CPSIG.batch_sig_combine_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_int
)
CPSIG.batch_sig_combine_f.restype = c_int

CPSIG.batch_sig_combine_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_int
)
CPSIG.batch_sig_combine_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # sig_combine_cuda
    ######################################################

    CUSIG.sig_combine_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64
    )
    CUSIG.sig_combine_cuda_f.restype = c_int

    CUSIG.sig_combine_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64
    )
    CUSIG.sig_combine_cuda_d.restype = c_int

    ######################################################
    # batch_sig_combine_cuda
    ######################################################

    CUSIG.batch_sig_combine_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64
    )
    CUSIG.batch_sig_combine_cuda_f.restype = c_int

    CUSIG.batch_sig_combine_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64
    )
    CUSIG.batch_sig_combine_cuda_d.restype = c_int

    ######################################################
    # sig_combine_backprop_cuda
    ######################################################

    CUSIG.sig_combine_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64
    )
    CUSIG.sig_combine_backprop_cuda_f.restype = c_int

    CUSIG.sig_combine_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64
    )
    CUSIG.sig_combine_backprop_cuda_d.restype = c_int

    ######################################################
    # batch_sig_combine_backprop_cuda
    ######################################################

    CUSIG.batch_sig_combine_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64
    )
    CUSIG.batch_sig_combine_backprop_cuda_f.restype = c_int

    CUSIG.batch_sig_combine_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64
    )
    CUSIG.batch_sig_combine_backprop_cuda_d.restype = c_int

    ######################################################
    # sig_to_log_sig_cuda
    ######################################################

    CUSIG.sig_to_log_sig_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.sig_to_log_sig_cuda_f.restype = c_int

    CUSIG.sig_to_log_sig_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.sig_to_log_sig_cuda_d.restype = c_int

    ######################################################
    # batch_sig_to_log_sig_cuda
    ######################################################

    CUSIG.batch_sig_to_log_sig_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.batch_sig_to_log_sig_cuda_f.restype = c_int

    CUSIG.batch_sig_to_log_sig_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.batch_sig_to_log_sig_cuda_d.restype = c_int

    ######################################################
    # sig_to_log_sig_backprop_cuda
    ######################################################

    CUSIG.sig_to_log_sig_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.sig_to_log_sig_backprop_cuda_f.restype = c_int

    CUSIG.sig_to_log_sig_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.sig_to_log_sig_backprop_cuda_d.restype = c_int

    ######################################################
    # batch_sig_to_log_sig_backprop_cuda
    ######################################################

    CUSIG.batch_sig_to_log_sig_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.batch_sig_to_log_sig_backprop_cuda_f.restype = c_int

    CUSIG.batch_sig_to_log_sig_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_int
    )
    CUSIG.batch_sig_to_log_sig_backprop_cuda_d.restype = c_int

    ######################################################
    # prepare_log_sig_cuda
    ######################################################

    CUSIG.prepare_log_sig_cuda.argtypes = (
        c_uint64,
        c_uint64,
        c_int,
        c_bool
    )
    CUSIG.prepare_log_sig_cuda.restype = c_int

    ######################################################
    # clear_cache_cuda
    ######################################################

    CUSIG.clear_cache_cuda.argtypes = (c_bool,)
    CUSIG.clear_cache_cuda.restype = c_int

    ######################################################
    # set_cache_dir_cuda
    ######################################################

    CUSIG.set_cache_dir_cuda.argtypes = (c_char_p,)
    CUSIG.set_cache_dir_cuda.restype = c_int

    ######################################################
    # sig_coef_cuda
    ######################################################

    CUSIG.sig_coef_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.sig_coef_cuda_f.restype = c_int

    CUSIG.sig_coef_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.sig_coef_cuda_d.restype = c_int

    ######################################################
    # batch_sig_coef_cuda
    ######################################################

    CUSIG.batch_sig_coef_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.batch_sig_coef_cuda_f.restype = c_int

    CUSIG.batch_sig_coef_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.batch_sig_coef_cuda_d.restype = c_int

    ######################################################
    # sig_coef_backprop_cuda
    ######################################################

    CUSIG.sig_coef_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64
    )
    CUSIG.sig_coef_backprop_cuda_f.restype = c_int

    CUSIG.sig_coef_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64
    )
    CUSIG.sig_coef_backprop_cuda_d.restype = c_int

    ######################################################
    # batch_sig_coef_backprop_cuda
    ######################################################

    CUSIG.batch_sig_coef_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64,
        c_uint64
    )
    CUSIG.batch_sig_coef_backprop_cuda_f.restype = c_int

    CUSIG.batch_sig_coef_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_uint64),
        c_uint64,
        c_uint64,
        c_uint64
    )
    CUSIG.batch_sig_coef_backprop_cuda_d.restype = c_int

######################################################
# sig_combine_backprop
######################################################

CPSIG.sig_combine_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64
)
CPSIG.sig_combine_backprop_f.restype = c_int

CPSIG.sig_combine_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64
)
CPSIG.sig_combine_backprop_d.restype = c_int

######################################################
# batch_sig_combine_backprop
######################################################

CPSIG.batch_sig_combine_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_int
)
CPSIG.batch_sig_combine_backprop_f.restype = c_int

CPSIG.batch_sig_combine_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_int
)
CPSIG.batch_sig_combine_backprop_d.restype = c_int

######################################################
# sig_coef
######################################################

CPSIG.sig_coef_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_bool
)
CPSIG.sig_coef_f.restype = c_int

CPSIG.sig_coef_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_bool
)
CPSIG.sig_coef_d.restype = c_int

######################################################
# sig_coef_backprop
######################################################

CPSIG.sig_coef_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float
)
CPSIG.sig_coef_backprop_f.restype = c_int

CPSIG.sig_coef_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double
)
CPSIG.sig_coef_backprop_d.restype = c_int

######################################################
# signature
######################################################

CPSIG.signature_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_bool
)
CPSIG.signature_f.restype = c_int

CPSIG.signature_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_bool
)
CPSIG.signature_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # signature_cuda
    ######################################################

    CUSIG.signature_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float,
        c_bool
    )
    CUSIG.signature_cuda_f.restype = c_int

    CUSIG.signature_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double,
        c_bool
    )
    CUSIG.signature_cuda_d.restype = c_int

######################################################
# sig_backprop
######################################################

CPSIG.sig_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float
)
CPSIG.sig_backprop_f.restype = c_int

CPSIG.sig_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double
)
CPSIG.sig_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # sig_backprop_cuda
    ######################################################

    CUSIG.sig_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float
    )
    CUSIG.sig_backprop_cuda_f.restype = c_int

    CUSIG.sig_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double
    )
    CUSIG.sig_backprop_cuda_d.restype = c_int

######################################################
# batch_sig_coef
######################################################

CPSIG.batch_sig_coef_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_bool,
    c_int
)
CPSIG.batch_sig_coef_f.restype = c_int

CPSIG.batch_sig_coef_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_bool,
    c_int
)
CPSIG.batch_sig_coef_d.restype = c_int

######################################################
# batch_sig_coef_backprop
######################################################

CPSIG.batch_sig_coef_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_int
)
CPSIG.batch_sig_coef_backprop_f.restype = c_int

CPSIG.batch_sig_coef_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_uint64),
    c_uint64,
    POINTER(c_uint64),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_int
)
CPSIG.batch_sig_coef_backprop_d.restype = c_int

######################################################
# batch_signature
######################################################

CPSIG.batch_signature_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_bool,
    c_int
)
CPSIG.batch_signature_f.restype = c_int

CPSIG.batch_signature_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_bool,
    c_int
)
CPSIG.batch_signature_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # batch_signature_cuda
    ######################################################

    CUSIG.batch_signature_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float,
        c_bool
    )
    CUSIG.batch_signature_cuda_f.restype = c_int

    CUSIG.batch_signature_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double,
        c_bool
    )
    CUSIG.batch_signature_cuda_d.restype = c_int

######################################################
# batch_sig_backprop
######################################################

CPSIG.batch_sig_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_float,
    c_int
)
CPSIG.batch_sig_backprop_f.restype = c_int

CPSIG.batch_sig_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_double,
    c_int
)
CPSIG.batch_sig_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # batch_sig_backprop_cuda
    ######################################################

    CUSIG.batch_sig_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_float
    )
    CUSIG.batch_sig_backprop_cuda_f.restype = c_int

    CUSIG.batch_sig_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool,
        c_bool,
        c_double
    )
    CUSIG.batch_sig_backprop_cuda_d.restype = c_int

######################################################
# log_sig_length
######################################################

CPSIG.log_sig_length.argtypes = (
    c_uint64,
    c_uint64
)
CPSIG.log_sig_length.restype = c_uint64

######################################################
# set_cache_dir
######################################################

CPSIG.set_cache_dir.argtypes = (
    c_char_p,
)
CPSIG.set_cache_dir.restype = c_int

######################################################
# prepare_log_sig
######################################################

CPSIG.prepare_log_sig.argtypes = (
    c_uint64,
    c_uint64,
    c_int,
    c_bool
)
CPSIG.prepare_log_sig.restype = c_int

######################################################
# clear_cache
######################################################

CPSIG.clear_cache.argtypes = (
    c_bool,
)
CPSIG.clear_cache.restype = c_int

######################################################
# sig_to_log_sig
######################################################

CPSIG.sig_to_log_sig_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int
)
CPSIG.sig_to_log_sig_f.restype = c_int

CPSIG.sig_to_log_sig_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int
)
CPSIG.sig_to_log_sig_d.restype = c_int

######################################################
# batch_sig_to_log_sig
######################################################

CPSIG.batch_sig_to_log_sig_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int,
    c_int
)
CPSIG.batch_sig_to_log_sig_f.restype = c_int

CPSIG.batch_sig_to_log_sig_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int,
    c_int
)
CPSIG.batch_sig_to_log_sig_d.restype = c_int

######################################################
# sig_to_log_sig_backprop
######################################################

CPSIG.sig_to_log_sig_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int
)
CPSIG.sig_to_log_sig_backprop_f.restype = c_int

CPSIG.sig_to_log_sig_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int
)
CPSIG.sig_to_log_sig_backprop_d.restype = c_int

######################################################
# batch_sig_to_log_sig_backprop
######################################################

CPSIG.batch_sig_to_log_sig_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int,
    c_int
)
CPSIG.batch_sig_to_log_sig_backprop_f.restype = c_int

CPSIG.batch_sig_to_log_sig_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_bool,
    c_int,
    c_int
)
CPSIG.batch_sig_to_log_sig_backprop_d.restype = c_int

######################################################
# batch_sig_kernel
######################################################

CPSIG.batch_sig_kernel_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_int
)
CPSIG.batch_sig_kernel_f.restype = c_int

CPSIG.batch_sig_kernel_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_int
)
CPSIG.batch_sig_kernel_d.restype = c_int

if BUILT_WITH_CUDA:
    CUSIG.batch_sig_kernel_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.batch_sig_kernel_cuda_f.restype = c_int

    CUSIG.batch_sig_kernel_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.batch_sig_kernel_cuda_d.restype = c_int

######################################################
# batch_sig_kernel_backprop
######################################################

CPSIG.batch_sig_kernel_backprop_f.argtypes = (
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_int
)
CPSIG.batch_sig_kernel_backprop_f.restype = c_int

CPSIG.batch_sig_kernel_backprop_d.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64,
    c_bool,
    c_int
)
CPSIG.batch_sig_kernel_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    CUSIG.batch_sig_kernel_backprop_cuda_f.argtypes = (
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.batch_sig_kernel_backprop_cuda_f.restype = c_int

    CUSIG.batch_sig_kernel_backprop_cuda_d.argtypes = (
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_uint64,
        c_bool
    )
    CUSIG.batch_sig_kernel_backprop_cuda_d.restype = c_int
