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
# Figure out how pysiglib was built
######################################################

try:
    from ._config import SYSTEM, BUILT_WITH_AVX
except ImportError as exc:
    raise RuntimeError("Could not import configuration properties from _config.py - package may not have been built correctly.") from exc

try:
    from ._config import BUILT_WITH_JAX_FFI
except ImportError:
    BUILT_WITH_JAX_FFI = False

if SYSTEM != platform.system():
    raise RuntimeError("System on which pySigLib was built does not match the current system - package may not have been built correctly.")

######################################################
# Load cpsig + discover cusig via the pysiglib-cuda plugin
######################################################

def native_lib_filename(base_name):
    """Platform-specific filename for a shared library (no directory prefix)."""
    if SYSTEM == 'Windows':
        return f'{base_name}.dll'
    if SYSTEM == 'Linux':
        return f'lib{base_name}.so'
    if SYSTEM == 'Darwin':
        return f'lib{base_name}.dylib'
    raise RuntimeError(f"Unsupported OS: {SYSTEM}")


# winmode=0 on Windows - see https://github.com/NVIDIA/warp/issues/24
def _load_native_lib(directory, base_name):
    path = os.path.join(directory, native_lib_filename(base_name))
    if SYSTEM == 'Windows':
        return ctypes.CDLL(path, winmode=0)
    return ctypes.CDLL(path)


DIR_ = os.path.dirname(sys.modules['pysiglib'].__file__)
CPSIG = _load_native_lib(DIR_, 'cpsig')

CUSIG = None
BUILT_WITH_CUDA = False
PYSIGLIB_CUDA_DIR = None
try:
    import pysiglib_cuda as _cuda_plugin
except ImportError:
    _cuda_plugin = None
if _cuda_plugin is not None:
    from pysiglib._version import __version__ as _pysiglib_version
    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            _installed_plugin = _pkg_version("pysiglib-cuda")
        except PackageNotFoundError:
            _installed_plugin = _pysiglib_version  # editable install: trust
    except ImportError:
        _installed_plugin = _pysiglib_version
    if _installed_plugin != _pysiglib_version:
        raise ImportError(
            "pysiglib-cuda " + _installed_plugin + " is installed but pysiglib " +
            _pysiglib_version + " requires pysiglib-cuda==" + _pysiglib_version +
            ". Reinstall the matching plugin (`pip install --force-reinstall "
            "pysiglib[cuda]`) or uninstall pysiglib-cuda to disable the CUDA "
            "backend."
        )
    PYSIGLIB_CUDA_DIR = os.path.dirname(_cuda_plugin.__file__)
    try:
        CUSIG = _load_native_lib(PYSIGLIB_CUDA_DIR, 'cusig')
        BUILT_WITH_CUDA = True
    except OSError:
        pass

######################################################
# Set argtypes and restypes for all imported functions
######################################################

######################################################
# transform_path
######################################################

CPSIG.transform_path_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_int)
CPSIG.transform_path_f.restype = c_int

CPSIG.transform_path_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_int)
CPSIG.transform_path_d.restype = c_int

######################################################
# transform_path_backprop
######################################################

CPSIG.transform_path_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_int)
CPSIG.transform_path_backprop_f.restype = c_int

CPSIG.transform_path_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_int)
CPSIG.transform_path_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # transform_path_cuda
    ######################################################

    CUSIG.transform_path_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float)
    CUSIG.transform_path_cuda_f.restype = c_int

    CUSIG.transform_path_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double)
    CUSIG.transform_path_cuda_d.restype = c_int

    ######################################################
    # transform_path_backprop_cuda
    ######################################################

    CUSIG.transform_path_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float)
    CUSIG.transform_path_backprop_cuda_f.restype = c_int

    CUSIG.transform_path_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double)
    CUSIG.transform_path_backprop_cuda_d.restype = c_int

######################################################
# sig_length
######################################################

CPSIG.sig_length.argtypes = (c_uint64, c_uint64)
CPSIG.sig_length.restype = c_uint64

######################################################
# log_sig_combine
######################################################

CPSIG.log_sig_combine_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_combine_f.restype = c_int

CPSIG.log_sig_combine_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_combine_d.restype = c_int

######################################################
# log_sig_from_path
######################################################

CPSIG.log_sig_from_path_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_from_path_f.restype = c_int

CPSIG.log_sig_from_path_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_from_path_d.restype = c_int

######################################################
# log_sig_from_path_backprop
######################################################

CPSIG.log_sig_from_path_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_from_path_backprop_f.restype = c_int

CPSIG.log_sig_from_path_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_from_path_backprop_d.restype = c_int

######################################################
# log_sig_combine_backprop
######################################################

CPSIG.log_sig_combine_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_combine_backprop_f.restype = c_int

CPSIG.log_sig_combine_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_combine_backprop_d.restype = c_int

######################################################
# sig_combine
######################################################

CPSIG.sig_combine_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_combine_f.restype = c_int

CPSIG.sig_combine_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_combine_d.restype = c_int

######################################################
# sig_join
######################################################


CPSIG.sig_join_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int)
CPSIG.sig_join_f.restype = c_int

CPSIG.sig_join_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int)
CPSIG.sig_join_d.restype = c_int

######################################################
# sig_join_backprop
######################################################


CPSIG.sig_join_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int)
CPSIG.sig_join_backprop_f.restype = c_int

CPSIG.sig_join_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int)
CPSIG.sig_join_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # sig_combine_cuda
    ######################################################

    CUSIG.sig_combine_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_combine_cuda_f.restype = c_int

    CUSIG.sig_combine_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_combine_cuda_d.restype = c_int

    ######################################################
    # sig_join_cuda
    ######################################################

    CUSIG.sig_join_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.sig_join_cuda_f.restype = c_int
    CUSIG.sig_join_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.sig_join_cuda_d.restype = c_int

    CUSIG.sig_join_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.sig_join_backprop_cuda_f.restype = c_int
    CUSIG.sig_join_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.sig_join_backprop_cuda_d.restype = c_int

    ######################################################
    # sig_combine_backprop_cuda
    ######################################################

    CUSIG.sig_combine_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_combine_backprop_cuda_f.restype = c_int

    CUSIG.sig_combine_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_combine_backprop_cuda_d.restype = c_int

    ######################################################
    # sig_to_log_sig_cuda
    ######################################################

    CUSIG.sig_to_log_sig_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.sig_to_log_sig_cuda_f.restype = c_int

    CUSIG.sig_to_log_sig_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.sig_to_log_sig_cuda_d.restype = c_int

    ######################################################
    # sig_to_log_sig_backprop_cuda
    ######################################################

    CUSIG.sig_to_log_sig_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.sig_to_log_sig_backprop_cuda_f.restype = c_int

    CUSIG.sig_to_log_sig_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.sig_to_log_sig_backprop_cuda_d.restype = c_int

    ######################################################
    # prepare_log_sig_cuda
    ######################################################

    CUSIG.prepare_log_sig_cuda.argtypes = (c_uint64, c_uint64, c_int, c_bool)
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

    CUSIG.sig_coef_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_coef_cuda_f.restype = c_int

    CUSIG.sig_coef_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_coef_cuda_d.restype = c_int

    ######################################################
    # sig_coef_backprop_cuda
    ######################################################

    CUSIG.sig_coef_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64)
    CUSIG.sig_coef_backprop_cuda_f.restype = c_int

    CUSIG.sig_coef_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64)
    CUSIG.sig_coef_backprop_cuda_d.restype = c_int

    ######################################################
    # log_sig_combine_cuda
    ######################################################

    CUSIG.log_sig_combine_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_combine_cuda_f.restype = c_int

    CUSIG.log_sig_combine_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_combine_cuda_d.restype = c_int

    ######################################################
    # log_sig_from_path_cuda
    ######################################################

    CUSIG.log_sig_from_path_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_from_path_cuda_f.restype = c_int

    CUSIG.log_sig_from_path_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_from_path_cuda_d.restype = c_int

    ######################################################
    # log_sig_combine_backprop_cuda
    ######################################################

    CUSIG.log_sig_combine_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_combine_backprop_cuda_f.restype = c_int

    CUSIG.log_sig_combine_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_combine_backprop_cuda_d.restype = c_int

    ######################################################
    # log_sig_from_path_backprop_cuda
    ######################################################

    CUSIG.log_sig_from_path_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_from_path_backprop_cuda_f.restype = c_int

    CUSIG.log_sig_from_path_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_from_path_backprop_cuda_d.restype = c_int

######################################################
# sig_combine_backprop
######################################################

CPSIG.sig_combine_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_combine_backprop_f.restype = c_int

CPSIG.sig_combine_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_combine_backprop_d.restype = c_int




######################################################
# sig_coef
######################################################

CPSIG.sig_coef_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_bool, c_int)
CPSIG.sig_coef_f.restype = c_int

CPSIG.sig_coef_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool, c_int)
CPSIG.sig_coef_d.restype = c_int

######################################################
# sig_coef_backprop
######################################################

CPSIG.sig_coef_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_int)
CPSIG.sig_coef_backprop_f.restype = c_int

CPSIG.sig_coef_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_int)
CPSIG.sig_coef_backprop_d.restype = c_int

######################################################
# signature
######################################################

CPSIG.signature_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_bool, c_bool, c_int, POINTER(c_float), c_uint64, c_uint64, c_uint64)
CPSIG.signature_f.restype = c_int

CPSIG.signature_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool, c_bool, c_int, POINTER(c_double), c_uint64, c_uint64, c_uint64)
CPSIG.signature_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # signature_cuda
    ######################################################

    CUSIG.signature_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_bool, c_bool, POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.signature_cuda_f.restype = c_int

    CUSIG.signature_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool, c_bool, POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.signature_cuda_d.restype = c_int

######################################################
# sig_backprop
######################################################

CPSIG.sig_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_bool, c_int, POINTER(c_float), c_uint64, c_uint64, c_uint64)
CPSIG.sig_backprop_f.restype = c_int

CPSIG.sig_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool, c_int, POINTER(c_double), c_uint64, c_uint64, c_uint64)
CPSIG.sig_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    ######################################################
    # sig_backprop_cuda
    ######################################################

    CUSIG.sig_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_bool, POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.sig_backprop_cuda_f.restype = c_int

    CUSIG.sig_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool, POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.sig_backprop_cuda_d.restype = c_int

######################################################
# log_sig_length
######################################################

CPSIG.log_sig_length.argtypes = (c_uint64, c_uint64)
CPSIG.log_sig_length.restype = c_uint64

######################################################
# set_cache_dir
######################################################

CPSIG.set_cache_dir.argtypes = (c_char_p,)
CPSIG.set_cache_dir.restype = c_int

######################################################
# prepare_log_sig
######################################################

CPSIG.prepare_log_sig.argtypes = (c_uint64, c_uint64, c_int, c_bool)
CPSIG.prepare_log_sig.restype = c_int

######################################################
# clear_cache
######################################################

CPSIG.clear_cache.argtypes = (c_bool,)
CPSIG.clear_cache.restype = c_int

######################################################
# sig_to_log_sig
######################################################

CPSIG.sig_to_log_sig_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.sig_to_log_sig_f.restype = c_int

CPSIG.sig_to_log_sig_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.sig_to_log_sig_d.restype = c_int

######################################################
# sig_to_log_sig_backprop
######################################################

CPSIG.sig_to_log_sig_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.sig_to_log_sig_backprop_f.restype = c_int

CPSIG.sig_to_log_sig_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.sig_to_log_sig_backprop_d.restype = c_int

######################################################
# sig_kernel
######################################################

CPSIG.sig_kernel_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_f.restype = c_int

CPSIG.sig_kernel_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_d.restype = c_int

if BUILT_WITH_CUDA:
    CUSIG.sig_kernel_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_cuda_f.restype = c_int

    CUSIG.sig_kernel_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_cuda_d.restype = c_int

######################################################
# branched_sig_kernel
######################################################

CPSIG.branched_sig_kernel_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.branched_sig_kernel_f.restype = c_int

CPSIG.branched_sig_kernel_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.branched_sig_kernel_d.restype = c_int

if BUILT_WITH_CUDA:
    CUSIG.branched_sig_kernel_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.branched_sig_kernel_cuda_f.restype = c_int

    CUSIG.branched_sig_kernel_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.branched_sig_kernel_cuda_d.restype = c_int

######################################################
# sig_kernel_backprop
######################################################

CPSIG.sig_kernel_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_backprop_f.restype = c_int

CPSIG.sig_kernel_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_backprop_d.restype = c_int

######################################################
# sig_kernel_log_pde
######################################################

CPSIG.sig_kernel_log_pde_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_log_pde_f.restype = c_int

CPSIG.sig_kernel_log_pde_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_log_pde_d.restype = c_int

CPSIG.sig_kernel_log_pde_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_log_pde_backprop_f.restype = c_int

CPSIG.sig_kernel_log_pde_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.sig_kernel_log_pde_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    CUSIG.sig_kernel_log_pde_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_log_pde_cuda_f.restype = c_int

    CUSIG.sig_kernel_log_pde_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_log_pde_cuda_d.restype = c_int

    CUSIG.sig_kernel_log_pde_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_log_pde_backprop_cuda_f.restype = c_int

    CUSIG.sig_kernel_log_pde_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_log_pde_backprop_cuda_d.restype = c_int

    CUSIG.sig_kernel_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_backprop_cuda_f.restype = c_int

    CUSIG.sig_kernel_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.sig_kernel_backprop_cuda_d.restype = c_int

######################################################
# branched_sig_kernel_backprop
######################################################

CPSIG.branched_sig_kernel_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.branched_sig_kernel_backprop_f.restype = c_int

CPSIG.branched_sig_kernel_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.branched_sig_kernel_backprop_d.restype = c_int

if BUILT_WITH_CUDA:
    CUSIG.branched_sig_kernel_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.branched_sig_kernel_backprop_cuda_f.restype = c_int

    CUSIG.branched_sig_kernel_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.branched_sig_kernel_backprop_cuda_d.restype = c_int

######################################################
# branched_sig
######################################################

CPSIG.prepare_branched_sig.argtypes = (c_uint64, c_uint64, c_bool, c_bool)
CPSIG.prepare_branched_sig.restype = c_int

CPSIG.branched_sig_length.argtypes = (c_uint64, c_uint64, c_bool)
CPSIG.branched_sig_length.restype = c_uint64


CPSIG.branched_sig_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool, c_float, c_bool, c_bool, POINTER(c_float), c_uint64, c_uint64, c_uint64)
CPSIG.branched_sig_f.restype = c_int

CPSIG.branched_sig_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool, c_double, c_bool, c_bool, POINTER(c_double), c_uint64, c_uint64, c_uint64)
CPSIG.branched_sig_d.restype = c_int


CPSIG.branched_sig_combine_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_combine_f.restype = c_int

CPSIG.branched_sig_combine_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_combine_d.restype = c_int

CPSIG.branched_sig_to_log_sig_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_to_log_sig_f.restype = c_int

CPSIG.branched_sig_to_log_sig_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_to_log_sig_d.restype = c_int

CPSIG.branched_sig_to_log_sig_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_to_log_sig_backprop_f.restype = c_int

CPSIG.branched_sig_to_log_sig_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_to_log_sig_backprop_d.restype = c_int

CPSIG.branched_sig_combine_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_combine_backprop_f.restype = c_int

CPSIG.branched_sig_combine_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.branched_sig_combine_backprop_d.restype = c_int


CPSIG.branched_sig_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool, c_float, c_bool, c_bool, POINTER(c_float), c_uint64, c_uint64, c_uint64)
CPSIG.branched_sig_backprop_f.restype = c_int

CPSIG.branched_sig_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool, c_double, c_bool, c_bool, POINTER(c_double), c_uint64, c_uint64, c_uint64)
CPSIG.branched_sig_backprop_d.restype = c_int

if BUILT_WITH_CUDA:

    CUSIG.branched_sig_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_bool, c_bool, POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.branched_sig_cuda_f.restype = c_int

    CUSIG.branched_sig_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool, c_bool, POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.branched_sig_cuda_d.restype = c_int

    CUSIG.branched_sig_combine_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_combine_cuda_f.restype = c_int
    CUSIG.branched_sig_combine_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_combine_cuda_d.restype = c_int

    CUSIG.branched_sig_combine_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_combine_backprop_cuda_f.restype = c_int
    CUSIG.branched_sig_combine_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_combine_backprop_cuda_d.restype = c_int

    CUSIG.branched_sig_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_float, c_bool, c_bool, POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.branched_sig_backprop_cuda_f.restype = c_int

    CUSIG.branched_sig_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_double, c_bool, c_bool, POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.branched_sig_backprop_cuda_d.restype = c_int

    CUSIG.branched_sig_to_log_sig_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_to_log_sig_cuda_f.restype = c_int

    CUSIG.branched_sig_to_log_sig_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_to_log_sig_cuda_d.restype = c_int

    CUSIG.branched_sig_to_log_sig_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_to_log_sig_backprop_cuda_f.restype = c_int

    CUSIG.branched_sig_to_log_sig_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.branched_sig_to_log_sig_backprop_cuda_d.restype = c_int

    ######################################################
    # logsig_to_sig_cuda
    ######################################################

    CUSIG.logsig_to_sig_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.logsig_to_sig_cuda_f.restype = c_int
    CUSIG.logsig_to_sig_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.logsig_to_sig_cuda_d.restype = c_int
    CUSIG.logsig_to_sig_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.logsig_to_sig_backprop_cuda_f.restype = c_int
    CUSIG.logsig_to_sig_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int, c_bool)
    CUSIG.logsig_to_sig_backprop_cuda_d.restype = c_int

    ######################################################
    # linear_sig_cuda
    ######################################################

    CUSIG.linear_sig_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.linear_sig_cuda_f.restype = c_int
    CUSIG.linear_sig_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool)
    CUSIG.linear_sig_cuda_d.restype = c_int

    ######################################################
    # log_sig_join_cuda
    ######################################################

    CUSIG.log_sig_join_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_join_cuda_f.restype = c_int
    CUSIG.log_sig_join_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_join_cuda_d.restype = c_int
    CUSIG.log_sig_join_backprop_cuda_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_join_backprop_cuda_f.restype = c_int
    CUSIG.log_sig_join_backprop_cuda_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64)
    CUSIG.log_sig_join_backprop_cuda_d.restype = c_int


######################################################
# linear_sig
######################################################

CPSIG.linear_sig_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.linear_sig_f.restype = c_int
CPSIG.linear_sig_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_int)
CPSIG.linear_sig_d.restype = c_int

######################################################
# log_sig_join
######################################################

CPSIG.log_sig_join_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_join_f.restype = c_int
CPSIG.log_sig_join_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_join_d.restype = c_int
CPSIG.log_sig_join_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_join_backprop_f.restype = c_int
CPSIG.log_sig_join_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int)
CPSIG.log_sig_join_backprop_d.restype = c_int

######################################################
# logsig_to_sig
######################################################

CPSIG.logsig_to_sig_f.argtypes = (POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.logsig_to_sig_f.restype = c_int
CPSIG.logsig_to_sig_d.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.logsig_to_sig_d.restype = c_int
CPSIG.logsig_to_sig_backprop_f.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.logsig_to_sig_backprop_f.restype = c_int
CPSIG.logsig_to_sig_backprop_d.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int)
CPSIG.logsig_to_sig_backprop_d.restype = c_int

######################################################
# shutdown
######################################################

CPSIG.cpsig_shutdown.argtypes = ()
CPSIG.cpsig_shutdown.restype = None

if BUILT_WITH_CUDA:
    CUSIG.cusig_shutdown.argtypes = ()
    CUSIG.cusig_shutdown.restype = None

import atexit

def _shutdown():
    try:
        CPSIG.cpsig_shutdown()
    except Exception:
        pass
    if CUSIG is not None:
        try:
            CUSIG.cusig_shutdown()
        except Exception:
            pass

atexit.register(_shutdown)
