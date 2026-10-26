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

import os
import sys
import platform
import ctypes
from ctypes import c_char_p, c_float, c_double, c_int, c_uint64, c_bool, c_void_p, POINTER
from collections.abc import Callable
from ._lasso_types import LassoOptions, LassoShape

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
TBB = None
if SYSTEM == 'Windows':
    tbb_path = os.path.join(DIR_, 'tbb12.dll')
    if os.path.isfile(tbb_path):
        TBB = ctypes.CDLL(tbb_path, winmode=0)
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

if hasattr(CPSIG, "cpsig_last_error_message"):
    CPSIG.cpsig_last_error_message.argtypes = ()
    CPSIG.cpsig_last_error_message.restype = c_char_p
if BUILT_WITH_CUDA and hasattr(CUSIG, "cusig_last_error_message"):
    CUSIG.cusig_last_error_message.argtypes = ()
    CUSIG.cusig_last_error_message.restype = c_char_p


def _bind_float_pair(
    library,
    name,
    argument_types: Callable[[type[c_float | c_double]], tuple[type[object], ...]],
):
    """Bind both precisions once and return the native functions by dtype."""
    if library is None:
        return None
    functions = {}
    for dtype, suffix, scalar in (
        ("float32", "f", c_float),
        ("float64", "d", c_double),
    ):
        function = getattr(library, f"{name}_{suffix}")
        function.argtypes = argument_types(scalar)
        function.restype = c_int
        functions[dtype] = function
    return functions


# Length queries and cache management use fixed, explicit signatures.
_lasso_args = lambda scalar: (POINTER(scalar), POINTER(scalar), POINTER(c_double), POINTER(c_double), POINTER(c_uint64), POINTER(c_uint64), c_uint64, POINTER(LassoOptions))
_lasso_result_args = lambda scalar: (c_uint64, POINTER(c_uint64), POINTER(c_uint64), POINTER(scalar), POINTER(scalar), POINTER(scalar), POINTER(scalar), POINTER(c_uint64), POINTER(c_uint64))


def _lasso_abi_matches(library, suffix):
    if library is None or not hasattr(library, 'sig_lasso_abi_version' + suffix):
        return False
    query = getattr(library, 'sig_lasso_abi_version' + suffix)
    query.argtypes = ()
    query.restype = c_int
    required = ['sig_lasso_v2' + suffix + kind for kind in ('_f', '_d')]
    required += ['sig_lasso_result_v2' + suffix + kind for kind in ('_f', '_d')]
    required += ['sig_lasso_shape_v2' + suffix, 'sig_lasso_release_v2' + suffix]
    return query() == 2 and all(hasattr(library, name) for name in required)


CPSIG_SIG_LASSO = _bind_float_pair(CPSIG, 'sig_lasso_v2', lambda scalar: (*_lasso_args(scalar), POINTER(c_uint64))) if _lasso_abi_matches(CPSIG, '') else None
CPSIG_SIG_LASSO_RESULT = _bind_float_pair(CPSIG, 'sig_lasso_result_v2', _lasso_result_args) if CPSIG_SIG_LASSO else None
CUSIG_SIG_LASSO = _bind_float_pair(CUSIG, 'sig_lasso_v2_cuda', lambda scalar: (*_lasso_args(scalar), c_void_p, c_int, POINTER(c_uint64))) if _lasso_abi_matches(CUSIG, '_cuda') else None
CUSIG_SIG_LASSO_RESULT = _bind_float_pair(CUSIG, 'sig_lasso_result_v2_cuda', _lasso_result_args) if CUSIG_SIG_LASSO else None
for _lasso_library, _lasso_suffix, _lasso_available in ((CPSIG, '', CPSIG_SIG_LASSO), (CUSIG, '_cuda', CUSIG_SIG_LASSO)):
    if _lasso_available:
        _lasso_shape = getattr(_lasso_library, 'sig_lasso_shape_v2' + _lasso_suffix)
        _lasso_shape.argtypes = (c_uint64, POINTER(LassoShape))
        _lasso_shape.restype = c_int
        _lasso_release = getattr(_lasso_library, 'sig_lasso_release_v2' + _lasso_suffix)
        _lasso_release.argtypes = (c_uint64,)
        _lasso_release.restype = c_int

CPSIG.sig_length.argtypes = (c_uint64, c_uint64)
CPSIG.sig_length.restype = c_uint64

CPSIG.log_sig_length.argtypes = (c_uint64, c_uint64)
CPSIG.log_sig_length.restype = c_uint64

CPSIG.set_cache_dir.argtypes = (c_char_p,)
CPSIG.set_cache_dir.restype = c_int

CPSIG.prepare_log_sig.argtypes = (c_uint64, c_uint64, c_int, c_bool)
CPSIG.prepare_log_sig.restype = c_int

CPSIG.clear_cache.argtypes = (c_bool,)
CPSIG.clear_cache.restype = c_int

CPSIG.prepare_branched_sig.argtypes = (c_uint64, c_uint64, c_bool, c_bool)
CPSIG.prepare_branched_sig.restype = c_int

CPSIG.prepare_branched_log_sig.argtypes = (c_uint64, c_uint64, c_int, c_bool, c_bool)
CPSIG.prepare_branched_log_sig.restype = c_int

CPSIG.branched_sig_length.argtypes = (c_uint64, c_uint64, c_bool)
CPSIG.branched_sig_length.restype = c_uint64

CPSIG.branched_log_sig_length.argtypes = (c_uint64, c_uint64, c_bool)
CPSIG.branched_log_sig_length.restype = c_uint64

CPSIG.prepare_branched_sig_coef.argtypes = (POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool)
CPSIG.prepare_branched_sig_coef.restype = c_int


if BUILT_WITH_CUDA:
    CUSIG.prepare_log_sig_cuda.argtypes = (c_uint64, c_uint64, c_int, c_bool)
    CUSIG.prepare_log_sig_cuda.restype = c_int

    CUSIG.clear_cache_cuda.argtypes = (c_bool,)
    CUSIG.clear_cache_cuda.restype = c_int

    CUSIG.set_cache_dir_cuda.argtypes = (c_char_p,)
    CUSIG.set_cache_dir_cuda.restype = c_int

    CUSIG.prepare_branched_sig_cuda.argtypes = (c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.prepare_branched_sig_cuda.restype = c_int

    CUSIG.prepare_branched_log_sig_cuda.argtypes = (c_uint64, c_uint64, c_int, c_bool, c_bool)
    CUSIG.prepare_branched_log_sig_cuda.restype = c_int

    CUSIG.prepare_branched_sig_coef_cuda.argtypes = (POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool)
    CUSIG.prepare_branched_sig_coef_cuda.restype = c_int


# CPU floating-point operations.
CPSIG_TRANSFORM_PATH = _bind_float_pair(
    CPSIG, "transform_path",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, T, c_int),
)

CPSIG_TRANSFORM_PATH_BACKPROP = _bind_float_pair(
    CPSIG, "transform_path_backprop",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, T, c_int),
)

CPSIG_LOG_SIG_COMBINE = _bind_float_pair(
    CPSIG, "log_sig_combine",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int),
)

CPSIG_LOG_SIG_FROM_PATH = _bind_float_pair(
    CPSIG, "log_sig_from_path",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_int),
)

CPSIG_LOG_SIG_FROM_PATH_BACKPROP = _bind_float_pair(
    CPSIG, "log_sig_from_path_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_int),
)

CPSIG_LOG_SIG_COMBINE_BACKPROP = _bind_float_pair(
    CPSIG, "log_sig_combine_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_int),
)

CPSIG_SIG_COMBINE = _bind_float_pair(
    CPSIG, "sig_combine",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_int),
)

CPSIG_SIG_JOIN = _bind_float_pair(
    CPSIG, "sig_join",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int),
)

CPSIG_SIG_JOIN_BACKPROP = _bind_float_pair(
    CPSIG, "sig_join_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_bool, c_bool, c_int),
)

CPSIG_SIG_COMBINE_BACKPROP = _bind_float_pair(
    CPSIG, "sig_combine_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_bool, c_int),
)

CPSIG_SIG_COEF = _bind_float_pair(
    CPSIG, "sig_coef",
    lambda T: (POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64,
               c_uint64, c_bool, c_bool, T, c_bool, c_int),
)

CPSIG_SIG_COEF_BACKPROP = _bind_float_pair(
    CPSIG, "sig_coef_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64,
               POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_bool, c_bool, T, c_int),
)

CPSIG_SIGNATURE = _bind_float_pair(
    CPSIG, "signature",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, T, c_bool,
               c_bool, c_int, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CPSIG_SIG_BACKPROP = _bind_float_pair(
    CPSIG, "sig_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_bool, c_bool, T, c_bool, c_int, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CPSIG_SIG_TO_LOG_SIG = _bind_float_pair(
    CPSIG, "sig_to_log_sig",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int),
)

CPSIG_SIG_TO_LOG_SIG_BACKPROP = _bind_float_pair(
    CPSIG, "sig_to_log_sig_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int,
               c_bool, c_int),
)

CPSIG_SIG_KERNEL = _bind_float_pair(
    CPSIG, "sig_kernel",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool,
               c_int),
)

CPSIG_SIG_KERNEL_POLY = _bind_float_pair(
    CPSIG, "sig_kernel_poly",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool,
               c_int),
)

CPSIG_SIG_KERNEL_POLY_BACKPROP = _bind_float_pair(
    CPSIG, "sig_kernel_poly_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_bool, c_int),
)

CPSIG_BRANCHED_SIG_KERNEL = _bind_float_pair(
    CPSIG, "branched_sig_kernel",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,
               c_bool, c_int),
)

CPSIG_SIG_KERNEL_BACKPROP = _bind_float_pair(
    CPSIG, "sig_kernel_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_uint64, c_bool, c_int),
)

CPSIG_BRANCHED_SIG_KERNEL_BACKPROP = _bind_float_pair(
    CPSIG, "branched_sig_kernel_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_uint64, c_uint64, c_bool, c_int),
)

CPSIG_BRANCHED_SIG_KERNEL_POLY = _bind_float_pair(
    CPSIG, "branched_sig_kernel_poly",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_bool, c_int),
)

CPSIG_BRANCHED_SIG_KERNEL_POLY_BACKPROP = _bind_float_pair(
    CPSIG, "branched_sig_kernel_poly_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_uint64, c_bool, c_int),
)

CPSIG_BRANCHED_SIG = _bind_float_pair(
    CPSIG, "branched_sig",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool, T,
               c_bool, c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CPSIG_BRANCHED_SIG_COEF = _bind_float_pair(
    CPSIG, "branched_sig_coef",
    lambda T: (POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,
               c_int, c_bool, c_bool, T, c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CPSIG_BRANCHED_SIG_COEF_BACKPROP = _bind_float_pair(
    CPSIG, "branched_sig_coef_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64, c_uint64,
               c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool, T, c_bool, POINTER(T), c_uint64, c_uint64,
               c_uint64),
)

CPSIG_BRANCHED_SIG_COMBINE = _bind_float_pair(
    CPSIG, "branched_sig_combine",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool),
)

CPSIG_BRANCHED_SIG_TO_LOG_SIG = _bind_float_pair(
    CPSIG, "branched_sig_to_log_sig",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_int, c_bool, c_bool),
)

CPSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP = _bind_float_pair(
    CPSIG, "branched_sig_to_log_sig_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_int, c_bool,
               c_bool),
)

CPSIG_BRANCHED_LOG_SIG_JOIN = _bind_float_pair(
    CPSIG, "branched_log_sig_join",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_int),
)

CPSIG_BRANCHED_LOG_SIG_JOIN_BACKPROP = _bind_float_pair(
    CPSIG, "branched_log_sig_join_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_bool, c_int),
)

CPSIG_BRANCHED_LOG_SIG_FROM_PATH = _bind_float_pair(
    CPSIG, "branched_log_sig_from_path",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_int),
)

CPSIG_BRANCHED_LOG_SIG_FROM_PATH_BACKPROP = _bind_float_pair(
    CPSIG, "branched_log_sig_from_path_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_int),
)

CPSIG_BRANCHED_SIG_COMBINE_BACKPROP = _bind_float_pair(
    CPSIG, "branched_sig_combine_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_int, c_bool, c_bool),
)

CPSIG_BRANCHED_SIG_BACKPROP = _bind_float_pair(
    CPSIG, "branched_sig_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_int,
               c_bool, c_bool, T, c_bool, c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CPSIG_LINEAR_SIG = _bind_float_pair(
    CPSIG, "linear_sig",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_int),
)

CPSIG_LOG_SIG_JOIN = _bind_float_pair(
    CPSIG, "log_sig_join",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int),
)

CPSIG_LOG_SIG_JOIN_BACKPROP = _bind_float_pair(
    CPSIG, "log_sig_join_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_int),
)

CPSIG_LOGSIG_TO_SIG = _bind_float_pair(
    CPSIG, "logsig_to_sig",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int, c_bool, c_int),
)

CPSIG_LOGSIG_TO_SIG_BACKPROP = _bind_float_pair(
    CPSIG, "logsig_to_sig_backprop",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_int,
               c_bool, c_int),
)


# CUDA floating-point operations.
# Absent CUDA libraries yield None, preserving the optional-backend API.
CUSIG_TRANSFORM_PATH = _bind_float_pair(
    CUSIG, "transform_path_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, T),
)

CUSIG_TRANSFORM_PATH_BACKPROP = _bind_float_pair(
    CUSIG, "transform_path_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool, T),
)

CUSIG_LOG_SIG_COMBINE = _bind_float_pair(
    CUSIG, "log_sig_combine_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_LOG_SIG_FROM_PATH = _bind_float_pair(
    CUSIG, "log_sig_from_path_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64),
)

CUSIG_LOG_SIG_FROM_PATH_BACKPROP = _bind_float_pair(
    CUSIG, "log_sig_from_path_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64),
)

CUSIG_LOG_SIG_COMBINE_BACKPROP = _bind_float_pair(
    CUSIG, "log_sig_combine_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_SIG_COMBINE = _bind_float_pair(
    CUSIG, "sig_combine_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool),
)

CUSIG_SIG_JOIN = _bind_float_pair(
    CUSIG, "sig_join_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool),
)

CUSIG_SIG_JOIN_BACKPROP = _bind_float_pair(
    CUSIG, "sig_join_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_bool, c_bool),
)

CUSIG_SIG_COMBINE_BACKPROP = _bind_float_pair(
    CUSIG, "sig_combine_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_bool),
)

CUSIG_SIG_COEF = _bind_float_pair(
    CUSIG, "sig_coef_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64, POINTER(c_uint64), c_uint64, c_uint64,
               c_uint64, c_bool),
)

CUSIG_SIG_COEF_BACKPROP = _bind_float_pair(
    CUSIG, "sig_coef_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64,
               POINTER(c_uint64), c_uint64, c_uint64, c_uint64),
)

CUSIG_SIGNATURE = _bind_float_pair(
    CUSIG, "signature_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, T, c_bool,
               c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_SIG_BACKPROP = _bind_float_pair(
    CUSIG, "sig_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_bool, c_bool, T, c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_SIG_TO_LOG_SIG = _bind_float_pair(
    CUSIG, "sig_to_log_sig_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_bool),
)

CUSIG_SIG_TO_LOG_SIG_BACKPROP = _bind_float_pair(
    CUSIG, "sig_to_log_sig_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_bool),
)

CUSIG_SIG_KERNEL = _bind_float_pair(
    CUSIG, "sig_kernel_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool),
)

CUSIG_SIG_KERNEL_POLY = _bind_float_pair(
    CUSIG, "sig_kernel_poly_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_bool),
)

CUSIG_SIG_KERNEL_POLY_BACKPROP = _bind_float_pair(
    CUSIG, "sig_kernel_poly_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_bool),
)

CUSIG_BRANCHED_SIG_KERNEL = _bind_float_pair(
    CUSIG, "branched_sig_kernel_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,
               c_bool),
)

CUSIG_SIG_KERNEL_BACKPROP = _bind_float_pair(
    CUSIG, "sig_kernel_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_uint64, c_bool),
)

CUSIG_BRANCHED_SIG_KERNEL_BACKPROP = _bind_float_pair(
    CUSIG, "branched_sig_kernel_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_uint64, c_uint64, c_bool),
)

CUSIG_BRANCHED_SIG_KERNEL_POLY = _bind_float_pair(
    CUSIG, "branched_sig_kernel_poly_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_bool),
)

CUSIG_BRANCHED_SIG_KERNEL_POLY_BACKPROP = _bind_float_pair(
    CUSIG, "branched_sig_kernel_poly_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_uint64, c_uint64, c_bool),
)

CUSIG_BRANCHED_SIG = _bind_float_pair(
    CUSIG, "branched_sig_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, T, c_bool,
               c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_BRANCHED_SIG_COEF = _bind_float_pair(
    CUSIG, "branched_sig_coef_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,
               c_bool, c_bool, T, c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_BRANCHED_SIG_COEF_BACKPROP = _bind_float_pair(
    CUSIG, "branched_sig_coef_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(c_uint64), c_uint64, c_uint64,
               c_uint64, c_uint64, c_uint64, c_bool, c_bool, T, c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_BRANCHED_SIG_COMBINE = _bind_float_pair(
    CUSIG, "branched_sig_combine_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool, c_bool),
)

CUSIG_BRANCHED_SIG_TO_LOG_SIG = _bind_float_pair(
    CUSIG, "branched_sig_to_log_sig_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool),
)

CUSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP = _bind_float_pair(
    CUSIG, "branched_sig_to_log_sig_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_bool, c_bool),
)

CUSIG_BRANCHED_LOG_SIG_JOIN = _bind_float_pair(
    CUSIG, "branched_log_sig_join_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool),
)

CUSIG_BRANCHED_LOG_SIG_JOIN_BACKPROP = _bind_float_pair(
    CUSIG, "branched_log_sig_join_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_bool),
)

CUSIG_BRANCHED_LOG_SIG_FROM_PATH = _bind_float_pair(
    CUSIG, "branched_log_sig_from_path_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64),
)

CUSIG_BRANCHED_LOG_SIG_FROM_PATH_BACKPROP = _bind_float_pair(
    CUSIG, "branched_log_sig_from_path_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64),
)

CUSIG_BRANCHED_SIG_COMBINE_BACKPROP = _bind_float_pair(
    CUSIG, "branched_sig_combine_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64,
               c_bool, c_bool),
)

CUSIG_BRANCHED_SIG_BACKPROP = _bind_float_pair(
    CUSIG, "branched_sig_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_uint64,
               c_bool, c_bool, T, c_bool, c_bool, POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_LINEAR_SIG = _bind_float_pair(
    CUSIG, "linear_sig_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_bool),
)

CUSIG_LOG_SIG_JOIN = _bind_float_pair(
    CUSIG, "log_sig_join_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_LOG_SIG_JOIN_BACKPROP = _bind_float_pair(
    CUSIG, "log_sig_join_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64),
)

CUSIG_LOGSIG_TO_SIG = _bind_float_pair(
    CUSIG, "logsig_to_sig_cuda",
    lambda T: (POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_bool),
)

CUSIG_LOGSIG_TO_SIG_BACKPROP = _bind_float_pair(
    CUSIG, "logsig_to_sig_backprop_cuda",
    lambda T: (POINTER(T), POINTER(T), POINTER(T), c_uint64, c_uint64, c_uint64, c_int, c_bool),
)

# Library shutdown.
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
