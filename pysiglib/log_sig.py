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

from typing import Union
from pathlib import Path

import numpy as np
import torch

from .param_checks import check_type, check_non_neg, check_log_sig_method
from .error_codes import err_msg
from .dtypes import (CPSIG_SIG_TO_LOG_SIG, CPSIG_BATCH_SIG_TO_LOG_SIG,
                     CUSIG_SIG_TO_LOG_SIG_CUDA, CUSIG_BATCH_SIG_TO_LOG_SIG_CUDA,
                     CPSIG_BATCH_LOG_SIG_FROM_PATH, CUSIG_BATCH_LOG_SIG_FROM_PATH_CUDA)
from .sig_length import sig_length, log_sig_length
from .sig import sig
from .data_handlers import SigOutputHandler, SigInputHandler, PathInputHandler
from .load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA
from .transform_path import transform_path


######################################################
# Python wrappers
######################################################

def set_cache_dir(
        dir : str
):
    """
    Sets the cache directory to use in ``pysiglib.prepare_log_sig``
    when ``use_disk=True``. If the cache directory is not explicitly
    set by a call to this function, a default directory will be used:

    - Windows: ``%LOCALAPPDATA%``
    - Linux: ``~/.cache``
    - Mac: ``~/Library/Caches``

    This function is not thread safe.

    :param dir: Path to cache directory
    :type dir: str

    Example usage:
    ----------------

    .. code-block::

        import pysiglib

        # Set cache dir to a folder "my_cache_dir" in the current working directory
        pysiglib.set_cache_dir("./my_cache_dir")

        pysiglib.prepare_log_sig(5, 3, lead_lag=True, method=2, use_disk=True)

        X = torch.rand((32,100,5))
        X_log_sig = pysiglib.log_sig(X, 3, lead_lag=True, method=2)

    """
    check_type(dir, "dir", str)
    p = Path(dir)
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {p}")

    err_code = CPSIG.set_cache_dir(dir.encode("utf-8"))
    if err_code:
        raise Exception("Error in pysiglib.set_cache_dir: " + err_msg(err_code))

    if BUILT_WITH_CUDA:
        err_code = CUSIG.set_cache_dir_cuda(dir.encode("utf-8"))
        if err_code:
            raise Exception("Error in pysiglib.set_cache_dir (CUDA): " + err_msg(err_code))


def prepare_log_sig(
        dimension : int,
        degree : int,
        method : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        use_disk : bool = False,
        device : str = "both"
):
    """
    Prepares for log signature computations. For details concerning the ``method`` parameter,
    see the page :doc:`Computing Log Signatures </pages/log_signatures/log_sig_methods>`.
    This function is not thread safe.

    This function populates in-memory caches for the CPU and/or GPU, controlled by the
    ``device`` parameter. The CPU cache stores Lyndon words and projection matrices in
    host memory, while the GPU cache (if CUDA is available) stores Lyndon indices, level
    indices, and sparse matrices in device memory.

    When ``use_disk=True``, both the CPU and GPU libraries read from and write to a
    shared disk cache in the same binary format. If the disk cache already exists
    (e.g. from a previous run), the data is loaded from disk instead of being recomputed.
    For ``method=0``, no preparation is needed and this function returns immediately.

    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the log signature.
    :type degree: int
    :param method: Method for the log signature computation. Must be one of `0`, `1`, `2` or `3`.
        Methods `0`-`2` require preparation; method `3` does not.
    :type method: int
    :param time_aug: Whether time augmentation will be used in the computation.
    :type time_aug: bool
    :param lead_lag: Whether the lead lag transform will be used in the computation.
    :type lead_lag: bool
    :param use_disk: If ``False``, will cache prepared objects in memory only.
        If ``True``, will also save these objects in a shared disk cache to be
        re-used for future runs. The CPU and GPU libraries share the same
        disk cache format and directory.
        See additionally the documentation for ``pysiglib.set_cache_dir``.
    :type use_disk: bool
    :param device: Which device caches to prepare. Must be one of ``"cpu"``, ``"cuda"``,
        or ``"both"`` (default). Use ``"cpu"`` to prepare only the CPU cache,
        ``"cuda"`` to prepare only the GPU cache, or ``"both"`` to prepare both.
    :type device: str

    Example usage:
    ----------------

    .. code-block::

        import pysiglib

        pysiglib.prepare_log_sig(5, 3, lead_lag=True, method=2, use_disk=True)

        X = torch.rand((32,100,5))
        X_log_sig = pysiglib.log_sig(X, 3, lead_lag=True, method=2)

    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(method, "method", int)
    check_log_sig_method(method)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(device, "device", str)

    if device not in ("cpu", "cuda", "both"):
        raise ValueError("device must be 'cpu', 'cuda', or 'both'")

    if method == 0 or method == 3:
        return

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    if device in ("cpu", "both"):
        err_code = CPSIG.prepare_log_sig(
            aug_dimension,
            degree,
            method,
            use_disk
        )

        if err_code:
            raise Exception("Error in pysiglib.prepare_log_sig: " + err_msg(err_code))

    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.prepare_log_sig_cuda(aug_dimension, degree, method, use_disk)
        if err_code:
            raise Exception("Error in pysiglib.prepare_log_sig (CUDA): " + err_msg(err_code))

def clear_cache(
        use_disk : bool = False,
        device : str = "both"
):
    """
    Clears the cache generated by ``pysiglib.prepare_log_sig``.

    :param use_disk: If ``False``, will clear the cache from memory only.
        If ``True``, will also clear the shared disk cache directory.
        See additionally the documentation for
        ``pysiglib.set_cache_dir``.
    :type use_disk: bool
    :param device: Which device caches to clear. Must be one of ``"cpu"``, ``"cuda"``,
        or ``"both"`` (default).
    :type device: str

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        pysiglib.prepare_log_sig(dimension=5, degree=4, method=2, use_disk=True)

        path = torch.rand((10, 100, 5))
        log_sig = pysiglib.log_sig(path, 4, n_jobs = -1)
        print(log_sig)

        pysiglib.clear_cache() # Clear cache from memory but keep on disk

    """
    if device not in ("cpu", "cuda", "both"):
        raise ValueError("device must be 'cpu', 'cuda', or 'both'")

    if device in ("cpu", "both"):
        err_code = CPSIG.clear_cache(use_disk)
        if err_code:
            raise Exception("Error in pysiglib.clear_cache: " + err_msg(err_code))

    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.clear_cache_cuda(use_disk)
        if err_code:
            raise Exception("Error in pysiglib.clear_cache (CUDA): " + err_msg(err_code))

def sig_to_log_sig_(data, result, data_dimension, degree, time_aug, lead_lag, method):
    err_code = CPSIG_SIG_TO_LOG_SIG[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data_dimension,
        degree,
        time_aug,
        lead_lag,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_to_log_sig: " + err_msg(err_code))
    return result.data

def batch_sig_to_log_sig_(data, result, data_dimension, degree, time_aug, lead_lag, method, n_jobs = 1):
    err_code = CPSIG_BATCH_SIG_TO_LOG_SIG[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        data_dimension,
        degree,
        time_aug,
        lead_lag,
        method,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_to_log_sig: " + err_msg(err_code))
    return result.data

def sig_to_log_sig_cuda_(data, result, aug_dimension, degree, method):
    err_code = CUSIG_SIG_TO_LOG_SIG_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        aug_dimension,
        degree,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_to_log_sig: " + err_msg(err_code))
    return result.data

def batch_sig_to_log_sig_cuda_(data, result, aug_dimension, degree, method):
    err_code = CUSIG_BATCH_SIG_TO_LOG_SIG_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        aug_dimension,
        degree,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_to_log_sig: " + err_msg(err_code))
    return result.data

def sig_to_log_sig(
        sig : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the log signature from the signature, using the specified method. For details,
    see the page :doc:`Computing Log Signatures </pages/log_signatures/log_sig_methods>`.

    :param sig: The signature or batch of signatures, given as a `numpy.ndarray` or `torch.tensor`.
        For a single signature, this must be of shape ``sig_length``. For a batch of paths, this must
        be of shape ``(batch_size, sig_length)``.
    :type sig: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the (log) signature(s).
    :type degree: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param method: Method to use for the log signature computation (`0`, `1` or `2`).
        Method `3` is not supported here; use ``pysiglib.log_sig`` with ``method=3`` instead.
    :type method: int
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Log signature or a batch of log signatures.
    :rtype: numpy.ndarray | torch.tensor

    Example usage:
    ----------------

    .. code-block:: python

        import torch
        import pysiglib

        pysiglib.prepare_log_sig(5, 3, lead_lag=True, method=2)

        X = torch.rand((32,100,5))
        X_sig = pysiglib.sig(X, 3, lead_lag=True)
        X_log_sig = pysiglib.sig_to_log_sig(X_sig, 5, 3, lead_lag=True, method=2)
    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(method, "method", int)
    check_log_sig_method(method)
    if method == 3:
        raise ValueError("method=3 is not supported in sig_to_log_sig. Use log_sig(path, degree, method=3) instead.")

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    sig_len = sig_length(aug_dimension, degree)
    data = SigInputHandler(sig, sig_len, "sig")
    out_len = log_sig_length(aug_dimension, degree) if method else sig_len
    result = SigOutputHandler(data, out_len)

    if data.device == "cpu":
        if data.is_batch:
            check_type(n_jobs, "n_jobs", int)
            if n_jobs == 0:
                raise ValueError("n_jobs cannot be 0")
            return batch_sig_to_log_sig_(data, result, dimension, degree, time_aug, lead_lag, method, n_jobs)
        return sig_to_log_sig_(data, result, dimension, degree, time_aug, lead_lag, method)
    else:
        if data.is_batch:
            return batch_sig_to_log_sig_cuda_(data, result, aug_dimension, degree, method)
        return sig_to_log_sig_cuda_(data, result, aug_dimension, degree, method)

def log_sig(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the log signature using the specified method. For details,
    see the page :doc:`Computing Log Signatures </pages/log_signatures/log_sig_methods>`.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param degree: Truncation degree of the (log) signature(s).
    :type degree: int
    :param time_aug: If set to True, will compute the log signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math:`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the log signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param method: Method to use for the log signature computation (`0`, `1`, `2` or `3`).
        Methods `0`-`2` first compute the full signature and then project to the log signature.
        Method `3` uses the Baker-Campbell-Hausdorff formula to compute the log signature
        directly from the path without ever computing the full signature. This uses less
        memory but is slower than methods `0`-`2` for typical dimensions and degrees.
    :type method: int
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Log signature or a batch of log signatures.
    :rtype: numpy.ndarray | torch.tensor

    Example usage:
    ----------------

    .. code-block:: python

        import torch
        import pysiglib

        pysiglib.prepare_log_sig(5, 3, lead_lag=True, method=2)

        X = torch.rand((32,100,5))
        X_log_sig = pysiglib.log_sig(X, 3, lead_lag=True, method=2)
    """
    if method == 3:
        return _log_sig_via_combine(path, degree, time_aug, lead_lag, end_time, n_jobs)

    sig_ = sig(path, degree, time_aug, lead_lag, end_time, True, n_jobs)
    dimension = path.shape[-1]
    log_sig_ = sig_to_log_sig(sig_, dimension, degree, time_aug, lead_lag, method, n_jobs)
    return log_sig_


def _log_sig_via_combine(path, degree, time_aug, lead_lag, end_time, n_jobs):
    """Compute log-signature via sequential BCH combination of per-segment log-signatures."""
    if time_aug or lead_lag:
        path = transform_path(path, time_aug, lead_lag, end_time, n_jobs)

    aug_dim = path.shape[-1]
    ls_len = log_sig_length(aug_dim, degree)

    data = PathInputHandler(path, False, False, 1.0, "path")
    result = SigOutputHandler(data, ls_len)

    if data.device == "cpu":
        err_code = CPSIG_BATCH_LOG_SIG_FROM_PATH[data.dtype](
            data.data_ptr,
            result.data_ptr,
            data.batch_size,
            data.data_length,
            aug_dim,
            degree,
            n_jobs
        )
        if err_code:
            raise Exception("Error in pysiglib.log_sig (method=3): " + err_msg(err_code))
    else:
        err_code = CUSIG_BATCH_LOG_SIG_FROM_PATH_CUDA[data.dtype](
            data.data_ptr,
            result.data_ptr,
            data.batch_size,
            data.data_length,
            aug_dim,
            degree
        )
        if err_code:
            raise Exception("Error in pysiglib.log_sig (method=3, CUDA): " + err_msg(err_code))

    return result.data
