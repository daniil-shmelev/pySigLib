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

from .param_checks import check_type, check_non_neg, check_log_sig_method, check_n_jobs
from .error_codes import err_msg
from .dtypes import (CPSIG_SIG_TO_LOG_SIG,
                     CUSIG_SIG_TO_LOG_SIG,
                     CPSIG_LOG_SIG_FROM_PATH, CUSIG_LOG_SIG_FROM_PATH)
from .sig_length import sig_length, log_sig_length, aug_dim, _infer_scalar_term
from .sig import sig
from .data_handlers import SigOutputHandler, SigInputHandler, PathInputHandler, CorrectionInputHandler
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
        raise Exception(
            "Error in pysiglib.set_cache_dir: " + err_msg(err_code, "cpu"))

    if BUILT_WITH_CUDA:
        err_code = CUSIG.set_cache_dir_cuda(dir.encode("utf-8"))
        if err_code:
            raise Exception(
                "Error in pysiglib.set_cache_dir (CUDA): "
                + err_msg(err_code, "cuda"))


def prepare_log_sig(
        dimension : int,
        degree : int,
        method : int,
        *,
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
    ``device`` parameter.

    When ``use_disk=True``, both the CPU and GPU libraries read from and write to a
    shared disk cache in the same binary format. If the disk cache already exists
    (e.g. from a previous run), the data is loaded from disk instead of being recomputed.
    For ``method=0``, no preparation is needed and this function returns immediately.

    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the log signature.
    :type degree: int
    :param method: Method for the log signature computation. Must be one of `0`, `1`, `2` or `3`.
        Methods `1`, `2`, and `3` require preparation; method `0` does not.
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

    if method == 0:
        return

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)

    if device in ("cpu", "both"):
        err_code = CPSIG.prepare_log_sig(
            aug_dimension,
            degree,
            method,
            use_disk
        )

        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_log_sig: "
                + err_msg(err_code, "cpu"))

    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.prepare_log_sig_cuda(aug_dimension, degree, method, use_disk)
        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_log_sig (CUDA): "
                + err_msg(err_code, "cuda"))

def clear_cache(
        *,
        use_disk : bool = False,
        device : str = "both"
):
    """
    Clears the in-memory caches generated by pySigLib preparation functions.

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
            raise Exception(
                "Error in pysiglib.clear_cache: "
                + err_msg(err_code, "cpu"))

    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.clear_cache_cuda(use_disk)
        if err_code:
            raise Exception(
                "Error in pysiglib.clear_cache (CUDA): "
                + err_msg(err_code, "cuda"))

def sig_to_log_sig(
        sig : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the log signature from the signature, using the specified method. For details,
    see the page :doc:`Computing Log Signatures </pages/log_signatures/log_sig_methods>`.

    :param sig: The signature or batch of signatures, given as a `numpy.ndarray` or `torch.Tensor`.
        For a single signature, this must be of shape ``sig_length``. For a batch of paths, this must
        be of shape ``(batch_size, sig_length)``.
    :type sig: numpy.ndarray | torch.Tensor
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
    :return: Log signature or a batch of log signatures. For method ``0`` (expanded), the output
        matches the scalar-term format of the input ``sig``. Methods ``1`` and ``2`` produce
        log-sig-shaped output (no scalar term).
    :rtype: numpy.ndarray | torch.Tensor

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

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    scalar_term = _infer_scalar_term(sig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag)

    sig_len = sig_length(aug_dimension, degree, scalar_term=scalar_term)
    data = SigInputHandler(sig, sig_len, "sig")
    out_len = log_sig_length(aug_dimension, degree) if method else sig_length(aug_dimension, degree, scalar_term=scalar_term)
    result = SigOutputHandler(data, out_len)

    if data.batch_size == 0:
        return result.data

    check_n_jobs(n_jobs)
    if data.device == "cpu":
        err_code = CPSIG_SIG_TO_LOG_SIG[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs)
    else:
        err_code = CUSIG_SIG_TO_LOG_SIG[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            aug_dimension, degree, method, scalar_term)
    if err_code:
        raise Exception(
            "Error in pysiglib.sig_to_log_sig: "
            + err_msg(err_code, result.device))
    return result.data

def log_sig(
        path : Union[np.ndarray, torch.Tensor],
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        method : int = 1,
        scalar_term : bool = False,
        correction = None,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the log signature using the specified method. For details,
    see the page :doc:`Computing Log Signatures </pages/log_signatures/log_sig_methods>`.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.Tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.Tensor
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
    :param scalar_term: If True, the output includes the leading constant 1 at index 0
        (the empty-word term). If False (default), this leading element is stripped from the output.
        Only affects method ``0`` (expanded) output; methods ``1`` and ``2`` produce
        log-sig-shaped output with no scalar term.
    :type scalar_term: bool
    :param correction: Optional per-segment correction of level
        :math:`\\geq 2` added locally before exponentiating each path segment.
        See :func:`sig` for the accepted constant, shared per-segment, and
        batch-specific layouts. For non-Lie correction such as
        Ito level-2 diagonal terms, use ``method=0`` to retain the full tensor
        logarithm. Cannot be combined with ``lead_lag=True``.
    :type correction: numpy.ndarray | torch.Tensor | None
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Log signature or a batch of log signatures.
    :rtype: numpy.ndarray | torch.Tensor

    Example usage:
    ----------------

    .. code-block:: python

        import torch
        import pysiglib

        pysiglib.prepare_log_sig(5, 3, lead_lag=True, method=2)

        X = torch.rand((32,100,5))
        X_log_sig = pysiglib.log_sig(X, 3, lead_lag=True, method=2)

    Ito-lifted log signature of a sampled Brownian path. For Brownian
    motion with instantaneous covariance :math:`\\Sigma`, setting the
    level-2 correction to :math:`c^{(2)}_{ij} = -\\frac{1}{2}\\Sigma_{ij}\\,\\Delta t`
    per segment gives the Ito correction. The Ito level-2 term is not
    Lie-valued (its symmetric part is not in the free Lie algebra), so
    ``method=0`` is used to retain the full tensor logarithm.

    .. code-block:: python

        import numpy as np
        import pysiglib

        d, N, T = 2, 3, 1.0
        n_steps = 100
        dt = T / n_steps
        rng = np.random.default_rng(42)

        # 2D standard Brownian motion sample (Sigma = I)
        path = np.zeros((n_steps + 1, d))
        path[1:] = np.cumsum(rng.normal(0, np.sqrt(dt), (n_steps, d)), axis=0)

        # Ito level-2 correction: -0.5 * dt * Sigma per path segment.
        correction = np.broadcast_to((-0.5 * np.eye(d) * dt).reshape(1, -1), (n_steps, d * d))

        ito_log_sig = pysiglib.log_sig(
            path, N, correction=correction, end_time=T, method=0)
        print(ito_log_sig)
    """
    if method == 3:
        if correction is not None:
            correction_data = CorrectionInputHandler(
                correction, PathInputHandler(path, time_aug, lead_lag, end_time, "path"), degree)
            if correction_data.length != 0:
                raise ValueError("correction is not supported with log_sig method=3")
        if time_aug or lead_lag:
            path = transform_path(path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)
        aug_dim = path.shape[-1]
        ls_len = log_sig_length(aug_dim, degree)
        data = PathInputHandler(path, False, False, 1.0, "path")
        result = SigOutputHandler(data, ls_len)
        if data.batch_size == 0:
            return result.data
        if data.device == "cpu":
            err_code = CPSIG_LOG_SIG_FROM_PATH[data.dtype](
                data.data_ptr, result.data_ptr, data.batch_size,
                data.data_length, aug_dim, degree, n_jobs)
        else:
            err_code = CUSIG_LOG_SIG_FROM_PATH[data.dtype](
                data.data_ptr, result.data_ptr, data.batch_size,
                data.data_length, aug_dim, degree)
        if err_code:
            raise Exception(
                "Error in pysiglib.log_sig (method=3): "
                + err_msg(err_code, result.device))
        return result.data

    # Methods 0-2: compute sig then project to log sig.
    sig_ = sig(path, degree, scalar_term=scalar_term, time_aug=time_aug, lead_lag=lead_lag,
               end_time=end_time, horner=True, correction=correction, n_jobs=n_jobs)
    dimension = path.shape[-1]
    return sig_to_log_sig(sig_, dimension, degree, time_aug=time_aug, lead_lag=lead_lag, method=method, n_jobs=n_jobs)
