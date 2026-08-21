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

import numpy as np
import torch

from .param_checks import check_type, check_non_neg, check_log_sig_method, check_n_jobs
from .error_codes import err_msg
from .dtypes import (CPSIG_SIG_TO_LOG_SIG_BACKPROP,
                     CUSIG_SIG_TO_LOG_SIG_BACKPROP,
                     CPSIG_LOG_SIG_FROM_PATH_BACKPROP, CUSIG_LOG_SIG_FROM_PATH_BACKPROP)
from .sig_length import sig_length, log_sig_length, aug_dim, _infer_scalar_term
from .data_handlers import SigOutputHandler, SigInputHandler, PathInputHandler, PathOutputHandler


def sig_to_log_sig_backprop(
        sig : Union[np.ndarray, torch.Tensor],
        log_sig_derivs : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Backpropagates through the ``pysiglib.sig_to_log_sig`` function.
    Given the derivatives of a scalar function :math:`F` with respect to the
    log signature, :math:`\\partial F / \\partial \\log(S(x))`, returns the
    derivatives of :math:`F` with respect to the signature,
    :math:`\\partial F / \\partial S(x)`.

    :param sig: The signature or batch of signatures, given as a `numpy.ndarray` or `torch.Tensor`.
        For a single signature, this must be of shape ``sig_length``. For a batch of paths, this must
        be of shape ``(batch_size, sig_length)``.
    :type sig: numpy.ndarray | torch.Tensor
    :param log_sig_derivs: Derivatives of the scalar function :math:`F` with respect to the log signature(s),
        :math:`\\partial F / \\partial S(x)`. This must be an array of the same shape as the
        log signature(s).
    :type log_sig_derivs: numpy.ndarray | torch.Tensor
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the (log) signature(s).
    :type degree: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param method: Method used for the log signature computation (`0`, `1` or `2`).
    :type method: int
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives of the scalar function :math:`F` with respect to the signature(s),
        :math:`\\partial F / \\partial S(x)`.
        This is an array of the same shape as the provided signature(s).
    :rtype: numpy.ndarray | torch.Tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        batch, length, dimension, degree = 10, 100, 5, 4
        pysiglib.prepare_log_sig(dimension, degree, time_aug=True, method=1)

        path = torch.rand((batch, length, dimension))
        sigs = pysiglib.sig(path, degree, time_aug=True)
        log_sigs = pysiglib.sig_to_log_sig(sigs, dimension, degree, time_aug=True, method=1)
        log_sig_derivs = torch.ones_like(log_sigs)
        sig_derivs = pysiglib.sig_to_log_sig_backprop(
            sigs, log_sig_derivs, dimension, degree,
            time_aug=True, method=1,
        )
        print(sig_derivs)
    """
    scalar_term = _infer_scalar_term(sig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag)
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(method, "method", int)
    check_log_sig_method(method)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)

    sig_len = sig_length(dimension, degree, time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term)
    log_sig_len = log_sig_length(dimension, degree, time_aug=time_aug, lead_lag=lead_lag) if method else sig_length(dimension, degree, time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term)
    data = SigInputHandler(sig, sig_len, "sig")
    derivs_data = SigInputHandler(log_sig_derivs, log_sig_len, "log_sig_derivs")

    if data.dtype != derivs_data.dtype:
        raise ValueError("sig and log_sig_derivs must have the same dtype")

    result = SigOutputHandler(data, sig_len)

    if data.batch_size == 0:
        return result.data

    check_n_jobs(n_jobs)
    if data.device == "cpu":
        err_code = CPSIG_SIG_TO_LOG_SIG_BACKPROP[data.dtype](
            data.data_ptr, result.data_ptr, derivs_data.data_ptr,
            data.batch_size, dimension, degree,
            time_aug, lead_lag, method, scalar_term, n_jobs)
    else:
        err_code = CUSIG_SIG_TO_LOG_SIG_BACKPROP[data.dtype](
            data.data_ptr, result.data_ptr, derivs_data.data_ptr,
            data.batch_size, aug_dimension, degree, method, scalar_term)
    if err_code:
        raise Exception("Error in pysiglib.sig_to_log_sig_backprop: " + err_msg(err_code))
    return result.data


def _log_sig_from_path_backprop(
        grad_output : Union[np.ndarray, torch.Tensor],
        path : Union[np.ndarray, torch.Tensor],
        degree : int,
        *,
        n_jobs : int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """Backpropagates through the method=3 log-signature-from-path computation."""
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    data = PathInputHandler(path, False, False, 1.0, "path")
    ls_len = log_sig_length(data.data_dimension, degree)
    derivs_data = SigInputHandler(grad_output, ls_len, "grad_output")

    if data.dtype != derivs_data.dtype:
        raise ValueError("grad_output and path must have the same dtype")

    result = PathOutputHandler(data.data_length, data.data_dimension, data)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_LOG_SIG_FROM_PATH_BACKPROP[data.dtype](
            derivs_data.data_ptr, result.data_ptr, data.data_ptr,
            data.batch_size, data.data_length, data.data_dimension, degree, n_jobs)
    else:
        err_code = CUSIG_LOG_SIG_FROM_PATH_BACKPROP[data.dtype](
            derivs_data.data_ptr, result.data_ptr, data.data_ptr,
            data.batch_size, data.data_length, data.data_dimension, degree)
    if err_code:
        raise Exception("Error in pysiglib.log_sig_from_path_backprop: " + err_msg(err_code))
    return result.data
