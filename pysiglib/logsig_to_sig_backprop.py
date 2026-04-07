# Copyright 2026 Daniil Shmelev
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

from .param_checks import check_type, check_non_neg
from .error_codes import err_msg
from .dtypes import (CPSIG_LOGSIG_TO_SIG_BACKPROP, CPSIG_BATCH_LOGSIG_TO_SIG_BACKPROP,
                     CUSIG_LOGSIG_TO_SIG_BACKPROP_CUDA, CUSIG_BATCH_LOGSIG_TO_SIG_BACKPROP_CUDA)
from .sig_length import sig_length, log_sig_length
from .data_handlers import SigOutputHandler, SigInputHandler


######################################################
# Python wrappers
######################################################

def logsig_to_sig_backprop_(data, derivs_data, result, dimension, degree, time_aug, lead_lag, method):
    err_code = CPSIG_LOGSIG_TO_SIG_BACKPROP[data.dtype](
        data.data_ptr,
        result.data_ptr,
        derivs_data.data_ptr,
        dimension,
        degree,
        time_aug,
        lead_lag,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.logsig_to_sig_backprop: " + err_msg(err_code))
    return result.data

def batch_logsig_to_sig_backprop_(data, derivs_data, result, dimension, degree, time_aug, lead_lag, method, n_jobs=1):
    err_code = CPSIG_BATCH_LOGSIG_TO_SIG_BACKPROP[data.dtype](
        data.data_ptr,
        result.data_ptr,
        derivs_data.data_ptr,
        data.batch_size,
        dimension,
        degree,
        time_aug,
        lead_lag,
        method,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.logsig_to_sig_backprop: " + err_msg(err_code))
    return result.data

def logsig_to_sig_backprop_cuda_(data, derivs_data, result, aug_dimension, degree, method):
    err_code = CUSIG_LOGSIG_TO_SIG_BACKPROP_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        derivs_data.data_ptr,
        aug_dimension,
        degree,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.logsig_to_sig_backprop: " + err_msg(err_code))
    return result.data

def batch_logsig_to_sig_backprop_cuda_(data, derivs_data, result, aug_dimension, degree, method):
    err_code = CUSIG_BATCH_LOGSIG_TO_SIG_BACKPROP_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        derivs_data.data_ptr,
        data.batch_size,
        aug_dimension,
        degree,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.logsig_to_sig_backprop: " + err_msg(err_code))
    return result.data


######################################################
# Public API
######################################################

def logsig_to_sig_backprop(
        log_sig : Union[np.ndarray, torch.tensor],
        sig_derivs : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 1,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Backpropagation through :func:`pysiglib.logsig_to_sig`.

    Given upstream derivatives ``sig_derivs`` (dL/d(sig)), computes the gradient
    dL/d(log_sig).

    Supports all methods (``0``, ``1``, ``2``).

    :param log_sig: The log-signature used in the forward pass.
    :type log_sig: numpy.ndarray | torch.tensor
    :param sig_derivs: Upstream derivatives dL/d(sig), same shape as the signature output.
    :type sig_derivs: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree.
    :type degree: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param method: Method to use (``0``, ``1``, or ``2``). Must match the method used in the forward pass.
    :type method: int
    :param n_jobs: Number of threads to run in parallel.
    :type n_jobs: int
    :return: Gradient dL/d(log_sig), same shape as ``log_sig``.
    :rtype: numpy.ndarray | torch.tensor
    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(method, "method", int)
    if method not in (0, 1, 2):
        raise ValueError("method must be 0, 1, or 2")

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    input_len = sig_length(aug_dimension, degree) if method == 0 else log_sig_length(aug_dimension, degree)
    sig_len = sig_length(aug_dimension, degree)
    data = SigInputHandler(log_sig, input_len, "log_sig")
    derivs_data = SigInputHandler(sig_derivs, sig_len, "sig_derivs")
    result = SigOutputHandler(data, input_len)

    if data.device == "cpu":
        if data.is_batch:
            check_type(n_jobs, "n_jobs", int)
            if n_jobs == 0:
                raise ValueError("n_jobs cannot be 0")
            return batch_logsig_to_sig_backprop_(data, derivs_data, result, dimension, degree, time_aug, lead_lag, method, n_jobs)
        return logsig_to_sig_backprop_(data, derivs_data, result, dimension, degree, time_aug, lead_lag, method)
    else:
        if data.is_batch:
            return batch_logsig_to_sig_backprop_cuda_(data, derivs_data, result, aug_dimension, degree, method)
        return logsig_to_sig_backprop_cuda_(data, derivs_data, result, aug_dimension, degree, method)
