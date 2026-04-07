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
from .dtypes import CPSIG_LOGSIG_TO_SIG, CPSIG_BATCH_LOGSIG_TO_SIG
from .sig_length import sig_length
from .data_handlers import SigOutputHandler, SigInputHandler


######################################################
# Python wrappers
######################################################

def logsig_to_sig_(data, result, dimension, degree, time_aug, lead_lag, method):
    err_code = CPSIG_LOGSIG_TO_SIG[data.dtype](
        data.data_ptr,
        result.data_ptr,
        dimension,
        degree,
        time_aug,
        lead_lag,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.logsig_to_sig: " + err_msg(err_code))
    return result.data

def batch_logsig_to_sig_(data, result, dimension, degree, time_aug, lead_lag, method, n_jobs=1):
    err_code = CPSIG_BATCH_LOGSIG_TO_SIG[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        dimension,
        degree,
        time_aug,
        lead_lag,
        method,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.logsig_to_sig: " + err_msg(err_code))
    return result.data


######################################################
# Public API
######################################################

def logsig_to_sig(
        log_sig : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : int = 0,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the signature from the log-signature via the truncated tensor exponential.
    This is the inverse of :func:`pysiglib.sig_to_log_sig`.

    Currently only ``method=0`` (expanded tensor form) is supported.

    :param log_sig: The log-signature or batch of log-signatures, given as a `numpy.ndarray` or `torch.tensor`.
        For a single log-signature, this must be of shape ``(sig_length,)``. For a batch, this must be
        of shape ``(batch_size, sig_length)``.
    :type log_sig: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the (log) signature(s).
    :type degree: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param method: Must be ``0`` (expanded tensor form). Methods ``1`` and ``2`` are not yet supported.
    :type method: int
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Signature or a batch of signatures.
    :rtype: numpy.ndarray | torch.tensor

    Example usage::

        import pysiglib
        import numpy as np

        path = np.random.uniform(size=(32, 100, 5))
        degree = 4

        sig = pysiglib.sig(path, degree)
        log_sig = pysiglib.sig_to_log_sig(sig, 5, degree, method=0)
        sig_recovered = pysiglib.logsig_to_sig(log_sig, 5, degree, method=0)
        # sig_recovered ≈ sig
    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(method, "method", int)
    if method != 0:
        raise ValueError("logsig_to_sig currently only supports method=0 (expanded tensor form)")

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    sig_len = sig_length(aug_dimension, degree)
    data = SigInputHandler(log_sig, sig_len, "log_sig")
    result = SigOutputHandler(data, sig_len)

    if data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        return batch_logsig_to_sig_(data, result, dimension, degree, time_aug, lead_lag, method, n_jobs)
    return logsig_to_sig_(data, result, dimension, degree, time_aug, lead_lag, method)
