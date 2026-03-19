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
from .dtypes import CPSIG_BATCH_LOG_SIG_COMBINE, CUSIG_BATCH_LOG_SIG_COMBINE_CUDA
from .sig_length import log_sig_length
from .data_handlers import MultipleSigInputHandler, SigOutputHandler


def log_sig_combine_(data, result, aug_dimension, degree, n_jobs):
    err_code = CPSIG_BATCH_LOG_SIG_COMBINE[data.dtype](
        data.sig_ptr[0],
        data.sig_ptr[1],
        result.data_ptr,
        data.batch_size,
        aug_dimension,
        degree,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.log_sig_combine: " + err_msg(err_code))
    return result.data


def log_sig_combine_cuda_(data, result, aug_dimension, degree):
    err_code = CUSIG_BATCH_LOG_SIG_COMBINE_CUDA[data.dtype](
        data.sig_ptr[0],
        data.sig_ptr[1],
        result.data_ptr,
        data.batch_size,
        aug_dimension,
        degree
    )

    if err_code:
        raise Exception("Error in pysiglib.log_sig_combine (CUDA): " + err_msg(err_code))
    return result.data


def log_sig_combine(
        log_sig1: Union[np.ndarray, torch.tensor],
        log_sig2: Union[np.ndarray, torch.tensor],
        dimension: int,
        degree: int,
        time_aug: bool = False,
        lead_lag: bool = False,
        n_jobs: int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Combines two truncated log-signatures (Lyndon basis) of the same degree and dimension
    using the Baker-Campbell-Hausdorff (BCH) formula. In particular, let :math:`x_1, x_2`
    be two paths such that the first point of :math:`x_2` is the last point of :math:`x_1`.
    Let :math:`L(x_1), L(x_2)` be the truncated log-signatures of :math:`x_1, x_2`
    respectively (computed with ``method=2``). Then calling this function on
    :math:`L(x_1), L(x_2)` returns the truncated log-signature of the concatenated path,

    .. math::

        L(x_1 * x_2) = \\text{BCH}(L(x_1), L(x_2)).

    :param log_sig1: The first truncated log-signature (Lyndon basis, method=2)
    :type log_sig1: numpy.ndarray | torch.tensor
    :param log_sig2: The second truncated log-signature (Lyndon basis, method=2).
        Must have the same degree and dimension as the first.
    :type log_sig2: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the log-signatures, :math:`N`
    :type degree: int
    :param time_aug: Whether time augmentation was applied before computing
        the log-signature.
    :type time_aug: bool
    :param lead_lag: Whether the lead lag transformation was applied before computing
        the log-signature.
    :type lead_lag: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation
        is run serially. If set to -1, all available threads are used.
    :type n_jobs: int
    :return: Combined log-signature (Lyndon basis)
    :rtype: numpy.ndarray | torch.tensor
    """

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    ls_len = log_sig_length(aug_dimension, degree)
    data = MultipleSigInputHandler([log_sig1, log_sig2], ls_len, ["log_sig1", "log_sig2"])
    result = SigOutputHandler(data, ls_len)

    if data.device != "cpu":
        if CUSIG_BATCH_LOG_SIG_COMBINE_CUDA is None:
            raise RuntimeError("CUDA log_sig_combine requires cusig (built with CUDA support)")
        return log_sig_combine_cuda_(data, result, aug_dimension, degree)

    return log_sig_combine_(data, result, aug_dimension, degree, n_jobs)
