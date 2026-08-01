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

from .param_checks import check_type, check_non_neg, check_n_jobs
from .error_codes import err_msg
from .dtypes import (
    CPSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP,
    CUSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP_CUDA,
)
from .data_handlers import MultipleSigInputHandler, SigOutputHandler
from .sig_length import aug_dim
from .branched_sig import (
    _infer_branched_scalar_term,
    branched_sig_length,
)


def branched_sig_to_log_sig_backprop(
        bsig: Union[np.ndarray, torch.Tensor],
        blogsig_derivs: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        planar: bool = False,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Backpropagates through the ``pysiglib.branched_sig_to_log_sig`` function.
    Given the derivatives of a scalar function :math:`F` with respect to the
    branched log signature, returns the derivatives of :math:`F` with respect
    to the branched signature.

    :param bsig: The branched signature or batch of branched signatures used in the forward pass,
        given as a `numpy.ndarray` or `torch.Tensor`. For a single branched signature, this must be
        of shape ``branched_sig_length``. For a batch of paths, this must be of shape
        ``(batch_size, branched_sig_length)``. The leading scalar term may be present or omitted.
    :type bsig: numpy.ndarray | torch.Tensor
    :param blogsig_derivs: Derivatives of the scalar function :math:`F` with respect to the
        branched log signature(s). This must be an array of the same shape as the branched log
        signature(s).
    :type blogsig_derivs: numpy.ndarray | torch.Tensor
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the branched signature(s).
    :type degree: int
    :param time_aug: Whether the branched signature(s) were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the branched signature(s) were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param planar: If True, use planar branched signatures.
    :type planar: bool
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives of the scalar function :math:`F` with respect to the branched
        signature(s). This is an array of the same shape as ``bsig``.
    :rtype: numpy.ndarray | torch.Tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        bsig = pysiglib.branched_sig(path, 3)
        blogsig = pysiglib.branched_sig_to_log_sig(bsig, 5, 3)
        blogsig_derivs = torch.ones_like(blogsig)
        bsig_derivs = pysiglib.branched_sig_to_log_sig_backprop(
            bsig, blogsig_derivs, 5, 3,
        )
        print(bsig_derivs)
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    scalar_term = _infer_branched_scalar_term(bsig, aug_dimension, degree, planar=planar)
    bsig_len = branched_sig_length(aug_dimension, degree, planar=planar, scalar_term=scalar_term)
    data = MultipleSigInputHandler([bsig, blogsig_derivs], bsig_len, ["bsig", "blogsig_derivs"])
    result = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, aug_dimension, degree, n_jobs, planar, scalar_term)
    else:
        err_code = CUSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP_CUDA[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, aug_dimension, degree, planar, scalar_term)
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_to_log_sig_backprop: " + err_msg(err_code))

    return result.data
