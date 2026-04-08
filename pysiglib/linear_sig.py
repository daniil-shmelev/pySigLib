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
from .dtypes import CPSIG_LINEAR_SIG, CPSIG_BATCH_LINEAR_SIG
from .load_siglib import BUILT_WITH_CUDA
from .sig_length import sig_length
from .data_handlers import SigInputHandler, SigOutputHandler

if BUILT_WITH_CUDA:
    from .dtypes import CUSIG_LINEAR_SIG_CUDA, CUSIG_BATCH_LINEAR_SIG_CUDA


def linear_sig_(data, result, dimension, degree):
    err_code = CPSIG_LINEAR_SIG[data.dtype](
        data.data_ptr,
        result.data_ptr,
        dimension,
        degree
    )

    if err_code:
        raise Exception("Error in pysiglib.linear_sig: " + err_msg(err_code))
    return result.data

def batch_linear_sig_(data, result, dimension, degree, n_jobs):
    err_code = CPSIG_BATCH_LINEAR_SIG[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        dimension,
        degree,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.linear_sig: " + err_msg(err_code))
    return result.data


def linear_sig_cuda_(data, result, dimension, degree):
    err_code = CUSIG_BATCH_LINEAR_SIG_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        dimension,
        degree
    )

    if err_code:
        raise Exception("Error in pysiglib.linear_sig (CUDA): " + err_msg(err_code))
    return result.data


def linear_sig(
        displacement : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the truncated signature of a single linear segment defined by a displacement vector.
    Given a displacement vector :math:`v \\in \\mathbb{R}^d`, this computes the signature of the
    linear path from :math:`0` to :math:`v`,

    .. math::

        S(v) = \\left(1, v, \\frac{v^{\\otimes 2}}{2!}, \\ldots, \\frac{v^{\\otimes N}}{N!}\\right).

    :param displacement: The displacement vector or batch of displacement vectors.
        For a single displacement, this must be of shape ``(dimension,)``.
        For a batch, this must be of shape ``(batch_size, dimension)``.
    :type displacement: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`.
    :type degree: int
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Truncated signature of the linear segment, or a batch of truncated signatures.
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import pysiglib
        import numpy as np

        dimension = 5
        degree = 3

        displacement = np.random.uniform(size=(dimension,))
        lsig = pysiglib.linear_sig(displacement, dimension, degree)

        # Batch version
        batch_size = 32
        displacements = np.random.uniform(size=(batch_size, dimension))
        lsigs = pysiglib.linear_sig(displacements, dimension, degree, n_jobs=-1)
    """

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    sig_len = sig_length(dimension, degree)
    data = SigInputHandler(displacement, dimension, "displacement")
    result = SigOutputHandler(data, sig_len)

    if data.device == "cpu":
        if data.is_batch:
            return batch_linear_sig_(data, result, dimension, degree, n_jobs)
        return linear_sig_(data, result, dimension, degree)
    else:
        return linear_sig_cuda_(data, result, dimension, degree)
