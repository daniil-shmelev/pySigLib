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
from .dtypes import CPSIG_SIG_JOIN, CPSIG_BATCH_SIG_JOIN
from .load_siglib import BUILT_WITH_CUDA
from .sig_length import sig_length
from .data_handlers import SigInputHandler, SigOutputHandler

if BUILT_WITH_CUDA:
    from .dtypes import CUSIG_BATCH_SIG_JOIN_CUDA


def sig_join_(sig_data, disp_data, result, dimension, degree, prepend):
    err_code = CPSIG_SIG_JOIN[sig_data.dtype](
        sig_data.data_ptr,
        disp_data.data_ptr,
        result.data_ptr,
        dimension,
        degree,
        prepend
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_join: " + err_msg(err_code))
    return result.data

def batch_sig_join_(sig_data, disp_data, result, dimension, degree, prepend, n_jobs):
    err_code = CPSIG_BATCH_SIG_JOIN[sig_data.dtype](
        sig_data.data_ptr,
        disp_data.data_ptr,
        result.data_ptr,
        sig_data.batch_size,
        dimension,
        degree,
        prepend,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_join: " + err_msg(err_code))
    return result.data


def sig_join_cuda_(sig_data, disp_data, result, dimension, degree, prepend):
    err_code = CUSIG_BATCH_SIG_JOIN_CUDA[sig_data.dtype](
        sig_data.data_ptr,
        disp_data.data_ptr,
        result.data_ptr,
        sig_data.batch_size,
        dimension,
        degree,
        prepend
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_join (CUDA): " + err_msg(err_code))
    return result.data


def sig_join(
        sig : Union[np.ndarray, torch.tensor],
        displacement : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        prepend : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Extends a truncated signature by a single displacement vector. This is equivalent to
    computing ``sig_combine(sig, linear_sig(displacement))``, but is more efficient as it
    avoids constructing the intermediate linear signature.

    Given a signature :math:`S(x)` and a displacement :math:`v`, this computes

    .. math::

        S(x * v) = S(x) \\otimes S(v),

    where :math:`S(v)` is the signature of the linear path defined by :math:`v`.

    :param sig: The existing truncated signature, of shape ``(sig_length,)`` or ``(batch_size, sig_length)``.
    :type sig: numpy.ndarray | torch.tensor
    :param displacement: The displacement vector, of shape ``(dimension,)`` or ``(batch_size, dimension)``.
    :type displacement: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`.
    :type degree: int
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Extended signature, :math:`S(x * v)`.
    :rtype: numpy.ndarray | torch.tensor

    Example usage::

        import pysiglib
        import numpy as np

        dimension = 5
        degree = 3

        path = np.random.uniform(size=(100, dimension))
        sig = pysiglib.sig(path, degree)

        displacement = np.random.uniform(size=(dimension,))
        extended_sig = pysiglib.sig_join(sig, displacement, dimension, degree)
    """

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(n_jobs, "n_jobs", int)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    sig_len = sig_length(dimension, degree)
    sig_data = SigInputHandler(sig, sig_len, "sig")
    disp_data = SigInputHandler(displacement, dimension, "displacement")

    if sig_data.type_ != disp_data.type_:
        raise ValueError("sig and displacement must both be numpy arrays or both torch tensors")
    if sig_data.dtype != disp_data.dtype:
        raise ValueError("sig and displacement must have the same dtype")
    if sig_data.is_batch != disp_data.is_batch:
        raise ValueError("sig and displacement must both be unbatched or both batched")
    if sig_data.is_batch and sig_data.batch_size != disp_data.batch_size:
        raise ValueError("sig and displacement must have the same batch size")
    if sig_data.device != disp_data.device:
        raise ValueError("sig and displacement must be on the same device")

    result = SigOutputHandler(sig_data, sig_len)

    if sig_data.device == "cpu":
        if sig_data.is_batch:
            return batch_sig_join_(sig_data, disp_data, result, dimension, degree, prepend, n_jobs)
        return sig_join_(sig_data, disp_data, result, dimension, degree, prepend)
    else:
        return sig_join_cuda_(sig_data, disp_data, result, dimension, degree, prepend)
