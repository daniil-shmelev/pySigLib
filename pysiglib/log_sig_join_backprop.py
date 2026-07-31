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
from .dtypes import CPSIG_LOG_SIG_JOIN_BACKPROP, CUSIG_LOG_SIG_JOIN_BACKPROP_CUDA
from .sig_length import log_sig_length
from .data_handlers import SigInputHandler, SigOutputHandler


def log_sig_join_backprop(
        d_out : Union[np.ndarray, torch.Tensor],
        log_sig : Union[np.ndarray, torch.Tensor],
        displacement : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        n_jobs : int = 1
):
    """
    This function is required to backpropagate through ``pysiglib.log_sig_join``.
    Given the derivatives of a scalar function :math:`F` with respect to the
    result of ``pysiglib.log_sig_join``, :math:`\\partial F / \\partial L(x * v)`,
    returns the derivatives of :math:`F` with respect to the original log-signature
    and the displacement vector,
    :math:`\\partial F / \\partial L(x)` and :math:`\\partial F / \\partial v`.

    .. note::

        ``log_sig`` is expected in the Lyndon bracket basis (``method=2`` output). You
        must call ``pysiglib.prepare_log_sig(dimension, degree, method=2)`` before using
        this function. This precomputes the Lyndon basis and BCH coefficients needed
        internally.

    :param d_out: Derivative with respect to the output of log_sig_join,
        :math:`\\partial F / \\partial L(x * v)`, of shape ``(..., log_sig_length)``.
    :type d_out: numpy.ndarray | torch.tensor
    :param log_sig: The original truncated log-signature, :math:`L(x)`,
        of shape ``(..., log_sig_length)``.
    :type log_sig: numpy.ndarray | torch.tensor
    :param displacement: The displacement vector, :math:`v`, of shape ``(..., dimension)``.
    :type displacement: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the log-signatures, :math:`N`
    :type degree: int
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives with respect to ``log_sig`` and ``displacement``
    :rtype: Tuple[numpy.ndarray | torch.tensor, numpy.ndarray | torch.tensor]

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        dimension, degree = 5, 3
        pysiglib.prepare_log_sig(dimension, degree, method=2)

        path = np.random.uniform(size=(100, dimension))
        ls = pysiglib.log_sig(path, degree, method=2)
        displacement = np.random.uniform(size=(dimension,))

        extended = pysiglib.log_sig_join(ls, displacement, dimension, degree)
        d_out = np.ones_like(extended)

        d_logsig, d_displacement = pysiglib.log_sig_join_backprop(
            d_out, ls, displacement, dimension, degree
        )
        print(d_logsig.shape, d_displacement.shape)

    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    ls_len = log_sig_length(dimension, degree)

    d_out_data = SigInputHandler(d_out, ls_len, "d_out")
    logsig_data = SigInputHandler(log_sig, ls_len, "log_sig")
    disp_data = SigInputHandler(displacement, dimension, "displacement")

    if d_out_data.type_ != logsig_data.type_ or logsig_data.type_ != disp_data.type_:
        raise ValueError("d_out, log_sig and displacement must all be numpy arrays or all torch tensors")
    if d_out_data.dtype != logsig_data.dtype or logsig_data.dtype != disp_data.dtype:
        raise ValueError("d_out, log_sig and displacement must have the same dtype")
    if not (d_out_data.batch_shape == logsig_data.batch_shape == disp_data.batch_shape):
        raise ValueError("d_out, log_sig and displacement must have the same batch shape")
    if d_out_data.device != logsig_data.device or logsig_data.device != disp_data.device:
        raise ValueError("d_out, log_sig and displacement must be on the same device")

    d_logsig = SigOutputHandler(d_out_data, ls_len)
    d_disp = SigOutputHandler(d_out_data, dimension)

    if d_out_data.batch_size == 0:
        return d_logsig.data, d_disp.data

    if d_out_data.device == "cpu":
        err_code = CPSIG_LOG_SIG_JOIN_BACKPROP[d_out_data.dtype](
            d_out_data.data_ptr, d_logsig.data_ptr, d_disp.data_ptr,
            logsig_data.data_ptr, disp_data.data_ptr,
            d_out_data.batch_size, dimension, degree, n_jobs)
    else:
        err_code = CUSIG_LOG_SIG_JOIN_BACKPROP_CUDA[d_out_data.dtype](
            d_out_data.data_ptr, d_logsig.data_ptr, d_disp.data_ptr,
            logsig_data.data_ptr, disp_data.data_ptr,
            d_out_data.batch_size, dimension, degree)
    if err_code:
        raise Exception("Error in pysiglib.log_sig_join_backprop: " + err_msg(err_code))
    return d_logsig.data, d_disp.data
