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
from .dtypes import CPSIG_LOG_SIG_JOIN, CUSIG_LOG_SIG_JOIN_CUDA
from .sig_length import log_sig_length
from .data_handlers import SigInputHandler, SigOutputHandler


def log_sig_join(
        log_sig : Union[np.ndarray, torch.tensor],
        displacement : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Extends a truncated log-signature by a single displacement vector using the
    Baker-Campbell-Hausdorff (BCH) formula. This is the log-signature analogue of
    ``sig_join``.

    Given a log-signature :math:`\\log S(x)` and a displacement :math:`v`, this computes

    .. math::

        \\log S(x * v) = \\text{BCH}(\\log S(x), v),

    where :math:`v` is embedded as a degree-1 element of the free Lie algebra.

    .. note::

        You must call ``pysiglib.prepare_log_sig(dimension, degree)`` before using this
        function. This precomputes the Lyndon basis and BCH coefficients needed internally.

    :param log_sig: The existing truncated log-signature, of shape ``(log_sig_length,)`` or ``(batch_size, log_sig_length)``.
    :type log_sig: numpy.ndarray | torch.tensor
    :param displacement: The displacement vector, of shape ``(dimension,)`` or ``(batch_size, dimension)``.
    :type displacement: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the log-signature, :math:`N`.
    :type degree: int
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Extended log-signature, :math:`\\log S(x * v)`.
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import pysiglib
        import numpy as np

        dimension = 5
        degree = 3
        pysiglib.prepare_log_sig(dimension, degree)

        path = np.random.uniform(size=(100, dimension))
        ls = pysiglib.log_sig(path, degree)

        displacement = np.random.uniform(size=(dimension,))
        extended_ls = pysiglib.log_sig_join(ls, displacement, dimension, degree)
    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    ls_len = log_sig_length(dimension, degree)
    logsig_data = SigInputHandler(log_sig, ls_len, "log_sig")
    disp_data = SigInputHandler(displacement, dimension, "displacement")

    if logsig_data.type_ != disp_data.type_:
        raise ValueError("log_sig and displacement must both be numpy arrays or both torch tensors")
    if logsig_data.dtype != disp_data.dtype:
        raise ValueError("log_sig and displacement must have the same dtype")
    if logsig_data.batch_shape != disp_data.batch_shape:
        raise ValueError("log_sig and displacement must have the same batch shape")
    if logsig_data.device != disp_data.device:
        raise ValueError("log_sig and displacement must be on the same device")

    result = SigOutputHandler(logsig_data, ls_len)

    if logsig_data.batch_size == 0:
        return result.data

    if logsig_data.device == "cpu":
        err_code = CPSIG_LOG_SIG_JOIN[logsig_data.dtype](
            logsig_data.data_ptr, disp_data.data_ptr, result.data_ptr,
            logsig_data.batch_size, dimension, degree, n_jobs)
    else:
        err_code = CUSIG_LOG_SIG_JOIN_CUDA[logsig_data.dtype](
            logsig_data.data_ptr, disp_data.data_ptr, result.data_ptr,
            logsig_data.batch_size, dimension, degree)
    if err_code:
        raise Exception("Error in pysiglib.log_sig_join: " + err_msg(err_code))
    return result.data
