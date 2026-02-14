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
from ctypes import c_uint64, POINTER, cast

import numpy as np
import torch

from .param_checks import check_word_or_word_list, check_type, check_non_neg
from .error_codes import err_msg
from .dtypes import CPSIG_SIG_COEF_BACKPROP, CPSIG_BATCH_SIG_COEF_BACKPROP
from .words import word_to_idx
from .data_handlers import SigInputHandler, PathInputHandler, SigOutputHandler, DeviceToHost, MultipleSigInputHandler, PathOutputHandler


def sig_coef_backprop_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr):
    err_code = CPSIG_SIG_COEF_BACKPROP[data.dtype](
        data.data_ptr,
        result.data_ptr,
        deriv_data.data[0].data_ptr,
        deriv_data.data[1].data_ptr,
        multi_indices_ptr,
        num_multi_indices,
        degrees_ptr,
        data.data_dimension,
        data.data_length,
        data.time_aug,
        data.lead_lag,
        data.end_time
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_coef: " + err_msg(err_code))
    return result.data

def batch_sig_coef_backprop_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr, n_jobs = 1):
    err_code = CPSIG_BATCH_SIG_COEF_BACKPROP[data.dtype](
        data.data_ptr,
        result.data_ptr,
        deriv_data.data[0].data_ptr,
        deriv_data.data[1].data_ptr,
        multi_indices_ptr,
        num_multi_indices,
        degrees_ptr,
        data.batch_size,
        data.data_dimension,
        data.data_length,
        data.time_aug,
        data.lead_lag,
        data.end_time,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_coef: " + err_msg(err_code))
    return result.data

def sig_coef_backprop(
        path : Union[np.ndarray, torch.tensor],
        word : Union[tuple[int, ...], list[tuple[int, ...]]],
        coef : Union[np.ndarray, torch.tensor],
        deriv : Union[np.ndarray, torch.tensor],
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes specific signature coefficients for a single path or a batch of paths. For
    a single path :math:`x`, the signature coeficient at a multi-index
    :math:`I = (i_1, i_2, \\ldots, i_k)` is given by

    .. math::

        S(x)^I_{[s,t]} := \\int_{s < t_1 < \\cdots < t_k < t} dx^{i_1}_{t_1} \\otimes dx^{i_2}_{t_2} \\otimes \\cdots \\otimes dx^{i_k}_{t_k}.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param word: Multi-indices :math:`I` at which to evaluate signature coefficients, given as a list
        of lists of integers in :math:`[0, d-1]`, where :math:`d` is the dimension of the path(s). For example,
        for a 2-dimensional path, one could pass ``[[0], [1,0], [0,1,1]]`` to compute the coefficients at
        the three multi-indices :math:`I = (0), (1,0), (0,1,1)`.
    :type word: tuple[int, ...] | list[tuple[int, ...]]
    :param time_aug: If set to True, will compute signature coefficients of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math:`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute signature coefficients of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param prefixes: If ``True``, will additionally return all prefixes of signature coefficients.
        These prefixes are extracted for free as a by-product of the computation.
        For example, passing ``word=[[1,2], [3,2,1]]`` with ``prefixes=True`` returns an
        output equivalent to passing ``word=[[1], [1,2], [3], [3,2], [3,2,1]]`` with ``prefixes=False``.
    :type prefixes: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Array of signature coefficients of shape ``(batch_size, num_coefs)``.
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        If the number of requested coefficients is large relative to the size of the full truncated signature,
        it is usually faster to call ``pysiglib.signature`` and extract the required coefficients using
        ``pysiglib.extract_sig_coefs``. This function is only faster when a very sparse collection
        of coefficients is required.

    .. note::

        Ideally, any array passed to ``pysiglib.sig_coef`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_coef`` will internally create a contiguous copy, which may be
        inefficient.

    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)

    # If path is on GPU, move to CPU
    device_handler = DeviceToHost([path], ["path"])
    path = device_handler.data[0]
    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    word = check_word_or_word_list(word, data.dimension, "word")

    deriv_data = MultipleSigInputHandler([coef, deriv], coef.shape[-1], ["coef", "deriv"])

    num_multi_indices = len(word)
    degrees = [len(idx) for idx in word]

    word = [torch.tensor(idx, dtype=torch.uint64, device = data.device) for idx in word]
    word = torch.concatenate(word, axis = 0)
    degrees = torch.tensor(degrees, dtype=torch.uint64, device=data.device)

    multi_indices_ptr = cast(word.data_ptr(), POINTER(c_uint64))
    degrees_ptr = cast(degrees.data_ptr(), POINTER(c_uint64))

    result = PathOutputHandler(data.data_length, data.data_dimension, data)

    if data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        res = batch_sig_coef_backprop_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr, n_jobs)
    else:
        res = sig_coef_backprop_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr)

    if device_handler.device is not None:
        res = res.to(device_handler.device)
    return res
