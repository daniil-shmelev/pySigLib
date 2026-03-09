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

from .param_checks import check_word_or_word_list, check_type
from .error_codes import err_msg
from .dtypes import CPSIG_SIG_COEF_BACKPROP, CPSIG_BATCH_SIG_COEF_BACKPROP, CUSIG_SIG_COEF_BACKPROP_CUDA, CUSIG_BATCH_SIG_COEF_BACKPROP_CUDA
from .data_handlers import PathInputHandler, MultipleSigInputHandler, PathOutputHandler


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
        raise Exception("Error in pysiglib.sig_coef_backprop: " + err_msg(err_code))
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
        raise Exception("Error in pysiglib.sig_coef_backprop: " + err_msg(err_code))
    return result.data

def sig_coef_backprop_cuda_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr):
    err_code = CUSIG_SIG_COEF_BACKPROP_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        deriv_data.data[0].data_ptr,
        deriv_data.data[1].data_ptr,
        multi_indices_ptr,
        num_multi_indices,
        degrees_ptr,
        data.data_dimension,
        data.data_length
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_coef_backprop (CUDA): " + err_msg(err_code))
    return result.data

def batch_sig_coef_backprop_cuda_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr):
    err_code = CUSIG_BATCH_SIG_COEF_BACKPROP_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        deriv_data.data[0].data_ptr,
        deriv_data.data[1].data_ptr,
        multi_indices_ptr,
        num_multi_indices,
        degrees_ptr,
        data.batch_size,
        data.data_dimension,
        data.data_length
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_coef_backprop (CUDA): " + err_msg(err_code))
    return result.data

def sig_coef_backprop(
        path : Union[np.ndarray, torch.tensor],
        words : Union[tuple[int, ...], list[tuple[int, ...]]],
        coefs : Union[np.ndarray, torch.tensor],
        derivs : Union[np.ndarray, torch.tensor],
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    This function is required to backpropagate through signature coefficient computation.
    Given the derivatives of a scalar function :math:`F` with respect to the
    signature coefficients, :math:`\\partial F / \\partial S(x)^I`, returns the
    derivatives of :math:`F` with respect to the underlying path,
    :math:`\\partial F / \\partial x`. Note that ``coefs`` must be generated using
    ``pysiglib.sig_coef`` using ``prefixes=True``, and ``derivs`` must be the derivatives
    with respect to this extended array.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param words: Multi-indices :math:`I` indexing the signature coefficients, given as a list
        of lists of integers in :math:`[0, d-1]`, where :math:`d` is the dimension of the path(s).
    :type words: tuple[int, ...] | list[tuple[int, ...]]
    :param coefs: Signature coefficients of the path or batch of paths, generated using
        ``pysiglib.sig_coef`` using ``prefixes=True``.
    :type coefs: numpy.ndarray | torch.tensor
    :param derivs: Derivatives of the scalar function :math:`F` with respect to the signature coefficients,
        :math:`\\partial F / \\partial S(x)^I`. This must be an array of the same shape as the
        provided coefficients.
    :type derivs: numpy.ndarray | torch.tensor
    :param time_aug: Whether the signature coefficients were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signature coefficients were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives of the scalar function :math:`F` with respect to the path(s), :math:`\\partial F / \\partial x`.
        This is an array of the same shape as the provided path(s).
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        words = [(0,), (1, 0), (1, 2, 3)]
        # Must generate coefs with prefixes=True for backprop
        coefs = pysiglib.sig_coef(path, words, prefixes=True)
        derivs = torch.ones_like(coefs)
        path_derivs = pysiglib.sig_coef_backprop(path, words, coefs, derivs)
        print(path_derivs)

    .. code-block:: python

        # Backprop with time augmentation and lead-lag
        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        words = [(0,), (1, 2)]
        # Must generate coefs with prefixes=True for backprop
        coefs = pysiglib.sig_coef(path, words, time_aug=True, lead_lag=True, prefixes=True)
        derivs = torch.ones_like(coefs)
        path_derivs = pysiglib.sig_coef_backprop(
            path, words, coefs, derivs, time_aug=True, lead_lag=True,
        )
        print(path_derivs)

    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    words = check_word_or_word_list(words, data.dimension, "word")

    coefs_len = 0
    for idx in words:
        coefs_len += len(idx) if idx else 1

    if coefs.shape[-1] != coefs_len:
        raise ValueError("Expected coefs.shape[-1] == " + str(coefs_len) + ". Please make sure coefs was generated using prefixes=True.")

    deriv_data = MultipleSigInputHandler([coefs, derivs], coefs_len, ["coef", "deriv"])

    num_multi_indices = len(words)
    degrees = [len(idx) for idx in words]

    flat_indices = [i for idx in words for i in idx]

    if data.device == "cpu":
        words_t = torch.tensor(flat_indices, dtype=torch.uint64)
        degrees_t = torch.tensor(degrees, dtype=torch.uint64)
    else:
        words_t = torch.tensor(flat_indices, dtype=torch.uint64, device=path.device)
        degrees_t = torch.tensor(degrees, dtype=torch.uint64, device=path.device)

    multi_indices_ptr = cast(words_t.data_ptr(), POINTER(c_uint64))
    degrees_ptr = cast(degrees_t.data_ptr(), POINTER(c_uint64))

    result = PathOutputHandler(data.data_length, data.data_dimension, data)

    if data.device == "cpu":
        if data.is_batch:
            check_type(n_jobs, "n_jobs", int)
            if n_jobs == 0:
                raise ValueError("n_jobs cannot be 0")
            res = batch_sig_coef_backprop_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr, n_jobs)
        else:
            res = sig_coef_backprop_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr)
    else:
        if data.is_batch:
            res = batch_sig_coef_backprop_cuda_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr)
        else:
            res = sig_coef_backprop_cuda_(data, result, deriv_data, multi_indices_ptr, num_multi_indices, degrees_ptr)

    return res
