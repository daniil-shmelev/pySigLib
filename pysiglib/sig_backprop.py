# Copyright 2025 Daniil Shmelev
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
from .data_handlers import PathInputHandler, SigOutputHandler, PathOutputHandler, MultipleSigInputHandler, DeviceToHost
from .dtypes import CPSIG_SIG_BACKPROP, CPSIG_BATCH_SIG_BACKPROP, CPSIG_SIG_COMBINE_BACKPROP, CPSIG_BATCH_SIG_COMBINE_BACKPROP
from .sig_length import sig_length

def sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, dimension, degree):
    err_code = CPSIG_SIG_COMBINE_BACKPROP[sig_data.dtype](
        sig_data.sig_ptr[2],
        sig1_deriv.data_ptr,
        sig2_deriv.data_ptr,
        sig_data.sig_ptr[0],
        sig_data.sig_ptr[1],
        dimension,
        degree
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_combine_backprop: " + err_msg(err_code))
    return sig1_deriv.data, sig2_deriv.data

def batch_sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, dimension, degree, n_jobs):
    err_code = CPSIG_BATCH_SIG_COMBINE_BACKPROP[sig_data.dtype](
        sig_data.sig_ptr[2],
        sig1_deriv.data_ptr,
        sig2_deriv.data_ptr,
        sig_data.sig_ptr[0],
        sig_data.sig_ptr[1],
        sig_data.batch_size,
        dimension,
        degree,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_combine_backprop: " + err_msg(err_code))
    return sig1_deriv.data, sig2_deriv.data

def sig_combine_backprop(
        deriv : Union[np.ndarray, torch.tensor],
        sig1 : Union[np.ndarray, torch.tensor],
        sig2 : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        n_jobs : int = 1
):
    """
    This function is required to backpropagate through ``pysiglib.sig_combine``.
    Given the derivatives of a scalar function :math:`F` with respect to the
    result of ``pysiglib.sig_combine``, :math:`\\partial F / \\partial S(x_1 * x_2)`,
    returns the derivatives of :math:`F` with respect to the original two signatures,
    :math:`\\partial F / \\partial S(x_1)` and :math:`\\partial F / \\partial S(x_2)`.

    :param deriv: Derivative with respect to the combined signature,
        :math:`\\partial F / \\partial S(x_1 * x_2)`
    :type sig_combine_deriv: numpy.ndarray | torch.tensor
    :param sig1: The first truncated signature
    :type sig1: numpy.ndarray | torch.tensor
    :param sig2: The second truncated signature. Must have the same degree and dimension as the first.
    :type sig2: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signatures, :math:`N`
    :type degree: int
    :param time_aug: Whether time augmentation was applied before computing
        the signature.
    :type time_aug: bool
    :param lead_lag: Whether the lead lag transformation was applied before computing
        the signature.
    :type lead_lag: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives with respect to ``sig1`` and ``sig2``
    :rtype: Tuple[numpy.ndarray | torch.tensor, numpy.ndarray | torch.tensor]

    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)

    device_handler = DeviceToHost([deriv, sig1, sig2], ["deriv", "sig1", "sig2"])
    deriv, sig1, sig2 = device_handler.data

    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    sig_len = sig_length(aug_dimension, degree)
    sig_data = MultipleSigInputHandler([sig1, sig2, deriv], sig_len, ["sig1", "sig2", "sig_combined_deriv"])

    sig1_deriv = SigOutputHandler(sig_data, sig_len)
    sig2_deriv = SigOutputHandler(sig_data, sig_len)

    if sig_data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        res1, res2 = batch_sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, aug_dimension, degree, n_jobs)
    else:
        res1, res2 = sig_combine_backprop_(sig_data, sig1_deriv, sig2_deriv, aug_dimension, degree)

    if device_handler.device is not None:
        res1 = res1.to(device_handler.device)
        res2 = res2.to(device_handler.device)
    return res1, res2

def sig_backprop_(path_data, sig_data, result, degree):
    err_code = CPSIG_SIG_BACKPROP[path_data.dtype](
        path_data.data_ptr,
        result.data_ptr,
        sig_data.sig_ptr[1],
        sig_data.sig_ptr[0],
        path_data.data_dimension,
        path_data.data_length,
        degree,
        path_data. time_aug,
        path_data.lead_lag,
        path_data.end_time
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_backprop: " + err_msg(err_code))
    return result.data

def batch_sig_backprop_(path_data, sig_data, result, degree, n_jobs):
    err_code = CPSIG_BATCH_SIG_BACKPROP[path_data.dtype](
        path_data.data_ptr,
        result.data_ptr,
        sig_data.sig_ptr[1],
        sig_data.sig_ptr[0],
        path_data.batch_size,
        path_data.data_dimension,
        path_data.data_length,
        degree,
        path_data.time_aug,
        path_data.lead_lag,
        path_data.end_time,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_backprop: " + err_msg(err_code))
    return result.data

def sig_backprop(
        path : Union[np.ndarray, torch.tensor],
        sig : Union[np.ndarray, torch.tensor],
        sig_derivs : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    This function is required to backpropagate through the signature computation.
    Given the derivatives of a scalar function :math:`F` with respect to the
    signature, :math:`\\partial F / \\partial S(x)`, returns the
    derivatives of :math:`F` with respect to the underlying path,
    :math:`\\partial F / \\partial x`.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param sig: Signature(s) of the path or batch of paths.
    :type sig: numpy.ndarray | torch.tensor
    :param sig_derivs: Derivatives of the scalar function :math:`F` with respect to the signature(s),
        :math:`\\partial F / \\partial S(x)`. This must be an array of the same shape as the
        provided signature(s).
    :type sig_derivs: numpy.ndarray | torch.tensor
    :param degree: The truncation level of the signature, :math:`N`.
    :type degree: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :return: Derivatives of the scalar function :math:`F` with respect to the path(s), :math:`\\partial F / \\partial x`.
        This is an array of the same shape as the provided path(s).
    :rtype: numpy.ndarray | torch.tensor

    """
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)

    device_handler = DeviceToHost([path, sig, sig_derivs], ["path", "sig", "sig_derivs"])
    path, sig, sig_derivs = device_handler.data

    path_data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    sig_len = sig_length(path_data.dimension, degree)
    sig_data = MultipleSigInputHandler([sig, sig_derivs], sig_len, ["sig", "sig_derivs"])

    if path_data.type_ != sig_data.type_:
        raise ValueError("path, sig and sig_derivs must all be numpy arrays or torch tensors")
    if path_data.dtype != sig_data.dtype:
        raise ValueError("path, sig and sig_derivs must have the same dtype")

    result = PathOutputHandler(path_data.data_length, path_data.data_dimension, path_data)

    if path_data.is_batch != sig_data.is_batch or path_data.batch_size != sig_data.batch_size:
        raise ValueError("path, sig and sig_derivs must have the same batch sizes")

    if path_data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        res = batch_sig_backprop_(path_data, sig_data, result, degree, n_jobs)
    else:
        res = sig_backprop_(path_data, sig_data, result, degree)

    if device_handler.device is not None:
        res = res.to(device_handler.device)
    return res
