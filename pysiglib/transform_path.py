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

from .load_siglib import BUILT_WITH_CUDA
from .data_handlers import PathOutputHandler
from .param_checks import check_type
from .error_codes import err_msg
from .dtypes import CPSIG_TRANSFORM_PATH, CPSIG_BATCH_TRANSFORM_PATH, CUSIG_TRANSFORM_PATH_CUDA, CUSIG_BATCH_TRANSFORM_PATH_CUDA

from .data_handlers import PathInputHandler

def transform_path_(data, result):
    err_code = CPSIG_TRANSFORM_PATH[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.data_dimension,
        data.data_length,
        data.time_aug,
        data.lead_lag,
        data.end_time
    )

    if err_code:
        raise Exception("Error in pysiglib.transform_path: " + err_msg(err_code))
    return result.data

def transform_path_cuda_(data, result):
    err_code = CUSIG_TRANSFORM_PATH_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.data_dimension,
        data.data_length,
        data.time_aug,
        data.lead_lag,
        data.end_time
    )

    if err_code:
        raise Exception("Error in pysiglib.transform_path: " + err_msg(err_code))
    return result.data

def batch_transform_path_(data, result, n_jobs):
    err_code = CPSIG_BATCH_TRANSFORM_PATH[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        data.data_dimension,
        data.data_length,
        data.time_aug,
        data.lead_lag,
        data.end_time,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.transform_path: " + err_msg(err_code))
    return result.data

def batch_transform_path_cuda_(data, result, n_jobs):
    err_code = CUSIG_BATCH_TRANSFORM_PATH_CUDA[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        data.data_dimension,
        data.data_length,
        data.time_aug,
        data.lead_lag,
        data.end_time
    )

    if err_code:
        raise Exception("Error in pysiglib.transform_path: " + err_msg(err_code))
    return result.data

def transform_path(
    path : Union[np.ndarray, torch.tensor],
    time_aug : bool = False,
    lead_lag : bool = False,
    end_time : float = 1.,
    n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    This function applies time-augmentation and/or the lead-lag transformation to a path. Given a
    :math:`d`-dimensional path :math:`(X_{t_i})_{i=1}^L`, the time-augmented path is formed by adding time as the
    last channel of the path,

    .. math::

        \\widehat{X}_{t_i} := (X_{t_i}, t_i).

    The lead-lag transformation is defined by

    .. math::

        X^{LL}_{t_i} := (X^{\\text{Lag}}_{t_i}, X^{\\text{Lead}}_{t_i})

    .. math::

        X^{\\text{Lead}}_{t_i} :=
        \\begin{cases}
            X_{t_k} & \\text{if } i = 2k, \\\\
            X_{t_k} & \\text{if } i = 2k - 1,
        \\end{cases}

    .. math::

        X^{\\text{Lag}}_{t_i} :=
        \\begin{cases}
            X_{t_k} & \\text{if } i = 2k, \\\\
            X_{t_k} & \\text{if } i = 2k + 1,
        \\end{cases}

    so that

    .. math::

        (X^{\\text{Lag}}_{t_i})_{i=0}^L = (X_{t_0}, X_{t_0}, X_{t_1}, X_{t_1}, X_{t_2}, \\ldots),

    .. math::

        (X^{\\text{Lead}}_{t_i})_{i=0}^L = (X_{t_0}, X_{t_1}, X_{t_1}, X_{t_2}, X_{t_2}, \\ldots).

    When both ``time_aug`` and ``lead_lag`` are set to ``True``, time-augmentation is applied
    after the lead-lag transformation.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param time_aug: If ``True``, applies time-augmentation by adding a linear channel to the path
        spanning :math:`[0, t_L]`. :math:`t_L` is given by the parameter ``end_time`` and defaults to 1.
    :type time_aug: bool
    :param lead_lag: If ``True``, applies the lead-lag transform.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Transformed paths.
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        Note that in the definition of the lead-lag transformation, we have intentionally chosen
        :math:`X^{LL} := (X^{\\text{Lag}}, X^{\\text{Lead}})` rather than the more commonly used
        order of channels :math:`X^{LL} := (X^{\\text{Lead}}, X^{\\text{Lag}})`.

    .. important::

        This function is provided for convenience only, and one should prefer the in-built flags for
        these transformations within ``pysiglib`` functions where available. For example, running
        ``pysiglib.sig`` with ``lead_lag=True`` will be faster and more memory-efficient than
        pre-computing the lead-lag transform and passing it to ``pysiglib.sig``, as the former
        method will never explicitly compute or store the lead-lag transform, and will instead
        modify the signature computation directly.

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.tensor([[0., 1.], [2., 3.]])
        transformed = pysiglib.transform_path(path, time_aug=True, lead_lag=True)
        print(transformed)
    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)

    if (not time_aug) and (not lead_lag):
        return path

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    result = PathOutputHandler(data.length, data.dimension, data)
    if data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")

        if data.device == "cpu":
            return batch_transform_path_(data, result, n_jobs)
        else:
            if not BUILT_WITH_CUDA:
                raise RuntimeError("pySigLib was built without CUDA - data must be moved to CPU.")
            return batch_transform_path_cuda_(data, result, n_jobs)

    if data.device == "cpu":
        return transform_path_(data, result)
    else:
        if not BUILT_WITH_CUDA:
            raise RuntimeError("pySigLib was built without CUDA - data must be moved to CPU.")
        return transform_path_cuda_(data, result)