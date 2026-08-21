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

from .data_handlers import PathOutputHandler
from .param_checks import check_type, check_n_jobs
from .error_codes import err_msg
from .dtypes import CPSIG_TRANSFORM_PATH_BACKPROP, CUSIG_TRANSFORM_PATH_BACKPROP

from .data_handlers import PathInputHandler

def transform_path_backprop(
    derivs : Union[np.ndarray, torch.Tensor],
    *,
    time_aug : bool = False,
    lead_lag : bool = False,
    end_time : float = 1.,
    n_jobs : int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    This function is required to backpropagate through ``pysiglib.transform_path``.
    Given the derivatives of a scalar function :math:`F` with respect to the
    result of ``pysiglib.transform_path``,
    :math:`\\{\\partial F / \\partial \\tilde{x}_{t_i}\\}_{i=0}^\\tilde{L}`,
    returns the derivatives of :math:`F` with respect to the original path,
    :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^L`.

    :param derivs: The derivatives with respect to the result of ``pysiglib.transform_path``,
        :math:`\\{\\partial F / \\partial \\tilde{x}_{t_i}\\}_{i=0}^\\tilde{L}`.
    :type derivs: numpy.ndarray | torch.Tensor
    :param time_aug: If ``True``, assumes the derivatives are with respect to a time
        augmented path.
    :type time_aug: bool
    :param lead_lag: If ``True``, assumes the derivatives are with respect to a lead-lag
        transformed path.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives with respect to the original path,
        :math:`\\{\\partial F / x_{t_i}\\}_{i=0}^L`.
    :rtype: numpy.ndarray | torch.Tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.tensor([[0., 1.], [2., 3.]])
        transformed = pysiglib.transform_path(path, time_aug=True, lead_lag=True)
        derivs = torch.ones_like(transformed)
        path_derivs = pysiglib.transform_path_backprop(derivs, time_aug=True, lead_lag=True)
        print(path_derivs)
    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)

    if (not time_aug) and (not lead_lag):
        return derivs

    check_n_jobs(n_jobs)
    data = PathInputHandler(derivs, False, False, end_time, "path")
    length = (data.length + 1) // 2 if lead_lag else data.length
    dimension = data.dimension - 1 if time_aug else data.dimension
    if lead_lag:
        dimension = dimension // 2
    result = PathOutputHandler(length, dimension, data)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_TRANSFORM_PATH_BACKPROP[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size, dimension, length,
            time_aug, lead_lag, end_time, n_jobs)
    else:
        err_code = CUSIG_TRANSFORM_PATH_BACKPROP[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size, dimension, length,
            time_aug, lead_lag, end_time)
    if err_code:
        raise Exception("Error in pysiglib.transform_path_backprop: " + err_msg(err_code))
    return result.data