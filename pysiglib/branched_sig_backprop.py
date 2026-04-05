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
from .dtypes import (CPSIG_BRANCHED_SIG_BACKPROP, CPSIG_BATCH_BRANCHED_SIG_BACKPROP,
                     CPSIG_BRANCHED_SIG_COMBINE_BACKPROP, CPSIG_BATCH_BRANCHED_SIG_COMBINE_BACKPROP)
from .data_handlers import PathInputHandler, PathOutputHandler, MultipleSigInputHandler, SigOutputHandler
from .load_siglib import CPSIG


def branched_sig_backprop(
        path: Union[np.ndarray, torch.Tensor],
        bsig: Union[np.ndarray, torch.Tensor],
        bsig_derivs: Union[np.ndarray, torch.Tensor],
        degree: int,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        n_jobs: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Backpropagates through the branched signature computation.

    Given the forward branched signature ``bsig = branched_sig(path, degree)``
    and upstream derivatives ``bsig_derivs = dF/d(bsig)``, computes
    ``dF/d(path)``.

    :param path: Input path, shape ``(length, dimension)`` or ``(batch, length, dimension)``.
    :param bsig: Forward branched signature output.
    :param bsig_derivs: Upstream derivatives w.r.t. the branched signature.
    :param degree: Maximum tree order (must match forward call).
    :param time_aug: Whether time augmentation was used in the forward pass.
    :param lead_lag: Whether lead-lag was used in the forward pass.
    :param end_time: End time for time augmentation.
    :param n_jobs: Number of parallel threads for batch processing.
    :return: Path derivatives, same shape as ``path``.
    """
    check_type(degree, "degree", int)
    check_type(n_jobs, "n_jobs", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_non_neg(degree, "degree")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    path_data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    dimension = path_data.data_dimension
    aug_dimension = path_data.dimension
    bsig_len = CPSIG.branched_sig_length(aug_dimension, degree)
    sig_data = MultipleSigInputHandler([bsig, bsig_derivs], bsig_len, ["bsig", "bsig_derivs"])
    result = PathOutputHandler(path_data.data_length, path_data.data_dimension, path_data)

    if path_data.device == "cuda":
        from .dtypes import CUSIG_BRANCHED_SIG_BACKPROP, CUSIG_BATCH_BRANCHED_SIG_BACKPROP
        if path_data.is_batch:
            err_code = CUSIG_BATCH_BRANCHED_SIG_BACKPROP[path_data.dtype](
                path_data.data_ptr,
                result.data_ptr,
                sig_data.sig_ptr[1],
                sig_data.sig_ptr[0],
                path_data.batch_size,
                dimension,
                path_data.data_length,
                degree,
                path_data.time_aug,
                path_data.lead_lag,
                path_data.end_time
            )
        else:
            err_code = CUSIG_BRANCHED_SIG_BACKPROP[path_data.dtype](
                path_data.data_ptr,
                result.data_ptr,
                sig_data.sig_ptr[1],
                sig_data.sig_ptr[0],
                dimension,
                path_data.data_length,
                degree,
                path_data.time_aug,
                path_data.lead_lag,
                path_data.end_time
            )
    else:
        if path_data.is_batch:
            err_code = CPSIG_BATCH_BRANCHED_SIG_BACKPROP[path_data.dtype](
                path_data.data_ptr,
                result.data_ptr,
                sig_data.sig_ptr[1],
                sig_data.sig_ptr[0],
                path_data.batch_size,
                dimension,
                path_data.data_length,
                degree,
                n_jobs,
                path_data.time_aug,
                path_data.lead_lag,
                path_data.end_time
            )
        else:
            err_code = CPSIG_BRANCHED_SIG_BACKPROP[path_data.dtype](
                path_data.data_ptr,
                result.data_ptr,
                sig_data.sig_ptr[1],
                sig_data.sig_ptr[0],
                dimension,
                path_data.data_length,
                degree,
                path_data.time_aug,
                path_data.lead_lag,
                path_data.end_time
            )

    if err_code:
        raise Exception("Error in pysiglib.branched_sig_backprop: " + err_msg(err_code))
    return result.data


def branched_sig_combine_backprop(
        derivs: Union[np.ndarray, torch.Tensor],
        bsig1: Union[np.ndarray, torch.Tensor],
        bsig2: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        n_jobs: int = 1
) -> tuple:
    """
    Backpropagates through the branched signature combine (Butcher product).

    Given ``out = branched_sig_combine(bsig1, bsig2, dimension, degree)``
    and upstream derivatives ``derivs = dF/d(out)``, computes
    ``(dF/d(bsig1), dF/d(bsig2))``.

    :param derivs: Upstream derivatives, same shape as combine output.
    :param bsig1: First branched signature input to the forward combine.
    :param bsig2: Second branched signature input to the forward combine.
    :param dimension: Dimension of the underlying path.
    :param degree: Maximum tree order.
    :param n_jobs: Number of parallel threads for batch processing.
    :return: Tuple ``(dF/d(bsig1), dF/d(bsig2))``.
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(n_jobs, "n_jobs", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    bsig_len = CPSIG.branched_sig_length(dimension, degree)
    data = MultipleSigInputHandler([derivs, bsig1, bsig2], bsig_len, ["derivs", "bsig1", "bsig2"])
    result1 = SigOutputHandler(data, bsig_len)
    result2 = SigOutputHandler(data, bsig_len)

    if data.is_batch:
        err_code = CPSIG_BATCH_BRANCHED_SIG_COMBINE_BACKPROP[data.dtype](
            data.sig_ptr[1],
            data.sig_ptr[2],
            data.sig_ptr[0],
            result1.data_ptr,
            result2.data_ptr,
            data.batch_size,
            dimension,
            degree,
            n_jobs
        )
    else:
        err_code = CPSIG_BRANCHED_SIG_COMBINE_BACKPROP[data.dtype](
            data.sig_ptr[1],
            data.sig_ptr[2],
            data.sig_ptr[0],
            result1.data_ptr,
            result2.data_ptr,
            dimension,
            degree
        )

    if err_code:
        raise Exception("Error in pysiglib.branched_sig_combine_backprop: " + err_msg(err_code))
    return result1.data, result2.data
