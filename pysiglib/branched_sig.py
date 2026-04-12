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
from .sig_length import aug_dim
from .dtypes import (CPSIG_BRANCHED_SIG, CPSIG_BRANCHED_SIG_COMBINE,
                     CUSIG_BRANCHED_SIG_CUDA, CUSIG_BRANCHED_SIG_COMBINE_CUDA)
from .data_handlers import PathInputHandler, SigOutputHandler, MultipleSigInputHandler
from .load_siglib import CPSIG


def prepare_branched_sig(
        dimension: int,
        degree: int,
        use_disk: bool = False,
        time_aug: bool = False,
        lead_lag: bool = False
):
    """
    Precomputes the tree enumeration and Connes-Kreimer coproduct tables
    needed for branched signature computation. Must be called before
    ``branched_sig()`` for a given ``(dimension, degree)`` pair.

    If ``time_aug`` or ``lead_lag`` are set, the cache is prepared for
    the augmented dimension automatically.

    :param dimension: Dimension of the underlying path.
    :param degree: Maximum tree order (number of nodes).
    :param use_disk: If True, cache the precomputed tables to disk for
        faster loading in future sessions. Uses the same cache directory
        as ``set_cache_dir()`` / ``prepare_log_sig()``.
    :param time_aug: If True, prepare for time-augmented paths (dim + 1).
    :param lead_lag: If True, prepare for lead-lag transformed paths (2 * dim).
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(use_disk, "use_disk", bool)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    err_code = CPSIG.prepare_branched_sig(aug_dimension, degree, use_disk)
    if err_code:
        raise Exception("Error in pysiglib.prepare_branched_sig: " + err_msg(err_code))


def branched_sig_length(dimension: int, degree: int) -> int:
    """
    Returns the length of a truncated branched signature.

    :param dimension: Dimension of the underlying path.
    :param degree: Maximum tree order (number of nodes).
    :return: Length of the branched signature array.
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    out = CPSIG.branched_sig_length(dimension, degree)
    if out == 0:
        raise ValueError("Invalid parameters or integer overflow in branched_sig_length")
    return out


def branched_sig(
        path: Union[np.ndarray, torch.Tensor],
        degree: int,
        n_jobs: int = 1,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the truncated branched signature of a path or batch of paths.

    The branched signature extends the standard path signature to iterated
    integrals indexed by decorated rooted trees, following Gubinelli (2010).

    Must call ``prepare_branched_sig(dimension, degree)`` before first use,
    where ``dimension`` is the augmented dimension (accounting for
    ``time_aug`` and ``lead_lag``).

    :param path: Path of shape ``(length, dimension)`` or ``(batch_size, length, dimension)``.
    :param degree: Maximum tree order (number of nodes).
    :param n_jobs: Number of parallel threads for batch processing.
    :param time_aug: If True, prepend a time channel to the path.
    :param lead_lag: If True, apply the lead-lag transformation.
    :param end_time: End time for the time augmentation channel.
    :return: Branched signature array of shape ``(bsig_len,)`` or ``(batch_size, bsig_len)``.
    """
    check_type(degree, "degree", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    dimension = data.data_dimension
    aug_dimension = data.dimension
    bsig_len = CPSIG.branched_sig_length(aug_dimension, degree)
    result = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            dimension, data.data_length, degree, n_jobs,
            data.time_aug, data.lead_lag, data.end_time)
    else:
        err_code = CUSIG_BRANCHED_SIG_CUDA[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            dimension, data.data_length, degree,
            data.time_aug, data.lead_lag, data.end_time)
    if err_code:
        raise Exception("Error in pysiglib.branched_sig: " + err_msg(err_code))
    return result.data


def branched_sig_combine(
        bsig1: Union[np.ndarray, torch.Tensor],
        bsig2: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        n_jobs: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Combines two truncated branched signatures via the Butcher product
    (the analogue of Chen's identity for branched rough paths).

    :param bsig1: First branched signature.
    :param bsig2: Second branched signature.
    :param dimension: Dimension of the underlying path.
    :param degree: Maximum tree order (number of nodes).
    :param n_jobs: Number of parallel threads for batch processing.
    :return: Combined branched signature.
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    bsig_len = CPSIG.branched_sig_length(dimension, degree)
    data = MultipleSigInputHandler([bsig1, bsig2], bsig_len, ["bsig1", "bsig2"])
    result = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_COMBINE[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, dimension, degree, n_jobs)
    else:
        err_code = CUSIG_BRANCHED_SIG_COMBINE_CUDA[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, dimension, degree)
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_combine: " + err_msg(err_code))
    return result.data
