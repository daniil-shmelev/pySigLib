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
from .dtypes import (CPSIG_BRANCHED_SIG_BACKPROP,
                     CPSIG_BRANCHED_SIG_COMBINE_BACKPROP,
                     CUSIG_BRANCHED_SIG_BACKPROP_CUDA,
                     CUSIG_BRANCHED_SIG_COMBINE_BACKPROP_CUDA)
from .data_handlers import PathInputHandler, PathOutputHandler, MultipleSigInputHandler, SigOutputHandler
from .branched_sig import _inv_permute_bsig, _permute_bsig, branched_sig_length, _infer_branched_scalar_term, _check_cuda_num_trees


def branched_sig_backprop(
        path: Union[np.ndarray, torch.Tensor],
        bsig: Union[np.ndarray, torch.Tensor],
        bsig_derivs: Union[np.ndarray, torch.Tensor],
        degree: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        tree_order: str = "recursive",
        planar: bool = False,
        n_jobs: int = 1,
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
    :param tree_order: Tree ordering convention of ``bsig`` and ``bsig_derivs``.
        ``"recursive"`` (default) uses the recursive construction order.
        ``"canonical"`` uses the shape-first order matching :func:`tree_to_idx`.
    :param planar: If True, backpropagate through planar branched signature.
    :param n_jobs: Number of parallel threads for batch processing.
    :return: Path derivatives, same shape as ``path``.
    """
    if tree_order not in ("recursive", "canonical"):
        raise ValueError(f"tree_order must be 'recursive' or 'canonical', got {tree_order!r}")
    check_type(degree, "degree", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(planar, "planar", bool)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    path_data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    dimension = path_data.data_dimension
    aug_dimension = path_data.dimension

    scalar_term = _infer_branched_scalar_term(bsig, aug_dimension, degree, planar=planar)
    if tree_order != "recursive":
        bsig = _inv_permute_bsig(bsig, aug_dimension, degree, planar=planar, scalar_term=scalar_term)
        bsig_derivs = _inv_permute_bsig(bsig_derivs, aug_dimension, degree, planar=planar, scalar_term=scalar_term)

    bsig_len = branched_sig_length(aug_dimension, degree, planar=planar, scalar_term=scalar_term)
    sig_data = MultipleSigInputHandler([bsig, bsig_derivs], bsig_len, ["bsig", "bsig_derivs"])
    result = PathOutputHandler(path_data.data_length, path_data.data_dimension, path_data)

    if path_data.batch_size == 0:
        return result.data

    if path_data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_BACKPROP[path_data.dtype](
            path_data.data_ptr, result.data_ptr,
            sig_data.sig_ptr[1], sig_data.sig_ptr[0],
            path_data.batch_size, dimension, path_data.data_length, degree, n_jobs,
            path_data.time_aug, path_data.lead_lag, path_data.end_time, planar, scalar_term)
    else:
        _check_cuda_num_trees(aug_dimension, degree, planar, "branched_sig_backprop")
        err_code = CUSIG_BRANCHED_SIG_BACKPROP_CUDA[path_data.dtype](
            path_data.data_ptr, result.data_ptr,
            sig_data.sig_ptr[1], sig_data.sig_ptr[0],
            path_data.batch_size, dimension, path_data.data_length, degree,
            path_data.time_aug, path_data.lead_lag, path_data.end_time, planar, scalar_term)
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_backprop: " + err_msg(err_code))
    return result.data


def branched_sig_combine_backprop(
        derivs: Union[np.ndarray, torch.Tensor],
        bsig1: Union[np.ndarray, torch.Tensor],
        bsig2: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        *,
        tree_order: str = "recursive",
        planar: bool = False,
        n_jobs: int = 1,
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
    :param tree_order: Tree ordering convention of ``derivs``, ``bsig1``, ``bsig2`` and the returned gradients.
        ``"recursive"`` (default) uses the recursive construction order.
        ``"canonical"`` uses the shape-first order matching :func:`tree_to_idx`.
    :param planar: If True, backpropagate through planar branched sig combine.
    :param n_jobs: Number of parallel threads for batch processing.
    :return: Tuple ``(dF/d(bsig1), dF/d(bsig2))``, in the same scalar-term format as the inputs.
    """
    if tree_order not in ("recursive", "canonical"):
        raise ValueError(f"tree_order must be 'recursive' or 'canonical', got {tree_order!r}")
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    scalar_term = _infer_branched_scalar_term(bsig1, dimension, degree, planar=planar)
    if tree_order != "recursive":
        derivs = _inv_permute_bsig(derivs, dimension, degree, planar=planar, scalar_term=scalar_term)
        bsig1 = _inv_permute_bsig(bsig1, dimension, degree, planar=planar, scalar_term=scalar_term)
        bsig2 = _inv_permute_bsig(bsig2, dimension, degree, planar=planar, scalar_term=scalar_term)

    bsig_len = branched_sig_length(dimension, degree, planar=planar, scalar_term=scalar_term)
    data = MultipleSigInputHandler([derivs, bsig1, bsig2], bsig_len, ["derivs", "bsig1", "bsig2"])
    result1 = SigOutputHandler(data, bsig_len)
    result2 = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result1.data, result2.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_COMBINE_BACKPROP[data.dtype](
            data.sig_ptr[1], data.sig_ptr[2], data.sig_ptr[0],
            result1.data_ptr, result2.data_ptr,
            data.batch_size, dimension, degree, n_jobs, planar, scalar_term)
    else:
        _check_cuda_num_trees(dimension, degree, planar, "branched_sig_combine_backprop")
        err_code = CUSIG_BRANCHED_SIG_COMBINE_BACKPROP_CUDA[data.dtype](
            data.sig_ptr[1], data.sig_ptr[2], data.sig_ptr[0],
            result1.data_ptr, result2.data_ptr,
            data.batch_size, dimension, degree, planar, scalar_term)
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_combine_backprop: " + err_msg(err_code))
    if tree_order != "recursive":
        _permute_bsig(result1.data, dimension, degree, planar=planar, scalar_term=scalar_term)
        _permute_bsig(result2.data, dimension, degree, planar=planar, scalar_term=scalar_term)
    return result1.data, result2.data
