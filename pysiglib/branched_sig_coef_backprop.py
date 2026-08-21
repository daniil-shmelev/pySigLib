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

from ctypes import POINTER, c_uint64, cast
from typing import Union

import numpy as np
import torch

from .branched_sig_coef import _branched_coef_data
from .data_handlers import CorrectionInputHandler, MultipleSigInputHandler, PathInputHandler, PathOutputHandler
from .dtypes import CPSIG_BRANCHED_SIG_COEF_BACKPROP, CUSIG_BRANCHED_SIG_COEF_BACKPROP
from .error_codes import err_msg
from .param_checks import check_n_jobs, check_type


def branched_sig_coef_backprop(
        path: Union[np.ndarray, torch.Tensor],
        trees,
        coefs: Union[np.ndarray, torch.Tensor],
        derivs: Union[np.ndarray, torch.Tensor],
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        planar: bool = False,
        correction=None,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Backpropagates through selected branched-signature coefficients.

    The basis elements and options must match the forward call and the preceding
    :func:`prepare_branched_sig_coef` call. ``coefs`` contains only the requested
    forward values.

    :param path: Path or batch of paths used in the forward call, with shape
        ``(..., length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param trees: Decorated trees or planar ordered forests requested in the
        forward call. See :func:`branched_sig_coef`.
    :type trees: tuple | None | list[tuple | None]
    :param coefs: Forward output from :func:`branched_sig_coef`.
    :type coefs: numpy.ndarray | torch.tensor
    :param derivs: Derivatives of a scalar objective with respect to ``coefs``.
    :type derivs: numpy.ndarray | torch.tensor
    :param time_aug: Whether time augmentation was used in the forward call.
    :type time_aug: bool
    :param lead_lag: Whether lead-lag was used in the forward call.
    :type lead_lag: bool
    :param end_time: End time used for time augmentation.
    :type end_time: float
    :param planar: Whether the planar MKW ordered-forest basis was used.
    :type planar: bool
    :param correction: Correction data passed to the forward call. It is treated
        as constant, so this function returns derivatives only for ``path``.
    :type correction: numpy.ndarray | torch.tensor | None
    :param n_jobs: Number of CPU threads. Ignored for CUDA input. Use 1 for
        serial CPU execution or -1 for all available CPU threads.
    :type n_jobs: int
    :return: Path derivatives with the same shape and container type as ``path``.
    :rtype: numpy.ndarray | torch.tensor

    Example:
    --------

    .. code-block:: python

        import numpy as np
        import pysiglib

        path = np.random.default_rng(0).normal(size=(100, 2))
        requested = [(0,), ((0,), 1)]
        pysiglib.prepare_branched_sig_coef(2, requested)
        coefs = pysiglib.branched_sig_coef(path, requested)
        grad = pysiglib.branched_sig_coef_backprop(
            path, requested, coefs, np.ones_like(coefs))

    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(planar, "planar", bool)
    check_n_jobs(n_jobs)

    path_data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    if path_data.lead_lag and path_data.data_length == 0:
        raise ValueError("lead_lag requires a path with at least one point")
    basis_elements, degree, tree_data = _branched_coef_data(
        trees, path_data.dimension, planar)
    coef_data = MultipleSigInputHandler(
        [coefs, derivs], len(basis_elements), ["coefs", "derivs"])
    if coef_data.type_ != path_data.type_:
        raise ValueError("coefs, derivs, and path must have the same array type")
    if coef_data.dtype != path_data.dtype:
        raise ValueError("coefs, derivs, and path must have the same dtype")
    if coef_data.device != path_data.device:
        raise ValueError("coefs, derivs, and path must be on the same device")
    if coef_data.batch_shape != path_data.batch_shape:
        raise ValueError("coefs and derivs must have the same batch shape as path")
    correction_data = CorrectionInputHandler(correction, path_data, degree)
    tree_data_tensor = torch.tensor(tree_data, dtype=torch.uint64)
    tree_data_ptr = cast(tree_data_tensor.data_ptr(), POINTER(c_uint64))
    result = PathOutputHandler(
        path_data.data_length, path_data.data_dimension, path_data)

    if path_data.batch_size == 0:
        return result.data

    if path_data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_COEF_BACKPROP[path_data.dtype](
            path_data.data_ptr, result.data_ptr, coef_data.data[0].data_ptr,
            coef_data.data[1].data_ptr, tree_data_ptr, len(tree_data),
            path_data.batch_size, path_data.data_dimension,
            path_data.data_length, degree, n_jobs, path_data.time_aug,
            path_data.lead_lag, path_data.end_time, planar,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride,
        )
    else:
        err_code = CUSIG_BRANCHED_SIG_COEF_BACKPROP[path_data.dtype](
            path_data.data_ptr, result.data_ptr, coef_data.data[0].data_ptr,
            coef_data.data[1].data_ptr, tree_data_ptr, len(tree_data),
            path_data.batch_size, path_data.data_dimension,
            path_data.data_length, degree, path_data.time_aug,
            path_data.lead_lag, path_data.end_time, planar,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride,
        )
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_coef_backprop: " + err_msg(err_code))
    return result.data
