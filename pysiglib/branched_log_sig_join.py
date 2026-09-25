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

import numpy as np
import torch

from .param_checks import check_type, check_type_multiple, check_non_neg, check_n_jobs
from .data_handlers import SigInputHandler, SigOutputHandler
from .branched_log_sig import branched_log_sig_length
from .dtypes import (
    CPSIG_BRANCHED_LOG_SIG_JOIN, CUSIG_BRANCHED_LOG_SIG_JOIN,
    CPSIG_BRANCHED_LOG_SIG_JOIN_BACKPROP, CUSIG_BRANCHED_LOG_SIG_JOIN_BACKPROP,
)
from .error_codes import err_msg


def _join_inputs(blogsig, displacement, dimension, degree, prepend, n_jobs, derivs=None):
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(prepend, "prepend", bool)
    check_n_jobs(n_jobs)
    length = branched_log_sig_length(dimension, degree, planar=True)
    inputs = [(blogsig, "blogsig", length), (displacement, "displacement", dimension)]
    if derivs is not None:
        inputs.append((derivs, "derivs", length))
    data = []
    for value, name, width in inputs:
        check_type_multiple(value, name, (np.ndarray, torch.Tensor))
        if value.ndim < 1 or value.shape[-1] != width:
            raise ValueError(name + " must have shape (..., " + str(width) + ")")
        item = SigInputHandler(value, width, name)
        if isinstance(value, torch.Tensor):
            item.device = str(value.device)
        data.append(item)
    for item in data[1:]:
        for field in ("type_", "dtype", "device", "batch_shape"):
            if getattr(item, field) != getattr(data[0], field):
                raise ValueError("join inputs must have the same " + field)
    if data[0].device != "cpu" and degree > 20:
        raise NotImplementedError("CUDA MKW BCH method supports degree at most 20")
    return data


def branched_log_sig_join(blogsig, displacement, dimension: int, degree: int, *,
                          prepend: bool = False, n_jobs: int = 1):
    """Append or prepend a linear segment to a planar branched log signature.

    Accepts scalar-free planar method-2/3 coordinates only. Call
    ``prepare_branched_log_sig(dimension, degree, 3, planar=True)`` first.
    The segment is added through the BCH formula in the same coordinates.

    :param blogsig: Planar method-2/3 log coordinates of shape ``(..., branched_log_sig_length)``.
    :param displacement: Segment displacement of shape ``(..., dimension)``.
        Type, dtype, device, and batch shape must match ``blogsig``.
    :param dimension: Number of channels, including any channels already added by path transforms.
    :param degree: Maximum number of nodes. CUDA supports degree at most 20.
    :param prepend: Prepend the segment if True. Otherwise append it.
    :param n_jobs: Number of CPU threads. Use -1 for all threads. Ignored on CUDA.
    :return: Updated scalar-free planar method-2/3 coordinates.
    """
    logsig_data, disp_data = _join_inputs(blogsig, displacement, dimension, degree, prepend, n_jobs)
    result = SigOutputHandler(logsig_data, logsig_data.sig.shape[-1])
    if logsig_data.batch_size == 0:
        return result.data
    args = (logsig_data.data_ptr, disp_data.data_ptr, result.data_ptr,
            logsig_data.batch_size, dimension, degree, prepend)
    if logsig_data.device == "cpu":
        err_code = CPSIG_BRANCHED_LOG_SIG_JOIN[logsig_data.dtype](*args, n_jobs)
    else:
        err_code = CUSIG_BRANCHED_LOG_SIG_JOIN[logsig_data.dtype](*args)
    if err_code:
        raise Exception("Error in pysiglib.branched_log_sig_join: " + err_msg(err_code, result.device))
    return result.data


def branched_log_sig_join_backprop(derivs, blogsig, displacement, dimension: int, degree: int, *,
                                   prepend: bool = False, n_jobs: int = 1):
    """Backpropagate through :func:`branched_log_sig_join`.

    Requires the same method-3 preparation as the forward call.

    :param derivs: Output derivatives, with the same shape as ``blogsig``.
    :param blogsig: Scalar-free planar method-2/3 coordinates used in the forward call.
    :param displacement: Displacement used in the forward call.
    :param dimension: Number of channels.
    :param degree: Maximum number of nodes.
    :param prepend: Segment order used in the forward call.
    :param n_jobs: Number of CPU threads.
    :return: Pair of derivatives with respect to ``blogsig`` and ``displacement``.
    """
    logsig_data, disp_data, derivs_data = _join_inputs(blogsig, displacement, dimension, degree, prepend, n_jobs, derivs)
    d_logsig = SigOutputHandler(logsig_data, logsig_data.sig.shape[-1])
    d_disp = SigOutputHandler(disp_data, dimension)
    if logsig_data.batch_size == 0:
        return d_logsig.data, d_disp.data
    args = (derivs_data.data_ptr, d_logsig.data_ptr, d_disp.data_ptr,
            logsig_data.data_ptr, disp_data.data_ptr, logsig_data.batch_size,
            dimension, degree, prepend)
    if logsig_data.device == "cpu":
        err_code = CPSIG_BRANCHED_LOG_SIG_JOIN_BACKPROP[logsig_data.dtype](*args, n_jobs)
    else:
        err_code = CUSIG_BRANCHED_LOG_SIG_JOIN_BACKPROP[logsig_data.dtype](*args)
    if err_code:
        raise Exception("Error in pysiglib.branched_log_sig_join_backprop: " + err_msg(err_code, d_logsig.device))
    return d_logsig.data, d_disp.data
