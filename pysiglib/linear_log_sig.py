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

from .param_checks import check_type, check_type_multiple, check_dtype, check_non_neg, check_n_jobs, check_log_sig_method
from .sig_length import sig_length, log_sig_length


def _linear_log_sig_length(dimension, degree, method, scalar_term, n_jobs):
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(method, "method", int)
    check_log_sig_method(method)
    check_type(scalar_term, "scalar_term", bool)
    check_n_jobs(n_jobs)
    if degree == 0 or dimension == 0:
        return int(method == 0 and scalar_term)
    return sig_length(dimension, degree, scalar_term=scalar_term) if method == 0 else log_sig_length(dimension, degree)


def linear_log_sig(displacement, dimension: int, degree: int, *,
                   method: int = 1, scalar_term: bool = False, n_jobs: int = 1):
    """Compute the log signature of the linear path from zero to ``displacement``.

    The degree-1 coordinates equal the displacement. All higher coordinates
    are zero. No preparation call is required.

    :param displacement: NumPy array or Torch tensor of shape ``(..., dimension)``.
    :param dimension: Number of channels.
    :param degree: Truncation degree.
    :param method: Log signature format, as in :func:`log_sig` (0, 1, 2, or 3).
    :param scalar_term: Include a leading zero for method 0. Ignored for other methods.
    :param n_jobs: Number of CPU threads. Use -1 for all threads. Ignored on CUDA.
    :return: Log signature with the input batch shape, type, dtype, and device.
    """
    length = _linear_log_sig_length(dimension, degree, method, scalar_term, n_jobs)
    check_type_multiple(displacement, "displacement", (np.ndarray, torch.Tensor))
    check_dtype(displacement, "displacement")
    if displacement.ndim < 1 or displacement.shape[-1] != dimension:
        raise ValueError("displacement must have shape (..., dimension)")
    offset = int(method == 0 and scalar_term)
    width = dimension if degree else 0
    if isinstance(displacement, torch.Tensor):
        prefix = displacement.new_zeros((*displacement.shape[:-1], offset))
        suffix = displacement.new_zeros((*displacement.shape[:-1], length - offset - width))
        return torch.cat([prefix, displacement[..., :width], suffix], dim=-1)
    prefix = np.zeros((*displacement.shape[:-1], offset), dtype=displacement.dtype)
    suffix = np.zeros((*displacement.shape[:-1], length - offset - width), dtype=displacement.dtype)
    return np.concatenate([prefix, displacement[..., :width], suffix], axis=-1)


def linear_log_sig_backprop(derivs, displacement, dimension: int, degree: int, *,
                            method: int = 1, scalar_term: bool = False, n_jobs: int = 1):
    """Backpropagate through :func:`linear_log_sig`.

    :param derivs: Output derivatives, with the same shape and format as the forward output.
    :param displacement: Displacement used in the forward call.
    :param dimension: Number of channels.
    :param degree: Truncation degree.
    :param method: Method used in the forward call.
    :param scalar_term: Scalar-term format used in the forward call.
    :param n_jobs: Number of CPU threads.
    :return: Derivatives with respect to ``displacement``.
    """
    length = _linear_log_sig_length(dimension, degree, method, scalar_term, n_jobs)
    check_type_multiple(displacement, "displacement", (np.ndarray, torch.Tensor))
    check_type_multiple(derivs, "derivs", (np.ndarray, torch.Tensor))
    check_dtype(displacement, "displacement")
    check_dtype(derivs, "derivs")
    if displacement.ndim < 1 or displacement.shape[-1] != dimension:
        raise ValueError("displacement must have shape (..., dimension)")
    if derivs.shape != (*displacement.shape[:-1], length):
        raise ValueError("derivs must have the same shape as the forward output")
    if isinstance(displacement, torch.Tensor) != isinstance(derivs, torch.Tensor) or displacement.dtype != derivs.dtype:
        raise ValueError("displacement and derivs must have the same type and dtype")
    offset = int(method == 0 and scalar_term)
    if isinstance(displacement, torch.Tensor):
        if displacement.device != derivs.device:
            raise ValueError("displacement and derivs must be on the same device")
        return derivs[..., offset:offset + dimension].clone() if degree else torch.zeros_like(displacement)
    return derivs[..., offset:offset + dimension].copy() if degree else np.zeros_like(displacement)
