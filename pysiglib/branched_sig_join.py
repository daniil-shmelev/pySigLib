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

from .param_checks import check_type, check_type_multiple
from .branched_sig import branched_sig_combine, _infer_branched_scalar_term
from .branched_sig_backprop import branched_sig_combine_backprop
from .linear_branched_sig import linear_branched_sig, linear_branched_sig_backprop


def _branched_join_scalar_term(bsig, displacement, dimension, degree, planar):
    check_type_multiple(bsig, "bsig", (np.ndarray, torch.Tensor))
    check_type_multiple(displacement, "displacement", (np.ndarray, torch.Tensor))
    if bsig.ndim < 1 or displacement.ndim < 1:
        raise ValueError("bsig and displacement must have at least rank 1")
    if isinstance(bsig, torch.Tensor) != isinstance(displacement, torch.Tensor) or bsig.dtype != displacement.dtype:
        raise ValueError("bsig and displacement must have the same type and dtype")
    if bsig.shape[:-1] != displacement.shape[:-1]:
        raise ValueError("bsig and displacement must have the same batch shape")
    if isinstance(bsig, torch.Tensor) and bsig.device != displacement.device:
        raise ValueError("bsig and displacement must be on the same device")
    return _infer_branched_scalar_term(bsig, dimension, degree, planar=planar)


def branched_sig_join(bsig, displacement, dimension: int, degree: int, *,
                      planar: bool = False, prepend: bool = False, n_jobs: int = 1):
    """Append or prepend a linear segment to a branched signature.

    Call ``prepare_branched_sig(dimension, degree, planar=planar)`` first.
    The output preserves the scalar-term format of ``bsig``.

    :param bsig: Branched signature of shape ``(..., branched_sig_length)``.
    :param displacement: Segment displacement of shape ``(..., dimension)``.
        Type, dtype, device, and batch shape must match ``bsig``.
    :param dimension: Number of channels, including any channels already added by path transforms.
    :param degree: Maximum number of nodes.
    :param planar: Use planar branched signatures if True.
    :param prepend: Prepend the segment if True. Otherwise append it.
    :param n_jobs: Number of CPU threads. Use -1 for all threads. Ignored on CUDA.
    :return: Updated branched signature.
    """
    check_type(prepend, "prepend", bool)
    scalar_term = _branched_join_scalar_term(bsig, displacement, dimension, degree, planar)
    segment = linear_branched_sig(displacement, dimension, degree, planar=planar, scalar_term=scalar_term, n_jobs=n_jobs)
    left, right = (segment, bsig) if prepend else (bsig, segment)
    return branched_sig_combine(left, right, dimension, degree, planar=planar, n_jobs=n_jobs)


def branched_sig_join_backprop(derivs, bsig, displacement, dimension: int, degree: int, *,
                               planar: bool = False, prepend: bool = False, n_jobs: int = 1):
    """Backpropagate through :func:`branched_sig_join`.

    :param derivs: Output derivatives, with the same shape as ``bsig``.
    :param bsig: Branched signature used in the forward call.
    :param displacement: Displacement used in the forward call.
    :param dimension: Number of channels.
    :param degree: Maximum number of nodes.
    :param planar: Planarity used in the forward call.
    :param prepend: Segment order used in the forward call.
    :param n_jobs: Number of CPU threads.
    :return: Pair of derivatives with respect to ``bsig`` and ``displacement``.
    """
    check_type(prepend, "prepend", bool)
    scalar_term = _branched_join_scalar_term(bsig, displacement, dimension, degree, planar)
    segment = linear_branched_sig(displacement, dimension, degree, planar=planar, scalar_term=scalar_term, n_jobs=n_jobs)
    left, right = (segment, bsig) if prepend else (bsig, segment)
    d_left, d_right = branched_sig_combine_backprop(derivs, left, right, dimension, degree, planar=planar, n_jobs=n_jobs)
    d_bsig, d_segment = (d_right, d_left) if prepend else (d_left, d_right)
    d_displacement = linear_branched_sig_backprop(d_segment, displacement, dimension, degree, planar=planar, scalar_term=scalar_term, n_jobs=n_jobs)
    return d_bsig, d_displacement
