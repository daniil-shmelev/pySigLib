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

from typing import Optional

from .data_handlers import displacement_to_path
from .param_checks import check_type, check_non_neg, check_n_jobs
from .branched_log_sig import branched_log_sig, _resolve_branched_log_sig_method
from .branched_log_sig_backprop import (
    _branched_log_sig_from_path_backprop, branched_sig_to_log_sig_backprop,
)
from .branched_sig import branched_sig
from .branched_sig_backprop import branched_sig_backprop


def linear_branched_log_sig(displacement, dimension: int, degree: int, *,
                            planar: bool = False, scalar_term: bool = False,
                            method: Optional[int] = None, n_jobs: int = 1):
    """Compute the branched log signature of a linear segment from zero to ``displacement``.

    Call ``prepare_branched_log_sig(dimension, degree, method, planar=planar)`` first.

    :param displacement: NumPy array or Torch tensor of shape ``(..., dimension)``.
    :param dimension: Number of channels.
    :param degree: Maximum number of nodes.
    :param planar: Use planar branched log signatures if True.
    :param scalar_term: Include a leading zero for method 0. Ignored for other methods.
    :param method: Format used by :func:`branched_log_sig`. Defaults to method 0 for
        nonplanar signatures and method 1 for planar signatures. Methods 1, 2, and 3
        require ``planar=True``.
    :param n_jobs: Number of CPU threads. Use -1 for all threads. Ignored on CUDA.
    :return: Branched log signature with the input batch shape, type, dtype, and device.
    """
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)
    path = displacement_to_path(displacement, dimension)
    return branched_log_sig(path, degree, planar=planar, scalar_term=scalar_term, method=method, n_jobs=n_jobs)


def linear_branched_log_sig_backprop(derivs, displacement, dimension: int, degree: int, *,
                                     planar: bool = False, scalar_term: bool = False,
                                     method: Optional[int] = None, n_jobs: int = 1):
    """Backpropagate through :func:`linear_branched_log_sig`.

    :param derivs: Output derivatives, with the same shape and format as the forward output.
    :param displacement: Displacement used in the forward call.
    :param dimension: Number of channels.
    :param degree: Maximum number of nodes.
    :param planar: Planarity used in the forward call.
    :param scalar_term: Scalar-term format used in the forward call.
    :param method: Method used in the forward call.
    :param n_jobs: Number of CPU threads.
    :return: Derivatives with respect to ``displacement``.
    """
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)
    method = _resolve_branched_log_sig_method(method, planar)
    path = displacement_to_path(displacement, dimension)
    if method == 3:
        return _branched_log_sig_from_path_backprop(derivs, path, degree, n_jobs=n_jobs)[..., 1, :]
    signature = branched_sig(path, degree, planar=planar, scalar_term=scalar_term, n_jobs=n_jobs)
    d_sig = branched_sig_to_log_sig_backprop(signature, derivs, dimension, degree, planar=planar, method=method, n_jobs=n_jobs)
    return branched_sig_backprop(path, signature, d_sig, degree, planar=planar, n_jobs=n_jobs)[..., 1, :]
