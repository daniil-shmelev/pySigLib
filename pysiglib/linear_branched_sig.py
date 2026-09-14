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

from .data_handlers import displacement_to_path
from .branched_sig import branched_sig
from .branched_sig_backprop import branched_sig_backprop


def linear_branched_sig(displacement, dimension: int, degree: int, *,
                        planar: bool = False, scalar_term: bool = False, n_jobs: int = 1):
    """Compute the branched signature of a linear segment from zero to ``displacement``.

    Call ``prepare_branched_sig(dimension, degree, planar=planar)`` first.

    :param displacement: NumPy array or Torch tensor of shape ``(..., dimension)``.
    :param dimension: Number of channels.
    :param degree: Maximum number of nodes.
    :param planar: Use planar branched signatures if True.
    :param scalar_term: Include the leading scalar coefficient, equal to one.
    :param n_jobs: Number of CPU threads. Use -1 for all threads. Ignored on CUDA.
    :return: Branched signature with the input batch shape, type, dtype, and device.
    """
    path = displacement_to_path(displacement, dimension)
    return branched_sig(path, degree, planar=planar, scalar_term=scalar_term, n_jobs=n_jobs)


def linear_branched_sig_backprop(derivs, displacement, dimension: int, degree: int, *,
                                 planar: bool = False, scalar_term: bool = False, n_jobs: int = 1):
    """Backpropagate through :func:`linear_branched_sig`.

    :param derivs: Output derivatives, with the same shape as the forward output.
    :param displacement: Displacement used in the forward call.
    :param dimension: Number of channels.
    :param degree: Maximum number of nodes.
    :param planar: Planarity used in the forward call.
    :param scalar_term: Scalar-term format used in the forward call.
    :param n_jobs: Number of CPU threads.
    :return: Derivatives with respect to ``displacement``.
    """
    path = displacement_to_path(displacement, dimension)
    signature = branched_sig(path, degree, planar=planar, scalar_term=scalar_term, n_jobs=n_jobs)
    return branched_sig_backprop(path, signature, derivs, degree, planar=planar, n_jobs=n_jobs)[..., 1, :]
