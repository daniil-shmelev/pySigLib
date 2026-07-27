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

from .data_handlers import CorrectionInputHandler, PathInputHandler, SigOutputHandler
from .dtypes import CPSIG_BRANCHED_SIG_COEF
from .error_codes import err_msg
from .param_checks import check_n_jobs, check_type
from .trees import _as_planar_forest_tuple, _basis_element_order, tree_to_idx


def _branched_coef_indices(basis_elements, dimension, planar):
    basis_elements = basis_elements if isinstance(basis_elements, list) else [basis_elements]
    if len(basis_elements) == 0:
        raise ValueError("trees must be a non-empty list of basis elements")

    try:
        orders = [
            0 if basis_element is None else _basis_element_order(
                _as_planar_forest_tuple(basis_element) if planar else basis_element,
                planar,
            )
            for basis_element in basis_elements
        ]
    except (IndexError, TypeError) as exc:
        raise ValueError("trees contains an invalid decorated tree or ordered forest") from exc

    degree = max(orders)
    indices = [
        tree_to_idx(
            basis_element,
            dimension,
            degree,
            planar=planar,
            scalar_term=True,
        )
        for basis_element in basis_elements
    ]
    return basis_elements, degree, indices


def branched_sig_coef(
        path: Union[np.ndarray, torch.Tensor],
        trees,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        planar: bool = False,
        correction=None,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes selected branched-signature coefficients on CPU.

    The truncation degree is inferred as the largest number of nodes in the
    requested trees or forests. Call :func:`prepare_branched_sig` for that
    degree and the augmented path dimension before first use. The empty basis
    element ``None`` is accepted and has constant coefficient 1.

    :param path: Path or batch of paths, with shape ``(..., length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param trees: A decorated rooted tree, or a list of decorated rooted trees.
        With ``planar=True``, each requested basis element is an ordered forest;
        a bare planar tree is accepted as shorthand for a one-tree forest. See
        :func:`tree_to_idx` for the tuple convention.
    :type trees: tuple | None | list[tuple | None]
    :param time_aug: Whether to prepend a time channel to the path.
    :type time_aug: bool
    :param lead_lag: Whether to apply the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time augmentation.
    :type end_time: float
    :param planar: Whether to use the planar MKW ordered-forest basis.
    :type planar: bool
    :param correction: Optional segment correction with the layout described by
        :func:`branched_sig`. It is supported with time augmentation but not
        with lead-lag.
    :type correction: numpy.ndarray | torch.tensor | None
    :param n_jobs: Number of CPU threads. Use 1 for serial execution or -1 for
        all available threads.
    :type n_jobs: int
    :return: Requested coefficients, with shape ``(..., num_trees)``.
    :rtype: numpy.ndarray | torch.tensor

    Example:
    --------

    .. code-block:: python

        import numpy as np
        import pysiglib

        path = np.random.default_rng(0).normal(size=(100, 2))
        requested = [(0,), ((0,), 1), ((0,), (1,), 0)]
        pysiglib.prepare_branched_sig(2, 3)
        coefs = pysiglib.branched_sig_coef(path, requested)

    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(planar, "planar", bool)
    check_n_jobs(n_jobs)

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    if data.lead_lag and data.data_length == 0:
        raise ValueError("lead_lag requires a path with at least one point")
    if data.device != "cpu":
        raise ValueError("branched_sig_coef is currently CPU-only")

    basis_elements, degree, indices = _branched_coef_indices(trees, data.dimension, planar)
    correction_data = CorrectionInputHandler(correction, data, degree)
    indices_tensor = torch.tensor(indices, dtype=torch.uint64)
    indices_ptr = cast(indices_tensor.data_ptr(), POINTER(c_uint64))
    result = SigOutputHandler(data, len(basis_elements))

    if data.batch_size == 0:
        return result.data

    err_code = CPSIG_BRANCHED_SIG_COEF[data.dtype](
        data.data_ptr,
        result.data_ptr,
        indices_ptr,
        len(indices),
        data.batch_size,
        data.data_dimension,
        data.data_length,
        degree,
        n_jobs,
        data.time_aug,
        data.lead_lag,
        data.end_time,
        planar,
        correction_data.data_ptr,
        correction_data.length,
        correction_data.batch_stride,
        correction_data.segment_stride,
    )
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_coef: " + err_msg(err_code))
    return result.data
