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

from .data_handlers import CorrectionInputHandler, PathInputHandler, SigInputHandler, SigOutputHandler
from .dtypes import CPSIG_BRANCHED_SIG_COEF, CUSIG_BRANCHED_SIG_COEF
from .error_codes import err_msg
from .load_siglib import BUILT_WITH_CUDA, CPSIG, CUSIG
from .param_checks import check_n_jobs, check_non_neg, check_type
from .sig_length import aug_dim
from .trees import _as_planar_forest_tuple, tree_to_idx


def _encode_branched_tree(tree, dimension, planar):
    if not isinstance(tree, tuple) or len(tree) == 0:
        raise ValueError("invalid decorated tree")
    label = tree[-1]
    if not isinstance(label, (int, np.integer)) or label < 0 or label >= dimension:
        raise ValueError("tree label is outside the path dimension")

    children = [
        _encode_branched_tree(child, dimension, planar)
        for child in tree[:-1]
    ]
    if not planar:
        children.sort(key=lambda child: child[1])
    nodes = 1 + sum(child[0] for child in children)
    key = (nodes, int(label), tuple(child[1] for child in children))
    data = [int(label), len(children)]
    for child in children:
        data.extend(child[2])
    return nodes, key, data


def _branched_coef_data(basis_elements, dimension, planar):
    basis_elements = basis_elements if isinstance(basis_elements, list) else [basis_elements]
    if len(basis_elements) == 0:
        raise ValueError("trees must be a non-empty list of basis elements")

    tree_data = [len(basis_elements)]
    orders = []
    try:
        for basis_element in basis_elements:
            if basis_element is None:
                roots = ()
            elif planar:
                roots = _as_planar_forest_tuple(basis_element)
            else:
                roots = (basis_element,)
            encoded_roots = [
                _encode_branched_tree(root, dimension, planar)
                for root in roots
            ]
            tree_data.append(len(encoded_roots))
            orders.append(sum(root[0] for root in encoded_roots))
            for root in encoded_roots:
                tree_data.extend(root[2])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("trees contains an invalid decorated tree or ordered forest") from exc

    return basis_elements, max(orders), tree_data


def extract_branched_sig_coef(
        bsig: Union[np.ndarray, torch.Tensor],
        trees,
        dimension: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        planar: bool = False,
        scalar_term: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extracts coefficients from a branched signature or batch of branched
    signatures.

    :param bsig: Branched signature or batch of branched signatures, with shape
        ``(..., branched_sig_length)``.
    :type bsig: numpy.ndarray | torch.Tensor
    :param trees: A decorated rooted tree, or a list of decorated rooted trees.
        With ``planar=True``, each requested basis element is an ordered forest;
        a bare planar tree is accepted as shorthand for a one-tree forest. See
        :func:`tree_to_idx` for the tuple convention.
    :type trees: tuple | None | list[tuple | None]
    :param dimension: Dimension of the underlying path.
    :type dimension: int
    :param time_aug: Whether the branched signatures were computed with
        ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the branched signatures were computed with
        ``lead_lag=True``.
    :type lead_lag: bool
    :param planar: Whether the branched signatures use the planar MKW
        ordered-forest basis.
    :type planar: bool
    :param scalar_term: Whether ``bsig`` includes the leading scalar term.
        This must match the format used to compute ``bsig``.
    :type scalar_term: bool
    :return: Branched-signature coefficients with shape ``(..., num_trees)``.
    :rtype: numpy.ndarray | torch.Tensor

    Example:
    --------

    .. code-block:: python

        import numpy as np
        import pysiglib

        path = np.random.default_rng(0).normal(size=(100, 2))
        pysiglib.prepare_branched_sig(2, 3)
        bsig = pysiglib.branched_sig(path, 3)
        requested = [(0,), ((0,), 1), ((0,), (1,), 0)]
        coefs = pysiglib.extract_branched_sig_coef(
            bsig, requested, dimension=2
        )

    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(planar, "planar", bool)
    check_type(scalar_term, "scalar_term", bool)

    augmented_dimension = aug_dim(dimension, time_aug, lead_lag)
    basis_elements, degree, _ = _branched_coef_data(
        trees, augmented_dimension, planar)

    sig_length = bsig.shape[-1]
    SigInputHandler(bsig, sig_length, "bsig")

    indices = [
        tree_to_idx(
            basis_element,
            augmented_dimension,
            degree,
            planar=planar,
            scalar_term=scalar_term,
        )
        for basis_element in basis_elements
    ]
    return bsig[..., indices]


def prepare_branched_sig_coef(
        dimension: int,
        trees,
        *,
        use_disk: bool = False,
        time_aug: bool = False,
        lead_lag: bool = False,
        planar: bool = False,
        device: str = "both",
):
    """
    Precomputes the data needed for selected branched-signature coefficients.
    Must be called before :func:`branched_sig_coef` for a given dimension and
    list of trees.

    The degree is inferred from ``trees``. The same trees, dimension, and
    augmentation options must be passed to :func:`branched_sig_coef` and its
    backpropagation.

    :param dimension: Dimension of the underlying path.
    :type dimension: int
    :param trees: A decorated rooted tree, or a list of decorated rooted trees.
        With ``planar=True``, each requested basis element is an ordered forest.
    :type trees: tuple | None | list[tuple | None]
    :param use_disk: If ``False``, will cache prepared objects in memory only.
        If ``True``, will also save these objects in a shared disk cache to be
        re-used for future runs. The CPU and GPU libraries share the same
        disk cache format and directory.
        See additionally the documentation for ``pysiglib.set_cache_dir``.
    :type use_disk: bool
    :param time_aug: Whether to prepare for time-augmented paths.
    :type time_aug: bool
    :param lead_lag: Whether to prepare for lead-lag transformed paths.
    :type lead_lag: bool
    :param planar: Whether to use the planar MKW ordered-forest basis.
    :type planar: bool
    :param device: Which device caches to prepare. Must be ``"cpu"``,
        ``"cuda"``, or ``"both"``.
    :type device: str
    """
    check_type(dimension, "dimension", int)
    check_type(use_disk, "use_disk", bool)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(planar, "planar", bool)
    check_type(device, "device", str)
    check_non_neg(dimension, "dimension")
    if device not in ("cpu", "cuda", "both"):
        raise ValueError("device must be 'cpu', 'cuda', or 'both'")

    augmented_dimension = aug_dim(dimension, time_aug, lead_lag)
    _, degree, tree_data = _branched_coef_data(
        trees, augmented_dimension, planar)
    tree_data_tensor = torch.tensor(tree_data, dtype=torch.uint64)
    tree_data_ptr = cast(tree_data_tensor.data_ptr(), POINTER(c_uint64))
    if device in ("cpu", "both"):
        err_code = CPSIG.prepare_branched_sig_coef(
            tree_data_ptr, len(tree_data), dimension, augmented_dimension,
            degree, planar, use_disk,
        )
        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_branched_sig_coef: "
                + err_msg(err_code, "cpu")
            )

    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.prepare_branched_sig_coef_cuda(
            tree_data_ptr, len(tree_data), dimension, augmented_dimension,
            degree, planar, use_disk,
        )
        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_branched_sig_coef (CUDA): "
                + err_msg(err_code, "cuda")
            )


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
    Computes selected branched-signature coefficients.

    The truncation degree is inferred as the largest number of nodes in the
    requested trees or forests. Call :func:`prepare_branched_sig_coef` with the
    same trees and options before first use.

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
    :param n_jobs: Number of CPU threads. Ignored for CUDA input. Use 1 for
        serial CPU execution or -1 for all available CPU threads.
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
        pysiglib.prepare_branched_sig_coef(2, requested)
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
    basis_elements, degree, tree_data = _branched_coef_data(
        trees, data.dimension, planar)
    correction_data = CorrectionInputHandler(correction, data, degree)
    tree_data_tensor = torch.tensor(tree_data, dtype=torch.uint64)
    tree_data_ptr = cast(tree_data_tensor.data_ptr(), POINTER(c_uint64))
    result = SigOutputHandler(data, len(basis_elements))

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_COEF[data.dtype](
            data.data_ptr, result.data_ptr, tree_data_ptr, len(tree_data),
            data.batch_size, data.data_dimension, data.data_length, degree,
            n_jobs, data.time_aug, data.lead_lag, data.end_time, planar,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride,
        )
    else:
        err_code = CUSIG_BRANCHED_SIG_COEF[data.dtype](
            data.data_ptr, result.data_ptr, tree_data_ptr, len(tree_data),
            data.batch_size, data.data_dimension, data.data_length, degree,
            data.time_aug, data.lead_lag, data.end_time, planar,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride,
        )
    if err_code:
        raise Exception(
            "Error in pysiglib.branched_sig_coef: "
            + err_msg(err_code, result.device))
    return result.data
