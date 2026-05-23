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
from .dtypes import CPSIG_BRANCHED_SIG_TO_LOG_SIG, CUSIG_BRANCHED_SIG_TO_LOG_SIG_CUDA
from .data_handlers import MultipleSigInputHandler, SigOutputHandler
from .sig_length import aug_dim
from .branched_sig import (
    _infer_branched_scalar_term,
    _inv_permute_bsig,
    _permute_bsig,
    branched_sig,
    branched_sig_length,
)


def branched_sig_to_log_sig(
        bsig: Union[np.ndarray, torch.Tensor],
        dimension: int,
        degree: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        tree_order: str = "recursive",
        planar: bool = False,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the branched log signature from the branched signature.

    :param bsig: The branched signature or batch of branched signatures, given as a `numpy.ndarray`
        or `torch.tensor`. For a single branched signature, this must be of shape
        ``branched_sig_length``. For a batch of paths, this must be of shape
        ``(batch_size, branched_sig_length)``. The leading scalar term may be present or omitted.
    :type bsig: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the branched signature(s).
    :type degree: int
    :param time_aug: Whether the branched signature(s) were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the branched signature(s) were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param tree_order: Tree ordering convention for the input and output.
        ``"recursive"`` (default) uses the recursive construction order.
        ``"canonical"`` uses the shape-first order matching :func:`tree_to_idx`.
    :type tree_order: str
    :param planar: If True, use planar branched signatures.
    :type planar: bool
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: The branched log signature or batch of branched log signatures, in the same
        scalar-term format as ``bsig``.
    :rtype: numpy.ndarray | torch.tensor

    Example usage:
    ----------------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        bsig = pysiglib.branched_sig(path, 3)
        blogsig = pysiglib.branched_sig_to_log_sig(bsig, 5, 3)
        print(blogsig)
    """
    if tree_order not in ("recursive", "canonical"):
        raise ValueError(f"tree_order must be 'recursive' or 'canonical', got {tree_order!r}")
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    scalar_term = _infer_branched_scalar_term(bsig, aug_dimension, degree, planar=planar)
    if tree_order != "recursive":
        bsig = _inv_permute_bsig(bsig, aug_dimension, degree, planar=planar, scalar_term=scalar_term)

    bsig_len = branched_sig_length(aug_dimension, degree, planar=planar, scalar_term=scalar_term)
    data = MultipleSigInputHandler([bsig], bsig_len, ["bsig"])
    result = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_TO_LOG_SIG[data.dtype](
            data.sig_ptr[0], result.data_ptr, data.batch_size,
            aug_dimension, degree, n_jobs, planar, scalar_term)
    else:
        err_code = CUSIG_BRANCHED_SIG_TO_LOG_SIG_CUDA[data.dtype](
            data.sig_ptr[0], result.data_ptr, data.batch_size,
            aug_dimension, degree, planar, scalar_term)
    if err_code:
        raise Exception("Error in pysiglib.branched_sig_to_log_sig: " + err_msg(err_code))

    if tree_order != "recursive":
        _permute_bsig(result.data, aug_dimension, degree, planar=planar, scalar_term=scalar_term)
    return result.data


def branched_log_sig(
        path: Union[np.ndarray, torch.Tensor],
        degree: int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False,
        end_time: float = 1.0,
        tree_order: str = "recursive",
        planar: bool = False,
        scalar_term: bool = False,
        correction = None,
        n_jobs: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the branched log signature of a path.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths,
        this must be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param degree: Truncation degree of the branched (log) signature(s).
    :type degree: int
    :param time_aug: If set to True, will compute the branched log signature of the
        time-augmented path, :math:`\\hat{x}_t := (t, x_t)`, defined as the original path with
        an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math:`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the branched log signature of the path after
        applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param tree_order: Tree ordering convention for the output coefficients.
        ``"recursive"`` (default) uses the recursive construction order.
        ``"canonical"`` uses the shape-first order matching :func:`tree_to_idx`.
    :type tree_order: str
    :param planar: If True, compute the planar branched log signature.
    :type planar: bool
    :param scalar_term: If True, include the leading scalar coefficient, which is zero.
    :type scalar_term: bool
    :param correction: Optional per-segment correction of level
        :math:`\\geq 2` added to the path increment on each path
        segment, before the branched log signature is taken. The
        level-1 part of the local lift is the segment's path increment
        :math:`\\Delta x`, the higher levels come from the matching correction row,
        and the local branched signature on each segment is

        .. math::

            \\exp_* \\left( \\sum_i \\Delta x_i\\, e_i + \\sum_{k=2}^{m} \\sum_{i_1, \\ldots, i_k} c^{(k)}_{i_1 \\ldots i_k}\\, e_{i_1 \\cdots i_k} \\right),

        where :math:`e_w` is the chain (root-to-leaf path) tree with labels
        :math:`w` and :math:`\\exp_*` is the Hopf-algebra exponential under the
        Butcher product. ``correction`` must have shape
        ``path.shape[:-2] + (path.shape[-2] - 1, L)``, with one row per
        path segment. ``L = d^2 + d^3 + ... + d^m``, where :math:`d` is the
        underlying path dimension and :math:`2 \\leq m \\leq N` is the highest
        correction level supplied (missing higher levels are zero). Levels are
        concatenated in order, and within level :math:`k` the entry for chain
        :math:`(i_1, \\ldots, i_k)` lives at flat index
        :math:`i_1 d^{k-1} + i_2 d^{k-2} + \\cdots + i_k`. Passing ``None``
        (default) or an empty array is equivalent to all-zero correction. Indices
        are over the original path channels; with ``time_aug=True``, the
        appended time channel contributes no correction. Cannot be combined with
        ``lead_lag=True``.
    :type correction: numpy.ndarray | torch.tensor | None
    :param n_jobs: Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: The branched log signature or batch of branched log signatures.
    :rtype: numpy.ndarray | torch.tensor

    Example usage:
    ----------------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        blogsig = pysiglib.branched_log_sig(path, 3)
        print(blogsig)

    Ito-lifted branched log signature of a sampled Brownian path. For Brownian
    motion with instantaneous covariance :math:`\\Sigma`, setting the level-2
    correction to :math:`c^{(2)}_{ij} = \\Sigma_{ij}\\,\\Delta t` per segment
    gives the Ito correction.

    .. code-block:: python

        import numpy as np
        import pysiglib

        d, N, T = 2, 3, 1.0
        n_steps = 100
        dt = T / n_steps
        rng = np.random.default_rng(42)

        # 2D standard Brownian motion sample (Sigma = I)
        path = np.zeros((n_steps + 1, d))
        path[1:] = np.cumsum(rng.normal(0, np.sqrt(dt), (n_steps, d)), axis=0)

        # Ito level-2 correction: one dt * Sigma row per path segment.
        correction = np.broadcast_to(
            (np.eye(d) * dt).reshape(1, -1), (n_steps, d * d)).copy()

        pysiglib.prepare_branched_sig(d, N)
        ito_blogsig = pysiglib.branched_log_sig(
            path, N, correction=correction, end_time=T)
        print(ito_blogsig)
    """
    bsig = branched_sig(
        path, degree, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        tree_order=tree_order, planar=planar, scalar_term=scalar_term,
        correction=correction, n_jobs=n_jobs)
    dimension = path.shape[-1]
    return branched_sig_to_log_sig(
        bsig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag,
        tree_order=tree_order, planar=planar, n_jobs=n_jobs)
