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

from __future__ import annotations
from ._array import NativeArrayT as Array

from .param_checks import check_type, check_non_neg, check_n_jobs
from .error_codes import err_msg
from ..sig_length import aug_dim
from .dtypes import (CPSIG_BRANCHED_SIG, CPSIG_BRANCHED_SIG_COMBINE,
                     CUSIG_BRANCHED_SIG, CUSIG_BRANCHED_SIG_COMBINE)
from .data_handlers import PathInputHandler, SigOutputHandler, MultipleSigInputHandler, CorrectionInputHandler
from ..load_siglib import BUILT_WITH_CUDA, CPSIG, CUSIG


def prepare_branched_sig(
        dimension: int,
        degree: int,
        *,
        use_disk: bool = False,
        time_aug: bool = False,
        lead_lag: bool = False,
        planar: bool = False,
        device: str = "both",
):
    """
    Precomputes data required for branched signature computations. Must be called before
    branched signature computation for a given ``(dimension, degree)`` pair.

    With ``planar=False`` this prepares the BCK basis of non-planar rooted
    trees. With ``planar=True`` this prepares the MKW basis of ordered forests
    of planar rooted trees.

    If ``time_aug`` or ``lead_lag`` are set, the cache is prepared for
    the augmented dimension automatically.

    :param dimension: Dimension of the underlying path.
    :type dimension: int
    :param degree: Maximum order (number of nodes).
    :type degree: int
    :param use_disk: If ``False``, will cache prepared objects in memory only.
        If ``True``, will also save these objects in a shared disk cache to be
        re-used for future runs. The CPU and GPU libraries share the same
        disk cache format and directory.
        See additionally the documentation for ``pysiglib.set_cache_dir``.
    :type use_disk: bool
    :param time_aug: If True, prepare for time-augmented paths (dim + 1).
    :type time_aug: bool
    :param lead_lag: If True, prepare for lead-lag transformed paths (2 * dim).
    :type lead_lag: bool
    :param planar: If True, prepare for planar (ordered) branched signatures.
    :type planar: bool
    :param device: Which device caches to prepare. Must be ``"cpu"``,
        ``"cuda"``, or ``"both"``.
    :type device: str
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(use_disk, "use_disk", bool)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(planar, "planar", bool)
    check_type(device, "device", str)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    if device not in ("cpu", "cuda", "both"):
        raise ValueError("device must be 'cpu', 'cuda', or 'both'")

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    if device in ("cpu", "both"):
        err_code = CPSIG.prepare_branched_sig(
            aug_dimension, degree, use_disk, planar)
        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_branched_sig: "
                + err_msg(err_code, "cpu"))

    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.prepare_branched_sig_cuda(
            aug_dimension, degree, planar, use_disk)
        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_branched_sig (CUDA): "
                + err_msg(err_code, "cuda"))


def branched_sig_length(dimension: int, degree: int, *, planar: bool = False, scalar_term: bool = False) -> int:
    """
    Returns the length of a truncated branched signature.

    With ``planar=False`` this counts non-planar rooted trees up to ``degree``.
    With ``planar=True`` this counts ordered forests of planar rooted trees up
    to total degree.

    :param dimension: Dimension of the underlying path.
    :type dimension: int
    :param degree: Maximum order (number of nodes).
    :type degree: int
    :param planar: If True, return the length for planar (ordered) branched signatures.
    :type planar: bool
    :param scalar_term: If True, includes the empty-tree scalar term at index 0 in the length.
        If False (default), the returned length is one less (matching ``branched_sig``
        output with ``scalar_term=False``).
    :type scalar_term: bool
    :return: Length of the branched signature array.
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    out = CPSIG.branched_sig_length(dimension, degree, planar)
    if out == 0:
        raise ValueError("Invalid parameters or integer overflow in branched_sig_length")
    return out - (0 if scalar_term else 1)


def _infer_branched_scalar_term(bsig, dimension: int, degree: int, planar: bool = False) -> bool:
    """Return True iff ``bsig``'s trailing dimension includes the leading scalar 1.

    Raises ``ValueError`` if the shape matches neither the scalar_term=True nor
    the scalar_term=False branched-signature length for the given
    ``(dimension, degree, planar)``. Used by consumer-side branched-sig
    functions that accept bsigs in either format and match their output format
    to the input.
    """
    full_len = branched_sig_length(dimension, degree, planar=planar, scalar_term=True)
    actual = bsig.shape[-1]
    if actual == full_len:
        return True
    if actual == full_len - 1:
        return False
    raise ValueError(
        "bsig has incompatible length " + str(actual) + " for dimension=" + str(dimension) +
        ", degree=" + str(degree) + ", planar=" + str(planar) +
        " (expected " + str(full_len) + " or " + str(full_len - 1) + ")."
    )


def branched_sig(
    path: Array,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    planar: bool = False,
    scalar_term: bool = False,
    correction=None,
    n_jobs: int = 1,
) -> Array:
    """
    Computes the truncated branched signature of a path or batch of paths.

    The branched signature extends the standard path signature to iterated
    integrals indexed by decorated rooted trees, following Gubinelli (2010).
    For ``planar=True``, the output is instead indexed by the MKW ordered
    forest basis.

    Must call ``prepare_branched_sig(dimension, degree, planar=planar)``
    before first use, where ``dimension`` is the augmented dimension
    (accounting for ``time_aug`` and ``lead_lag``).

    :param path: Path of shape ``(length, dimension)`` or ``(..., length, dimension)``.
    :type path: Array
    :param degree: Maximum order (number of nodes).
    :type degree: int
    :param time_aug: If True, prepend a time channel to the path.
    :type time_aug: bool
    :param lead_lag: If True, apply the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for the time augmentation channel.
    :type end_time: float
    :param planar: If True, compute the planar MKW branched signature.
    :type planar: bool
    :param scalar_term: If True, the output includes the leading constant 1 at index 0
        (the empty-word term). If False (default), this leading element is stripped from the output.
    :type scalar_term: bool
    :param correction: Optional per-segment correction of level
        :math:`\\geq 2` added to the path increment on each path
        segment. The level-1 part of the local lift is the segment's
        path increment :math:`\\Delta x`, the higher levels come from
        the matching correction row, and the local branched signature on each
        segment is

        .. math::

            \\exp_* \\left( \\sum_i \\Delta x_i\\, e_i + \\sum_{k=2}^{m} \\sum_{i_1, \\ldots, i_k} c^{(k)}_{i_1 \\ldots i_k}\\, e_{i_1 \\cdots i_k} \\right),

        where :math:`e_w` is the chain (root-to-leaf path) tree with labels
        :math:`w` and :math:`\\exp_*` is the Hopf-algebra exponential under the
        Butcher product. A non-empty ``correction`` may have shape ``(C,)``
        for one constant correction shared by every segment and batch item,
        ``(path.shape[-2] - 1, C)`` for one correction row per segment shared
        by the batch, or ``path.shape[:-2] + (path.shape[-2] - 1, C)`` for
        batch-specific segment corrections. Here ``C`` is the correction
        width, with ``C = d^2 + d^3 + ... + d^m``, where :math:`d` is the
        underlying path dimension and
        :math:`2 \\leq m \\leq N` is the highest correction level supplied
        (missing higher levels are zero). Levels are concatenated in order,
        and within level :math:`k` the entry for chain
        :math:`(i_1, \\ldots, i_k)` lives at flat index
        :math:`i_1 d^{k-1} + i_2 d^{k-2} + \\cdots + i_k`. Passing ``None``
        (default) or an empty array is equivalent to all-zero correction. Indices
        are over the original path channels; with ``time_aug=True``, the
        appended time channel contributes no correction. Cannot be combined with
        ``lead_lag=True``.
    :type correction: Array | None
    :param n_jobs: Number of parallel threads for batch processing.
    :type n_jobs: int
    :return: Branched signature array of shape ``(bsig_len,)`` or ``(..., bsig_len)``.

    Example usage:
    ----------------

    Ito-lifted branched signature of a sampled Brownian path. For Brownian
    motion with instantaneous covariance :math:`\\Sigma`, setting the level-2
    correction to :math:`c^{(2)}_{ij} = -\\frac{1}{2}\\Sigma_{ij}\\,\\Delta t` per segment
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

        # Ito level-2 correction: -0.5 * dt * Sigma per path segment.
        correction = np.broadcast_to(
            (-0.5 * np.eye(d) * dt).reshape(1, -1), (n_steps, d * d)).copy()

        pysiglib.prepare_branched_sig(d, N)
        ito_bsig = pysiglib.branched_sig(
            path, N, correction=correction, end_time=T)
        print(ito_bsig)
    """
    check_type(degree, "degree", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(planar, "planar", bool)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    correction_data = CorrectionInputHandler(correction, data, degree)
    dimension = data.data_dimension
    aug_dimension = data.dimension
    bsig_len = branched_sig_length(aug_dimension, degree, planar=planar, scalar_term=scalar_term)
    result = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            dimension, data.data_length, degree, n_jobs,
            data.time_aug, data.lead_lag, data.end_time, planar, scalar_term,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    else:
        err_code = CUSIG_BRANCHED_SIG[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            dimension, data.data_length, degree,
            data.time_aug, data.lead_lag, data.end_time, planar, scalar_term,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    if err_code:
        raise Exception(
            "Error in pysiglib.branched_sig: "
            + err_msg(err_code, result.device))
    return result.data


def branched_sig_combine(
    bsig1: Array,
    bsig2: Array,
    dimension: int,
    degree: int,
    *,
    planar: bool = False,
    n_jobs: int = 1,
) -> Array:
    """
    Combines two truncated branched signatures via the Hopf product
    (the analogue of Chen's identity for branched rough paths).

    :param bsig1: First branched signature.
    :type bsig1: Array
    :param bsig2: Second branched signature.
    :type bsig2: Array
    :param dimension: Dimension of the underlying path.
    :type dimension: int
    :param degree: Maximum order (number of nodes).
    :type degree: int
    :param planar: If True, combine planar MKW branched signatures.
    :type planar: bool
    :param n_jobs: Number of parallel threads for batch processing.
    :type n_jobs: int
    :return: Combined branched signature, in the same ordering and scalar-term format as the inputs.
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    scalar_term = _infer_branched_scalar_term(bsig1, dimension, degree, planar=planar)
    bsig_len = branched_sig_length(dimension, degree, planar=planar, scalar_term=scalar_term)
    data = MultipleSigInputHandler([bsig1, bsig2], bsig_len, ["bsig1", "bsig2"])
    result = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_COMBINE[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, dimension, degree, n_jobs, planar, scalar_term)
    else:
        err_code = CUSIG_BRANCHED_SIG_COMBINE[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, dimension, degree, planar, scalar_term)
    if err_code:
        raise Exception(
            "Error in pysiglib.branched_sig_combine: "
            + err_msg(err_code, result.device))
    return result.data
