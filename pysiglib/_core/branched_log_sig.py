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

from typing import Optional


from .param_checks import check_type, check_non_neg, check_n_jobs
from .error_codes import err_msg
from .dtypes import (
    CPSIG_BRANCHED_LOG_SIG_FROM_PATH,
    CPSIG_BRANCHED_SIG_TO_LOG_SIG,
    CUSIG_BRANCHED_LOG_SIG_FROM_PATH,
    CUSIG_BRANCHED_SIG_TO_LOG_SIG,
)
from .data_handlers import (
    CorrectionInputHandler,
    MultipleSigInputHandler,
    PathInputHandler,
    SigOutputHandler,
)
from ..sig_length import aug_dim
from ..load_siglib import BUILT_WITH_CUDA, CPSIG, CUSIG
from .branched_sig import (
    _infer_branched_scalar_term,
    branched_sig,
    branched_sig_length,
)
from .transform_path import transform_path


def prepare_branched_log_sig(
        dimension: int,
        degree: int,
        method: int,
        *,
        use_disk: bool = False,
        time_aug: bool = False,
        lead_lag: bool = False,
        planar: bool = False,
        device: str = "both",
):
    """
    Precomputes data required for branched log signature computations. Must be called before
    ``branched_log_sig()`` or ``branched_sig_to_log_sig()`` for a given
    ``(dimension, degree, planar)`` combination. This also prepares the corresponding
    branched-signature cache.

    :param dimension: Dimension of the underlying path.
    :type dimension: int
    :param degree: Maximum order (number of nodes).
    :type degree: int
    :param method: Method to prepare. Method 0 computes the expanded branched
        log signature. Methods 1, 2, and 3 compute compressed MKW log signatures
        and require ``planar=True``.
    :type method: int
    :param use_disk: If ``True``, load or save the branched-signature cache on disk.
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
    check_type(method, "method", int)
    method = _resolve_branched_log_sig_method(method, planar)
    check_type(device, "device", str)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    if device not in ("cpu", "cuda", "both"):
        raise ValueError("device must be 'cpu', 'cuda', or 'both'")
    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    if device in ("cpu", "both"):
        err_code = CPSIG.prepare_branched_log_sig(
            aug_dimension, degree, method, use_disk, planar)
        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_branched_log_sig: "
                + err_msg(err_code, "cpu"))

    if (method == 3 and degree > 20 and BUILT_WITH_CUDA
            and device in ("cuda", "both")):
        raise NotImplementedError(
            "CUDA MKW BCH method supports degree at most 20")
    if BUILT_WITH_CUDA and device in ("cuda", "both"):
        err_code = CUSIG.prepare_branched_log_sig_cuda(
            aug_dimension, degree, method, planar, use_disk)
        if err_code:
            raise Exception(
                "Error in pysiglib.prepare_branched_log_sig (CUDA): "
                + err_msg(err_code, "cuda"))


def _resolve_branched_log_sig_method(method: Optional[int], planar: bool) -> int:
    check_type(planar, "planar", bool)
    if method is None:
        return 1 if planar else 0
    check_type(method, "method", int)
    if method not in (0, 1, 2, 3):
        raise ValueError("method must be one of 0, 1, 2 or 3. Got " + str(method) + " instead.")
    if method and not planar:
        raise ValueError(
            "branched log signature methods 1, 2 and 3 require planar=True")
    return method


def branched_log_sig_length(
        dimension: int,
        degree: int,
        *,
        planar: bool = False,
) -> int:
    """Returns the scalar-free length of a branched log signature.

    For ``planar=True``, this is the number of weighted Lyndon forests. For
    ``planar=False``, this is the number of decorated nonplanar rooted trees.
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    if dimension > 255:
        raise ValueError("branched log signature dimension must be <= 255")
    out = CPSIG.branched_log_sig_length(dimension, degree, planar)
    if out == 0 and dimension > 0 and degree > 0:
        raise ValueError(
            "Invalid parameters or integer overflow in branched_log_sig_length")
    return out


def branched_sig_to_log_sig(
    bsig: Array,
    dimension: int,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    planar: bool = False,
    method: Optional[int] = None,
    n_jobs: int = 1,
) -> Array:
    """
    Computes the branched log signature from the branched signature.

    :param bsig: The branched signature or batch of branched signatures, given as a `numpy.ndarray`
        or `Array`. For a single branched signature, this must be of shape
        ``branched_sig_length``. For a batch of paths, this must be of shape
        ``(batch_size, branched_sig_length)``. The leading scalar term may be present or omitted.
    :type bsig: Array
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the branched signature(s).
    :type degree: int
    :param time_aug: Whether the branched signature(s) were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the branched signature(s) were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param planar: If True, use planar branched signatures.
    :type planar: bool
    :param method: Method to use. Method 0 returns the expanded branched log
        signature. Methods 1 and 2 return compressed MKW coordinates and
        require ``planar=True``. If omitted, method 0 is used for nonplanar
        signatures and method 1 is used for planar signatures. Method 3 is not
        available for conversion from a branched signature.
    :type method: int | None
    :param n_jobs: Number of CPU threads to run in parallel. CUDA execution
        ignores this value.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: The branched log signature or batch of branched log signatures. Method 0
        preserves the scalar-term format of ``bsig``. Methods 1 and 2 are scalar-free.
    :rtype: Array

    Example usage:
    ----------------

    .. code-block:: python

        import numpy as np
        import pysiglib

        pysiglib.prepare_branched_log_sig(5, 3, 0)
        path = np.random.random((10, 100, 5))
        bsig = pysiglib.branched_sig(path, 3)
        blogsig = pysiglib.branched_sig_to_log_sig(bsig, 5, 3)
        print(blogsig)
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(planar, "planar", bool)
    method = _resolve_branched_log_sig_method(method, planar)
    if method == 3:
        raise ValueError(
            "method=3 is not supported in branched_sig_to_log_sig. "
            "Use branched_log_sig(path, degree, method=3) instead.")
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)
    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    scalar_term = _infer_branched_scalar_term(bsig, aug_dimension, degree, planar=planar)
    bsig_len = branched_sig_length(aug_dimension, degree, planar=planar, scalar_term=scalar_term)
    data = MultipleSigInputHandler([bsig], bsig_len, ["bsig"])
    out_len = (bsig_len if method == 0 else
               branched_log_sig_length(aug_dimension, degree, planar=True))
    result = SigOutputHandler(data, out_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_TO_LOG_SIG[data.dtype](
            data.sig_ptr[0], result.data_ptr, data.batch_size,
            aug_dimension, degree, method, n_jobs, planar, scalar_term)
    else:
        err_code = CUSIG_BRANCHED_SIG_TO_LOG_SIG[data.dtype](
            data.sig_ptr[0], result.data_ptr, data.batch_size,
            aug_dimension, degree, method, planar, scalar_term)
    if err_code:
        raise Exception(
            "Error in pysiglib.branched_sig_to_log_sig: "
            + err_msg(err_code, result.device))

    return result.data


def branched_log_sig(
    path: Array,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    planar: bool = False,
    scalar_term: bool = False,
    method: Optional[int] = None,
    correction=None,
    n_jobs: int = 1,
) -> Array:
    """
    Computes the branched log signature of a path.

    :param path: The underlying path or batch of paths, given as a `Array`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths,
        this must be of shape ``(batch_size, length, dimension)``.
    :type path: Array
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
    :param planar: If True, compute the planar branched log signature.
    :type planar: bool
    :param scalar_term: If True, include the leading scalar coefficient, which is zero.
    :type scalar_term: bool
    :param method: Method to use. Method 0 returns the expanded branched log
        signature. Methods 1, 2, and 3 return compressed MKW coordinates and
        require ``planar=True``. Method 3 uses direct sequential BCH updates
        and does not support non-empty corrections. If omitted, method 0 is
        used for nonplanar signatures and method 1 is used for planar
        signatures. ``scalar_term`` affects only method 0.
    :type method: int | None
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
    :param n_jobs: Number of CPU threads to run in parallel. CUDA execution
        ignores this value.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: The branched log signature or batch of branched log signatures.
    :rtype: Array

    Example usage:
    ----------------

    .. code-block:: python

        import numpy as np
        import pysiglib

        pysiglib.prepare_branched_log_sig(5, 3, 0)
        path = np.random.random((10, 100, 5))
        blogsig = pysiglib.branched_log_sig(path, 3)
        print(blogsig)

    Ito-lifted branched log signature of a sampled Brownian path. For Brownian
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

        pysiglib.prepare_branched_log_sig(d, N, 0)
        ito_blogsig = pysiglib.branched_log_sig(
            path, N, correction=correction, end_time=T)
        print(ito_blogsig)
    """
    method = _resolve_branched_log_sig_method(method, planar)
    if method == 3:
        input_data = PathInputHandler(
            path, time_aug, lead_lag, end_time, "path")
        if input_data.data_length == 0:
            raise ValueError(
                "branched_log_sig method 3 received an empty path")
        if correction is not None:
            correction_data = CorrectionInputHandler(
                correction, input_data, degree,
            )
            if correction_data.length != 0:
                raise ValueError(
                    "correction is not supported with branched_log_sig method=3")
        if time_aug or lead_lag:
            path = transform_path(
                path, time_aug=time_aug, lead_lag=lead_lag,
                end_time=end_time, n_jobs=n_jobs)
            data = PathInputHandler(path, False, False, 1.0, "path")
        else:
            data = input_data
        if data.device != "cpu" and degree > 20:
            raise NotImplementedError(
                "CUDA MKW BCH method supports degree at most 20")
        out_len = branched_log_sig_length(
            data.data_dimension, degree, planar=True)
        result = SigOutputHandler(data, out_len)
        if data.batch_size == 0:
            return result.data
        if data.device == "cpu":
            err_code = CPSIG_BRANCHED_LOG_SIG_FROM_PATH[data.dtype](
                data.data_ptr, result.data_ptr, data.batch_size,
                data.data_length, data.data_dimension, degree, n_jobs)
        else:
            err_code = CUSIG_BRANCHED_LOG_SIG_FROM_PATH[data.dtype](
                data.data_ptr, result.data_ptr, data.batch_size,
                data.data_length, data.data_dimension, degree)
        if err_code:
            raise Exception(
                "Error in pysiglib.branched_log_sig (method=3): "
                + err_msg(err_code, result.device))
        return result.data

    bsig = branched_sig(
        path, degree, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        planar=planar, scalar_term=scalar_term, correction=correction, n_jobs=n_jobs)
    dimension = path.shape[-1]
    return branched_sig_to_log_sig(
        bsig, dimension, degree, time_aug=time_aug, lead_lag=lead_lag,
        planar=planar, method=method, n_jobs=n_jobs)
