# Copyright 2025 Daniil Shmelev
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

import warnings


from .param_checks import check_type, check_non_neg, check_n_jobs
from .error_codes import err_msg
from .dtypes import CPSIG_SIGNATURE, CPSIG_SIG_COMBINE, CUSIG_SIGNATURE, CUSIG_SIG_COMBINE
from ..sig_length import sig_length, aug_dim, _infer_scalar_term
from .data_handlers import PathInputHandler, MultipleSigInputHandler, SigOutputHandler, CorrectionInputHandler


def sig_combine(
    sig1: Array,
    sig2: Array,
    dimension: int,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    n_jobs: int = 1,
) -> Array:
    """
    Combines two truncated signatures of the same degree and dimension into one signature. In particular, let :math:`x_1, x_2`
    be two paths such that the first point of :math:`x_2` is the last point of :math:`x_1`. Let :math:`S(x_1), S(x_2)`
    be the truncated signatures of :math:`x_1, x_2` respectively. Then calling this function on :math:`S(x_1), S(x_2)` returns
    the truncated signature of the concatenated path,

    .. math::

        S(x_1 * x_2) = S(x_1) \\otimes S(x_2),

    where :math:`x_1 * x_2` is the concatenation of the two paths :math:`x_1, x_2`.

    :param sig1: The first truncated signature
    :type sig1: Array
    :param sig2: The second truncated signature. Must have the same degree and dimension as the first.
    :type sig2: Array
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signatures, :math:`N`
    :type degree: int
    :param time_aug: Whether time augmentation was applied before computing
        the signature.
    :type time_aug: bool
    :param lead_lag: Whether the lead lag transformation was applied before computing
        the signature.
    :type lead_lag: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Combined signature, :math:`S(x_1 * x_2)`, in the same scalar-term format as the inputs.
    :rtype: Array

    Example usage::

        import pysiglib
        import numpy as np

        batch_size = 32
        length = 100
        dimension = 5
        degree = 3

        X1 = np.random.uniform(size=(batch_size, length, dimension))
        X2 = np.random.uniform(size=(batch_size, length, dimension))
        X_concat = np.concatenate((X1, X2), axis=1)

        X2 = np.concatenate((X1[:, [-1], :], X2), axis=1) # Make sure first pt of X2 is last pt of X1
        sig1 = pysiglib.sig(X1, degree)
        sig2 = pysiglib.sig(X2, degree)

        # The tensor product...
        sig_mult = pysiglib.sig_combine(sig1, sig2, dimension, degree)

        # ... is the same as the signature of the concatenated path:
        sig = pysiglib.sig(X_concat, degree)
    """

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_n_jobs(n_jobs)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    scalar_term = _infer_scalar_term(sig1, dimension, degree, time_aug=time_aug, lead_lag=lead_lag)

    sig_len = sig_length(aug_dimension, degree, scalar_term=scalar_term)
    data = MultipleSigInputHandler([sig1, sig2], sig_len, ["sig1", "sig2"])
    result = SigOutputHandler(data, sig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_SIG_COMBINE[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, aug_dimension, degree, scalar_term, n_jobs)
    else:
        err_code = CUSIG_SIG_COMBINE[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, aug_dimension, degree, scalar_term)
    if err_code:
        raise Exception(
            "Error in pysiglib.sig_combine: "
            + err_msg(err_code, result.device))
    return result.data


def sig(
    path: Array,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    horner: bool = True,
    scalar_term: bool = False,
    correction=None,
    n_jobs: int = 1,
) -> Array:
    """
    Computes the truncated signature of single path or a batch of paths. For
    a single path :math:`x`, the signature is given by

    .. math::

        S(x)_{[s,t]} := \\left( 1, S(x)^{(1)}_{[s,t]}, \\ldots, S(x)^{(N)}_{[s,t]}\\right) \\in T((\\mathbb{R}^d)),
    .. math::

        S(x)^{(k)}_{[s,t]} := \\int_{s < t_1 < \\cdots < t_k < t} dx_{t_1} \\otimes dx_{t_2} \\otimes \\cdots \\otimes dx_{t_k} \\in \\left(\\mathbb{R}^d\\right)^{\\otimes k}.

    :param path: The underlying path or batch of paths, given as a `Array`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: Array
    :param degree: The truncation level of the signature, :math:`N`.
    :type degree: int
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math:`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param horner: If True, will use Horner's algorithm for polynomial multiplication.
    :type horner: bool
    :param scalar_term: If True, the output includes the leading constant 1 at index 0
        (the empty-word term). If False (default), this leading element is stripped from the output.
    :type scalar_term: bool
    :param correction: Optional per-segment correction of level
        :math:`\\geq 2` added to each path increment before exponentiation.
        The level-1 part of the local
        lift is the segment's path increment :math:`\\Delta x`, the
        higher levels come from the corresponding correction row, and the local signature
        on each segment is

        .. math::

            \\exp \\left( \\sum_i \\Delta x_i\\, e_i + \\sum_{k=2}^{m} \\sum_{i_1, \\ldots, i_k} c^{(k)}_{i_1 \\ldots i_k}\\, e_{i_1} \\otimes \\cdots \\otimes e_{i_k} \\right),

        where :math:`(e_{i_1} \\otimes \\cdots \\otimes e_{i_k})` is the
        tensor basis. A non-empty ``correction`` may have shape ``(C,)``
        for one constant correction shared by every segment and batch item,
        ``(path.shape[-2] - 1, C)`` for one correction row per segment shared
        by the batch, or ``path.shape[:-2] + (path.shape[-2] - 1, C)`` for
        batch-specific segment corrections. Here ``C`` is the correction
        width, with :math:`C = d^2 + d^3 + \\cdots + d^m`, where
        :math:`d` is the original
        path dimension and :math:`2 \\leq m \\leq N` is the highest correction
        level supplied (missing higher levels are zero). Levels are
        concatenated in each row, and within level :math:`k` the entry for
        index tuple :math:`(i_1, \\ldots, i_k)` lives at flat index
        :math:`i_1 d^{k-1} + i_2 d^{k-2} + \\cdots + i_k`. Passing ``None``
        or an empty array is equivalent to all-zero correction. With
        ``time_aug=True``, the appended time channel contributes no
        correction entries. Cannot be combined with ``lead_lag=True``.
    :type correction: Array | None
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Truncated signature, or a batch of truncated signatures.
    :rtype: Array

    .. note::

        ``pysiglib.signature`` is an alias of ``pysiglib.sig`` included for backward
        compatibility with versions ``< 1.0.0``.

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        path = np.random.random((10, 100, 5))
        sigs = pysiglib.sig(path, degree=4)
        print(sigs)

    .. code-block:: python

        # Using time augmentation, lead-lag, and parallel threads
        import numpy as np
        import pysiglib

        path = np.random.random((10, 100, 5))
        sigs = pysiglib.sig(
            path,
            degree=4,
            time_aug=True,
            lead_lag=True,
            end_time=2.0,
            n_jobs=-1,
        )
        print(sigs)

    Ito-lifted signature of a sampled Brownian path. For Brownian motion
    with instantaneous covariance :math:`\\Sigma`, setting the level-2
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
        correction = np.broadcast_to((-0.5 * np.eye(d) * dt).reshape(1, -1), (n_steps, d * d))

        ito_sig = pysiglib.sig(path, N, correction=correction, end_time=T)
        print(ito_sig)

    """
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(horner, "horner", bool)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)

    check_n_jobs(n_jobs)
    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    correction_data = CorrectionInputHandler(correction, data, degree)
    sig_len = sig_length(data.dimension, degree, scalar_term=scalar_term)
    result = SigOutputHandler(data, sig_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_SIGNATURE[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            data.data_dimension, data.data_length, degree,
            data.time_aug, data.lead_lag, data.end_time, horner, scalar_term, n_jobs,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    else:
        err_code = CUSIG_SIGNATURE[data.dtype](
            data.data_ptr, result.data_ptr, data.batch_size,
            data.data_dimension, data.data_length, degree,
            data.time_aug, data.lead_lag, data.end_time, horner, scalar_term,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    if err_code:
        raise Exception(
            "Error in pysiglib.sig: " + err_msg(err_code, result.device))

    has_bad = not data.xp.all(data.xp.isfinite(result.data))
    if has_bad:
        warnings.warn(
            "sig produced NaN or Inf values. This is typically caused by paths "
            "with large increments, leading to numerical overflow. Consider "
            "normalizing your paths.",
            RuntimeWarning,
            stacklevel=2
        )

    return result.data
