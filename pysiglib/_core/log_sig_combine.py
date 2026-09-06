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
from .dtypes import CPSIG_LOG_SIG_COMBINE, CUSIG_LOG_SIG_COMBINE, CPSIG_LOG_SIG_COMBINE_BACKPROP, CUSIG_LOG_SIG_COMBINE_BACKPROP
from ..sig_length import log_sig_length, aug_dim
from .data_handlers import MultipleSigInputHandler, SigOutputHandler


def log_sig_combine(
    log_sig1: Array,
    log_sig2: Array,
    dimension: int,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    n_jobs: int = 1,
) -> Array:
    """
    Combines two truncated log-signatures (Lyndon basis) of the same degree and dimension
    using the Baker-Campbell-Hausdorff (BCH) formula. In particular, let :math:`x_1, x_2`
    be two paths such that the first point of :math:`x_2` is the last point of :math:`x_1`.
    Let :math:`L(x_1), L(x_2)` be the truncated log-signatures of :math:`x_1, x_2`
    respectively (computed with ``method=2`` or ``method=3``). Then calling this function on
    :math:`L(x_1), L(x_2)` returns the truncated log-signature of the concatenated path,

    .. math::

        L(x_1 * x_2) = \\text{BCH}(L(x_1), L(x_2)).

    .. note::

        This function requires a call to
        ``pysiglib.prepare_log_sig(dimension, degree, method=3)``.

    :param log_sig1: The first truncated log-signature (Lyndon basis, method=2 or method=3)
    :type log_sig1: Array
    :param log_sig2: The second truncated log-signature (Lyndon basis, method=2 or method=3).
        Must have the same degree and dimension as the first.
    :type log_sig2: Array
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the log-signatures, :math:`N`
    :type degree: int
    :param time_aug: Whether time augmentation was applied before computing
        the log-signature.
    :type time_aug: bool
    :param lead_lag: Whether the lead lag transformation was applied before computing
        the log-signature.
    :type lead_lag: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation
        is run serially. If set to -1, all available threads are used.
    :type n_jobs: int
    :return: Combined log-signature (Lyndon basis)
    :rtype: Array
    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_n_jobs(n_jobs)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)

    ls_len = log_sig_length(aug_dimension, degree)
    data = MultipleSigInputHandler([log_sig1, log_sig2], ls_len, ["log_sig1", "log_sig2"])
    result = SigOutputHandler(data, ls_len)

    if data.batch_size == 0:
        return result.data

    if data.device == "cpu":
        err_code = CPSIG_LOG_SIG_COMBINE[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, aug_dimension, degree, n_jobs)
    else:
        err_code = CUSIG_LOG_SIG_COMBINE[data.dtype](
            data.sig_ptr[0], data.sig_ptr[1], result.data_ptr,
            data.batch_size, aug_dimension, degree)
    if err_code:
        raise Exception(
            "Error in pysiglib.log_sig_combine: "
            + err_msg(err_code, result.device))
    return result.data


def log_sig_combine_backprop(
    deriv: Array,
    ls1: Array,
    ls2: Array,
    dimension: int,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    n_jobs: int = 1,
):
    """
    Backpropagation through :func:`log_sig_combine`. Given derivatives of a scalar
    function :math:`F` with respect to the output of ``log_sig_combine``,
    computes the derivatives with respect to the two input log-signatures.

    :param deriv: Derivative with respect to the combined log-signature
    :type deriv: Array
    :param ls1: The first truncated log-signature (Lyndon basis, method=2 or method=3)
    :type ls1: Array
    :param ls2: The second truncated log-signature (Lyndon basis, method=2 or method=3)
    :type ls2: Array
    :param dimension: Dimension of the underlying space
    :type dimension: int
    :param degree: Truncation level of the log-signatures
    :type degree: int
    :param time_aug: Whether time augmentation was applied
    :type time_aug: bool
    :param lead_lag: Whether the lead-lag transformation was applied
    :type lead_lag: bool
    :param n_jobs: Number of threads (CPU only)
    :type n_jobs: int
    :return: Derivatives with respect to ``ls1`` and ``ls2``
    :rtype: Tuple[Array, Array]
    """

    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_n_jobs(n_jobs)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)

    ls_len = log_sig_length(aug_dimension, degree)
    data = MultipleSigInputHandler([ls1, ls2, deriv], ls_len, ["ls1", "ls2", "deriv"])
    result1 = SigOutputHandler(data, ls_len)
    result2 = SigOutputHandler(data, ls_len)

    if data.batch_size == 0:
        return result1.data, result2.data

    if data.device == "cpu":
        err_code = CPSIG_LOG_SIG_COMBINE_BACKPROP[data.dtype](
            data.sig_ptr[2], result1.data_ptr, result2.data_ptr,
            data.sig_ptr[0], data.sig_ptr[1],
            data.batch_size, aug_dimension, degree, n_jobs)
    else:
        err_code = CUSIG_LOG_SIG_COMBINE_BACKPROP[data.dtype](
            data.sig_ptr[2], result1.data_ptr, result2.data_ptr,
            data.sig_ptr[0], data.sig_ptr[1],
            data.batch_size, aug_dimension, degree)
    if err_code:
        raise Exception(
            "Error in pysiglib.log_sig_combine_backprop: "
            + err_msg(err_code, result1.device))
    return result1.data, result2.data
