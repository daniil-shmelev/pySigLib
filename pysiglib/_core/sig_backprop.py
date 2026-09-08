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

from .param_checks import check_type, check_non_neg, check_n_jobs
from .error_codes import err_msg
from .data_handlers import PathInputHandler, SigOutputHandler, PathOutputHandler, MultipleSigInputHandler, CorrectionInputHandler
from .dtypes import CPSIG_SIG_BACKPROP, CPSIG_SIG_COMBINE_BACKPROP, CUSIG_SIG_BACKPROP, CUSIG_SIG_COMBINE_BACKPROP
from ..sig_length import sig_length, aug_dim, _infer_scalar_term


def sig_combine_backprop(
    deriv: Array,
    sig1: Array,
    sig2: Array,
    dimension: int,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    n_jobs: int = 1,
):
    """
    This function is required to backpropagate through ``pysiglib.sig_combine``.
    Given the derivatives of a scalar function :math:`F` with respect to the
    result of ``pysiglib.sig_combine``, :math:`\\partial F / \\partial S(x_1 * x_2)`,
    returns the derivatives of :math:`F` with respect to the original two signatures,
    :math:`\\partial F / \\partial S(x_1)` and :math:`\\partial F / \\partial S(x_2)`.

    :param deriv: Derivative with respect to the combined signature,
        :math:`\\partial F / \\partial S(x_1 * x_2)`
    :type deriv: Array
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
    :return: Derivatives with respect to ``sig1`` and ``sig2``, in the same scalar-term format as the inputs.
    :rtype: Tuple[Array, Array]

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        batch_size, length, dimension, degree = 10, 100, 5, 3
        X1 = np.random.uniform(size=(batch_size, length, dimension))
        X2 = np.random.uniform(size=(batch_size, length, dimension))

        sig1 = pysiglib.sig(X1, degree, time_aug=True)
        sig2 = pysiglib.sig(X2, degree, time_aug=True)
        combined = pysiglib.sig_combine(sig1, sig2, dimension, degree, time_aug=True)

        derivs = np.ones_like(combined)
        dsig1, dsig2 = pysiglib.sig_combine_backprop(
            derivs, sig1, sig2, dimension, degree, time_aug=True
        )
        print(dsig1)

    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)
    scalar_term = _infer_scalar_term(sig1, dimension, degree, time_aug=time_aug, lead_lag=lead_lag)
    sig_len = sig_length(aug_dimension, degree, scalar_term=scalar_term)

    check_n_jobs(n_jobs)
    sig_data = MultipleSigInputHandler([sig1, sig2, deriv], sig_len, ["sig1", "sig2", "sig_combined_deriv"])

    sig1_deriv = SigOutputHandler(sig_data, sig_len)
    sig2_deriv = SigOutputHandler(sig_data, sig_len)

    if sig_data.batch_size == 0:
        return sig1_deriv.data, sig2_deriv.data

    if sig_data.device == "cpu":
        err_code = CPSIG_SIG_COMBINE_BACKPROP[sig_data.dtype](
            sig_data.sig_ptr[2], sig1_deriv.data_ptr, sig2_deriv.data_ptr,
            sig_data.sig_ptr[0], sig_data.sig_ptr[1],
            sig_data.batch_size, aug_dimension, degree, scalar_term, n_jobs)
    else:
        err_code = CUSIG_SIG_COMBINE_BACKPROP[sig_data.dtype](
            sig_data.sig_ptr[2], sig1_deriv.data_ptr, sig2_deriv.data_ptr,
            sig_data.sig_ptr[0], sig_data.sig_ptr[1],
            sig_data.batch_size, aug_dimension, degree, scalar_term)
    if err_code:
        raise Exception(
            "Error in pysiglib.sig_combine_backprop: "
            + err_msg(err_code, sig1_deriv.device))
    return sig1_deriv.data, sig2_deriv.data


def sig_backprop(
    path: Array,
    sig: Array,
    sig_derivs: Array,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    correction=None,
    n_jobs: int = 1,
) -> Array:
    """
    This function is required to backpropagate through the signature computation.
    Given the derivatives of a scalar function :math:`F` with respect to the
    signature, :math:`\\partial F / \\partial S(x)`, returns the
    derivatives of :math:`F` with respect to the underlying path,
    :math:`\\partial F / \\partial x`.

    :param path: The underlying path or batch of paths, given as a `Array`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: Array
    :param sig: Signature(s) of the path or batch of paths.
    :type sig: Array
    :param sig_derivs: Derivatives of the scalar function :math:`F` with respect to the signature(s),
        :math:`\\partial F / \\partial S(x)`. This must be an array of the same shape as the
        provided signature(s).
    :type sig_derivs: Array
    :param degree: The truncation level of the signature, :math:`N`.
    :type degree: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param correction: The same correction supplied to the forward ``sig`` call.
        Treated as a constant: no derivatives are
        returned with respect to ``correction``. Cannot be combined with
        ``lead_lag=True``.
    :type correction: Array | None
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives of the scalar function :math:`F` with respect to the path(s), :math:`\\partial F / \\partial x`.
        This is an array of the same shape as the provided path(s).
    :rtype: Array

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        path = np.random.random((10, 100, 5))
        degree = 4
        sigs = pysiglib.sig(path, degree)
        sig_derivs = np.ones_like(sigs)
        path_derivs = pysiglib.sig_backprop(path, sigs, sig_derivs, degree)
        print(path_derivs)

    .. code-block:: python

        # Backprop with time augmentation and lead-lag
        import numpy as np
        import pysiglib

        path = np.random.random((10, 100, 5))
        degree = 4
        sigs = pysiglib.sig(path, degree, time_aug=True, lead_lag=True, end_time=2.0)
        sig_derivs = np.ones_like(sigs)
        path_derivs = pysiglib.sig_backprop(
            path, sigs, sig_derivs, degree,
            time_aug=True, lead_lag=True, end_time=2.0,
        )
        print(path_derivs)

    """
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)

    path_data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    correction_data = CorrectionInputHandler(correction, path_data, degree)
    scalar_term = _infer_scalar_term(sig, path_data.data_dimension, degree, time_aug=time_aug, lead_lag=lead_lag)
    sig_len = sig_length(path_data.dimension, degree, scalar_term=scalar_term)
    sig_data = MultipleSigInputHandler([sig, sig_derivs], sig_len, ["sig", "sig_derivs"])

    if path_data.type_ != sig_data.type_:
        raise ValueError("path, sig and sig_derivs must all be numpy arrays or torch tensors")
    if path_data.dtype != sig_data.dtype:
        raise ValueError("path, sig and sig_derivs must have the same dtype")

    result = PathOutputHandler(path_data.data_length, path_data.data_dimension, path_data)

    if path_data.batch_size == 0:
        return result.data

    if path_data.batch_shape != sig_data.batch_shape:
        raise ValueError("path, sig and sig_derivs must have the same batch shape")

    check_n_jobs(n_jobs)
    if path_data.device == "cpu":
        err_code = CPSIG_SIG_BACKPROP[path_data.dtype](
            path_data.data_ptr, result.data_ptr,
            sig_data.sig_ptr[1], sig_data.sig_ptr[0],
            path_data.batch_size, path_data.data_dimension, path_data.data_length,
            degree, path_data.time_aug, path_data.lead_lag, path_data.end_time, scalar_term, n_jobs,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    else:
        err_code = CUSIG_SIG_BACKPROP[path_data.dtype](
            path_data.data_ptr, result.data_ptr,
            sig_data.sig_ptr[1], sig_data.sig_ptr[0],
            path_data.batch_size, path_data.data_dimension, path_data.data_length,
            degree, path_data.time_aug, path_data.lead_lag, path_data.end_time, scalar_term,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    if err_code:
        raise Exception(
            "Error in pysiglib.sig_backprop: "
            + err_msg(err_code, result.device))
    return result.data
