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
from .dtypes import (CPSIG_BRANCHED_SIG_BACKPROP,
                     CPSIG_BRANCHED_SIG_COMBINE_BACKPROP,
                     CUSIG_BRANCHED_SIG_BACKPROP,
                     CUSIG_BRANCHED_SIG_COMBINE_BACKPROP)
from .data_handlers import PathInputHandler, PathOutputHandler, MultipleSigInputHandler, SigOutputHandler, CorrectionInputHandler
from .branched_sig import branched_sig_length, _infer_branched_scalar_term


def branched_sig_backprop(
    path: Array,
    bsig: Array,
    bsig_derivs: Array,
    degree: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    end_time: float = 1.0,
    planar: bool = False,
    correction=None,
    n_jobs: int = 1,
) -> Array:
    """
    Backpropagates through the branched signature computation.

    Given the forward branched signature
    ``bsig = branched_sig(path, degree, correction=correction)`` and upstream
    derivatives ``bsig_derivs = dF/d(bsig)``, computes ``dF/d(path)``.

    :param path: Input path, shape ``(length, dimension)`` or ``(batch, length, dimension)``.
    :type path: Array
    :param bsig: Forward branched signature output.
    :type bsig: Array
    :param bsig_derivs: Upstream derivatives w.r.t. the branched signature.
    :type bsig_derivs: Array
    :param degree: Maximum order (must match forward call).
    :type degree: int
    :param time_aug: Whether time augmentation was used in the forward pass.
    :type time_aug: bool
    :param lead_lag: Whether lead-lag was used in the forward pass.
    :type lead_lag: bool
    :param end_time: End time for time augmentation.
    :type end_time: float
    :param planar: If True, backpropagate through planar branched signature.
    :type planar: bool
    :param correction: The same correction supplied to the forward call
        (see :func:`branched_sig` for layout and semantics).
        Treated as a constant: no derivatives are returned with respect to
        ``correction``. Cannot be combined with ``lead_lag=True``.
    :type correction: Array | None
    :param n_jobs: Number of parallel threads for batch processing.
    :type n_jobs: int
    :return: Path derivatives, same shape as ``path``.

    Example usage:
    ----------------

    Forward and backward pass through the Ito-lifted branched signature of
    a sampled Brownian path. The same ``correction`` array must be passed to
    both calls.

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
        bsig = pysiglib.branched_sig(
            path, N, correction=correction, end_time=T)
        bsig_derivs = np.ones_like(bsig)
        grad = pysiglib.branched_sig_backprop(
            path, bsig, bsig_derivs, N, correction=correction, end_time=T)
        print(grad.shape)
    """
    check_type(degree, "degree", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(planar, "planar", bool)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    path_data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    correction_data = CorrectionInputHandler(correction, path_data, degree)
    dimension = path_data.data_dimension
    aug_dimension = path_data.dimension

    scalar_term = _infer_branched_scalar_term(bsig, aug_dimension, degree, planar=planar)
    bsig_len = branched_sig_length(aug_dimension, degree, planar=planar, scalar_term=scalar_term)
    sig_data = MultipleSigInputHandler([bsig, bsig_derivs], bsig_len, ["bsig", "bsig_derivs"])
    result = PathOutputHandler(path_data.data_length, path_data.data_dimension, path_data)

    if path_data.batch_size == 0:
        return result.data

    if path_data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_BACKPROP[path_data.dtype](
            path_data.data_ptr, result.data_ptr,
            sig_data.sig_ptr[1], sig_data.sig_ptr[0],
            path_data.batch_size, dimension, path_data.data_length, degree, n_jobs,
            path_data.time_aug, path_data.lead_lag, path_data.end_time, planar, scalar_term,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    else:
        err_code = CUSIG_BRANCHED_SIG_BACKPROP[path_data.dtype](
            path_data.data_ptr, result.data_ptr,
            sig_data.sig_ptr[1], sig_data.sig_ptr[0],
            path_data.batch_size, dimension, path_data.data_length, degree,
            path_data.time_aug, path_data.lead_lag, path_data.end_time, planar, scalar_term,
            correction_data.data_ptr, correction_data.length,
            correction_data.batch_stride, correction_data.segment_stride)
    if err_code:
        raise Exception(
            "Error in pysiglib.branched_sig_backprop: "
            + err_msg(err_code, result.device))
    return result.data


def branched_sig_combine_backprop(
    derivs: Array,
    bsig1: Array,
    bsig2: Array,
    dimension: int,
    degree: int,
    *,
    planar: bool = False,
    n_jobs: int = 1,
) -> tuple:
    """
    Backpropagates through the branched signature combine (Butcher product).

    Given ``out = branched_sig_combine(bsig1, bsig2, dimension, degree)``
    and upstream derivatives ``derivs = dF/d(out)``, computes
    ``(dF/d(bsig1), dF/d(bsig2))``.

    :param derivs: Upstream derivatives, same shape as combine output.
    :type derivs: Array
    :param bsig1: First branched signature input to the forward combine.
    :type bsig1: Array
    :param bsig2: Second branched signature input to the forward combine.
    :type bsig2: Array
    :param dimension: Dimension of the underlying path.
    :type dimension: int
    :param degree: Maximum order.
    :type degree: int
    :param planar: If True, backpropagate through planar branched sig combine.
    :type planar: bool
    :param n_jobs: Number of parallel threads for batch processing.
    :type n_jobs: int
    :return: Tuple ``(dF/d(bsig1), dF/d(bsig2))``, in the same scalar-term format as the inputs.
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    scalar_term = _infer_branched_scalar_term(bsig1, dimension, degree, planar=planar)
    bsig_len = branched_sig_length(dimension, degree, planar=planar, scalar_term=scalar_term)
    data = MultipleSigInputHandler([derivs, bsig1, bsig2], bsig_len, ["derivs", "bsig1", "bsig2"])
    result1 = SigOutputHandler(data, bsig_len)
    result2 = SigOutputHandler(data, bsig_len)

    if data.batch_size == 0:
        return result1.data, result2.data

    if data.device == "cpu":
        err_code = CPSIG_BRANCHED_SIG_COMBINE_BACKPROP[data.dtype](
            data.sig_ptr[1], data.sig_ptr[2], data.sig_ptr[0],
            result1.data_ptr, result2.data_ptr,
            data.batch_size, dimension, degree, n_jobs, planar, scalar_term)
    else:
        err_code = CUSIG_BRANCHED_SIG_COMBINE_BACKPROP[data.dtype](
            data.sig_ptr[1], data.sig_ptr[2], data.sig_ptr[0],
            result1.data_ptr, result2.data_ptr,
            data.batch_size, dimension, degree, planar, scalar_term)
    if err_code:
        raise Exception(
            "Error in pysiglib.branched_sig_combine_backprop: "
            + err_msg(err_code, result1.device))
    return result1.data, result2.data
