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
from .dtypes import CPSIG_SIG_JOIN_BACKPROP, CUSIG_SIG_JOIN_BACKPROP
from .sig_length import sig_length, _infer_scalar_term
from .data_handlers import SigInputHandler, SigOutputHandler


def sig_join_backprop(
        d_out : Union[np.ndarray, torch.Tensor],
        sig : Union[np.ndarray, torch.Tensor],
        displacement : Union[np.ndarray, torch.Tensor],
        dimension : int,
        degree : int,
        *,
        prepend : bool = False,
        n_jobs : int = 1
):
    """
    This function is required to backpropagate through ``pysiglib.sig_join``.
    Given the derivatives of a scalar function :math:`F` with respect to the
    result of ``pysiglib.sig_join``, :math:`\\partial F / \\partial S(x * v)`,
    returns the derivatives of :math:`F` with respect to the original signature
    and the displacement vector,
    :math:`\\partial F / \\partial S(x)` and :math:`\\partial F / \\partial v`.

    :param d_out: Derivative with respect to the output of sig_join,
        :math:`\\partial F / \\partial S(x * v)`
    :type d_out: numpy.ndarray | torch.Tensor
    :param sig: The original truncated signature, :math:`S(x)`
    :type sig: numpy.ndarray | torch.Tensor
    :param displacement: The displacement vector, :math:`v`
    :type displacement: numpy.ndarray | torch.Tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signatures, :math:`N`
    :type degree: int
    :param prepend: Must match the value used in the forward ``sig_join`` call. If True, the linear
        segment was prepended to the front of the path rather than appended. Default is False.
    :type prepend: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Derivatives with respect to ``sig`` and ``displacement``
    :rtype: Tuple[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor]

    Example:
    ---------

    .. code-block:: python

        import numpy as np
        import pysiglib

        dimension, degree = 5, 3
        path = np.random.uniform(size=(100, dimension))
        sig = pysiglib.sig(path, degree)
        displacement = np.random.uniform(size=(dimension,))

        extended = pysiglib.sig_join(sig, displacement, dimension, degree)
        d_out = np.ones_like(extended)

        d_sig, d_displacement = pysiglib.sig_join_backprop(
            d_out, sig, displacement, dimension, degree
        )
        print(d_sig.shape, d_displacement.shape)

    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_n_jobs(n_jobs)

    scalar_term = _infer_scalar_term(sig, dimension, degree)
    sig_len = sig_length(dimension, degree, scalar_term=scalar_term)

    d_out_data = SigInputHandler(d_out, sig_len, "d_out")
    sig_data = SigInputHandler(sig, sig_len, "sig")
    disp_data = SigInputHandler(displacement, dimension, "displacement")

    if d_out_data.type_ != sig_data.type_ or sig_data.type_ != disp_data.type_:
        raise ValueError("d_out, sig and displacement must all be numpy arrays or all torch tensors")
    if d_out_data.dtype != sig_data.dtype or sig_data.dtype != disp_data.dtype:
        raise ValueError("d_out, sig and displacement must have the same dtype")
    if not (d_out_data.batch_shape == sig_data.batch_shape == disp_data.batch_shape):
        raise ValueError("d_out, sig and displacement must have the same batch shape")
    if d_out_data.device != sig_data.device or sig_data.device != disp_data.device:
        raise ValueError("d_out, sig and displacement must be on the same device")

    d_sig = SigOutputHandler(d_out_data, sig_len)
    d_disp = SigOutputHandler(d_out_data, dimension)

    if d_out_data.batch_size == 0:
        return d_sig.data, d_disp.data

    if d_out_data.device == "cpu":
        err_code = CPSIG_SIG_JOIN_BACKPROP[d_out_data.dtype](
            d_out_data.data_ptr, d_sig.data_ptr, d_disp.data_ptr,
            sig_data.data_ptr, disp_data.data_ptr,
            d_out_data.batch_size, dimension, degree, prepend, scalar_term, n_jobs)
    else:
        err_code = CUSIG_SIG_JOIN_BACKPROP[d_out_data.dtype](
            d_out_data.data_ptr, d_sig.data_ptr, d_disp.data_ptr,
            sig_data.data_ptr, disp_data.data_ptr,
            d_out_data.batch_size, dimension, degree, prepend, scalar_term)
    if err_code:
        raise Exception("Error in pysiglib.sig_join_backprop: " + err_msg(err_code))
    return d_sig.data, d_disp.data
