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
from ctypes import c_uint64, POINTER, cast

import numpy as np
import torch

from .param_checks import check_word_or_word_list, check_type, check_non_neg, check_n_jobs
from .error_codes import err_msg
from .sig_length import aug_dim
from .dtypes import CPSIG_SIG_COEF, CUSIG_SIG_COEF_CUDA
from .words import word_to_idx
from .data_handlers import SigInputHandler, PathInputHandler, SigOutputHandler

def extract_sig_coef(
        sig : Union[np.ndarray, torch.tensor],
        words: Union[tuple[int, ...], list[tuple[int, ...]]],
        dimension: int,
        time_aug: bool = False,
        lead_lag: bool = False
) -> Union[np.ndarray, torch.tensor]:
    """
    Extracts signature coefficients from a signature or batch of signatures.

    :param sig: The signature or batch of signatures, given as a `numpy.ndarray` or `torch.tensor`.
        For a single signature, this must be of shape ``sig_length``. For a batch of paths, this must
        be of shape ``(batch_size, sig_length)``.
    :type sig: numpy.ndarray | torch.tensor
    :param words: Word or list of words at which to extract coefficients.
    :type words: tuple[int, ...] | list[tuple[int, ...]]]
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :return: Signature coefficients of shape ``num_words`` or batch of signature
        coefficients of shape ``(batch_size, num_words)``.
    :rtype: numpy.ndarray | torch.tensor

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        sigs = pysiglib.sig(path, degree=4)
        words = [(0,), (1, 0), (1, 2, 3)]
        coefs = pysiglib.extract_sig_coef(sigs, words, dimension=5)
        print(coefs)

    .. code-block:: python

        # Extract coefficients from signatures computed with time_aug and lead_lag
        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        sigs = pysiglib.sig(path, degree=4, time_aug=True, lead_lag=True)

        # With lead_lag the dimension doubles (10), and time_aug adds one (11).
        # Words now index into the augmented dimension.
        words = [(6,), (10, 9)]
        coefs = pysiglib.extract_sig_coef(
            sigs, words, dimension=5, time_aug=True, lead_lag=True
        )
        print(coefs)

    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)

    words = check_word_or_word_list(words, aug_dimension, "word")

    sig_len = sig.shape[-1]
    data = SigInputHandler(sig, sig_len, "sig")

    idx = [word_to_idx(w, aug_dimension) for w in words]

    return sig[..., idx]

def sig_coef(
        path : Union[np.ndarray, torch.tensor],
        words : Union[tuple[int, ...], list[tuple[int, ...]]],
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        prefixes : bool = False,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes specific signature coefficients for a single path or a batch of paths. For
    a single path :math:`x`, the signature coefficient at a multi-index
    :math:`I = (i_1, i_2, \\ldots, i_k)` is given by

    .. math::

        S(x)^I_{[s,t]} := \\int_{s < t_1 < \\cdots < t_k < t} dx^{i_1}_{t_1} \\otimes dx^{i_2}_{t_2} \\otimes \\cdots \\otimes dx^{i_k}_{t_k}.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape ``(length, dimension)``. For a batch of paths, this must
        be of shape ``(batch_size, length, dimension)``.
    :type path: numpy.ndarray | torch.tensor
    :param words: Multi-indices :math:`I` at which to evaluate signature coefficients, given as a list
        of lists of integers in :math:`[0, d-1]`, where :math:`d` is the dimension of the path(s). For example,
        for a 2-dimensional path, one could pass ``[(0,), (1,0), (0,1,1)]`` to compute the coefficients at
        the three multi-indices :math:`I = (0), (1,0), (0,1,1)`.
    :type words: tuple[int, ...] | list[tuple[int, ...]]
    :param time_aug: If set to True, will compute signature coefficients of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math:`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute signature coefficients of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param prefixes: If ``True``, will additionally return all prefixes of signature coefficients.
        These prefixes are extracted for free as a by-product of the computation.
        For example, passing ``word=[(1,2), (3,2,1)]`` with ``prefixes=True`` returns an
        output equivalent to passing ``word=[(1,), (1,2), (3,), (3,2), (3,2,1)]`` with ``prefixes=False``.
    :type prefixes: bool
    :param n_jobs: Number of threads to run in parallel. If n_jobs = 1, the computation is run serially.
        If set to -1, all available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs)
        threads are used. For example if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :return: Signature coefficients of shape ``num_words`` or batch of signature
        coefficients of shape ``(batch_size, num_words)``.
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        If the number of requested coefficients is large relative to the size of the full truncated signature,
        it is usually faster to call ``pysiglib.signature`` and extract the required coefficients using
        ``pysiglib.extract_sig_coefs``. This function is only faster when a very sparse collection
        of coefficients is required.

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        words = [(0,), (1,0), (1,2,3)]
        coefs = pysiglib.sig_coef(path, words)

    .. code-block:: python

        # Using prefixes to return all prefix coefficients
        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        words = [(4, 3), (1, 2, 3)]
        coefs = pysiglib.sig_coef(path, words, prefixes=True)
        # Returns coefficients for (4,), (4,3), (1,), (1,2), and (1,2,3)
        print(coefs)

    .. code-block:: python

        # Computing specific coefficients with time_aug and lead_lag
        import torch
        import pysiglib

        path = torch.rand((10, 100, 5))
        # With lead_lag the dimension doubles (10), and time_aug adds one (11).
        # Words now index into the augmented dimension.
        words = [(6,), (10, 9)]
        coefs = pysiglib.sig_coef(path, words, lead_lag=True, time_aug=True, end_time=2.0)
        print(coefs)

    """
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)
    check_type(end_time, "end_time", float)
    check_type(prefixes, "prefixes", bool)

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")

    # CUDA sig_coef doesn't support time_aug/lead_lag natively —
    # transform the path first, then recurse with no augmentation flags.
    if data.device != "cpu" and (time_aug or lead_lag):
        from .transform_path import transform_path
        transformed = transform_path(path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time)
        return sig_coef(transformed, words, time_aug=False, lead_lag=False, end_time=1., prefixes=prefixes, n_jobs=n_jobs)

    words = check_word_or_word_list(words, data.dimension, "word")

    if len(words) == 0:
        raise ValueError("words must be a non-empty list of multi-indices.")

    num_multi_indices = len(words)
    degrees = [len(idx) for idx in words]
    if prefixes:
        result_length = 0
        for idx in words:
            result_length += len(idx) if idx else 1
    else:
        result_length = num_multi_indices

    flat_indices = [i for idx in words for i in idx]

    if data.device == "cpu":
        words_t = torch.tensor(flat_indices, dtype=torch.uint64)
        degrees_t = torch.tensor(degrees, dtype=torch.uint64)
    else:
        words_t = torch.tensor(flat_indices, dtype=torch.uint64, device=path.device)
        degrees_t = torch.tensor(degrees, dtype=torch.uint64, device=path.device)

    multi_indices_ptr = cast(words_t.data_ptr(), POINTER(c_uint64))
    degrees_ptr = cast(degrees_t.data_ptr(), POINTER(c_uint64))

    result = SigOutputHandler(data, result_length)

    if data.batch_size == 0:
        return result.data

    check_n_jobs(n_jobs)
    if data.device == "cpu":
        err_code = CPSIG_SIG_COEF[data.dtype](
            data.data_ptr, result.data_ptr,
            multi_indices_ptr, num_multi_indices, degrees_ptr,
            data.batch_size, data.data_dimension, data.data_length,
            data.time_aug, data.lead_lag, data.end_time, prefixes, n_jobs)
    else:
        err_code = CUSIG_SIG_COEF_CUDA[data.dtype](
            data.data_ptr, result.data_ptr,
            multi_indices_ptr, num_multi_indices, degrees_ptr,
            data.batch_size, data.data_dimension, data.data_length, prefixes)
    if err_code:
        raise Exception("Error in pysiglib.sig_coef: " + err_msg(err_code))
    return result.data
