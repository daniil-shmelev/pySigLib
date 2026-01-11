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

from .param_checks import check_word_or_word_list, check_type, check_non_neg
from .sig_length import sig_length
from .words import word_to_idx
from .data_handlers import SigInputHandler

def extract_sig_coef(
        word: Union[tuple[int, ...], list[tuple[int, ...]]],
        sig : Union[np.ndarray, torch.tensor],
        dimension: int,
        degree: int,
        time_aug: bool = False,
        lead_lag: bool = False
) -> Union[np.ndarray, torch.tensor]:
    """
    Extracts signature coefficients from a signature or batch of signatures.

    :param word: Word or list of words at which to extract coefficients.
    :type word: tuple[int, ...] | list[tuple[int, ...]]]
    :param sig: The signature or batch of signatures, given as a `numpy.ndarray` or `torch.tensor`.
        For a single signature, this must be of shape ``sig_length``. For a batch of paths, this must
        be of shape ``(batch_size, sig_length)``.
    :type sig: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying path(s).
    :type dimension: int
    :param degree: Truncation degree of the signature(s).
    :type degree: int
    :param time_aug: Whether the signatures were computed with ``time_aug=True``.
    :type time_aug: bool
    :param lead_lag: Whether the signatures were computed with ``lead_lag=True``.
    :type lead_lag: bool
    :return: Signature coefficients of shape ``num_words`` or batch of signature
        coefficients of shape ``(batch_size, num_words)``.
    :rtype: numpy.ndarray | torch.tensor
    """
    check_type(dimension, "dimension", int)
    check_non_neg(dimension, "dimension")
    check_type(degree, "degree", int)
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    word = check_word_or_word_list(word, aug_dimension, "word")

    sig_len = sig_length(aug_dimension, degree)
    data = SigInputHandler(sig, sig_len, "sig")

    idx = [word_to_idx(w, aug_dimension) for w in word]

    return sig[..., idx]