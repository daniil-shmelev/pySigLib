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

from ..load_siglib import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX
from ..words import words_of_length, words, lyndon_words_of_length, lyndon_words, is_lyndon, word_to_idx, idx_to_word
from ..transform_path import transform_path
from ..sig_length import sig_length, log_sig_length
from ..sig_coef import extract_sig_coef
from ..log_sig import set_cache_dir, prepare_log_sig, clear_cache
from ..static_kernels import Context, StaticKernel, LinearKernel, ScaledLinearKernel, RBFKernel, PolynomialKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel, RationalQuadraticKernel
from .torch_api import sig, sig_combine, sig_coef, transform_path, sig_to_log_sig, log_sig, log_sig_combine, logsig_to_sig, sig_kernel, sig_kernel_gram, sig_score, expected_sig_score, sig_mmd, branched_sig, prepare_branched_sig, branched_sig_length, branched_sig_combine, linear_sig, sig_join, log_sig_join

from ..streams import (
    SigStream as _SigStream,
    LogSigStream as _LogSigStream,
    SigWindowStream as _SigWindowStream,
    LogSigWindowStream as _LogSigWindowStream,
)

class SigStream(_SigStream):
    """SigStream with PyTorch autograd support. See :class:`pysiglib.SigStream`."""
    def __init__(self, dimension: int, degree: int):
        super().__init__(dimension, degree, _sig_join=sig_join, _sig_combine=sig_combine, _sig=sig)

class LogSigStream(_LogSigStream):
    """LogSigStream with PyTorch autograd support. See :class:`pysiglib.LogSigStream`."""
    def __init__(self, dimension: int, degree: int, method: int = 2):
        super().__init__(dimension, degree, method=method,
                         _log_sig_join=log_sig_join, _log_sig_combine=log_sig_combine,
                         _log_sig=lambda path, deg: log_sig(path, deg, method=method))

class SigWindowStream(_SigWindowStream):
    """SigWindowStream with PyTorch autograd support. See :class:`pysiglib.SigWindowStream`."""
    def __init__(self, dimension: int, degree: int, window_size: int, stride: int = 1):
        super().__init__(dimension, degree, window_size, stride, _sig=sig)

class LogSigWindowStream(_LogSigWindowStream):
    """LogSigWindowStream with PyTorch autograd support. See :class:`pysiglib.LogSigWindowStream`."""
    def __init__(self, dimension: int, degree: int, window_size: int, stride: int = 1, method: int = 2):
        super().__init__(dimension, degree, window_size, stride, method=method,
                         _log_sig=lambda path, deg: log_sig(path, deg, method=method))

signature = sig
