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

from .jax_api import (
    sig, sig_combine, transform_path,
    sig_to_log_sig, logsig_to_sig, log_sig, log_sig_combine,
    sig_join, log_sig_join, linear_sig,
    sig_coef, branched_sig_coef, sig_kernel, sig_kernel_gram,
    branched_sig_kernel, branched_sig_kernel_gram,
    sig_score, expected_sig_score, sig_mmd,
    branched_sig, branched_sig_combine,
    branched_sig_to_log_sig, branched_log_sig,
)
from .static_kernels_jax import (
    LinearKernel, ScaledLinearKernel, RBFKernel,
    PolynomialKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel,
    RationalQuadraticKernel,
)

# Re-export non-differentiable utilities from base pysiglib
from ..sig_length import sig_length, log_sig_length
from ..words import words_of_length, words, lyndon_words_of_length, lyndon_words, is_lyndon, word_to_idx, idx_to_word
from ..trees import trees, trees_of_order, tree_to_idx, idx_to_tree
from ..sig_coef import extract_sig_coef
from ..log_sig import set_cache_dir, prepare_log_sig, clear_cache
from ..branched_sig import prepare_branched_sig, branched_sig_length
from ..branched_log_sig import branched_log_sig_length, prepare_branched_log_sig
from ..branched_sig_coef import extract_branched_sig_coef, prepare_branched_sig_coef
from ..load_siglib import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX

# ``Context`` / ``StaticKernel`` are shared with the base API - users writing
# custom kernels inherit from the same ABC in both environments.
from ..static_kernels import Context, StaticKernel

from ..streams import (
    SigStream as _SigStream,
    LogSigStream as _LogSigStream,
    SigWindowStream as _SigWindowStream,
    LogSigWindowStream as _LogSigWindowStream,
)


class SigStream(_SigStream):
    __doc__ = _SigStream.__doc__

    def __init__(self, dimension: int, degree: int, *, scalar_term: bool = False, n_jobs: int = 1):
        super().__init__(dimension, degree, scalar_term=scalar_term, n_jobs=n_jobs,
                         _sig_join=sig_join, _sig_combine=sig_combine, _sig=sig)


class LogSigStream(_LogSigStream):
    __doc__ = _LogSigStream.__doc__

    def __init__(self, dimension: int, degree: int, *, method: int = 2, n_jobs: int = 1):
        super().__init__(dimension, degree, method=method, n_jobs=n_jobs,
                         _log_sig_join=log_sig_join, _log_sig_combine=log_sig_combine,
                         _log_sig=log_sig)


class SigWindowStream(_SigWindowStream):
    __doc__ = _SigWindowStream.__doc__

    def __init__(self, dimension: int, degree: int, window_size: int,
                 *,
                 stride: int = 1, scalar_term: bool = False, n_jobs: int = 1):
        super().__init__(dimension, degree, window_size, stride=stride,
                         scalar_term=scalar_term, n_jobs=n_jobs, _sig=sig)


class LogSigWindowStream(_LogSigWindowStream):
    __doc__ = _LogSigWindowStream.__doc__

    def __init__(self, dimension: int, degree: int, window_size: int,
                 *,
                 stride: int = 1, method: int = 2, n_jobs: int = 1):
        super().__init__(dimension, degree, window_size, stride=stride,
                         method=method, n_jobs=n_jobs, _log_sig=log_sig)

__all__ = [
    "sig", "sig_combine", "transform_path",
    "sig_to_log_sig", "logsig_to_sig", "log_sig", "log_sig_combine",
    "sig_join", "log_sig_join", "linear_sig",
    "sig_kernel", "sig_kernel_gram",
    "branched_sig_kernel", "branched_sig_kernel_gram",
    "sig_coef", "branched_sig_coef",
    "sig_score", "expected_sig_score", "sig_mmd",
    "branched_sig", "branched_sig_combine",
    "branched_sig_to_log_sig", "branched_log_sig",
    "prepare_branched_sig", "prepare_branched_log_sig", "prepare_branched_sig_coef",
    "branched_sig_length", "branched_log_sig_length",
    "sig_length", "log_sig_length",
    "words_of_length", "words", "lyndon_words_of_length", "lyndon_words",
    "is_lyndon", "word_to_idx", "idx_to_word",
    "trees", "trees_of_order", "tree_to_idx", "idx_to_tree",
    "extract_sig_coef", "extract_branched_sig_coef",
    "set_cache_dir", "prepare_log_sig", "clear_cache",
    "Context", "StaticKernel",
    "LinearKernel", "ScaledLinearKernel", "RBFKernel",
    "PolynomialKernel", "Matern12Kernel", "Matern32Kernel", "Matern52Kernel",
    "RationalQuadraticKernel",
    "SigStream", "LogSigStream", "SigWindowStream", "LogSigWindowStream",
    "signature",
]

signature = sig
