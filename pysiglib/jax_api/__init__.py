from __future__ import annotations
from .._core._docs import backend_doc as _backend_doc
import jax
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
    sig as sig,
    sig_combine as sig_combine,
    transform_path as transform_path,
    sig_to_log_sig as sig_to_log_sig,
    logsig_to_sig as logsig_to_sig,
    log_sig as log_sig,
    log_sig_combine as log_sig_combine,
    sig_join as sig_join,
    log_sig_join as log_sig_join,
    linear_sig as linear_sig,
    sig_coef as sig_coef,
    branched_sig_coef as branched_sig_coef,
    sig_kernel as sig_kernel,
    sig_kernel_gram as sig_kernel_gram,
    branched_sig_kernel as branched_sig_kernel,
    branched_sig_kernel_gram as branched_sig_kernel_gram,
    sig_score as sig_score,
    expected_sig_score as expected_sig_score,
    sig_mmd as sig_mmd,
    branched_sig as branched_sig,
    branched_sig_combine as branched_sig_combine,
    branched_sig_to_log_sig as branched_sig_to_log_sig,
    branched_log_sig as branched_log_sig,
)
from .static_kernels_jax import (
    LinearKernel as LinearKernel,
    ScaledLinearKernel as ScaledLinearKernel,
    RBFKernel as RBFKernel,
    PolynomialKernel as PolynomialKernel,
    Matern12Kernel as Matern12Kernel,
    Matern32Kernel as Matern32Kernel,
    Matern52Kernel as Matern52Kernel,
    RationalQuadraticKernel as RationalQuadraticKernel,
)

# Re-export non-differentiable utilities from base pysiglib
from ..sig_length import sig_length as sig_length, log_sig_length as log_sig_length
from ..words import (
    words_of_length as words_of_length,
    words as words,
    lyndon_words_of_length as lyndon_words_of_length,
    lyndon_words as lyndon_words,
    is_lyndon as is_lyndon,
    word_to_idx as word_to_idx,
    idx_to_word as idx_to_word,
)
from ..trees import (
    trees as trees,
    trees_of_order as trees_of_order,
    tree_to_idx as tree_to_idx,
    idx_to_tree as idx_to_tree,
)
from .._core.log_sig import (
    set_cache_dir as set_cache_dir,
    prepare_log_sig as prepare_log_sig,
    clear_cache as clear_cache,
)
from .._core.branched_sig import (
    prepare_branched_sig as prepare_branched_sig,
    branched_sig_length as branched_sig_length,
)
from .._core.branched_log_sig import (
    branched_log_sig_length as branched_log_sig_length,
    prepare_branched_log_sig as prepare_branched_log_sig,
)
from .._core.branched_sig_coef import (
    prepare_branched_sig_coef as prepare_branched_sig_coef,
)
from ..load_siglib import (
    SYSTEM as SYSTEM,
    BUILT_WITH_CUDA as BUILT_WITH_CUDA,
    BUILT_WITH_AVX as BUILT_WITH_AVX,
)

# ``Context`` / ``StaticKernel`` are shared with the base API - users writing
# custom kernels inherit from the same ABC in both environments.
from .static_kernels_jax import Context as Context, StaticKernel as StaticKernel

from .._core.streams import (
    SigStream as _SigStream,
    LogSigStream as _LogSigStream,
    SigWindowStream as _SigWindowStream,
    LogSigWindowStream as _LogSigWindowStream,
)


class SigStream(_SigStream[jax.Array]):
    _array_type = jax.Array
    __doc__ = _backend_doc(_SigStream.__doc__, "jax")

    def __init__(self, dimension: int, degree: int, *, scalar_term: bool = False, n_jobs: int = 1):
        super().__init__(dimension, degree, scalar_term=scalar_term, n_jobs=n_jobs,
                         _sig_join=sig_join, _sig_combine=sig_combine, _sig=sig)


class LogSigStream(_LogSigStream[jax.Array]):
    _array_type = jax.Array
    __doc__ = _backend_doc(_LogSigStream.__doc__, "jax")

    def __init__(self, dimension: int, degree: int, *, method: int = 2, n_jobs: int = 1):
        super().__init__(dimension, degree, method=method, n_jobs=n_jobs,
                         _log_sig_join=log_sig_join, _log_sig_combine=log_sig_combine,
                         _log_sig=log_sig)


class SigWindowStream(_SigWindowStream[jax.Array]):
    _array_type = jax.Array
    __doc__ = _backend_doc(_SigWindowStream.__doc__, "jax")

    def __init__(self, dimension: int, degree: int, window_size: int,
                 *,
                 stride: int = 1, scalar_term: bool = False, n_jobs: int = 1):
        super().__init__(dimension, degree, window_size, stride=stride,
                         scalar_term=scalar_term, n_jobs=n_jobs, _sig=sig)


class LogSigWindowStream(_LogSigWindowStream[jax.Array]):
    _array_type = jax.Array
    __doc__ = _backend_doc(_LogSigWindowStream.__doc__, "jax")

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

from .._core import sig_coef as _extract_sig_coef
from .._core._array import require_array as _require_array
from typing import Union as Union, Optional as Optional


def extract_sig_coef(
    sig: jax.Array,
    words: Union[tuple[int, ...], list[tuple[int, ...]]],
    dimension: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    scalar_term: bool = False,
) -> jax.Array:
    _require_array(sig, jax.Array, "sig")
    return _extract_sig_coef.extract_sig_coef(
        sig,
        words,
        dimension,
        time_aug=time_aug,
        lead_lag=lead_lag,
        scalar_term=scalar_term,
    )


extract_sig_coef.__doc__ = _backend_doc(
    _extract_sig_coef.extract_sig_coef.__doc__, "jax"
)

from .._core import branched_sig_coef as _extract_branched_sig_coef


def extract_branched_sig_coef(
    bsig: jax.Array,
    trees,
    dimension: int,
    *,
    time_aug: bool = False,
    lead_lag: bool = False,
    planar: bool = False,
    scalar_term: bool = False,
) -> jax.Array:
    _require_array(bsig, jax.Array, "bsig")
    return _extract_branched_sig_coef.extract_branched_sig_coef(
        bsig,
        trees,
        dimension,
        time_aug=time_aug,
        lead_lag=lead_lag,
        planar=planar,
        scalar_term=scalar_term,
    )


extract_branched_sig_coef.__doc__ = _backend_doc(
    _extract_branched_sig_coef.extract_branched_sig_coef.__doc__, "jax"
)
