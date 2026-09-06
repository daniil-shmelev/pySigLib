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

from .load_siglib import (
    SYSTEM as SYSTEM,
    BUILT_WITH_CUDA as BUILT_WITH_CUDA,
    BUILT_WITH_AVX as BUILT_WITH_AVX,
    BUILT_WITH_JAX_FFI as BUILT_WITH_JAX_FFI,
)
from .words import (
    words_of_length as words_of_length,
    words as words,
    lyndon_words_of_length as lyndon_words_of_length,
    lyndon_words as lyndon_words,
    is_lyndon as is_lyndon,
    word_to_idx as word_to_idx,
    idx_to_word as idx_to_word,
)
from .trees import (
    trees as trees,
    trees_of_order as trees_of_order,
    tree_to_idx as tree_to_idx,
    idx_to_tree as idx_to_tree,
)
from .sig_length import sig_length as sig_length, log_sig_length as log_sig_length
from .sig import sig_combine as sig_combine, sig as sig
from .sig_backprop import (
    sig_backprop as sig_backprop,
    sig_combine_backprop as sig_combine_backprop,
)
from .linear_sig import linear_sig as linear_sig
from .sig_join import sig_join as sig_join
from .sig_join_backprop import sig_join_backprop as sig_join_backprop
from .log_sig_join import log_sig_join as log_sig_join
from .log_sig_join_backprop import log_sig_join_backprop as log_sig_join_backprop
from .sig_coef import extract_sig_coef as extract_sig_coef, sig_coef as sig_coef
from .sig_coef_backprop import sig_coef_backprop as sig_coef_backprop
from .log_sig import (
    set_cache_dir as set_cache_dir,
    prepare_log_sig as prepare_log_sig,
    clear_cache as clear_cache,
    sig_to_log_sig as sig_to_log_sig,
    log_sig as log_sig,
)
from .log_sig_backprop import sig_to_log_sig_backprop as sig_to_log_sig_backprop
from .logsig_to_sig import logsig_to_sig as logsig_to_sig
from .logsig_to_sig_backprop import logsig_to_sig_backprop as logsig_to_sig_backprop
from .log_sig_combine import (
    log_sig_combine as log_sig_combine,
    log_sig_combine_backprop as log_sig_combine_backprop,
)
from .sig_kernel import sig_kernel as sig_kernel, sig_kernel_gram as sig_kernel_gram
from .sig_kernel_backprop import (
    sig_kernel_backprop as sig_kernel_backprop,
    sig_kernel_gram_backprop as sig_kernel_gram_backprop,
)
from .branched_sig_kernel import (
    branched_sig_kernel as branched_sig_kernel,
    branched_sig_kernel_gram as branched_sig_kernel_gram,
)
from .branched_sig_kernel_backprop import (
    branched_sig_kernel_backprop as branched_sig_kernel_backprop,
    branched_sig_kernel_gram_backprop as branched_sig_kernel_gram_backprop,
)
from .sig_metrics import (
    sig_score as sig_score,
    expected_sig_score as expected_sig_score,
    sig_mmd as sig_mmd,
)
from .transform_path import transform_path as transform_path
from .transform_path_backprop import transform_path_backprop as transform_path_backprop
from .static_kernels import (
    Context as Context,
    StaticKernel as StaticKernel,
    LinearKernel as LinearKernel,
    ScaledLinearKernel as ScaledLinearKernel,
    RBFKernel as RBFKernel,
    PolynomialKernel as PolynomialKernel,
    Matern12Kernel as Matern12Kernel,
    Matern32Kernel as Matern32Kernel,
    Matern52Kernel as Matern52Kernel,
    RationalQuadraticKernel as RationalQuadraticKernel,
)
from .branched_sig import (
    prepare_branched_sig as prepare_branched_sig,
    branched_sig as branched_sig,
    branched_sig_combine as branched_sig_combine,
    branched_sig_length as branched_sig_length,
)
from .branched_sig_backprop import (
    branched_sig_backprop as branched_sig_backprop,
    branched_sig_combine_backprop as branched_sig_combine_backprop,
)
from .branched_sig_coef import (
    extract_branched_sig_coef as extract_branched_sig_coef,
    prepare_branched_sig_coef as prepare_branched_sig_coef,
    branched_sig_coef as branched_sig_coef,
)
from .branched_sig_coef_backprop import (
    branched_sig_coef_backprop as branched_sig_coef_backprop,
)
from .branched_log_sig import (
    branched_log_sig as branched_log_sig,
    branched_log_sig_length as branched_log_sig_length,
    branched_sig_to_log_sig as branched_sig_to_log_sig,
    prepare_branched_log_sig as prepare_branched_log_sig,
)
from .branched_log_sig_backprop import (
    branched_sig_to_log_sig_backprop as branched_sig_to_log_sig_backprop,
)
from .streams import (
    SigStream as SigStream,
    LogSigStream as LogSigStream,
    SigWindowStream as SigWindowStream,
    LogSigWindowStream as LogSigWindowStream,
)

signature = sig


from ._version import __version__
