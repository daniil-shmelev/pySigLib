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
from .torch_api import sig, sig_combine, sig_coef, transform_path, sig_to_log_sig, log_sig, log_sig_combine, logsig_to_sig, sig_kernel, sig_kernel_gram, sig_score, expected_sig_score, sig_mmd, branched_sig, prepare_branched_sig, branched_sig_length, branched_sig_combine

signature = sig
