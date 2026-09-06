"""Shared native API used by tests parametrized over NumPy and Torch storage.

Public namespace contracts are tested separately in test_backend_apis.py.
"""
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

from pysiglib.load_siglib import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX, BUILT_WITH_JAX_FFI
from pysiglib.words import words_of_length, words, lyndon_words_of_length, lyndon_words, is_lyndon, word_to_idx, idx_to_word
from pysiglib.trees import trees, trees_of_order, tree_to_idx, idx_to_tree
from pysiglib.sig_length import sig_length, log_sig_length
from pysiglib._core.sig import sig_combine, sig
from pysiglib._core.sig_backprop import sig_backprop, sig_combine_backprop
from pysiglib._core.linear_sig import linear_sig
from pysiglib._core.sig_join import sig_join
from pysiglib._core.sig_join_backprop import sig_join_backprop
from pysiglib._core.log_sig_join import log_sig_join
from pysiglib._core.log_sig_join_backprop import log_sig_join_backprop
from pysiglib._core.sig_coef import extract_sig_coef, sig_coef
from pysiglib._core.sig_coef_backprop import sig_coef_backprop
from pysiglib._core.log_sig import set_cache_dir, prepare_log_sig, clear_cache, sig_to_log_sig, log_sig
from pysiglib._core.log_sig_backprop import sig_to_log_sig_backprop
from pysiglib._core.logsig_to_sig import logsig_to_sig
from pysiglib._core.logsig_to_sig_backprop import logsig_to_sig_backprop
from pysiglib._core.log_sig_combine import log_sig_combine, log_sig_combine_backprop
from pysiglib._core.sig_kernel import sig_kernel, sig_kernel_gram
from pysiglib._core.sig_kernel_backprop import sig_kernel_backprop, sig_kernel_gram_backprop
from pysiglib._core.branched_sig_kernel import branched_sig_kernel, branched_sig_kernel_gram
from pysiglib._core.branched_sig_kernel_backprop import branched_sig_kernel_backprop, branched_sig_kernel_gram_backprop
from pysiglib._core.sig_metrics import sig_score, expected_sig_score, sig_mmd
from pysiglib._core.transform_path import transform_path
from pysiglib._core.transform_path_backprop import transform_path_backprop
from pysiglib._core.static_kernels import Context, StaticKernel, LinearKernel, ScaledLinearKernel, RBFKernel, PolynomialKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel, RationalQuadraticKernel
from pysiglib._core.branched_sig import prepare_branched_sig, branched_sig, branched_sig_combine, branched_sig_length
from pysiglib._core.branched_sig_backprop import branched_sig_backprop, branched_sig_combine_backprop
from pysiglib._core.branched_sig_coef import extract_branched_sig_coef, prepare_branched_sig_coef, branched_sig_coef
from pysiglib._core.branched_sig_coef_backprop import branched_sig_coef_backprop
from pysiglib._core.branched_log_sig import (branched_log_sig, branched_log_sig_length,
                               branched_sig_to_log_sig, prepare_branched_log_sig)
from pysiglib._core.branched_log_sig_backprop import branched_sig_to_log_sig_backprop
from pysiglib._core.streams import SigStream, LogSigStream, SigWindowStream, LogSigWindowStream

signature = sig

import pysiglib.torch_api as torch_api

from pysiglib._version import __version__

from pysiglib import load_siglib
