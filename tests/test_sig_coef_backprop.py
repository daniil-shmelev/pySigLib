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

import pytest
import numpy as np
import torch
import pysiglib

EPSILON = 1e-10

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def test_sig_coef_backprop():
    X = np.random.uniform(size=(100, 3))
    word = (0,1,2)

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[pysiglib.word_to_idx(word, 3)] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, word, prefixes=True)
    derivs = np.array([0.,0.,1.])
    d2 = pysiglib.sig_coef_backprop(X, word, coef, derivs)

    check_close(d1, d2)

# def test_batch_sig_coef_full_time_aug():
#     X = np.random.uniform(size=(10, 100, 3))
#     multi_indices = pysiglib.words(4, 5)[1:]
#
#     coeff = pysiglib.sig_coef(X, multi_indices, time_aug = True)
#     sig = pysiglib.signature(X, 5, time_aug = True)
#     check_close(sig[:, 1:], coeff)
#
# def test_batch_sig_coef_full_lead_lag():
#     X = np.random.uniform(size=(10, 100, 3))
#     multi_indices = pysiglib.words(6, 5)[1:]
#
#     coeff = pysiglib.sig_coef(X, multi_indices, lead_lag = True)
#     sig = pysiglib.signature(X, 5, lead_lag = True)
#     check_close(sig[:, 1:], coeff)
#
# def test_batch_sig_coef_full_time_aug_lead_lag():
#     X = np.random.uniform(size=(10, 100, 3))
#     multi_indices = pysiglib.words(7, 5)[1:]
#
#     coeff = pysiglib.sig_coef(X, multi_indices, time_aug = True, lead_lag = True)
#     sig = pysiglib.signature(X, 5, time_aug = True, lead_lag = True)
#     check_close(sig[:, 1:], coeff)

