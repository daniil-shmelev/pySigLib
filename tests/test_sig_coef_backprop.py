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

def test_sig_coef_backprop_batch():
    X = np.random.uniform(size=(32, 100, 3))
    words = [(0,1,2), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.array([[0.,0.,1., 0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_batch_2():
    X = np.random.uniform(size=(32, 100, 3))
    words = [(2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.array([[0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_batch_3():
    X = np.random.uniform(size=(32, 100, 3))
    words = [tuple(), (2, 0, 1)]
    words_idx = [pysiglib.word_to_idx(w, 3) for w in words]

    sig = pysiglib.sig(X, 3)
    derivs = np.zeros_like(sig)
    derivs[:, words_idx] = 1.
    d1 = pysiglib.sig_backprop(X, sig, derivs, 3)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.array([[1., 0., 0., 1.]]*32)
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

def test_sig_coef_backprop_full_batch():
    X = np.random.uniform(size=(1, 2, 2))
    words = pysiglib.words(2, 2)

    sig = pysiglib.sig(X, 2)
    derivs = np.ones(sig.shape)
    d1 = pysiglib.sig_backprop(X, sig, derivs, 2)

    coef = pysiglib.sig_coef(X, words, prefixes=True)
    derivs = np.zeros_like(coef)
    i = 0
    for w in words:
        i += len(w)
        derivs[:, i] = 1.
    d2 = pysiglib.sig_coef_backprop(X, words, coef, derivs)

    check_close(d1, d2)

