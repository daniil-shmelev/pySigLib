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

import numpy as np
import torch
import pysiglib

EPSILON = 1e-10

def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)

def test_extract_sig_coef_all():
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension))
    sig = pysiglib.sig(x, degree)
    words = pysiglib.words(dimension, degree)
    coefs = pysiglib.extract_sig_coef(words, sig, dimension, degree)
    check_close(sig, coefs)

def test_extract_sig_coef_lyndon():
    dimension, degree = 3, 4
    x = torch.rand(size=(100, dimension))
    pysiglib.prepare_log_sig(dimension, degree, method=1)
    log_sig_full = pysiglib.log_sig(x, degree, method=0)
    log_sig = pysiglib.log_sig(x, degree, method=1)
    words = pysiglib.lyndon_words(dimension, degree)
    coefs = pysiglib.extract_sig_coef(words, log_sig_full, dimension, degree)
    check_close(log_sig, coefs)
