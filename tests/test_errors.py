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

import pytest
import numpy as np
import torch

import pysiglib

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-10

@pytest.mark.parametrize("dim, deg", [(-1, 2), (1, -2)])
def test_sig_length_value_error(dim, deg):
    with pytest.raises(ValueError):
        pysiglib.sig_length(dim, deg)


@pytest.mark.parametrize("args", [
    ('a', 2, False, False, False, False),
    (np.array(['a', 'b']), 2, False, False, False, False),
    (np.array([[0.], [1.]]), 'a', False, False, False, False),
    (np.array([[0.], [1.]]), 2, 'a', False, False, False),
    (np.array([[0.], [1.]]), 2, False, 'a', False, False),
    (np.array([[0.], [1.]]), 2, False, False, 'a', False),
    (np.array([[[0.], [1.]]]), 2, False, False, False, 'a'),
])
def test_signature_type_error(args):
    with pytest.raises(TypeError):
        pysiglib.sig(*args)


@pytest.mark.parametrize("X, deg", [
    (np.array([0., 1.]), 2),
    (np.array([[0.], [1.]]), -1),
])
def test_signature_value_error(X, deg):
    with pytest.raises(ValueError):
        pysiglib.sig(X, deg)


@pytest.mark.parametrize("x, y, d", [
    ('a', np.array([[0.], [1.]]), 2),
    (np.array([[0.], [1.]]), 'a', 2),
    (np.array([[0.], [1.]]), np.array([[0.], [1.]]), 'a'),
])
def test_sig_kernel_type_error(x, y, d):
    with pytest.raises(TypeError):
        pysiglib.sig_kernel(x, y, dyadic_order=d)


@pytest.mark.parametrize("x, y, d", [
    (np.array([0., 1.]), np.array([[0.], [1.]]), 2),
    (np.array([[0.], [1.]]), np.array([0., 1.]), 2),
    (np.array([[[[0.]]], [[[1.]]]]), np.array([[0.], [1.]]), 2),
    (np.array([[0.], [1.]]), np.array([[[[0.]]], [[[1.]]]]), 2),
    (np.array([[0.], [1.]]), np.array([[0.], [1.]]), -2),
])
def test_sig_kernel_value_error(x, y, d):
    with pytest.raises(ValueError):
        pysiglib.sig_kernel(x, y, dyadic_order=d)

def test_signature_n_jobs_zero():
    with pytest.raises(ValueError):
        pysiglib.sig(np.array([[[0.], [1.]]]), 2, n_jobs = 0)

def test_sig_combine_n_jobs_zero():
    sig = pysiglib.sig(np.array([[0.], [1.]]), 2)
    with pytest.raises(ValueError):
        pysiglib.sig_combine(sig, sig, 1, 2, n_jobs = 0)

def test_sig_kernel_n_jobs_zero():
    with pytest.raises(ValueError):
        pysiglib.sig_kernel(np.array([[0.], [1.]]), np.array([[0.], [1.]]), dyadic_order=0, n_jobs = 0)
