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
import torch

import pysiglib

EPSILON = 1e-5

from conftest import DEVICES, load_fixtures

FIXTURES = load_fixtures("reference_data.npz")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random(device, dyadic_order):
    X = torch.tensor(FIXTURES["ess_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["ess_Y"], device=device, dtype=torch.double)
    expected = float(FIXTURES[f"ess_linear__do{dyadic_order}"])

    d2 = float(pysiglib.expected_sig_score(X, Y, dyadic_order=dyadic_order))

    assert not abs(expected - d2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random_rbf(device, dyadic_order):
    X = torch.tensor(FIXTURES["ess_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["ess_Y"], device=device, dtype=torch.double)
    expected = float(FIXTURES[f"ess_rbf__do{dyadic_order}"])

    d2 = float(pysiglib.expected_sig_score(X, Y, dyadic_order=dyadic_order, static_kernel=pysiglib.RBFKernel(2.)))

    assert not abs(expected - d2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(10, 20), (20, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_expected_sig_score_random_non_square(device, len1, len2, dyadic_order):
    X = torch.tensor(FIXTURES[f"ess_nonsq_{len1}_{len2}__do{dyadic_order}_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES[f"ess_nonsq_{len1}_{len2}__do{dyadic_order}_Y"], device=device, dtype=torch.double)
    expected = float(FIXTURES[f"ess_nonsq_{len1}_{len2}__do{dyadic_order}"])

    d2 = float(pysiglib.expected_sig_score(X, Y, dyadic_order=dyadic_order))

    assert not abs(expected - d2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random(device, dyadic_order):
    X = torch.tensor(FIXTURES["ess_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["ess_Y"], device=device, dtype=torch.double)
    expected = float(FIXTURES[f"mmd_linear__do{dyadic_order}"])

    mmd2 = float(pysiglib.sig_mmd(X, Y, dyadic_order=dyadic_order))

    assert not abs(expected - mmd2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random_rbf(device, dyadic_order):
    X = torch.tensor(FIXTURES["ess_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES["ess_Y"], device=device, dtype=torch.double)
    expected = float(FIXTURES[f"mmd_rbf__do{dyadic_order}"])

    mmd2 = float(pysiglib.sig_mmd(X, Y, dyadic_order=dyadic_order, static_kernel=pysiglib.RBFKernel(2.)))

    assert not abs(expected - mmd2) > EPSILON

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("len1", "len2"), [(10, 20), (20, 10)])
@pytest.mark.parametrize("dyadic_order", range(3))
def test_sig_mmd_random_non_square(device, len1, len2, dyadic_order):
    X = torch.tensor(FIXTURES[f"ess_nonsq_{len1}_{len2}__do{dyadic_order}_X"], device=device, dtype=torch.double)
    Y = torch.tensor(FIXTURES[f"ess_nonsq_{len1}_{len2}__do{dyadic_order}_Y"], device=device, dtype=torch.double)
    expected = float(FIXTURES[f"mmd_nonsq_{len1}_{len2}__do{dyadic_order}"])

    mmd2 = float(pysiglib.sig_mmd(X, Y, dyadic_order=dyadic_order))

    assert not abs(expected - mmd2) > EPSILON
