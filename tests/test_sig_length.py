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

import numpy as np
import torch
import kauri

import pysiglib

np.random.seed(42)
torch.manual_seed(42)

def test_sig_length():
    assert pysiglib.sig_length(0, 0) == 0
    assert pysiglib.sig_length(0, 1) == 0
    assert pysiglib.sig_length(1, 0) == 0
    assert pysiglib.sig_length(9, 9) == 435848049
    assert pysiglib.sig_length(10, 10) == 11111111110
    assert pysiglib.sig_length(11, 11) == 313842837671
    assert pysiglib.sig_length(400, 5) == 10265664160400

def test_log_sig_length():
    assert pysiglib.log_sig_length(2, 3) == 5
    assert pysiglib.log_sig_length(9, 9) == 49212093
    assert pysiglib.log_sig_length(10, 10) == 1125217654
    assert pysiglib.log_sig_length(5, 12) == 26039187

def test_branched_sig_length():
    assert pysiglib.branched_sig_length(1, 1) == 1
    assert pysiglib.branched_sig_length(2, 1) == 2
    assert pysiglib.branched_sig_length(2, 2) == 6
    assert pysiglib.branched_sig_length(2, 3) == 20
    assert pysiglib.branched_sig_length(2, 4) == 72
    assert pysiglib.branched_sig_length(3, 3) == 57
    assert pysiglib.branched_sig_length(5, 5) == 19880

def test_branched_sig_length_scalar_term():
    assert pysiglib.branched_sig_length(0, 0, scalar_term=True) == 1
    assert pysiglib.branched_sig_length(0, 1, scalar_term=True) == 1
    assert pysiglib.branched_sig_length(1, 0, scalar_term=True) == 1
    assert pysiglib.branched_sig_length(1, 1, scalar_term=True) == 2
    assert pysiglib.branched_sig_length(2, 1, scalar_term=True) == 3
    assert pysiglib.branched_sig_length(2, 2, scalar_term=True) == 7
    assert pysiglib.branched_sig_length(2, 3, scalar_term=True) == 21
    assert pysiglib.branched_sig_length(2, 4, scalar_term=True) == 73
    assert pysiglib.branched_sig_length(3, 3, scalar_term=True) == 58
    assert pysiglib.branched_sig_length(5, 5, scalar_term=True) == 19881

def test_branched_sig_length_matches_bck_tree_basis():
    for dim, deg in [(1, 3), (2, 3), (3, 2)]:
        expected = len(kauri.colored_trees(dim, deg))
        assert pysiglib.branched_sig_length(dim, deg, scalar_term=True) == expected
        assert pysiglib.branched_sig_length(dim, deg) == expected - 1

def test_branched_sig_length_planar():
    assert pysiglib.branched_sig_length(2, 3, planar=True) == 50
    assert pysiglib.branched_sig_length(2, 4, planar=True) == 274
    assert pysiglib.branched_sig_length(3, 3, planar=True) == 156

def test_branched_sig_length_planar_scalar_term():
    assert pysiglib.branched_sig_length(2, 3, planar=True, scalar_term=True) == 51
    assert pysiglib.branched_sig_length(2, 4, planar=True, scalar_term=True) == 275
    assert pysiglib.branched_sig_length(3, 3, planar=True, scalar_term=True) == 157

def test_planar_branched_sig_length_matches_mkw_ordered_forest_basis():
    for dim, deg in [(1, 3), (2, 3), (3, 2)]:
        expected = len(kauri.colored_ordered_forests(dim, deg))
        assert pysiglib.branched_sig_length(dim, deg, planar=True, scalar_term=True) == expected
        assert pysiglib.branched_sig_length(dim, deg, planar=True) == expected - 1

    assert len(kauri.colored_ordered_forests(2, 3)) > len(kauri.colored_trees(2, 3))
