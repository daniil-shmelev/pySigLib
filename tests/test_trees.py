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
import pytest

import pysiglib


def test_tree_native_order():
    # Root label is the innermost loop, after the child-ID sequence.
    assert pysiglib.trees(2, 2) == (
        (0,), (1,), ((0,), 0), ((0,), 1), ((1,), 0), ((1,), 1),
    )


def test_forest_native_order():
    assert pysiglib.forests(2, 2) == (
        ((0,),), ((1,),),
        ((0,), (0,)), ((0,), (1,)),
        ((1,), (0,)), ((1,), (1,)),
        (((0,), 0),), (((0,), 1),), (((1,), 0),), (((1,), 1),),
    )


def test_weighted_lyndon_forests():
    a, b = (0,), ((0,), 0)
    # b is a single letter of weight two. ab has weight three; aa is not Lyndon.
    assert pysiglib.lyndon_forests(1, 2) == ((a,), (b,))
    assert pysiglib.lyndon_forests(1, 3) == (
        (a,), (b,), (a, b), (((0,), (0,), 0),), ((((0,), 0), 0),),
    )
    assert pysiglib.forest_to_idx((b,), 1, 2) == 2
    assert pysiglib.lyndon_forest_to_idx((b,), 1, 2) == 1


def test_lyndon_uses_tree_ids_not_nested_tuple_order():
    # These letters have increasing native IDs; Python cannot compare their tuples.
    leaf, chain = (0,), ((0,), 0)
    assert (leaf, chain) in pysiglib.lyndon_forests(1, 3)
    assert (chain, leaf) not in pysiglib.lyndon_forests(1, 3)


@pytest.mark.parametrize("dimension,degree", [(1, 4), (2, 4), (3, 3)])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_exact_tree_orders(dimension, degree, scalar_term):
    reconstructed = tuple(
        tree
        for order in range(degree + 1)
        for tree in pysiglib.trees_of_order(dimension, order, scalar_term=scalar_term)
    )
    assert reconstructed == pysiglib.trees(dimension, degree, scalar_term=scalar_term)


@pytest.mark.parametrize("scalar_term", [False, True])
def test_planar_index_matches_planar_output(scalar_term):
    dimension, degree = 2, 3
    path = np.random.default_rng(7).normal(size=(8, dimension))
    pysiglib.prepare_branched_sig(dimension, degree, device="cpu")
    pysiglib.prepare_branched_sig(dimension, degree, planar=True, device="cpu")
    bck = pysiglib.branched_sig(path, degree, scalar_term=scalar_term)
    mkw = pysiglib.branched_sig(path, degree, planar=True, scalar_term=scalar_term)
    tree = (((0,), 0), 0)
    bck_idx = pysiglib.tree_to_idx(tree, dimension, degree, scalar_term=scalar_term)
    mkw_idx = pysiglib.forest_to_idx((tree,), dimension, degree, scalar_term=scalar_term)
    assert bck_idx != mkw_idx
    assert bck[bck_idx] == pytest.approx(mkw[mkw_idx])
    assert mkw[bck_idx] != pytest.approx(mkw[mkw_idx])


@pytest.mark.parametrize("family,element", [
    (pysiglib.tree_to_idx, ()),
    (pysiglib.tree_to_idx, (2,)),
    (pysiglib.tree_to_idx, ((1,), (0,), 0)),  # Noncanonical BCK child order.
    (pysiglib.forest_to_idx, (((2,),),)),
    (pysiglib.lyndon_forest_to_idx, ((0,), (0,))),
    (pysiglib.lyndon_forest_to_idx, ((1,), (0,))),
])
def test_invalid_basis_element(family, element):
    with pytest.raises((TypeError, ValueError)):
        family(element, 2, 3)


@pytest.mark.parametrize("family", [pysiglib.forest_to_idx, pysiglib.lyndon_forest_to_idx])
def test_forest_requires_explicit_tuple_of_trees(family):
    with pytest.raises(TypeError):
        family((0,), 2, 3)
    with pytest.raises(TypeError):
        family(None, 2, 3)


def test_empty_planar_forest_coefficient():
    path = np.array([[0.0], [1.0]])
    pysiglib.prepare_branched_sig_coef(1, [()], planar=True, device="cpu")
    np.testing.assert_array_equal(pysiglib.branched_sig_coef(path, [()], planar=True), [1.0])
    np.testing.assert_array_equal(
        pysiglib.extract_branched_sig_coef(np.array([1.0]), [()], 1, planar=True, scalar_term=True),
        [1.0],
    )
