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
import native_api as pysiglib


PARAMS = [(2, 3), (2, 4), (3, 3), (3, 2), (4, 2)]


class TestRoundTrip:
    @pytest.mark.parametrize("dim, deg", PARAMS)
    def test_idx_to_tree_to_idx(self, dim, deg):
        bsig_len = pysiglib.branched_sig_length(dim, deg)
        for i in range(bsig_len):
            tree = pysiglib.idx_to_tree(i, dim, deg)
            assert pysiglib.tree_to_idx(tree, dim, deg) == i

    @pytest.mark.parametrize("dim, deg", PARAMS)
    def test_planar_idx_to_tree_to_idx(self, dim, deg):
        bsig_len = pysiglib.branched_sig_length(dim, deg, planar=True)
        for i in range(bsig_len):
            tree = pysiglib.idx_to_tree(i, dim, deg, planar=True)
            assert pysiglib.tree_to_idx(tree, dim, deg, planar=True) == i

    @pytest.mark.parametrize("dim, deg", PARAMS)
    def test_tree_to_idx_to_tree(self, dim, deg):
        # ``trees(...)`` always includes the empty tree (None) at index 0,
        # so use scalar_term=True to make the indexing consistent.
        all_trees = pysiglib.trees(dim, deg)
        for tree in all_trees:
            idx = pysiglib.tree_to_idx(tree, dim, deg, scalar_term=True)
            assert pysiglib.idx_to_tree(idx, dim, deg, scalar_term=True) == tree


class TestPlanarOrderedForestBasis:
    def test_planar_trees_returns_ordered_forests(self):
        basis = pysiglib.trees(2, 2, planar=True)
        assert ((0,),) in basis
        assert (0,) not in basis
        assert ((0,), (1,)) in basis
        assert ((1,), (0,)) in basis

    def test_planar_idx_to_tree_returns_ordered_forest(self):
        assert pysiglib.idx_to_tree(0, 2, 2, planar=True) == ((0,),)

        forest = ((0,), (1,))
        idx = pysiglib.tree_to_idx(forest, 2, 2, planar=True)
        assert pysiglib.idx_to_tree(idx, 2, 2, planar=True) == forest

    def test_planar_tree_to_idx_accepts_single_tree_shorthand(self):
        assert (
            pysiglib.tree_to_idx((0,), 2, 2, planar=True)
            == pysiglib.tree_to_idx(((0,),), 2, 2, planar=True)
        )


class TestLengthMatch:
    @pytest.mark.parametrize("dim, deg", PARAMS)
    def test_trees_length_matches_branched_sig_length(self, dim, deg):
        # ``trees(...)`` always includes the empty tree, so its length matches
        # branched_sig_length with scalar_term=True.
        assert len(pysiglib.trees(dim, deg)) == pysiglib.branched_sig_length(dim, deg, scalar_term=True)

    @pytest.mark.parametrize("dim, deg", PARAMS)
    def test_trees_of_order_sum(self, dim, deg):
        total = 1  # empty tree
        for order in range(1, deg + 1):
            total += len(pysiglib.trees_of_order(dim, order))
        assert total == pysiglib.branched_sig_length(dim, deg, scalar_term=True)


class TestEmptyTree:
    def test_empty_tree_at_idx_0(self):
        assert pysiglib.idx_to_tree(0, 2, 3, scalar_term=True) is None

    def test_tree_to_idx_empty(self):
        assert pysiglib.tree_to_idx(None, 2, 3, scalar_term=True) == 0

    def test_trees_of_order_0(self):
        assert pysiglib.trees_of_order(2, 0) == (None,)


class TestScalarTermFlag:
    @pytest.mark.parametrize("dim, deg", [(2, 3), (3, 2)])
    def test_index_shift_between_scalar_term_formats(self, dim, deg):
        # For scalar_term=True, None is at 0 and non-empty trees shift up by 1
        # relative to scalar_term=False indices.
        pysiglib.prepare_branched_sig(dim, deg)
        non_empty = [t for t in pysiglib.trees(dim, deg) if t is not None]
        for t in non_empty:
            idx_false = pysiglib.tree_to_idx(t, dim, deg, scalar_term=False)
            idx_true = pysiglib.tree_to_idx(t, dim, deg, scalar_term=True)
            assert idx_true == idx_false + 1
            assert pysiglib.idx_to_tree(idx_false, dim, deg, scalar_term=False) == t
            assert pysiglib.idx_to_tree(idx_true, dim, deg, scalar_term=True) == t

    def test_empty_tree_requires_scalar_term(self):
        assert pysiglib.tree_to_idx(None, 2, 3, scalar_term=True) == 0
        with pytest.raises(ValueError):
            pysiglib.tree_to_idx(None, 2, 3, scalar_term=False)


class TestSingleNodeTrees:
    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_order_1_trees(self, dim):
        t = pysiglib.trees_of_order(dim, 1)
        assert len(t) == dim
        for label in range(dim):
            assert t[label] == (label,)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_single_node_indices(self, dim):
        # With scalar_term=False (default), single-node tree (label,) is at idx label.
        for label in range(dim):
            assert pysiglib.tree_to_idx((label,), dim, 3) == label


class TestCoefficientExtraction:
    @pytest.mark.parametrize("dim, deg", [(2, 3), (3, 3)])
    def test_single_node_matches_sig_level1(self, dim, deg):
        rng = np.random.default_rng(42)
        path = rng.standard_normal((20, dim))

        pysiglib.prepare_branched_sig(dim, deg)
        bsig = pysiglib.branched_sig(path, deg)
        sig = pysiglib.sig(path, 1)

        for label in range(dim):
            idx = pysiglib.tree_to_idx((label,), dim, deg)
            np.testing.assert_allclose(bsig[idx], sig[label], atol=1e-10)

    def test_planar_index_matches_planar_output(self):
        dim, deg = 2, 3
        rng = np.random.default_rng(7)
        path = rng.standard_normal((8, dim)).astype(np.float64)

        pysiglib.prepare_branched_sig(dim, deg, planar=False)
        bsig_nonplanar = pysiglib.branched_sig(path, deg, planar=False)
        pysiglib.prepare_branched_sig(dim, deg, planar=True)
        bsig_planar = pysiglib.branched_sig(path, deg, planar=True)

        tree = (((0,), 0), 0)
        forest = (tree,)
        idx_nonplanar = pysiglib.tree_to_idx(tree, dim, deg, planar=False)
        idx_planar = pysiglib.tree_to_idx(forest, dim, deg, planar=True)

        # scalar_term=False default: lengths and indices shift down by 1 vs the
        # scalar_term=True convention.
        assert len(bsig_nonplanar) == 20
        assert len(bsig_planar) == pysiglib.branched_sig_length(dim, deg, planar=True)
        assert idx_planar != idx_nonplanar
        assert pysiglib.idx_to_tree(idx_planar, dim, deg, planar=True) == forest
        assert pysiglib.idx_to_tree(idx_nonplanar, dim, deg, planar=False) == tree
        assert bsig_planar[idx_planar] == pytest.approx(bsig_nonplanar[idx_nonplanar])
        assert bsig_planar[idx_nonplanar] != pytest.approx(bsig_nonplanar[idx_nonplanar])


class TestValidation:
    def test_idx_out_of_range(self):
        bsig_len = pysiglib.branched_sig_length(2, 3)
        with pytest.raises(ValueError):
            pysiglib.idx_to_tree(bsig_len, 2, 3)

class TestImplementationOrder:
    @pytest.mark.parametrize("dim, deg", PARAMS)
    @pytest.mark.parametrize("planar", [False, True])
    def test_idx_to_tree_to_idx_roundtrip(self, dim, deg, planar):
        bsig_len = pysiglib.branched_sig_length(dim, deg, planar=planar)
        for i in range(bsig_len):
            tree = pysiglib.idx_to_tree(i, dim, deg, planar=planar)
            assert pysiglib.tree_to_idx(tree, dim, deg, planar=planar) == i

    @pytest.mark.parametrize("dim, deg", PARAMS)
    @pytest.mark.parametrize("planar", [False, True])
    def test_trees_enumeration_matches_tree_to_idx(self, dim, deg, planar):
        # ``trees(...)`` always includes the empty tree at index 0, matching the
        # scalar_term=True indexing convention.
        all_trees = pysiglib.trees(dim, deg, planar=planar)
        assert len(all_trees) == pysiglib.branched_sig_length(dim, deg, planar=planar, scalar_term=True)
        assert all_trees[0] is None
        for i, tree in enumerate(all_trees):
            assert pysiglib.tree_to_idx(tree, dim, deg, planar=planar, scalar_term=True) == i

    @pytest.mark.parametrize("dim, deg", PARAMS)
    @pytest.mark.parametrize("planar", [False, True])
    def test_trees_of_order_consistency_with_trees(self, dim, deg, planar):
        """Concatenating trees_of_order per stratum reproduces trees()."""
        reconstructed = []
        for order in range(deg + 1):
            reconstructed.extend(
                pysiglib.trees_of_order(dim, order, planar=planar)
            )
        assert tuple(reconstructed) == pysiglib.trees(dim, deg, planar=planar)
