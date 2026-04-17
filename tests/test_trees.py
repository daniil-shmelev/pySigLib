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
        all_trees = pysiglib.trees(dim, deg)
        for tree in all_trees:
            idx = pysiglib.tree_to_idx(tree, dim, deg)
            assert pysiglib.idx_to_tree(idx, dim, deg) == tree


class TestLengthMatch:
    @pytest.mark.parametrize("dim, deg", PARAMS)
    def test_trees_length_matches_branched_sig_length(self, dim, deg):
        assert len(pysiglib.trees(dim, deg)) == pysiglib.branched_sig_length(dim, deg)

    @pytest.mark.parametrize("dim, deg", PARAMS)
    def test_trees_of_order_sum(self, dim, deg):
        total = 1  # empty tree
        for order in range(1, deg + 1):
            total += len(pysiglib.trees_of_order(dim, order))
        assert total == pysiglib.branched_sig_length(dim, deg)


class TestEmptyTree:
    def test_empty_tree_at_idx_0(self):
        assert pysiglib.idx_to_tree(0, 2, 3) is None

    def test_tree_to_idx_empty(self):
        assert pysiglib.tree_to_idx(None, 2, 3) == 0

    def test_trees_of_order_0(self):
        assert pysiglib.trees_of_order(2, 0) == (None,)


class TestSingleNodeTrees:
    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_order_1_trees(self, dim):
        t = pysiglib.trees_of_order(dim, 1)
        assert len(t) == dim
        for label in range(dim):
            assert t[label] == (label,)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_single_node_indices(self, dim):
        for label in range(dim):
            assert pysiglib.tree_to_idx((label,), dim, 3) == label + 1


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
            np.testing.assert_allclose(bsig[idx], sig[label + 1], atol=1e-10)

    def test_planar_canonical_index_matches_planar_output(self):
        dim, deg = 2, 3
        rng = np.random.default_rng(7)
        path = rng.standard_normal((8, dim)).astype(np.float64)

        pysiglib.prepare_branched_sig(dim, deg, planar=False)
        bsig_nonplanar = pysiglib.branched_sig(path, deg, planar=False, tree_order="canonical")
        pysiglib.prepare_branched_sig(dim, deg, planar=True)
        bsig_planar = pysiglib.branched_sig(path, deg, planar=True, tree_order="canonical")

        tree = (((0,), 0), 0)
        idx_nonplanar = pysiglib.tree_to_idx(tree, dim, deg, planar=False)
        idx_planar = pysiglib.tree_to_idx(tree, dim, deg, planar=True)

        assert len(bsig_nonplanar) == 21
        assert len(bsig_planar) == 23
        assert idx_nonplanar == 7
        assert idx_planar != idx_nonplanar
        assert pysiglib.idx_to_tree(idx_planar, dim, deg, planar=True) == tree
        assert pysiglib.idx_to_tree(idx_nonplanar, dim, deg, planar=False) == tree
        assert bsig_planar[idx_planar] == pytest.approx(bsig_nonplanar[idx_nonplanar])
        assert bsig_planar[idx_nonplanar] != pytest.approx(bsig_nonplanar[idx_nonplanar])


class TestValidation:
    def test_idx_out_of_range(self):
        bsig_len = pysiglib.branched_sig_length(2, 3)
        with pytest.raises(ValueError):
            pysiglib.idx_to_tree(bsig_len, 2, 3)

    def test_invalid_tree_order(self):
        with pytest.raises(ValueError, match="tree_order must be"):
            pysiglib.tree_to_idx((0,), 2, 3, tree_order="weird")
        with pytest.raises(ValueError, match="tree_order must be"):
            pysiglib.idx_to_tree(0, 2, 3, tree_order="weird")
        with pytest.raises(ValueError, match="tree_order must be"):
            pysiglib.trees(2, 3, tree_order="weird")
        with pytest.raises(ValueError, match="tree_order must be"):
            pysiglib.trees_of_order(2, 1, tree_order="weird")


# ---------------------------------------------------------------------------
# Recursive tree order
# ---------------------------------------------------------------------------

TREE_ORDER_PARAMS = [
    (dim, deg, tree_order, planar)
    for dim, deg in PARAMS
    for tree_order in ("canonical", "recursive")
    for planar in (False, True)
]


class TestRecursiveTreeOrder:
    @pytest.mark.parametrize("dim, deg, tree_order, planar", TREE_ORDER_PARAMS)
    def test_idx_to_tree_to_idx_roundtrip(self, dim, deg, tree_order, planar):
        bsig_len = pysiglib.branched_sig_length(dim, deg, planar=planar)
        for i in range(bsig_len):
            tree = pysiglib.idx_to_tree(i, dim, deg, tree_order=tree_order, planar=planar)
            assert pysiglib.tree_to_idx(tree, dim, deg, tree_order=tree_order, planar=planar) == i

    @pytest.mark.parametrize("dim, deg, tree_order, planar", TREE_ORDER_PARAMS)
    def test_trees_enumeration_matches_tree_to_idx(self, dim, deg, tree_order, planar):
        all_trees = pysiglib.trees(dim, deg, tree_order=tree_order, planar=planar)
        assert len(all_trees) == pysiglib.branched_sig_length(dim, deg, planar=planar)
        assert all_trees[0] is None
        for i, tree in enumerate(all_trees):
            assert pysiglib.tree_to_idx(tree, dim, deg, tree_order=tree_order, planar=planar) == i

    @pytest.mark.parametrize("dim, deg, tree_order, planar", TREE_ORDER_PARAMS)
    def test_trees_of_order_consistency_with_trees(self, dim, deg, tree_order, planar):
        """Concatenating trees_of_order per stratum reproduces trees()."""
        reconstructed = []
        for order in range(deg + 1):
            reconstructed.extend(
                pysiglib.trees_of_order(dim, order, tree_order=tree_order, planar=planar)
            )
        assert tuple(reconstructed) == pysiglib.trees(
            dim, deg, tree_order=tree_order, planar=planar
        )

    @pytest.mark.parametrize("dim, deg", [(2, 3), (3, 3)])
    @pytest.mark.parametrize("planar", [False, True])
    def test_recursive_index_matches_branched_sig_output(self, dim, deg, planar):
        """tree_to_idx(..., tree_order="recursive") must match recursive bsig output."""
        rng = np.random.default_rng(123)
        path = rng.standard_normal((15, dim)).astype(np.float64)
        pysiglib.prepare_branched_sig(dim, deg, planar=planar)

        bsig_rec = pysiglib.branched_sig(path, deg, tree_order="recursive", planar=planar)
        bsig_can = pysiglib.branched_sig(path, deg, tree_order="canonical", planar=planar)

        # Every tree's coefficient must agree between the two orderings when indexed correctly.
        for tree in pysiglib.trees(dim, deg, tree_order="canonical", planar=planar):
            i_rec = pysiglib.tree_to_idx(tree, dim, deg, tree_order="recursive", planar=planar)
            i_can = pysiglib.tree_to_idx(tree, dim, deg, tree_order="canonical", planar=planar)
            np.testing.assert_allclose(bsig_rec[i_rec], bsig_can[i_can], atol=1e-10)

    @pytest.mark.parametrize("dim, deg", PARAMS)
    @pytest.mark.parametrize("planar", [False, True])
    def test_recursive_is_permutation_of_canonical(self, dim, deg, planar):
        """The recursive tree list is a permutation of the canonical list."""
        canonical = pysiglib.trees(dim, deg, tree_order="canonical", planar=planar)
        recursive = pysiglib.trees(dim, deg, tree_order="recursive", planar=planar)
        assert set(canonical) == set(recursive)
        assert recursive[0] is None  # empty tree anchors both orderings
        assert canonical[0] is None
