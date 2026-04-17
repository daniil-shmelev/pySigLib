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

"""
Decorated rooted tree enumeration and indexing for branched signatures.

Trees use the kauri tuple convention::

    Empty tree:   None
    Leaf:         (label,)
    Internal:     (child1, child2, ..., root_label)

where children are sorted and ``label`` is an int in ``[0, dimension)``.
"""

from functools import cache

import kauri

from .param_checks import check_type, check_non_neg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trees_of_order(
        dimension: int,
        order: int,
        planar: bool = False
) -> tuple[tuple]:
    """
    Returns all decorated rooted trees with exactly ``order`` nodes,
    in canonical ordering.

    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param order: Exact number of nodes.
    :type order: int
    :param planar: If True, return planar (ordered) trees.
    :type planar: bool
    :return: Tuple of trees as tuples in kauri convention.
    :rtype: tuple[tuple]

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        # All single-node trees over dimension 2
        t = pysiglib.trees_of_order(2, 1)
        print(t) # ((0,), (1,))

    """
    check_type(dimension, "dimension", int)
    check_type(order, "order", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(order, "order")
    check_type(planar, "planar", bool)
    return _trees_of_order_cached(dimension, order, planar)


@cache
def _trees_of_order_cached(dimension, order, planar=False):
    if order == 0:
        return (None,)
    if planar:
        return tuple(t.sorted_list_repr() for t in kauri.colored_planar_trees_of_order(order, dimension))
    return tuple(t.sorted_list_repr() for t in kauri.colored_trees_of_order(order, dimension))


def trees(
        dimension: int,
        degree: int,
        planar: bool = False
) -> tuple[tuple]:
    """
    Returns all decorated rooted trees up to a given degree (max nodes),
    starting with the empty tree (``None``), in canonical ordering.

    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes per tree.
    :type degree: int
    :param planar: If True, return planar (ordered) trees.
    :type planar: bool
    :return: All decorated rooted trees up to the given degree.
    :rtype: tuple[tuple]

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        t = pysiglib.trees(2, 2)
        print(t) # (None, (0,), (1,), ((0,), 0), ((1,), 0), ((0,), 1), ((1,), 1))

    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_type(planar, "planar", bool)
    return _trees_cached(dimension, degree, planar)


@cache
def _trees_cached(dimension, degree, planar=False):
    if planar:
        return tuple(t.sorted_list_repr() for t in kauri.colored_planar_trees_up_to_order(dimension, degree))
    return tuple(t.sorted_list_repr() for t in kauri.colored_trees(dimension, degree))


def tree_to_idx(
        tree,
        dimension: int,
        degree: int,
        planar: bool = False
) -> int:
    """
    Given a decorated rooted tree, returns its flat index in the
    canonical enumeration (matching :func:`branched_sig` with
    ``tree_order="canonical"``).

    Trees use the kauri tuple convention:

    - Empty tree: ``None`` -- index 0
    - Leaf: ``(label,)`` where ``label`` is in ``[0, dimension)``
    - Internal node: ``(child_1, child_2, ..., root_label)``

    :param tree: Decorated rooted tree as a tuple (or None for empty).
    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes (same as ``degree`` in :func:`branched_sig`).
    :type degree: int
    :param planar: If True, use planar (ordered) tree indexing.
    :type planar: bool
    :return: Flat index in the branched signature vector.
    :rtype: int

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand(size=(100, 2))
        pysiglib.prepare_branched_sig(2, 3)
        bsig = pysiglib.branched_sig(path, 3, tree_order="canonical")

        # Get coefficient at a specific tree
        tree = ((1,), 0)  # root 0 with one child labeled 1
        idx = pysiglib.tree_to_idx(tree, dimension=2, degree=3)
        print(f"Index: {idx}, Coefficient: {bsig[idx]}")

    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_type(planar, "planar", bool)
    return _tree_to_idx_cached(tree, dimension, degree, planar)


@cache
def _tree_to_idx_cached(tree, dimension, degree, planar=False):
    if tree is None:
        return 0
    if planar:
        return kauri.colored_planar_tree_to_idx(kauri.PlanarTree(tree), dimension, degree)
    return kauri.colored_tree_to_idx(kauri.Tree(tree), dimension, degree)


def idx_to_tree(
        idx: int,
        dimension: int,
        degree: int,
        planar: bool = False
) -> tuple:
    """
    Given a flat index in the canonical enumeration, returns the
    corresponding decorated rooted tree as a tuple.

    :param idx: Flat index in the branched signature vector.
    :type idx: int
    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes (same as ``degree`` in :func:`branched_sig`).
    :type degree: int
    :param planar: If True, use planar (ordered) tree indexing.
    :type planar: bool
    :return: Decorated rooted tree (None for empty tree, tuple otherwise).
    :rtype: tuple or None

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        tree = pysiglib.idx_to_tree(3, dimension=2, degree=3)
        print(tree)  # ((0,), 0)

    """
    check_type(idx, "idx", int)
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(idx, "idx")
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_type(planar, "planar", bool)
    return _idx_to_tree_cached(idx, dimension, degree, planar)


@cache
def _idx_to_tree_cached(idx, dimension, degree, planar=False):
    if planar:
        kt = kauri.idx_to_colored_planar_tree(idx, dimension, degree)
    else:
        kt = kauri.idx_to_colored_tree(idx, dimension, degree)
    return kt.sorted_list_repr()
