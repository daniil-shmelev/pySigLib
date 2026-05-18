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


def _check_tree_order(tree_order):
    if tree_order not in ("recursive", "canonical"):
        raise ValueError(f"tree_order must be 'recursive' or 'canonical', got {tree_order!r}")


def _canonical_to_recursive_perm(dimension, degree, planar):
    if planar:
        return kauri.planar_canonical_to_recursive_permutation(dimension, degree)
    return kauri.canonical_to_recursive_permutation(dimension, degree)


def _recursive_to_canonical_perm(dimension, degree, planar):
    if planar:
        return kauri.planar_recursive_to_canonical_permutation(dimension, degree)
    return kauri.recursive_to_canonical_permutation(dimension, degree)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trees_of_order(
        dimension: int,
        order: int,
        *,
        tree_order: str = "canonical",
        planar: bool = False,
) -> tuple[tuple]:
    """
    Returns all decorated rooted trees with exactly ``order`` nodes, in the
    specified ordering.

    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param order: Exact number of nodes.
    :type order: int
    :param tree_order: Tree ordering convention. ``"canonical"`` (default) uses
        the shape-first order matching :func:`tree_to_idx` and
        ``branched_sig(..., tree_order="canonical")``. ``"recursive"`` uses the
        recursive bottom-up construction order matching
        ``branched_sig(..., tree_order="recursive")`` (the default).
    :type tree_order: str
    :param planar: If True, enumerate planar (ordered) trees in which the
        order of children matters.
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
    _check_tree_order(tree_order)
    check_type(dimension, "dimension", int)
    check_type(order, "order", int)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(order, "order")
    return _trees_of_order_cached(dimension, order, tree_order, planar)


@cache
def _trees_of_order_cached(dimension, order, tree_order, planar):
    if order == 0:
        return (None,)
    if tree_order == "canonical":
        if planar:
            return tuple(t.sorted_list_repr() for t in kauri.colored_planar_trees_of_order(order, dimension))
        return tuple(t.sorted_list_repr() for t in kauri.colored_trees_of_order(order, dimension))
    # Recursive ordering: compute all trees up to `order` in recursive order,
    # then slice off the last stratum (trees of exactly `order` nodes).
    from .branched_sig import branched_sig_length
    all_trees = _trees_cached(dimension, order, "recursive", planar)
    lower = branched_sig_length(dimension, order - 1, scalar_term=True, planar=planar)
    upper = branched_sig_length(dimension, order, scalar_term=True, planar=planar)
    return all_trees[lower:upper]


def trees(
        dimension: int,
        degree: int,
        *,
        tree_order: str = "canonical",
        planar: bool = False,
) -> tuple[tuple]:
    """
    Returns all decorated rooted trees up to a given degree (max nodes),
    starting with the empty tree (``None``), in the specified ordering.

    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes per tree.
    :type degree: int
    :param tree_order: Tree ordering convention. ``"canonical"`` (default) uses
        the shape-first order matching :func:`tree_to_idx` and
        ``branched_sig(..., tree_order="canonical")``. ``"recursive"`` uses the
        recursive bottom-up construction order matching
        ``branched_sig(..., tree_order="recursive")`` (the default).
    :type tree_order: str
    :param planar: If True, enumerate planar (ordered) trees in which the
        order of children matters.
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
    _check_tree_order(tree_order)
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    return _trees_cached(dimension, degree, tree_order, planar)


@cache
def _trees_cached(dimension, degree, tree_order, planar):
    if tree_order == "canonical":
        if planar:
            canonical = tuple(
                t.sorted_list_repr() for t in kauri.colored_planar_trees_up_to_order(degree, dimension)
            )
        else:
            canonical = tuple(t.sorted_list_repr() for t in kauri.colored_trees(dimension, degree))
        return canonical
    # Recursive: start from canonical and permute the non-empty trees.
    canonical = _trees_cached(dimension, degree, "canonical", planar)
    if len(canonical) <= 1:
        return canonical
    perm = _canonical_to_recursive_perm(dimension, degree, planar)
    # perm[i] = recursive position of the canonical non-empty tree at index i.
    # Build recursive-ordered list: recursive[0] = None; recursive[perm[i] + 1] = canonical[i + 1].
    n_non_empty = len(canonical) - 1
    recursive = [None] * (n_non_empty + 1)
    for i in range(n_non_empty):
        recursive[perm[i] + 1] = canonical[i + 1]
    return tuple(recursive)


def tree_to_idx(
        tree,
        dimension: int,
        degree: int,
        *,
        tree_order: str = "canonical",
        planar: bool = False,
        scalar_term: bool = False,
) -> int:
    """
    Given a decorated rooted tree, returns its flat index in the
    branched-signature coefficient vector.

    Trees use the kauri tuple convention:

    - Empty tree: ``None`` -- index 0 when ``scalar_term=True``; invalid otherwise
    - Leaf: ``(label,)`` where ``label`` is in ``[0, dimension)``
    - Internal node: ``(child_1, child_2, ..., root_label)``

    With ``scalar_term=True`` the empty tree sits at index 0. With
    ``scalar_term=False`` (default) there is no empty-tree entry, so all
    indices shift down by 1 and ``None`` is invalid.

    :param tree: Decorated rooted tree as a tuple (or None for empty when ``scalar_term=True``).
    :type tree: tuple | None
    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes (same as ``degree`` in :func:`branched_sig`).
    :type degree: int
    :param tree_order: Tree ordering convention. ``"canonical"`` (default)
        matches ``branched_sig(..., tree_order="canonical")``. ``"recursive"``
        matches ``branched_sig(..., tree_order="recursive")`` (the default).
    :type tree_order: str
    :param planar: If True, use the planar (ordered) enumeration matching
        ``branched_sig(..., planar=True)``.
    :type planar: bool
    :param scalar_term: Whether the target branched signature includes the leading
        scalar 1 at index 0. Must match the format of the bsig you intend to
        index. Default ``False``.
    :type scalar_term: bool
    :return: Flat index in the branched signature vector.
    :rtype: int

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        path = torch.rand(size=(100, 2))
        pysiglib.prepare_branched_sig(2, 3)
        bsig = pysiglib.branched_sig(path, 3, tree_order="canonical")  # scalar_term=False

        tree = ((1,), 0)
        idx = pysiglib.tree_to_idx(tree, dimension=2, degree=3)
        print(f"Index: {idx}, Coefficient: {bsig[idx]}")

    """
    _check_tree_order(tree_order)
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    if tree is None and not scalar_term:
        raise ValueError(
            "The empty tree has no index in a branched signature with scalar_term=False. "
            "Pass scalar_term=True if your bsig includes the leading scalar 1."
        )
    return _tree_to_idx_cached(tree, dimension, degree, tree_order, planar, scalar_term)


@cache
def _tree_to_idx_cached(tree, dimension, degree, tree_order, planar, scalar_term):
    if tree is None:
        return 0
    if planar:
        canonical_idx = kauri.colored_planar_tree_to_idx(kauri.PlanarTree(tree), dimension, degree)
    else:
        canonical_idx = kauri.colored_tree_to_idx(kauri.Tree(tree), dimension, degree)
    if tree_order == "canonical":
        idx = canonical_idx
    else:
        perm = _canonical_to_recursive_perm(dimension, degree, planar)
        idx = int(perm[canonical_idx - 1]) + 1
    return idx if scalar_term else idx - 1


def idx_to_tree(
        idx: int,
        dimension: int,
        degree: int,
        *,
        tree_order: str = "canonical",
        planar: bool = False,
        scalar_term: bool = False,
) -> tuple:
    """
    Inverse of :func:`tree_to_idx`. Given a flat index in the branched-signature
    coefficient vector, returns the corresponding decorated rooted tree.

    With ``scalar_term=True``, index 0 maps to the empty tree (``None``).
    With ``scalar_term=False`` (default), all indices shift down by 1 (index 0
    maps to the first non-empty tree) and the empty tree is unreachable.

    :param idx: Flat index in the branched signature vector.
    :type idx: int
    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes (same as ``degree`` in :func:`branched_sig`).
    :type degree: int
    :param tree_order: Tree ordering convention. ``"canonical"`` (default)
        matches ``branched_sig(..., tree_order="canonical")``. ``"recursive"``
        matches ``branched_sig(..., tree_order="recursive")`` (the default).
    :type tree_order: str
    :param planar: If True, interpret ``idx`` in the planar (ordered)
        enumeration matching ``branched_sig(..., planar=True)``.
    :type planar: bool
    :param scalar_term: Whether the source branched signature includes the leading
        scalar 1 at index 0. Must match the format of the bsig the index was taken
        from. Default ``False``.
    :type scalar_term: bool
    :return: Decorated rooted tree (None for empty tree when ``scalar_term=True``,
        tuple otherwise).
    :rtype: tuple or None

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        tree = pysiglib.idx_to_tree(3, dimension=2, degree=3)
        print(tree)

    """
    check_type(idx, "idx", int)
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(planar, "planar", bool)
    check_non_neg(idx, "idx")
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    _check_tree_order(tree_order)
    return _idx_to_tree_cached(idx, dimension, degree, tree_order, planar, scalar_term)


@cache
def _idx_to_tree_cached(idx, dimension, degree, tree_order, planar, scalar_term):
    if not scalar_term:
        idx = idx + 1
    if idx == 0:
        return None
    if tree_order == "recursive":
        inv_perm = _recursive_to_canonical_perm(dimension, degree, planar)
        canonical_idx = int(inv_perm[idx - 1]) + 1
    else:
        canonical_idx = idx
    if planar:
        kt = kauri.idx_to_colored_planar_tree(canonical_idx, dimension, degree)
    else:
        kt = kauri.idx_to_colored_tree(canonical_idx, dimension, degree)
    return kt.sorted_list_repr()
