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
Decorated basis enumeration and indexing for branched signatures.

Non-planar BCK signatures are indexed by decorated rooted trees. Planar MKW
signatures are indexed by ordered forests of decorated planar rooted trees.

Trees use the native pySigLib tuple convention::

    Empty tree:   None
    Leaf:         (label,)
    Internal:     (child1, child2, ..., root_label)

where children are sorted in the non-planar case and ``label`` is an int in
``[0, dimension)``. Ordered forests are represented as ``(tree1, tree2, ...)``.
The order is the implementation order used by the native CPU and CUDA caches.
"""

from functools import cache

from .param_checks import check_type, check_non_neg


def _tree_node_count(tree):
    return 1 + sum(_tree_node_count(child) for child in tree[:-1])


def _is_planar_forest_tuple(obj):
    return isinstance(obj, tuple) and len(obj) > 0 and not isinstance(obj[-1], int)


def _as_planar_forest_tuple(tree):
    if tree is None:
        return None
    if _is_planar_forest_tuple(tree):
        return tree
    return (tree,)


def _child_sequences(target_nodes, trees, planar):
    out = []
    current = []

    def enumerate_(remaining, min_idx):
        if remaining == 0:
            out.append(tuple(current))
            return
        start = 0 if planar else min_idx
        for idx in range(start, len(trees)):
            nodes = _tree_node_count(trees[idx])
            if nodes > remaining:
                break
            current.append(idx)
            enumerate_(remaining - nodes, idx)
            current.pop()

    enumerate_(target_nodes, 0)
    return tuple(out)


@cache
def _native_tree_basis_cached(dimension, degree, planar):
    trees = []
    for order in range(1, degree + 1):
        if order == 1:
            for label in range(dimension):
                trees.append((label,))
            continue

        current_trees = tuple(trees)
        for children in _child_sequences(order - 1, current_trees, planar):
            if not children:
                continue
            for label in range(dimension):
                trees.append(tuple(current_trees[idx] for idx in children) + (label,))
    return tuple(trees)


@cache
def _native_forest_basis_cached(dimension, degree):
    trees = _native_tree_basis_cached(dimension, degree, True)
    forests = []
    current = []

    def enumerate_(remaining):
        if remaining == 0:
            forests.append(tuple(current))
            return
        for tree in trees:
            nodes = _tree_node_count(tree)
            if nodes > remaining:
                break
            current.append(tree)
            enumerate_(remaining - nodes)
            current.pop()

    for order in range(1, degree + 1):
        enumerate_(order)
    return tuple(forests)


@cache
def _basis_cached(dimension, degree, planar):
    if planar:
        non_empty = _native_forest_basis_cached(dimension, degree)
    else:
        non_empty = _native_tree_basis_cached(dimension, degree, False)
    return (None,) + non_empty


@cache
def _basis_index_cached(dimension, degree, planar):
    return {basis_element: idx for idx, basis_element in enumerate(_basis_cached(dimension, degree, planar))}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trees_of_order(
        dimension: int,
        order: int,
        *,
        planar: bool = False,
) -> tuple[tuple]:
    """
    Returns all basis elements with exactly ``order`` nodes.

    For ``planar=False`` these are non-planar decorated rooted trees. For
    ``planar=True`` these are ordered forests of decorated planar rooted trees,
    with ``order`` equal to the total number of nodes in the forest. The order
    matches the native pySigLib coefficient layout.

    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param order: Exact number of nodes.
    :type order: int
    :param planar: If True, enumerate ordered forests of planar rooted trees.
    :type planar: bool
    :return: Tuple of basis elements as tuples in native pySigLib convention.
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
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(order, "order")
    if order == 0:
        return (None,)
    return tuple(
        basis_element
        for basis_element in _basis_cached(dimension, order, planar)[1:]
        if _basis_element_order(basis_element, planar) == order
    )


def trees(
        dimension: int,
        degree: int,
        *,
        planar: bool = False,
) -> tuple[tuple]:
    """
    Returns all basis elements up to a given degree, starting with the empty
    tree (``None``).

    For ``planar=False`` these are non-planar decorated rooted trees. For
    ``planar=True`` these are ordered forests of decorated planar rooted trees,
    with degree equal to the maximum total number of nodes in the forest. The
    order matches the native pySigLib coefficient layout.

    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes per tree or forest.
    :type degree: int
    :param planar: If True, enumerate ordered forests of planar rooted trees.
    :type planar: bool
    :return: All basis elements up to the given degree.
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
    check_type(planar, "planar", bool)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    return _basis_cached(dimension, degree, planar)


def _basis_element_order(basis_element, planar):
    if planar:
        return sum(_tree_node_count(tree) for tree in basis_element)
    return _tree_node_count(basis_element)


def tree_to_idx(
        tree,
        dimension: int,
        degree: int,
        *,
        planar: bool = False,
        scalar_term: bool = False,
) -> int:
    """
    Given a basis element, returns its flat index in the branched-signature
    coefficient vector.

    Trees use the native pySigLib tuple convention:

    - Empty tree: ``None`` - index 0 when ``scalar_term=True``; invalid otherwise
    - Leaf: ``(label,)`` where ``label`` is in ``[0, dimension)``
    - Internal node: ``(child_1, child_2, ..., root_label)``
    - Planar ordered forest: ``(tree_1, tree_2, ...)`` when ``planar=True``

    With ``scalar_term=True`` the empty tree sits at index 0. With
    ``scalar_term=False`` (default) there is no empty-tree entry, so all
    indices shift down by 1 and ``None`` is invalid.

    In the planar case, the branched-signature basis is indexed by ordered
    forests. Passing a single tree is accepted as shorthand for the one-tree
    forest.

    :param tree: Decorated rooted tree, ordered forest, or None for empty when ``scalar_term=True``.
    :type tree: tuple | None
    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes (same as ``degree`` in :func:`branched_sig`).
    :type degree: int
    :param planar: If True, use the planar ordered-forest enumeration matching
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
        bsig = pysiglib.branched_sig(path, 3)  # scalar_term=False

        tree = ((1,), 0)
        idx = pysiglib.tree_to_idx(tree, dimension=2, degree=3)
        print(f"Index: {idx}, Coefficient: {bsig[idx]}")

    """
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
    basis_element = _as_planar_forest_tuple(tree) if planar else tree
    index = _basis_index_cached(dimension, degree, planar).get(basis_element)
    if index is None:
        raise ValueError("tree is not in the requested branched signature basis")
    return index if scalar_term else index - 1


def idx_to_tree(
        idx: int,
        dimension: int,
        degree: int,
        *,
        planar: bool = False,
        scalar_term: bool = False,
) -> tuple:
    """
    Inverse of :func:`tree_to_idx`. Given a flat index in the branched-signature
    coefficient vector, returns the corresponding basis element.

    With ``scalar_term=True``, index 0 maps to the empty tree (``None``).
    With ``scalar_term=False`` (default), all indices shift down by 1 (index 0
    maps to the first non-empty tree) and the empty tree is unreachable.

    :param idx: Flat index in the branched signature vector.
    :type idx: int
    :param dimension: Path dimension (alphabet size).
    :type dimension: int
    :param degree: Maximum number of nodes (same as ``degree`` in :func:`branched_sig`).
    :type degree: int
    :param planar: If True, interpret ``idx`` in the planar ordered-forest
        enumeration matching ``branched_sig(..., planar=True)``.
    :type planar: bool
    :param scalar_term: Whether the source branched signature includes the leading
        scalar 1 at index 0. Must match the format of the bsig the index was taken
        from. Default ``False``.
    :type scalar_term: bool
    In the planar case, the returned tuple is an ordered forest, represented as
    ``(tree_1, tree_2, ...)``.

    :return: Decorated rooted tree, ordered forest, or None for empty tree when
        ``scalar_term=True``.
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
    if not scalar_term:
        idx = idx + 1
    basis = _basis_cached(dimension, degree, planar)
    if idx >= len(basis):
        raise ValueError("idx is out of range for the requested branched signature basis")
    return basis[idx]
