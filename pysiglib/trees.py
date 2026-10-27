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

"""BCK trees, MKW ordered forests, and weighted Lyndon forests in coordinate order.

A decorated tree is ``(child_1, ..., child_k, root_label)``; a leaf is ``(label,)``. BCK children are sorted by native tree ID. A forest is a tuple of planar trees in their specified order. The empty BCK element is ``None`` and the empty forest is ``()``.
"""

from functools import cache

from .param_checks import _check_basis_index, _check_basis_parameters, check_type
from .words import _is_lyndon


def _tree_node_count(tree):
    return 1 + sum(_tree_node_count(child) for child in tree[:-1])


def _as_planar_forest_tuple(tree):
    # Coefficient APIs also accept a bare tree as a one-tree forest.
    if tree is None:
        return ()
    if isinstance(tree, tuple) and (not tree or not isinstance(tree[-1], int)):
        return tree
    return (tree,)


def _child_sequences(target_nodes, trees, planar):
    current = []
    weights = tuple(_tree_node_count(tree) for tree in trees)

    def enumerate_(remaining, min_idx):
        if remaining == 0:
            yield tuple(current)
            return
        for idx in range(0 if planar else min_idx, len(trees)):
            nodes = weights[idx]
            if nodes > remaining:
                break
            current.append(idx)
            yield from enumerate_(remaining - nodes, idx)
            current.pop()

    yield from enumerate_(target_nodes, 0)


@cache
def _native_tree_basis_cached(dimension, degree, planar):
    trees = []
    for order in range(1, degree + 1):
        current_trees = tuple(trees)
        for children in _child_sequences(order - 1, current_trees, planar):
            for label in range(dimension):
                trees.append(tuple(current_trees[idx] for idx in children) + (label,))
    return tuple(trees)


@cache
def _native_forest_basis_cached(dimension, degree, lyndon=False):
    trees = _native_tree_basis_cached(dimension, degree, planar=True)
    # Stream forest words so the compressed enumeration does not retain the full basis.
    return tuple(
        tuple(trees[idx] for idx in word)
        for order in range(1, degree + 1)
        for word in _child_sequences(order, trees, True)
        if not lyndon or _is_lyndon(word)
    )


@cache
def _tree_indices(dimension, degree):
    basis = _native_tree_basis_cached(dimension, degree, planar=False)
    return {tree: idx for idx, tree in enumerate(basis)}


@cache
def _forest_indices(dimension, degree, *, lyndon):
    basis = _native_forest_basis_cached(dimension, degree, lyndon=lyndon)
    return {forest: idx for idx, forest in enumerate(basis)}


def _check_tree(tree, dimension):
    check_type(tree, "tree", tuple)
    if not tree:
        raise ValueError("a decorated tree must contain a root label")
    check_type(tree[-1], "root label", int)
    if not 0 <= tree[-1] < dimension:
        raise ValueError("tree label is outside the path dimension")
    for child in tree[:-1]:
        _check_tree(child, dimension)


def _check_forest(forest, dimension):
    check_type(forest, "forest", tuple)
    for tree in forest:
        _check_tree(tree, dimension)


def _element_to_idx(element, indices, *, scalar_term=False):
    idx = indices.get(element)
    if idx is None:
        raise ValueError("element is not in the requested coordinate basis")
    return idx + int(scalar_term)


def _idx_to_element(idx, basis, *, scalar_term=False, empty=None):
    _check_basis_index(idx, len(basis) + int(scalar_term))
    if scalar_term and idx == 0:
        return empty
    return basis[idx - int(scalar_term)]


def trees(dimension: int, degree: int, *, scalar_term: bool = False) -> list:
    """Return decorated nonplanar BCK trees through ``degree`` in numerical output order.

    The order is node count, native child-ID sequence, then root label. Children are sorted by native tree ID. Increasing ``degree`` preserves existing indices. These labels index both BCK signatures and BCK log signatures. For planar MKW coordinates use :func:`forests`.

    :param dimension: Number of labels; use the effective dimension after any path augmentation.
    :param degree: Inclusive maximum number of nodes.
    :param scalar_term: Include the empty element ``None`` at index zero. Default ``False``.
    :return: Fresh list of tree tuples, matching :func:`idx_to_tree` under the same options.

    Example::

        >>> trees(2, 2)
        [(0,), (1,), ((0,), 0), ((0,), 1), ((1,), 0), ((1,), 1)]
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    basis = _native_tree_basis_cached(dimension, degree, planar=False)
    return [None, *basis] if scalar_term else list(basis)


def trees_of_order(dimension: int, order: int, *, scalar_term: bool = False) -> list:
    """Return nonplanar BCK trees of exactly ``order`` nodes in native order.

    :param dimension: Number of labels.
    :param order: Exact number of nodes.
    :param scalar_term: Include ``None`` when ``order=0``. Default ``False``.
    :return: Fresh list of tree tuples, or ``[None]`` for order zero with ``scalar_term=True``.
    """
    basis = trees(dimension, order, scalar_term=scalar_term)
    if order == 0:
        return basis
    return [tree for tree in basis if tree is not None and _tree_node_count(tree) == order]


def tree_to_idx(tree, dimension: int, degree: int, *, scalar_term: bool = False) -> int:
    """Return a canonical BCK tree's full coordinate index; inverse of :func:`idx_to_tree`.

    :param tree: Decorated tree with children sorted by native tree ID, or ``None`` for the scalar.
    :param dimension: Number of labels.
    :param degree: Inclusive maximum number of nodes.
    :param scalar_term: Include the empty element at index zero. Default ``False``.
    :return: Index in :func:`trees` under the same options.
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    if tree is None:
        if not scalar_term:
            raise ValueError("The empty tree has no index with scalar_term=False")
        return 0
    _check_tree(tree, dimension)
    return _element_to_idx(tree, _tree_indices(dimension, degree), scalar_term=scalar_term)


def idx_to_tree(idx: int, dimension: int, degree: int, *, scalar_term: bool = False):
    """Return the BCK tree at a full coordinate index; inverse of :func:`tree_to_idx`.

    :param idx: Nonnegative flat coordinate index.
    :param dimension: Number of labels.
    :param degree: Inclusive maximum number of nodes.
    :param scalar_term: Include ``None`` at index zero. Default ``False``.
    :return: Tree tuple or ``None``, equal to ``trees(dimension, degree, scalar_term=scalar_term)[idx]``.
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    basis = _native_tree_basis_cached(dimension, degree, planar=False)
    return _idx_to_element(idx, basis, scalar_term=scalar_term, empty=None)


def forests(dimension: int, degree: int, *, scalar_term: bool = False) -> list[tuple]:
    """Return ordered planar MKW forests through ``degree`` in numerical output order.

    Each forest is a tuple of decorated planar trees. The order is total node count, then lexicographic order of native tree IDs. Increasing ``degree`` preserves existing indices. These labels index MKW signatures and expanded MKW log signatures (method 0).

    :param dimension: Number of labels; use the effective dimension after any path augmentation.
    :param degree: Inclusive maximum total number of nodes across all component trees.
    :param scalar_term: Include the empty forest ``()`` at index zero. Default ``False``.
    :return: Fresh list of forest tuples, matching :func:`idx_to_forest` under the same options.

    Example::

        >>> forests(1, 2)
        [((0,),), ((0,), (0,)), (((0,), 0),)]
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    basis = _native_forest_basis_cached(dimension, degree, lyndon=False)
    return [(), *basis] if scalar_term else list(basis)


def forest_to_idx(forest: tuple, dimension: int, degree: int, *, scalar_term: bool = False) -> int:
    """Return an ordered planar forest's full MKW coordinate index.

    :param forest: Tuple of planar trees, including a singleton tuple for a one-tree forest.
    :param dimension: Number of labels.
    :param degree: Inclusive maximum total node count.
    :param scalar_term: Include the empty forest ``()`` at index zero. Default ``False``.
    :return: Index in :func:`forests`, distinct from the compressed :func:`lyndon_forest_to_idx` index.
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    _check_forest(forest, dimension)
    if not forest:
        if not scalar_term:
            raise ValueError("The empty forest has no index with scalar_term=False")
        return 0
    indices = _forest_indices(dimension, degree, lyndon=False)
    return _element_to_idx(forest, indices, scalar_term=scalar_term)


def idx_to_forest(idx: int, dimension: int, degree: int, *, scalar_term: bool = False) -> tuple:
    """Return the ordered forest at a full MKW index; inverse of :func:`forest_to_idx`.

    :param idx: Nonnegative flat coordinate index.
    :param dimension: Number of labels.
    :param degree: Inclusive maximum total node count.
    :param scalar_term: Include the empty forest at index zero. Default ``False``.
    :return: Forest tuple at ``forests(dimension, degree, scalar_term=scalar_term)[idx]``.
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    basis = _native_forest_basis_cached(dimension, degree, lyndon=False)
    return _idx_to_element(idx, basis, scalar_term=scalar_term, empty=())


def lyndon_forests(dimension: int, degree: int) -> list[tuple]:
    """Return weighted Lyndon-forest labels for compressed MKW log signatures.

    Each planar tree is a letter ordered by its native tree ID, with weight equal to its number of nodes. Lyndon forests are the nonempty Lyndon words over this alphabet, ordered by total weight and then lexicographically by tree IDs. Increasing ``degree`` preserves existing indices.

    Method 1 returns selected expanded forest coefficients; methods 2 and 3 return coefficients in the standard Lyndon bracket basis with the same labels. Use :func:`forest_to_idx` for full indices and :func:`lyndon_forest_to_idx` for compressed indices.

    :param dimension: Number of labels; use the effective dimension after any path augmentation.
    :param degree: Inclusive maximum total node count, regardless of the number of component trees.
    :return: Fresh list of forest tuples. The empty forest is always excluded.

    Example::

        >>> lyndon_forests(1, 2)
        [((0,),), (((0,), 0),)]
    """
    _check_basis_parameters(dimension, degree)
    return list(_native_forest_basis_cached(dimension, degree, lyndon=True))


def lyndon_forest_to_idx(forest: tuple, dimension: int, degree: int) -> int:
    """Return a weighted Lyndon forest's compressed MKW log-signature index.

    :param forest: Nonempty tuple of planar trees forming a Lyndon word in native tree-ID order.
    :param dimension: Number of labels.
    :param degree: Inclusive maximum total node count.
    :return: Index in :func:`lyndon_forests`. There is no scalar coordinate.
    """
    _check_basis_parameters(dimension, degree)
    _check_forest(forest, dimension)
    if not forest:
        raise ValueError("The empty forest has no compressed index")
    return _element_to_idx(forest, _forest_indices(dimension, degree, lyndon=True))


def idx_to_lyndon_forest(idx: int, dimension: int, degree: int) -> tuple:
    """Return a compressed MKW label; inverse of :func:`lyndon_forest_to_idx`.

    :param idx: Nonnegative compressed coordinate index.
    :param dimension: Number of labels.
    :param degree: Inclusive maximum total node count.
    :return: Forest tuple at ``lyndon_forests(dimension, degree)[idx]``. There is no scalar coordinate.
    """
    _check_basis_parameters(dimension, degree)
    basis = _native_forest_basis_cached(dimension, degree, lyndon=True)
    return _idx_to_element(idx, basis)
