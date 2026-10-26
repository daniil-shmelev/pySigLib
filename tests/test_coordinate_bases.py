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

"""Public coordinate contracts, including independent numerical layout checks."""

from functools import partial

import numpy as np
import pytest

import pysiglib as ps

FULL = [
    pytest.param(ps.words, ps.word_to_idx, ps.idx_to_word, ps.sig_length, (), id="words"),
    pytest.param(ps.trees, ps.tree_to_idx, ps.idx_to_tree, ps.branched_sig_length, None, id="trees"),
    pytest.param(ps.forests, ps.forest_to_idx, ps.idx_to_forest,
                 partial(ps.branched_sig_length, planar=True), (), id="forests"),
]
COMPRESSED = [
    pytest.param(ps.lyndon_words, ps.lyndon_word_to_idx, ps.idx_to_lyndon_word,
                 ps.log_sig_length, (), id="lyndon_words"),
    pytest.param(ps.lyndon_forests, ps.lyndon_forest_to_idx, ps.idx_to_lyndon_forest,
                 partial(ps.branched_log_sig_length, planar=True), (), id="lyndon_forests"),
]


@pytest.mark.parametrize("enumerate_,to_idx,from_idx,length,empty", FULL + COMPRESSED)
@pytest.mark.parametrize("dimension,degree", [(0, 0), (0, 3), (2, 0), (1, 5), (2, 4), (3, 3)])
def test_coordinate_contract(enumerate_, to_idx, from_idx, length, empty, dimension, degree):
    basis = enumerate_(dimension=dimension, degree=degree)
    assert isinstance(basis, tuple)
    # Ordinary log_sig_length requires positive arguments; enumeration also covers zero.
    expected = length(dimension, degree) if dimension and degree else 0
    assert len(set(basis)) == len(basis) == expected
    assert empty not in basis
    for i, element in enumerate(basis):
        assert from_idx(i, dimension=dimension, degree=degree) == element
        assert to_idx(element, dimension=dimension, degree=degree) == i
    for idx in (-1, len(basis)):
        with pytest.raises(ValueError, match="out of range"):
            from_idx(idx, dimension, degree)
    with pytest.raises(ValueError):
        to_idx(empty, dimension, degree)


@pytest.mark.parametrize("enumerate_,to_idx,from_idx,length,empty", FULL)
@pytest.mark.parametrize("dimension,degree", [(0, 0), (0, 3), (2, 0), (1, 3), (2, 3)])
def test_scalar_contract(enumerate_, to_idx, from_idx, length, empty, dimension, degree):
    basis = enumerate_(dimension, degree, scalar_term=True)
    assert basis == (empty,) + enumerate_(dimension, degree)
    assert len(basis) == length(dimension, degree, scalar_term=True)
    for i, element in enumerate(basis):
        assert to_idx(element, dimension, degree, scalar_term=True) == i
        assert from_idx(i, dimension, degree, scalar_term=True) == element
    with pytest.raises(ValueError):
        from_idx(len(basis), dimension, degree, scalar_term=True)


@pytest.mark.parametrize("enumerate_,to_idx,from_idx,length,empty", FULL + COMPRESSED)
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_degree_preserves_indices(enumerate_, to_idx, from_idx, length, empty, dimension):
    small = enumerate_(dimension, 2)
    large = enumerate_(dimension, 4)
    assert large[:len(small)] == small
    for i, element in enumerate(small):
        assert to_idx(element, dimension, 4) == i
        assert from_idx(i, dimension, 4) == element
    for element in large[len(small):]:
        with pytest.raises(ValueError):
            to_idx(element, dimension, 2)


@pytest.mark.parametrize("enumerate_,to_idx,from_idx,length,empty", FULL + COMPRESSED)
@pytest.mark.parametrize("dimension,degree,error", [(-1, 2, ValueError), (2, -1, ValueError),
                                                        (2.0, 2, TypeError), (2, 2.0, TypeError)])
def test_parameter_validation(enumerate_, to_idx, from_idx, length, empty, dimension, degree, error):
    for function, args in [(enumerate_, ()), (to_idx, (empty,)), (from_idx, (0,))]:
        with pytest.raises(error):
            function(*args, dimension, degree)


@pytest.mark.parametrize("enumerate_,to_idx,from_idx,length,empty", FULL)
def test_scalar_type_validation(enumerate_, to_idx, from_idx, length, empty):
    for function, args in [(enumerate_, ()), (to_idx, (empty,)), (from_idx, (0,))]:
        with pytest.raises(TypeError, match="scalar_term"):
            function(*args, 2, 3, scalar_term=1)


@pytest.mark.parametrize("enumerate_,to_idx,from_idx,length,empty", COMPRESSED)
def test_compressed_has_no_scalar_option(enumerate_, to_idx, from_idx, length, empty):
    for function, args in [(enumerate_, ()), (to_idx, (empty,)), (from_idx, (0,))]:
        with pytest.raises(TypeError):
            function(*args, 2, 3, scalar_term=True)


@pytest.mark.parametrize("enumerate_,to_idx,from_idx,length,empty", FULL + COMPRESSED)
def test_index_type_validation(enumerate_, to_idx, from_idx, length, empty):
    with pytest.raises(TypeError):
        from_idx(0.5, 2, 3)
    with pytest.raises(TypeError):
        to_idx([], 2, 3)


@pytest.mark.parametrize("dimension,degree", [(0, 0), (0, 3), (2, 0), (1, 4), (3, 3)])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_exact_word_lengths(dimension, degree, scalar_term):
    assert ps.words(dimension, degree, scalar_term=scalar_term) == tuple(
        word for n in range(degree + 1)
        for word in ps.words_of_length(dimension, n, scalar_term=scalar_term)
    )
    assert ps.lyndon_words(dimension, degree) == tuple(
        word for n in range(degree + 1) for word in ps.lyndon_words_of_length(dimension, n)
    )


@pytest.mark.parametrize("branched", [False, True])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_compressed_labels_match_numerical_projection(branched, scalar_term):
    dimension, degree = 2, 4
    path = np.random.default_rng(79).normal(size=(7, dimension))
    if branched:
        ps.prepare_branched_log_sig(dimension, degree, 2, planar=True, device="cpu")
        signature = ps.branched_sig(path, degree, planar=True, scalar_term=scalar_term)
        convert = partial(ps.branched_sig_to_log_sig, planar=True)
        full_index, labels = ps.forest_to_idx, ps.lyndon_forests(dimension, degree)
    else:
        ps.prepare_log_sig(dimension, degree, 2)
        signature = ps.sig(path, degree, scalar_term=scalar_term)
        convert = ps.sig_to_log_sig
        full_index, labels = ps.word_to_idx, ps.lyndon_words(dimension, degree)
    expanded = convert(signature, dimension, degree, method=0)
    selected = convert(signature, dimension, degree, method=1)
    indices = [full_index(label, dimension, degree, scalar_term=scalar_term) for label in labels]
    np.testing.assert_allclose(selected, expanded[indices], rtol=1e-12, atol=1e-12)

    # Independently expand standard brackets and reconstruct their selected coefficients.
    expansions = {}
    for label in labels:
        if len(label) == 1:
            expansions[label] = {label: 1}
            continue
        split = next(i for i in range(1, len(label)) if label[i:] in expansions)
        terms = {}
        for left, a in expansions[label[:split]].items():
            for right, b in expansions[label[split:]].items():
                terms[left + right] = terms.get(left + right, 0) + a * b
                terms[right + left] = terms.get(right + left, 0) - a * b
        expansions[label] = terms
    projection = np.array([[expansions[col].get(row, 0) for col in labels] for row in labels])
    bracket_coefficients = convert(signature, dimension, degree, method=2)
    np.testing.assert_allclose(projection @ bracket_coefficients, selected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("module_name", ["pysiglib.torch_api", "pysiglib.jax_api"])
def test_frontend_coordinate_helpers(module_name):
    api = pytest.importorskip(module_name)
    for name in (
        "words", "word_to_idx", "idx_to_word",
        "lyndon_words", "lyndon_word_to_idx", "idx_to_lyndon_word",
        "trees", "tree_to_idx", "idx_to_tree",
        "forests", "forest_to_idx", "idx_to_forest",
        "lyndon_forests", "lyndon_forest_to_idx", "idx_to_lyndon_forest",
    ):
        assert getattr(api, name) is getattr(ps, name)
