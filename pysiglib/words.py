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

"""Word and Lyndon-word coordinates in native numerical output order."""

import itertools
from functools import cache

from .param_checks import (
    _check_basis_index,
    _check_basis_parameters,
    check_type,
    check_word,
)


def words_of_length(dimension: int, length: int, *, scalar_term: bool = False) -> list[tuple[int, ...]]:
    """Return words of exactly ``length`` letters, in lexicographic order.

    :param dimension: Alphabet size; letters range from zero to ``dimension - 1``.
    :param length: Exact word length.
    :param scalar_term: Include the empty word when ``length=0``. Default ``False``.
    :return: Fresh list of word tuples.
    """
    _check_basis_parameters(dimension, length)
    check_type(scalar_term, "scalar_term", bool)
    if length == 0 and not scalar_term:
        return []
    return list(itertools.product(range(dimension), repeat=length))


def words(dimension: int, degree: int, *, scalar_term: bool = False) -> list[tuple[int, ...]]:
    """Return signature coordinate labels through ``degree`` in numerical output order.

    Words are ordered by length, then lexicographically. Increasing ``degree`` preserves the existing prefix and indices. The same labels index expanded log signatures (method 0).

    :param dimension: Alphabet size; use the effective dimension after any path augmentation.
    :param degree: Inclusive maximum word length.
    :param scalar_term: Include the empty word ``()`` at index zero. Default ``False``.
    :return: Fresh list of word tuples, matching :func:`idx_to_word` under the same options.

    Example::

        >>> words(2, 2)
        [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
        >>> words(2, 0, scalar_term=True)
        [()]
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    return [
        word
        for length in range(0 if scalar_term else 1, degree + 1)
        for word in itertools.product(range(dimension), repeat=length)
    ]


@cache
def _lyndon_words_of_length(dimension, length):
    if dimension == 0 or length == 0:
        return ()
    result = []
    word = [0]
    while word:
        period = len(word)
        if period == length:
            result.append(tuple(word))
        while len(word) < length:
            word.append(word[-period])
        while word and word[-1] == dimension - 1:
            word.pop()
        if word:
            word[-1] += 1
    return tuple(result)


def lyndon_words_of_length(dimension: int, length: int) -> list[tuple[int, ...]]:
    """Return Lyndon words of exactly ``length`` letters, in lexicographic order.

    :param dimension: Alphabet size.
    :param length: Exact word length. Zero returns an empty list.
    :return: Fresh list of nonempty Lyndon-word tuples.
    """
    _check_basis_parameters(dimension, length)
    return list(_lyndon_words_of_length(dimension, length))


@cache
def _lyndon_words(dimension, degree):
    return tuple(
        word
        for length in range(1, degree + 1)
        for word in _lyndon_words_of_length(dimension, length)
    )


def lyndon_words(dimension: int, degree: int) -> list[tuple[int, ...]]:
    """Return compressed log-signature labels through ``degree`` in numerical output order.

    The order is word length, then lexicographic order. Method 1 returns selected expanded word coefficients; methods 2 and 3 return coefficients in the standard Lyndon bracket basis with these same labels. Full and compressed indices are distinct. Increasing ``degree`` preserves the existing prefix and indices.

    :param dimension: Alphabet size; use the effective dimension after any path augmentation.
    :param degree: Inclusive maximum word length.
    :return: Fresh list of Lyndon-word tuples. The empty word is always excluded.
    """
    _check_basis_parameters(dimension, degree)
    return list(_lyndon_words(dimension, degree))


def _is_lyndon(word):
    return bool(word) and all(word < word[start:] for start in range(1, len(word)))


def is_lyndon(word: tuple[int, ...]) -> bool:
    """Return whether a tuple of nonnegative integer letters is a Lyndon word.

    The empty word is not Lyndon. Letters are compared by their integer values.
    """
    check_word(word, float("inf"), "word")
    return _is_lyndon(word)


def word_to_idx(word: tuple[int, ...], dimension: int, degree: int, *, scalar_term: bool = False) -> int:
    """Return a word's index in the full signature or expanded log-signature coordinates.

    :param word: Tuple of integer letters in ``[0, dimension)``.
    :param dimension: Alphabet size.
    :param degree: Inclusive maximum word length; words beyond this bound are rejected.
    :param scalar_term: Include the empty word at index zero. Default ``False``.
    :return: Index in :func:`words` with the same options.
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    check_word(word, dimension, "word")
    if len(word) > degree:
        raise ValueError("word exceeds the requested degree")
    if not word and not scalar_term:
        raise ValueError("The empty word has no index with scalar_term=False")
    idx = 0
    for letter in word:
        idx = idx * dimension + letter + 1
    return idx if scalar_term else idx - 1


def idx_to_word(idx: int, dimension: int, degree: int, *, scalar_term: bool = False) -> tuple:
    """Return the word at ``idx`` in the full coordinates; inverse of :func:`word_to_idx`.

    :param idx: Nonnegative flat coordinate index.
    :param dimension: Alphabet size.
    :param degree: Inclusive maximum word length; indices beyond this bound are rejected.
    :param scalar_term: Include the empty word at index zero. Default ``False``.
    :return: Word tuple, equal to ``words(dimension, degree, scalar_term=scalar_term)[idx]``.
    """
    _check_basis_parameters(dimension, degree)
    check_type(scalar_term, "scalar_term", bool)
    count = degree if dimension == 1 else (dimension ** (degree + 1) - dimension) // (dimension - 1)
    _check_basis_index(idx, count + int(scalar_term))
    if not scalar_term:
        idx += 1
    word = []
    while idx:
        idx, letter = divmod(idx - 1, dimension)
        word.append(letter)
    return tuple(reversed(word))


@cache
def _lyndon_word_indices(dimension, degree):
    return {word: idx for idx, word in enumerate(_lyndon_words(dimension, degree))}


def lyndon_word_to_idx(word: tuple[int, ...], dimension: int, degree: int) -> int:
    """Return a Lyndon word's compressed log-signature index.

    :param word: Nonempty Lyndon word over ``[0, dimension)``.
    :param dimension: Alphabet size.
    :param degree: Inclusive maximum word length.
    :return: Index in :func:`lyndon_words`, distinct from the full :func:`word_to_idx` index.
    """
    _check_basis_parameters(dimension, degree)
    check_word(word, dimension, "word")
    if len(word) > degree or not _is_lyndon(word):
        raise ValueError("word is not in the requested Lyndon-word basis")
    return _lyndon_word_indices(dimension, degree)[word]


def idx_to_lyndon_word(idx: int, dimension: int, degree: int) -> tuple:
    """Return a compressed log-signature label; inverse of :func:`lyndon_word_to_idx`.

    :param idx: Nonnegative compressed coordinate index.
    :param dimension: Alphabet size.
    :param degree: Inclusive maximum word length.
    :return: Word tuple at ``lyndon_words(dimension, degree)[idx]``. There is no scalar coordinate.
    """
    _check_basis_parameters(dimension, degree)
    basis = _lyndon_words(dimension, degree)
    _check_basis_index(idx, len(basis))
    return basis[idx]
