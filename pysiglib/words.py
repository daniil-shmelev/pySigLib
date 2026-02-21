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

import itertools
from functools import cache
from .param_checks import check_type, check_non_neg, check_word

def words_of_length(
        alphabet_size : int,
        length : int
) -> list[tuple[int, ...]]:
    """
    Returns all words of a given length.

    :param alphabet_size: Size of the alphabet.
    :type alphabet_size: int
    :param length: Length of words.
    :type length: int
    :return: All words of the given length.
    :rtype: list[tuple[int, ...]]

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        # All words of length 2 over alphabet {0, 1}
        w = pysiglib.words_of_length(2, 2)
        print(w) # [(0, 0), (0, 1), (1, 0), (1, 1)]

    """
    check_type(alphabet_size, "alphabet_size", int)
    check_type(length, "length", int)
    check_non_neg(alphabet_size, "alphabet_size")
    check_non_neg(length, "length")

    d = list(range(alphabet_size))
    return list(itertools.product(d, repeat = length))

def words(
        alphabet_size : int,
        max_length : int
) -> list[tuple[int, ...]]:
    """
    Returns all words up to a given length.

    :param alphabet_size: Size of the alphabet.
    :type alphabet_size: int
    :param max_length: Max length of words.
    :type max_length: int
    :return: All words up to the given length.
    :rtype: list[tuple[int, ...]]

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        # All words up to length 2 over alphabet {0, 1}
        w = pysiglib.words(2, 2)
        print(w) # [(), (0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]

    """
    check_type(alphabet_size, "alphabet_size", int)
    check_type(max_length, "max_length", int)
    check_non_neg(alphabet_size, "alphabet_size")
    check_non_neg(max_length, "max_length")

    d = list(range(alphabet_size))
    res = [tuple()]

    for level in range(1, max_length + 1):
        res += list(itertools.product(d, repeat = level))
    return res

def lyndon_words_of_length(
        alphabet_size: int,
        length: int
) -> list[tuple[int, ...]]:
    """
    Returns all Lyndon words of a given length.

    :param alphabet_size: Size of the alphabet.
    :type alphabet_size: int
    :param length: Length of words.
    :type length: int
    :return: All Lyndon words of the given length.
    :rtype: list[tuple[int, ...]]

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        # All Lyndon words of length 3 over alphabet {0, 1}
        w = pysiglib.lyndon_words_of_length(2, 3)
        print(w) # [(0, 0, 1), (0, 1, 1)]

    """
    check_type(alphabet_size, "alphabet_size", int)
    check_type(length, "length", int)
    check_non_neg(alphabet_size, "alphabet_size")
    check_non_neg(length, "length")

    res = []
    word = [0]
    while word:
        m = len(word)
        if m == length:
            res.append(tuple(word))

        while len(word) < length:
            word.append(word[-m])

        while word and word[-1] == alphabet_size - 1:
            word.pop()

        if word:
            word[-1] += 1

    return res


def lyndon_words(
        alphabet_size: int,
        max_length: int
) -> list[tuple[int, ...]]:
    """
    Returns all Lyndon words up to a given length.

    :param alphabet_size: Size of the alphabet.
    :type alphabet_size: int
    :param max_length: Max length of words.
    :type max_length: int
    :return: All Lyndon words up to the given length.
    :rtype: list[tuple[int, ...]]

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        # All Lyndon words up to length 3 over alphabet {0, 1}
        w = pysiglib.lyndon_words(2, 3)
        print(w) # [(0,), (1,), (0, 1), (0, 0, 1), (0, 1, 1)]

    """
    check_type(alphabet_size, "alphabet_size", int)
    check_type(max_length, "max_length", int)
    check_non_neg(alphabet_size, "alphabet_size")
    check_non_neg(max_length, "max_length")

    res = []
    for level in range(1, max_length + 1):
        res += lyndon_words_of_length(alphabet_size, level)
    return res

@cache
def is_lyndon(
    word : tuple[int, ...]
) -> bool:
    """
    Checks if a given word is a Lyndon word.

    :param word: Word
    :type word: tuple[int, ...]
    :return: Returns ``True`` if ``word`` is a Lyndon word and ``False`` otherwise.
    :rtype: bool

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        print(pysiglib.is_lyndon((0, 1)))    # True
        print(pysiglib.is_lyndon((1, 0)))    # False
        print(pysiglib.is_lyndon((0, 0, 1))) # True

    """
    check_word(word, float('inf'), "word")
    n = len(word)
    if n == 0:
        return False
    if n == 1:
        return True

    for i in range(1, n):
        if not (word < word[i:]):
            return False

    return True

@cache
def word_to_idx(
        word : tuple[int, ...],
        alphabet_size : int
) -> int:
    """
    Given a word :math:`(w_0, w_1, \\ldots, w_n)`, returns the corresponding flattened index

    .. math::

        \\sum_{i=0}^n (w_i + 1) d^i

    :param word: Word
    :type word: tuple[int, ...]
    :param alphabet_size: Size of the alphabet
    :type alphabet_size: int
    :return: Flattened index corresponding to word
    :rtype: int

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        length, dimension, degree = 100, 2, 3
        x = torch.rand(size=(length, dimension))
        sig = pysiglib.sig(x, degree)

        # Get coefficient at word
        word = (0, 1)
        coef_idx = pysiglib.extract_sig_coef(word, sig, dimension, degree)

        # Convert word to index and get coefficient
        idx = pysiglib.word_to_idx(word, dimension)
        coef_word = sig[idx]

        # Both methods give the same coefficient
        print("Idx: ", idx, " Coefficient: ", coef_idx)
        print("Word: ", word, " Coefficient: ", coef_word)

    """
    check_type(alphabet_size, "alphabet_size", int)
    check_non_neg(alphabet_size, "alphabet_size")
    check_word(word, alphabet_size, "word")

    if not word:
        return 0

    idx = 0
    for i in word:
        idx = idx * alphabet_size + (i+1)
    return idx

@cache
def idx_to_word(
        idx : int,
        alphabet_size : int
) -> tuple[int, ...]:
    """
    Given a flattened index

    .. math::

        \\sum_{i=0}^n (w_i + 1) d^i,

    returns the corresponding word :math:`(w_0, w_1, \\ldots, w_n)`.

    :param idx: Flattened index
    :type idx: int
    :param alphabet_size: Size of the alphabet
    :type alphabet_size: int
    :return: Word corresponding to flattened index
    :rtype: tuple[int, ...]

    Example:
    ---------

    .. code-block:: python

        import torch
        import pysiglib

        length, dimension, degree = 100, 2, 3
        x = torch.rand(size=(length, dimension))
        sig = pysiglib.sig(x, degree)

        # Get coefficient at index
        idx = 4
        coef_idx = sig[idx]

        # Convert index to word and get coefficient at word
        word = pysiglib.idx_to_word(idx, dimension)
        coef_word = pysiglib.extract_sig_coef(word, sig, dimension, degree)

        # Both methods give the same coefficient
        print("Idx: ", idx, " Coefficient: ", coef_idx)
        print("Word: ", word, " Coefficient: ", coef_word)

    """
    word = []

    while idx:
        idx, r = divmod(idx, alphabet_size)
        if not r:
            r = alphabet_size
            idx -= 1
        word.append(r - 1)

    return tuple(reversed(word))

