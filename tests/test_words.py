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

import pysiglib

def test_lyndon_words_1():
    dimension, degree = 2, 3
    true_ = [(0,), (1,), (0,1), (0,0,1), (0,1,1)]
    res = pysiglib.lyndon_words(dimension, degree)
    assert true_ == res

def test_lyndon_words_2():
    dimension, degree = 5, 2
    true_ = [(0,), (1,), (2,), (3,), (4,), (0,1), (0,2), (0,3), (0,4),
             (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    res = pysiglib.lyndon_words(dimension, degree)
    assert true_ == res

def test_lyndon_words_3():
    dimension, degree = 3, 4
    true_ = [(0,), (1,), (2,),
    (0, 1), (0, 2), (1, 2),
    (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2), (1, 1, 2), (1, 2, 2),
    (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 2, 1), (0, 0, 2, 2),
    (0, 1, 0, 2), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 2, 1), (0, 1, 2, 2), (0, 2, 1, 1),
    (0, 2, 1, 2), (0, 2, 2, 1), (0, 2, 2, 2), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2)]
    res = pysiglib.lyndon_words(dimension, degree)
    assert true_ == res

def test_idx_to_word():
    dimension, degree = 3, 4
    m = pysiglib.sig_length(dimension, degree)

    for idx in range(m):
        w = pysiglib.idx_to_word(idx, dimension)
        i = pysiglib.word_to_idx(w, dimension)
        assert idx == i


def test_word_idx_round_trip_scalar_term_true():
    dimension, degree = 3, 4
    m = pysiglib.sig_length(dimension, degree, scalar_term=True)
    for idx in range(m):
        w = pysiglib.idx_to_word(idx, dimension, scalar_term=True)
        assert pysiglib.word_to_idx(w, dimension, scalar_term=True) == idx


def test_word_idx_shift_between_scalar_term_formats():
    # For scalar_term=True, () is at 0 and non-empty words shift up by 1
    # relative to scalar_term=False indices.
    dimension = 4
    for w in [(0,), (1,), (3,), (0, 1), (3, 2, 1), (0, 1, 2, 3)]:
        idx_false = pysiglib.word_to_idx(w, dimension, scalar_term=False)
        idx_true = pysiglib.word_to_idx(w, dimension, scalar_term=True)
        assert idx_true == idx_false + 1


def test_empty_word_requires_scalar_term():
    assert pysiglib.word_to_idx((), 3, scalar_term=True) == 0
    import pytest
    with pytest.raises(ValueError):
        pysiglib.word_to_idx((), 3, scalar_term=False)


def test_is_lyndon():
    dimension, degree = 3, 4
    words = pysiglib.words(dimension, degree)
    lyndon_words = set(pysiglib.lyndon_words(dimension, degree))

    for w in words:
        assert pysiglib.is_lyndon(w) == (w in lyndon_words)

