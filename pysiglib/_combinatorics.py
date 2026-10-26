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

"""Shared validation for coordinate enumeration and indexing."""

from .param_checks import check_non_neg, check_type


def _check_parameters(dimension, degree):
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")


def _check_scalar_term(scalar_term):
    check_type(scalar_term, "scalar_term", bool)


def _check_index(idx, size):
    check_type(idx, "idx", int)
    if not 0 <= idx < size:
        raise ValueError("idx is out of range for the requested coordinate family")
