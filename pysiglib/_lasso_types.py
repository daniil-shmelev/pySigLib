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

from ctypes import Structure, c_uint64, c_uint32, c_int, c_double


class LassoOptions(Structure):
    _fields_ = [(name, c_uint64) for name in (
        'samples', 'length', 'dimension', 'degree', 'targets', 'max_iter',
        'max_epochs', 'batch_size', 'cache_bytes', 'memory_limit')]
    _fields_ += [('tolerance', c_double), ('n_jobs', c_int)]
    _fields_ += [(name, c_uint32) for name in ('full', 'intercept', 'single_precision', 'solver_flags')]


class LassoStats(Structure):
    _fields_ = [(name, c_uint64) for name in (
        'columns', 'evaluated', 'pruned', 'block_queries', 'cache_hits',
        'cache_misses', 'peak_bytes', 'coordinate_sweeps', 'screening_count', 'search_passes', 'constructed')]
    _fields_ += [(name, c_double) for name in (
        'preparation_seconds', 'search_seconds', 'insertion_seconds', 'solve_seconds', 'total_seconds')]


class LassoShape(Structure):
    _fields_ = [('words', c_uint64), ('letters', c_uint64), ('targets', c_uint64), ('stats', LassoStats)]
