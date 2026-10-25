/* Copyright 2026 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once
#include <cstdint>
namespace siglasso { struct Result; }

struct SigLassoOptions {
    uint64_t samples, length, dimension, degree, targets;
    uint64_t max_iter, max_epochs, batch_size, cache_bytes, memory_limit;
    double tolerance;
    int n_jobs;
    uint32_t full, intercept, single_precision, solver_flags;
};

struct SigLassoStats {
    uint64_t columns, evaluated, pruned, block_queries, cache_hits, cache_misses;
    uint64_t peak_bytes, coordinate_sweeps, screening_count, search_passes;
    uint64_t constructed;
    double preparation_seconds, search_seconds, insertion_seconds, solve_seconds, total_seconds;
};

struct SigLassoShape {
    uint64_t words, letters, targets;
    SigLassoStats stats;
};
