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
#include "prefix_suffix_selection.h"

namespace prefix_suffix_endpoint {

inline constexpr uint64_t cpu_budget = 512ULL << 20;
inline constexpr uint64_t gemm_row_grain = 64;

struct CpuWorkspace {
    uint64_t per_worker = 0, workers = 0;

    CpuWorkspace(uint64_t tasks, uint64_t total, uint64_t lower, uint64_t width, uint64_t dimension,
        uint64_t block, uint64_t threads, uint64_t budget_elements) {
        if (!tasks || total > budget_elements / 2 / tasks) return;
        const uint64_t remaining = budget_elements - 2 * tasks * total;
        per_worker = prefix_suffix::add(prefix_suffix::add(lower, width),
            prefix_suffix::mul(block, prefix_suffix::add(width, dimension)));
        workers = std::min({tasks, threads, remaining / per_worker});
    }
};

inline uint64_t select_cpu_block(uint64_t batch, uint64_t segments, const prefix_suffix::Layout& layout,
    uint64_t threads, uint64_t itemsize, bool augmented) {
    if (!batch || augmented || segments < 256 || layout.d < 12 || (layout.m != 2 && layout.m != 3)) return 0;
    const uint64_t block = prefix_suffix_selection::block_size(batch, segments, threads, 64, 1024);
    const uint64_t width = layout.power[layout.m - 1];
    const uint64_t blocks_per_path = (segments - 1) / block + 1;
    const uint64_t tasks = prefix_suffix::mul(batch, blocks_per_path);
    const CpuWorkspace workspace(tasks, layout.offset.back(), layout.offset[layout.m], width, layout.d,
        block, threads, cpu_budget / itemsize);
    if (!workspace.workers) return 0;

    if (layout.m == 2) {
        // Both products have one row per segment. Amortise block storage and merging.
        if (batch < threads || workspace.workers < threads || block / layout.d < 8) return 0;
    } else if (threads > 1) {
        // The balanced top-level product has twice as many rows of factors.
        const uint64_t row_groups = (width - 1) / gemm_row_grain + 1;
        const uint64_t balanced_workers = std::min(threads, prefix_suffix::mul(batch, row_groups));
        const uint64_t merges = std::max<uint64_t>(1, blocks_per_path - 1);
        if (2 * workspace.workers < balanced_workers || segments / merges / layout.d < 2) return 0;
    }
    return prefix_suffix_selection::automatic() ? block : 0;
}

} // namespace prefix_suffix_endpoint
