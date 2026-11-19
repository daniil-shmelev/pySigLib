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
#include "prefix_suffix.h"

namespace prefix_suffix_selection {

struct Choice { uint64_t a = 0, block = 0; };

inline bool automatic() {
    using prefix_suffix::option;
    return option("PYSIGLIB_PS_ENDPOINT_BLOCK", UINT64_MAX, true) == UINT64_MAX
        && !option("PYSIGLIB_PS_WORKSPACE_BYTES", 0, true);
}

inline uint64_t block_size(uint64_t batch, uint64_t segments, uint64_t workers, uint64_t minimum, uint64_t maximum) {
    uint64_t block = std::min(segments, maximum);
    const uint64_t blocks_per_path = (workers - 1) / batch + 1;
    while (block > minimum && (segments - 1) / block + 1 < blocks_per_path)
        block = std::max(minimum, block / 2);
    return block;
}

inline Choice select_cuda(bool backward, uint64_t batch, uint64_t segments,
    const prefix_suffix::Layout& l, int parallelism, bool augmented) {
    if (augmented || segments < 128) return {};
    Choice choice;
    if (backward && l.m >= 3 && prefix_suffix::mul(batch, segments) >= 4096) {
        if (l.m == 3 && l.d <= 16) choice.a = 1;
        else {
            const uint64_t limit = 4 * std::min<uint64_t>(16, l.power[l.h]);
            uint64_t a = (l.m + 1) / 2;
            while (a > 1 && l.power[l.m - a + 1] <= limit) --a;
            if (a < (l.m + 1) / 2) choice.a = a;
        }
    } else if (!backward && segments >= 256 && l.m >= 3) {
        const uint64_t width = l.power[l.m - 1];
        if (batch >= 8 && ((l.m == 3 && l.d >= 16 && width <= 4096 && segments / l.d >= 16) || (l.m >= 7 && width <= 128)))
            choice.block = block_size(batch, segments, parallelism, 16, 256);
    }
    if (!choice.a && !choice.block) return {};
    return automatic() ? choice : Choice{};
}

} // namespace prefix_suffix_selection
