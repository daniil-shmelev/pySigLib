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
#include <array>
#include <numeric>

namespace prefix_suffix {

// Words that differ only by a permutation have the same increment monomial.
struct SymmetricPower {
    uint64_t size;
    std::vector<unsigned> representative, group, offsets, members;

    SymmetricPower(unsigned d, unsigned degree, unsigned count) : size(count), group(count), members(count) {
        std::array<unsigned, 32> letters{};
        for (unsigned word = 0; word < count; ++word) {
            unsigned value = word;
            for (unsigned j = 0; j < degree; ++j) {
                letters[j] = value % d;
                value /= d;
            }
            std::sort(letters.begin(), letters.begin() + degree);
            unsigned key = 0;
            for (unsigned j = 0; j < degree; ++j) key = key * d + letters[j];
            if (key == word) {
                group[word] = static_cast<unsigned>(representative.size());
                representative.push_back(word);
            } else group[word] = group[key];
        }
        offsets.resize(representative.size() + 1);
        for (unsigned g : group) ++offsets[g + 1];
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        auto cursor = offsets;
        for (unsigned word = 0; word < count; ++word) members[cursor[group[word]]++] = word;
    }
};

struct SymmetricPlan {
    std::vector<SymmetricPower> powers;
    uint64_t scratch = 0;
    static constexpr uint64_t tile_bytes = 2ULL << 20;

    static bool use(uint64_t n, uint64_t a, uint64_t p, uint64_t q) {
        return n >= 32 && a >= 2 && p <= 32768 && q >= 8 && p * q >= 32768;
    }

    SymmetricPlan(const Layout& l, uint64_t n, uint64_t rows, uint64_t itemsize) {
        const uint64_t top = l.left_degree(l.m);
        if (rows < 4 * top || l.scan_degree == l.m || !use(n, top, l.power[top], l.power[l.m - top])) return;
        for (unsigned r = 2; r <= top; ++r)
            powers.emplace_back(static_cast<unsigned>(l.d), r, static_cast<unsigned>(l.power[r]));
        for (uint64_t k = l.scan_degree + 1; k <= l.m; ++k) {
            const uint64_t a = l.left_degree(k), p = l.power[a], q = l.power[k - a];
            if (!use(n, a, p, q)) continue;
            const uint64_t height = std::min(n, (rows - 1) / a + 1) + q;
            for (uint64_t r = 2; r <= a; ++r) {
                const auto& map = powers[r - 2];
                const uint64_t unit = mul(height, map.representative.size());
                if (unit > tile_bytes / itemsize) {
                    powers.clear();
                    scratch = 0;
                    return;
                }
                const uint64_t heads = std::min(p / map.size, tile_bytes / itemsize / unit);
                scratch = std::max(scratch, mul(heads, unit));
            }
        }
    }
};
} // namespace prefix_suffix
