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

namespace prefix_suffix {

struct LegendreMode {
    unsigned head, depth, groups, cols, first_group;
    double weight[17]{};
    std::vector<unsigned> grow, parent, offsets, members, words;
    uint64_t rows() const { return uint64_t(head) * groups; }
    uint64_t size() const { return rows() * cols; }
};

// Mode r stores all words through degree (depth-r), with consecutive segment
// samples along each row of the factor matrix.
struct MomentRows {
    unsigned depth, size = 0, first[17]{};
    MomentRows(const Layout& l, unsigned depth_) : depth(depth_) {
        for (unsigned r = 0; r <= depth; ++r) {
            first[r] = size;
            size += unsigned(l.offset[depth - r + 1]);
        }
    }
};

// Orthogonal moments of the two segment polynomials. Mode r has a symmetric
// centre of 2r+1 increment letters and ordered words on either side.
struct LegendrePlan {
    unsigned a, b, groups = 0, max_rows = 0;
    uint64_t buffers[2]{};
    std::vector<LegendreMode> modes;

    explicit LegendrePlan(const Layout& l, unsigned split = 0, bool backward = true)
        : a(split ? split : unsigned((l.m + 1) / 2)), b(unsigned(l.m - a)) {
        const unsigned d = unsigned(l.d);
        std::vector<unsigned> keys(d), parents(d, 0), previous(d);
        for (unsigned c = 0; c < d; ++c) previous[c] = keys[c] = c;
        auto insert_letters = [&](const std::vector<unsigned>& shorter) {
            std::vector<unsigned> first(keys.size() + 1), children(keys.size() * d);
            for (size_t g = 0; g < keys.size(); ++g) first[g + 1] = first[g] + d - keys[g] % d;
            std::vector<unsigned> next(first.back()), next_parents(first.back());
            for (unsigned g = 0; g < keys.size(); ++g) {
                const unsigned last = keys[g] % d;
                for (unsigned c = 0; c < last; ++c) {
                    // Insert into the shorter sorted word, then append its former last letter.
                    const unsigned prefix = shorter[parents[g] * d + c];
                    children[g * d + c] = first[prefix] + last - keys[prefix] % d;
                }
                for (unsigned c = last; c < d; ++c) {
                    const unsigned id = first[g] + c - last;
                    next[id] = keys[g] * d + c;
                    next_parents[id] = g;
                    children[g * d + c] = id;
                }
            }
            parents.swap(next_parents);
            keys.swap(next);
            return children;
        };
        for (unsigned r = 0; r < a && r <= b; ++r) {
            auto& mode = modes.emplace_back();
            if (r) {
                auto& prev = modes[r - 1];
                prev.grow.resize(d * keys.size() * d);
                const auto half = insert_letters(previous);
                previous = insert_letters(half);
                mode.parent.resize(keys.size(), UINT_MAX);
                for (unsigned x = 0; x < d; ++x) for (unsigned g = 0; g < prev.groups; ++g) {
                    const unsigned mid = half[g * d + x];
                    for (unsigned y = 0; y < d; ++y) {
                        const unsigned id = previous[mid * d + y], ex = (x * prev.groups + g) * d + y;
                        if (mode.parent[id] == UINT_MAX) mode.parent[id] = ex;
                        prev.grow[ex] = id;
                    }
                }
                if (backward) {
                    mode.offsets.resize(keys.size() + 1);
                    for (unsigned id : prev.grow) ++mode.offsets[id + 1];
                    for (unsigned g = 0; g < keys.size(); ++g) mode.offsets[g + 1] += mode.offsets[g];
                    auto cursor = mode.offsets;
                    mode.members.resize(prev.grow.size());
                    for (unsigned ex = 0; ex < prev.grow.size(); ++ex) mode.members[cursor[prev.grow[ex]]++] = ex;
                }
            }
            mode.head = unsigned(l.power[a - 1 - r]);
            mode.depth = b - r;
            mode.cols = unsigned(l.offset[mode.depth + 1]);
            mode.first_group = groups;
            mode.groups = unsigned(keys.size());
            mode.words.assign(keys.begin(), keys.end());
            groups += mode.groups;
            max_rows = std::max(max_rows, unsigned(mode.rows()));
            buffers[r % 2] = std::max(buffers[r % 2], mode.size());
            mode.weight[0] = 1;
            for (unsigned t = 1; t <= std::max(a - 1 - r, b - r); ++t)
                mode.weight[t] = mode.weight[t - 1] * (r + t) / (t * double(2 * r + t + 1));
        }
    }

    uint64_t tile(uint64_t n, size_t itemsize, bool backward) const {
        const uint64_t bytes = (max_rows + modes.front().cols) * itemsize * (1 + backward);
        return std::min(n, std::max<uint64_t>(64, (512ULL << 10) / bytes / 32 * 32));
    }
    uint64_t workspace(const Layout& l, uint64_t n, uint64_t block, bool backward) const {
        const uint64_t raw = l.offset[a] + l.offset[b + 1];
        return buffers[0] + buffers[1] + (1 + backward) * ((n + 1) * raw + n * (l.d + groups)
            + block * (max_rows + modes.front().cols))
            + 8 * std::max(l.offset[a], l.offset[b + 1]);
    }
};

} // namespace prefix_suffix
