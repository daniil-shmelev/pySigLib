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

#include <gtest/gtest.h>
#include "sig_lasso_core.h"

namespace {
using siglasso::Word;

size_t longest_lyndon(const Word& w, size_t start, size_t end) {
    for (size_t k = end; k > start; --k) {
        bool lyndon = true;
        for (size_t j = start + 1; lyndon && j < k; ++j)
            lyndon = std::lexicographical_compare(w.begin() + start, w.begin() + k, w.begin() + j, w.begin() + k);
        if (lyndon) return k;
    }
    return start;
}

TEST(LassoFactorTree, EveryDescendantMatchesThePeriodicInterval) {
    std::vector<Word> level(1);
    for (size_t degree = 1; degree <= 8; ++degree) {
        std::vector<Word> next;
        for (const auto& prefix : level) for (uint64_t a = 0; a < 3; ++a) {
            auto w = prefix;
            w.push_back(a);
            next.push_back(w);
            std::vector<bool> ancestor(degree + 1);
            for (size_t k = degree; k;) {
                ancestor[k] = true;
                size_t last = 0;
                for (size_t j = 0; j < k; j = longest_lyndon(w, j, k)) last = j;
                k = last ? last : longest_lyndon(w, 0, k - 1);
            }
            for (size_t k = 1; k <= degree; ++k) {
                const Word u(w.begin(), w.begin() + k);
                const auto factor = siglasso::last_factor(u);
                size_t last = 0;
                for (size_t j = 0; j < k; j = longest_lyndon(w, j, k)) last = j;
                ASSERT_EQ(last, factor.start);
                int comparison = 0;
                for (size_t j = k; !comparison && j < degree; ++j) {
                    const auto limit = factor.letter(u, j - k);
                    comparison = (w[j] > limit) - (w[j] < limit);
                }
                ASSERT_EQ(ancestor[k], last == 0 || comparison <= 0);
            }
        }
        level.swap(next);
    }
}
} // namespace
