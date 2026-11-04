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
#include "sig_lasso_core.h"

namespace siglasso {
inline bool lyndon(const Word& w) {
    for (size_t k = 1; k < w.size(); ++k)
        if (!std::lexicographical_compare(w.begin(), w.end(), w.begin() + k, w.end())) return false;
    return !w.empty();
}

struct Hall {
    Word word;
    std::shared_ptr<Hall> left, right;
    size_t run = 0;
    size_t slot = 0;
};
using HallPtr = std::shared_ptr<Hall>;
inline int compare(const HallPtr& a, const HallPtr& b) {
    if (a == b) return 0;
    if (a->word.size() != b->word.size()) return a->word.size() > b->word.size() ? -1 : 1;
    if (!a->left) return a->word < b->word ? -1 : a->word == b->word ? 0 : 1;
    const int first = compare(a->left, b->left);
    return first ? first : compare(a->right, b->right);
}
inline bool admissible(const HallPtr& u, const HallPtr& v) {
    return compare(u, v) < 0 && (!u->right || compare(v, u->right) <= 0);
}
inline HallPtr bracket(const HallPtr& u, const HallPtr& v) {
    auto h = std::make_shared<Hall>(); h->left = u; h->right = v;
    h->word = u->word; h->word.insert(h->word.end(), v->word.begin(), v->word.end());
    h->run = u->right && compare(u->right, v) == 0 ? u->run + 1 : 1;
    return h;
}

} // namespace siglasso
