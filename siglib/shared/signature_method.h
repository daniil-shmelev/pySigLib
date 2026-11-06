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
#include <stdexcept>
#include <type_traits>

namespace signature_method {

enum Method { standard = 0, prefix_suffix = 1, automatic = 2 };

template<class T>
bool use_prefix_suffix(int method, uint64_t ps_split, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t degree, bool horner, bool augmented, uint64_t correction_len,
    bool cuda, bool backward) {
    if (method < standard || method > automatic)
        throw std::invalid_argument("Invalid signature method");
    if (ps_split && (method != prefix_suffix || ps_split > degree))
        throw std::invalid_argument("A prefix length from 1 to degree requires the prefix-suffix method");
    if (method == standard) return false;
    if (!horner) throw std::invalid_argument("Prefix-suffix and automatic selection require horner=true");
    if (method == prefix_suffix) return true;

    if (!std::is_same_v<T, float> || augmented || !batch || batch > 32 ||
        length < 17 || length > 4097 || dimension < 2 || dimension > 32 || degree < 3 || degree > 16)
        return false;
    const uint64_t segments = length - 1;
    if (cuda && batch * segments < 512) return false;

    constexpr uint64_t limit = 1ULL << 20;
    uint64_t width = 1;
    for (uint64_t k = 1; k <= degree; ++k) {
        width *= dimension;
        if (width > limit || (k == (degree + 1) / 2 && segments * width > limit)) return false;
    }
    if (correction_len)
        return !cuda && correction_len == dimension * dimension && degree >= 2 * dimension + 1 && width >= (1ULL << 17);
    const unsigned exponent = cuda ? (backward ? 16 : 19) : (backward ? 11 : 17);
    return width >= (1ULL << exponent);
}

} // namespace signature_method
