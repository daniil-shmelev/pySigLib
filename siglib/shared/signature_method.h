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
bool use_prefix_suffix_cuda(uint64_t batch, uint64_t dimension, uint64_t length,
    uint64_t degree, bool augmented, uint64_t correction_len, bool backward) {
    if (augmented || correction_len || !batch || batch > 128 || length < 17 || length > 4097
        || dimension < 2 || dimension > 32 || degree < 3 || degree > 16) return false;
    const uint64_t segments = length - 1, work = batch * segments;
    constexpr uint64_t limit = 1ULL << 24;
    uint64_t width = 1;
    for (uint64_t k = 1; k <= degree; ++k) {
        if (width > limit / dimension) return false;
        width *= dimension;
        if (k == (degree + 1) / 2 && width > limit / segments) return false;
    }
    // The grouped Horner kernels change the crossover with matrix multiplication.
    const bool grouped = (dimension <= 4 || dimension == 6 || dimension == 8 || dimension == 10
        || dimension == 12 || dimension == 16 || dimension == 24 || dimension == 32)
        && degree >= (dimension <= 4 ? 6 : 4) && degree <= (dimension <= 4 ? 16 : 8);
    if constexpr (std::is_same_v<T, double>) {
        if (!backward) return dimension > 4 && degree >= 4 && work >= 512
            && ((!grouped && width >= 4096) || (width >= 131072 && degree * (degree - 1) >= 3 * dimension));
        if (dimension == 2)
            return degree >= (batch <= 4 ? 10 : 12) && (segments >= 128 || width >= 16384)
                && (batch <= 32 || width >= 65536);
        if (dimension == 3)
            return degree >= (batch <= 4 ? 8 : 9) && (degree >= 10 || segments >= 64);
        if (dimension == 4)
            return degree >= 5 && (segments >= 64 || work >= 512 || width >= 65536);
        return width >= 4096 && (batch <= 32 || width >= 16384);
    }
    if (!backward) return dimension > 4 && width >= 65536 && batch * width >= 2097152 && work >= 512;
    if (!grouped) return width >= 4096 && work >= 128;
    if (dimension == 2) return false;
    if (dimension == 3) return width >= 196608 && (work >= 512 || (width >= 1048576 && work >= 128));
    if (dimension == 4) return width >= 65536 && work >= 128;
    return segments >= 64 && ((width >= 65536 && work >= 128)
        || (dimension >= 8 && width >= 32768 && work >= 4096));
}

template<class T>
bool use_prefix_suffix(int method, uint64_t ps_split, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t degree, bool augmented, uint64_t correction_len,
    bool cuda, bool backward) {
    if (method < standard || method > automatic)
        throw std::invalid_argument("Invalid signature method");
    if (ps_split && (method != prefix_suffix || ps_split > degree))
        throw std::invalid_argument("A prefix length from 1 to degree requires the prefix-suffix method");
    if (method == standard) return false;
    if (method == prefix_suffix) return true;
    if (cuda) return use_prefix_suffix_cuda<T>(batch, dimension, length, degree, augmented, correction_len, backward);

    if (!std::is_same_v<T, float> || augmented || !batch || batch > 32 ||
        length < 17 || length > 4097 || dimension < 2 || dimension > 32 || degree < 3 || degree > 16)
        return false;
    const uint64_t segments = length - 1;

    constexpr uint64_t limit = 1ULL << 20;
    uint64_t width = 1;
    for (uint64_t k = 1; k <= degree; ++k) {
        width *= dimension;
        if (width > limit || (k == (degree + 1) / 2 && segments * width > limit)) return false;
    }
    if (correction_len)
        return correction_len == dimension * dimension && degree >= 2 * dimension + 1 && width >= (1ULL << 17);
    const unsigned exponent = backward ? 11 : 17;
    return width >= (1ULL << exponent);
}

} // namespace signature_method
