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
    if (augmented || correction_len || !batch || length < 17
        || dimension < 2 || degree < 2 || degree > 16) return false;
    const uint64_t segments = length - 1;
    if (batch > 128) {
        const uint64_t wide_dimension = std::is_same_v<T, double> ? 128 : 96;
        return backward && degree == 2 && batch <= 256 && segments <= 64
            && dimension >= wide_dimension && dimension <= 128;
    }
    if (!backward && batch <= 4) {
        if (dimension == 5 && degree == 3 && batch == 1 && segments >= 3072 && segments <= 8192)
            return true;
        if constexpr (std::is_same_v<T, double>) {
            const bool fused_dimension = dimension == 2 || dimension == 3 || dimension == 4 || dimension == 8;
            if (fused_dimension && degree <= 3 && segments >= 2048 && segments <= 8192) return true;
        } else {
            if (dimension == 8 && degree == 2 && segments >= 2048 && segments <= 8192) return true;
            if (dimension >= 3 && dimension <= 4 && degree == 3 && segments >= 3072 && segments <= 8192)
                return true;
        }
        const uint64_t wide_dimension = std::is_same_v<T, double> ? 48 : 96;
        if (degree == 2 && dimension >= wide_dimension && dimension <= 128
            && segments >= 1024 && segments <= 4096) return true;
    }
    if (!backward && degree == 2 && dimension >= 96 && dimension <= 128
        && batch >= (dimension == 128 ? 8 : 16) && segments >= 256 && segments <= 4096)
        return true;
    if (backward && dimension >= 48 && dimension <= 128 && segments <= 4096) {
        if (degree == 2) {
            if (dimension >= 64 && batch > 16 && segments >= 128) return true;
            if constexpr (std::is_same_v<T, double>)
                return dimension == 128 && (batch >= 128 || (batch >= 3 && segments >= 64));
            return (dimension >= 96 && batch >= 64)
                || (dimension == 128 && batch >= 8 && segments >= 256);
        }
        if (degree == 3 && dimension <= 64 && batch <= 32 && segments >= 64 && segments <= 1024)
            return true;
    }
    if (length > 4097 || dimension > 32 || degree < 3) return false;
    const uint64_t work = batch * segments;
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
        && degree >= (dimension <= 4 ? 6 : 4) && degree <= (dimension <= 4 ? 16 : (dimension == 6 ? 9 : 8));
    const bool dimension_six_tree = dimension == 6 && degree >= 7 && degree <= 9;
    if (backward) {
        if (dimension == 5 && degree == 5 && segments >= 96) return true;
        if (degree == 3 && dimension >= 12 && dimension <= 24 && batch <= 4 && segments >= 512)
            return false;
        if (!grouped && dimension >= 5 && dimension <= 12 && batch >= 16
            && segments >= 256 && width >= 1728)
            return true;
        if (dimension >= 8 && dimension <= 12 && degree >= 4 && segments >= 1024
            && batch <= 4 && width >= 10000)
            return true;
    }
    if constexpr (std::is_same_v<T, double>) {
        if (!backward && dimension > 4 && degree >= 4 && !grouped && work >= 256 && width >= 32768)
            return true;
        if (!backward && dimension == 3 && degree >= 9 && batch <= 4 && segments >= 1024) return true;
        if (backward && dimension == 3 && degree == 5 && batch >= 8 && segments >= 256) return true;
        if (!backward) return !dimension_six_tree && dimension > 4 && degree >= 4 && work >= 512
            && ((!grouped && width >= 4096) || (width >= 131072 && degree * (degree - 1) >= 3 * dimension));
        if (batch <= 32 && dimension == 2 && degree >= 14 && segments <= 64)
            return false;
        if (batch <= 32 && dimension == 3 && degree >= 10 && degree <= 12
            && (segments <= 64 || (batch >= 8 && segments <= 256)))
            return false;
        if (batch <= 32 && dimension_six_tree && segments <= 64) return false;
        if (dimension == 2)
            return degree >= (batch <= 4 ? 10 : 12) && (segments >= 128 || width >= 16384)
                && (batch <= 32 || width >= 65536);
        if (dimension == 3)
            return degree >= (batch <= 4 ? 8 : 9) && (degree >= 10 || segments >= 64);
        if (dimension == 4)
            return degree >= 5 && (segments >= 64 || work >= 512 || width >= 65536);
        return width >= 4096 && (batch <= 32 || width >= 16384);
    }
    if (!backward) return !(dimension_six_tree && segments >= 64)
        && dimension > 4 && width >= 65536 && batch * width >= 2097152 && work >= 512;
    if (!grouped) return width >= 4096 && work >= 128;
    if (dimension == 2) return false;
    if (dimension == 3) return width >= 196608 && (work >= 512 || (width >= 1048576 && work >= 128));
    if (dimension == 4) return width >= 65536 && work >= 128;
    if (dimension_six_tree && (degree == 7 || (degree == 8 && (segments < 64 || batch < 8)))) return false;
    return (segments >= 64 || width >= 262144) && ((width >= 65536 && work >= 128)
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
