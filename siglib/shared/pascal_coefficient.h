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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#ifdef __CUDACC__
#define PASCAL_HD __host__ __device__
#define PASCAL_INLINE __forceinline__
#elif defined(_MSC_VER)
#define PASCAL_HD
#define PASCAL_INLINE __forceinline
#else
#define PASCAL_HD
#define PASCAL_INLINE inline __attribute__((always_inline))
#endif

namespace pascal_coefficient {

enum Method { horner = 0, pascal_method = 1, automatic = 2 };
enum Variant { scaled = 3, guarded = 4 };

inline uint64_t add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("Pascal size addition overflow");
    return a + b;
}
inline uint64_t mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("Pascal size multiplication overflow");
    return a * b;
}
inline uint64_t environment_option(const char* name, uint64_t fallback) {
#ifdef _WIN32
    char buffer[32];
    const auto size = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    if (!size) return fallback;
    if (size >= sizeof(buffer)) throw std::invalid_argument("Pascal environment option is too long");
    const char* value = buffer;
#else
    const char* value = std::getenv(name);
#endif
    if (!value) return fallback;
    uint64_t bytes = 0;
    for (const char* digit = value; *digit; ++digit) {
        if (*digit < '0' || *digit > '9') throw std::invalid_argument("Pascal environment option must be an unsigned integer");
        bytes = add(mul(bytes, 10), *digit - '0');
    }
    return bytes;
}
inline uint64_t workspace_limit() {
    const uint64_t bytes = environment_option("PYSIGLIB_PASCAL_WORKSPACE_BYTES", 256ULL << 20);
    if (!bytes) throw std::invalid_argument("PYSIGLIB_PASCAL_WORKSPACE_BYTES must be positive");
    return bytes;
}

struct Words {
    std::vector<uint64_t> packed;
    uint64_t count, width = 0, max_degree = 0, index_count = 0;
    Words(const uint64_t* indices, const uint64_t* degrees, uint64_t size, uint64_t dimension, bool prefixes)
        : count(size) {
        if (!dimension || !size || !degrees) throw std::invalid_argument("Invalid coefficient dimensions");
        for (uint64_t w = 0; w < size; ++w) index_count = add(index_count, degrees[w]);
        if (index_count && !indices) throw std::invalid_argument("Missing coefficient word indices");
        packed.resize(add(mul(size, 3), index_count));
        uint64_t offset = 0;
        for (uint64_t w = 0; w < size; ++w) {
            packed[w] = degrees[w];
            packed[size + w] = offset;
            packed[2 * size + w] = width;
            max_degree = std::max(max_degree, degrees[w]);
            width = add(width, prefixes ? std::max<uint64_t>(degrees[w], 1) : 1);
            offset += degrees[w];
        }
        for (uint64_t i = 0; i < index_count; ++i) {
            if (indices[i] >= dimension) throw std::invalid_argument("Coefficient word index out of range");
            packed[3 * size + i] = indices[i];
        }
    }
    const uint64_t* word(uint64_t w) const { return packed.data() + 3 * count + packed[count + w]; }
};

struct Checkpoints {
    uint64_t block, blocks, count, state_count;
    Checkpoints(uint64_t n, uint64_t m, uint64_t cutoff = 64) : block(1), blocks(n), count(0), state_count(m ? m - 1 : 0) {
        if (n > cutoff) {
            block = static_cast<uint64_t>(std::sqrt(static_cast<long double>(n)));
            if (!block) block = 1;
            if (n / block > block || n % block) ++block;
            blocks = n / block + (n % block != 0);
        }
        count = add(mul(add(blocks, block > 1 ? block : 0), state_count), mul(block > 1 ? block : blocks, m));
    }
};

template<class T>
struct ScalarOps {
    using V = T;
    using Scalar = T;
    static constexpr T maximum = std::numeric_limits<T>::max();
    static constexpr T minimum = std::numeric_limits<T>::min();
    PASCAL_HD static T set(T x) { return x; }
    PASCAL_HD static T madd(T x, T y, T z) { return x * y + z; }
    PASCAL_HD static bool finite(T x) { return x <= maximum && x >= -maximum; }
    PASCAL_HD static bool all_finite(const T* x, uint64_t count) {
        bool valid = true;
        for (uint64_t j = 0; j < count; ++j) valid = finite(x[j]) && valid;
        return valid;
    }
    PASCAL_HD static bool tiny(T raw, T x) {
        return raw != T(0) && x < minimum && x > -minimum;
    }
    PASCAL_HD static T mul_div(T x, T y, uint64_t j) {
        const T bound = T(j) * minimum;
        return x < bound && x > -bound ? (x * y) / T(j) : (x / T(j)) * y;
    }
};

template<class T>
PASCAL_HD PASCAL_INLINE T point(const T* path, uint64_t vertex, uint64_t channel, uint64_t dimension,
    uint64_t n, bool time_aug, bool lead_lag, T end_time) {
    if (time_aug && channel == dimension * (lead_lag ? 2 : 1))
        return n ? T(vertex) * (end_time / T(n)) : T(0);
    const uint64_t row = lead_lag ? vertex / 2 + ((vertex % 2) && channel >= dimension) : vertex;
    return path[row * dimension + (lead_lag && channel >= dimension ? channel - dimension : channel)];
}


inline int variant(int method, uint64_t degree) {
    if (method < horner || method > automatic) throw std::invalid_argument("Invalid coefficient method");
    if (method == horner) return horner;
    if (degree > 20) return guarded;
    return scaled;
}

} // namespace pascal_coefficient
