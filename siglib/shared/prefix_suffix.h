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
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace prefix_suffix {

inline uint64_t mul(uint64_t x, uint64_t y) {
    if (y && x > std::numeric_limits<uint64_t>::max() / y)
        throw std::overflow_error("prefix-suffix size multiplication overflow");
    return x * y;
}

inline uint64_t add(uint64_t x, uint64_t y) {
    if (x > std::numeric_limits<uint64_t>::max() - y)
        throw std::overflow_error("prefix-suffix size addition overflow");
    return x + y;
}

inline uint64_t option(const char* name, uint64_t fallback, bool allow_zero = false) {
#ifdef _WIN32
    char buffer[32];
    const auto size = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    if (!size) return fallback;
    if (size >= sizeof(buffer)) throw std::invalid_argument(std::string(name) + " is too long");
    const char* text = buffer;
#else
    const char* text = std::getenv(name);
#endif
    if (!text || !*text) return fallback;
    uint64_t value = 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9') throw std::invalid_argument(std::string(name) + " must be a positive integer");
        value = add(mul(value, 10), *text - '0');
    }
    if (!value && !allow_zero) throw std::invalid_argument(std::string(name) + " must be positive");
    return value;
}

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int small_power(int d, int k) {
    int value = 1;
    for (int i = 0; i < k; ++i) value *= d;
    return value;
}

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int small_offset(int d, int k) {
    int value = 0, power = 1;
    for (int i = 0; i < k; ++i) { value += power; power *= d; }
    return value;
}

struct Layout {
    uint64_t d, m, h, scan_degree, prefix_degree;
    std::vector<uint64_t> power, offset;
    Layout(uint64_t dimension, uint64_t degree) : d(dimension), m(degree), h(degree / 2), prefix_degree(degree / 2 + degree % 2 - (degree != 0)) {
        if (!d) throw std::invalid_argument("invalid prefix-suffix dimension");
        const uint64_t count = add(m, 2);
        scan_degree = option("PYSIGLIB_PS_SCAN_DEGREE", h, true);
        if (scan_degree < h || scan_degree > m)
            throw std::invalid_argument("PYSIGLIB_PS_SCAN_DEGREE must be between floor(depth/2) and depth");
        mul(count, sizeof(uint64_t));
        if (d > 1 && m >= 64) throw std::overflow_error("prefix-suffix tensor power overflow");
        power.resize(count - 1);
        offset.resize(count);
        power[0] = 1;
        for (uint64_t k = 0; k <= m; ++k) {
            if (k) power[k] = mul(power[k - 1], d);
            offset[k + 1] = add(offset[k], power[k]);
        }
    }
};

template<typename T>
void right_factor(T* result, const T* suffix, const T* v, uint64_t j, uint64_t b, const Layout& l) {
    result[0] = 1;
    T scale = 1;
    for (uint64_t f = 2; f <= j + 1; ++f) scale /= T(f);
    for (uint64_t level = 1; level <= b; ++level) {
        const uint64_t previous = l.power[level - 1];
        const T reciprocal = T(1) / T(j + b - level + 2);
        // Update channel zero last so every channel reads the previous level.
        for (uint64_t channel = l.d; channel-- > 0;) {
            T* destination = result + channel * previous;
            const T* source = suffix + l.offset[level] + channel * previous;
            const T multiplier = v[channel] * reciprocal;
            if (level == b) {
                for (uint64_t x = 0; x < previous; ++x)
                    destination[x] = (source[x] + multiplier * result[x]) * scale;
            } else {
                for (uint64_t x = 0; x < previous; ++x)
                    destination[x] = source[x] + multiplier * result[x];
            }
        }
    }
    if (!b) result[0] = scale;
}

template<typename T>
void extend_right_factor(T* result, const T* previous, const T* suffix, const T* v,
    uint64_t j, uint64_t b, const Layout& l) {
    T scale = 1;
    for (uint64_t f = 2; f <= j + 1; ++f) scale /= T(f);
    const uint64_t size = l.power[b];
    for (uint64_t channel = 0; channel < l.d; ++channel) {
        T* destination = result + channel * size;
        const T* source = suffix + l.offset[b + 1] + channel * size;
        const T value = v[channel];
        for (uint64_t x = 0; x < size; ++x)
            destination[x] = source[x] * scale + value * previous[x];
    }
}

template<typename T> struct Matrix {
    T* data;
    uint64_t rows, cols, rs, cs;
    uint64_t bs = 0;
    Matrix sub(uint64_t i, uint64_t j, uint64_t r, uint64_t c) const {
        return {data + i * rs + j * cs, r, c, rs, cs, bs};
    }
};

inline uint64_t tile_side(uint64_t p, uint64_t q, uint64_t k, uint64_t max_side) {
    uint64_t s = 1, minimum = std::min({p, q, k});
    while (s <= minimum / 2 && s <= max_side / 2) s *= 2;
    return s;
}

template<typename T, typename Ops>
void product(Ops& ops, Matrix<T> a, Matrix<T> b, Matrix<T> c,
    uint64_t max_side, T* scratch, bool first) {
    const bool oversized = std::max({a.rs, a.cs, b.rs, b.cs, c.rs, a.rows, a.cols, b.cols}) > INT_MAX;
    if (!oversized) {
        ops.gemm(a, b, c, first ? T(0) : T(1));
        return;
    }
    const uint64_t side = tile_side(a.rows, b.cols, a.cols, max_side);
    if (first) ops.clear(c);
    const uint64_t area = side * side;
    for (uint64_t i = 0; i < a.rows; i += side) {
        const uint64_t rows = std::min(side, a.rows - i);
        for (uint64_t j = 0; j < b.cols; j += side) {
            const uint64_t cols = std::min(side, b.cols - j);
            for (uint64_t k = 0; k < a.cols; k += side) {
                const uint64_t inner = std::min(side, a.cols - k);
                Matrix<T> pa{scratch, rows, inner, side, 1, ops.scratch_stride};
                Matrix<T> pb{scratch + area, inner, cols, side, 1, ops.scratch_stride};
                Matrix<T> pc{scratch + 2 * area, rows, cols, side, 1, ops.scratch_stride};
                ops.pack(a.sub(i, k, rows, inner), pa);
                ops.pack(b.sub(k, j, inner, cols), pb);
                ops.gemm(pa, pb, pc, T(0));
                ops.accumulate(pc, c.sub(i, j, rows, cols), T(1));
            }
        }
    }
}

} // namespace prefix_suffix
