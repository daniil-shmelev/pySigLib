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
#define WORD_HD __host__ __device__
#define WORD_INLINE __forceinline__
#elif defined(_MSC_VER)
#define WORD_HD
#define WORD_INLINE __forceinline
#else
#define WORD_HD
#define WORD_INLINE inline __attribute__((always_inline))
#endif

namespace word_coefficient {

enum Method { horner = 0, prefix_suffix = 1 };

inline void check_method(int method) {
    if (method != horner && method != prefix_suffix) throw std::invalid_argument("Invalid coefficient method");
}

inline uint64_t add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("Coefficient size addition overflow");
    return a + b;
}
inline uint64_t mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("Coefficient size multiplication overflow");
    return a * b;
}
inline uint64_t environment_option(const char* name, uint64_t fallback) {
#ifdef _WIN32
    char buffer[32];
    const auto size = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    if (!size) return fallback;
    if (size >= sizeof(buffer)) throw std::invalid_argument("Coefficient environment option is too long");
    const char* value = buffer;
#else
    const char* value = std::getenv(name);
#endif
    if (!value) return fallback;
    uint64_t bytes = 0;
    for (const char* digit = value; *digit; ++digit) {
        if (*digit < '0' || *digit > '9') throw std::invalid_argument("Coefficient environment option must be an unsigned integer");
        bytes = add(mul(bytes, 10), *digit - '0');
    }
    return bytes;
}
inline uint64_t workspace_limit() {
    const uint64_t bytes = environment_option("PYSIGLIB_COEFFICIENT_WORKSPACE_BYTES", 256ULL << 20);
    if (!bytes) throw std::invalid_argument("PYSIGLIB_COEFFICIENT_WORKSPACE_BYTES must be positive");
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

template<class T>
struct ScalarOps {
    static constexpr T maximum = std::numeric_limits<T>::max();
    static constexpr T minimum = std::numeric_limits<T>::min();
    WORD_HD static bool finite(T x) { return x <= maximum && x >= -maximum; }
    WORD_HD static bool tiny(T raw, T x) {
        return raw != T(0) && x < minimum && x > -minimum;
    }
    WORD_HD static T mul_div(T x, T y, uint64_t j) {
        const T bound = T(j) * minimum;
        return x < bound && x > -bound ? (x * y) / T(j) : (x / T(j)) * y;
    }
};

template<class T>
WORD_HD WORD_INLINE T point(const T* path, uint64_t vertex, uint64_t channel, uint64_t dimension,
    uint64_t n, bool time_aug, bool lead_lag, T end_time) {
    if (time_aug && channel == dimension * (lead_lag ? 2 : 1))
        return n ? T(vertex) * (end_time / T(n)) : T(0);
    const uint64_t row = lead_lag ? vertex / 2 + ((vertex % 2) && channel >= dimension) : vertex;
    return path[row * dimension + (lead_lag && channel >= dimension ? channel - dimension : channel)];
}

template<class T>
WORD_HD void axis_forward(T dx, uint64_t degree, T* result, bool prefixes) {
    T term = 1;
    for (uint64_t j = 1; j <= degree; ++j) {
        term = ScalarOps<T>::mul_div(dx, term, j);
        if (prefixes) result[j - 1] = term;
    }
    if (!prefixes || !degree) result[0] = term;
}

template<class T>
WORD_HD T axis_backward(T dx, const T* seeds, uint64_t degree, bool terminal) {
    T grad = seeds[terminal ? 0 : degree - 1];
    for (uint64_t j = degree - 1; j > 0; --j)
        grad = (terminal ? T(0) : seeds[j - 1]) + ScalarOps<T>::mul_div(dx, grad, j);
    return grad;
}

// Divided Horner rows keep the factorial inside the product. Used on range-sensitive inputs.
template<bool Backward, class T, class Scatter>
WORD_HD void horner_word(const T* x, T* output, const T* saved, const T* seeds, const uint64_t* word,
    uint64_t m, uint64_t dim, uint64_t length, bool ta, bool ll, T time, bool prefixes, bool terminal,
    T* scratch, Scatter scatter) {
    const uint64_t n = length ? (length - 1) * (ll ? 2 : 1) : 0;
    T* state = scratch;
    T* dx = state + m + 1;
    T* adj = dx + m;
    T* row = adj + m;
    T* grad = row + m;
    T* after = grad + m;
    state[0] = T(1);
    if (!m) { if constexpr (!Backward) output[0] = 1; return; }
    auto increments = [&](uint64_t t) {
        for (uint64_t j = 0; j < m; ++j)
            dx[j] = point(x, t + 1, word[j], dim, n, ta, ll, time) - point(x, t, word[j], dim, n, ta, ll, time);
    };
    auto update = [&](uint64_t depth, T sign) {
        for (uint64_t j = depth; j; --j) {
            T value = T(1);
            for (uint64_t k = 0; k < j; ++k)
                value = state[k + 1] + ScalarOps<T>::mul_div(sign * dx[k], value, j - k);
            state[j] = value;
        }
    };
    for (uint64_t j = 1; j <= m; ++j) state[j] = 0;
    if (!Backward || (!saved && n > 1)) {
        for (uint64_t t = 0; t < n; ++t) {
            increments(t);
            if (!t) {
                for (uint64_t j = 1; j <= m; ++j) state[j] = ScalarOps<T>::mul_div(dx[j-1], state[j-1], j);
            } else update(m, T(1));
        }
    }
    if constexpr (!Backward) {
        if (prefixes) for (uint64_t j = 0; j < m; ++j) output[j] = state[j+1];
        else output[0] = state[m];
    } else {
        for (uint64_t j = 0; j < m; ++j) {
            if (saved) state[j+1] = saved[j];
            adj[j] = terminal ? (j+1 == m ? seeds[0] : T(0)) : seeds[j];
            after[j] = 0;
        }
        for (uint64_t t = n; t-- > 0;) {
            increments(t);
            if (!t) {
                for (uint64_t j = 1; j < m; ++j) state[j] = ScalarOps<T>::mul_div(dx[j-1], state[j-1], j);
                for (uint64_t j = m; j; --j) {
                    grad[j-1] = ScalarOps<T>::mul_div(state[j-1], adj[j-1], j);
                    if (j > 1) adj[j-2] += ScalarOps<T>::mul_div(dx[j-1], adj[j-1], j);
                }
            } else {
                update(m-1, T(-1));
                for (uint64_t k = 0; k < m; ++k) grad[k] = 0;
                for (uint64_t j = 1; j <= m; ++j) {
                    row[0] = 1;
                    for (uint64_t k = 1; k < j; ++k)
                        row[k] = state[k] + ScalarOps<T>::mul_div(dx[k-1], row[k-1], j-k+1);
                    T weight = adj[j-1];
                    for (uint64_t k = j; k-- > 0;) {
                        grad[k] += ScalarOps<T>::mul_div(row[k], weight, j-k);
                        weight = ScalarOps<T>::mul_div(dx[k], weight, j-k);
                        if (k) adj[k-1] += weight;
                    }
                }
            }
            // A segment followed by its inverse has identity signature for every turnaround point.
            bool retraced = t + 1 < n;
            for (uint64_t j = 0; retraced && j < m; ++j) {
                const uint64_t c = word[j];
                retraced = point(x, t, c, dim, n, ta, ll, time) == point(x, t + 2, c, dim, n, ta, ll, time);
            }
            for (uint64_t j = 0; j < m; ++j) {
                const uint64_t c = word[j];
                if (ta && c == dim * (ll ? 2 : 1)) continue;
                const bool lead = ll && c >= dim;
                const uint64_t vertex = ll ? (t+1)/2 + ((t+1)%2 && lead) : t+1;
                const uint64_t col = lead ? c-dim : c;
                if (!retraced) scatter(output + vertex * dim + col, grad[j] - after[j]);
                if (!t) scatter(output+col, -grad[j]);
                after[j] = grad[j];
            }
        }
    }
}

} // namespace word_coefficient
