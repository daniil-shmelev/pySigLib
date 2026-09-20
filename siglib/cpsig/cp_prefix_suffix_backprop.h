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
#include "cp_vector_funcs.h"

namespace prefix_suffix::cpu_detail {

template<typename T>
void tensor_powers(const T* v, T* powers, uint64_t degree, const Layout& l) {
    powers[0] = 1;
    for (uint64_t k = 1; k <= degree; ++k)
        for (uint64_t x = 0; x < l.power[k - 1]; ++x)
            for (uint64_t c = 0; c < l.d; ++c)
                powers[l.offset[k] + x * l.d + c] = powers[l.offset[k - 1] + x] * v[c];
}

template<typename T>
void tensor_powers_backprop(const T* v, const T* powers, T* deriv, T* dv, uint64_t degree, const Layout& l) {
    for (uint64_t k = degree; k > 0; --k)
        for (uint64_t x = 0; x < l.power[k - 1]; ++x) {
            T value = 0;
            for (uint64_t c = 0; c < l.d; ++c) {
                const T g = deriv[l.offset[k] + x * l.d + c];
                value += g * v[c];
                dv[c] += g * powers[l.offset[k - 1] + x];
            }
            deriv[l.offset[k - 1] + x] += value;
        }
}

template<typename T>
void right_factor_backprop(const T* suffix, const T* v, const T* deriv, T* dsuffix, T* dv,
    uint64_t j, uint64_t b, const Layout& l, T* history, T* work) {
    history[0] = 1;
    T scale = 1;
    for (uint64_t f = 2; f <= j + 1; ++f) scale /= T(f);
    for (uint64_t k = 1; k < b; ++k) {
        const uint64_t previous = l.power[k - 1];
        const T reciprocal = T(1) / T(j + b - k + 2);
        for (uint64_t c = 0; c < l.d; ++c)
            for (uint64_t x = 0; x < previous; ++x)
                history[l.offset[k] + c * previous + x] = suffix[l.offset[k] + c * previous + x]
                    + v[c] * reciprocal * history[l.offset[k - 1] + x];
    }
    for (uint64_t x = 0; x < l.power[b]; ++x) work[l.offset[b] + x] = deriv[x] * scale;
    for (uint64_t k = b; k > 0; --k) {
        const uint64_t previous = l.power[k - 1];
        const T reciprocal = T(1) / T(j + b - k + 2);
        T* lower = work + l.offset[k - 1];
        std::fill_n(lower, previous, T(0));
        for (uint64_t c = 0; c < l.d; ++c) {
            const T multiplier = v[c] * reciprocal;
            const uint64_t word = l.offset[k] + c * previous;
            T channel_deriv = 0;
            if (previous >= 16) {
                vec_mult_add(dsuffix + word, work + word, T(1), previous);
                channel_deriv = dot_product_mult_add(history + l.offset[k - 1],
                    work + word, lower, multiplier, previous);
            } else {
                for (uint64_t x = 0; x < previous; ++x) {
                    const T g = work[word + x];
                    dsuffix[word + x] += g;
                    lower[x] += g * multiplier;
                    channel_deriv += g * history[l.offset[k - 1] + x];
                }
            }
            dv[c] += channel_deriv * reciprocal;
        }
    }
}

template<bool Prepend, typename T>
void horner_backprop(const T* state, const T* v, T* deriv, T* dv, uint64_t degree,
    const Layout& l, T* history, T* work) {
    for (uint64_t c = 0; c < l.d; ++c) dv[c] += deriv[1 + c];
    // Reverse the descending level updates in ascending order.
    for (uint64_t target = 2; target <= degree; ++target) {
        for (uint64_t c = 0; c < l.d; ++c) history[1 + c] = v[c] / T(target);
        for (uint64_t k = 1; k + 1 < target; ++k) {
            const uint64_t size = l.power[k];
            const T reciprocal = T(1) / T(target - k);
            for (uint64_t c = 0; c < l.d; ++c)
                for (uint64_t x = 0; x < size; ++x) {
                    const uint64_t word = Prepend ? c * size + x : x * l.d + c;
                    history[l.offset[k + 1] + word] =
                        (state[l.offset[k] + x] + history[l.offset[k] + x]) * v[c] * reciprocal;
                }
        }
        const T* upper = deriv + l.offset[target];
        for (uint64_t k = target - 1; k > 0; --k) {
            const uint64_t size = l.power[k];
            const T reciprocal = T(1) / T(target - k);
            T* lower = work + l.offset[k];
            std::fill_n(lower, size, T(0));
            for (uint64_t c = 0; c < l.d; ++c) {
                const T multiplier = v[c] * reciprocal;
                T channel_deriv = 0;
                for (uint64_t x = 0; x < size; ++x) {
                    const uint64_t word = Prepend ? c * size + x : x * l.d + c;
                    const T g = upper[word];
                    lower[x] += g * multiplier;
                    channel_deriv += g * (state[l.offset[k] + x] + history[l.offset[k] + x]);
                }
                dv[c] += channel_deriv * reciprocal;
            }
            for (uint64_t x = 0; x < size; ++x) deriv[l.offset[k] + x] += lower[x];
            upper = lower;
        }
        for (uint64_t c = 0; c < l.d; ++c) dv[c] += work[1 + c] / T(target);
    }
}

} // namespace prefix_suffix::cpu_detail
