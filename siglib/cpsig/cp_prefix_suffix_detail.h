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

#include "cp_simd_config.h"
#include "hwy/highway.h"
#if defined(CPSIG_CP_PREFIX_SUFFIX_DETAIL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_CP_PREFIX_SUFFIX_DETAIL_H_
#undef CPSIG_CP_PREFIX_SUFFIX_DETAIL_H_
#else
#define CPSIG_CP_PREFIX_SUFFIX_DETAIL_H_
#endif
#include "prefix_suffix.h"
#include "cp_simd-inl.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {
namespace cpu_detail {
using namespace prefix_suffix;

template<typename T, int D, int K, int Level>
inline void prepend_level(T* state, const T* v, T* work) {
    constexpr int previous = small_power(D, Level);
    constexpr T reciprocal = T(1) / T(K - Level);
    for (int channel = D; channel-- > 0;) {
        const T multiplier = v[channel] * reciprocal;
        T* destination = Level + 1 == K
            ? state + small_offset(D, K) + channel * previous
            : work + channel * previous;
        const T* source = state + small_offset(D, Level);
#if HWY_TARGET != HWY_SCALAR
        if constexpr (Level + 1 == K) {
            vec_mult_add(destination, source, multiplier, previous);
            vec_mult_add(destination, work, multiplier, previous);
        } else {
            vec_add_scaled(destination, source, work, multiplier, previous);
        }
#else
        for (int word = previous; word-- > 0;)
            destination[word] = (source[word] + work[word]) * multiplier
                + (Level + 1 == K ? destination[word] : T(0));
#endif
    }
    if constexpr (Level + 1 < K)
        prepend_level<T, D, K, Level + 1>(state, v, work);
}

template<typename T, int D, int K>
inline void prepend_target(T* state, const T* v, T* work) {
    constexpr T reciprocal = T(1) / T(K);
    for (int channel = 0; channel < D; ++channel) work[channel] = v[channel] * reciprocal;
    prepend_level<T, D, K, 1>(state, v, work);
}

template<typename T, int D, int K>
inline void prepend_targets(T* state, const T* v, T* work) {
    prepend_target<T, D, K>(state, v, work);
    if constexpr (K > 2) prepend_targets<T, D, K - 1>(state, v, work);
}

template<typename T, int D, int M>
inline void prepend_fixed(T* state, const T* v, T* work) {
    if constexpr (M >= 2) prepend_targets<T, D, M>(state, v, work);
    for (int channel = 0; channel < D; ++channel) state[1 + channel] += v[channel];
}

template<typename T>
inline void prepend_generic(T* state, const T* v, const Layout& layout, T* work) {
    for (uint64_t k = layout.scan_degree; k > 1; --k) {
        const T initial_reciprocal = T(1) / T(k);
        for (uint64_t channel = 0; channel < layout.d; ++channel)
            work[channel] = v[channel] * initial_reciprocal;
        for (uint64_t level = 1; level < k; ++level) {
            const uint64_t previous = layout.power[level];
            const T reciprocal = T(1) / T(k - level);
            for (uint64_t channel = layout.d; channel-- > 0;) {
                const T multiplier = v[channel] * reciprocal;
                T* destination = level + 1 == k
                    ? state + layout.offset[k] + channel * previous
                    : work + channel * previous;
                const T* source = state + layout.offset[level];
#if HWY_TARGET != HWY_SCALAR
                if (level + 1 == k) {
                    vec_mult_add(destination, source, multiplier, previous);
                    vec_mult_add(destination, work, multiplier, previous);
                } else {
                    vec_add_scaled(destination, source, work, multiplier, previous);
                }
#else
                for (uint64_t word = previous; word-- > 0;)
                    destination[word] = (source[word] + work[word]) * multiplier
                        + (level + 1 == k ? destination[word] : T(0));
#endif
            }
        }
    }
    for (uint64_t channel = 0; channel < layout.d; ++channel) state[1 + channel] += v[channel];
}

template<typename T>
inline void prepend(T* state, const T* v, const Layout& layout, T* work) {
    if (layout.scan_degree <= 3) {
        auto run = [&]<int D>() {
            switch (layout.scan_degree) {
            case 1: prepend_fixed<T, D, 1>(state, v, work); break;
            case 2: prepend_fixed<T, D, 2>(state, v, work); break;
            case 3: prepend_fixed<T, D, 3>(state, v, work); break;
            }
        };
        switch (layout.d) {
        case 2: run.template operator()<2>(); return;
        case 3: run.template operator()<3>(); return;
        case 4: run.template operator()<4>(); return;
        case 8: run.template operator()<8>(); return;
        }
    }
    prepend_generic(state, v, layout, work);
}

} // namespace cpu_detail
}
}
HWY_AFTER_NAMESPACE();
#endif
