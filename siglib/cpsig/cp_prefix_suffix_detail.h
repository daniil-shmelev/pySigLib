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

template<typename T>
void right_factor(T* result, const T* suffix, const T* v, uint64_t j, uint64_t b, const Layout& l) {
    const Simd<T> simd;
    const uint64_t lanes = hn::Lanes(simd);
    result[0] = 1;
    T scale = 1;
    for (uint64_t f = 2; f <= j + 1; ++f) scale /= T(f);
    for (uint64_t level = 1; level <= b; ++level) {
        const uint64_t previous = l.power[level - 1];
        const T reciprocal = T(1) / T(j + b - level + 2);
        for (uint64_t channel = l.d; channel-- > 0;) {
            T* destination = result + channel * previous;
            const T* source = suffix + l.offset[level] + channel * previous;
            const T multiplier = v[channel] * reciprocal;
            uint64_t x = 0;
            for (; x + lanes <= previous; x += lanes) {
                auto value = hn::Add(hn::LoadU(simd, source + x), hn::Mul(hn::Set(simd, multiplier), hn::LoadU(simd, result + x)));
                if (level == b) value = hn::Mul(value, hn::Set(simd, scale));
                hn::StoreU(value, simd, destination + x);
            }
            for (; x < previous; ++x) {
                const T value = source[x] + multiplier * result[x];
                destination[x] = level == b ? value * scale : value;
            }
        }
    }
    if (!b) result[0] = scale;
}

template<typename T>
HWY_INLINE void prepend_level_values(T* destination, const T* source, T* work, const T* v,
    uint64_t d, uint64_t previous, T reciprocal, bool last) {
    for (uint64_t channel = d; channel-- > 0;) {
        T* row = destination + channel * previous;
        const T multiplier = v[channel] * reciprocal;
#if HWY_TARGET != HWY_SCALAR
        if (last) {
            const Simd<T> simd;
            const uint64_t lanes = hn::Lanes(simd);
            const auto scale = hn::Set(simd, multiplier);
            uint64_t x = 0;
            for (; x + lanes <= previous; x += lanes) {
                const auto value = hn::MulAdd(hn::LoadU(simd, source + x), scale, hn::LoadU(simd, row + x));
                hn::StoreU(hn::MulAdd(hn::LoadU(simd, work + x), scale, value), simd, row + x);
            }
            for (; x < previous; ++x) { row[x] += source[x] * multiplier; row[x] += work[x] * multiplier; }
        } else {
            vec_add_scaled(row, source, work, multiplier, previous);
        }
#else
        for (uint64_t word = previous; word-- > 0;)
            row[word] = (source[word] + work[word]) * multiplier + (last ? row[word] : T(0));
#endif
    }
}

template<typename T, int D, int K, int Level>
inline void prepend_level(T* state, const T* v, T* work) {
    constexpr int previous = small_power(D, Level);
    constexpr T reciprocal = T(1) / T(K - Level);
    prepend_level_values(Level + 1 == K ? state + small_offset(D, K) : work,
        state + small_offset(D, Level), work, v, D, previous, reciprocal, Level + 1 == K);
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
            prepend_level_values(level + 1 == k ? state + layout.offset[k] : work,
                state + layout.offset[level], work, v, layout.d, previous, reciprocal, level + 1 == k);
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
