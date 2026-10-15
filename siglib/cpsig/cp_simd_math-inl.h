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
#if defined(CPSIG_CP_SIMD_MATH_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_CP_SIMD_MATH_INL_H_
#undef CPSIG_CP_SIMD_MATH_INL_H_
#else
#define CPSIG_CP_SIMD_MATH_INL_H_
#endif
#include "cp_simd-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include <cmath>

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template<typename T>
void vec_exp(T* values, uint64_t size) {
    const Simd<T> d;
    constexpr uint64_t lanes = vec_batch_lanes<T>;
    uint64_t i = 0;
    // Scalar double exp is faster in the measured CPU kernels.
    if constexpr (lanes > 1 && std::is_same_v<T, float>) {
        const T bound = T(80);
        const auto lower = hn::Set(d, -bound), upper = hn::Set(d, bound);
        for (; i + lanes <= size; i += lanes) {
            const auto x = hn::LoadU(d, values + i);
            const auto safe = hn::And(hn::Ge(x, lower), hn::Le(x, upper));
            const auto result = hn::Exp(d, hn::IfThenElse(safe, x, hn::Zero(d)));
            if (hn::AllTrue(d, safe)) {
                hn::StoreU(result, d, values + i);
            } else {
                T original[lanes];
                hn::StoreU(x, d, original);
                hn::StoreU(result, d, values + i);
                for (uint64_t k = 0; k < lanes; ++k)
                    if (!(original[k] >= -bound && original[k] <= bound))
                        values[i + k] = std::exp(original[k]);
            }
        }
    }
    for (; i < size; ++i) values[i] = std::exp(values[i]);
}

}
}
HWY_AFTER_NAMESPACE();
#endif
