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

#include "cppch.h"
#include "macros.h"
#include "cp_simd_config.h"
#include "hwy/highway.h"

#if defined(CPSIG_SIMD_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_SIMD_INL_H_
#undef CPSIG_SIMD_INL_H_
#else
#define CPSIG_SIMD_INL_H_
#endif

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#if HWY_ARCH_X86
inline constexpr size_t vec_batch_bytes = 32;
#else
inline constexpr size_t vec_batch_bytes = 16;
#endif
template<typename T>
using Simd = hn::CappedTag<T, vec_batch_bytes / sizeof(T)>;
template<typename T>
inline constexpr size_t vec_batch_lanes = hn::MaxLanes(Simd<T>());

template<typename T>
FORCE_INLINE void vec_mult_add(T* out, const T* other, T scalar, uint64_t size) {
    const Simd<T> d;
    constexpr size_t lanes = vec_batch_lanes<T>;
    if constexpr (std::is_same_v<T, float> && lanes >= 4) {
        if (size > 4 && size < 8) {
            const hn::CappedTag<T, 4> d4;
            const auto s = hn::Set(d4, scalar);
            const auto a = hn::LoadU(d4, other);
            const auto b = hn::LoadU(d4, other + size - 4);
            const auto x = hn::LoadU(d4, out);
            const auto y = hn::LoadU(d4, out + size - 4);
            hn::StoreU(hn::MulAdd(a, s, x), d4, out);
            hn::StoreU(hn::MulAdd(b, s, y), d4, out + size - 4);
            return;
        }
    }
    const auto s = hn::Set(d, scalar);
    const uint64_t vectors = size / lanes;
    const uint64_t tail4 = size & (lanes - 1) & 4;
    const uint64_t tail2 = size & (lanes - 1) & 2;
    const uint64_t tail1 = size & (lanes - 1) & 1;
    for (uint64_t k = 0; k < vectors; ++k) {
        hn::StoreU(hn::MulAdd(hn::LoadU(d, other), s, hn::LoadU(d, out)), d, out);
        other += lanes;
        out += lanes;
    }
    if constexpr (std::is_same_v<T, float> && lanes > 4) {
        if (tail4) {
            const hn::CappedTag<T, 4> d4;
            hn::StoreU(hn::MulAdd(hn::LoadU(d4, other), hn::Set(d4, scalar), hn::LoadU(d4, out)), d4, out);
            other += 4;
            out += 4;
        }
    }
    if constexpr (HWY_ARCH_X86 && lanes > 2) {
        if (tail2) {
            const hn::CappedTag<T, 2> d2;
            hn::StoreU(hn::MulAdd(hn::LoadU(d2, other), hn::Set(d2, scalar), hn::LoadU(d2, out)), d2, out);
            other += 2;
            out += 2;
        }
    }
    else if constexpr (lanes > 2) {
        if (tail2) {
            out[0] += other[0] * scalar;
            out[1] += other[1] * scalar;
            other += 2;
            out += 2;
        }
    }
    if (tail1) *out += *other * scalar;
}

template<typename T>
FORCE_INLINE void vec_mult_assign(T* out, const T* other, T scalar, uint64_t size) {
    const Simd<T> d;
    constexpr size_t lanes = vec_batch_lanes<T>;
    if constexpr (std::is_same_v<T, float> && lanes >= 4) {
        if (size > 4 && size < 8) {
            const hn::CappedTag<T, 4> d4;
            const auto s = hn::Set(d4, scalar);
            hn::StoreU(hn::Mul(hn::LoadU(d4, other), s), d4, out);
            hn::StoreU(hn::Mul(hn::LoadU(d4, other + size - 4), s), d4, out + size - 4);
            return;
        }
    }
    const auto s = hn::Set(d, scalar);
    const uint64_t vectors = size / lanes;
    const uint64_t tail_half = size & (lanes - 1) & (16 / sizeof(T));
    const uint64_t tail2 = size & (lanes - 1) & 2;
    const uint64_t tail1 = size & (lanes - 1) & 1;
    for (uint64_t k = 0; k < vectors; ++k) {
        hn::StoreU(hn::Mul(hn::LoadU(d, other), s), d, out);
        other += lanes;
        out += lanes;
    }
    if constexpr (lanes * sizeof(T) > 16) {
        const hn::CappedTag<T, 16 / sizeof(T)> dh;
        constexpr size_t half = 16 / sizeof(T);
        if (tail_half) {
            hn::StoreU(hn::Mul(hn::LoadU(dh, other), hn::Set(dh, scalar)), dh, out);
            other += half;
            out += half;
        }
    }
    if constexpr (std::is_same_v<T, float> && lanes > 2) {
        if (tail2) {
            out[0] = other[0] * scalar;
            out[1] = other[1] * scalar;
            other += 2;
            out += 2;
        }
    }
    if (tail1) *out = *other * scalar;
}

template<typename T>
FORCE_INLINE T reduce_sum(hn::Vec<Simd<T>> value) {
    const Simd<T> d;
#if HWY_TARGET != HWY_SCALAR
    if constexpr (std::is_same_v<T, float> && vec_batch_lanes<T> == 8) {
        const hn::Half<Simd<T>> dh;
        auto sum = hn::Add(hn::LowerHalf(dh, value), hn::UpperHalf(dh, value));
        sum = hn::Add(sum, hn::Shuffle2301(sum));
        return hn::GetLane(hn::Add(sum, hn::Shuffle1032(sum)));
    }
    else if constexpr (std::is_same_v<T, float> && vec_batch_lanes<T> == 4) {
        const auto sum = hn::Add(value, hn::Shuffle2301(value));
        return hn::GetLane(hn::Add(sum, hn::Shuffle1032(sum)));
    }
    else
#endif
    {
        T lanes[vec_batch_lanes<T>];
        hn::StoreU(value, d, lanes);
        T result = lanes[0];
        for (size_t k = 1; k < vec_batch_lanes<T>; ++k) result += lanes[k];
        return result;
    }
}

template<typename T>
FORCE_INLINE T dot_product(const T* a, const T* b, size_t size) {
    const Simd<T> d;
    constexpr size_t n = vec_batch_lanes<T>;
    auto s0 = hn::Zero(d);
    size_t k = 0;
    if constexpr (n * sizeof(T) > 16) {
        auto s1 = s0, s2 = s0, s3 = s0;
        for (; k + 4 * n <= size; k += 4 * n) {
            s0 = hn::MulAdd(hn::LoadU(d, a + k), hn::LoadU(d, b + k), s0);
            s1 = hn::MulAdd(hn::LoadU(d, a + k + n), hn::LoadU(d, b + k + n), s1);
            s2 = hn::MulAdd(hn::LoadU(d, a + k + 2 * n), hn::LoadU(d, b + k + 2 * n), s2);
            s3 = hn::MulAdd(hn::LoadU(d, a + k + 3 * n), hn::LoadU(d, b + k + 3 * n), s3);
        }
        s0 = hn::Add(hn::Add(s0, s1), hn::Add(s2, s3));
    }
    for (; k + n <= size; k += n)
        s0 = hn::MulAdd(hn::LoadU(d, a + k), hn::LoadU(d, b + k), s0);
    T result = reduce_sum<T>(s0);
    for (; k < size; ++k) result += a[k] * b[k];
    return result;
}

template<typename T>
FORCE_INLINE void dot_product_pair(const T* weights, const T* a, const T* b, size_t size, T& result_a, T& result_b) {
    if constexpr (HWY_ARCH_ARM || vec_batch_lanes<T> == 1) {
        result_a = result_b = T(0);
        for (size_t k = 0; k < size; ++k) {
            result_a += weights[k] * a[k];
            result_b += weights[k] * b[k];
        }
        return;
    }
    const Simd<T> d;
    constexpr size_t n = vec_batch_lanes<T>;
    auto sa = hn::Zero(d), sb = sa;
    size_t k = 0;
    for (; k + n <= size; k += n) {
        const auto w = hn::LoadU(d, weights + k);
        sa = hn::MulAdd(w, hn::LoadU(d, a + k), sa);
        sb = hn::MulAdd(w, hn::LoadU(d, b + k), sb);
    }
    result_a = reduce_sum<T>(sa);
    result_b = reduce_sum<T>(sb);
    for (; k < size; ++k) {
        result_a += weights[k] * a[k];
        result_b += weights[k] * b[k];
    }
}

template<typename T>
FORCE_INLINE T dot_product_mult_add(const T* a, const T* b, T* out, T scalar, size_t size) {
    const Simd<T> d;
    constexpr size_t n = vec_batch_lanes<T>;
    auto sum = hn::Zero(d);
    size_t k = 0;
    for (; k + n <= size; k += n) {
        const auto vb = hn::LoadU(d, b + k);
        sum = hn::MulAdd(hn::LoadU(d, a + k), vb, sum);
        hn::StoreU(hn::MulAdd(vb, hn::Set(d, scalar), hn::LoadU(d, out + k)), d, out + k);
    }
    T result = reduce_sum<T>(sum);
    for (; k < size; ++k) {
        result += a[k] * b[k];
        out[k] += b[k] * scalar;
    }
    return result;
}

template<typename T>
FORCE_INLINE void vec_add_scaled(T* out, const T* a, const T* b, T scalar, uint64_t size) {
    const Simd<T> d;
    uint64_t k = 0;
    for (; k + hn::Lanes(d) <= size; k += hn::Lanes(d))
        hn::StoreU(hn::Mul(hn::Add(hn::LoadU(d, a + k), hn::LoadU(d, b + k)), hn::Set(d, scalar)), d, out + k);
    for (; k < size; ++k) out[k] = (a[k] + b[k]) * scalar;
}

template<typename T, typename IndexFn>
FORCE_INLINE void vec_kernel_diag_step(T* RESTRICT next, const T* RESTRICT prev, const T* RESTRICT prev_m1, const T* RESTRICT prev_prev_m1, const T* RESTRICT gram_a, const T* RESTRICT gram_b, IndexFn idx, uint64_t count) {
    const Simd<T> d;
    constexpr size_t n = vec_batch_lanes<T>;
    uint64_t k = 0;
    for (; n > 1 && k + n <= count; k += n) {
        auto va = hn::Zero(d), vb = hn::Zero(d);
#if HWY_TARGET != HWY_SCALAR
        if constexpr (std::is_same_v<T, double> && n == 4) {
            const hn::Half<Simd<T>> dh;
            const auto gi0 = idx(k), gi1 = idx(k + 1), gi2 = idx(k + 2), gi3 = idx(k + 3);
            const auto a_lo = hn::InterleaveLower(hn::Set(dh, gram_a[gi0]), hn::Set(dh, gram_a[gi1]));
            const auto a_hi = hn::InterleaveLower(hn::Set(dh, gram_a[gi2]), hn::Set(dh, gram_a[gi3]));
            const auto b_lo = hn::InterleaveLower(hn::Set(dh, gram_b[gi0]), hn::Set(dh, gram_b[gi1]));
            const auto b_hi = hn::InterleaveLower(hn::Set(dh, gram_b[gi2]), hn::Set(dh, gram_b[gi3]));
            va = hn::Combine(d, a_hi, a_lo);
            vb = hn::Combine(d, b_hi, b_lo);
        }
        else
#endif
        {
            alignas(vec_batch_bytes) T a[n], b[n];
            for (size_t j = 0; j < n; ++j) {
                const auto gi = idx(k + j);
                a[j] = gram_a[gi]; b[j] = gram_b[gi];
            }
            va = hn::Load(d, a);
            vb = hn::Load(d, b);
        }
        if constexpr (HWY_ARCH_ARM) {
            auto result = hn::Mul(hn::LoadU(d, prev + k), va);
            result = hn::MulAdd(hn::LoadU(d, prev_m1 + k), va, result);
            result = hn::NegMulAdd(hn::LoadU(d, prev_prev_m1 + k), vb, result);
            hn::StoreU(result, d, next + k);
        }
        else {
            hn::StoreU(hn::MulAdd(hn::LoadU(d, prev + k), va, hn::MulSub(hn::LoadU(d, prev_m1 + k), va, hn::Mul(hn::LoadU(d, prev_prev_m1 + k), vb))), d, next + k);
        }
    }
    for (; k < count; ++k) {
        const auto gi = idx(k);
        next[k] = (prev[k] + prev_m1[k]) * gram_a[gi] - prev_prev_m1[k] * gram_b[gi];
    }
}

FORCE_INLINE void vec4_add(double* RESTRICT out, const double* RESTRICT a, const double* RESTRICT b, uint64_t count) {
    const Simd<double> d;
    for (uint64_t k = 0; k < count; ++k)
        for (size_t lane = 0; lane < 4; lane += vec_batch_lanes<double>) {
            const uint64_t offset = k * 4 + lane;
            hn::StoreU(hn::Add(hn::LoadU(d, a + offset), hn::LoadU(d, b + offset)), d, out + offset);
        }
}
FORCE_INLINE void vec4_fmadd(double* RESTRICT out, const double* RESTRICT a, double scalar, uint64_t count) {
    const Simd<double> d;
    const auto s = hn::Set(d, scalar);
    for (uint64_t k = 0; k < count; ++k)
        for (size_t lane = 0; lane < 4; lane += vec_batch_lanes<double>) {
            const uint64_t offset = k * 4 + lane;
            hn::StoreU(hn::MulAdd(s, hn::LoadU(d, a + offset), hn::LoadU(d, out + offset)), d, out + offset);
        }
}
FORCE_INLINE void vec4_scale(double* RESTRICT out, const double* RESTRICT a, double scalar, uint64_t count) {
    const Simd<double> d;
    const auto s = hn::Set(d, scalar);
    for (uint64_t k = 0; k < count; ++k)
        for (size_t lane = 0; lane < 4; lane += vec_batch_lanes<double>) {
            const uint64_t offset = k * 4 + lane;
            hn::StoreU(hn::Mul(s, hn::LoadU(d, a + offset)), d, out + offset);
        }
}
FORCE_INLINE void vec4_commutator_accum(double* RESTRICT result, const double* RESTRICT v1, const double* RESTRICT v2, const uint32_t* ki, const uint32_t* kj, const double* kval, uint32_t start, uint32_t end) {
    const Simd<double> d;
    if constexpr (HWY_ARCH_ARM && vec_batch_lanes<double> == 2) {
        // Share the sparse-index pass across both halves of the four-lane batch.
        auto lo = hn::Zero(d);
        auto hi = hn::Zero(d);
        for (uint32_t idx = start; idx < end; ++idx) {
            const uint32_t i = ki[idx] * 4, j = kj[idx] * 4;
            const auto coefficient = hn::Set(d, kval[idx]);
            const auto bracket_lo = hn::MulSub(
                hn::LoadU(d, v1 + i), hn::LoadU(d, v2 + j),
                hn::Mul(hn::LoadU(d, v1 + j), hn::LoadU(d, v2 + i)));
            const auto bracket_hi = hn::MulSub(
                hn::LoadU(d, v1 + i + 2), hn::LoadU(d, v2 + j + 2),
                hn::Mul(hn::LoadU(d, v1 + j + 2), hn::LoadU(d, v2 + i + 2)));
            lo = hn::MulAdd(coefficient, bracket_lo, lo);
            hi = hn::MulAdd(coefficient, bracket_hi, hi);
        }
        hn::StoreU(lo, d, result);
        hn::StoreU(hi, d, result + 2);
        return;
    }
    for (size_t k = 0; k < 4; k += hn::Lanes(d)) {
        auto sum = hn::Zero(d);
        for (uint32_t idx = start; idx < end; ++idx) {
            const uint32_t i = ki[idx] * 4, j = kj[idx] * 4;
            const auto bracket = hn::MulSub(hn::LoadU(d, v1 + i + k), hn::LoadU(d, v2 + j + k), hn::Mul(hn::LoadU(d, v1 + j + k), hn::LoadU(d, v2 + i + k)));
            sum = hn::MulAdd(hn::Set(d, kval[idx]), bracket, sum);
        }
        hn::StoreU(sum, d, result + k);
    }
}
FORCE_INLINE void vec4_bracket_grad(double* RESTRICT dm_lf, double* RESTRICT dm_rf, const double* RESTRICT dm_w, const double* RESTRICT v1, const double* RESTRICT v2, uint32_t i, uint32_t j, const uint32_t* ij_k, const double* ij_c, uint32_t start, uint32_t end) {
    const Simd<double> d;
    if constexpr (HWY_ARCH_ARM && vec_batch_lanes<double> == 2) {
        // Reuse each coefficient and index for the low and high halves.
        auto lo = hn::Zero(d);
        auto hi = hn::Zero(d);
        for (uint32_t idx = start; idx < end; ++idx) {
            const auto coefficient = hn::Set(d, ij_c[idx]);
            const uint32_t offset = ij_k[idx] * 4;
            lo = hn::Add(lo, hn::Mul(coefficient, hn::LoadU(d, dm_w + offset)));
            hi = hn::Add(hi, hn::Mul(coefficient, hn::LoadU(d, dm_w + offset + 2)));
        }
        for (size_t k = 0; k < 4; k += 2) {
            const auto sum = k == 0 ? lo : hi;
            hn::StoreU(hn::MulAdd(sum, hn::LoadU(d, v2 + j * 4 + k), hn::LoadU(d, dm_lf + i * 4 + k)), d, dm_lf + i * 4 + k);
            hn::StoreU(hn::NegMulAdd(sum, hn::LoadU(d, v2 + i * 4 + k), hn::LoadU(d, dm_lf + j * 4 + k)), d, dm_lf + j * 4 + k);
            hn::StoreU(hn::MulAdd(sum, hn::LoadU(d, v1 + i * 4 + k), hn::LoadU(d, dm_rf + j * 4 + k)), d, dm_rf + j * 4 + k);
            hn::StoreU(hn::NegMulAdd(sum, hn::LoadU(d, v1 + j * 4 + k), hn::LoadU(d, dm_rf + i * 4 + k)), d, dm_rf + i * 4 + k);
        }
        return;
    }
    for (size_t k = 0; k < 4; k += hn::Lanes(d)) {
        auto s = hn::Zero(d);
        for (uint32_t idx = start; idx < end; ++idx)
            s = hn::Add(s, hn::Mul(hn::Set(d, ij_c[idx]), hn::LoadU(d, dm_w + ij_k[idx] * 4 + k)));
        hn::StoreU(hn::MulAdd(s, hn::LoadU(d, v2 + j * 4 + k), hn::LoadU(d, dm_lf + i * 4 + k)), d, dm_lf + i * 4 + k);
        hn::StoreU(hn::NegMulAdd(s, hn::LoadU(d, v2 + i * 4 + k), hn::LoadU(d, dm_lf + j * 4 + k)), d, dm_lf + j * 4 + k);
        hn::StoreU(hn::MulAdd(s, hn::LoadU(d, v1 + i * 4 + k), hn::LoadU(d, dm_rf + j * 4 + k)), d, dm_rf + j * 4 + k);
        hn::StoreU(hn::NegMulAdd(s, hn::LoadU(d, v1 + j * 4 + k), hn::LoadU(d, dm_rf + i * 4 + k)), d, dm_rf + i * 4 + k);
    }
}

template<typename T>
FORCE_INLINE void vec_batch_fill(T* out, T value) { const Simd<T> d; hn::Store(hn::Set(d, value), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_copy(T* out, const T* value) { const Simd<T> d; hn::Store(hn::Load(d, value), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_add(T* out, const T* left, const T* right) { const Simd<T> d; hn::Store(hn::Add(hn::Load(d, left), hn::Load(d, right)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_add_inplace(T* out, const T* value) { vec_batch_add(out, out, value); }
template<typename T>
FORCE_INLINE void vec_batch_subtract(T* out, const T* left, const T* right) { const Simd<T> d; hn::Store(hn::Sub(hn::Load(d, left), hn::Load(d, right)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_subtract_inplace(T* out, const T* value) { vec_batch_subtract(out, out, value); }
template<typename T>
FORCE_INLINE void vec_batch_multiply(T* out, const T* left, const T* right) { const Simd<T> d; hn::Store(hn::Mul(hn::Load(d, left), hn::Load(d, right)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_multiply_inplace(T* out, const T* value) { vec_batch_multiply(out, out, value); }
template<typename T>
FORCE_INLINE void vec_batch_scale(T* out, const T* value, T scalar) { const Simd<T> d; hn::Store(hn::Mul(hn::Load(d, value), hn::Set(d, scalar)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_negate_inplace(T* out) { const Simd<T> d; hn::Store(hn::Neg(hn::Load(d, out)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_multiply_add(T* out, const T* left, const T* right) { const Simd<T> d; hn::Store(hn::MulAdd(hn::Load(d, left), hn::Load(d, right), hn::Load(d, out)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_scaled_add(T* out, const T* value, T scalar) { const Simd<T> d; hn::Store(hn::MulAdd(hn::Load(d, value), hn::Set(d, scalar), hn::Load(d, out)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_multiply_add_scaled(T* out, const T* left, const T* right, T scalar) { const Simd<T> d; hn::Store(hn::MulAdd(hn::Mul(hn::Load(d, left), hn::Load(d, right)), hn::Set(d, scalar), hn::Load(d, out)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_multiply_add3(T* out, const T* first, const T* second, const T* third) { const Simd<T> d; hn::Store(hn::MulAdd(hn::Mul(hn::Load(d, first), hn::Load(d, second)), hn::Load(d, third), hn::Load(d, out)), d, out); }
template<typename T>
FORCE_INLINE void vec_batch_subtract_product(T* out, const T* left, const T* right) { const Simd<T> d; hn::Store(hn::NegMulAdd(hn::Load(d, left), hn::Load(d, right), hn::Load(d, out)), d, out); }
template<typename T>
FORCE_INLINE bool vec_batch_is_zero(const T* value) { const Simd<T> d; return hn::AllTrue(d, hn::Eq(hn::Load(d, value), hn::Zero(d))); }

}  // namespace HWY_NAMESPACE
}  // namespace cpsig
HWY_AFTER_NAMESPACE();
#endif
