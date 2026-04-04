/* Copyright 2025 Daniil Shmelev
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
#include "cppch.h"
#include "macros.h"

#ifdef VEC

// Vectorised signature kernel diagonal step with gather (scalar tail).
// Computes: next[k] = (prev[k] + prev_m1[k]) * gram_a[idx(k)] - prev_prev_m1[k] * gram_b[idx(k)]
// for k in [k, count).  IndexFn maps k -> gram index (non-contiguous gather).

template<typename T, typename IndexFn>
FORCE_INLINE void vec_kernel_diag_step_tail(
	T* RESTRICT next, const T* RESTRICT prev, const T* RESTRICT prev_m1,
	const T* RESTRICT prev_prev_m1, const T* RESTRICT gram_a,
	const T* RESTRICT gram_b, IndexFn idx, uint64_t k, uint64_t count)
{
	for (; k < count; ++k) {
		const uint64_t gi = idx(k);
		next[k] = (prev[k] + prev_m1[k]) * gram_a[gi] - prev_prev_m1[k] * gram_b[gi];
	}
}

#ifndef __APPLE__

#include <immintrin.h>

FORCE_INLINE void vec_mult_add(float* out, const float* other, float scalar, uint64_t size)
{
	if (size > 4 && size < 8) {
		const __m128 s = _mm_set1_ps(scalar);
		const __m128 v0 = _mm_loadu_ps(other);
		const __m128 v1 = _mm_loadu_ps(other + size - 4);
		const __m128 o0 = _mm_loadu_ps(out);
		const __m128 o1 = _mm_loadu_ps(out + size - 4);
		_mm_storeu_ps(out, _mm_fmadd_ps(v0, s, o0));
		_mm_storeu_ps(out + size - 4, _mm_fmadd_ps(v1, s, o1));
		return;
	}

	const uint64_t N = size / 8UL;
	const uint64_t tail4 = size & 4UL;
	const uint64_t tail2 = size & 2UL;
	const uint64_t tail1 = size & 1UL;

	__m256 a, b;
	const __m256 scalar_256 = _mm256_set1_ps(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		a = _mm256_loadu_ps(other);
		b = _mm256_loadu_ps(out);
		b = _mm256_fmadd_ps(a, scalar_256, b);
		_mm256_storeu_ps(out, b);
		other += 8;
		out += 8;
	}

	if (tail4) {
		__m128 c, d;
		const __m128 scalar_128 = _mm_set1_ps(scalar);

		c = _mm_loadu_ps(other);
		d = _mm_loadu_ps(out);
		d = _mm_fmadd_ps(c, scalar_128, d);
		_mm_storeu_ps(out, d);
		other += 4;
		out += 4;
	}

	if (tail2) {
		__m128 c, d;
		const __m128 scalar_128 = _mm_set1_ps(scalar);

		c = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(other)));
		d = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(out)));
		d = _mm_fmadd_ps(c, scalar_128, d);
		_mm_store_sd(reinterpret_cast<double*>(out), _mm_castps_pd(d));
		other += 2;
		out += 2;
	}

	if (tail1) {
		*out += *other * scalar;
	}
}

FORCE_INLINE void vec_mult_add(double* out, const double* other, double scalar, uint64_t size)
{
	const uint64_t N = size / 4UL;
	const uint64_t tail2 = size & 2UL;
	const uint64_t tail1 = size & 1UL;

	__m256d a, b;
	const __m256d scalar_256 = _mm256_set1_pd(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		a = _mm256_loadu_pd(other);
		b = _mm256_loadu_pd(out);
		b = _mm256_fmadd_pd(a, scalar_256, b);
		_mm256_storeu_pd(out, b);
		other += 4;
		out += 4;
	}
	if (tail2) {
		__m128d c, d;
		__m128d scalar_128 = _mm_set1_pd(scalar);

		c = _mm_loadu_pd(other);
		d = _mm_loadu_pd(out);
		d = _mm_fmadd_pd(c, scalar_128, d);
		_mm_storeu_pd(out, d);
		other += 2;
		out += 2;
	}
	if (tail1) {
		*out += *other * scalar;
	}
}

FORCE_INLINE void vec_mult_assign(float* out, const float* other, float scalar, uint64_t size)
{
	if (size > 4 && size < 8) {
		const __m128 s = _mm_set1_ps(scalar);
		_mm_storeu_ps(out, _mm_mul_ps(_mm_loadu_ps(other), s));
		_mm_storeu_ps(out + size - 4, _mm_mul_ps(_mm_loadu_ps(other + size - 4), s));
		return;
	}

	const uint64_t N = size / 8UL;
	const uint64_t tail4 = size & 4UL;
	const uint64_t tail2 = size & 2UL;
	const uint64_t tail1 = size & 1UL;

	__m256 a;
	const __m256 scalar_ = _mm256_set1_ps(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		a = _mm256_loadu_ps(other);
		a = _mm256_mul_ps(a, scalar_);
		_mm256_storeu_ps(out, a);
		other += 8;
		out += 8;
	}

	if (tail4) {
		__m128 c;
		const __m128 scalar_128 = _mm_set1_ps(scalar);

		c = _mm_loadu_ps(other);
		c = _mm_mul_ps(c, scalar_128);
		_mm_storeu_ps(out, c);
		other += 4;
		out += 4;
	}

	if (tail2) {
		out[0] = other[0] * scalar;
		out[1] = other[1] * scalar;
		other += 2;
		out += 2;
	}

	if (tail1) {
		*out = *other * scalar;
	}
}

FORCE_INLINE void vec_mult_assign(double* out, const double* other, double scalar, uint64_t size)
{
	const uint64_t N = size / 4UL;
	const uint64_t tail2 = size & 2UL;
	const uint64_t tail1 = size & 1UL;

	__m256d a;
	const __m256d scalar_ = _mm256_set1_pd(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		a = _mm256_loadu_pd(other);
		a = _mm256_mul_pd(a, scalar_);
		_mm256_storeu_pd(out, a);
		other += 4;
		out += 4;
	}
	if (tail2) {
		__m128d c;
		__m128d scalar_128 = _mm_set1_pd(scalar);

		c = _mm_loadu_pd(other);
		c = _mm_mul_pd(c, scalar_128);
		_mm_storeu_pd(out, c);
		other += 2;
		out += 2;
	}
	if (tail1) {
		*out = *other * scalar;
	}
}

FORCE_INLINE double dot_product(const double* a, const double* b, size_t N) {
	__m256d sum = _mm256_setzero_pd();

	size_t k = 0;
	size_t limit = N & ~3UL;
	for (; k < limit; k += 4) {
		__m256d va = _mm256_loadu_pd(&a[k]);
		__m256d vb = _mm256_loadu_pd(&b[k]);
		sum = _mm256_fmadd_pd(va, vb, sum);
	}

	double tmp[4];
	_mm256_storeu_pd(tmp, sum);
	double out = tmp[0] + tmp[1] + tmp[2] + tmp[3];

	for (; k < N; ++k) {
		out += a[k] * b[k];
	}

	return out;
}

template<typename IndexFn>
FORCE_INLINE void vec_kernel_diag_step(
	double* RESTRICT next,
	const double* RESTRICT prev,
	const double* RESTRICT prev_m1,
	const double* RESTRICT prev_prev_m1,
	const double* RESTRICT gram_a,
	const double* RESTRICT gram_b,
	IndexFn idx,
	uint64_t count)
{
	uint64_t k = 0;
	for (; k + 4 <= count; k += 4) {
		const uint64_t gi0 = idx(k), gi1 = idx(k + 1), gi2 = idx(k + 2), gi3 = idx(k + 3);
		__m256d va = _mm256_set_pd(gram_a[gi3], gram_a[gi2], gram_a[gi1], gram_a[gi0]);
		__m256d vb = _mm256_set_pd(gram_b[gi3], gram_b[gi2], gram_b[gi1], gram_b[gi0]);
		__m256d vpj = _mm256_loadu_pd(prev + k);
		__m256d vpjm1 = _mm256_loadu_pd(prev_m1 + k);
		__m256d vppjm1 = _mm256_loadu_pd(prev_prev_m1 + k);

		_mm256_storeu_pd(next + k,
			_mm256_fmadd_pd(vpj, va, _mm256_fmsub_pd(vpjm1, va, _mm256_mul_pd(vppjm1, vb))));
	}
	vec_kernel_diag_step_tail(next, prev, prev_m1, prev_prev_m1, gram_a, gram_b, idx, k, count);
}

template<typename IndexFn>
FORCE_INLINE void vec_kernel_diag_step(
	float* RESTRICT next,
	const float* RESTRICT prev,
	const float* RESTRICT prev_m1,
	const float* RESTRICT prev_prev_m1,
	const float* RESTRICT gram_a,
	const float* RESTRICT gram_b,
	IndexFn idx,
	uint64_t count)
{
	uint64_t k = 0;
	for (; k + 8 <= count; k += 8) {
		alignas(32) float a_vals[8], b_vals[8];
		for (int m = 0; m < 8; ++m) {
			const uint64_t gi = idx(k + m);
			a_vals[m] = gram_a[gi];
			b_vals[m] = gram_b[gi];
		}
		__m256 va = _mm256_load_ps(a_vals);
		__m256 vb = _mm256_load_ps(b_vals);
		__m256 vpj = _mm256_loadu_ps(prev + k);
		__m256 vpjm1 = _mm256_loadu_ps(prev_m1 + k);
		__m256 vppjm1 = _mm256_loadu_ps(prev_prev_m1 + k);

		_mm256_storeu_ps(next + k,
			_mm256_fmadd_ps(vpj, va, _mm256_fmsub_ps(vpjm1, va, _mm256_mul_ps(vppjm1, vb))));
	}
	vec_kernel_diag_step_tail(next, prev, prev_m1, prev_prev_m1, gram_a, gram_b, idx, k, count);
}

#else

FORCE_INLINE void vec_mult_add(float* out, const float* other, float scalar, uint64_t size)
{
	if (size > 4 && size < 8) {
		const float32x4_t s = vdupq_n_f32(scalar);
		const float32x4_t v0 = vld1q_f32(other);
		const float32x4_t v1 = vld1q_f32(other + size - 4);
		const float32x4_t o0 = vld1q_f32(out);
		const float32x4_t o1 = vld1q_f32(out + size - 4);
		vst1q_f32(out, vfmaq_f32(o0, v0, s));
		vst1q_f32(out + size - 4, vfmaq_f32(o1, v1, s));
		return;
	}

	const uint64_t N = size / 4;
	const uint64_t tail = size & 3;

	float32x4_t scalar_v = vdupq_n_f32(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		float32x4_t a = vld1q_f32(other);
		float32x4_t b = vld1q_f32(out);
		b = vfmaq_f32(b, a, scalar_v);
		vst1q_f32(out, b);

		other += 4;
		out += 4;
	}

	for (uint64_t i = 0; i < tail; ++i) {
		out[i] += other[i] * scalar;
	}
}

FORCE_INLINE void vec_mult_add(double* out, const double* other, double scalar, uint64_t size) {
    const uint64_t N = size / 2;
    const uint64_t tail = size & 1;

    float64x2_t scalar_v = vdupq_n_f64(scalar);

    for (uint64_t i = 0; i < N; ++i) {
        float64x2_t a = vld1q_f64(other);
        float64x2_t b = vld1q_f64(out);
        b = vfmaq_f64(b, a, scalar_v);
        vst1q_f64(out, b);

        other += 2;
        out += 2;
    }
    if (tail) {
        *out += (*other) * scalar;
    }
}

FORCE_INLINE void vec_mult_assign(float* out, const float* other, float scalar, uint64_t size)
{
	if (size > 4 && size < 8) {
		const float32x4_t s = vdupq_n_f32(scalar);
		vst1q_f32(out, vmulq_f32(vld1q_f32(other), s));
		vst1q_f32(out + size - 4, vmulq_f32(vld1q_f32(other + size - 4), s));
		return;
	}

	const uint64_t N = size / 4;
	const uint64_t tail = size & 3;

	float32x4_t scalar_v = vdupq_n_f32(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		float32x4_t a = vld1q_f32(other);
		a = vmulq_f32(a, scalar_v);
		vst1q_f32(out, a);

		other += 4;
		out += 4;
	}

	for (uint64_t i = 0; i < tail; ++i) {
		out[i] = other[i] * scalar;
	}
}

FORCE_INLINE void vec_mult_assign(double* out, const double* other, double scalar, uint64_t size) {
    const uint64_t N = size / 2;
    const uint64_t tail = size & 1;

    float64x2_t scalar_v = vdupq_n_f64(scalar);

    for (uint64_t i = 0; i < N; ++i) {
        float64x2_t a = vld1q_f64(other);
        a = vmulq_f64(a, scalar_v);
        vst1q_f64(out, a);

        other += 2;
        out += 2;
    }
    if (tail) {
        *out = (*other) * scalar;
    }
}

template<typename IndexFn>
FORCE_INLINE void vec_kernel_diag_step(
	double* RESTRICT next,
	const double* RESTRICT prev,
	const double* RESTRICT prev_m1,
	const double* RESTRICT prev_prev_m1,
	const double* RESTRICT gram_a,
	const double* RESTRICT gram_b,
	IndexFn idx,
	uint64_t count)
{
	uint64_t k = 0;
	for (; k + 2 <= count; k += 2) {
		const uint64_t gi0 = idx(k), gi1 = idx(k + 1);
		double a_vals[2] = { gram_a[gi0], gram_a[gi1] };
		double b_vals[2] = { gram_b[gi0], gram_b[gi1] };
		float64x2_t va = vld1q_f64(a_vals);
		float64x2_t vb = vld1q_f64(b_vals);
		float64x2_t vpj = vld1q_f64(prev + k);
		float64x2_t vpjm1 = vld1q_f64(prev_m1 + k);
		float64x2_t vppjm1 = vld1q_f64(prev_prev_m1 + k);

		float64x2_t result = vmulq_f64(vpj, va);
		result = vfmaq_f64(result, vpjm1, va);
		result = vfmsq_f64(result, vppjm1, vb);
		vst1q_f64(next + k, result);
	}
	vec_kernel_diag_step_tail(next, prev, prev_m1, prev_prev_m1, gram_a, gram_b, idx, k, count);
}

template<typename IndexFn>
FORCE_INLINE void vec_kernel_diag_step(
	float* RESTRICT next,
	const float* RESTRICT prev,
	const float* RESTRICT prev_m1,
	const float* RESTRICT prev_prev_m1,
	const float* RESTRICT gram_a,
	const float* RESTRICT gram_b,
	IndexFn idx,
	uint64_t count)
{
	uint64_t k = 0;
	for (; k + 4 <= count; k += 4) {
		float a_vals[4], b_vals[4];
		for (int m = 0; m < 4; ++m) {
			const uint64_t gi = idx(k + m);
			a_vals[m] = gram_a[gi];
			b_vals[m] = gram_b[gi];
		}
		float32x4_t va = vld1q_f32(a_vals);
		float32x4_t vb = vld1q_f32(b_vals);
		float32x4_t vpj = vld1q_f32(prev + k);
		float32x4_t vpjm1 = vld1q_f32(prev_m1 + k);
		float32x4_t vppjm1 = vld1q_f32(prev_prev_m1 + k);

		float32x4_t result = vmulq_f32(vpj, va);
		result = vfmaq_f32(result, vpjm1, va);
		result = vfmsq_f32(result, vppjm1, vb);
		vst1q_f32(next + k, result);
	}
	vec_kernel_diag_step_tail(next, prev, prev_m1, prev_prev_m1, gram_a, gram_b, idx, k, count);
}

#endif

#endif
