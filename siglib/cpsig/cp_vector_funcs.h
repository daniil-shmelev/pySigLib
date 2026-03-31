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

#else

FORCE_INLINE void vec_mult_add(float* out, const float* other, float scalar, uint64_t size)
{
	const uint64_t N = size / 4;
	const uint64_t tail = size & 3;

	float32x4_t scalar_v = vdupq_n_f32(scalar);

	for (uint64_t i = 0; i < N; ++i) {
		float32x4_t a = vld1q_f32(other);
		float32x4_t b = vld1q_f32(out);

		a = vmulq_f32(a, scalar_v);
		b = vaddq_f32(b, a);

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

        a = vmulq_f64(a, scalar_v);
        b = vaddq_f64(b, a);

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


#endif

#endif
