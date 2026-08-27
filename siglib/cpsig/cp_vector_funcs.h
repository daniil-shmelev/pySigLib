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

FORCE_INLINE float dot_product(const float* a, const float* b, size_t N) {
	__m256 sum0 = _mm256_setzero_ps();
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();

	size_t k = 0;
	size_t limit = N & ~31UL;
	for (; k < limit; k += 32) {
		sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[k]), _mm256_loadu_ps(&b[k]), sum0);
		sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[k + 8]), _mm256_loadu_ps(&b[k + 8]), sum1);
		sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[k + 16]), _mm256_loadu_ps(&b[k + 16]), sum2);
		sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[k + 24]), _mm256_loadu_ps(&b[k + 24]), sum3);
	}
	__m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
	limit = N & ~7UL;
	for (; k < limit; k += 8) {
		__m256 va = _mm256_loadu_ps(&a[k]);
		__m256 vb = _mm256_loadu_ps(&b[k]);
		sum = _mm256_fmadd_ps(va, vb, sum);
	}

	__m128 hi = _mm256_extractf128_ps(sum, 1);
	__m128 lo = _mm256_castps256_ps128(sum);
	__m128 s128 = _mm_add_ps(lo, hi);
	s128 = _mm_hadd_ps(s128, s128);
	s128 = _mm_hadd_ps(s128, s128);
	float out = _mm_cvtss_f32(s128);

	for (; k < N; ++k) {
		out += a[k] * b[k];
	}

	return out;
}

FORCE_INLINE double dot_product(const double* a, const double* b, size_t N) {
	__m256d sum0 = _mm256_setzero_pd();
	__m256d sum1 = _mm256_setzero_pd();
	__m256d sum2 = _mm256_setzero_pd();
	__m256d sum3 = _mm256_setzero_pd();

	size_t k = 0;
	size_t limit = N & ~15UL;
	for (; k < limit; k += 16) {
		sum0 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[k]), _mm256_loadu_pd(&b[k]), sum0);
		sum1 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[k + 4]), _mm256_loadu_pd(&b[k + 4]), sum1);
		sum2 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[k + 8]), _mm256_loadu_pd(&b[k + 8]), sum2);
		sum3 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[k + 12]), _mm256_loadu_pd(&b[k + 12]), sum3);
	}
	__m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
	limit = N & ~3UL;
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

FORCE_INLINE void dot_product_pair(
	const float* weights, const float* a, const float* b, size_t N,
	float& result_a, float& result_b
) {
	__m256 sum_a = _mm256_setzero_ps();
	__m256 sum_b = _mm256_setzero_ps();
	size_t k = 0;
	for (; k + 8 <= N; k += 8) {
		const __m256 w = _mm256_loadu_ps(weights + k);
		sum_a = _mm256_fmadd_ps(w, _mm256_loadu_ps(a + k), sum_a);
		sum_b = _mm256_fmadd_ps(w, _mm256_loadu_ps(b + k), sum_b);
	}

	__m128 sum_a_128 = _mm_add_ps(
		_mm256_castps256_ps128(sum_a), _mm256_extractf128_ps(sum_a, 1));
	__m128 sum_b_128 = _mm_add_ps(
		_mm256_castps256_ps128(sum_b), _mm256_extractf128_ps(sum_b, 1));
	sum_a_128 = _mm_hadd_ps(sum_a_128, sum_a_128);
	sum_a_128 = _mm_hadd_ps(sum_a_128, sum_a_128);
	sum_b_128 = _mm_hadd_ps(sum_b_128, sum_b_128);
	sum_b_128 = _mm_hadd_ps(sum_b_128, sum_b_128);
	result_a = _mm_cvtss_f32(sum_a_128);
	result_b = _mm_cvtss_f32(sum_b_128);
	for (; k < N; ++k) {
		result_a += weights[k] * a[k];
		result_b += weights[k] * b[k];
	}
}

FORCE_INLINE void dot_product_pair(
	const double* weights, const double* a, const double* b, size_t N,
	double& result_a, double& result_b
) {
	__m256d sum_a = _mm256_setzero_pd();
	__m256d sum_b = _mm256_setzero_pd();
	size_t k = 0;
	for (; k + 4 <= N; k += 4) {
		const __m256d w = _mm256_loadu_pd(weights + k);
		sum_a = _mm256_fmadd_pd(w, _mm256_loadu_pd(a + k), sum_a);
		sum_b = _mm256_fmadd_pd(w, _mm256_loadu_pd(b + k), sum_b);
	}

	double values_a[4];
	double values_b[4];
	_mm256_storeu_pd(values_a, sum_a);
	_mm256_storeu_pd(values_b, sum_b);
	result_a = values_a[0] + values_a[1] + values_a[2] + values_a[3];
	result_b = values_b[0] + values_b[1] + values_b[2] + values_b[3];
	for (; k < N; ++k) {
		result_a += weights[k] * a[k];
		result_b += weights[k] * b[k];
	}
}

FORCE_INLINE float dot_product_mult_add(
	const float* a, const float* b, float* out, float scalar, size_t N
) {
	__m256 sum = _mm256_setzero_ps();
	const __m256 scalar_v = _mm256_set1_ps(scalar);
	size_t k = 0;
	for (; k + 8 <= N; k += 8) {
		const __m256 va = _mm256_loadu_ps(&a[k]);
		const __m256 vb = _mm256_loadu_ps(&b[k]);
		sum = _mm256_fmadd_ps(va, vb, sum);
		_mm256_storeu_ps(
			&out[k], _mm256_fmadd_ps(vb, scalar_v, _mm256_loadu_ps(&out[k])));
	}

	__m128 sum128 = _mm_add_ps(
		_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
	sum128 = _mm_hadd_ps(sum128, sum128);
	sum128 = _mm_hadd_ps(sum128, sum128);
	float result = _mm_cvtss_f32(sum128);
	for (; k < N; ++k) {
		result += a[k] * b[k];
		out[k] += b[k] * scalar;
	}
	return result;
}

FORCE_INLINE double dot_product_mult_add(
	const double* a, const double* b, double* out, double scalar, size_t N
) {
	__m256d sum = _mm256_setzero_pd();
	const __m256d scalar_v = _mm256_set1_pd(scalar);
	size_t k = 0;
	for (; k + 4 <= N; k += 4) {
		const __m256d va = _mm256_loadu_pd(&a[k]);
		const __m256d vb = _mm256_loadu_pd(&b[k]);
		sum = _mm256_fmadd_pd(va, vb, sum);
		_mm256_storeu_pd(
			&out[k], _mm256_fmadd_pd(vb, scalar_v, _mm256_loadu_pd(&out[k])));
	}

	double values[4];
	_mm256_storeu_pd(values, sum);
	double result = values[0] + values[1] + values[2] + values[3];
	for (; k < N; ++k) {
		result += a[k] * b[k];
		out[k] += b[k] * scalar;
	}
	return result;
}

FORCE_INLINE void vec_add_scaled(float* out, const float* a, const float* b, float scalar, uint64_t size) {
	const __m256 sv = _mm256_set1_ps(scalar);
	uint64_t k = 0;
	uint64_t limit = size & ~7ULL;
	for (; k < limit; k += 8) {
		__m256 va = _mm256_loadu_ps(&a[k]);
		__m256 vb = _mm256_loadu_ps(&b[k]);
		_mm256_storeu_ps(&out[k], _mm256_mul_ps(_mm256_add_ps(va, vb), sv));
	}
	for (; k < size; ++k) out[k] = (a[k] + b[k]) * scalar;
}

FORCE_INLINE void vec_add_scaled(double* out, const double* a, const double* b, double scalar, uint64_t size) {
	const __m256d sv = _mm256_set1_pd(scalar);
	uint64_t k = 0;
	uint64_t limit = size & ~3ULL;
	for (; k < limit; k += 4) {
		__m256d va = _mm256_loadu_pd(&a[k]);
		__m256d vb = _mm256_loadu_pd(&b[k]);
		_mm256_storeu_pd(&out[k], _mm256_mul_pd(_mm256_add_pd(va, vb), sv));
	}
	for (; k < size; ++k) out[k] = (a[k] + b[k]) * scalar;
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

// 4-wide batched double operations (interleaved layout: 4 doubles per logical element)

FORCE_INLINE void vec4_add(double* RESTRICT out, const double* RESTRICT a, const double* RESTRICT b, uint64_t count) {
	for (uint64_t k = 0; k < count; ++k)
		_mm256_storeu_pd(&out[k * 4], _mm256_add_pd(_mm256_loadu_pd(&a[k * 4]), _mm256_loadu_pd(&b[k * 4])));
}

FORCE_INLINE void vec4_add_inplace(double* RESTRICT a, const double* RESTRICT b, uint64_t count) {
	for (uint64_t k = 0; k < count; ++k)
		_mm256_storeu_pd(&a[k * 4], _mm256_add_pd(_mm256_loadu_pd(&a[k * 4]), _mm256_loadu_pd(&b[k * 4])));
}

FORCE_INLINE void vec4_fmadd(double* RESTRICT out, const double* RESTRICT a, double scalar, uint64_t count) {
	__m256d s = _mm256_set1_pd(scalar);
	for (uint64_t k = 0; k < count; ++k)
		_mm256_storeu_pd(&out[k * 4], _mm256_fmadd_pd(s, _mm256_loadu_pd(&a[k * 4]), _mm256_loadu_pd(&out[k * 4])));
}

FORCE_INLINE void vec4_scale(double* RESTRICT out, const double* RESTRICT a, double scalar, uint64_t count) {
	__m256d s = _mm256_set1_pd(scalar);
	for (uint64_t k = 0; k < count; ++k)
		_mm256_storeu_pd(&out[k * 4], _mm256_mul_pd(s, _mm256_loadu_pd(&a[k * 4])));
}

FORCE_INLINE void vec4_commutator_accum(
	double* RESTRICT result,
	const double* RESTRICT v1, const double* RESTRICT v2,
	const uint32_t* ki, const uint32_t* kj, const double* kval,
	uint32_t start, uint32_t end
) {
	__m256d sum = _mm256_setzero_pd();
	for (uint32_t idx = start; idx < end; ++idx) {
		const uint32_t ci = ki[idx], cj = kj[idx];
		const __m256d bracket = _mm256_fmsub_pd(
			_mm256_loadu_pd(&v1[ci * 4]), _mm256_loadu_pd(&v2[cj * 4]),
			_mm256_mul_pd(_mm256_loadu_pd(&v1[cj * 4]), _mm256_loadu_pd(&v2[ci * 4])));
		sum = _mm256_fmadd_pd(_mm256_set1_pd(kval[idx]),
			bracket,
			sum);
	}
	_mm256_storeu_pd(result, sum);
}

FORCE_INLINE void vec4_bracket_grad(
	double* RESTRICT dm_lf, double* RESTRICT dm_rf,
	const double* RESTRICT dm_w, const double* RESTRICT v1, const double* RESTRICT v2,
	uint32_t i, uint32_t j,
	const uint32_t* ij_k, const double* ij_c,
	uint32_t start, uint32_t end
) {
	__m256d S = _mm256_setzero_pd();
	for (uint32_t idx = start; idx < end; ++idx)
		S = _mm256_add_pd(S, _mm256_mul_pd(
			_mm256_set1_pd(ij_c[idx]), _mm256_loadu_pd(&dm_w[ij_k[idx] * 4])));
	const __m256d v1i = _mm256_loadu_pd(&v1[i * 4]);
	const __m256d v1j = _mm256_loadu_pd(&v1[j * 4]);
	const __m256d v2i = _mm256_loadu_pd(&v2[i * 4]);
	const __m256d v2j = _mm256_loadu_pd(&v2[j * 4]);
	_mm256_storeu_pd(&dm_lf[i * 4], _mm256_fmadd_pd(S, v2j, _mm256_loadu_pd(&dm_lf[i * 4])));
	_mm256_storeu_pd(&dm_lf[j * 4], _mm256_fnmadd_pd(S, v2i, _mm256_loadu_pd(&dm_lf[j * 4])));
	_mm256_storeu_pd(&dm_rf[j * 4], _mm256_fmadd_pd(S, v1i, _mm256_loadu_pd(&dm_rf[j * 4])));
	_mm256_storeu_pd(&dm_rf[i * 4], _mm256_fnmadd_pd(S, v1j, _mm256_loadu_pd(&dm_rf[i * 4])));
}

// Fixed-width operations on aligned coefficients from several batch elements.

inline constexpr uint64_t vec_batch_bytes = 32;

FORCE_INLINE void vec_batch_fill(float* out, float value) {
	_mm256_store_ps(out, _mm256_set1_ps(value));
}

FORCE_INLINE void vec_batch_fill(double* out, double value) {
	_mm256_store_pd(out, _mm256_set1_pd(value));
}

FORCE_INLINE void vec_batch_copy(float* out, const float* value) {
	_mm256_store_ps(out, _mm256_load_ps(value));
}

FORCE_INLINE void vec_batch_copy(double* out, const double* value) {
	_mm256_store_pd(out, _mm256_load_pd(value));
}

FORCE_INLINE void vec_batch_add(
	float* out, const float* left, const float* right
) {
	_mm256_store_ps(out,
		_mm256_add_ps(_mm256_load_ps(left), _mm256_load_ps(right)));
}

FORCE_INLINE void vec_batch_add(
	double* out, const double* left, const double* right
) {
	_mm256_store_pd(out,
		_mm256_add_pd(_mm256_load_pd(left), _mm256_load_pd(right)));
}

FORCE_INLINE void vec_batch_add_inplace(float* out, const float* value) {
	_mm256_store_ps(out,
		_mm256_add_ps(_mm256_load_ps(out), _mm256_load_ps(value)));
}

FORCE_INLINE void vec_batch_add_inplace(double* out, const double* value) {
	_mm256_store_pd(out,
		_mm256_add_pd(_mm256_load_pd(out), _mm256_load_pd(value)));
}

FORCE_INLINE void vec_batch_subtract(
	float* out, const float* left, const float* right
) {
	_mm256_store_ps(out,
		_mm256_sub_ps(_mm256_load_ps(left), _mm256_load_ps(right)));
}

FORCE_INLINE void vec_batch_subtract(
	double* out, const double* left, const double* right
) {
	_mm256_store_pd(out,
		_mm256_sub_pd(_mm256_load_pd(left), _mm256_load_pd(right)));
}

FORCE_INLINE void vec_batch_subtract_inplace(float* out, const float* value) {
	_mm256_store_ps(out,
		_mm256_sub_ps(_mm256_load_ps(out), _mm256_load_ps(value)));
}

FORCE_INLINE void vec_batch_subtract_inplace(double* out, const double* value) {
	_mm256_store_pd(out,
		_mm256_sub_pd(_mm256_load_pd(out), _mm256_load_pd(value)));
}

FORCE_INLINE void vec_batch_multiply(
	float* out, const float* left, const float* right
) {
	_mm256_store_ps(out,
		_mm256_mul_ps(_mm256_load_ps(left), _mm256_load_ps(right)));
}

FORCE_INLINE void vec_batch_multiply(
	double* out, const double* left, const double* right
) {
	_mm256_store_pd(out,
		_mm256_mul_pd(_mm256_load_pd(left), _mm256_load_pd(right)));
}

FORCE_INLINE void vec_batch_multiply_inplace(float* out, const float* value) {
	_mm256_store_ps(out,
		_mm256_mul_ps(_mm256_load_ps(out), _mm256_load_ps(value)));
}

FORCE_INLINE void vec_batch_multiply_inplace(double* out, const double* value) {
	_mm256_store_pd(out,
		_mm256_mul_pd(_mm256_load_pd(out), _mm256_load_pd(value)));
}

FORCE_INLINE void vec_batch_scale(float* out, const float* value, float scalar) {
	_mm256_store_ps(out,
		_mm256_mul_ps(_mm256_load_ps(value), _mm256_set1_ps(scalar)));
}

FORCE_INLINE void vec_batch_scale(double* out, const double* value, double scalar) {
	_mm256_store_pd(out,
		_mm256_mul_pd(_mm256_load_pd(value), _mm256_set1_pd(scalar)));
}

FORCE_INLINE void vec_batch_negate_inplace(float* out) {
	_mm256_store_ps(out,
		_mm256_xor_ps(_mm256_load_ps(out), _mm256_set1_ps(-0.0f)));
}

FORCE_INLINE void vec_batch_negate_inplace(double* out) {
	_mm256_store_pd(out,
		_mm256_xor_pd(_mm256_load_pd(out), _mm256_set1_pd(-0.0)));
}

FORCE_INLINE void vec_batch_multiply_add(
	float* out, const float* left, const float* right
) {
	_mm256_store_ps(out, _mm256_fmadd_ps(
		_mm256_load_ps(left), _mm256_load_ps(right), _mm256_load_ps(out)));
}

FORCE_INLINE void vec_batch_multiply_add(
	double* out, const double* left, const double* right
) {
	_mm256_store_pd(out, _mm256_fmadd_pd(
		_mm256_load_pd(left), _mm256_load_pd(right), _mm256_load_pd(out)));
}

FORCE_INLINE void vec_batch_scaled_add(
	float* out, const float* value, float scalar
) {
	_mm256_store_ps(out, _mm256_fmadd_ps(
		_mm256_load_ps(value), _mm256_set1_ps(scalar), _mm256_load_ps(out)));
}

FORCE_INLINE void vec_batch_scaled_add(
	double* out, const double* value, double scalar
) {
	_mm256_store_pd(out, _mm256_fmadd_pd(
		_mm256_load_pd(value), _mm256_set1_pd(scalar), _mm256_load_pd(out)));
}

FORCE_INLINE void vec_batch_multiply_add_scaled(
	float* out, const float* left, const float* right, float scalar
) {
	const __m256 product = _mm256_mul_ps(
		_mm256_load_ps(left), _mm256_load_ps(right));
	_mm256_store_ps(out, _mm256_fmadd_ps(
		product, _mm256_set1_ps(scalar), _mm256_load_ps(out)));
}

FORCE_INLINE void vec_batch_multiply_add_scaled(
	double* out, const double* left, const double* right, double scalar
) {
	const __m256d product = _mm256_mul_pd(
		_mm256_load_pd(left), _mm256_load_pd(right));
	_mm256_store_pd(out, _mm256_fmadd_pd(
		product, _mm256_set1_pd(scalar), _mm256_load_pd(out)));
}

FORCE_INLINE void vec_batch_multiply_add3(
	float* out, const float* first, const float* second, const float* third
) {
	const __m256 product = _mm256_mul_ps(
		_mm256_load_ps(first), _mm256_load_ps(second));
	_mm256_store_ps(out, _mm256_fmadd_ps(
		product, _mm256_load_ps(third), _mm256_load_ps(out)));
}

FORCE_INLINE void vec_batch_multiply_add3(
	double* out, const double* first, const double* second, const double* third
) {
	const __m256d product = _mm256_mul_pd(
		_mm256_load_pd(first), _mm256_load_pd(second));
	_mm256_store_pd(out, _mm256_fmadd_pd(
		product, _mm256_load_pd(third), _mm256_load_pd(out)));
}

FORCE_INLINE void vec_batch_subtract_product(
	float* out, const float* left, const float* right
) {
	_mm256_store_ps(out, _mm256_fnmadd_ps(
		_mm256_load_ps(left), _mm256_load_ps(right), _mm256_load_ps(out)));
}

FORCE_INLINE void vec_batch_subtract_product(
	double* out, const double* left, const double* right
) {
	_mm256_store_pd(out, _mm256_fnmadd_pd(
		_mm256_load_pd(left), _mm256_load_pd(right), _mm256_load_pd(out)));
}

FORCE_INLINE bool vec_batch_is_zero(const float* value) {
	const __m256 unequal = _mm256_cmp_ps(
		_mm256_load_ps(value), _mm256_setzero_ps(), _CMP_NEQ_UQ);
	return _mm256_movemask_ps(unequal) == 0;
}

FORCE_INLINE bool vec_batch_is_zero(const double* value) {
	const __m256d unequal = _mm256_cmp_pd(
		_mm256_load_pd(value), _mm256_setzero_pd(), _CMP_NEQ_UQ);
	return _mm256_movemask_pd(unequal) == 0;
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

FORCE_INLINE float dot_product(const float* a, const float* b, size_t N) {
	float32x4_t sum = vdupq_n_f32(0.0f);
	size_t k = 0;
	for (; k + 4 <= N; k += 4)
		sum = vfmaq_f32(sum, vld1q_f32(&a[k]), vld1q_f32(&b[k]));
	float out = vaddvq_f32(sum);
	for (; k < N; ++k) out += a[k] * b[k];
	return out;
}

FORCE_INLINE double dot_product(const double* a, const double* b, size_t N) {
	float64x2_t sum = vdupq_n_f64(0.0);
	size_t k = 0;
	for (; k + 2 <= N; k += 2)
		sum = vfmaq_f64(sum, vld1q_f64(&a[k]), vld1q_f64(&b[k]));
	double out = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
	for (; k < N; ++k) out += a[k] * b[k];
	return out;
}

FORCE_INLINE float dot_product_mult_add(
	const float* a, const float* b, float* out, float scalar, size_t N
) {
	float32x4_t sum = vdupq_n_f32(0.0f);
	const float32x4_t scalar_v = vdupq_n_f32(scalar);
	size_t k = 0;
	for (; k + 4 <= N; k += 4) {
		const float32x4_t va = vld1q_f32(&a[k]);
		const float32x4_t vb = vld1q_f32(&b[k]);
		sum = vfmaq_f32(sum, va, vb);
		vst1q_f32(&out[k], vfmaq_f32(vld1q_f32(&out[k]), vb, scalar_v));
	}
	float result = vaddvq_f32(sum);
	for (; k < N; ++k) {
		result += a[k] * b[k];
		out[k] += b[k] * scalar;
	}
	return result;
}

FORCE_INLINE double dot_product_mult_add(
	const double* a, const double* b, double* out, double scalar, size_t N
) {
	float64x2_t sum = vdupq_n_f64(0.0);
	const float64x2_t scalar_v = vdupq_n_f64(scalar);
	size_t k = 0;
	for (; k + 2 <= N; k += 2) {
		const float64x2_t va = vld1q_f64(&a[k]);
		const float64x2_t vb = vld1q_f64(&b[k]);
		sum = vfmaq_f64(sum, va, vb);
		vst1q_f64(&out[k], vfmaq_f64(vld1q_f64(&out[k]), vb, scalar_v));
	}
	double result = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
	for (; k < N; ++k) {
		result += a[k] * b[k];
		out[k] += b[k] * scalar;
	}
	return result;
}

FORCE_INLINE void vec_add_scaled(float* out, const float* a, const float* b, float scalar, uint64_t size) {
	float32x4_t sv = vdupq_n_f32(scalar);
	uint64_t k = 0;
	for (; k + 4 <= size; k += 4)
		vst1q_f32(&out[k], vmulq_f32(vaddq_f32(vld1q_f32(&a[k]), vld1q_f32(&b[k])), sv));
	for (; k < size; ++k) out[k] = (a[k] + b[k]) * scalar;
}

FORCE_INLINE void vec_add_scaled(double* out, const double* a, const double* b, double scalar, uint64_t size) {
	float64x2_t sv = vdupq_n_f64(scalar);
	uint64_t k = 0;
	for (; k + 2 <= size; k += 2)
		vst1q_f64(&out[k], vmulq_f64(vaddq_f64(vld1q_f64(&a[k]), vld1q_f64(&b[k])), sv));
	for (; k < size; ++k) out[k] = (a[k] + b[k]) * scalar;
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

// 4-wide batched double operations (interleaved layout: 4 doubles per logical element)

FORCE_INLINE void vec4_add(double* RESTRICT out, const double* RESTRICT a, const double* RESTRICT b, uint64_t count) {
	for (uint64_t k = 0; k < count; ++k) {
		const uint64_t off = k * 4;
		vst1q_f64(&out[off],     vaddq_f64(vld1q_f64(&a[off]),     vld1q_f64(&b[off])));
		vst1q_f64(&out[off + 2], vaddq_f64(vld1q_f64(&a[off + 2]), vld1q_f64(&b[off + 2])));
	}
}

FORCE_INLINE void vec4_add_inplace(double* RESTRICT a, const double* RESTRICT b, uint64_t count) {
	for (uint64_t k = 0; k < count; ++k) {
		const uint64_t off = k * 4;
		vst1q_f64(&a[off],     vaddq_f64(vld1q_f64(&a[off]),     vld1q_f64(&b[off])));
		vst1q_f64(&a[off + 2], vaddq_f64(vld1q_f64(&a[off + 2]), vld1q_f64(&b[off + 2])));
	}
}

FORCE_INLINE void vec4_fmadd(double* RESTRICT out, const double* RESTRICT a, double scalar, uint64_t count) {
	float64x2_t s = vdupq_n_f64(scalar);
	for (uint64_t k = 0; k < count; ++k) {
		const uint64_t off = k * 4;
		vst1q_f64(&out[off],     vfmaq_f64(vld1q_f64(&out[off]),     vld1q_f64(&a[off]),     s));
		vst1q_f64(&out[off + 2], vfmaq_f64(vld1q_f64(&out[off + 2]), vld1q_f64(&a[off + 2]), s));
	}
}

FORCE_INLINE void vec4_scale(double* RESTRICT out, const double* RESTRICT a, double scalar, uint64_t count) {
	float64x2_t s = vdupq_n_f64(scalar);
	for (uint64_t k = 0; k < count; ++k) {
		const uint64_t off = k * 4;
		vst1q_f64(&out[off],     vmulq_f64(vld1q_f64(&a[off]),     s));
		vst1q_f64(&out[off + 2], vmulq_f64(vld1q_f64(&a[off + 2]), s));
	}
}

FORCE_INLINE void vec4_commutator_accum(
	double* RESTRICT result,
	const double* RESTRICT v1, const double* RESTRICT v2,
	const uint32_t* ki, const uint32_t* kj, const double* kval,
	uint32_t start, uint32_t end
) {
	float64x2_t sum_lo = vdupq_n_f64(0.0), sum_hi = vdupq_n_f64(0.0);
	for (uint32_t idx = start; idx < end; ++idx) {
		const uint32_t ci = ki[idx], cj = kj[idx];
		float64x2_t val = vdupq_n_f64(kval[idx]);
		float64x2_t blo = vfmaq_f64(vnegq_f64(vmulq_f64(vld1q_f64(&v1[cj * 4]),     vld1q_f64(&v2[ci * 4]))),
		                             vld1q_f64(&v1[ci * 4]),     vld1q_f64(&v2[cj * 4]));
		float64x2_t bhi = vfmaq_f64(vnegq_f64(vmulq_f64(vld1q_f64(&v1[cj * 4 + 2]), vld1q_f64(&v2[ci * 4 + 2]))),
		                             vld1q_f64(&v1[ci * 4 + 2]), vld1q_f64(&v2[cj * 4 + 2]));
		sum_lo = vfmaq_f64(sum_lo, val, blo);
		sum_hi = vfmaq_f64(sum_hi, val, bhi);
	}
	vst1q_f64(result, sum_lo);
	vst1q_f64(result + 2, sum_hi);
}

FORCE_INLINE void vec4_bracket_grad(
	double* RESTRICT dm_lf, double* RESTRICT dm_rf,
	const double* RESTRICT dm_w, const double* RESTRICT v1, const double* RESTRICT v2,
	uint32_t i, uint32_t j,
	const uint32_t* ij_k, const double* ij_c,
	uint32_t start, uint32_t end
) {
	float64x2_t slo = vdupq_n_f64(0.0), shi = vdupq_n_f64(0.0);
	for (uint32_t idx = start; idx < end; ++idx) {
		float64x2_t c = vdupq_n_f64(ij_c[idx]);
		const uint32_t k = ij_k[idx];
		slo = vaddq_f64(slo, vmulq_f64(c, vld1q_f64(&dm_w[k * 4])));
		shi = vaddq_f64(shi, vmulq_f64(c, vld1q_f64(&dm_w[k * 4 + 2])));
	}
	const float64x2_t v1ilo = vld1q_f64(&v1[i * 4]), v1ihi = vld1q_f64(&v1[i * 4 + 2]);
	const float64x2_t v1jlo = vld1q_f64(&v1[j * 4]), v1jhi = vld1q_f64(&v1[j * 4 + 2]);
	const float64x2_t v2ilo = vld1q_f64(&v2[i * 4]), v2ihi = vld1q_f64(&v2[i * 4 + 2]);
	const float64x2_t v2jlo = vld1q_f64(&v2[j * 4]), v2jhi = vld1q_f64(&v2[j * 4 + 2]);
	vst1q_f64(&dm_lf[i * 4],     vfmaq_f64(vld1q_f64(&dm_lf[i * 4]),     slo, v2jlo));
	vst1q_f64(&dm_lf[i * 4 + 2], vfmaq_f64(vld1q_f64(&dm_lf[i * 4 + 2]), shi, v2jhi));
	vst1q_f64(&dm_lf[j * 4],     vfmsq_f64(vld1q_f64(&dm_lf[j * 4]),     slo, v2ilo));
	vst1q_f64(&dm_lf[j * 4 + 2], vfmsq_f64(vld1q_f64(&dm_lf[j * 4 + 2]), shi, v2ihi));
	vst1q_f64(&dm_rf[j * 4],     vfmaq_f64(vld1q_f64(&dm_rf[j * 4]),     slo, v1ilo));
	vst1q_f64(&dm_rf[j * 4 + 2], vfmaq_f64(vld1q_f64(&dm_rf[j * 4 + 2]), shi, v1ihi));
	vst1q_f64(&dm_rf[i * 4],     vfmsq_f64(vld1q_f64(&dm_rf[i * 4]),     slo, v1jlo));
	vst1q_f64(&dm_rf[i * 4 + 2], vfmsq_f64(vld1q_f64(&dm_rf[i * 4 + 2]), shi, v1jhi));
}

// Fixed-width operations on coefficients from several batch elements.

inline constexpr uint64_t vec_batch_bytes = 16;

FORCE_INLINE void vec_batch_fill(float* out, float value) {
	vst1q_f32(out, vdupq_n_f32(value));
}

FORCE_INLINE void vec_batch_fill(double* out, double value) {
	vst1q_f64(out, vdupq_n_f64(value));
}

FORCE_INLINE void vec_batch_copy(float* out, const float* value) {
	vst1q_f32(out, vld1q_f32(value));
}

FORCE_INLINE void vec_batch_copy(double* out, const double* value) {
	vst1q_f64(out, vld1q_f64(value));
}

FORCE_INLINE void vec_batch_add(
	float* out, const float* left, const float* right
) {
	vst1q_f32(out, vaddq_f32(vld1q_f32(left), vld1q_f32(right)));
}

FORCE_INLINE void vec_batch_add(
	double* out, const double* left, const double* right
) {
	vst1q_f64(out, vaddq_f64(vld1q_f64(left), vld1q_f64(right)));
}

FORCE_INLINE void vec_batch_add_inplace(float* out, const float* value) {
	vst1q_f32(out, vaddq_f32(vld1q_f32(out), vld1q_f32(value)));
}

FORCE_INLINE void vec_batch_add_inplace(double* out, const double* value) {
	vst1q_f64(out, vaddq_f64(vld1q_f64(out), vld1q_f64(value)));
}

FORCE_INLINE void vec_batch_subtract(
	float* out, const float* left, const float* right
) {
	vst1q_f32(out, vsubq_f32(vld1q_f32(left), vld1q_f32(right)));
}

FORCE_INLINE void vec_batch_subtract(
	double* out, const double* left, const double* right
) {
	vst1q_f64(out, vsubq_f64(vld1q_f64(left), vld1q_f64(right)));
}

FORCE_INLINE void vec_batch_subtract_inplace(float* out, const float* value) {
	vst1q_f32(out, vsubq_f32(vld1q_f32(out), vld1q_f32(value)));
}

FORCE_INLINE void vec_batch_subtract_inplace(double* out, const double* value) {
	vst1q_f64(out, vsubq_f64(vld1q_f64(out), vld1q_f64(value)));
}

FORCE_INLINE void vec_batch_multiply(
	float* out, const float* left, const float* right
) {
	vst1q_f32(out, vmulq_f32(vld1q_f32(left), vld1q_f32(right)));
}

FORCE_INLINE void vec_batch_multiply(
	double* out, const double* left, const double* right
) {
	vst1q_f64(out, vmulq_f64(vld1q_f64(left), vld1q_f64(right)));
}

FORCE_INLINE void vec_batch_multiply_inplace(float* out, const float* value) {
	vst1q_f32(out, vmulq_f32(vld1q_f32(out), vld1q_f32(value)));
}

FORCE_INLINE void vec_batch_multiply_inplace(double* out, const double* value) {
	vst1q_f64(out, vmulq_f64(vld1q_f64(out), vld1q_f64(value)));
}

FORCE_INLINE void vec_batch_scale(float* out, const float* value, float scalar) {
	vst1q_f32(out, vmulq_f32(vld1q_f32(value), vdupq_n_f32(scalar)));
}

FORCE_INLINE void vec_batch_scale(double* out, const double* value, double scalar) {
	vst1q_f64(out, vmulq_f64(vld1q_f64(value), vdupq_n_f64(scalar)));
}

FORCE_INLINE void vec_batch_negate_inplace(float* out) {
	vst1q_f32(out, vnegq_f32(vld1q_f32(out)));
}

FORCE_INLINE void vec_batch_negate_inplace(double* out) {
	vst1q_f64(out, vnegq_f64(vld1q_f64(out)));
}

FORCE_INLINE void vec_batch_multiply_add(
	float* out, const float* left, const float* right
) {
	vst1q_f32(out, vfmaq_f32(
		vld1q_f32(out), vld1q_f32(left), vld1q_f32(right)));
}

FORCE_INLINE void vec_batch_multiply_add(
	double* out, const double* left, const double* right
) {
	vst1q_f64(out, vfmaq_f64(
		vld1q_f64(out), vld1q_f64(left), vld1q_f64(right)));
}

FORCE_INLINE void vec_batch_scaled_add(
	float* out, const float* value, float scalar
) {
	vst1q_f32(out, vfmaq_f32(
		vld1q_f32(out), vld1q_f32(value), vdupq_n_f32(scalar)));
}

FORCE_INLINE void vec_batch_scaled_add(
	double* out, const double* value, double scalar
) {
	vst1q_f64(out, vfmaq_f64(
		vld1q_f64(out), vld1q_f64(value), vdupq_n_f64(scalar)));
}

FORCE_INLINE void vec_batch_multiply_add_scaled(
	float* out, const float* left, const float* right, float scalar
) {
	const float32x4_t product = vmulq_f32(vld1q_f32(left), vld1q_f32(right));
	vst1q_f32(out, vfmaq_f32(
		vld1q_f32(out), product, vdupq_n_f32(scalar)));
}

FORCE_INLINE void vec_batch_multiply_add_scaled(
	double* out, const double* left, const double* right, double scalar
) {
	const float64x2_t product = vmulq_f64(vld1q_f64(left), vld1q_f64(right));
	vst1q_f64(out, vfmaq_f64(
		vld1q_f64(out), product, vdupq_n_f64(scalar)));
}

FORCE_INLINE void vec_batch_multiply_add3(
	float* out, const float* first, const float* second, const float* third
) {
	const float32x4_t product = vmulq_f32(vld1q_f32(first), vld1q_f32(second));
	vst1q_f32(out, vfmaq_f32(
		vld1q_f32(out), product, vld1q_f32(third)));
}

FORCE_INLINE void vec_batch_multiply_add3(
	double* out, const double* first, const double* second, const double* third
) {
	const float64x2_t product = vmulq_f64(vld1q_f64(first), vld1q_f64(second));
	vst1q_f64(out, vfmaq_f64(
		vld1q_f64(out), product, vld1q_f64(third)));
}

FORCE_INLINE void vec_batch_subtract_product(
	float* out, const float* left, const float* right
) {
	vst1q_f32(out, vfmsq_f32(
		vld1q_f32(out), vld1q_f32(left), vld1q_f32(right)));
}

FORCE_INLINE void vec_batch_subtract_product(
	double* out, const double* left, const double* right
) {
	vst1q_f64(out, vfmsq_f64(
		vld1q_f64(out), vld1q_f64(left), vld1q_f64(right)));
}

FORCE_INLINE bool vec_batch_is_zero(const float* value) {
	const uint32x4_t equal = vceqq_f32(vld1q_f32(value), vdupq_n_f32(0.0f));
	return vgetq_lane_u32(equal, 0) != 0
		&& vgetq_lane_u32(equal, 1) != 0
		&& vgetq_lane_u32(equal, 2) != 0
		&& vgetq_lane_u32(equal, 3) != 0;
}

FORCE_INLINE bool vec_batch_is_zero(const double* value) {
	const uint64x2_t equal = vceqq_f64(vld1q_f64(value), vdupq_n_f64(0.0));
	return vgetq_lane_u64(equal, 0) != 0
		&& vgetq_lane_u64(equal, 1) != 0;
}

#endif

#endif

template<std::floating_point T>
FORCE_INLINE void dot_product_pair(
	const T* weights, const T* a, const T* b, size_t size,
	T& result_a, T& result_b
) {
	result_a = static_cast<T>(0);
	result_b = static_cast<T>(0);
	for (size_t k = 0; k < size; ++k) {
		result_a += weights[k] * a[k];
		result_b += weights[k] * b[k];
	}
}

#ifndef VEC

template<std::floating_point T>
FORCE_INLINE void vec_mult_add(T* out, const T* other, T scalar, uint64_t size) {
	for (uint64_t k = 0; k < size; ++k)
		out[k] += other[k] * scalar;
}

template<std::floating_point T>
FORCE_INLINE T dot_product(const T* a, const T* b, size_t size) {
	T result = static_cast<T>(0.);
	for (size_t k = 0; k < size; ++k)
		result += a[k] * b[k];
	return result;
}

template<std::floating_point T>
FORCE_INLINE T dot_product_mult_add(
	const T* a, const T* b, T* out, T scalar, size_t size
) {
	T result = static_cast<T>(0.);
	for (size_t k = 0; k < size; ++k) {
		result += a[k] * b[k];
		out[k] += b[k] * scalar;
	}
	return result;
}

#endif
