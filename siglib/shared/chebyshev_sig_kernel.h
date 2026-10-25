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
#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#define SIG_CHEB_INLINE __host__ __device__ __forceinline__
#else
#define SIG_CHEB_INLINE inline
#endif

namespace chebyshev_sig_kernel {

template<typename T>
struct Corner { T hi; T lo; };

template<typename T>
struct CellAdjoints { T coefficient; T start; T bottom_end; T left_end; };

template<typename T>
SIG_CHEB_INLINE T rounded_add(T a, T b) {
#ifdef __CUDA_ARCH__
	if constexpr (std::is_same_v<T, float>) return __fadd_rn(a, b);
	else return __dadd_rn(a, b);
#else
	volatile T result = a + b;
	return result;
#endif
}

template<typename T>
SIG_CHEB_INLINE Corner<T> two_sum(T a, T b) {
	const T value = rounded_add(a, b);
#ifdef __CUDA_ARCH__
	if constexpr (std::is_same_v<T, double>) {
		const bool ordered = fabs(a) >= fabs(b);
		const T large = ordered ? a : b;
		const T small = ordered ? b : a;
		return { value, rounded_add(rounded_add(large, -value), small) };
	}
#endif
	const T tail = rounded_add(value, -a);
	const T error = rounded_add(rounded_add(a, -rounded_add(value, -tail)), rounded_add(b, -tail));
	return { value, error };
}

template<typename T>
SIG_CHEB_INLINE Corner<T> corner_add(Corner<T> a, Corner<T> b) {
	const auto sum = two_sum(a.hi, b.hi);
	return two_sum(sum.hi, rounded_add(rounded_add(sum.lo, a.lo), b.lo));
}

template<typename T>
SIG_CHEB_INLINE Corner<T> corner_accumulate(Corner<T> a, Corner<T> b) {
	// Sum2-style accumulation (Ogita, Rump and Oishi, 2005).
	const auto sum = two_sum(a.hi, b.hi);
	return { sum.hi, rounded_add(a.lo, rounded_add(sum.lo, b.lo)) };
}

template<typename T>
SIG_CHEB_INLINE Corner<T> corner_product(Corner<T> a, T factor) {
#ifdef __CUDA_ARCH__
	T product;
	T error;
	if constexpr (std::is_same_v<T, float>) {
		product = __fmul_rn(a.hi, factor);
		error = __fmaf_rn(a.hi, factor, -product);
	}
	else {
		product = __dmul_rn(a.hi, factor);
		error = __fma_rn(a.hi, factor, -product);
	}
#else
	volatile T rounded_product = a.hi * factor;
	const T product = rounded_product;
	const T error = std::fma(a.hi, factor, -product);
#endif
	return { product, rounded_add(error, a.lo * factor) };
}

template<typename T>
SIG_CHEB_INLINE Corner<T> corner_scale(Corner<T> a, T factor) {
	const auto product = corner_product(a, factor);
	return two_sum(product.hi, product.lo);
}

template<typename T>
SIG_CHEB_INLINE T corner_value(Corner<T> a) { return a.hi + a.lo; }

template<typename T>
SIG_CHEB_INLINE T corner_add_value(Corner<T> a, T b) {
	const auto sum = two_sum(a.hi, b);
	return rounded_add(sum.hi, rounded_add(sum.lo, a.lo));
}

// The input has length coefficients; the output has length + 1.
template<typename T>
SIG_CHEB_INLINE void integrate(T* a, uint64_t length, T scale) {
	const T a0 = a[0];
	const T a2 = length > 2 ? a[2] : T(0);
	T upper = T(0);
	T constant = T(0);
	for (uint64_t k = length; k > 1; --k) {
		const T old = k < length ? a[k] : T(0);
		const T value = (a[k - 1] - upper) * (scale * (T(1) / T(4 * k)));
		a[k] = value;
		constant += k % 2 ? value : -value;
		upper = old;
	}
	a[1] = (T(0.5) * a0 - T(0.25) * a2) * scale;
	a[0] = constant + a[1];
}

template<typename T>
SIG_CHEB_INLINE void multiply_coordinate(T* a, uint64_t length) {
	T upper = T(0);
	for (uint64_t k = length; k > 0; --k) {
		const T old = k < length ? a[k] : T(0);
		a[k] = T(0.5) * old + (k == 1 ? T(0.5) : T(0.25)) * a[k - 1] + T(0.25) * upper;
		upper = old;
	}
	a[0] = T(0.5) * a[0] + T(0.25) * upper;
}

template<typename T>
SIG_CHEB_INLINE T coefficient_sum(const T* a, uint64_t length) {
	T result = T(0);
	for (uint64_t k = 0; k < length; ++k) result += a[k];
	return result;
}

SIG_CHEB_INLINE uint64_t folded_index(uint64_t k, uint64_t p) {
	const uint64_t remainder = k % (2 * p);
	return remainder < p ? remainder : 2 * p - remainder;
}

template<typename T>
SIG_CHEB_INLINE void finish_edge(T* out, Corner<T> start, Corner<T> end, uint64_t p) {
	T odd = T(0);
	for (uint64_t k = 2; k <= p; ++k) {
		if (k % 2) odd += out[k];
	}
	// The constant coefficient is implicit in the endpoint constraints.
	out[0] = T(0);
	const auto difference = two_sum(end.hi, -start.hi);
	const T correction = T(0.5) * ((difference.lo + end.lo) - start.lo);
	if (p <= 2)
		out[1] = T(0.5) * difference.hi + correction;
	else {
		const auto leading = two_sum(T(0.5) * difference.hi, -odd);
		out[1] = leading.hi + (leading.lo + correction);
	}
}

template<typename T>
SIG_CHEB_INLINE void fold(const T* raw, T* out, Corner<T> start, Corner<T> end, uint64_t p, uint64_t terms) {
	for (uint64_t k = 2; k <= p; ++k) out[k] = raw[k];
	for (uint64_t k = p + 1; k <= p + terms; ++k) {
		const uint64_t target = folded_index(k, p);
		if (target >= 2) out[target] += raw[k];
	}
	finish_edge(out, start, end, p);
}

template<typename T>
SIG_CHEB_INLINE Corner<T> finish_cell_increment(
	T integral, const T* bottom, const T* left, Corner<T> left_end,
	T* top, T* right, uint64_t p
) {
	// Compatible edges have increment twice the sum of their odd modes.
	T bottom_odd = T(0), left_odd = T(0);
	T top_odd = T(0), right_odd = T(0);
	for (uint64_t k = 1; k <= p; k += 2) {
		bottom_odd += bottom[k]; left_odd += left[k];
	}
	for (uint64_t k = 2; k <= p; ++k) {
		if (k % 2) { top_odd += top[k]; right_odd += right[k]; }
	}
	const T top_delta = bottom_odd + T(0.5) * integral;
	const T right_delta = left_odd + T(0.5) * integral;
	const auto corner = two_sum(left_end.hi, T(2) * top_delta + left_end.lo);
	top[1] = top_delta - top_odd;
	right[1] = right_delta - right_odd;
	top[0] = right[0] = T(0);
	return corner;
}

template<typename T, int P = -1, int K = -1, bool CompatibleEdges = false>
SIG_CHEB_INLINE Corner<T> cell_forward(
	T c, const T* bottom, const T* left, Corner<T> start,
	Corner<T> bottom_end, Corner<T> left_end, T* top, T* right,
	uint64_t degree, uint64_t terms
) {
	const uint64_t p = P < 0 ? degree : static_cast<uint64_t>(P);
	const uint64_t count = K < 0 ? terms : static_cast<uint64_t>(K);
	if constexpr (P == 1 && K >= 1 && K <= 3) {
		const T q2 = K >= 2 ? c * (c * T(0.25)) : T(0);
		const T q3 = K == 3 ? q2 * (c * T(1.0 / 9.0)) : T(0);
		const T integral = (c + q2 + q3) * corner_value(start)
			+ (c + T(2.0 / 3.0) * q2 + T(0.5) * q3) * (bottom[1] + left[1]);
		if constexpr (CompatibleEdges)
			return finish_cell_increment(integral, bottom, left, left_end, top, right, p);
		auto corner = corner_accumulate(left_end, { -start.hi, -start.lo });
		corner = corner_accumulate(corner, bottom_end);
		corner = corner_accumulate(corner, { integral, T(0) });
		corner = two_sum(corner.hi, corner.lo);
		finish_edge(top, left_end, corner, p);
		finish_edge(right, bottom_end, corner, p);
		return corner;
	}
	if constexpr ((P == 2 && (K == 2 || K == 3)) || (P == 3 && K == 3)) {
		const T f1 = bottom[1], f2 = bottom[2], g1 = left[1], g2 = left[2];
		const T f3 = P == 3 ? bottom[3] : T(0), g3 = P == 3 ? left[3] : T(0);
		const T q2 = c * (c * T(0.25));
		const T a1 = c * (f1 - T(4.0 / 3.0) * f2 + f3);
		const T b1 = c * (g1 - T(4.0 / 3.0) * g2 + g3);
		const T a2 = q2 * (T(2.0 / 3.0) * f1 - T(4.0 / 3.0) * f2 + T(6.0 / 5.0) * f3);
		const T b2 = q2 * (T(2.0 / 3.0) * g1 - T(4.0 / 3.0) * g2 + T(6.0 / 5.0) * g3);
		const T weighted1 = corner_value(start) * c;
		const T weighted2 = corner_value(start) * q2;
		T integral = (weighted1 + (a1 + b1)) + (weighted2 + (a2 + b2));
		const T cross_top2 = weighted2 + b2;
		const T cross_right2 = weighted2 + a2;
		if constexpr (P == 2) {
			top[2] = f2 + c * (T(0.125) * f1) + q2 * (T(0.125) * f1 - T(5.0 / 24.0) * f2) + T(0.125) * cross_top2;
			right[2] = g2 + c * (T(0.125) * g1) + q2 * (T(0.125) * g1 - T(5.0 / 24.0) * g2) + T(0.125) * cross_right2;
		}
		else {
			top[2] = f2 + c * (T(0.125) * f1 - T(0.0625) * f3) + q2 * (T(0.125) * f1 - T(19.0 / 96.0) * f2 + T(0.125) * f3) + T(0.125) * cross_top2;
			right[2] = g2 + c * (T(0.125) * g1 - T(0.0625) * g3) + q2 * (T(0.125) * g1 - T(19.0 / 96.0) * g2 + T(0.125) * g3) + T(0.125) * cross_right2;
		}
		if constexpr (K == 3) {
			const T q3 = q2 * (c * T(1.0 / 9.0));
			const T a3 = q3 * (T(0.5) * f1 - T(6.0 / 5.0) * f2 + T(13.0 / 10.0) * f3);
			const T b3 = q3 * (T(0.5) * g1 - T(6.0 / 5.0) * g2 + T(13.0 / 10.0) * g3);
			const T weighted3 = corner_value(start) * q3;
			integral += weighted3 + (a3 + b3);
			const T cross_top3 = weighted3 + b3;
			const T cross_right3 = weighted3 + a3;
			const T f1_weight = P == 2 ? T(7.0 / 64.0) : T(29.0 / 256.0);
			top[2] += q3 * (f1_weight * f1 - T(0.25) * f2 + T(153.0 / 640.0) * f3) + T(0.1875) * cross_top3;
			right[2] += q3 * (f1_weight * g1 - T(0.25) * g2 + T(153.0 / 640.0) * g3) + T(0.1875) * cross_right3;
			if constexpr (P == 3) {
				top[3] = f3 + c * (T(1.0 / 12.0) * f2) + q2 * (T(1.0 / 48.0) * f1 - T(0.03125) * f3) + q3 * (T(0.03125) * f1 - T(7.0 / 128.0) * f2 + T(0.03125) * f3) + T(0.03125) * cross_top3;
				right[3] = g3 + c * (T(1.0 / 12.0) * g2) + q2 * (T(1.0 / 48.0) * g1 - T(0.03125) * g3) + q3 * (T(0.03125) * g1 - T(7.0 / 128.0) * g2 + T(0.03125) * g3) + T(0.03125) * cross_right3;
			}
		}
		if constexpr (CompatibleEdges)
			return finish_cell_increment(integral, bottom, left, left_end, top, right, p);
		auto corner = corner_accumulate(left_end, { -start.hi, -start.lo });
		corner = corner_accumulate(corner, bottom_end);
		corner = corner_accumulate(corner, { integral, T(0) });
		corner = two_sum(corner.hi, corner.lo);
		finish_edge(top, left_end, corner, p);
		finish_edge(right, bottom_end, corner, p);
		return corner;
	}
	constexpr uint64_t capacity = P < 0 || K < 0 ? 32 : static_cast<uint64_t>(P + K + 1);
	T a[capacity] = {}, b[capacity] = {}, power[capacity] = {};
	T raw_top[capacity] = {}, raw_right[capacity] = {};
	for (uint64_t k = 1; k <= p; ++k) {
		a[k] = bottom[k]; b[k] = left[k];
		a[0] += k % 2 ? a[k] : -a[k];
		b[0] += k % 2 ? b[k] : -b[k];
	}
	for (uint64_t k = 0; k <= p; ++k) { raw_top[k] = a[k]; raw_right[k] = b[k]; }
	raw_top[0] += left_end.hi;
	raw_right[0] += bottom_end.hi;
	// Preserve the compensated endpoints in the zero-order transfer.
	auto corner = corner_accumulate(left_end, { -start.hi, -start.lo });
	corner = corner_accumulate(corner, bottom_end);
	power[0] = T(1);
	T q = T(1);
#if defined(__CUDA_ARCH__)
#pragma unroll (K < 1 ? 1 : K)
#endif
	for (uint64_t r = 1; r <= count; ++r) {
		integrate(a, p + r, c * (T(1) / T(r)));
		integrate(b, p + r, c * (T(1) / T(r)));
		multiply_coordinate(power, r);
		q *= c * (T(1) / T(r * r));
		const T a_end = coefficient_sum(a, p + r + 1);
		const T b_end = coefficient_sum(b, p + r + 1);
		const auto weighted = corner_product(start, q);
		corner = corner_accumulate(corner, two_sum(a_end, b_end));
		corner = corner_accumulate(corner, weighted);
		const T cross_top = corner_add_value(weighted, b_end);
		const T cross_right = corner_add_value(weighted, a_end);
		for (uint64_t k = 0; k <= p + r; ++k) { raw_top[k] += a[k]; raw_right[k] += b[k]; }
		for (uint64_t k = 0; k <= r; ++k) { raw_top[k] += cross_top * power[k]; raw_right[k] += cross_right * power[k]; }
	}
	corner = two_sum(corner.hi, corner.lo);
	fold(raw_top, top, left_end, corner, p, count);
	fold(raw_right, right, bottom_end, corner, p, count);
	return corner;
}

template<typename T>
SIG_CHEB_INLINE void integrate_transpose(T* a, uint64_t length, T scale) {
	const T a0 = a[0], a1 = a[1];
	a[0] = T(0.5) * (a0 + a1) * scale;
	if (length > 2) a[1] = (a[2] - a0) * (scale * T(0.125));
	T lower = a1;
	for (uint64_t k = 2; k + 1 < length; ++k) {
		const T old = a[k];
		const T constant = (k % 2 ? T(1) : T(-1)) * (T(1) / T(2 * (k * k - 1)));
		a[k] = scale * (a[k + 1] * (T(1) / T(4 * (k + 1))) - lower * (T(1) / T(4 * (k - 1))) + constant * a0);
		lower = old;
	}
}

// Reverse the real-arithmetic cell map; compensation controls forward rounding.
template<typename T, int P = -1, int K = -1>
SIG_CHEB_INLINE CellAdjoints<T> cell_backward(
	T c, const T* bottom, const T* left, Corner<T> start,
	const T* top_adj, const T* right_adj, T corner_adj,
	T* bottom_adj, T* left_adj, uint64_t degree, uint64_t terms
) {
	const uint64_t p = P < 0 ? degree : static_cast<uint64_t>(P);
	const uint64_t count = K < 0 ? terms : static_cast<uint64_t>(K);
	constexpr uint64_t capacity = P < 0 || K < 0 ? 32 : static_cast<uint64_t>(P + K + 1);
	T top_raw[capacity] = {}, right_raw[capacity] = {};
	T a[capacity] = {}, b[capacity] = {}, power[capacity] = {};
	T top_dot[capacity] = {}, right_dot[capacity] = {};
	CellAdjoints<T> result{ T(0), T(0), -T(0.5) * right_adj[1], -T(0.5) * top_adj[1] };
	corner_adj += T(0.5) * (top_adj[1] + right_adj[1]);
	result.bottom_end += corner_adj;
	result.left_end += corner_adj;
	for (uint64_t k = 0; k <= p + count; ++k) {
		const uint64_t target = folded_index(k, p);
		if (target > 1) {
			top_raw[k] = top_adj[target] - (target % 2 ? top_adj[1] : T(0));
			right_raw[k] = right_adj[target] - (target % 2 ? right_adj[1] : T(0));
		}
	}
	for (uint64_t k = 1; k <= p; ++k) {
		a[k] = bottom[k]; b[k] = left[k];
		a[0] += k % 2 ? a[k] : -a[k];
		b[0] += k % 2 ? b[k] : -b[k];
	}
	power[0] = T(1);
	T q = T(1), divided_q = T(1);
	result.start = -corner_adj;
#if defined(__CUDA_ARCH__)
#pragma unroll (K < 1 ? 1 : K)
#endif
	for (uint64_t r = 1; r <= count; ++r) {
		const T scale = r == 1 ? T(1) : c * (T(1) / T(r));
		integrate(a, p + r, scale);
		integrate(b, p + r, scale);
		multiply_coordinate(power, r);
		q *= c * (T(1) / T(r * r));
		if (r > 1) divided_q *= c * (T(1) / T(r * r));
		for (uint64_t k = 0; k <= r; ++k) {
			top_dot[r] += top_raw[k] * power[k];
			right_dot[r] += right_raw[k] * power[k];
		}
		const T common = top_dot[r] + right_dot[r] + corner_adj;
		T derivative = corner_value(start) * divided_q * common;
		for (uint64_t k = 0; k <= p + r; ++k) {
			derivative += a[k] * (top_raw[k] + right_dot[r] + corner_adj);
			derivative += b[k] * (right_raw[k] + top_dot[r] + corner_adj);
		}
		result.coefficient += T(r) * derivative;
		result.start += q * common;
	}
	for (uint64_t k = 0; k <= p + count; ++k) { a[k] = T(0); b[k] = T(0); }
#if defined(__CUDA_ARCH__)
#pragma unroll (K < 1 ? 1 : K)
#endif
	for (uint64_t r = count; r > 0; --r) {
		for (uint64_t k = 0; k <= p + r; ++k) {
			a[k] += top_raw[k] + right_dot[r] + corner_adj;
			b[k] += right_raw[k] + top_dot[r] + corner_adj;
		}
		integrate_transpose(a, p + r + 1, c * (T(1) / T(r)));
		integrate_transpose(b, p + r + 1, c * (T(1) / T(r)));
	}
	for (uint64_t k = 0; k <= p; ++k) { a[k] += top_raw[k]; b[k] += right_raw[k]; }
	bottom_adj[0] = T(0); left_adj[0] = T(0);
	for (uint64_t k = 1; k <= p; ++k) {
		bottom_adj[k] = a[k] + (k % 2 ? a[0] : -a[0]);
		left_adj[k] = b[k] + (k % 2 ? b[0] : -b[0]);
	}
	return result;
}

}  // namespace chebyshev_sig_kernel

#undef SIG_CHEB_INLINE
