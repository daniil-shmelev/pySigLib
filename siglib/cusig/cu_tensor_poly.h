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
#include "cupch.h"

// =========================================================================
// Shared host-side helpers for signature length and level index computation
// =========================================================================

inline uint64_t host_power(uint64_t base, uint64_t exp) {
	uint64_t result = 1;
	while (exp > 0) {
		if (exp & 1) {
			// result * base overflows iff base > UINT64_MAX / result
			if (result != 0 && base > UINT64_MAX / result)
				return 0;
			result *= base;
		}
		exp >>= 1;
		if (exp > 0) {
			// base * base overflows iff base > UINT64_MAX / base
			if (base != 0 && base > UINT64_MAX / base)
				return 0;
			base *= base;
		}
	}
	return result;
}

inline uint64_t host_sig_length(uint64_t dimension, uint64_t degree) {
	if (dimension == 0) return 1;
	if (dimension == 1) {
		if (degree == UINT64_MAX)
			throw std::overflow_error("host_sig_length: degree overflow");
		return degree + 1;
	}
	const auto pwr = host_power(dimension, degree + 1);
	if (!pwr)
		throw std::overflow_error("host_sig_length: sig length overflow");
	return (pwr - 1) / (dimension - 1);
}

inline void host_populate_level_index(uint64_t* level_index, uint64_t dimension, uint64_t count) {
	level_index[0] = 0;
	for (uint64_t i = 1; i < count; ++i) {
		if (dimension != 0 && level_index[i - 1] > UINT64_MAX / dimension)
			throw std::overflow_error("host_populate_level_index: level_index overflow");
		const uint64_t mul = level_index[i - 1] * dimension;
		if (mul > UINT64_MAX - 1)
			throw std::overflow_error("host_populate_level_index: level_index overflow");
		level_index[i] = mul + 1;
	}
}

// =========================================================================
// Shared helper: choose threads per block from largest level size
// =========================================================================

inline unsigned int host_choose_threads_per_block(uint64_t max_level_size) {
	unsigned int tpb = 32;
	if (max_level_size > 32)   tpb = 64;
	if (max_level_size > 64)   tpb = 128;
	if (max_level_size > 128)  tpb = 256;
	if (max_level_size > 512)  tpb = 512;
	if (max_level_size > 1024) tpb = 1024;
	return tpb;
}

// =========================================================================
// CUDA device function: sig_combine_inplace
//
// Computes sig1 = sig1 (x) sig2  (tensor product / Chen's identity)
// where sig2 is another truncated signature.
//
// Used by both the signature forward pass (cu_signature.cu) and the
// standalone sig_combine kernel (cu_tensor_poly.cu).
// =========================================================================

template<typename T>
__device__ void sig_combine_inplace_device(
	T* __restrict__ sig1,
	const T* __restrict__ sig2,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// A valid signature has level-0 = 1; the loop below only touches 1..N.
	// Only thread 0 writes; no subsequent thread reads sig1[0], so no sync.
	if (tid == 0) sig1[0] = static_cast<T>(1);

	// Process from highest level down to 1
	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {

		const uint64_t target_start = level_index[target_level];
		const uint64_t target_size = level_index[target_level + 1] - level_index[target_level];

		// Accumulate contributions from left_level x right_level where left_level + right_level = target_level
		for (int64_t left_level = target_level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			const uint64_t left_start = level_index[left_level];
			const uint64_t right_start = level_index[right_level];
			const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];

			for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
				const uint64_t l_idx = idx / right_size;
				const uint64_t r_idx = idx % right_size;
				sig1[target_start + idx] += sig1[left_start + l_idx] * sig2[right_start + r_idx];
			}
			__syncthreads();
		}

		// left_level = 0: just add sig2 at this level
		for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
			sig1[target_start + idx] += sig2[level_index[target_level] + idx];
		}
		__syncthreads();
	}
}

// =========================================================================
// linear_signature_device: signature of a single linear segment from
// a displacement vector (dimension elements).
// =========================================================================

template<typename T>
__device__ void linear_signature_device(
	const T* __restrict__ increments,
	T* __restrict__ out,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	if (tid == 0) out[0] = static_cast<T>(1);
	for (uint64_t i = tid; i < dimension; i += nthreads)
		out[i + 1] = increments[i];
	__syncthreads();

	const unsigned int dim32 = static_cast<unsigned int>(dimension);
	const unsigned int stride_l = static_cast<unsigned int>(nthreads) / dimension;
	const unsigned int stride_r = static_cast<unsigned int>(nthreads) % dimension;
	const unsigned int base_l = static_cast<unsigned int>(tid) / dimension;
	const unsigned int base_r = static_cast<unsigned int>(tid) % dimension;

	for (uint64_t level = 2; level <= degree; ++level) {
		const T one_over_level = static_cast<T>(1) / static_cast<T>(level);
		const uint64_t prev_start = level_index[level - 1];
		const uint64_t prev_size = level_index[level] - prev_start;
		const uint64_t cur_start = level_index[level];
		const unsigned int cur_size = static_cast<unsigned int>(prev_size * dimension);

		unsigned int cur_l = base_l;
		unsigned int cur_r = base_r;
		for (unsigned int idx = static_cast<unsigned int>(tid); idx < cur_size; idx += static_cast<unsigned int>(nthreads)) {
			out[cur_start + idx] = out[prev_start + cur_l] * increments[cur_r] * one_over_level;
			cur_l += stride_l;
			cur_r += stride_r;
			if (cur_r >= dim32) { cur_r -= dim32; cur_l++; }
		}
		__syncthreads();
	}
}
