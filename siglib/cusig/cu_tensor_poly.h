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
		if (exp & 1) result *= base;
		base *= base;
		exp >>= 1;
	}
	return result;
}

inline uint64_t host_sig_length(uint64_t dimension, uint64_t degree) {
	if (dimension == 0) return 1;
	if (dimension == 1) return degree + 1;
	return (host_power(dimension, degree + 1) - 1) / (dimension - 1);
}

inline void host_populate_level_index(uint64_t* level_index, uint64_t dimension, uint64_t count) {
	level_index[0] = 0;
	for (uint64_t i = 1; i < count; ++i)
		level_index[i] = level_index[i - 1] * dimension + 1;
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

	// Process from highest level down to 1
	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {

		const uint64_t target_start = level_index[target_level];
		const uint64_t target_size = level_index[target_level + 1] - level_index[target_level];

		// Accumulate contributions from left_level x right_level where left_level + right_level = target_level
		for (int64_t left_level = target_level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			const uint64_t left_start = level_index[left_level];
			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
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
