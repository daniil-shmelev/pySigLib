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
// CUDA device function: tensor_log_inplace
//
// Computes the tensor logarithm of a truncated signature in-place.
// The result is the expanded log signature (method=0).
//
// sig:          pointer to signature data (will be modified in-place)
// buff1, buff2: scratch buffers of size sig_length(dim, degree-1) and
//               sig_length(dim, degree) respectively
// degree:       truncation degree
// level_index:  precomputed level index array (degree+2 entries)
// =========================================================================

template<typename T>
__device__ void tensor_log_inplace_device(
	T* __restrict__ sig,
	T* __restrict__ buff1,
	T* __restrict__ buff2,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// sig[0] = 0
	if (tid == 0) sig[0] = static_cast<T>(0);
	__syncthreads();

	if (degree <= 1) return;

	// Zero buff1 and buff2
	const uint64_t buff1_size = level_index[degree];   // sig_length(dim, degree-1)
	const uint64_t buff2_size = level_index[degree + 1]; // sig_length(dim, degree)
	for (uint64_t i = tid; i < buff1_size; i += nthreads)
		buff1[i] = static_cast<T>(0);
	for (uint64_t i = tid; i < buff2_size; i += nthreads)
		buff2[i] = static_cast<T>(0);
	__syncthreads();

	for (int64_t k = static_cast<int64_t>(degree); k > 0; --k) {
		const T constant = static_cast<T>(1) / static_cast<T>(k);

		// Compute buff2 tensor products: for target_level 2..1+degree-k
		const uint64_t max_target = 1 + degree - static_cast<uint64_t>(k);
		for (uint64_t target_level = 2; target_level <= max_target; ++target_level) {

			const uint64_t target_start = level_index[target_level];
			const uint64_t target_size = level_index[target_level + 1] - target_start;

			// Zero buff2 at this level
			for (uint64_t i = tid; i < target_size; i += nthreads)
				buff2[target_start + i] = static_cast<T>(0);
			__syncthreads();

			// Accumulate tensor products: sig[left] * buff1[right] where left+right=target
			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				const uint64_t right_level = target_level - left_level;

				const uint64_t left_start = level_index[left_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
					const uint64_t l_idx = idx / right_size;
					const uint64_t r_idx = idx % right_size;
					buff2[target_start + idx] += sig[left_start + l_idx] * buff1[right_start + r_idx];
				}
				__syncthreads();
			}
		}

		if (k > 1) {
			// Update buff1: buff1[level] = constant * sig[level] - buff2[level]
			// for levels 1..max_target
			for (uint64_t target_level = 1; target_level <= max_target; ++target_level) {
				const uint64_t target_start = level_index[target_level];
				const uint64_t target_size = level_index[target_level + 1] - target_start;

				for (uint64_t i = tid; i < target_size; i += nthreads) {
					buff1[target_start + i] = constant * sig[target_start + i] - buff2[target_start + i];
				}
			}
			__syncthreads();
		}
	}

	// Final step: sig[level] -= buff2[level] for levels 2..degree
	for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
		const uint64_t target_start = level_index[target_level];
		const uint64_t target_size = level_index[target_level + 1] - target_start;

		for (uint64_t i = tid; i < target_size; i += nthreads) {
			sig[target_start + i] -= buff2[target_start + i];
		}
	}
	__syncthreads();
}
