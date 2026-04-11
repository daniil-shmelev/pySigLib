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
#include "cu_log_signature.h"

// =========================================================================
// CUDA device function: uncombine_sig_deriv_zero
//
// Backpropagation through the tensor product (Chen's identity).
// Ported from cp_tensor_poly.h uncombine_sig_deriv_zero.
//
// sig1:             original signature (read-only)
// sig2:             partial log (read-only)
// sig_concat_deriv: input/output derivatives (modified in-place)
// sig2_deriv:       output derivatives w.r.t. sig2 (written)
// degree:           truncation degree for the uncombine
// level_index:      precomputed level index
// reduction_buf:    shared memory buffer of size >= nthreads for tiled reduction
// =========================================================================

template<typename T>
__device__ void uncombine_sig_deriv_zero_device(
	const T* __restrict__ sig1,
	const T* __restrict__ sig2,
	T* __restrict__ sig_concat_deriv,
	T* __restrict__ sig2_deriv,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* __restrict__ level_index,
	T* __restrict__ reduction_buf
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// sig_len for degree-1
	const uint64_t sig2_len = level_index[degree];
	for (uint64_t i = tid; i < sig2_len; i += nthreads)
		sig2_deriv[i] = static_cast<T>(0);
	__syncthreads();

	// First loop: accumulate into sig2_deriv (process levels top-down)
	// sig2_deriv[right] += sig_concat_deriv[level][left*right_size + right] * sig1[left]
	for (int64_t level = static_cast<int64_t>(degree); level > 0; --level) {
		for (int64_t left_level = level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			const uint64_t level_start = level_index[level];
			const uint64_t left_start = level_index[left_level];
			const uint64_t left_size = level_index[left_level + 1] - left_start;
			const uint64_t right_start = level_index[right_level];
			const uint64_t right_size = level_index[right_level + 1] - right_start;

			if (right_size >= static_cast<uint64_t>(nthreads)) {
				// Standard: enough outputs to keep all threads busy
				for (uint64_t r = tid; r < right_size; r += nthreads) {
					T acc = static_cast<T>(0);
					for (uint64_t l = 0; l < left_size; ++l) {
						acc += sig_concat_deriv[level_start + l * right_size + r] * sig1[left_start + l];
					}
					sig2_deriv[right_start + r] += acc;
				}
			} else {
				// Tiled: assign multiple threads per output r, tile the l sum
				uint64_t tpr = 1;
				while (tpr * 2 <= static_cast<uint64_t>(nthreads) / right_size) tpr *= 2;

				uint64_t my_r = tid / tpr;
				uint64_t my_lane = tid % tpr;

				T acc = static_cast<T>(0);
				if (my_r < right_size) {
					for (uint64_t l = my_lane; l < left_size; l += tpr)
						acc += sig_concat_deriv[level_start + l * right_size + my_r] * sig1[left_start + l];
				}

				reduction_buf[tid] = acc;
				__syncthreads();

				for (uint64_t s = tpr / 2; s > 0; s >>= 1) {
					if (my_lane < s)
						reduction_buf[tid] += reduction_buf[tid + s];
					__syncthreads();
				}

				if (my_lane == 0 && my_r < right_size)
					sig2_deriv[right_start + my_r] += reduction_buf[tid];
			}
			__syncthreads();
		}
	}

	// Second loop: accumulate into sig_concat_deriv (process left_levels bottom-up)
	// Zero sig_concat_deriv[left_level], then:
	// sig_concat_deriv[left] += sig_concat_deriv[level][left*right_size + right] * sig2[right]
	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		const uint64_t left_start = level_index[left_level];
		const uint64_t left_size = level_index[left_level + 1] - left_start;

		// Zero this left level
		for (uint64_t i = tid; i < left_size; i += nthreads)
			sig_concat_deriv[left_start + i] = static_cast<T>(0);
		__syncthreads();

		for (uint64_t level = left_level + 1, right_level = 1;
			level <= degree;
			++level, ++right_level) {

			const uint64_t level_start = level_index[level];
			const uint64_t right_start = level_index[right_level];
			const uint64_t right_size = level_index[right_level + 1] - right_start;

			if (left_size >= static_cast<uint64_t>(nthreads)) {
				// Standard: enough outputs to keep all threads busy
				for (uint64_t l = tid; l < left_size; l += nthreads) {
					T acc = static_cast<T>(0);
					for (uint64_t r = 0; r < right_size; ++r) {
						acc += sig_concat_deriv[level_start + l * right_size + r] * sig2[right_start + r];
					}
					sig_concat_deriv[left_start + l] += acc;
				}
			} else {
				// Tiled: assign multiple threads per output l, tile the r sum
				uint64_t tpl = 1;
				while (tpl * 2 <= static_cast<uint64_t>(nthreads) / left_size) tpl *= 2;

				uint64_t my_l = tid / tpl;
				uint64_t my_lane = tid % tpl;

				T acc = static_cast<T>(0);
				if (my_l < left_size) {
					for (uint64_t r = my_lane; r < right_size; r += tpl)
						acc += sig_concat_deriv[level_start + my_l * right_size + r] * sig2[right_start + r];
				}

				reduction_buf[tid] = acc;
				__syncthreads();

				for (uint64_t s = tpl / 2; s > 0; s >>= 1) {
					if (my_lane < s)
						reduction_buf[tid] += reduction_buf[tid + s];
					__syncthreads();
				}

				if (my_lane == 0 && my_l < left_size)
					sig_concat_deriv[left_start + my_l] += reduction_buf[tid];
			}
			__syncthreads();
		}
	}
}

// =========================================================================
// CUDA device function: tensor_log_backprop
//
// Full backpropagation through the tensor log computation.
// Ported from cp_log_signature.h tensor_log_backprop_.
//
// out:           output buffer (sig_len), receives dF/d(sig)
// derivs:        scratch buffer (sig_len), initially holds dF/d(log_sig) in expanded form
// other_derivs:  scratch buffer (sig_len)
// sig:           original signature (read-only, sig_len)
// sig_copy:      scratch buffer (sig_len), will be used to recompute tensor_log with partials
// partial_logs:  scratch buffer ((degree-1) * buff1_size)
// buff1:         scratch buffer (buff1_size)
// buff2:         scratch buffer (sig_len)
// dimension, degree, level_index: as usual
// reduction_buf: shared memory buffer of size >= nthreads for tiled reduction
// =========================================================================

template<typename T>
__device__ void tensor_log_backprop_device(
	T* __restrict__ out,
	T* __restrict__ derivs,
	T* __restrict__ other_derivs,
	const T* __restrict__ sig,
	T* __restrict__ sig_copy,
	T* __restrict__ partial_logs,
	T* __restrict__ buff1,
	T* __restrict__ buff2,
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t buff1_size,
	const uint64_t* __restrict__ level_index,
	T* __restrict__ reduction_buf
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// Copy derivs to out
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		out[i] = derivs[i];
	// Forward forces log_sig[0] = 0 (constant), so d/d(sig[0]) = 0.
	if (tid == 0) out[0] = static_cast<T>(0);
	__syncthreads();

	if (degree <= 1) return;

	// Copy sig to sig_copy and compute tensor_log with partial_logs
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		sig_copy[i] = sig[i];
	__syncthreads();

	tensor_log_inplace_device<T>(
		sig_copy, buff1, buff2,
		degree, level_index,
		partial_logs, buff1_size
	);

	// Zero other_derivs
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		other_derivs[i] = static_cast<T>(0);
	__syncthreads();

	T factor = static_cast<T>(-1);
	for (uint64_t depth = 1; depth + 1 < degree; ++depth) {
		T scalar = static_cast<T>(1) / static_cast<T>(1 + depth);
		T* partial = partial_logs + (degree - 2 - depth) * buff1_size;

		uncombine_sig_deriv_zero_device<T>(
			sig, partial, derivs, other_derivs,
			dimension, degree + 1 - depth, level_index,
			reduction_buf
		);

		// out[i] += factor * (derivs[i] + scalar * other_derivs[i])
		for (uint64_t lev = 1; lev <= degree - depth; ++lev) {
			const uint64_t lev_start = level_index[lev];
			const uint64_t lev_size = level_index[lev + 1] - lev_start;

			for (uint64_t i = tid; i < lev_size; i += nthreads) {
				out[lev_start + i] += factor * (derivs[lev_start + i] + scalar * other_derivs[lev_start + i]);
			}
		}
		__syncthreads();

		// Swap derivs and other_derivs pointers
		T* tmp = other_derivs;
		other_derivs = derivs;
		derivs = tmp;

		factor = -factor;
	}

	// Backprop level 2: the final "uncombine" step
	T scalar_final = factor / static_cast<T>(degree);
	const uint64_t lev1_start = level_index[1];
	const uint64_t lev2_start = level_index[2];

	// out[lev1 + k] += scalar * (sum_j derivs[lev2 + k*dim + j] * sig[lev1 + j]
	//                           + sum_j derivs[lev2 + j*dim + k] * sig[lev1 + j])
	for (uint64_t k = tid; k < dimension; k += nthreads) {
		T acc = static_cast<T>(0);
		for (uint64_t j = 0; j < dimension; ++j) {
			T sig_j = sig[lev1_start + j];
			acc += derivs[lev2_start + k * dimension + j] * sig_j;  // row k, col j
			acc += derivs[lev2_start + j * dimension + k] * sig_j;  // row j, col k
		}
		out[lev1_start + k] += scalar_final * acc;
	}
	__syncthreads();
}
