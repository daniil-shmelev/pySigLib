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

#include "cupch.h"
#include "cusig.h"
#include "cu_tensor_poly.h"

// =========================================================================
// CUDA sig_combine kernel
//
// Computes out = sig1 (x) sig2  (Chen's identity / tensor product)
// for a batch of signature pairs.  Each block handles one batch element.
// =========================================================================

template<typename T>
__global__ void sig_combine_kernel(
	const T* __restrict__ sig1,          // [batch_size * sig_len]
	const T* __restrict__ sig2,          // [batch_size * sig_len]
	T* __restrict__ out,                 // [batch_size * sig_len]
	uint64_t dimension,
	uint64_t degree
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// Compute level_index in shared memory (no device malloc needed)
	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);

	if (tid == 0) {
		level_index[0] = 0;
		for (uint64_t i = 1; i < degree + 2; ++i)
			level_index[i] = level_index[i - 1] * dimension + 1;
	}
	__syncthreads();

	const uint64_t sig_len = level_index[degree + 1];
	const T* my_sig1 = sig1 + batch_idx * sig_len;
	const T* my_sig2 = sig2 + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;

	// Level 0: scalar component = 1
	if (tid == 0)
		my_out[0] = static_cast<T>(1);

	// Each level reads only from the original sig1 and sig2,
	// so all levels are independent — no syncs needed between them.
	for (uint64_t target_level = 1; target_level <= degree; ++target_level) {
		const uint64_t target_start = level_index[target_level];
		const uint64_t target_size = level_index[target_level + 1] - target_start;

		for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
			// S1^(k) + S2^(k)
			T val = my_sig1[target_start + idx] + my_sig2[target_start + idx];

			// + sum_{i=1}^{k-1} S1^(i) tensor S2^(k-i)
			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				const uint64_t l_idx = idx / right_size;
				const uint64_t r_idx = idx % right_size;
				val += my_sig1[left_start + l_idx] * my_sig2[right_start + r_idx];
			}

			my_out[target_start + idx] = val;
		}
	}
}

// =========================================================================
// Host-side core launch
// =========================================================================

template<typename T>
void sig_combine_cuda_core_(
	const T* sig1,
	const T* sig2,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);

	if (degree == 0) {
		// Signature of degree 0 is just the scalar 1; combine is trivially 1
		auto ones = std::make_unique<T[]>(batch_size);
		std::fill(ones.get(), ones.get() + batch_size, static_cast<T>(1));
		cudaMemcpy(out, ones.get(), batch_size * sizeof(T), cudaMemcpyHostToDevice);
		return;
	}

	// Choose threads per block based on largest level size
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	// Shared memory: level_index (computed inside kernel, no device malloc needed)
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_combine_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig1, sig2, out, dimension, degree
	);

	check_cuda_kernel_launch();
}

template<typename T>
void sig_combine_cuda_(
	const T* sig1,
	const T* sig2,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("sig_combine_cuda received dimension 0");
	sig_combine_cuda_core_<T>(sig1, sig2, out, batch_size, dimension, degree);
}

// =========================================================================
// CUDA sig_combine_backprop kernel
//
// Given d_out = dF/d(sig_combine(sig1, sig2)), computes:
//   sig1_deriv = dF/d(sig1)
//   sig2_deriv = dF/d(sig2)
//
// 2D grid: gridDim.y = batch_size, gridDim.x covers sig_len.
// Each thread processes exactly one (batch, element) pair.
// =========================================================================

template<typename T>
__global__ void __launch_bounds__(256)
sig_combine_backprop_kernel(
	const T* __restrict__ d_out,         // [batch_size * sig_len]
	T* __restrict__ sig1_deriv,          // [batch_size * sig_len]
	T* __restrict__ sig2_deriv,          // [batch_size * sig_len]
	const T* __restrict__ sig1,          // [batch_size * sig_len]
	const T* __restrict__ sig2,          // [batch_size * sig_len]
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len
) {
	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);
	if (threadIdx.x == 0) {
		level_index[0] = 0;
		for (uint64_t i = 1; i < degree + 2; ++i)
			level_index[i] = level_index[i - 1] * dimension + 1;
	}
	__syncthreads();

	const uint64_t elem = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (elem >= sig_len) return;

	const uint64_t offset = (uint64_t)blockIdx.y * sig_len;
	const T* my_d_out = d_out + offset;
	const T* my_sig2 = sig2 + offset;
	const T* my_sig1 = sig1 + offset;

	if (elem == 0) {
		sig1_deriv[offset] = static_cast<T>(0);
		sig2_deriv[offset] = static_cast<T>(0);
		return;
	}

	// Find level
	uint64_t k = 1;
	while (elem >= level_index[k + 1]) ++k;

	const T d_out_val = my_d_out[elem];

	// Top level: no inner loop work, just copy
	if (k == degree) {
		sig1_deriv[offset + elem] = d_out_val;
		sig2_deriv[offset + elem] = d_out_val;
		return;
	}

	const uint64_t k_start = level_index[k];
	const uint64_t k_size = level_index[k + 1] - k_start;
	const uint64_t idx = elem - k_start;

	// sig1_deriv
	T val1 = d_out_val;
	for (uint64_t rl = 1; rl <= degree - k; ++rl) {
		const uint64_t comb_start = level_index[k + rl];
		const uint64_t r_start = level_index[rl];
		const uint64_t rs = level_index[rl + 1] - r_start;
		const T* d_out_row = my_d_out + comb_start + idx * rs;
		for (uint64_t r = 0; r < rs; ++r) {
			val1 += d_out_row[r] * my_sig2[r_start + r];
		}
	}
	sig1_deriv[offset + elem] = val1;

	// sig2_deriv
	T val2 = d_out_val;
	for (uint64_t ll = 1; ll <= degree - k; ++ll) {
		const uint64_t comb_start = level_index[ll + k];
		const uint64_t l_start = level_index[ll];
		const uint64_t ls = level_index[ll + 1] - l_start;
		const T* d_out_col = my_d_out + comb_start + idx;
		for (uint64_t l = 0; l < ls; ++l) {
			val2 += d_out_col[l * k_size] * my_sig1[l_start + l];
		}
	}
	sig2_deriv[offset + elem] = val2;
}

// =========================================================================
// Host-side core launch for backprop
// =========================================================================

template<typename T>
void sig_combine_backprop_cuda_core_(
	const T* sig_combined_deriv,
	T* sig1_deriv,
	T* sig2_deriv,
	const T* sig1,
	const T* sig2,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	if (degree == 0) {
		cudaMemcpy(sig1_deriv, sig_combined_deriv, batch_size * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaMemcpy(sig2_deriv, sig_combined_deriv, batch_size * sizeof(T), cudaMemcpyDeviceToDevice);
		return;
	}

	uint64_t sig_len = host_sig_length(dimension, degree);

	unsigned int threads_per_block = 256;
	unsigned int grid_x = static_cast<unsigned int>((sig_len + threads_per_block - 1) / threads_per_block);
	dim3 grid(grid_x, static_cast<unsigned int>(batch_size));

	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_combine_backprop_kernel<T><<<grid, threads_per_block, smem_size>>>(
		sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2,
		dimension, degree, sig_len
	);

	check_cuda_kernel_launch();
}

template<typename T>
void sig_combine_backprop_cuda_(
	const T* sig_combined_deriv,
	T* sig1_deriv,
	T* sig2_deriv,
	const T* sig1,
	const T* sig2,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("sig_combine_backprop_cuda received dimension 0");
	sig_combine_backprop_cuda_core_<T>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree);
}

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#include "cu_macros.h"

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {

	CUSIG_API int sig_combine_cuda_f(
		const float* sig1, const float* sig2, float* out,
		uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<float>(sig1, sig2, out, 1, dimension, degree));
	}

	CUSIG_API int sig_combine_cuda_d(
		const double* sig1, const double* sig2, double* out,
		uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<double>(sig1, sig2, out, 1, dimension, degree));
	}

	CUSIG_API int batch_sig_combine_cuda_f(
		const float* sig1, const float* sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<float>(sig1, sig2, out, batch_size, dimension, degree));
	}

	CUSIG_API int batch_sig_combine_cuda_d(
		const double* sig1, const double* sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<double>(sig1, sig2, out, batch_size, dimension, degree));
	}

	CUSIG_API int sig_combine_backprop_cuda_f(
		const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv,
		const float* sig1, const float* sig2,
		uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<float>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, 1, dimension, degree));
	}

	CUSIG_API int sig_combine_backprop_cuda_d(
		const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv,
		const double* sig1, const double* sig2,
		uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<double>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, 1, dimension, degree));
	}

	CUSIG_API int batch_sig_combine_backprop_cuda_f(
		const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv,
		const float* sig1, const float* sig2,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<float>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree));
	}

	CUSIG_API int batch_sig_combine_backprop_cuda_d(
		const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv,
		const double* sig1, const double* sig2,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<double>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree));
	}
}
