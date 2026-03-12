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
#include "cu_sig_coef.h"
#include "cu_atomic.h"
#include <type_traits>

// =========================================================================
// Constant memory for 1/k! values (max 21 doubles = 168 bytes)
// =========================================================================
__constant__ double c_one_over_fact_d[SIG_COEF_CUDA_MAX_DEGREE + 1];
__constant__ float  c_one_over_fact_f[SIG_COEF_CUDA_MAX_DEGREE + 1];

// =========================================================================
// Upload 1/k! to constant memory and return device pointer
// =========================================================================
template<typename T>
const T* upload_one_over_fact(uint64_t max_degree) {
	std::vector<T> h_ovf(max_degree + 1);
	h_ovf[0] = T(1);
	for (uint64_t i = 1; i <= max_degree; ++i)
		h_ovf[i] = h_ovf[i - 1] / T(i);

	const T* d_ptr;
	if constexpr (std::is_same_v<T, double>) {
		cudaMemcpyToSymbol(c_one_over_fact_d, h_ovf.data(), sizeof(T) * (max_degree + 1));
		cudaGetSymbolAddress((void**)&d_ptr, c_one_over_fact_d);
	}
	else {
		cudaMemcpyToSymbol(c_one_over_fact_f, h_ovf.data(), sizeof(T) * (max_degree + 1));
		cudaGetSymbolAddress((void**)&d_ptr, c_one_over_fact_f);
	}
	return d_ptr;
}

// =========================================================================
// Fixed-degree kernel dispatch macro (cases 1-10)
// =========================================================================
#define CUSIG_DISPATCH_FIXED_DEGREE(KERNEL, NUM_BLOCKS, TPB, ...) \
	switch (max_degree) { \
	case 1:  KERNEL<T, 1> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 2:  KERNEL<T, 2> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 3:  KERNEL<T, 3> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 4:  KERNEL<T, 4> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 5:  KERNEL<T, 5> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 6:  KERNEL<T, 6> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 7:  KERNEL<T, 7> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 8:  KERNEL<T, 8> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 9:  KERNEL<T, 9> <<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	case 10: KERNEL<T, 10><<<NUM_BLOCKS, TPB>>>(__VA_ARGS__); break; \
	}

// =========================================================================
// Generic CUDA sig_coef kernel (Horner scheme)
//
// The Chen identity update at each timestep is:
//   coefs[i] += sum_{j=0}^{i-1} coefs[j] * prod(dx[j..i-1]) * ovf[i-j]
//
// Horner evaluation (high-to-low for in-place update):
//   temp = ovf[i]; for k: temp = temp*dx[k] + coefs[k+1]*ovf[i-k-1]
//   coefs[i] += temp * dx[i-1]
//
// Benefits over the original buff-based approach:
//   - 25% fewer FP64 ops (no separate buff multiply pass)
//   - No cross-level serial dependency (ILP across levels)
//   - Eliminates the buff[] array entirely
// =========================================================================

template<typename T>
__global__ void sig_coef_kernel(
	const T* __restrict__ path,
	T* __restrict__ out,
	const uint64_t* __restrict__ multi_idx,
	const uint64_t* __restrict__ degrees,
	const T* __restrict__ one_over_fact,
	const uint64_t* __restrict__ idx_offsets,
	const uint64_t* __restrict__ out_offsets,
	uint64_t num_multi_idx,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t path_length,
	uint64_t result_length,
	bool prefixes
) {
	const uint64_t global_idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	const uint64_t total_work = batch_size * num_multi_idx;
	if (global_idx >= total_work) return;

	const uint64_t batch_idx = global_idx / num_multi_idx;
	const uint64_t word_idx = global_idx % num_multi_idx;

	const uint64_t idx_offset = idx_offsets[word_idx];
	const uint64_t out_offset = out_offsets[word_idx];
	const uint64_t degree = degrees[word_idx];
	const uint64_t* my_multi_idx_ptr = multi_idx + idx_offset;
	T* my_out = out + batch_idx * result_length + out_offset;
	const T* my_path = path + batch_idx * path_length * dimension;

	if (path_length <= 1) {
		const uint64_t n_out = (prefixes && degree) ? degree : 1;
		for (uint64_t i = 0; i < n_out; ++i) my_out[i] = T(0);
		return;
	}

	if (degree == 0) {
		*my_out = T(1);
		return;
	}

	// Cache channel indices in registers
	uint64_t channels[SIG_COEF_CUDA_MAX_DEGREE];
	for (uint64_t i = 0; i < degree; ++i) {
		channels[i] = my_multi_idx_ptr[i];
	}

	T coefs[SIG_COEF_CUDA_MAX_DEGREE + 1];

	// First segment: coefs[i] = prod(dx_0[0..i-1]) * ovf[i]
	coefs[0] = T(1);
	T prod = T(1);
	for (uint64_t i = 0; i < degree; ++i) {
		prod *= (my_path[dimension + channels[i]] - my_path[channels[i]]);
		coefs[i + 1] = prod * one_over_fact[i + 1];
	}

	// Main loop: Horner-based Chen's relation
	for (uint64_t t = 2; t < path_length; ++t) {
		T dx[SIG_COEF_CUDA_MAX_DEGREE];
		for (uint64_t i = 0; i < degree; ++i) {
			dx[i] = my_path[t * dimension + channels[i]]
			      - my_path[(t - 1) * dimension + channels[i]];
		}

		for (uint64_t i = degree; i >= 1; --i) {
			T temp = one_over_fact[i];
			for (uint64_t k = 0; k + 1 < i; ++k) {
				temp = temp * dx[k] + coefs[k + 1] * one_over_fact[i - k - 1];
			}
			coefs[i] += temp * dx[i - 1];
		}
	}

	if (!prefixes) {
		*my_out = coefs[degree];
	}
	else {
		for (uint64_t i = 0; i < degree; ++i) {
			my_out[i] = coefs[i + 1];
		}
	}
}

// =========================================================================
// Fixed-degree kernel with Horner scheme
//
// Compile-time degree enables full unrolling of all loops and register
// allocation for all arrays. Combined with Horner, each level's update
// is independent, enabling instruction-level parallelism.
// =========================================================================

template<typename T, uint64_t DEGREE>
__global__ void sig_coef_kernel_fixed_degree(
	const T* __restrict__ path,
	T* __restrict__ out,
	const uint64_t* __restrict__ multi_idx,
	const T* __restrict__ one_over_fact,
	const uint64_t* __restrict__ idx_offsets,
	const uint64_t* __restrict__ out_offsets,
	uint64_t num_multi_idx,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t path_length,
	uint64_t result_length,
	bool prefixes
) {
	const uint64_t global_idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	const uint64_t total_work = batch_size * num_multi_idx;
	if (global_idx >= total_work) return;

	const uint64_t batch_idx = global_idx / num_multi_idx;
	const uint64_t word_idx = global_idx % num_multi_idx;

	const uint64_t idx_offset = idx_offsets[word_idx];
	const uint64_t out_offset = out_offsets[word_idx];

	const uint64_t* my_multi_idx_ptr = multi_idx + idx_offset;
	T* my_out = out + batch_idx * result_length + out_offset;
	const T* my_path = path + batch_idx * path_length * dimension;

	// Cache channel indices and 1/k! in registers
	uint64_t channels[DEGREE];
	T ovf[DEGREE + 1];
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i) {
		channels[i] = my_multi_idx_ptr[i];
		ovf[i + 1] = one_over_fact[i + 1];
	}

	T coefs[DEGREE + 1];

	// First segment
	coefs[0] = T(1);
	T init_prod = T(1);
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i) {
		init_prod *= (my_path[dimension + channels[i]] - my_path[channels[i]]);
		coefs[i + 1] = init_prod * ovf[i + 1];
	}

	// Main loop: Horner-based Chen's relation
	for (uint64_t t = 2; t < path_length; ++t) {
		T dx[DEGREE];
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i) {
			dx[i] = my_path[t * dimension + channels[i]]
			      - my_path[(t - 1) * dimension + channels[i]];
		}

		// Horner scheme, high-to-low for in-place coefs update
		#pragma unroll
		for (uint64_t i = DEGREE; i >= 1; --i) {
			T temp = ovf[i];
			#pragma unroll
			for (uint64_t k = 0; k + 1 < i; ++k) {
				temp = temp * dx[k] + coefs[k + 1] * ovf[i - k - 1];
			}
			coefs[i] += temp * dx[i - 1];
		}
	}

	if (!prefixes) {
		*my_out = coefs[DEGREE];
	}
	else {
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i) {
			my_out[i] = coefs[i + 1];
		}
	}
}

// =========================================================================
// Host-side launch function
// =========================================================================

template<typename T>
void sig_coef_cuda_(
	const T* d_path,
	T* d_out,
	const uint64_t* d_multi_idx,
	uint64_t num_multi_idx,
	const uint64_t* d_degrees,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool prefixes
) {
	if (dimension == 0) throw std::invalid_argument("sig_coef_cuda received dimension 0");
	if (num_multi_idx == 0) return;

	// Copy degrees to host to compute max_degree, result_length, and prefix sums
	std::vector<uint64_t> h_degrees(num_multi_idx);
	cudaMemcpy(h_degrees.data(), d_degrees, sizeof(uint64_t) * num_multi_idx, cudaMemcpyDeviceToHost);

	uint64_t max_degree = 0;
	uint64_t result_length = 0;
	bool all_same_degree = true;

	// Compute prefix sums on host (single pass)
	std::vector<uint64_t> h_offsets(num_multi_idx * 2);
	uint64_t idx_sum = 0, out_sum = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		h_offsets[i] = idx_sum;
		h_offsets[num_multi_idx + i] = out_sum;
		const uint64_t deg_i = h_degrees[i];
		max_degree = std::max(max_degree, deg_i);
		idx_sum += deg_i;
		out_sum += (prefixes && deg_i) ? deg_i : 1;
		if (deg_i != h_degrees[0]) all_same_degree = false;
	}
	result_length = out_sum;

	if (max_degree > SIG_COEF_CUDA_MAX_DEGREE)
		throw std::invalid_argument("sig_coef_cuda: degree > 20 not supported");

	if (length <= 1) {
		cudaMemset(d_out, 0, sizeof(T) * batch_size * result_length);
		return;
	}

	const T* d_one_over_fact = upload_one_over_fact<T>(max_degree);

	// Upload prefix sums (single allocation, single memcpy)
	uint64_t* d_offsets;
	cudaMalloc(&d_offsets, sizeof(uint64_t) * num_multi_idx * 2);
	cudaMemcpy(d_offsets, h_offsets.data(), sizeof(uint64_t) * num_multi_idx * 2, cudaMemcpyHostToDevice);
	uint64_t* d_idx_offsets = d_offsets;
	uint64_t* d_out_offsets = d_offsets + num_multi_idx;

	// Launch kernel
	uint64_t total_work = batch_size * num_multi_idx;
	unsigned int tpb = 256;
	unsigned int num_blocks = static_cast<unsigned int>((total_work + tpb - 1) / tpb);

	if (all_same_degree && max_degree > 0 && max_degree <= 10) {
		CUSIG_DISPATCH_FIXED_DEGREE(sig_coef_kernel_fixed_degree, num_blocks, tpb,
			d_path, d_out, d_multi_idx, d_one_over_fact,
			d_idx_offsets, d_out_offsets,
			num_multi_idx, batch_size, dimension, length, result_length, prefixes);
	}
	else {
		sig_coef_kernel<T><<<num_blocks, tpb>>>(
			d_path, d_out, d_multi_idx, d_degrees, d_one_over_fact,
			d_idx_offsets, d_out_offsets,
			num_multi_idx, batch_size, dimension, length, result_length, prefixes);
	}

	cudaFree(d_offsets);
	check_cuda_kernel_launch();
}

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#include "cu_macros.h"

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {

	CUSIG_API int sig_coef_cuda_f(
		const float* path, float* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_cuda_<float>(path, out, multi_idx, num_multi_idx, degrees, 1, dimension, length, prefixes));
	}

	CUSIG_API int sig_coef_cuda_d(
		const double* path, double* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_cuda_<double>(path, out, multi_idx, num_multi_idx, degrees, 1, dimension, length, prefixes));
	}

	CUSIG_API int batch_sig_coef_cuda_f(
		const float* path, float* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_cuda_<float>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, prefixes));
	}

	CUSIG_API int batch_sig_coef_cuda_d(
		const double* path, double* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_cuda_<double>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, prefixes));
	}
}

// =========================================================================
// Backpropagation
// =========================================================================

// Note: signed_one_over_fact ((-1)^k / k!) is computed inline from one_over_fact
// to reduce register pressure — sign flip is free with compile-time unrolling.

// =========================================================================
// Generic backprop kernel (variable degree per word)
// - sovf computed inline from ovf to reduce register pressure
// - Merged atomicAdds: adjacent timesteps combined (~2x fewer atomics)
// =========================================================================

template<typename T>
__global__ void sig_coef_backprop_kernel(
	const T* __restrict__ path,
	T* __restrict__ out,
	const T* __restrict__ coefs_in,
	const T* __restrict__ derivs_in,
	const uint64_t* __restrict__ multi_idx,
	const uint64_t* __restrict__ degrees,
	const T* __restrict__ one_over_fact,
	const uint64_t* __restrict__ idx_offsets,
	const uint64_t* __restrict__ coef_offsets,
	uint64_t num_multi_idx,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t path_length,
	uint64_t coefs_length
) {
	const uint64_t global_idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	const uint64_t total_work = batch_size * num_multi_idx;
	if (global_idx >= total_work) return;

	const uint64_t batch_idx = global_idx / num_multi_idx;
	const uint64_t word_idx = global_idx % num_multi_idx;

	const uint64_t degree = degrees[word_idx];
	if (degree == 0 || path_length <= 1) return;

	const uint64_t idx_offset = idx_offsets[word_idx];
	const uint64_t coef_offset = coef_offsets[word_idx];

	const uint64_t* my_idx = multi_idx + idx_offset;
	const T* my_coefs = coefs_in + batch_idx * coefs_length + coef_offset;
	const T* my_derivs = derivs_in + batch_idx * coefs_length + coef_offset;
	const T* my_path = path + batch_idx * path_length * dimension;
	T* my_out = out + batch_idx * path_length * dimension;

	uint32_t ch[SIG_COEF_CUDA_MAX_DEGREE];
	for (uint64_t i = 0; i < degree; ++i)
		ch[i] = static_cast<uint32_t>(my_idx[i]);

	T c[SIG_COEF_CUDA_MAX_DEGREE + 1];
	c[0] = T(1);
	for (uint64_t i = 0; i < degree; ++i)
		c[i + 1] = my_coefs[i];

	T d[SIG_COEF_CUDA_MAX_DEGREE];
	for (uint64_t i = 0; i < degree; ++i)
		d[i] = my_derivs[i];

	T buff[SIG_COEF_CUDA_MAX_DEGREE];
	T prev_upd[SIG_COEF_CUDA_MAX_DEGREE];
	for (uint64_t i = 0; i < degree; ++i)
		prev_upd[i] = T(0);

	for (uint64_t t = path_length - 1; t >= 1; --t) {
		T dx[SIG_COEF_CUDA_MAX_DEGREE];
		for (uint64_t i = 0; i < degree; ++i)
			dx[i] = my_path[t * dimension + ch[i]]
			      - my_path[(t - 1) * dimension + ch[i]];

		// uncombine_coefs_ (sovf computed inline)
		{
			T last_c = c[0];
			for (uint64_t i = 1; i <= degree; ++i) {
				const T inc = dx[i - 1];
				for (uint64_t j = 0; j + 1 < i; ++j)
					buff[j] *= inc;
				buff[i - 1] = last_c * inc;
				T acc = T(0);
				for (uint64_t j = 0; j < i; ++j)
					acc += buff[j] * (((i - j) & 1) ? -one_over_fact[i - j] : one_over_fact[i - j]);
				last_c = c[i];
				c[i] += acc;
			}
		}

		// update_path_derivs_ → collect into cur_upd
		T cur_upd[SIG_COEF_CUDA_MAX_DEGREE];
		{
			T upd = d[0];
			T rp = T(1);
			for (uint64_t m = 1; m < degree; ++m) {
				rp *= dx[m];
				upd += d[m] * rp * one_over_fact[m + 1];
			}
			cur_upd[0] = upd;

			for (uint64_t i = 1; i < degree; ++i) {
				const T inc = dx[i - 1];
				for (uint64_t k = 0; k + 1 < i; ++k)
					buff[k] *= inc;
				buff[i - 1] = c[i - 1] * inc;

				T s = c[i];
				for (uint64_t k = 0; k < i; ++k)
					s += buff[k] * one_over_fact[i - k + 1];
				upd = d[i] * s;

				rp = T(1);
				for (uint64_t m = i + 1; m < degree; ++m) {
					rp *= dx[m];
					s = c[i] * one_over_fact[m - i + 1];
					for (uint64_t k = 0; k < i; ++k)
						s += buff[k] * one_over_fact[m - k + 1];
					s *= rp;
					upd += d[m] * s;
				}

				cur_upd[i] = upd;
			}
		}

		// Merged atomicAdds: net = cur_upd - prev_upd
		for (uint64_t i = 0; i < degree; ++i)
			myAtomicAdd(&my_out[t * dimension + ch[i]], cur_upd[i] - prev_upd[i]);

		for (uint64_t i = 0; i < degree; ++i)
			prev_upd[i] = cur_upd[i];

		// update_coef_derivs_
		{
			for (uint64_t i = 0; i + 1 < degree; ++i) {
				T acc = T(0);
				T rp = T(1);
				for (uint64_t k = i + 1; k < degree; ++k) {
					rp *= dx[k];
					acc += d[k] * rp * one_over_fact[k - i];
				}
				d[i] += acc;
			}
		}
	}

	// Final: write -prev_upd to position 0
	for (uint64_t i = 0; i < degree; ++i)
		myAtomicAdd(&my_out[ch[i]], -prev_upd[i]);
}

// =========================================================================
// Fixed-degree backprop kernel (compile-time unrolling)
// - sovf computed inline from ovf to reduce register pressure
// - Merged atomicAdds: adjacent timesteps combined (~2x fewer atomics)
// =========================================================================

template<typename T, uint64_t DEGREE>
__global__ void sig_coef_backprop_kernel_fixed_degree(
	const T* __restrict__ path,
	T* __restrict__ out,
	const T* __restrict__ coefs_in,
	const T* __restrict__ derivs_in,
	const uint64_t* __restrict__ multi_idx,
	const T* __restrict__ one_over_fact,
	const uint64_t* __restrict__ idx_offsets,
	const uint64_t* __restrict__ coef_offsets,
	uint64_t num_multi_idx,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t path_length,
	uint64_t coefs_length
) {
	const uint64_t global_idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	const uint64_t total_work = batch_size * num_multi_idx;
	if (global_idx >= total_work) return;

	const uint64_t batch_idx = global_idx / num_multi_idx;
	const uint64_t word_idx = global_idx % num_multi_idx;

	const uint64_t idx_offset = idx_offsets[word_idx];
	const uint64_t coef_offset = coef_offsets[word_idx];

	const uint64_t* my_idx = multi_idx + idx_offset;
	const T* my_coefs = coefs_in + batch_idx * coefs_length + coef_offset;
	const T* my_derivs = derivs_in + batch_idx * coefs_length + coef_offset;
	const T* my_path = path + batch_idx * path_length * dimension;
	T* my_out = out + batch_idx * path_length * dimension;

	uint32_t ch[DEGREE];
	T ovf[DEGREE + 1];

	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i) {
		ch[i] = static_cast<uint32_t>(my_idx[i]);
		ovf[i + 1] = one_over_fact[i + 1];
	}
	ovf[0] = T(1);

	T c[DEGREE + 1];
	c[0] = T(1);
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i)
		c[i + 1] = my_coefs[i];

	T d[DEGREE];
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i)
		d[i] = my_derivs[i];

	T buff[DEGREE];
	T prev_upd[DEGREE];
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i)
		prev_upd[i] = T(0);

	for (uint64_t t = path_length - 1; t >= 1; --t) {
		T dx[DEGREE];
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i)
			dx[i] = my_path[t * dimension + ch[i]]
			      - my_path[(t - 1) * dimension + ch[i]];

		// uncombine_coefs_ (sovf computed inline: (-1)^k * ovf[k])
		{
			T last_c = c[0];
			#pragma unroll
			for (uint64_t i = 1; i <= DEGREE; ++i) {
				const T inc = dx[i - 1];
				#pragma unroll
				for (uint64_t j = 0; j + 1 < i; ++j)
					buff[j] *= inc;
				buff[i - 1] = last_c * inc;
				T acc = T(0);
				#pragma unroll
				for (uint64_t j = 0; j < i; ++j)
					acc += buff[j] * (((i - j) & 1) ? -ovf[i - j] : ovf[i - j]);
				last_c = c[i];
				c[i] += acc;
			}
		}

		// update_path_derivs_ → collect into cur_upd
		T cur_upd[DEGREE];
		{
			T upd = d[0];
			T rp = T(1);
			#pragma unroll
			for (uint64_t m = 1; m < DEGREE; ++m) {
				rp *= dx[m];
				upd += d[m] * rp * ovf[m + 1];
			}
			cur_upd[0] = upd;

			#pragma unroll
			for (uint64_t i = 1; i < DEGREE; ++i) {
				const T inc = dx[i - 1];
				#pragma unroll
				for (uint64_t k = 0; k + 1 < i; ++k)
					buff[k] *= inc;
				buff[i - 1] = c[i - 1] * inc;

				T s = c[i];
				#pragma unroll
				for (uint64_t k = 0; k < i; ++k)
					s += buff[k] * ovf[i - k + 1];
				upd = d[i] * s;

				rp = T(1);
				#pragma unroll
				for (uint64_t m = i + 1; m < DEGREE; ++m) {
					rp *= dx[m];
					s = c[i] * ovf[m - i + 1];
					#pragma unroll
					for (uint64_t k = 0; k < i; ++k)
						s += buff[k] * ovf[m - k + 1];
					s *= rp;
					upd += d[m] * s;
				}

				cur_upd[i] = upd;
			}
		}

		// Merged atomicAdds: net = cur_upd - prev_upd
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i)
			myAtomicAdd(&my_out[t * dimension + ch[i]], cur_upd[i] - prev_upd[i]);

		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i)
			prev_upd[i] = cur_upd[i];

		// update_coef_derivs_
		{
			#pragma unroll
			for (uint64_t i = 0; i + 1 < DEGREE; ++i) {
				T acc = T(0);
				T rp = T(1);
				#pragma unroll
				for (uint64_t k = i + 1; k < DEGREE; ++k) {
					rp *= dx[k];
					acc += d[k] * rp * ovf[k - i];
				}
				d[i] += acc;
			}
		}
	}

	// Final: write -prev_upd to position 0
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i)
		myAtomicAdd(&my_out[ch[i]], -prev_upd[i]);
}

// =========================================================================
// Backprop host-side launch function
// =========================================================================

template<typename T>
void sig_coef_backprop_cuda_(
	const T* d_path,
	T* d_out,
	const T* d_coefs,
	const T* d_derivs,
	const uint64_t* d_multi_idx,
	uint64_t num_multi_idx,
	const uint64_t* d_degrees,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length
) {
	if (dimension == 0) throw std::invalid_argument("sig_coef_backprop_cuda received dimension 0");
	if (num_multi_idx == 0) return;

	// Copy degrees to host
	std::vector<uint64_t> h_degrees(num_multi_idx);
	cudaMemcpy(h_degrees.data(), d_degrees, sizeof(uint64_t) * num_multi_idx, cudaMemcpyDeviceToHost);

	uint64_t max_degree = 0;
	uint64_t coefs_length = 0;
	bool all_same_degree = true;

	// Compute prefix sums on host
	std::vector<uint64_t> h_offsets(num_multi_idx * 2);
	uint64_t idx_sum = 0, coef_sum = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		h_offsets[i] = idx_sum;
		h_offsets[num_multi_idx + i] = coef_sum;
		const uint64_t deg_i = h_degrees[i];
		max_degree = std::max(max_degree, deg_i);
		idx_sum += deg_i;
		coef_sum += deg_i ? deg_i : 1;
		if (deg_i != h_degrees[0]) all_same_degree = false;
	}
	coefs_length = coef_sum;

	if (max_degree > SIG_COEF_CUDA_MAX_DEGREE)
		throw std::invalid_argument("sig_coef_backprop_cuda: degree > 20 not supported");

	// Zero the output
	cudaMemset(d_out, 0, sizeof(T) * batch_size * length * dimension);

	if (length <= 1 || max_degree == 0) return;

	const T* d_ovf = upload_one_over_fact<T>(max_degree);

	// Upload prefix sums
	uint64_t* d_offsets;
	cudaMalloc(&d_offsets, sizeof(uint64_t) * num_multi_idx * 2);
	cudaMemcpy(d_offsets, h_offsets.data(), sizeof(uint64_t) * num_multi_idx * 2, cudaMemcpyHostToDevice);
	uint64_t* d_idx_offsets = d_offsets;
	uint64_t* d_coef_offsets = d_offsets + num_multi_idx;

	// Launch kernel
	uint64_t total_work = batch_size * num_multi_idx;
	unsigned int tpb = 256;
	unsigned int num_blocks = static_cast<unsigned int>((total_work + tpb - 1) / tpb);

	if (all_same_degree && max_degree > 0 && max_degree <= 10) {
		CUSIG_DISPATCH_FIXED_DEGREE(sig_coef_backprop_kernel_fixed_degree, num_blocks, tpb,
			d_path, d_out, d_coefs, d_derivs, d_multi_idx, d_ovf,
			d_idx_offsets, d_coef_offsets,
			num_multi_idx, batch_size, dimension, length, coefs_length);
	}
	else {
		sig_coef_backprop_kernel<T><<<num_blocks, tpb>>>(
			d_path, d_out, d_coefs, d_derivs, d_multi_idx, d_degrees, d_ovf,
			d_idx_offsets, d_coef_offsets,
			num_multi_idx, batch_size, dimension, length, coefs_length);
	}

	cudaFree(d_offsets);
	check_cuda_kernel_launch();
}

// =========================================================================
// Backprop exported C functions
// =========================================================================

extern "C" {

	CUSIG_API int sig_coef_backprop_cuda_f(
		const float* path, float* out, const float* coefs, const float* derivs,
		const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t dimension, uint64_t length
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_backprop_cuda_<float>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, 1, dimension, length));
	}

	CUSIG_API int sig_coef_backprop_cuda_d(
		const double* path, double* out, const double* coefs, const double* derivs,
		const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t dimension, uint64_t length
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_backprop_cuda_<double>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, 1, dimension, length));
	}

	CUSIG_API int batch_sig_coef_backprop_cuda_f(
		const float* path, float* out, const float* coefs, const float* derivs,
		const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t batch_size, uint64_t dimension, uint64_t length
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_backprop_cuda_<float>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length));
	}

	CUSIG_API int batch_sig_coef_backprop_cuda_d(
		const double* path, double* out, const double* coefs, const double* derivs,
		const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t batch_size, uint64_t dimension, uint64_t length
	) noexcept {
		CUSIG_SAFE_CALL(sig_coef_backprop_cuda_<double>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length));
	}
}
