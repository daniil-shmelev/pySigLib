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
#include <type_traits>

// =========================================================================
// Constant memory for 1/k! values (max 21 doubles = 168 bytes)
// =========================================================================
__constant__ double c_one_over_fact_d[SIG_COEF_CUDA_MAX_DEGREE + 1];
__constant__ float  c_one_over_fact_f[SIG_COEF_CUDA_MAX_DEGREE + 1];

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

	// Upload 1/k! to constant memory
	std::vector<T> h_one_over_fact(max_degree + 1);
	h_one_over_fact[0] = T(1);
	for (uint64_t i = 1; i <= max_degree; ++i)
		h_one_over_fact[i] = h_one_over_fact[i - 1] / T(i);

	const T* d_one_over_fact;
	if constexpr (std::is_same_v<T, double>) {
		cudaMemcpyToSymbol(c_one_over_fact_d, h_one_over_fact.data(),
			sizeof(T) * (max_degree + 1));
		cudaGetSymbolAddress((void**)&d_one_over_fact, c_one_over_fact_d);
	}
	else {
		cudaMemcpyToSymbol(c_one_over_fact_f, h_one_over_fact.data(),
			sizeof(T) * (max_degree + 1));
		cudaGetSymbolAddress((void**)&d_one_over_fact, c_one_over_fact_f);
	}

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

	if (all_same_degree && max_degree > 0) {
		switch (max_degree) {
		case 1:
			sig_coef_kernel_fixed_degree<T, 1><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 2:
			sig_coef_kernel_fixed_degree<T, 2><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 3:
			sig_coef_kernel_fixed_degree<T, 3><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 4:
			sig_coef_kernel_fixed_degree<T, 4><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 5:
			sig_coef_kernel_fixed_degree<T, 5><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 6:
			sig_coef_kernel_fixed_degree<T, 6><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 7:
			sig_coef_kernel_fixed_degree<T, 7><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 8:
			sig_coef_kernel_fixed_degree<T, 8><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 9:
			sig_coef_kernel_fixed_degree<T, 9><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		case 10:
			sig_coef_kernel_fixed_degree<T, 10><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		default:
			sig_coef_kernel<T><<<num_blocks, tpb>>>(
				d_path, d_out, d_multi_idx, d_degrees, d_one_over_fact,
				d_idx_offsets, d_out_offsets,
				num_multi_idx, batch_size, dimension, length, result_length, prefixes);
			break;
		}
	}
	else {
		sig_coef_kernel<T><<<num_blocks, tpb>>>(
			d_path, d_out, d_multi_idx, d_degrees, d_one_over_fact,
			d_idx_offsets, d_out_offsets,
			num_multi_idx, batch_size, dimension, length, result_length, prefixes);
	}

	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	cudaFree(d_offsets);

	if (err != cudaSuccess)
		throw std::runtime_error(std::string("sig_coef_cuda kernel failed: ") + cudaGetErrorString(err));
}

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#ifndef CU_SIG_COEF_SAFE_CALL
#define CU_SIG_COEF_SAFE_CALL(function_call)                    \
    try {                                                       \
        function_call;                                          \
    }                                                           \
    catch (std::bad_alloc&) {                                   \
        std::cerr << "Failed to allocate memory";               \
        return 1;                                               \
    }                                                           \
    catch (std::invalid_argument& e) {                          \
        std::cerr << e.what();                                  \
        return 2;                                               \
    }                                                           \
    catch (std::exception& e) {                                 \
        std::cerr << e.what();                                  \
        return 3;                                               \
    }                                                           \
    catch (...) {                                               \
        std::cerr << "Unknown error";                           \
        return 4;                                               \
    }                                                           \
    return 0;
#endif

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {

	CUSIG_API int sig_coef_cuda_f(
		const float* path, float* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CU_SIG_COEF_SAFE_CALL(sig_coef_cuda_<float>(path, out, multi_idx, num_multi_idx, degrees, 1, dimension, length, prefixes));
	}

	CUSIG_API int sig_coef_cuda_d(
		const double* path, double* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CU_SIG_COEF_SAFE_CALL(sig_coef_cuda_<double>(path, out, multi_idx, num_multi_idx, degrees, 1, dimension, length, prefixes));
	}

	CUSIG_API int batch_sig_coef_cuda_f(
		const float* path, float* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CU_SIG_COEF_SAFE_CALL(sig_coef_cuda_<float>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, prefixes));
	}

	CUSIG_API int batch_sig_coef_cuda_d(
		const double* path, double* out, const uint64_t* multi_idx,
		uint64_t num_multi_idx, const uint64_t* degrees,
		uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes
	) noexcept {
		CU_SIG_COEF_SAFE_CALL(sig_coef_cuda_<double>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, prefixes));
	}
}
