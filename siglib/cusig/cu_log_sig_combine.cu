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
#include "cu_log_sig_combine.h"
#include "cu_macros.h"

// =========================================================================
// CUDA kernel: one block per batch element, threads cooperate on lie_bracket
// =========================================================================

// Kernel with shared memory: v1/v2 loaded into shared memory each BCH iteration
// to eliminate scattered global memory reads in the inner loop.
template<typename T>
__global__ void batch_log_sig_combine_kernel_(
	const T* __restrict__ log_sig1,
	const T* __restrict__ log_sig2,
	T* __restrict__ out,
	T* __restrict__ memo,
	const double* __restrict__ bch_coefs,
	const uint64_t* __restrict__ bch_lf,
	const uint64_t* __restrict__ bch_rf,
	const uint32_t* __restrict__ comm_k_ptr,
	const uint32_t* __restrict__ comm_k_i,
	const uint32_t* __restrict__ comm_k_j,
	const int* __restrict__ comm_k_val,
	uint64_t m, uint64_t m2
) {
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;

	extern __shared__ char smem[];
	T* s_v1 = reinterpret_cast<T*>(smem);
	T* s_v2 = s_v1 + m;

	const T* ls1 = log_sig1 + batch_idx * m;
	const T* ls2 = log_sig2 + batch_idx * m;
	T* o = out + batch_idx * m;
	T* my_memo = memo + batch_idx * m2 * m;

	// out = ls1 + ls2
	for (uint64_t k = tid; k < m; k += stride) {
		o[k] = ls1[k] + ls2[k];
	}

	if (m2 <= 2) return;

	// Initialize leaves: memo[0] = ls1, memo[1] = ls2
	for (uint64_t k = tid; k < m; k += stride) {
		my_memo[k] = ls1[k];
		my_memo[m + k] = ls2[k];
	}

	__syncthreads();

	// BCH loop: indices are topologically sorted (children < parent)
	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t lf = bch_lf[w];
		const uint64_t rf = bch_rf[w];
		T* result = my_memo + w * m;

		// Cooperatively load v1 and v2 into shared memory
		const T* v1_global = my_memo + lf * m;
		const T* v2_global = my_memo + rf * m;
		for (uint64_t k = tid; k < m; k += stride) {
			s_v1[k] = v1_global[k];
			s_v2[k] = v2_global[k];
		}
		__syncthreads();

		// lie_bracket via transposed commutator table + accumulate into output
		const T c_w = T(bch_coefs[w]);
		for (uint64_t k = tid; k < m; k += stride) {
			T sum = T(0);
			const uint32_t start = comm_k_ptr[k];
			const uint32_t end = comm_k_ptr[k + 1];
			for (uint32_t idx = start; idx < end; ++idx) {
				const uint32_t i = comm_k_i[idx];
				const uint32_t j = comm_k_j[idx];
				const int c = comm_k_val[idx];
				sum += T(c) * (s_v1[i] * s_v2[j] - s_v1[j] * s_v2[i]);
			}
			result[k] = sum;
			if (c_w != T(0)) o[k] += c_w * sum;
		}

		__syncthreads();
	}
}

// Fallback kernel without shared memory (for large m where 2*m*sizeof(T) > 48KB)
template<typename T>
__global__ void batch_log_sig_combine_kernel_noshmem_(
	const T* __restrict__ log_sig1,
	const T* __restrict__ log_sig2,
	T* __restrict__ out,
	T* __restrict__ memo,
	const double* __restrict__ bch_coefs,
	const uint64_t* __restrict__ bch_lf,
	const uint64_t* __restrict__ bch_rf,
	const uint32_t* __restrict__ comm_k_ptr,
	const uint32_t* __restrict__ comm_k_i,
	const uint32_t* __restrict__ comm_k_j,
	const int* __restrict__ comm_k_val,
	uint64_t m, uint64_t m2
) {
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;

	const T* ls1 = log_sig1 + batch_idx * m;
	const T* ls2 = log_sig2 + batch_idx * m;
	T* o = out + batch_idx * m;
	T* my_memo = memo + batch_idx * m2 * m;

	for (uint64_t k = tid; k < m; k += stride) {
		o[k] = ls1[k] + ls2[k];
	}

	if (m2 <= 2) return;

	for (uint64_t k = tid; k < m; k += stride) {
		my_memo[k] = ls1[k];
		my_memo[m + k] = ls2[k];
	}

	__syncthreads();

	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t lf = bch_lf[w];
		const uint64_t rf = bch_rf[w];
		const T* v1 = my_memo + lf * m;
		const T* v2 = my_memo + rf * m;
		T* result = my_memo + w * m;

		const T c_w = T(bch_coefs[w]);
		for (uint64_t k = tid; k < m; k += stride) {
			T sum = T(0);
			const uint32_t start = comm_k_ptr[k];
			const uint32_t end = comm_k_ptr[k + 1];
			for (uint32_t idx = start; idx < end; ++idx) {
				const uint32_t i = comm_k_i[idx];
				const uint32_t j = comm_k_j[idx];
				const int c = comm_k_val[idx];
				sum += T(c) * (v1[i] * v2[j] - v1[j] * v2[i]);
			}
			result[k] = sum;
			if (c_w != T(0)) o[k] += c_w * sum;
		}

		__syncthreads();
	}
}

// Degree < 2 special case: output = ls1 + ls2
template<typename T>
__global__ void batch_log_sig_add_kernel_(
	const T* __restrict__ log_sig1,
	const T* __restrict__ log_sig2,
	T* __restrict__ out,
	uint64_t m
) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		out[idx] = log_sig1[idx] + log_sig2[idx];
	}
}

// =========================================================================
// Host-side launcher
// =========================================================================

template<typename T>
void batch_log_sig_combine_cuda_(
	const T* log_sig1, const T* log_sig2, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine_cuda received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine_cuda received degree 0");

	const CUDABchCache& cache = get_cuda_bch_cache_(dimension, degree);
	uint64_t m = cache.m;
	uint64_t m2 = cache.m2;

	if (degree < 2) {
		uint64_t total = batch_size * m;
		unsigned int threads = 256;
		unsigned int blocks = static_cast<unsigned int>((total + threads - 1) / threads);
		batch_log_sig_add_kernel_<T><<<blocks, threads>>>(log_sig1, log_sig2, out, total);
		check_cuda_kernel_launch();
		return;
	}

	// Determine chunk size based on available GPU memory
	uint64_t memo_per_batch = m2 * m;
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);

	// Use at most half of free memory for memo
	uint64_t max_batch = free_mem / (memo_per_batch * sizeof(T) * 2);
	if (max_batch < 1) max_batch = 1;
	uint64_t chunk_size = std::min(batch_size, max_batch);

	T* d_memo = nullptr;
	cudaMalloc(&d_memo, chunk_size * memo_per_batch * sizeof(T));
	check_cuda_error();

	unsigned int threads = std::min(static_cast<uint64_t>(256), m);
	threads = ((threads + 31) / 32) * 32;
	if (threads < 32) threads = 32;

	// Decide whether shared memory kernel fits (2 vectors of size m)
	size_t shared_size = 2 * m * sizeof(T);
	bool use_shmem = (shared_size <= 48 * 1024);

	for (uint64_t offset = 0; offset < batch_size; offset += chunk_size) {
		uint64_t current_batch = std::min(chunk_size, batch_size - offset);

		if (use_shmem) {
			batch_log_sig_combine_kernel_<T><<<static_cast<unsigned int>(current_batch), threads, shared_size>>>(
				log_sig1 + offset * m,
				log_sig2 + offset * m,
				out + offset * m,
				d_memo,
				cache.d_bch_coefficients, cache.d_bch_left_factor, cache.d_bch_right_factor,
				cache.d_comm_k_ptr, cache.d_comm_k_i, cache.d_comm_k_j, cache.d_comm_k_val,
				m, m2
			);
		} else {
			batch_log_sig_combine_kernel_noshmem_<T><<<static_cast<unsigned int>(current_batch), threads>>>(
				log_sig1 + offset * m,
				log_sig2 + offset * m,
				out + offset * m,
				d_memo,
				cache.d_bch_coefficients, cache.d_bch_left_factor, cache.d_bch_right_factor,
				cache.d_comm_k_ptr, cache.d_comm_k_i, cache.d_comm_k_j, cache.d_comm_k_val,
				m, m2
			);
		}

		check_cuda_kernel_launch();
	}

	cudaFree(d_memo);
	check_cuda_error();
}

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {

CUSIG_API int log_sig_combine_cuda_f(
	const float* log_sig1, const float* log_sig2, float* out,
	uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_cuda_<float>(log_sig1, log_sig2, out, 1, dimension, degree));
}

CUSIG_API int log_sig_combine_cuda_d(
	const double* log_sig1, const double* log_sig2, double* out,
	uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_cuda_<double>(log_sig1, log_sig2, out, 1, dimension, degree));
}

CUSIG_API int batch_log_sig_combine_cuda_f(
	const float* log_sig1, const float* log_sig2, float* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_cuda_<float>(log_sig1, log_sig2, out, batch_size, dimension, degree));
}

CUSIG_API int batch_log_sig_combine_cuda_d(
	const double* log_sig1, const double* log_sig2, double* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_cuda_<double>(log_sig1, log_sig2, out, batch_size, dimension, degree));
}

} // extern "C"
