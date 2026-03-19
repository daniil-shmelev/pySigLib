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
// CUDA backward kernels for log_sig_combine
// =========================================================================

template<typename T>
__global__ void batch_log_sig_combine_backprop_kernel_(
	const T* __restrict__ d_out,
	T* __restrict__ d_ls1,
	T* __restrict__ d_ls2,
	const T* __restrict__ ls1,
	const T* __restrict__ ls2,
	T* __restrict__ workspace,
	const double* __restrict__ bch_coefs,
	const uint64_t* __restrict__ bch_lf,
	const uint64_t* __restrict__ bch_rf,
	const uint32_t* __restrict__ comm_k_ptr,
	const uint32_t* __restrict__ comm_k_i,
	const uint32_t* __restrict__ comm_k_j,
	const int* __restrict__ comm_k_val,
	const uint32_t* __restrict__ comm_a_ptr,
	const uint32_t* __restrict__ comm_a_k,
	const uint32_t* __restrict__ comm_a_partner,
	const int* __restrict__ comm_a_signed_c,
	uint64_t m, uint64_t m2
) {
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;

	extern __shared__ char smem[];
	T* s_v1 = reinterpret_cast<T*>(smem);
	T* s_v2 = s_v1 + m;

	const T* my_dout = d_out + batch_idx * m;
	T* my_dls1 = d_ls1 + batch_idx * m;
	T* my_dls2 = d_ls2 + batch_idx * m;
	const T* my_ls1 = ls1 + batch_idx * m;
	const T* my_ls2 = ls2 + batch_idx * m;
	// workspace: memo[m2*m] then d_memo[m2*m]
	T* memo = workspace + batch_idx * 2 * m2 * m;
	T* d_memo = memo + m2 * m;

	// Phase 1: d_ls1 = d_out, d_ls2 = d_out
	for (uint64_t k = tid; k < m; k += stride) {
		my_dls1[k] = my_dout[k];
		my_dls2[k] = my_dout[k];
	}

	if (m2 <= 2) return;

	// Phase 2: Recompute forward memo
	for (uint64_t k = tid; k < m; k += stride) {
		memo[k] = my_ls1[k];
		memo[m + k] = my_ls2[k];
	}
	__syncthreads();

	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t lf = bch_lf[w];
		const uint64_t rf = bch_rf[w];
		T* result = memo + w * m;

		const T* v1_global = memo + lf * m;
		const T* v2_global = memo + rf * m;
		for (uint64_t k = tid; k < m; k += stride) {
			s_v1[k] = v1_global[k];
			s_v2[k] = v2_global[k];
		}
		__syncthreads();

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
		}
		__syncthreads();
	}

	// Phase 3: Initialize d_memo
	for (uint64_t k = tid; k < m; k += stride) {
		d_memo[k] = T(0);
		d_memo[m + k] = T(0);
	}
	for (uint64_t w = 2; w < m2; ++w) {
		for (uint64_t k = tid; k < m; k += stride) {
			d_memo[w * m + k] = T(bch_coefs[w]) * my_dout[k];
		}
	}
	__syncthreads();

	// Phase 4: Reverse BCH loop using input-grouped table
	for (uint64_t w = m2 - 1; w >= 2; --w) {
		const uint64_t lf = bch_lf[w];
		const uint64_t rf = bch_rf[w];
		const T* dm_w = d_memo + w * m;
		T* dm_lf = d_memo + lf * m;
		T* dm_rf = d_memo + rf * m;

		// Load v1=memo[lf], v2=memo[rf] into shared memory
		const T* v1_global = memo + lf * m;
		const T* v2_global = memo + rf * m;
		for (uint64_t k = tid; k < m; k += stride) {
			s_v1[k] = v1_global[k];
			s_v2[k] = v2_global[k];
		}
		__syncthreads();

		// Gather per input index a (no race conditions)
		for (uint64_t a = tid; a < m; a += stride) {
			T acc_dv1 = T(0);
			T acc_dv2 = T(0);
			const uint32_t start = comm_a_ptr[a];
			const uint32_t end = comm_a_ptr[a + 1];
			for (uint32_t idx = start; idx < end; ++idx) {
				const uint32_t k = comm_a_k[idx];
				const uint32_t partner = comm_a_partner[idx];
				const int sc = comm_a_signed_c[idx];
				const T dk = dm_w[k];
				acc_dv1 += T(sc) * s_v2[partner] * dk;
				acc_dv2 -= T(sc) * s_v1[partner] * dk;
			}
			dm_lf[a] += acc_dv1;
			dm_rf[a] += acc_dv2;
		}
		__syncthreads();
	}

	// Phase 5: Accumulate leaf gradients
	for (uint64_t k = tid; k < m; k += stride) {
		my_dls1[k] += d_memo[k];
		my_dls2[k] += d_memo[m + k];
	}
}

template<typename T>
__global__ void batch_log_sig_combine_backprop_kernel_noshmem_(
	const T* __restrict__ d_out,
	T* __restrict__ d_ls1,
	T* __restrict__ d_ls2,
	const T* __restrict__ ls1,
	const T* __restrict__ ls2,
	T* __restrict__ workspace,
	const double* __restrict__ bch_coefs,
	const uint64_t* __restrict__ bch_lf,
	const uint64_t* __restrict__ bch_rf,
	const uint32_t* __restrict__ comm_k_ptr,
	const uint32_t* __restrict__ comm_k_i,
	const uint32_t* __restrict__ comm_k_j,
	const int* __restrict__ comm_k_val,
	const uint32_t* __restrict__ comm_a_ptr,
	const uint32_t* __restrict__ comm_a_k,
	const uint32_t* __restrict__ comm_a_partner,
	const int* __restrict__ comm_a_signed_c,
	uint64_t m, uint64_t m2
) {
	// Same as above but without shared memory for v1/v2
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;

	const T* my_dout = d_out + batch_idx * m;
	T* my_dls1 = d_ls1 + batch_idx * m;
	T* my_dls2 = d_ls2 + batch_idx * m;
	const T* my_ls1 = ls1 + batch_idx * m;
	const T* my_ls2 = ls2 + batch_idx * m;
	T* memo = workspace + batch_idx * 2 * m2 * m;
	T* d_memo = memo + m2 * m;

	for (uint64_t k = tid; k < m; k += stride) {
		my_dls1[k] = my_dout[k];
		my_dls2[k] = my_dout[k];
	}

	if (m2 <= 2) return;

	for (uint64_t k = tid; k < m; k += stride) {
		memo[k] = my_ls1[k];
		memo[m + k] = my_ls2[k];
	}
	__syncthreads();

	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t lf = bch_lf[w];
		const uint64_t rf = bch_rf[w];
		const T* v1 = memo + lf * m;
		const T* v2 = memo + rf * m;
		T* result = memo + w * m;

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
		}
		__syncthreads();
	}

	for (uint64_t k = tid; k < m; k += stride) {
		d_memo[k] = T(0);
		d_memo[m + k] = T(0);
	}
	for (uint64_t w = 2; w < m2; ++w) {
		for (uint64_t k = tid; k < m; k += stride) {
			d_memo[w * m + k] = T(bch_coefs[w]) * my_dout[k];
		}
	}
	__syncthreads();

	for (uint64_t w = m2 - 1; w >= 2; --w) {
		const uint64_t lf = bch_lf[w];
		const uint64_t rf = bch_rf[w];
		const T* v1 = memo + lf * m;
		const T* v2 = memo + rf * m;
		const T* dm_w = d_memo + w * m;
		T* dm_lf = d_memo + lf * m;
		T* dm_rf = d_memo + rf * m;

		for (uint64_t a = tid; a < m; a += stride) {
			T acc_dv1 = T(0);
			T acc_dv2 = T(0);
			const uint32_t start = comm_a_ptr[a];
			const uint32_t end = comm_a_ptr[a + 1];
			for (uint32_t idx = start; idx < end; ++idx) {
				const uint32_t k = comm_a_k[idx];
				const uint32_t partner = comm_a_partner[idx];
				const int sc = comm_a_signed_c[idx];
				const T dk = dm_w[k];
				acc_dv1 += T(sc) * v2[partner] * dk;
				acc_dv2 -= T(sc) * v1[partner] * dk;
			}
			dm_lf[a] += acc_dv1;
			dm_rf[a] += acc_dv2;
		}
		__syncthreads();
	}

	for (uint64_t k = tid; k < m; k += stride) {
		my_dls1[k] += d_memo[k];
		my_dls2[k] += d_memo[m + k];
	}
}

// =========================================================================
// Host-side backward launcher
// =========================================================================

template<typename T>
void batch_log_sig_combine_backprop_cuda_(
	const T* d_out, T* d_ls1, T* d_ls2,
	const T* ls1, const T* ls2,
	uint64_t batch_size, uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine_backprop_cuda received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine_backprop_cuda received degree 0");

	const CUDABchCache& cache = get_cuda_bch_cache_(dimension, degree);
	uint64_t m = cache.m;
	uint64_t m2 = cache.m2;

	if (degree < 2) {
		// d_ls1 = d_out, d_ls2 = d_out
		uint64_t total = batch_size * m;
		cudaMemcpy(d_ls1, d_out, total * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_ls2, d_out, total * sizeof(T), cudaMemcpyDeviceToDevice);
		check_cuda_error();
		return;
	}

	// Workspace: 2 * m2 * m per batch element (memo + d_memo)
	uint64_t ws_per_batch = 2 * m2 * m;

	// Cached workspace to avoid cudaMalloc/cudaFree per call
	static T* s_workspace = nullptr;
	static size_t s_workspace_elems = 0;

	size_t needed_elems = batch_size * ws_per_batch;
	if (needed_elems > s_workspace_elems) {
		if (s_workspace) { cudaFree(s_workspace); s_workspace = nullptr; s_workspace_elems = 0; }
		// Check available memory and determine chunk size
		size_t free_mem, total_mem;
		cudaMemGetInfo(&free_mem, &total_mem);
		uint64_t max_batch = free_mem / (ws_per_batch * sizeof(T) * 2);
		if (max_batch < 1) max_batch = 1;
		uint64_t alloc_batch = std::min(batch_size, max_batch);
		size_t alloc_elems = alloc_batch * ws_per_batch;
		cudaMalloc(&s_workspace, alloc_elems * sizeof(T));
		check_cuda_error();
		s_workspace_elems = alloc_elems;
	}

	uint64_t chunk_size = s_workspace_elems / ws_per_batch;
	if (chunk_size > batch_size) chunk_size = batch_size;

	unsigned int threads = std::min(static_cast<uint64_t>(64), m);
	threads = ((threads + 31) / 32) * 32;
	if (threads < 32) threads = 32;

	size_t shared_size = 2 * m * sizeof(T);
	bool use_shmem = (shared_size <= 48 * 1024);

	for (uint64_t offset = 0; offset < batch_size; offset += chunk_size) {
		uint64_t current_batch = std::min(chunk_size, batch_size - offset);

		if (use_shmem) {
			batch_log_sig_combine_backprop_kernel_<T><<<static_cast<unsigned int>(current_batch), threads, shared_size>>>(
				d_out + offset * m,
				d_ls1 + offset * m,
				d_ls2 + offset * m,
				ls1 + offset * m,
				ls2 + offset * m,
				s_workspace,
				cache.d_bch_coefficients, cache.d_bch_left_factor, cache.d_bch_right_factor,
				cache.d_comm_k_ptr, cache.d_comm_k_i, cache.d_comm_k_j, cache.d_comm_k_val,
				cache.d_comm_a_ptr, cache.d_comm_a_k, cache.d_comm_a_partner, cache.d_comm_a_signed_c,
				m, m2
			);
		} else {
			batch_log_sig_combine_backprop_kernel_noshmem_<T><<<static_cast<unsigned int>(current_batch), threads>>>(
				d_out + offset * m,
				d_ls1 + offset * m,
				d_ls2 + offset * m,
				ls1 + offset * m,
				ls2 + offset * m,
				s_workspace,
				cache.d_bch_coefficients, cache.d_bch_left_factor, cache.d_bch_right_factor,
				cache.d_comm_k_ptr, cache.d_comm_k_i, cache.d_comm_k_j, cache.d_comm_k_val,
				cache.d_comm_a_ptr, cache.d_comm_a_k, cache.d_comm_a_partner, cache.d_comm_a_signed_c,
				m, m2
			);
		}

		check_cuda_kernel_launch();
	}
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

CUSIG_API int log_sig_combine_backprop_cuda_f(
	const float* d_out, float* d_ls1, float* d_ls2,
	const float* ls1, const float* ls2,
	uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_backprop_cuda_<float>(d_out, d_ls1, d_ls2, ls1, ls2, 1, dimension, degree));
}

CUSIG_API int log_sig_combine_backprop_cuda_d(
	const double* d_out, double* d_ls1, double* d_ls2,
	const double* ls1, const double* ls2,
	uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_backprop_cuda_<double>(d_out, d_ls1, d_ls2, ls1, ls2, 1, dimension, degree));
}

CUSIG_API int batch_log_sig_combine_backprop_cuda_f(
	const float* d_out, float* d_ls1, float* d_ls2,
	const float* ls1, const float* ls2,
	uint64_t batch_size, uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_backprop_cuda_<float>(d_out, d_ls1, d_ls2, ls1, ls2, batch_size, dimension, degree));
}

CUSIG_API int batch_log_sig_combine_backprop_cuda_d(
	const double* d_out, double* d_ls1, double* d_ls2,
	const double* ls1, const double* ls2,
	uint64_t batch_size, uint64_t dimension, uint64_t degree
) noexcept {
	CUSIG_SAFE_CALL(batch_log_sig_combine_backprop_cuda_<double>(d_out, d_ls1, d_ls2, ls1, ls2, batch_size, dimension, degree));
}

} // extern "C"
