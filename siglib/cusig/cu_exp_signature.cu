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

#include "cusig.h"
#include "cu_utils.h"
#include "cu_atomic.h"
#include "cu_exp_host.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

// =========================================================================
// tensor_exp: exp(x) = 1 + P_1 + ... + P_N, P_1=x, P_n=x \otimes P_{n-1}/n
// P_n has min level n -> level-skipping. buff must be 2*sig_len.
// =========================================================================

template<typename T>
__device__ void tensor_exp_device(
	const T* __restrict__ log_sig,
	T* __restrict__ out,
	T* __restrict__ buff,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;
	const uint64_t sig_len = level_index[degree + 1];

	if (tid == 0) out[0] = static_cast<T>(1);
	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		out[i] = log_sig[i];
	__syncthreads();

	if (degree <= 1) return;

	T* P_prev = buff;
	T* P_curr = buff + sig_len;

	for (uint64_t i = tid; i < sig_len; i += nthreads)
		P_prev[i] = log_sig[i];
	__syncthreads();

	for (uint64_t n = 2; n <= degree; ++n) {
		T inv_n = static_cast<T>(1) / static_cast<T>(n);

		for (uint64_t target_level = n; target_level <= degree; ++target_level) {
			const uint64_t target_start = level_index[target_level];
			const uint64_t target_size = level_index[target_level + 1] - target_start;
			const uint64_t max_left = target_level - (n - 1);

			for (uint64_t i = tid; i < target_size; i += nthreads)
				P_curr[target_start + i] = static_cast<T>(0);
			__syncthreads();

			for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
					const uint64_t l_idx = idx / right_size;
					const uint64_t r_idx = idx % right_size;
					P_curr[target_start + idx] += (log_sig[left_start + l_idx] * inv_n) * P_prev[right_start + r_idx];
				}
				__syncthreads();
			}

			// Fuse accumulation into per-level loop
			for (uint64_t i = tid; i < target_size; i += nthreads)
				out[target_start + i] += P_curr[target_start + i];
			__syncthreads();
		}

		T* tmp = P_prev;
		P_prev = P_curr;
		P_curr = tmp;
		__syncthreads();
	}
}

// =========================================================================
// tensor_exp_backprop: recomputes P_1..P_N, backprops from n=degree to 2.
// P_all: degree*sig_len scratch. dP+dP_next: 2*sig_len scratch.
// =========================================================================

template<typename T>
__device__ void tensor_exp_backprop_device(
	T* __restrict__ d_logsig,
	const T* __restrict__ d_sig,
	const T* __restrict__ log_sig,
	T* __restrict__ P_all,
	T* __restrict__ dP,
	T* __restrict__ dP_next,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;
	const uint64_t sig_len = level_index[degree + 1];

	// Zero d_logsig
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		d_logsig[i] = static_cast<T>(0);
	__syncthreads();

	if (degree <= 1) {
		for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
			d_logsig[i] = d_sig[i];
		__syncthreads();
		return;
	}

	for (uint64_t i = tid; i < sig_len; i += nthreads)
		P_all[i] = log_sig[i];
	__syncthreads();

	for (uint64_t n = 2; n <= degree; ++n) {
		T inv_n = static_cast<T>(1) / static_cast<T>(n);
		T* P_curr = P_all + (n - 1) * sig_len;
		const T* P_prev = P_all + (n - 2) * sig_len;

		for (uint64_t i = level_index[n] + tid; i < sig_len; i += nthreads)
			P_curr[i] = static_cast<T>(0);
		__syncthreads();

		for (uint64_t target_level = n; target_level <= degree; ++target_level) {
			const uint64_t target_start = level_index[target_level];
			const uint64_t target_size = level_index[target_level + 1] - target_start;
			const uint64_t max_left = target_level - (n - 1);

			for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
					const uint64_t l_idx = idx / right_size;
					const uint64_t r_idx = idx % right_size;
					P_curr[target_start + idx] += (log_sig[left_start + l_idx] * inv_n) * P_prev[right_start + r_idx];
				}
				__syncthreads();
			}
		}
	}

	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		d_logsig[i] = d_sig[i];
	__syncthreads();

	for (uint64_t i = tid; i < sig_len; i += nthreads)
		dP[i] = static_cast<T>(0);
	__syncthreads();

	for (int64_t n = static_cast<int64_t>(degree); n >= 2; --n) {
		T inv_n = static_cast<T>(1) / static_cast<T>(n);
		const T* P_prev = P_all + (n - 2) * sig_len;

		for (uint64_t i = tid; i < sig_len; i += nthreads)
			dP_next[i] = static_cast<T>(0);
		__syncthreads();

		for (uint64_t target_level = static_cast<uint64_t>(n); target_level <= degree; ++target_level) {
			const uint64_t target_start = level_index[target_level];
			const uint64_t max_left = target_level - (n - 1);

			for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t left_size = level_index[left_level + 1] - left_start;
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				for (uint64_t l_idx = tid; l_idx < left_size; l_idx += nthreads) {
					T acc = static_cast<T>(0);
					for (uint64_t r_idx = 0; r_idx < right_size; ++r_idx) {
						uint64_t t_idx = l_idx * right_size + r_idx;
						acc += (d_sig[target_start + t_idx] + dP[target_start + t_idx]) * P_prev[right_start + r_idx];
					}
					d_logsig[left_start + l_idx] += acc * inv_n;
				}
				__syncthreads();

				for (uint64_t r_idx = tid; r_idx < right_size; r_idx += nthreads) {
					T acc = static_cast<T>(0);
					for (uint64_t l_idx = 0; l_idx < left_size; ++l_idx) {
						uint64_t t_idx = l_idx * right_size + r_idx;
						acc += (d_sig[target_start + t_idx] + dP[target_start + t_idx]) * log_sig[left_start + l_idx];
					}
					dP_next[right_start + r_idx] += acc * inv_n;
				}
				__syncthreads();
			}
		}

		T* tmp = dP;
		dP = dP_next;
		dP_next = tmp;
		__syncthreads();
	}

	// dP now holds chain gradient to P_1 = x
	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		d_logsig[i] += dP[i];
	__syncthreads();
}

// =========================================================================
// Forward kernel: one block per batch element
// =========================================================================

template<typename T>
__global__ void logsig_to_sig_kernel(
	const T* __restrict__ log_sig,
	T* __restrict__ out,
	T* __restrict__ buff,
	const uint64_t* __restrict__ d_level_index,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_log_sig = log_sig + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;
	T* my_buff = buff + batch_idx * 2 * sig_len; // needs 2*sig_len for P_prev/P_curr

	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	tensor_exp_device<T>(my_log_sig, my_out, my_buff, degree, level_index_smem);
}

// =========================================================================
// Backward kernel: one block per batch element
// =========================================================================

template<typename T>
__global__ void logsig_to_sig_backprop_kernel(
	T* __restrict__ d_logsig,
	const T* __restrict__ d_sig,
	const T* __restrict__ log_sig,
	T* __restrict__ P_all_buf,
	T* __restrict__ dP_buf,
	const uint64_t* __restrict__ d_level_index,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	T* my_d_logsig = d_logsig + batch_idx * sig_len;
	const T* my_d_sig = d_sig + batch_idx * sig_len;
	const T* my_log_sig = log_sig + batch_idx * sig_len;
	T* my_P_all = P_all_buf + batch_idx * degree * sig_len;
	T* my_dP = dP_buf + batch_idx * 2 * sig_len;
	T* my_dP_next = my_dP + sig_len;

	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	tensor_exp_backprop_device<T>(
		my_d_logsig, my_d_sig, my_log_sig,
		my_P_all, my_dP, my_dP_next,
		degree, level_index_smem
	);
}

// =========================================================================
// Methods 1/2 kernels: expansion matrix multiply + tensor_exp
// =========================================================================

template<typename T>
__global__ void logsig_to_sig_m12_kernel(
	const T* __restrict__ coefs,
	T* __restrict__ out,
	T* __restrict__ expanded_buf,
	T* __restrict__ buff,
	const T* __restrict__ d_expand_mat,
	const uint64_t* __restrict__ d_level_index,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t m,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_coefs = coefs + batch_idx * m;
	T* my_out = out + batch_idx * sig_len;
	T* my_expanded = expanded_buf + batch_idx * sig_len;
	T* my_buff = buff + batch_idx * 2 * sig_len;

	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	for (uint64_t j = tid; j < sig_len; j += nthreads) {
		T acc = static_cast<T>(0);
		for (uint64_t i = 0; i < m; ++i)
			acc += d_expand_mat[j * m + i] * my_coefs[i];
		my_expanded[j] = acc;
	}
	__syncthreads();

	tensor_exp_device<T>(my_expanded, my_out, my_buff, degree, level_index_smem);
}

template<typename T>
__global__ void logsig_to_sig_m12_backprop_kernel(
	T* __restrict__ d_coefs,
	const T* __restrict__ d_sig,
	const T* __restrict__ coefs,
	const T* __restrict__ d_expand_mat,
	T* __restrict__ expanded_buf,
	T* __restrict__ d_expanded_buf,
	T* __restrict__ P_all_buf,
	T* __restrict__ dP_buf,
	const uint64_t* __restrict__ d_level_index,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t m,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_coefs = coefs + batch_idx * m;
	const T* my_d_sig = d_sig + batch_idx * sig_len;
	T* my_d_coefs = d_coefs + batch_idx * m;
	T* my_expanded = expanded_buf + batch_idx * sig_len;
	T* my_d_expanded = d_expanded_buf + batch_idx * sig_len;
	T* my_P_all = P_all_buf + batch_idx * degree * sig_len;
	T* my_dP = dP_buf + batch_idx * 2 * sig_len;
	T* my_dP_next = my_dP + sig_len;

	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	for (uint64_t j = tid; j < sig_len; j += nthreads) {
		T acc = static_cast<T>(0);
		for (uint64_t i = 0; i < m; ++i)
			acc += d_expand_mat[j * m + i] * my_coefs[i];
		my_expanded[j] = acc;
	}
	__syncthreads();

	tensor_exp_backprop_device<T>(
		my_d_expanded, my_d_sig, my_expanded,
		my_P_all, my_dP, my_dP_next,
		degree, level_index_smem
	);

	for (uint64_t i = tid; i < m; i += nthreads) {
		T acc = static_cast<T>(0);
		for (uint64_t j = 0; j < sig_len; ++j)
			acc += my_d_expanded[j] * d_expand_mat[j * m + i];
		my_d_coefs[i] = acc;
	}
	__syncthreads();
}

// =========================================================================
// Workspace
// =========================================================================

struct CUDAExpSigWorkspace {
	void* d_buff = nullptr;
	void* d_intermediates = nullptr;
	void* d_dS = nullptr;
	void* d_expanded = nullptr;
	void* d_d_expanded = nullptr;
	void* d_expand_mat = nullptr;
	size_t buff_bytes = 0;
	size_t intermediates_bytes = 0;
	size_t dS_bytes = 0;
	size_t expanded_bytes = 0;
	size_t d_expanded_bytes = 0;
	size_t expand_mat_bytes = 0;

	void ensure(void*& ptr, size_t& current, size_t need) {
		if (need > current) {
			if (ptr) { cudaFree(ptr); ptr = nullptr; current = 0; }
			CUDA_CHECK(cudaMalloc(&ptr, need));
			current = need;
		}
	}

	void ensure_forward(size_t need_buff) { ensure(d_buff, buff_bytes, need_buff); }
	void ensure_expanded(size_t need) { ensure(d_expanded, expanded_bytes, need); }
	void ensure_d_expanded(size_t need) { ensure(d_d_expanded, d_expanded_bytes, need); }
	void ensure_expand_mat(size_t need) { ensure(d_expand_mat, expand_mat_bytes, need); }

	void ensure_backward(size_t need_intermediates, size_t need_dS) {
		ensure(d_intermediates, intermediates_bytes, need_intermediates);
		ensure(d_dS, dS_bytes, need_dS);
	}

	void free() {
		auto f = [](void*& p, size_t& s) { if (p) { cudaFree(p); p = nullptr; s = 0; } };
		f(d_buff, buff_bytes);
		f(d_intermediates, intermediates_bytes);
		f(d_dS, dS_bytes);
		f(d_expanded, expanded_bytes);
		f(d_d_expanded, d_expanded_bytes);
		f(d_expand_mat, expand_mat_bytes);
	}

	~CUDAExpSigWorkspace() { free(); }
};

static CUDAExpSigWorkspace g_exp_workspace;
static std::mutex g_exp_workspace_mu;

void release_exp_sig_state() {
	std::lock_guard<std::mutex> lock(g_exp_workspace_mu);
	g_exp_workspace.free();
}

// =========================================================================
// Host-side forward launch
// =========================================================================

// Helper kernels for scalar_term=false staging.
template<typename T>
__global__ void exp_prepend_scalar_one_kernel(
	const T* __restrict__ in_stripped,
	T* __restrict__ out_full,
	uint64_t full_len,
	uint64_t total
) {
	for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		flat < total; flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t b = flat / full_len;
		const uint64_t i = flat - b * full_len;
		if (i == 0) out_full[flat] = static_cast<T>(1);
		else out_full[flat] = in_stripped[b * (full_len - 1) + (i - 1)];
	}
}

template<typename T>
__global__ void exp_strip_scalar_kernel(
	const T* __restrict__ in_full,
	T* __restrict__ out_stripped,
	uint64_t full_len,
	uint64_t total
) {
	const uint64_t stripped_len = full_len - 1;
	for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		flat < total; flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t b = flat / stripped_len;
		const uint64_t i = flat - b * stripped_len;
		out_stripped[flat] = in_full[b * full_len + i + 1];
	}
}

template<typename T>
__global__ void exp_prepend_zero_kernel(
	const T* __restrict__ in_stripped,   // stripped log-sig (log-sig[0]=0 is implicit)
	T* __restrict__ out_full,
	uint64_t full_len,
	uint64_t total
) {
	for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		flat < total; flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t b = flat / full_len;
		const uint64_t i = flat - b * full_len;
		if (i == 0) out_full[flat] = static_cast<T>(0);
		else out_full[flat] = in_stripped[b * (full_len - 1) + (i - 1)];
	}
}

template<typename T>
static void exp_stage_prepend_(const T* in_stripped, T* out_full, uint64_t batch_size, uint64_t full_len, bool prepend_one) {
	const unsigned int block = 256;
	const uint64_t total = batch_size * full_len;
	const unsigned int grid = static_cast<unsigned int>(std::min<uint64_t>(
		(total + block - 1) / block, CUDA_BATCH_GRID_LIMIT));
	if (prepend_one)
		exp_prepend_scalar_one_kernel<T><<<grid, block>>>(in_stripped, out_full, full_len, total);
	else
		exp_prepend_zero_kernel<T><<<grid, block>>>(in_stripped, out_full, full_len, total);
	check_cuda_error();
}

template<typename T>
static void exp_stage_strip_(const T* in_full, T* out_stripped, uint64_t batch_size, uint64_t full_len) {
	if (full_len == 1) return;
	const unsigned int block = 256;
	const uint64_t total = batch_size * (full_len - 1);
	const unsigned int grid = static_cast<unsigned int>(std::min<uint64_t>(
		(total + block - 1) / block, CUDA_BATCH_GRID_LIMIT));
	exp_strip_scalar_kernel<T><<<grid, block>>>(in_full, out_stripped, full_len, total);
	check_cuda_error();
}

// Core kernel launches factored out - same body as before; callers now handle
// scalar_term staging. This keeps scalar_term=true zero-overhead.
template<typename T>
void logsig_to_sig_cuda_core_(
	const T* log_sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method
);
template<typename T>
void logsig_to_sig_backprop_cuda_core_(
	const T* log_sig,
	T* d_logsig,
	const T* d_sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method
);

template<typename T>
void logsig_to_sig_cuda_(
	const T* log_sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method,
	bool scalar_term = true
) {
	if (method < 0 || method > 2)
		throw std::invalid_argument("logsig_to_sig_cuda: method must be 0, 1, or 2");
	if (dimension == 0)
		throw std::invalid_argument("logsig_to_sig_cuda received dimension 0");
	if (degree == 0)
		throw std::invalid_argument("logsig_to_sig_cuda received degree 0");

	if (scalar_term) {
		logsig_to_sig_cuda_core_<T>(log_sig, out, batch_size, dimension, degree, method);
		return;
	}


	const uint64_t full_len = host_sig_length(dimension, degree);
	CudaBuf<T> d_out_full(batch_size * full_len * sizeof(T));

	if (method == 0) {
		CudaBuf<T> d_ls_full(batch_size * full_len * sizeof(T));
		exp_stage_prepend_<T>(log_sig, d_ls_full.get(), batch_size, full_len, /*prepend_one=*/false);
		logsig_to_sig_cuda_core_<T>(d_ls_full.get(), d_out_full.get(), batch_size, dimension, degree, method);
	} else {
		logsig_to_sig_cuda_core_<T>(log_sig, d_out_full.get(), batch_size, dimension, degree, method);
	}
	exp_stage_strip_<T>(d_out_full.get(), out, batch_size, full_len);
}

template<typename T>
void logsig_to_sig_backprop_cuda_(
	const T* log_sig,
	T* d_logsig,
	const T* d_sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method,
	bool scalar_term = true
) {
	if (method < 0 || method > 2)
		throw std::invalid_argument("logsig_to_sig_backprop_cuda: method must be 0, 1, or 2");
	if (dimension == 0)
		throw std::invalid_argument("logsig_to_sig_backprop_cuda received dimension 0");
	if (degree == 0)
		throw std::invalid_argument("logsig_to_sig_backprop_cuda received degree 0");

	if (scalar_term) {
		logsig_to_sig_backprop_cuda_core_<T>(log_sig, d_logsig, d_sig, batch_size, dimension, degree, method);
		return;
	}

	// scalar_term=false. Python wrapper layouts:
	//   method==0: log_sig is sig-shaped stripped (prepend 0 to reach full); d_sig
	//              is sig-shaped stripped (prepend 0); d_logsig is sig-shaped stripped
	//              (compute into full buffer, strip afterwards).
	//   method==1/2: log_sig and d_logsig are log-sig-shaped (no scalar concept);
	//                only d_sig is stripped.
	const uint64_t full_len = host_sig_length(dimension, degree);

	CudaBuf<T> d_dsig_full(batch_size * full_len * sizeof(T));
	exp_stage_prepend_<T>(d_sig, d_dsig_full.get(), batch_size, full_len, /*prepend_one=*/false);

	if (method == 0) {
		CudaBuf<T> d_ls_full(batch_size * full_len * sizeof(T));
		exp_stage_prepend_<T>(log_sig, d_ls_full.get(), batch_size, full_len, /*prepend_one=*/false);
		CudaBuf<T> d_dlogsig_full(batch_size * full_len * sizeof(T));
		logsig_to_sig_backprop_cuda_core_<T>(d_ls_full.get(), d_dlogsig_full.get(), d_dsig_full.get(), batch_size, dimension, degree, method);
		exp_stage_strip_<T>(d_dlogsig_full.get(), d_logsig, batch_size, full_len);
	} else {
		logsig_to_sig_backprop_cuda_core_<T>(log_sig, d_logsig, d_dsig_full.get(), batch_size, dimension, degree, method);
	}
}

// Original forward implementation, renamed to _core_.
template<typename T>
void logsig_to_sig_cuda_core_(
	const T* log_sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);

	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	CudaBuf<uint64_t> d_level_index(level_index_bytes);
	CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	uint64_t m = 0;
	std::unique_ptr<T[]> h_expand;
	if (method != 0) {
		m = get_lyndon_count(dimension, degree);
		h_expand = std::make_unique<T[]>(sig_len * m);
		if constexpr (std::is_same_v<T, float>)
			build_expansion_matrix_f(h_expand.get(), sig_len, m, dimension, degree, method);
		else
			build_expansion_matrix_d(h_expand.get(), sig_len, m, dimension, degree, method);
	}

	std::lock_guard<std::mutex> lock(g_exp_workspace_mu);

	if (method == 0) {
		g_exp_workspace.ensure_forward(sizeof(T) * batch_size * 2 * sig_len);

		configure_dynamic_smem(
			logsig_to_sig_kernel<T>, smem_size, "CUDA log sig to sig");
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset);
			logsig_to_sig_kernel<T><<<batch_chunk.grid, threads, smem_size>>>(
				log_sig, out, static_cast<T*>(g_exp_workspace.d_buff),
				d_level_index.get(), degree, sig_len,
				batch_chunk.offset, batch_chunk.size
			);
			batch_offset += batch_chunk.size;
		}
	}
	else {
		size_t mat_bytes = sizeof(T) * sig_len * m;
		g_exp_workspace.ensure_expand_mat(mat_bytes);
		CUDA_CHECK(cudaMemcpy(g_exp_workspace.d_expand_mat, h_expand.get(), mat_bytes, cudaMemcpyHostToDevice));

		g_exp_workspace.ensure_forward(sizeof(T) * batch_size * 2 * sig_len);
		g_exp_workspace.ensure_expanded(sizeof(T) * batch_size * sig_len);

		configure_dynamic_smem(
			logsig_to_sig_m12_kernel<T>, smem_size,
			"CUDA log sig to sig projected");
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset);
			logsig_to_sig_m12_kernel<T><<<batch_chunk.grid, threads, smem_size>>>(
				log_sig, out,
				static_cast<T*>(g_exp_workspace.d_expanded),
				static_cast<T*>(g_exp_workspace.d_buff),
				static_cast<T*>(g_exp_workspace.d_expand_mat),
				d_level_index.get(), degree, sig_len, m,
				batch_chunk.offset, batch_chunk.size
			);
			batch_offset += batch_chunk.size;
		}
	}

	check_cuda_kernel_launch();
}

// =========================================================================
// Host-side backward launch
// =========================================================================

// Renamed: original backprop body, now called from the dispatcher.
template<typename T>
void logsig_to_sig_backprop_cuda_core_(
	const T* log_sig,
	T* d_logsig,
	const T* d_sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);

	if (degree == 1 && method == 0) {
		cudaMemcpy(d_logsig, d_sig, batch_size * sig_len * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaMemset2D(d_logsig, sig_len * sizeof(T), 0, sizeof(T), batch_size);
		return;
	}

	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	CudaBuf<uint64_t> d_level_index(level_index_bytes);
	CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	uint64_t m = 0;
	std::unique_ptr<T[]> h_expand;
	if (method != 0) {
		m = get_lyndon_count(dimension, degree);
		h_expand = std::make_unique<T[]>(sig_len * m);
		if constexpr (std::is_same_v<T, float>)
			build_expansion_matrix_f(h_expand.get(), sig_len, m, dimension, degree, method);
		else
			build_expansion_matrix_d(h_expand.get(), sig_len, m, dimension, degree, method);
	}

	std::lock_guard<std::mutex> lock(g_exp_workspace_mu);

	if (method == 0) {
		g_exp_workspace.ensure_backward(
			sizeof(T) * batch_size * degree * sig_len,   // P_all
			sizeof(T) * batch_size * 2 * sig_len         // dP + dP_next
		);

		configure_dynamic_smem(
			logsig_to_sig_backprop_kernel<T>, smem_size,
			"CUDA log sig to sig backprop");
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset);
			logsig_to_sig_backprop_kernel<T><<<
				batch_chunk.grid, threads, smem_size>>>(
					d_logsig, d_sig, log_sig,
					static_cast<T*>(g_exp_workspace.d_intermediates),
					static_cast<T*>(g_exp_workspace.d_dS),
					d_level_index.get(), degree, sig_len,
					batch_chunk.offset, batch_chunk.size
				);
			batch_offset += batch_chunk.size;
		}
	}
	else {
		size_t mat_bytes = sizeof(T) * sig_len * m;
		g_exp_workspace.ensure_expand_mat(mat_bytes);
		CUDA_CHECK(cudaMemcpy(g_exp_workspace.d_expand_mat, h_expand.get(), mat_bytes, cudaMemcpyHostToDevice));

		g_exp_workspace.ensure_expanded(sizeof(T) * batch_size * sig_len);
		g_exp_workspace.ensure_backward(
			sizeof(T) * batch_size * degree * sig_len,     // P_all
			sizeof(T) * batch_size * 2 * sig_len           // dP + dP_next
		);

		g_exp_workspace.ensure_d_expanded(sizeof(T) * batch_size * sig_len);

		configure_dynamic_smem(
			logsig_to_sig_m12_backprop_kernel<T>, smem_size,
			"CUDA log sig to sig projected backprop");
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset);
			logsig_to_sig_m12_backprop_kernel<T><<<
				batch_chunk.grid, threads, smem_size>>>(
					d_logsig, d_sig, log_sig,
					static_cast<T*>(g_exp_workspace.d_expand_mat),
					static_cast<T*>(g_exp_workspace.d_expanded),
					static_cast<T*>(g_exp_workspace.d_d_expanded),
					static_cast<T*>(g_exp_workspace.d_intermediates),
					static_cast<T*>(g_exp_workspace.d_dS),
					d_level_index.get(), degree, sig_len, m,
					batch_chunk.offset, batch_chunk.size
				);
			batch_offset += batch_chunk.size;
		}
	}

	check_cuda_kernel_launch();
}

// =========================================================================
// Exported C functions
// =========================================================================

#include "cu_macros.h"

extern "C" {


	CUSIG_API int logsig_to_sig_cuda_f(
		const float* log_sig, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_cuda_<float>(log_sig, out, batch_size, dimension, degree, method, scalar_term));
	}

	CUSIG_API int logsig_to_sig_cuda_d(
		const double* log_sig, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_cuda_<double>(log_sig, out, batch_size, dimension, degree, method, scalar_term));
	}


	CUSIG_API int logsig_to_sig_backprop_cuda_f(
		const float* log_sig, float* d_logsig, const float* d_sig,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_backprop_cuda_<float>(log_sig, d_logsig, d_sig, batch_size, dimension, degree, method, scalar_term));
	}

	CUSIG_API int logsig_to_sig_backprop_cuda_d(
		const double* log_sig, double* d_logsig, const double* d_sig,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_backprop_cuda_<double>(log_sig, d_logsig, d_sig, batch_size, dimension, degree, method, scalar_term));
	}

}
