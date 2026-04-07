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
#include "cu_atomic.h"

// =========================================================================
// Device function: tensor_exp (Horner scheme)
//
// Computes exp(log_sig) via: S = 1 + x/N, then S = 1 + (x/k)*S for k=N-1..1
// Each block handles one batch element. Threads cooperate on levels.
//
// log_sig:     input expanded log-signature (level 0 = 0)
// out:         output signature (level 0 = 1)
// buff:        scratch buffer of size sig_len
// degree:      truncation degree
// level_index: precomputed (degree+2 entries, in shared memory)
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

	// Initialize: out = 1 + x/N
	if (tid == 0) out[0] = static_cast<T>(1);
	T inv_k = static_cast<T>(1) / static_cast<T>(degree);
	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		out[i] = log_sig[i] * inv_k;
	__syncthreads();

	if (degree <= 1) return;

	// Horner iterations: k = N-1 down to 1
	for (int64_t k = static_cast<int64_t>(degree) - 1; k >= 1; --k) {
		inv_k = static_cast<T>(1) / static_cast<T>(k);

		// Compute buff[l] = sum_{l1+l2=l, l1>=1, l2>=1} (x[l1]*inv_k) * out[l2]
		for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
			const uint64_t target_start = level_index[target_level];
			const uint64_t target_size = level_index[target_level + 1] - target_start;

			// Zero buff at this level
			for (uint64_t i = tid; i < target_size; i += nthreads)
				buff[target_start + i] = static_cast<T>(0);
			__syncthreads();

			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
					const uint64_t l_idx = idx / right_size;
					const uint64_t r_idx = idx % right_size;
					buff[target_start + idx] += (log_sig[left_start + l_idx] * inv_k) * out[right_start + r_idx];
				}
				__syncthreads();
			}
		}

		// Update out: out[l] = x[l]/k + buff[l] for l >= 2, out[1] = x[1]/k
		const uint64_t level1_start = level_index[1];
		const uint64_t level1_size = level_index[2] - level1_start;
		for (uint64_t i = tid; i < level1_size; i += nthreads)
			out[level1_start + i] = log_sig[level1_start + i] * inv_k;

		for (uint64_t l = 2; l <= degree; ++l) {
			const uint64_t l_start = level_index[l];
			const uint64_t l_size = level_index[l + 1] - l_start;
			for (uint64_t i = tid; i < l_size; i += nthreads)
				out[l_start + i] = log_sig[l_start + i] * inv_k + buff[l_start + i];
		}
		__syncthreads();
	}
}

// =========================================================================
// Device function: tensor_exp_backprop
//
// Given dL/d(sig), computes dL/d(log_sig).
// Recomputes Horner intermediates internally.
//
// d_logsig:    output gradient w.r.t. log_sig
// d_sig:       input upstream gradient w.r.t. signature
// log_sig:     original log-signature input
// intermediates: scratch buffer of size (degree-1) * sig_len for S_2..S_N
// buff:        scratch buffer of size sig_len
// dS, dS_next: two sig_len buffers for gradient propagation
// degree, level_index: as above
// =========================================================================

template<typename T>
__device__ void tensor_exp_backprop_device(
	T* __restrict__ d_logsig,
	const T* __restrict__ d_sig,
	const T* __restrict__ log_sig,
	T* __restrict__ intermediates,
	T* __restrict__ buff,
	T* __restrict__ dS,
	T* __restrict__ dS_next,
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
		// exp(x) = 1 + x, so d_logsig = d_sig (levels 1+)
		for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
			d_logsig[i] = d_sig[i];
		__syncthreads();
		return;
	}

	const uint64_t num_intermediates = degree - 1;

	// Recompute S_N
	T* S_current = intermediates + (num_intermediates - 1) * sig_len;
	if (tid == 0) S_current[0] = static_cast<T>(1);
	T inv_k = static_cast<T>(1) / static_cast<T>(degree);
	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		S_current[i] = log_sig[i] * inv_k;
	__syncthreads();

	// Recompute S_{N-1}, ..., S_2
	for (int64_t k = static_cast<int64_t>(degree) - 1; k >= 2; --k) {
		T* S_prev = S_current;
		S_current = intermediates + (k - 2) * sig_len;
		inv_k = static_cast<T>(1) / static_cast<T>(k);

		// Compute S_k = 1 + (x/k) * S_{k+1}
		for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
			const uint64_t target_start = level_index[target_level];
			const uint64_t target_size = level_index[target_level + 1] - target_start;

			for (uint64_t i = tid; i < target_size; i += nthreads)
				buff[target_start + i] = static_cast<T>(0);
			__syncthreads();

			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
					const uint64_t l_idx = idx / right_size;
					const uint64_t r_idx = idx % right_size;
					buff[target_start + idx] += (log_sig[left_start + l_idx] * inv_k) * S_prev[right_start + r_idx];
				}
				__syncthreads();
			}
		}

		if (tid == 0) S_current[0] = static_cast<T>(1);
		const uint64_t l1_start = level_index[1];
		const uint64_t l1_size = level_index[2] - l1_start;
		for (uint64_t i = tid; i < l1_size; i += nthreads)
			S_current[l1_start + i] = log_sig[l1_start + i] * inv_k;

		for (uint64_t l = 2; l <= degree; ++l) {
			const uint64_t l_start = level_index[l];
			const uint64_t l_size = level_index[l + 1] - l_start;
			for (uint64_t i = tid; i < l_size; i += nthreads)
				S_current[l_start + i] = log_sig[l_start + i] * inv_k + buff[l_start + i];
		}
		__syncthreads();
	}

	// Copy d_sig to dS
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		dS[i] = d_sig[i];
	__syncthreads();

	// Backward: for k = 1, ..., N-1
	for (int64_t k = 1; k < static_cast<int64_t>(degree); ++k) {
		inv_k = static_cast<T>(1) / static_cast<T>(k);
		const T* S_kp1 = intermediates + (k - 1) * sig_len;

		// d_logsig += dS / k (additive term)
		for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
			d_logsig[i] += dS[i] * inv_k;
		__syncthreads();

		// Zero dS_next
		for (uint64_t i = tid; i < sig_len; i += nthreads)
			dS_next[i] = static_cast<T>(0);
		__syncthreads();

		// Backprop through tensor product
		for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
			const uint64_t target_start = level_index[target_level];

			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t left_size = level_index[left_level + 1] - left_start;
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				// d_logsig[l1] += sum_{l2} dS[l1+l2] * S_{k+1}[l2] * inv_k
				for (uint64_t l_idx = tid; l_idx < left_size; l_idx += nthreads) {
					T acc = static_cast<T>(0);
					for (uint64_t r_idx = 0; r_idx < right_size; ++r_idx) {
						acc += dS[target_start + l_idx * right_size + r_idx] * S_kp1[right_start + r_idx];
					}
					myAtomicAdd(&d_logsig[left_start + l_idx], acc * inv_k);
				}
				__syncthreads();

				// dS_next[l2] += sum_{l1} dS[l1+l2] * x[l1] * inv_k
				for (uint64_t r_idx = tid; r_idx < right_size; r_idx += nthreads) {
					T acc = static_cast<T>(0);
					for (uint64_t l_idx = 0; l_idx < left_size; ++l_idx) {
						acc += dS[target_start + l_idx * right_size + r_idx] * log_sig[left_start + l_idx];
					}
					myAtomicAdd(&dS_next[right_start + r_idx], acc * inv_k);
				}
				__syncthreads();
			}
		}

		// Swap dS and dS_next
		T* tmp = dS;
		dS = dS_next;
		dS_next = tmp;
		__syncthreads();
	}

	// Final step: k = N
	inv_k = static_cast<T>(1) / static_cast<T>(degree);
	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		d_logsig[i] += dS[i] * inv_k;
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
	uint64_t sig_len
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_log_sig = log_sig + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;
	T* my_buff = buff + batch_idx * sig_len;

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
	T* __restrict__ intermediates,
	T* __restrict__ buff,
	T* __restrict__ dS_buf,
	const uint64_t* __restrict__ d_level_index,
	uint64_t degree,
	uint64_t sig_len
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const uint64_t num_intermediates = degree - 1;

	T* my_d_logsig = d_logsig + batch_idx * sig_len;
	const T* my_d_sig = d_sig + batch_idx * sig_len;
	const T* my_log_sig = log_sig + batch_idx * sig_len;
	T* my_intermediates = intermediates + batch_idx * num_intermediates * sig_len;
	T* my_buff = buff + batch_idx * sig_len;
	T* my_dS = dS_buf + batch_idx * 2 * sig_len;
	T* my_dS_next = my_dS + sig_len;

	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	tensor_exp_backprop_device<T>(
		my_d_logsig, my_d_sig, my_log_sig,
		my_intermediates, my_buff, my_dS, my_dS_next,
		degree, level_index_smem
	);
}

// =========================================================================
// Workspace
// =========================================================================

struct CUDAExpSigWorkspace {
	void* d_buff = nullptr;
	void* d_intermediates = nullptr;
	void* d_dS = nullptr;
	size_t buff_bytes = 0;
	size_t intermediates_bytes = 0;
	size_t dS_bytes = 0;

	void ensure_forward(size_t need_buff) {
		if (need_buff > buff_bytes) {
			if (d_buff) cudaFree(d_buff);
			cudaMalloc(&d_buff, need_buff);
			buff_bytes = need_buff;
		}
	}

	void ensure_backward(size_t need_buff, size_t need_intermediates, size_t need_dS) {
		ensure_forward(need_buff);
		if (need_intermediates > intermediates_bytes) {
			if (d_intermediates) cudaFree(d_intermediates);
			cudaMalloc(&d_intermediates, need_intermediates);
			intermediates_bytes = need_intermediates;
		}
		if (need_dS > dS_bytes) {
			if (d_dS) cudaFree(d_dS);
			cudaMalloc(&d_dS, need_dS);
			dS_bytes = need_dS;
		}
	}

	void free() {
		if (d_buff) { cudaFree(d_buff); d_buff = nullptr; buff_bytes = 0; }
		if (d_intermediates) { cudaFree(d_intermediates); d_intermediates = nullptr; intermediates_bytes = 0; }
		if (d_dS) { cudaFree(d_dS); d_dS = nullptr; dS_bytes = 0; }
	}

	~CUDAExpSigWorkspace() { free(); }
};

static CUDAExpSigWorkspace g_exp_workspace;

// =========================================================================
// Host-side forward launch
// =========================================================================

template<typename T>
void logsig_to_sig_cuda_(
	const T* log_sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	if (method != 0)
		throw std::invalid_argument("logsig_to_sig_cuda: only method=0 is supported on CUDA");

	const uint64_t sig_len = host_sig_length(dimension, degree);

	if (degree == 0) {
		cudaMemset(out, 0, batch_size * sizeof(T));
		return;
	}

	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	uint64_t* d_level_index = nullptr;
	cudaMalloc(&d_level_index, level_index_bytes);
	cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);

	g_exp_workspace.ensure_forward(sizeof(T) * batch_size * sig_len);

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	logsig_to_sig_kernel<T><<<static_cast<unsigned int>(batch_size), threads, smem_size>>>(
		log_sig, out,
		static_cast<T*>(g_exp_workspace.d_buff),
		d_level_index, degree, sig_len
	);

	cudaFree(d_level_index);
	check_cuda_kernel_launch();
}

// =========================================================================
// Host-side backward launch
// =========================================================================

template<typename T>
void logsig_to_sig_backprop_cuda_(
	const T* log_sig,
	T* d_logsig,
	const T* d_sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	if (method != 0)
		throw std::invalid_argument("logsig_to_sig_backprop_cuda: only method=0 is supported on CUDA");

	const uint64_t sig_len = host_sig_length(dimension, degree);

	if (degree <= 1) {
		cudaMemcpy(d_logsig, d_sig, batch_size * sig_len * sizeof(T), cudaMemcpyDeviceToDevice);
		return;
	}

	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	uint64_t* d_level_index = nullptr;
	cudaMalloc(&d_level_index, level_index_bytes);
	cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);

	const uint64_t num_intermediates = degree - 1;
	g_exp_workspace.ensure_backward(
		sizeof(T) * batch_size * sig_len,
		sizeof(T) * batch_size * num_intermediates * sig_len,
		sizeof(T) * batch_size * 2 * sig_len
	);

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	logsig_to_sig_backprop_kernel<T><<<static_cast<unsigned int>(batch_size), threads, smem_size>>>(
		d_logsig, d_sig, log_sig,
		static_cast<T*>(g_exp_workspace.d_intermediates),
		static_cast<T*>(g_exp_workspace.d_buff),
		static_cast<T*>(g_exp_workspace.d_dS),
		d_level_index, degree, sig_len
	);

	cudaFree(d_level_index);
	check_cuda_kernel_launch();
}

// =========================================================================
// Exported C functions
// =========================================================================

#include "cu_macros.h"

extern "C" {

	CUSIG_API int logsig_to_sig_cuda_f(
		const float* log_sig, float* out,
		uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_cuda_<float>(log_sig, out, 1, dimension, degree, method));
	}

	CUSIG_API int logsig_to_sig_cuda_d(
		const double* log_sig, double* out,
		uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_cuda_<double>(log_sig, out, 1, dimension, degree, method));
	}

	CUSIG_API int batch_logsig_to_sig_cuda_f(
		const float* log_sig, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_cuda_<float>(log_sig, out, batch_size, dimension, degree, method));
	}

	CUSIG_API int batch_logsig_to_sig_cuda_d(
		const double* log_sig, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_cuda_<double>(log_sig, out, batch_size, dimension, degree, method));
	}

	CUSIG_API int logsig_to_sig_backprop_cuda_f(
		const float* log_sig, float* d_logsig, const float* d_sig,
		uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_backprop_cuda_<float>(log_sig, d_logsig, d_sig, 1, dimension, degree, method));
	}

	CUSIG_API int logsig_to_sig_backprop_cuda_d(
		const double* log_sig, double* d_logsig, const double* d_sig,
		uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_backprop_cuda_<double>(log_sig, d_logsig, d_sig, 1, dimension, degree, method));
	}

	CUSIG_API int batch_logsig_to_sig_backprop_cuda_f(
		const float* log_sig, float* d_logsig, const float* d_sig,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_backprop_cuda_<float>(log_sig, d_logsig, d_sig, batch_size, dimension, degree, method));
	}

	CUSIG_API int batch_logsig_to_sig_backprop_cuda_d(
		const double* log_sig, double* d_logsig, const double* d_sig,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CUSIG_SAFE_CALL(logsig_to_sig_backprop_cuda_<double>(log_sig, d_logsig, d_sig, batch_size, dimension, degree, method));
	}

}
