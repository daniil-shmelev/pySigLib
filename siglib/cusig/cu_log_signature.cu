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
#include "cu_utils.h"
#include "cu_log_signature.h"
#include "cu_log_sig_backprop.h"
#include "cu_log_sig_cache.h"

// =========================================================================
// Scratch buffer workspace - cached across calls to avoid per-call
// cudaMalloc/cudaFree of large buffers
// =========================================================================

struct CUDALogSigWorkspace {
	void* d_temp = nullptr;
	void* d_buff1 = nullptr;
	void* d_buff2 = nullptr;
	size_t temp_bytes = 0;
	size_t buff1_bytes = 0;
	size_t buff2_bytes = 0;

	void ensure(size_t need_temp, size_t need_buff1, size_t need_buff2) {
		if (need_temp > temp_bytes) {
			if (d_temp) { cudaFree(d_temp); d_temp = nullptr; temp_bytes = 0; }
			CUDA_CHECK(cudaMalloc(&d_temp, need_temp));
			temp_bytes = need_temp;
		}
		if (need_buff1 > buff1_bytes) {
			if (d_buff1) { cudaFree(d_buff1); d_buff1 = nullptr; buff1_bytes = 0; }
			CUDA_CHECK(cudaMalloc(&d_buff1, need_buff1));
			buff1_bytes = need_buff1;
		}
		if (need_buff2 > buff2_bytes) {
			if (d_buff2) { cudaFree(d_buff2); d_buff2 = nullptr; buff2_bytes = 0; }
			CUDA_CHECK(cudaMalloc(&d_buff2, need_buff2));
			buff2_bytes = need_buff2;
		}
	}

	void free() {
		if (d_temp) { cudaFree(d_temp); d_temp = nullptr; temp_bytes = 0; }
		if (d_buff1) { cudaFree(d_buff1); d_buff1 = nullptr; buff1_bytes = 0; }
		if (d_buff2) { cudaFree(d_buff2); d_buff2 = nullptr; buff2_bytes = 0; }
	}

	~CUDALogSigWorkspace() { free(); }
};

static CUDALogSigWorkspace g_workspace;
static std::mutex g_workspace_mu;

void free_cuda_log_sig_workspace_() {
	std::lock_guard<std::mutex> lock(g_workspace_mu);
	g_workspace.free();
}

// =========================================================================
// CUDA sig_to_log_sig kernel (method 0 - expanded tensor log)
//
// Each block handles one batch element.
// =========================================================================

template<typename T>
__global__ void sig_to_log_sig_kernel(
	const T* __restrict__ sig,           // [batch_size * sig_len]
	T* __restrict__ out,                 // [batch_size * sig_len]
	T* __restrict__ buff1,               // [batch_size * buff1_len]
	T* __restrict__ buff2,               // [batch_size * sig_len]
	const uint64_t* __restrict__ d_level_index,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t buff1_len
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	T* my_out = out + batch_idx * sig_len;
	T* my_buff1 = buff1 + batch_idx * buff1_len;
	T* my_buff2 = buff2 + batch_idx * sig_len;
	const T* my_sig = sig + batch_idx * sig_len;

	// Load level_index into shared memory
	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);

	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// Copy sig to out
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		my_out[i] = my_sig[i];
	__syncthreads();

	// Compute tensor log in-place
	tensor_log_inplace_device<T>(my_out, my_buff1, my_buff2, degree, level_index_smem);
}

// =========================================================================
// Host-side sig_to_log_sig core launch (method 0)
// =========================================================================

// Helper kernels: scalar_term=false staging between full and stripped layouts.
// These run once per call, only on the scalar_term=false branch - zero cost
// when scalar_term=true.
template<typename T>
__global__ void prepend_scalar_one_kernel(
	const T* __restrict__ in_stripped,   // [batch, full_len-1]
	T* __restrict__ out_full,            // [batch, full_len]
	uint64_t full_len
) {
	const uint64_t b = blockIdx.y;
	const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= full_len) return;
	T* dst = out_full + b * full_len;
	if (i == 0) dst[0] = static_cast<T>(1);
	else dst[i] = in_stripped[b * (full_len - 1) + (i - 1)];
}

template<typename T>
__global__ void strip_scalar_kernel(
	const T* __restrict__ in_full,       // [batch, full_len]
	T* __restrict__ out_stripped,        // [batch, full_len-1]
	uint64_t full_len
) {
	const uint64_t b = blockIdx.y;
	const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i + 1 >= full_len) return;
	out_stripped[b * (full_len - 1) + i] = in_full[b * full_len + (i + 1)];
}

template<typename T>
static void stage_prepend_(const T* in_stripped, T* out_full, uint64_t batch_size, uint64_t full_len) {
	const unsigned int block = 256;
	const unsigned int grid_x = static_cast<unsigned int>((full_len + block - 1) / block);
	dim3 grid(grid_x, static_cast<unsigned int>(batch_size));
	prepend_scalar_one_kernel<T><<<grid, block>>>(in_stripped, out_full, full_len);
}

template<typename T>
static void stage_strip_(const T* in_full, T* out_stripped, uint64_t batch_size, uint64_t full_len) {
	const unsigned int block = 256;
	const unsigned int grid_x = static_cast<unsigned int>((full_len + block - 1) / block);
	dim3 grid(grid_x, static_cast<unsigned int>(batch_size));
	strip_scalar_kernel<T><<<grid, block>>>(in_full, out_stripped, full_len);
}

template<typename T>
void sig_to_log_sig_cuda_core_(
	const T* sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);

	if (degree == 0) {
		cudaMemset(out, 0, batch_size * sizeof(T));
		return;
	}

	// Build level_index on host and copy to device
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);

	CudaBuf<uint64_t> d_level_index(level_index_bytes);
	CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));

	// Allocate scratch buffers via workspace cache
	const uint64_t buff1_len = degree >= 2 ? host_sig_length(dimension, degree - 1) : 1;

	std::lock_guard<std::mutex> lock(g_workspace_mu);
	g_workspace.ensure(0, sizeof(T) * batch_size * buff1_len, sizeof(T) * batch_size * sig_len);

	// Choose threads per block based on largest level size
	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	// Shared memory: level_index only
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_to_log_sig_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig, out,
		static_cast<T*>(g_workspace.d_buff1),
		static_cast<T*>(g_workspace.d_buff2),
		d_level_index.get(), degree, sig_len, buff1_len
	);

	check_cuda_kernel_launch();
}

// =========================================================================
// CUDA sig_to_log_sig method 1 kernel (Lyndon words)
//
// Phase 1: Copy sig -> temp, compute tensor log in-place
// Phase 2: Gather - out[i] = temp[lyndon_idx[i]]
// =========================================================================

template<typename T>
__global__ void sig_to_log_sig_m1_kernel(
	const T* __restrict__ sig,
	T* __restrict__ out,
	T* __restrict__ temp,
	T* __restrict__ buff1,
	T* __restrict__ buff2,
	const uint64_t* __restrict__ d_level_index,
	const uint64_t* __restrict__ d_lyndon_idx,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t buff1_len,
	uint64_t log_sig_len
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	T* my_temp = temp + batch_idx * sig_len;
	T* my_buff1 = buff1 + batch_idx * buff1_len;
	T* my_buff2 = buff2 + batch_idx * sig_len;
	T* my_out = out + batch_idx * log_sig_len;
	const T* my_sig = sig + batch_idx * sig_len;

	// Load level_index into shared memory
	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);

	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// Copy sig to temp
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		my_temp[i] = my_sig[i];
	__syncthreads();

	// Compute tensor log in-place on temp
	tensor_log_inplace_device<T>(my_temp, my_buff1, my_buff2, degree, level_index_smem);

	// Gather: out[i] = temp[lyndon_idx[i]]
	for (uint64_t i = tid; i < log_sig_len; i += nthreads)
		my_out[i] = my_temp[d_lyndon_idx[i]];
}

// =========================================================================
// CUDA sig_to_log_sig method 2 kernel (Lyndon basis)
//
// Phase 1: Copy sig -> temp, compute tensor log in-place
// Phase 2: Gather - out[i] = temp[lyndon_idx[i]]
// Phase 3: Apply sparse lower-triangular matrix multiply (parallel)
//          Copy gathered values to temp scratch, then:
//          out[i] = temp[i] + sum_j(sparse_mat[i][j] * temp[j])
// =========================================================================

template<typename T>
__global__ void sig_to_log_sig_m2_kernel(
	const T* __restrict__ sig,
	T* __restrict__ out,
	T* __restrict__ temp,
	T* __restrict__ buff1,
	T* __restrict__ buff2,
	const uint64_t* __restrict__ d_level_index,
	const uint64_t* __restrict__ d_lyndon_idx,
	const int* __restrict__ d_sparse_vals,
	const uint64_t* __restrict__ d_sparse_cols,
	const uint64_t* __restrict__ d_sparse_row_ptr,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t buff1_len,
	uint64_t log_sig_len
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	T* my_temp = temp + batch_idx * sig_len;
	T* my_buff1 = buff1 + batch_idx * buff1_len;
	T* my_buff2 = buff2 + batch_idx * sig_len;
	T* my_out = out + batch_idx * log_sig_len;
	const T* my_sig = sig + batch_idx * sig_len;

	// Load level_index into shared memory
	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);

	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// Copy sig to temp
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		my_temp[i] = my_sig[i];
	__syncthreads();

	// Compute tensor log in-place on temp
	tensor_log_inplace_device<T>(my_temp, my_buff1, my_buff2, degree, level_index_smem);

	// Gather: out[i] = temp[lyndon_idx[i]]
	for (uint64_t i = tid; i < log_sig_len; i += nthreads)
		my_out[i] = my_temp[d_lyndon_idx[i]];
	__syncthreads();

	// Copy gathered values to temp scratch (reuse my_temp since tensor log is done)
	// so we can read from scratch while writing to out in parallel
	for (uint64_t i = tid; i < log_sig_len; i += nthreads)
		my_temp[i] = my_out[i];
	__syncthreads();

	// Apply inverse projection matrix in parallel
	// out[i] = temp[i] + sum_j mat[i][j] * temp[j]
	for (uint64_t i = tid; i < log_sig_len; i += nthreads) {
		uint64_t row_start = d_sparse_row_ptr[i];
		uint64_t row_end = d_sparse_row_ptr[i + 1];
		T acc = static_cast<T>(0);
		for (uint64_t k = row_start; k < row_end; ++k) {
			acc += static_cast<T>(d_sparse_vals[k]) * my_temp[d_sparse_cols[k]];
		}
		my_out[i] = my_temp[i] + acc;
	}
}

// =========================================================================
// Host-side method 1 launch
// =========================================================================

template<typename T>
void sig_to_log_sig_cuda_m1_core_(
	const T* sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	const auto& cache = get_cuda_log_sig_cache(dimension, degree);
	const uint64_t sig_len = cache.sig_len;
	const uint64_t buff1_len = cache.buff1_len;
	const uint64_t log_sig_len = cache.log_sig_len;

	std::lock_guard<std::mutex> lock(g_workspace_mu);
	g_workspace.ensure(
		sizeof(T) * batch_size * sig_len,
		sizeof(T) * batch_size * buff1_len,
		sizeof(T) * batch_size * sig_len
	);

	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_to_log_sig_m1_kernel<T><<<static_cast<unsigned int>(batch_size), cache.threads_per_block, smem_size>>>(
		sig, out,
		static_cast<T*>(g_workspace.d_temp),
		static_cast<T*>(g_workspace.d_buff1),
		static_cast<T*>(g_workspace.d_buff2),
		cache.d_level_index,
		cache.d_lyndon_idx, degree, sig_len, buff1_len, log_sig_len
	);

	check_cuda_kernel_launch();
}

// =========================================================================
// Host-side method 2 launch
// =========================================================================

template<typename T>
void sig_to_log_sig_cuda_m2_core_(
	const T* sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	const auto& cache = get_cuda_log_sig_cache(dimension, degree, 2);
	const uint64_t sig_len = cache.sig_len;
	const uint64_t buff1_len = cache.buff1_len;
	const uint64_t log_sig_len = cache.log_sig_len;

	std::lock_guard<std::mutex> lock(g_workspace_mu);
	g_workspace.ensure(
		sizeof(T) * batch_size * sig_len,
		sizeof(T) * batch_size * buff1_len,
		sizeof(T) * batch_size * sig_len
	);

	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_to_log_sig_m2_kernel<T><<<static_cast<unsigned int>(batch_size), cache.threads_per_block, smem_size>>>(
		sig, out,
		static_cast<T*>(g_workspace.d_temp),
		static_cast<T*>(g_workspace.d_buff1),
		static_cast<T*>(g_workspace.d_buff2),
		cache.d_level_index,
		cache.d_lyndon_idx,
		cache.d_sparse_vals, cache.d_sparse_cols, cache.d_sparse_row_ptr,
		degree, sig_len, buff1_len, log_sig_len
	);

	check_cuda_kernel_launch();
}

// =========================================================================
// Backprop workspace - cached across calls
// =========================================================================

struct CUDALogSigBackpropWorkspace {
	void* d_buf = nullptr;
	size_t buf_bytes = 0;
	void* d_derivs = nullptr;
	size_t derivs_bytes = 0;

	void ensure(size_t need_buf, size_t need_derivs) {
		if (need_buf > buf_bytes) {
			if (d_buf) { cudaFree(d_buf); d_buf = nullptr; buf_bytes = 0; }
			CUDA_CHECK(cudaMalloc(&d_buf, need_buf));
			buf_bytes = need_buf;
		}
		if (need_derivs > derivs_bytes) {
			if (d_derivs) { cudaFree(d_derivs); d_derivs = nullptr; derivs_bytes = 0; }
			CUDA_CHECK(cudaMalloc(&d_derivs, need_derivs));
			derivs_bytes = need_derivs;
		}
	}

	void free() {
		if (d_buf) { cudaFree(d_buf); d_buf = nullptr; buf_bytes = 0; }
		if (d_derivs) { cudaFree(d_derivs); d_derivs = nullptr; derivs_bytes = 0; }
	}

	~CUDALogSigBackpropWorkspace() { free(); }
};

static CUDALogSigBackpropWorkspace g_bp_workspace;
static std::mutex g_bp_workspace_mu;

void free_cuda_log_sig_backprop_workspace_() {
	std::lock_guard<std::mutex> lock(g_bp_workspace_mu);
	g_bp_workspace.free();
}

void release_log_sig_state() {
	free_cuda_log_sig_workspace_();
	free_cuda_log_sig_backprop_workspace_();
}

// =========================================================================
// CUDA sig_to_log_sig_backprop kernel (method 0 - expanded tensor log)
//
// Each block handles one batch element.
// Scratch per element: sig_copy(sig_len) + partial_logs((degree-1)*buff1_len)
//                    + other_derivs(sig_len) + buff1(buff1_len) + buff2(sig_len)
// =========================================================================

template<typename T>
__global__ void sig_to_log_sig_backprop_kernel(
	const T* __restrict__ sig,
	T* __restrict__ out,
	T* __restrict__ derivs,
	T* __restrict__ scratch,
	const uint64_t* __restrict__ d_level_index,
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t buff1_len,
	uint64_t scratch_per_element
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_sig = sig + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;
	T* my_derivs = derivs + batch_idx * sig_len;
	T* my_scratch = scratch + batch_idx * scratch_per_element;

	// Partition scratch: sig_copy | partial_logs | other_derivs | buff1 | buff2
	T* sig_copy = my_scratch;
	T* partial_logs = sig_copy + sig_len;
	T* other_derivs = partial_logs + (degree - 1) * buff1_len;
	T* buff1 = other_derivs + sig_len;
	T* buff2 = buff1 + buff1_len;

	// Load level_index into shared memory; reduction buffer follows
	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	T* reduction_buf = reinterpret_cast<T*>(smem + (degree + 2) * sizeof(uint64_t));

	tensor_log_backprop_device<T>(
		my_out, my_derivs, other_derivs, my_sig,
		sig_copy, partial_logs, buff1, buff2,
		dimension, degree, sig_len, buff1_len, level_index_smem,
		reduction_buf
	);
}

// =========================================================================
// Host-side sig_to_log_sig_backprop core launch (method 0)
// =========================================================================

template<typename T>
void sig_to_log_sig_backprop_cuda_core_(
	const T* sig,
	T* out,
	const T* log_sig_derivs,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);

	if (degree <= 1) {
		cudaMemcpy(out, log_sig_derivs, batch_size * sig_len * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaMemset2D(out, sig_len * sizeof(T), 0, sizeof(T), batch_size);
		return;
	}

	const uint64_t buff1_len = host_sig_length(dimension, degree - 1);

	// Build level_index
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);
	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);

	CudaBuf<uint64_t> d_level_index(level_index_bytes);
	CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));

	// Scratch per element: sig_copy + partial_logs + other_derivs + buff1 + buff2
	const uint64_t scratch_per_element =
		sig_len +                    // sig_copy
		(degree - 1) * buff1_len +   // partial_logs
		sig_len +                    // other_derivs
		buff1_len +                  // buff1
		sig_len;                     // buff2

	const size_t derivs_size = sizeof(T) * batch_size * sig_len;

	std::lock_guard<std::mutex> lock(g_bp_workspace_mu);
	g_bp_workspace.ensure(sizeof(T) * batch_size * scratch_per_element, derivs_size);

	CUDA_CHECK(cudaMemcpy(g_bp_workspace.d_derivs, log_sig_derivs, derivs_size, cudaMemcpyDeviceToDevice));

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);
	size_t smem_size = (degree + 2) * sizeof(uint64_t) + threads_per_block * sizeof(T);

	sig_to_log_sig_backprop_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig, out, static_cast<T*>(g_bp_workspace.d_derivs),
		static_cast<T*>(g_bp_workspace.d_buf),
		d_level_index.get(), dimension, degree, sig_len, buff1_len, scratch_per_element
	);

	check_cuda_kernel_launch();
}

// =========================================================================
// CUDA sig_to_log_sig_backprop method 1 kernel (Lyndon words)
//
// Phase 1: Scatter log_sig_derivs to expanded form via lyndon_idx
// Phase 2: Run tensor_log_backprop on expanded form
// =========================================================================

template<typename T>
__global__ void sig_to_log_sig_backprop_m1_kernel(
	const T* __restrict__ sig,
	T* __restrict__ out,
	const T* __restrict__ log_sig_derivs,
	T* __restrict__ derivs_expanded,
	T* __restrict__ scratch,
	const uint64_t* __restrict__ d_level_index,
	const uint64_t* __restrict__ d_lyndon_idx,
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t buff1_len,
	uint64_t log_sig_len,
	uint64_t scratch_per_element
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_sig = sig + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;
	const T* my_log_derivs = log_sig_derivs + batch_idx * log_sig_len;
	T* my_derivs = derivs_expanded + batch_idx * sig_len;
	T* my_scratch = scratch + batch_idx * scratch_per_element;

	T* sig_copy = my_scratch;
	T* partial_logs = sig_copy + sig_len;
	T* other_derivs = partial_logs + (degree - 1) * buff1_len;
	T* buff1 = other_derivs + sig_len;
	T* buff2 = buff1 + buff1_len;

	// Load level_index into shared memory; reduction buffer follows
	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// Zero expanded derivs, then scatter from log_sig_derivs
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		my_derivs[i] = static_cast<T>(0);
	__syncthreads();

	for (uint64_t i = tid; i < log_sig_len; i += nthreads)
		my_derivs[d_lyndon_idx[i]] = my_log_derivs[i];
	__syncthreads();

	T* reduction_buf = reinterpret_cast<T*>(smem + (degree + 2) * sizeof(uint64_t));

	tensor_log_backprop_device<T>(
		my_out, my_derivs, other_derivs, my_sig,
		sig_copy, partial_logs, buff1, buff2,
		dimension, degree, sig_len, buff1_len, level_index_smem,
		reduction_buf
	);
}

// =========================================================================
// Host-side method 1 backprop launch
// =========================================================================

template<typename T>
void sig_to_log_sig_backprop_cuda_m1_core_(
	const T* sig,
	T* out,
	const T* log_sig_derivs,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	const auto& cache = get_cuda_log_sig_cache(dimension, degree);
	const uint64_t sig_len = cache.sig_len;
	const uint64_t buff1_len = cache.buff1_len;
	const uint64_t log_sig_len = cache.log_sig_len;

	const uint64_t scratch_per_element =
		sig_len + (degree > 1 ? (degree - 1) : 1) * buff1_len + sig_len + buff1_len + sig_len;

	const size_t derivs_size = sizeof(T) * batch_size * sig_len;

	std::lock_guard<std::mutex> lock(g_bp_workspace_mu);
	g_bp_workspace.ensure(sizeof(T) * batch_size * scratch_per_element, derivs_size);

	size_t smem_size = (degree + 2) * sizeof(uint64_t) + cache.threads_per_block * sizeof(T);

	sig_to_log_sig_backprop_m1_kernel<T><<<static_cast<unsigned int>(batch_size), cache.threads_per_block, smem_size>>>(
		sig, out, log_sig_derivs, static_cast<T*>(g_bp_workspace.d_derivs),
		static_cast<T*>(g_bp_workspace.d_buf),
		cache.d_level_index, cache.d_lyndon_idx,
		dimension, degree, sig_len, buff1_len, log_sig_len, scratch_per_element
	);

	check_cuda_kernel_launch();
}

// =========================================================================
// CUDA sig_to_log_sig_backprop method 2 kernel (Lyndon basis)
//
// Phase 1: Apply transpose sparse matrix to log_sig_derivs
// Phase 2: Scatter to expanded form via lyndon_idx
// Phase 3: Run tensor_log_backprop on expanded form
// =========================================================================

template<typename T>
__global__ void sig_to_log_sig_backprop_m2_kernel(
	const T* __restrict__ sig,
	T* __restrict__ out,
	const T* __restrict__ log_sig_derivs,
	T* __restrict__ derivs_expanded,
	T* __restrict__ scratch,
	const uint64_t* __restrict__ d_level_index,
	const uint64_t* __restrict__ d_lyndon_idx,
	const int* __restrict__ d_sparse_vals_t,
	const uint64_t* __restrict__ d_sparse_cols_t,
	const uint64_t* __restrict__ d_sparse_row_ptr_t,
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t buff1_len,
	uint64_t log_sig_len,
	uint64_t scratch_per_element
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_sig = sig + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;
	const T* my_log_derivs = log_sig_derivs + batch_idx * log_sig_len;
	T* my_derivs = derivs_expanded + batch_idx * sig_len;
	T* my_scratch = scratch + batch_idx * scratch_per_element;

	T* sig_copy = my_scratch;
	T* partial_logs = sig_copy + sig_len;
	T* other_derivs = partial_logs + (degree - 1) * buff1_len;
	T* buff1 = other_derivs + sig_len;
	T* buff2 = buff1 + buff1_len;

	// Load level_index into shared memory; reduction buffer follows
	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// Apply transpose of inverse projection matrix via parallel gather,
	// and scatter to expanded form in one step.
	// result[j] = log_sig_derivs[j] + sum_k M^T[j][k] * log_sig_derivs[k]
	// Then: derivs_expanded[lyndon_idx[j]] = result[j]
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		my_derivs[i] = static_cast<T>(0);
	__syncthreads();

	for (uint64_t j = tid; j < log_sig_len; j += nthreads) {
		T acc = my_log_derivs[j];
		uint64_t row_start = d_sparse_row_ptr_t[j];
		uint64_t row_end = d_sparse_row_ptr_t[j + 1];
		for (uint64_t k = row_start; k < row_end; ++k) {
			acc += static_cast<T>(d_sparse_vals_t[k]) * my_log_derivs[d_sparse_cols_t[k]];
		}
		my_derivs[d_lyndon_idx[j]] = acc;
	}
	__syncthreads();

	T* reduction_buf = reinterpret_cast<T*>(smem + (degree + 2) * sizeof(uint64_t));

	tensor_log_backprop_device<T>(
		my_out, my_derivs, other_derivs, my_sig,
		sig_copy, partial_logs, buff1, buff2,
		dimension, degree, sig_len, buff1_len, level_index_smem,
		reduction_buf
	);
}

// =========================================================================
// Host-side method 2 backprop launch
// =========================================================================

template<typename T>
void sig_to_log_sig_backprop_cuda_m2_core_(
	const T* sig,
	T* out,
	const T* log_sig_derivs,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree
) {
	const auto& cache = get_cuda_log_sig_cache(dimension, degree, 2);
	const uint64_t sig_len = cache.sig_len;
	const uint64_t buff1_len = cache.buff1_len;
	const uint64_t log_sig_len = cache.log_sig_len;

	const uint64_t scratch_per_element =
		sig_len + (degree > 1 ? (degree - 1) : 1) * buff1_len + sig_len + buff1_len + sig_len;

	const size_t derivs_size = sizeof(T) * batch_size * sig_len;

	std::lock_guard<std::mutex> lock(g_bp_workspace_mu);
	g_bp_workspace.ensure(sizeof(T) * batch_size * scratch_per_element, derivs_size);

	size_t smem_size = (degree + 2) * sizeof(uint64_t) + cache.threads_per_block * sizeof(T);

	sig_to_log_sig_backprop_m2_kernel<T><<<static_cast<unsigned int>(batch_size), cache.threads_per_block, smem_size>>>(
		sig, out, log_sig_derivs, static_cast<T*>(g_bp_workspace.d_derivs),
		static_cast<T*>(g_bp_workspace.d_buf),
		cache.d_level_index, cache.d_lyndon_idx,
		cache.d_sparse_vals_t, cache.d_sparse_cols_t, cache.d_sparse_row_ptr_t,
		dimension, degree, sig_len, buff1_len, log_sig_len, scratch_per_element
	);

	check_cuda_kernel_launch();
}

// =========================================================================
// Backprop method dispatch
// =========================================================================

template<typename T>
void sig_to_log_sig_backprop_cuda_(
	const T* sig,
	T* out,
	const T* log_sig_derivs,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method,
	bool scalar_term = true
) {
	if (dimension == 0) throw std::invalid_argument("sig_to_log_sig_backprop_cuda received dimension 0");
	if (degree == 0) throw std::invalid_argument("sig_to_log_sig_backprop_cuda received degree 0");

	if (scalar_term) {
		// Hot path: unchanged.
		if (method == 0) {
			sig_to_log_sig_backprop_cuda_core_<T>(sig, out, log_sig_derivs, batch_size, dimension, degree);
		}
		else if (method == 1) {
			sig_to_log_sig_backprop_cuda_m1_core_<T>(sig, out, log_sig_derivs, batch_size, dimension, degree);
		}
		else if (method == 2) {
			sig_to_log_sig_backprop_cuda_m2_core_<T>(sig, out, log_sig_derivs, batch_size, dimension, degree);
		}
		else {
			throw std::invalid_argument("sig_to_log_sig_backprop_cuda: method must be 0, 1, or 2");
		}
		return;
	}

	// scalar_term=false for backprop. The Python wrapper passes:
	//   sig: FULL length (always)
	//   log_sig_derivs: sig-shaped stripped (method=0) or log-sig-shaped (method>0)
	//   out (d_sig): FULL length (Python strips afterward)
	const uint64_t full_len = host_sig_length(dimension, degree);

	if (method == 0) {
		// log_sig_derivs is stripped; stage into a full-size buffer.
		CudaBuf<T> d_lsd_full(batch_size * full_len * sizeof(T));
		// d(log_sig)/d... leading slot is 0 (log_sig[0] is constant).
		CUDA_CHECK(cudaMemset(d_lsd_full.get(), 0, batch_size * full_len * sizeof(T)));
		stage_prepend_<T>(log_sig_derivs, d_lsd_full.get(), batch_size, full_len);
		sig_to_log_sig_backprop_cuda_core_<T>(sig, out, d_lsd_full.get(), batch_size, dimension, degree);
	}
	else if (method == 1) {
		sig_to_log_sig_backprop_cuda_m1_core_<T>(sig, out, log_sig_derivs, batch_size, dimension, degree);
	}
	else if (method == 2) {
		sig_to_log_sig_backprop_cuda_m2_core_<T>(sig, out, log_sig_derivs, batch_size, dimension, degree);
	}
	else {
		throw std::invalid_argument("sig_to_log_sig_backprop_cuda: method must be 0, 1, or 2");
	}
}

// =========================================================================
// Method dispatch
// =========================================================================

template<typename T>
void sig_to_log_sig_cuda_(
	const T* sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int method,
	bool scalar_term = true
) {
	if (dimension == 0) throw std::invalid_argument("sig_to_log_sig_cuda received dimension 0");
	if (degree == 0) throw std::invalid_argument("sig_to_log_sig_cuda received degree 0");

	if (scalar_term) {
		// Hot path: unchanged.
		if (method == 0) {
			sig_to_log_sig_cuda_core_<T>(sig, out, batch_size, dimension, degree);
		}
		else if (method == 1) {
			sig_to_log_sig_cuda_m1_core_<T>(sig, out, batch_size, dimension, degree);
		}
		else if (method == 2) {
			sig_to_log_sig_cuda_m2_core_<T>(sig, out, batch_size, dimension, degree);
		}
		else {
			throw std::invalid_argument("sig_to_log_sig_cuda: method must be 0, 1, or 2");
		}
		return;
	}

	// scalar_term=false: stage input/output through full-sized buffers.
	const uint64_t full_len = host_sig_length(dimension, degree);
	CudaBuf<T> d_sig_full(batch_size * full_len * sizeof(T));
	stage_prepend_<T>(sig, d_sig_full.get(), batch_size, full_len);

	if (method == 0) {
		// Output is sig-shaped. Stage through a full-sized output buffer.
		CudaBuf<T> d_out_full(batch_size * full_len * sizeof(T));
		sig_to_log_sig_cuda_core_<T>(d_sig_full.get(), d_out_full.get(), batch_size, dimension, degree);
		stage_strip_<T>(d_out_full.get(), out, batch_size, full_len);
	}
	else if (method == 1) {
		// Output is log-sig-shaped (no scalar concept); write directly.
		sig_to_log_sig_cuda_m1_core_<T>(d_sig_full.get(), out, batch_size, dimension, degree);
	}
	else if (method == 2) {
		sig_to_log_sig_cuda_m2_core_<T>(d_sig_full.get(), out, batch_size, dimension, degree);
	}
	else {
		throw std::invalid_argument("sig_to_log_sig_cuda: method must be 0, 1, or 2");
	}
}

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#include "cu_macros.h"

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {


	CUSIG_API int sig_to_log_sig_cuda_f(
		const float* sig, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_to_log_sig_cuda_<float>(sig, out, batch_size, dimension, degree, method, scalar_term));
	}

	CUSIG_API int sig_to_log_sig_cuda_d(
		const double* sig, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_to_log_sig_cuda_<double>(sig, out, batch_size, dimension, degree, method, scalar_term));
	}


	CUSIG_API int sig_to_log_sig_backprop_cuda_f(
		const float* sig, float* out, const float* log_sig_derivs,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_to_log_sig_backprop_cuda_<float>(sig, out, log_sig_derivs, batch_size, dimension, degree, method, scalar_term));
	}

	CUSIG_API int sig_to_log_sig_backprop_cuda_d(
		const double* sig, double* out, const double* log_sig_derivs,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_to_log_sig_backprop_cuda_<double>(sig, out, log_sig_derivs, batch_size, dimension, degree, method, scalar_term));
	}
}
