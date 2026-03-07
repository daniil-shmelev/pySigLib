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
#include "cu_log_signature.h"
#include "cu_log_sig_cache.h"

// =========================================================================
// CUDA sig_to_log_sig kernel
//
// Computes the expanded tensor logarithm of a truncated signature.
// Each block handles one batch element.
// Requires two scratch buffers (buff1, buff2) per batch element,
// allocated in global memory.
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
// Host-side sig_to_log_sig core launch
// =========================================================================

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
		// degree 0: just copy the scalar 1 → out[0] = 0 (log of unit)
		auto zeros = std::make_unique<T[]>(batch_size);
		std::fill(zeros.get(), zeros.get() + batch_size, static_cast<T>(0));
		cudaMemcpy(out, zeros.get(), batch_size * sizeof(T), cudaMemcpyHostToDevice);
		return;
	}

	if (degree == 1) {
		// degree 1: just set out[0]=0, copy the rest
		// We launch the kernel anyway (it handles this case)
	}

	// Build level_index on host and copy to device
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);

	uint64_t* d_level_index = nullptr;
	cudaMalloc(&d_level_index, level_index_bytes);
	cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);

	// Allocate scratch buffers
	const uint64_t buff1_len = degree >= 2 ? host_sig_length(dimension, degree - 1) : 1;

	T* d_buff1 = nullptr;
	T* d_buff2 = nullptr;
	cudaMalloc(&d_buff1, sizeof(T) * batch_size * buff1_len);
	cudaMalloc(&d_buff2, sizeof(T) * batch_size * sig_len);

	// Choose threads per block based on largest level size
	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = 32;
	if (max_level_size > 32)   threads_per_block = 64;
	if (max_level_size > 64)   threads_per_block = 128;
	if (max_level_size > 128)  threads_per_block = 256;
	if (max_level_size > 512)  threads_per_block = 512;
	if (max_level_size > 1024) threads_per_block = 1024;

	// Shared memory: level_index only
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_to_log_sig_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig, out, d_buff1, d_buff2, d_level_index, degree, sig_len, buff1_len
	);

	cudaDeviceSynchronize();
	cudaFree(d_level_index);
	cudaFree(d_buff1);
	cudaFree(d_buff2);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const int error_code = static_cast<int>(err);
		throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
	}
}

// =========================================================================
// CUDA sig_to_log_sig method 1 kernel (Lyndon words)
//
// Phase 1: Copy sig → temp, compute tensor log in-place
// Phase 2: Gather — out[i] = temp[lyndon_idx[i]]
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
// Phase 1: Copy sig → temp, compute tensor log in-place
// Phase 2: Gather — gathered[i] = temp[lyndon_idx[i]]
// Phase 3: Apply sparse lower-triangular matrix multiply (serial per row)
//          out[i] = gathered[i] + sum_j(sparse_mat[i][j] * gathered[j])
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

	// Apply inverse projection matrix (lower triangular, diagonal = identity, dropped)
	// out[i] += sum_j mat[i][j] * out[j] for j < i
	// Process rows bottom-up so that out[j] values used in row i are still original
	if (tid == 0) {
		for (uint64_t i_ = 0; i_ < log_sig_len; ++i_) {
			uint64_t i = log_sig_len - i_ - 1;
			uint64_t row_start = d_sparse_row_ptr[i];
			uint64_t row_end = d_sparse_row_ptr[i + 1];
			T acc = static_cast<T>(0);
			for (uint64_t k = row_start; k < row_end; ++k) {
				acc += static_cast<T>(d_sparse_vals[k]) * my_out[d_sparse_cols[k]];
			}
			my_out[i] += acc;
		}
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
	const uint64_t sig_len = host_sig_length(dimension, degree);
	const auto& cache = get_cuda_log_sig_cache(dimension, degree);
	const uint64_t log_sig_len = cache.log_sig_len;

	// Build level_index on host and copy to device
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);

	uint64_t* d_level_index = nullptr;
	cudaMalloc(&d_level_index, level_index_bytes);
	cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);

	// Allocate scratch buffers
	const uint64_t buff1_len = degree >= 2 ? host_sig_length(dimension, degree - 1) : 1;

	T* d_temp = nullptr;
	T* d_buff1 = nullptr;
	T* d_buff2 = nullptr;
	cudaMalloc(&d_temp, sizeof(T) * batch_size * sig_len);
	cudaMalloc(&d_buff1, sizeof(T) * batch_size * buff1_len);
	cudaMalloc(&d_buff2, sizeof(T) * batch_size * sig_len);

	// Choose threads per block
	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = 32;
	if (max_level_size > 32)   threads_per_block = 64;
	if (max_level_size > 64)   threads_per_block = 128;
	if (max_level_size > 128)  threads_per_block = 256;
	if (max_level_size > 512)  threads_per_block = 512;
	if (max_level_size > 1024) threads_per_block = 1024;

	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_to_log_sig_m1_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig, out, d_temp, d_buff1, d_buff2, d_level_index,
		cache.d_lyndon_idx, degree, sig_len, buff1_len, log_sig_len
	);

	cudaDeviceSynchronize();
	cudaFree(d_level_index);
	cudaFree(d_temp);
	cudaFree(d_buff1);
	cudaFree(d_buff2);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const int error_code = static_cast<int>(err);
		throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
	}
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
	const uint64_t sig_len = host_sig_length(dimension, degree);
	const auto& cache = get_cuda_log_sig_cache(dimension, degree);
	const uint64_t log_sig_len = cache.log_sig_len;

	// Build level_index on host and copy to device
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);

	uint64_t* d_level_index = nullptr;
	cudaMalloc(&d_level_index, level_index_bytes);
	cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);

	// Allocate scratch buffers
	const uint64_t buff1_len = degree >= 2 ? host_sig_length(dimension, degree - 1) : 1;

	T* d_temp = nullptr;
	T* d_buff1 = nullptr;
	T* d_buff2 = nullptr;
	cudaMalloc(&d_temp, sizeof(T) * batch_size * sig_len);
	cudaMalloc(&d_buff1, sizeof(T) * batch_size * buff1_len);
	cudaMalloc(&d_buff2, sizeof(T) * batch_size * sig_len);

	// Choose threads per block
	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = 32;
	if (max_level_size > 32)   threads_per_block = 64;
	if (max_level_size > 64)   threads_per_block = 128;
	if (max_level_size > 128)  threads_per_block = 256;
	if (max_level_size > 512)  threads_per_block = 512;
	if (max_level_size > 1024) threads_per_block = 1024;

	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_to_log_sig_m2_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig, out, d_temp, d_buff1, d_buff2, d_level_index,
		cache.d_lyndon_idx,
		cache.d_sparse_vals, cache.d_sparse_cols, cache.d_sparse_row_ptr,
		degree, sig_len, buff1_len, log_sig_len
	);

	cudaDeviceSynchronize();
	cudaFree(d_level_index);
	cudaFree(d_temp);
	cudaFree(d_buff1);
	cudaFree(d_buff2);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const int error_code = static_cast<int>(err);
		throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
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
	int method
) {
	if (dimension == 0) throw std::invalid_argument("sig_to_log_sig_cuda received dimension 0");
	if (degree == 0) throw std::invalid_argument("sig_to_log_sig_cuda received degree 0");

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
}

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#ifndef CU_LOG_SIG_SAFE_CALL
#define CU_LOG_SIG_SAFE_CALL(function_call)                     \
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
    catch (std::out_of_range& e) {                              \
        std::cerr << e.what();                                  \
        return 3;                                               \
    }                                                           \
    catch (std::runtime_error& e) {                             \
        std::string msg = e.what();                             \
        std::regex pattern(R"(CUDA Error \((\d+)\):)");         \
        std::smatch match;                                      \
        int ret_code = 10;                                      \
        if (std::regex_search(msg, match, pattern)) {           \
            ret_code = 100000 + std::stoi(match[1]);            \
        }                                                       \
        std::cerr << e.what();                                  \
        return ret_code;                                        \
    }                                                           \
    catch (...) {                                               \
        std::cerr << "Unknown exception";                       \
        return 11;                                              \
    }                                                           \
    return 0;
#endif

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {

	CUSIG_API int sig_to_log_sig_cuda_f(
		const float* sig, float* out,
		uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CU_LOG_SIG_SAFE_CALL(sig_to_log_sig_cuda_<float>(sig, out, 1, dimension, degree, method));
	}

	CUSIG_API int sig_to_log_sig_cuda_d(
		const double* sig, double* out,
		uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CU_LOG_SIG_SAFE_CALL(sig_to_log_sig_cuda_<double>(sig, out, 1, dimension, degree, method));
	}

	CUSIG_API int batch_sig_to_log_sig_cuda_f(
		const float* sig, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CU_LOG_SIG_SAFE_CALL(sig_to_log_sig_cuda_<float>(sig, out, batch_size, dimension, degree, method));
	}

	CUSIG_API int batch_sig_to_log_sig_cuda_d(
		const double* sig, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CU_LOG_SIG_SAFE_CALL(sig_to_log_sig_cuda_<double>(sig, out, batch_size, dimension, degree, method));
	}
}
