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
	const uint64_t* __restrict__ d_level_index,
	uint64_t degree,
	uint64_t sig_len
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_sig1 = sig1 + batch_idx * sig_len;
	const T* my_sig2 = sig2 + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;

	// Load level_index into shared memory
	extern __shared__ char smem[];
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem);

	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// Copy sig1 to out
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		my_out[i] = my_sig1[i];
	__syncthreads();

	// Combine in-place: out = out (x) sig2
	sig_combine_inplace_device<T>(my_out, my_sig2, degree, level_index_smem);
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

	// Build level_index on host and copy to device
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);

	uint64_t* d_level_index = nullptr;
	cudaMalloc(&d_level_index, level_index_bytes);
	cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);

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

	sig_combine_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig1, sig2, out, d_level_index, degree, sig_len
	);

	cudaDeviceSynchronize();
	cudaFree(d_level_index);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const int error_code = static_cast<int>(err);
		throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
	}
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
// SAFE_CALL macro
// =========================================================================

#ifndef CU_TENSOR_POLY_SAFE_CALL
#define CU_TENSOR_POLY_SAFE_CALL(function_call)                 \
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

	CUSIG_API int sig_combine_cuda_f(
		const float* sig1, const float* sig2, float* out,
		uint64_t dimension, uint64_t degree
	) noexcept {
		CU_TENSOR_POLY_SAFE_CALL(sig_combine_cuda_<float>(sig1, sig2, out, 1, dimension, degree));
	}

	CUSIG_API int sig_combine_cuda_d(
		const double* sig1, const double* sig2, double* out,
		uint64_t dimension, uint64_t degree
	) noexcept {
		CU_TENSOR_POLY_SAFE_CALL(sig_combine_cuda_<double>(sig1, sig2, out, 1, dimension, degree));
	}

	CUSIG_API int batch_sig_combine_cuda_f(
		const float* sig1, const float* sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CU_TENSOR_POLY_SAFE_CALL(sig_combine_cuda_<float>(sig1, sig2, out, batch_size, dimension, degree));
	}

	CUSIG_API int batch_sig_combine_cuda_d(
		const double* sig1, const double* sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CU_TENSOR_POLY_SAFE_CALL(sig_combine_cuda_<double>(sig1, sig2, out, batch_size, dimension, degree));
	}
}
