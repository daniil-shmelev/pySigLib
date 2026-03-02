/* Copyright 2025 Daniil Shmelev
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
#include "cu_signature.h"

// =========================================================================
// Helper: compute sig_length on the host
// =========================================================================

static uint64_t host_power(uint64_t base, uint64_t exp) {
	uint64_t result = 1;
	while (exp > 0) {
		if (exp & 1) result *= base;
		base *= base;
		exp >>= 1;
	}
	return result;
}

static uint64_t host_sig_length(uint64_t dimension, uint64_t degree) {
	if (dimension == 0) return 1;
	if (dimension == 1) return degree + 1;
	return (host_power(dimension, degree + 1) - 1) / (dimension - 1);
}

// =========================================================================
// Helper: populate level_index on the host
//   level_index[k] = start offset of level k in the flat signature array
//   level_index[0] = 0, level_index[1] = 1, level_index[k] = level_index[k-1]*dim + 1
// =========================================================================

static void host_populate_level_index(uint64_t* level_index, uint64_t dimension, uint64_t count) {
	level_index[0] = 0;
	for (uint64_t i = 1; i < count; ++i)
		level_index[i] = level_index[i - 1] * dimension + 1;
}

// =========================================================================
// CUDA Kernel: compute the linear signature of a single segment
//
// Given increments z[0..dim-1] = end_pt - start_pt, the linear signature is:
//   level 0: 1
//   level 1: z
//   level k: tensor product of level (k-1) with z, divided by k
//
// Each thread handles a strided portion of the output.
// =========================================================================

template<typename T>
__device__ void linear_signature_device(
	const T* __restrict__ increments,   // dimension elements
	T* __restrict__ out,                // sig_length elements
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// Level 0
	if (tid == 0) out[0] = static_cast<T>(1);

	// Level 1
	for (uint64_t i = tid; i < dimension; i += nthreads)
		out[i + 1] = increments[i];

	__syncthreads();

	// Higher levels
	for (uint64_t level = 2; level <= degree; ++level) {
		const T one_over_level = static_cast<T>(1) / static_cast<T>(level);
		const uint64_t prev_start = level_index[level - 1];
		const uint64_t prev_size = level_index[level] - level_index[level - 1];
		const uint64_t cur_start = level_index[level];
		const uint64_t cur_size = prev_size * dimension;

		for (uint64_t idx = tid; idx < cur_size; idx += nthreads) {
			const uint64_t left_idx = idx / dimension;
			const uint64_t right_idx = idx % dimension;
			out[cur_start + idx] = out[prev_start + left_idx] * increments[right_idx] * one_over_level;
		}
		__syncthreads();
	}
}

// =========================================================================
// CUDA Kernel: sig_combine_inplace on device
//
// Computes sig1 = sig1 (x) sig2  (tensor product / Chen's identity)
// where sig2 is the signature of a linear segment.
// =========================================================================

template<typename T>
__device__ void sig_combine_inplace_device(
	T* __restrict__ sig1,
	const T* __restrict__ sig2,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// Process from highest level down to 1
	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {

		const uint64_t target_start = level_index[target_level];
		const uint64_t target_size = level_index[target_level + 1] - level_index[target_level];

		// Accumulate contributions from left_level x right_level where left_level + right_level = target_level
		for (int64_t left_level = target_level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			const uint64_t left_start = level_index[left_level];
			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
			const uint64_t right_start = level_index[right_level];
			const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];

			for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
				const uint64_t l_idx = idx / right_size;
				const uint64_t r_idx = idx % right_size;
				sig1[target_start + idx] += sig1[left_start + l_idx] * sig2[right_start + r_idx];
			}
			__syncthreads();
		}

		// left_level = 0: just add sig2 at this level
		for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
			sig1[target_start + idx] += sig2[level_index[target_level] + idx];
		}
		__syncthreads();
	}
}

// =========================================================================
// CUDA Kernel: Horner-based signature step on device
//
// Combines the current signature with the signature of a linear segment
// (given by increments) using Horner's method.  This avoids materialising
// the full linear signature and is the same algorithm as the CPU version.
//
// Workspace layout: horner_step has size 2 * dim^degree.
//   - buf_A = horner_step[0 .. dim^degree - 1]
//   - buf_B = horner_step[dim^degree .. 2*dim^degree - 1]
//   We ping-pong between buf_A and buf_B during the multiply steps
//   to avoid read-write conflicts under parallel execution.
// =========================================================================

template<typename T>
__device__ void signature_horner_step_device(
	T* __restrict__ sig,
	const T* __restrict__ increments,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* __restrict__ level_index,
	T* __restrict__ horner_workspace,  // size = 2 * (level_index[degree+1] - level_index[degree])
	uint64_t half_size                 // = level_index[degree+1] - level_index[degree]
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	T* buf_A = horner_workspace;
	T* buf_B = horner_workspace + half_size;

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 1; --target_level) {

		T one_over_level = static_cast<T>(1) / static_cast<T>(target_level);

		// Current read buffer: start with buf_A
		T* src = buf_A;
		T* dst = buf_B;

		// left_level = 0: assign increments * one_over_level into src
		for (uint64_t i = tid; i < dimension; i += nthreads)
			src[i] = increments[i] * one_over_level;
		__syncthreads();

		for (int64_t left_level = 1, right_level = target_level - 1;
			left_level < target_level - 1;
			++left_level, --right_level) {

			const uint64_t left_level_size = level_index[left_level + 1] - level_index[left_level];
			one_over_level = static_cast<T>(1) / static_cast<T>(right_level);

			// Add sig at left_level into src (current horner data)
			for (uint64_t i = tid; i < left_level_size; i += nthreads)
				src[i] += sig[level_index[left_level] + i];
			__syncthreads();

			// Multiply: dst = src (x) increments * one_over_level
			const uint64_t result_size = left_level_size * dimension;
			for (uint64_t idx = tid; idx < result_size; idx += nthreads) {
				const uint64_t l_idx = idx / dimension;
				const uint64_t r_idx = idx % dimension;
				dst[idx] = src[l_idx] * increments[r_idx] * one_over_level;
			}
			__syncthreads();

			// Swap src and dst
			T* tmp = src;
			src = dst;
			dst = tmp;
		}

		// Last iteration: left_level = target_level - 1
		{
			const uint64_t left_level_size = level_index[target_level] - level_index[target_level - 1];

			// Add sig at target_level - 1 into src
			for (uint64_t i = tid; i < left_level_size; i += nthreads)
				src[i] += sig[level_index[target_level - 1] + i];
			__syncthreads();

			// Multiply and add directly into sig at target_level (right_level = 1)
			const uint64_t out_size = left_level_size * dimension;
			for (uint64_t idx = tid; idx < out_size; idx += nthreads) {
				const uint64_t l_idx = idx / dimension;
				const uint64_t r_idx = idx % dimension;
				sig[level_index[target_level] + idx] += src[l_idx] * increments[r_idx];
			}
			__syncthreads();
		}
	}

	// Update level 1
	for (uint64_t i = tid; i < dimension; i += nthreads)
		sig[i + 1] += increments[i];
	__syncthreads();
}

// =========================================================================
// Main kernel: compute signature for a batch of paths
// One block per batch element. Path data is already transformed
// (time_aug / lead_lag handled by cu_path_transforms).
//
// path: [batch_size, length, dimension] in row-major (on GPU)
// out:  [batch_size, sig_length] (on GPU, preallocated)
// =========================================================================

template<typename T>
__global__ void signature_kernel(
	const T* __restrict__ path,
	T* __restrict__ out,
	const uint64_t* __restrict__ d_level_index,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t path_flat_len,        // dimension * length
	T* __restrict__ workspace,     // [batch_size * 2 * horner_half_size]
	uint64_t horner_half_size      // = level_index[degree+1] - level_index[degree]
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * sig_len;
	T* my_horner = workspace + batch_idx * 2 * horner_half_size;

	// ---- Compute increments for step 0 and linear signature ----
	// Use shared memory for increments (dimension elements)
	extern __shared__ char smem[];
	T* increments = reinterpret_cast<T*>(smem);

	const T* prev_pt = my_path;
	const T* next_pt = my_path + dimension;

	// Compute increments = next_pt - prev_pt
	for (uint64_t i = tid; i < dimension; i += nthreads)
		increments[i] = next_pt[i] - prev_pt[i];
	__syncthreads();

	// Linear signature of first segment
	linear_signature_device(increments, my_out, dimension, degree, d_level_index);
	__syncthreads();

	if (length <= 2) return;

	// ---- Iterate over remaining segments using Horner ----
	for (uint64_t step = 2; step < length; ++step) {
		prev_pt = my_path + (step - 1) * dimension;
		next_pt = my_path + step * dimension;

		// Compute increments
		for (uint64_t i = tid; i < dimension; i += nthreads)
			increments[i] = next_pt[i] - prev_pt[i];
		__syncthreads();

		// Horner step: combine current sig with linear sig of this segment
		signature_horner_step_device(my_out, increments, dimension, degree, d_level_index, my_horner, horner_half_size);
	}
}

// =========================================================================
// Naive kernel (fallback, uses sig_combine instead of Horner)
// =========================================================================

template<typename T>
__global__ void signature_naive_kernel(
	const T* __restrict__ path,
	T* __restrict__ out,
	const uint64_t* __restrict__ d_level_index,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t path_flat_len,
	T* __restrict__ linear_sig_workspace  // [batch_size * sig_len]
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * sig_len;
	T* my_linear_sig = linear_sig_workspace + batch_idx * sig_len;

	extern __shared__ char smem[];
	T* increments = reinterpret_cast<T*>(smem);

	const T* prev_pt = my_path;
	const T* next_pt = my_path + dimension;

	// Increments for first segment
	for (uint64_t i = tid; i < dimension; i += nthreads)
		increments[i] = next_pt[i] - prev_pt[i];
	__syncthreads();

	// Linear signature of first segment
	linear_signature_device(increments, my_out, dimension, degree, d_level_index);
	__syncthreads();

	if (length <= 2) return;

	for (uint64_t step = 2; step < length; ++step) {
		prev_pt = my_path + (step - 1) * dimension;
		next_pt = my_path + step * dimension;

		for (uint64_t i = tid; i < dimension; i += nthreads)
			increments[i] = next_pt[i] - prev_pt[i];
		__syncthreads();

		linear_signature_device(increments, my_linear_sig, dimension, degree, d_level_index);
		__syncthreads();

		sig_combine_inplace_device(my_out, my_linear_sig, degree, d_level_index);
		__syncthreads();
	}
}

// =========================================================================
// Host-side core launch (operates on already-transformed path data)
// =========================================================================

template<typename T>
void signature_cuda_core_(
	const T* path,          // GPU pointer, shape [batch_size, length, dimension] flattened
	T* out,                 // GPU pointer, shape [batch_size, sig_len] flattened
	uint64_t batch_size,
	uint64_t dimension,     // transformed dimension
	uint64_t length,        // transformed length
	uint64_t degree,
	bool horner
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);
	const uint64_t path_flat_len = dimension * length;

	// Handle trivial cases
	if (length <= 1) {
		// sig = (1, 0, 0, ..., 0) for each batch element
		cudaMemset(out, 0, batch_size * sig_len * sizeof(T));
		auto ones = std::make_unique<T[]>(batch_size);
		std::fill(ones.get(), ones.get() + batch_size, static_cast<T>(1));
		for (uint64_t i = 0; i < batch_size; ++i)
			cudaMemcpy(out + i * sig_len, ones.get() + i, sizeof(T), cudaMemcpyHostToDevice);
		return;
	}

	if (degree == 0) {
		auto ones = std::make_unique<T[]>(batch_size);
		std::fill(ones.get(), ones.get() + batch_size, static_cast<T>(1));
		cudaMemcpy(out, ones.get(), batch_size * sizeof(T), cudaMemcpyHostToDevice);
		return;
	}

	// Build level_index on host and copy to device
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	uint64_t* d_level_index;
	cudaMalloc(&d_level_index, (degree + 2) * sizeof(uint64_t));
	cudaMemcpy(d_level_index, level_index_host.get(), (degree + 2) * sizeof(uint64_t), cudaMemcpyHostToDevice);

	// Choose number of threads per block based on largest level size
	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = 32; // minimum: one warp
	if (max_level_size > 32) threads_per_block = 64;
	if (max_level_size > 64) threads_per_block = 128;
	if (max_level_size > 128) threads_per_block = 256;
	if (max_level_size > 512) threads_per_block = 512;

	// Shared memory: space for increments
	size_t smem_size = dimension * sizeof(T);

	if (horner && degree >= 2) {
		// Allocate Horner workspace on device (2x for ping-pong buffers)
		uint64_t horner_half_size = max_level_size;
		T* d_workspace;
		cudaMalloc(&d_workspace, batch_size * 2 * horner_half_size * sizeof(T));

		signature_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
			path, out, d_level_index,
			dimension, length, degree, sig_len, path_flat_len,
			d_workspace, horner_half_size
		);

		cudaDeviceSynchronize();
		cudaFree(d_workspace);
	}
	else {
		// Allocate linear sig workspace on device
		T* d_linear_sig;
		cudaMalloc(&d_linear_sig, batch_size * sig_len * sizeof(T));

		signature_naive_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
			path, out, d_level_index,
			dimension, length, degree, sig_len, path_flat_len,
			d_linear_sig
		);

		cudaDeviceSynchronize();
		cudaFree(d_linear_sig);
	}

	cudaFree(d_level_index);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const int error_code = static_cast<int>(err);
		throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
	}
}

// =========================================================================
// Host-side launch function (with time_aug / lead_lag support)
//
// The input path is on GPU with shape [batch_size, length, dimension].
// If time_aug or lead_lag is set, the path is first transformed on-GPU
// using the existing cu_path_transforms infrastructure, then the signature
// is computed on the transformed path.
// =========================================================================

// Forward-declare transform_path_ from cu_path_transforms.cu
template<typename T>
void transform_path_(
	const T* data_in,
	T* data_out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time
);

template<typename T>
void signature_cuda_(
	const T* path,          // GPU pointer
	T* out,                 // GPU pointer
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool time_aug,
	bool lead_lag,
	T end_time,
	bool horner
) {
	if (dimension == 0) throw std::invalid_argument("signature_cuda received path of dimension 0");

	// Compute transformed dimensions
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		// Transform path on GPU
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		T* d_transformed;
		cudaMalloc(&d_transformed, t_path_size * sizeof(T));

		transform_path_<T>(path, d_transformed, batch_size, dimension, length, time_aug, lead_lag, end_time);
		cudaDeviceSynchronize();

		signature_cuda_core_<T>(d_transformed, out, batch_size, t_dimension, t_length, degree, horner);

		cudaFree(d_transformed);
	}
	else {
		signature_cuda_core_<T>(path, out, batch_size, dimension, length, degree, horner);
	}
}

// =========================================================================
// SAFE_CALL macro (same pattern as cu_sig_kernel.cu)
// =========================================================================

#ifndef CU_SIGNATURE_SAFE_CALL
#define CU_SIGNATURE_SAFE_CALL(function_call)                   \
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

	CUSIG_API int signature_cuda_f(
		const float* path, float* out,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time,
		bool horner
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(signature_cuda_<float>(path, out, 1, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CUSIG_API int signature_cuda_d(
		const double* path, double* out,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time,
		bool horner
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(signature_cuda_<double>(path, out, 1, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CUSIG_API int batch_signature_cuda_f(
		const float* path, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time,
		bool horner
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(signature_cuda_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CUSIG_API int batch_signature_cuda_d(
		const double* path, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time,
		bool horner
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(signature_cuda_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}
}
