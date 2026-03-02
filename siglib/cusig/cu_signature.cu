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
// Custom atomicAdd for double: CAS-based implementation for sm_50/52
// (Native atomicAdd(double*,double) requires sm_60+)
// For float, we just forward to the built-in atomicAdd.
// =========================================================================

template<typename T>
__device__ __forceinline__ void myAtomicAdd(T* address, T val);

template<>
__device__ __forceinline__ void myAtomicAdd<float>(float* address, float val) {
	atomicAdd(address, val);
}

template<>
__device__ __forceinline__ void myAtomicAdd<double>(double* address, double val) {
	unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
}

// =========================================================================
// CUDA Device: uncombine_sig_deriv
//
// Given sig1, sig2 are two signatures, and sig_concat = sig1 * sig2.
// sig_concat_deriv is dF/d(sig_concat).
// Computes dF/d(sig1) into sig_concat_deriv and dF/d(sig2) into sig2_deriv.
// =========================================================================

template<typename T>
__device__ void uncombine_sig_deriv_device(
	const T* __restrict__ sig1,
	const T* __restrict__ sig2,
	T* __restrict__ sig_concat_deriv,
	T* __restrict__ sig2_deriv,
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// Copy sig_concat_deriv to sig2_deriv
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		sig2_deriv[i] = sig_concat_deriv[i];
	__syncthreads();

	// First loop: accumulate sig2_deriv using sig1 values
	for (uint64_t level = degree; level > 0; --level) {
		for (uint64_t left_level = level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
			const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
			const uint64_t target_size = left_size * right_size;

			for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
				const uint64_t l_idx = idx / right_size;
				const uint64_t r_idx = idx % right_size;
				// result_ptr[idx] * left_ptr[l_idx] -> right_ptr[r_idx]
				T val = sig_concat_deriv[level_index[level] + idx] * sig1[level_index[left_level] + l_idx];
				myAtomicAdd(&sig2_deriv[level_index[right_level] + r_idx], val);
			}
			__syncthreads();
		}
	}

	// Second loop: accumulate sig1 (sig_concat_deriv) derivatives using sig2 values
	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		for (uint64_t level = left_level + 1, right_level = 1;
			level <= degree;
			++level, ++right_level) {

			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
			const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
			const uint64_t target_size = left_size * right_size;

			for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
				const uint64_t l_idx = idx / right_size;
				const uint64_t r_idx = idx % right_size;
				// result_ptr[idx] * right_ptr[r_idx] -> left_ptr[l_idx]
				T val = sig_concat_deriv[level_index[level] + idx] * sig2[level_index[right_level] + r_idx];
				myAtomicAdd(&sig_concat_deriv[level_index[left_level] + l_idx], val);
			}
			__syncthreads();
		}
	}
}

// =========================================================================
// CUDA Device: linear_sig_deriv_to_increment_deriv
//
// Given sig is the signature of a line segment [a,b] and sig_deriv is
// dF/d(sig), computes dF/d(b-a) and writes it into sig_deriv[1..dimension].
// =========================================================================

template<typename T>
__device__ void linear_sig_deriv_to_increment_deriv_device(
	const T* __restrict__ sig,
	T* __restrict__ sig_deriv,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	for (uint64_t level = degree; level > 1; --level) {
		const T one_over_level = static_cast<T>(1) / static_cast<T>(level);
		const uint64_t level_size = level_index[level] - level_index[level - 1];

		for (uint64_t j = tid; j < level_size; j += nthreads) {
			const uint64_t offs1 = level_index[level] + dimension * j - 1;
			const uint64_t offs2 = level_index[level - 1] + j;
			for (uint64_t dd = 1; dd <= dimension; ++dd) {
				const T ii = sig_deriv[offs1 + dd] * one_over_level;
				myAtomicAdd(&sig_deriv[offs2], sig[dd] * ii);
				myAtomicAdd(&sig_deriv[dd], sig[offs2] * ii);
			}
		}
		__syncthreads();
	}
}

// =========================================================================
// Main backprop kernel: one block per batch element
//
// path:       [batch_size, length, dimension] (GPU, transformed path)
// out:        [batch_size, length, dimension] (GPU, output path derivs)
// sig_derivs: [batch_size, sig_len] (GPU, dF/d(sig), will be modified)
// sig:        [batch_size, sig_len] (GPU, forward signature, will be modified)
// workspace:  per-batch workspace for local_derivs, linear_signature,
//             horner buffers
// =========================================================================

template<typename T>
__global__ void sig_backprop_kernel(
	const T* __restrict__ path,
	T* __restrict__ out,
	T* __restrict__ sig_derivs,
	T* __restrict__ sig,
	const uint64_t* __restrict__ d_level_index,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t sig_len,
	uint64_t path_flat_len,
	T* __restrict__ workspace,
	uint64_t workspace_per_batch,
	uint64_t horner_half_size
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * path_flat_len;
	T* my_sig_derivs = sig_derivs + batch_idx * sig_len;
	T* my_sig = sig + batch_idx * sig_len;

	// Workspace layout per batch:
	//   [0 .. sig_len-1]                     = local_derivs
	//   [sig_len .. 2*sig_len-1]             = linear_signature
	//   [2*sig_len .. 2*sig_len+2*hhs-1]    = horner_workspace
	T* my_workspace = workspace + batch_idx * workspace_per_batch;
	T* local_derivs = my_workspace;
	T* linear_sig = my_workspace + sig_len;
	T* horner_ws = my_workspace + 2 * sig_len;

	// Shared memory for increments
	extern __shared__ char smem[];
	T* increments = reinterpret_cast<T*>(smem);

	// Zero the output
	for (uint64_t i = tid; i < path_flat_len; i += nthreads)
		my_out[i] = static_cast<T>(0);
	__syncthreads();

	// Iterate BACKWARDS through path segments
	// Segment indices: from (length-1) down to 1
	// For segment step, prev_pt = path[step-1], next_pt = path[step]
	for (int64_t step = static_cast<int64_t>(length) - 1; step >= 1; --step) {
		const T* prev_pt = my_path + (step - 1) * dimension;
		const T* next_pt = my_path + step * dimension;

		// Compute FORWARD increments = next_pt - prev_pt (for linear signature)
		for (uint64_t i = tid; i < dimension; i += nthreads)
			increments[i] = next_pt[i] - prev_pt[i];
		__syncthreads();

		// Compute linear signature of the FORWARD segment
		linear_signature_device(increments, linear_sig, dimension, degree, d_level_index);
		__syncthreads();

		// Negate increments to get REVERSED = prev_pt - next_pt (for Horner uncombine)
		for (uint64_t i = tid; i < dimension; i += nthreads)
			increments[i] = -increments[i];
		__syncthreads();

		// Horner step to "uncombine": sig = sig uncombined with reversed segment
		signature_horner_step_device(my_sig, increments, dimension, degree,
			d_level_index, horner_ws, horner_half_size);
		__syncthreads();

		// uncombine_sig_deriv: split derivatives
		// After this: my_sig_derivs contains dF/d(sig1), local_derivs contains dF/d(sig2)
		uncombine_sig_deriv_device(my_sig, linear_sig, my_sig_derivs, local_derivs,
			dimension, degree, sig_len, d_level_index);
		__syncthreads();

		// linear_sig_deriv_to_increment_deriv: convert dF/d(linear_sig) to dF/d(increment)
		// Result is in local_derivs[1..dimension]
		linear_sig_deriv_to_increment_deriv_device(linear_sig, local_derivs,
			dimension, degree, d_level_index);
		__syncthreads();

		// Accumulate into output: pos += s, neg -= s
		// pos = out + step * dimension, neg = out + (step-1) * dimension
		// s = local_derivs + 1
		T* pos = my_out + step * dimension;
		T* neg = my_out + (step - 1) * dimension;
		for (uint64_t d = tid; d < dimension; d += nthreads) {
			T s = local_derivs[1 + d];
			pos[d] += s;
			neg[d] -= s;
		}
		__syncthreads();
	}
}

// =========================================================================
// Host-side core launch for backprop (operates on already-transformed path)
// =========================================================================

template<typename T>
void sig_backprop_cuda_core_(
	const T* path,
	T* out,
	const T* sig_derivs,
	const T* sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);
	const uint64_t path_flat_len = dimension * length;

	// Handle trivial cases
	if (length <= 1 || degree == 0) {
		cudaMemset(out, 0, batch_size * path_flat_len * sizeof(T));
		return;
	}

	// Build level_index on host and copy to device
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	uint64_t* d_level_index;
	cudaMalloc(&d_level_index, (degree + 2) * sizeof(uint64_t));
	cudaMemcpy(d_level_index, level_index_host.get(), (degree + 2) * sizeof(uint64_t), cudaMemcpyHostToDevice);

	// Make mutable copies of sig_derivs and sig on device
	T* d_sig_derivs_copy;
	T* d_sig_copy;
	cudaMalloc(&d_sig_derivs_copy, batch_size * sig_len * sizeof(T));
	cudaMalloc(&d_sig_copy, batch_size * sig_len * sizeof(T));
	cudaMemcpy(d_sig_derivs_copy, sig_derivs, batch_size * sig_len * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_sig_copy, sig, batch_size * sig_len * sizeof(T), cudaMemcpyDeviceToDevice);

	// Workspace per batch element:
	//   local_derivs: sig_len
	//   linear_signature: sig_len
	//   horner_workspace: 2 * horner_half_size
	uint64_t horner_half_size = level_index_host[degree + 1] - level_index_host[degree];
	uint64_t workspace_per_batch = 2 * sig_len + 2 * horner_half_size;

	T* d_workspace;
	cudaMalloc(&d_workspace, batch_size * workspace_per_batch * sizeof(T));

	// Choose threads per block
	uint64_t max_level_size = horner_half_size;
	unsigned int threads_per_block = 32;
	if (max_level_size > 32) threads_per_block = 64;
	if (max_level_size > 64) threads_per_block = 128;
	if (max_level_size > 128) threads_per_block = 256;
	if (max_level_size > 512) threads_per_block = 512;

	// Shared memory for increments
	size_t smem_size = dimension * sizeof(T);

	sig_backprop_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		path, out, d_sig_derivs_copy, d_sig_copy,
		d_level_index,
		dimension, length, degree, sig_len, path_flat_len,
		d_workspace, workspace_per_batch, horner_half_size
	);

	cudaError_t err = cudaDeviceSynchronize();

	cudaFree(d_workspace);
	cudaFree(d_sig_derivs_copy);
	cudaFree(d_sig_copy);
	cudaFree(d_level_index);

	if (err != cudaSuccess) {
		throw std::runtime_error("CUDA Error (" + std::to_string(static_cast<int>(err)) + "): " + cudaGetErrorString(err));
	}
}

// =========================================================================
// Host-side launch function for backprop (with time_aug / lead_lag support)
//
// For the backprop, when time_aug or lead_lag is set:
// 1. Transform the path on GPU
// 2. Compute the backprop on the transformed path (gives derivs w.r.t. transformed path)
// 3. Apply transform_path_backprop to get derivs w.r.t. original path
// =========================================================================

// Forward-declare transform_path_backprop_ from cu_path_transforms.cu
template<typename T>
void transform_path_backprop_(
	const T* derivs,
	T* data_out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time
);

template<typename T>
void sig_backprop_cuda_(
	const T* path,
	T* out,
	const T* sig_derivs,
	const T* sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool time_aug,
	bool lead_lag,
	T end_time
) {
	if (dimension == 0) throw std::invalid_argument("sig_backprop_cuda received path of dimension 0");

	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		// Transform path on GPU
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		T* d_transformed = nullptr;
		T* d_transformed_derivs = nullptr;

		try {
			cudaMalloc(&d_transformed, t_path_size * sizeof(T));

			transform_path_<T>(path, d_transformed, batch_size, dimension, length, time_aug, lead_lag, end_time);

			// Backprop on transformed path -> derivs w.r.t. transformed path
			cudaMalloc(&d_transformed_derivs, t_path_size * sizeof(T));

			sig_backprop_cuda_core_<T>(d_transformed, d_transformed_derivs, sig_derivs, sig,
				batch_size, t_dimension, t_length, degree);

			cudaFree(d_transformed);
			d_transformed = nullptr;

			// Apply transform_path_backprop to get derivs w.r.t. original path
			transform_path_backprop_<T>(d_transformed_derivs, out, batch_size, dimension, length, time_aug, lead_lag, end_time);

			cudaFree(d_transformed_derivs);
		} catch (...) {
			if (d_transformed) cudaFree(d_transformed);
			if (d_transformed_derivs) cudaFree(d_transformed_derivs);
			throw;
		}
	}
	else {
		sig_backprop_cuda_core_<T>(path, out, sig_derivs, sig, batch_size, dimension, length, degree);
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

	// =====================================================================
	// Signature backpropagation exports
	// =====================================================================

	CUSIG_API int sig_backprop_cuda_f(
		const float* path, float* out,
		const float* sig_derivs, const float* sig,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(sig_backprop_cuda_<float>(path, out, sig_derivs, sig, 1, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CUSIG_API int sig_backprop_cuda_d(
		const double* path, double* out,
		const double* sig_derivs, const double* sig,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(sig_backprop_cuda_<double>(path, out, sig_derivs, sig, 1, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_sig_backprop_cuda_f(
		const float* path, float* out,
		const float* sig_derivs, const float* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(sig_backprop_cuda_<float>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_sig_backprop_cuda_d(
		const double* path, double* out,
		const double* sig_derivs, const double* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time
	) noexcept {
		CU_SIGNATURE_SAFE_CALL(sig_backprop_cuda_<double>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time));
	}
}
