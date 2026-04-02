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
#include "cu_signature.h"
#include "cu_tensor_poly.h"
#include "cu_atomic.h"

template<typename T>
__device__ void linear_signature_device(
	const T* __restrict__ increments,   // dimension many elements
	T* __restrict__ out,                // sig_length many elements
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	// compute the linear signature of a single segment
	// Each thread handles a strided portion of the output.

	const int thread_id = threadIdx.x;
	const int nthreads = blockDim.x;

	// Level 0
	if (thread_id == 0) out[0] = static_cast<T>(1.);

	// Level 1
	for (uint64_t i = thread_id; i < dimension; i += nthreads)
		out[i + 1] = increments[i];

	__syncthreads();

	// Precompute stride decomposition for incremental index tracking (avoids repeated integer division)
	// Use 32-bit integers for loop variables (faster on GPU than 64-bit)
	const unsigned int dim32 = static_cast<unsigned int>(dimension);
	const unsigned int stride_l = static_cast<unsigned int>(nthreads) / dimension;
	const unsigned int stride_r = static_cast<unsigned int>(nthreads) % dimension;
	const unsigned int base_l = static_cast<unsigned int>(thread_id) / dimension;
	const unsigned int base_r = static_cast<unsigned int>(thread_id) % dimension;

	// Higher levels
	for (uint64_t level = 2; level <= degree; ++level) {
		const T one_over_level = static_cast<T>(1) / static_cast<T>(level);
		const uint64_t prev_start = level_index[level - 1];
		const uint64_t prev_size = level_index[level] - level_index[level - 1];
		const uint64_t cur_start = level_index[level];
		const unsigned int cur_size = static_cast<unsigned int>(prev_size * dimension);

		unsigned int cur_l = base_l;
		unsigned int cur_r = base_r;
		for (unsigned int idx = static_cast<unsigned int>(thread_id); idx < cur_size; idx += static_cast<unsigned int>(nthreads)) {
			out[cur_start + idx] = out[prev_start + cur_l] * increments[cur_r] * one_over_level;
			cur_l += stride_l;
			cur_r += stride_r;
			if (cur_r >= dim32) {
				cur_r -= dim32;
				cur_l++;
			}
		}
		__syncthreads();
	}
}

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
	// Combines the current signature with the signature of a linear segment
	// (given by increments) using Horner's method.

	// Workspace layout: horner_step has size 2 * dim^degree.
	//   - buf_A = horner_step[0 .. dim^degree - 1]
	//   - buf_B = horner_step[dim^degree .. 2*dim^degree - 1]
	//   We ping-pong between buf_A and buf_B during the multiply steps
	//   to avoid read-write conflicts under parallel execution.

	const unsigned int thread_id = static_cast<unsigned int>(threadIdx.x);
	const int nthreads = blockDim.x;

	// Precompute stride decomposition for incremental index tracking (avoids repeated integer division)
	// Use 32-bit integers for loop variables (faster on GPU than 64-bit)
	const unsigned int dim32 = static_cast<unsigned int>(dimension);
	const unsigned int nthreads32 = static_cast<unsigned int>(nthreads);
	const unsigned int stride_l = nthreads32 / dim32;
	const unsigned int stride_r = nthreads32 % dim32;
	const unsigned int base_l = thread_id / dim32;
	const unsigned int base_r = thread_id % dim32;

	T* buf_A = horner_workspace;
	T* buf_B = horner_workspace + half_size;

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 1; --target_level) {

		T one_over_level = static_cast<T>(1.) / static_cast<T>(target_level);

		// Current read buffer: start with buf_A
		T* src = buf_A;
		T* dst = buf_B;

		// left_level = 0: assign increments * one_over_level into src
		for (unsigned int i = thread_id; i < dim32; i += nthreads32)
			src[i] = increments[i] * one_over_level;
		__syncthreads();

		for (int64_t left_level = 1, right_level = target_level - 1;
			left_level < target_level - 1;
			++left_level, --right_level) {

			const unsigned int left_level_size = static_cast<unsigned int>(level_index[left_level + 1] - level_index[left_level]);
			one_over_level = static_cast<T>(1.) / static_cast<T>(right_level);

			// Add sig at left_level into src (current horner data)
			const uint64_t left_start = level_index[left_level];
			for (unsigned int i = thread_id; i < left_level_size; i += nthreads32)
				src[i] += sig[left_start + i];
			__syncthreads();

			// Multiply: dst = src (x) increments * one_over_level
			const unsigned int result_size = left_level_size * dim32;
			unsigned int cur_l = base_l;
			unsigned int cur_r = base_r;
			for (unsigned int idx = thread_id; idx < result_size; idx += nthreads32) {
				dst[idx] = src[cur_l] * increments[cur_r] * one_over_level;
				cur_l += stride_l;
				cur_r += stride_r;
				if (cur_r >= dim32) {
					cur_r -= dim32;
					cur_l++;
				}
			}
			__syncthreads();

			// Swap src and dst
			T* tmp = src;
			src = dst;
			dst = tmp;
		}

		// Last iteration: left_level = target_level - 1
		{
			const unsigned int left_level_size = static_cast<unsigned int>(level_index[target_level] - level_index[target_level - 1]);

			// Add sig at target_level - 1 into src
			const uint64_t last_left_start = level_index[target_level - 1];
			for (unsigned int i = thread_id; i < left_level_size; i += nthreads32)
				src[i] += sig[last_left_start + i];
			__syncthreads();

			// Multiply and add directly into sig at target_level (right_level = 1)
			const unsigned int out_size = left_level_size * dim32;
			const uint64_t target_start = level_index[target_level];
			unsigned int cur_l = base_l;
			unsigned int cur_r = base_r;
			for (unsigned int idx = thread_id; idx < out_size; idx += nthreads32) {
				sig[target_start + idx] += src[cur_l] * increments[cur_r];
				cur_l += stride_l;
				cur_r += stride_r;
				if (cur_r >= dim32) {
					cur_r -= dim32;
					cur_l++;
				}
			}
			__syncthreads();
		}
	}

	// Update level 1
	for (unsigned int i = thread_id; i < dim32; i += nthreads32)
		sig[i + 1] += increments[i];
	__syncthreads();
}

template<typename T>
__global__ void signature_naive_ker(
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
	const int thread_id = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * sig_len;
	T* my_linear_sig = linear_sig_workspace + batch_idx * sig_len;

	// ---- Shared memory: increments + level_index ----
	extern __shared__ char smem[];
	T* increments = reinterpret_cast<T*>(smem);
	const size_t inc_bytes = dimension * sizeof(T);
	const size_t aligned_off = (inc_bytes + 7) & ~size_t(7);
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem + aligned_off);

	for (uint64_t i = thread_id; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	const T* prev_pt = my_path;
	const T* next_pt = my_path + dimension;

	// Increments for first segment
	for (uint64_t i = thread_id; i < dimension; i += nthreads)
		increments[i] = next_pt[i] - prev_pt[i];
	__syncthreads();

	// Linear signature of first segment
	linear_signature_device(increments, my_out, dimension, degree, level_index_smem);
	__syncthreads();

	if (length <= 2) return;

	for (uint64_t step = 2; step < length; ++step) {
		prev_pt = my_path + (step - 1) * dimension;
		next_pt = my_path + step * dimension;

		for (uint64_t i = thread_id; i < dimension; i += nthreads)
			increments[i] = next_pt[i] - prev_pt[i];
		__syncthreads();

		linear_signature_device(increments, my_linear_sig, dimension, degree, level_index_smem);
		__syncthreads();

		sig_combine_inplace_device(my_out, my_linear_sig, degree, level_index_smem);
		__syncthreads();
	}
}

// ---------------------------------------------------------------------------
// Per-word kernel: each thread computes ONE signature coefficient across all
// time steps. Launched once per level on separate CUDA streams.
// ---------------------------------------------------------------------------

__constant__ float c_recip_f32[13] = {
	0.f, 1.f, 0.5f, 1.f/3.f, 0.25f, 0.2f, 1.f/6.f, 1.f/7.f,
	0.125f, 1.f/9.f, 0.1f, 1.f/11.f, 1.f/12.f
};
__constant__ double c_recip_f64[13] = {
	0.0, 1.0, 0.5, 1.0/3.0, 0.25, 0.2, 1.0/6.0, 1.0/7.0,
	0.125, 1.0/9.0, 0.1, 1.0/11.0, 1.0/12.0
};

template<typename T> __device__ __forceinline__ T d_recip(int n);
template<> __device__ __forceinline__ float d_recip<float>(int n) { return c_recip_f32[n]; }
template<> __device__ __forceinline__ double d_recip<double>(int n) { return c_recip_f64[n]; }

// Batches multiple time steps into shared memory per chunk to reduce
// __syncthreads() overhead from 2 per step to 2 per chunk.
constexpr int SIG_CHUNK = 128;

template<typename T, int DEGREE>
__global__ __launch_bounds__(128)
void signature_per_word_ker(
	const T* __restrict__ path,       // [batch, length, dim]
	T* __restrict__ out,
	const int dim,
	const int steps,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride        // length * dim
) {
	static_assert(DEGREE >= 1 && DEGREE <= 12, "DEGREE must be 1-12");

	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t batch_idx = blockIdx.y;
	const bool active = word_idx < level_size;

	extern __shared__ char smem[];
	T* shared_inc = reinterpret_cast<T*>(smem);

	int letters[DEGREE];
	if (active) {
		uint64_t w = word_idx;
		for (int i = DEGREE - 1; i >= 0; --i) {
			letters[i] = static_cast<int>(w % dim);
			w /= dim;
		}
	}

	T pref[DEGREE + 1];
	T comp[DEGREE + 1];
	for (int i = 0; i <= DEGREE; ++i) { pref[i] = T(0); comp[i] = T(0); }
	pref[0] = T(1);

	const T* batch_path = path + batch_idx * path_stride;

	for (int chunk_start = 0; chunk_start < steps; chunk_start += SIG_CHUNK) {
		const int chunk_end = (chunk_start + SIG_CHUNK < steps) ? chunk_start + SIG_CHUNK : steps;
		const int chunk_len = chunk_end - chunk_start;

		// Cooperatively load increments for this chunk (compute on the fly)
		__syncthreads();
		{
			const int total_elems = chunk_len * dim;
			for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
				const int t_local = i / dim;
				const int d_idx = i - t_local * dim;
				const int base = (chunk_start + t_local) * dim;
				shared_inc[i] = batch_path[base + dim + d_idx] - batch_path[base + d_idx];
			}
		}
		__syncthreads();

		if (!active) continue;

		for (int t_local = 0; t_local < chunk_len; ++t_local) {
			const T* inc = shared_inc + t_local * dim;

			for (int sd = DEGREE; sd > 0; --sd) {
				T h = T(0);
				for (int k = 0; k < sd; ++k) {
					const T scale = inc[letters[k]] * d_recip<T>(sd - k);
					h = scale * (pref[k] + h);
				}
				T y = h - comp[sd];
				T tmp = pref[sd] + y;
				comp[sd] = (tmp - pref[sd]) - y;
				pref[sd] = tmp;
			}
		}
	}

	if (active) {
		out[batch_idx * sig_size + level_offset + word_idx] = pref[DEGREE];
	}
}

// Set level 0 (scalar 1) for all batch elements
template<typename T>
__global__ void set_sig_level0(T* out, uint64_t sig_size, uint64_t batch_size) {
	uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (b < batch_size) out[b * sig_size] = static_cast<T>(1);
}

// Cached stream pool (created once, reused across calls)
static cudaStream_t s_per_word_streams[12] = {};
static bool s_streams_initialized = false;

static void ensure_streams() {
	if (!s_streams_initialized) {
		for (int i = 0; i < 12; ++i)
			cudaStreamCreate(&s_per_word_streams[i]);
		s_streams_initialized = true;
	}
}

template<typename T>
void signature_per_word_core_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);
	const uint64_t path_stride = length * dimension;
	const int steps = static_cast<int>(length - 1);
	const int dim = static_cast<int>(dimension);

	set_sig_level0<T><<<(unsigned int)((batch_size + 255) / 256), 256>>>(
		out, sig_len, batch_size);

	uint64_t li[14];
	host_populate_level_index(li, dimension, degree + 2);

	size_t smem = SIG_CHUNK * dimension * sizeof(T);

	ensure_streams();

	for (uint64_t k = 1; k <= degree; ++k) {
		uint64_t level_size = host_power(dimension, k);
		uint64_t level_offset = li[k];
		unsigned int block = 128;
		if (level_size < 128) block = 32;
		unsigned int grid_x = (unsigned int)((level_size + block - 1) / block);
		dim3 grid(grid_x, (unsigned int)batch_size, 1);

		#define LAUNCH_DEGREE(D) \
			case D: signature_per_word_ker<T, D><<<grid, block, smem, s_per_word_streams[k-1]>>>( \
				path, out, dim, steps, sig_len, level_offset, level_size, path_stride); break;

		switch (k) {
			LAUNCH_DEGREE(1)
			LAUNCH_DEGREE(2)
			LAUNCH_DEGREE(3)
			LAUNCH_DEGREE(4)
			LAUNCH_DEGREE(5)
			LAUNCH_DEGREE(6)
			LAUNCH_DEGREE(7)
			LAUNCH_DEGREE(8)
			LAUNCH_DEGREE(9)
			LAUNCH_DEGREE(10)
			LAUNCH_DEGREE(11)
			LAUNCH_DEGREE(12)
			default: break;
		}
		#undef LAUNCH_DEGREE
	}

	for (uint64_t k = 0; k < degree; ++k)
		cudaStreamSynchronize(s_per_word_streams[k]);

	check_cuda_error();
}

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
		T one = static_cast<T>(1);
		for (uint64_t i = 0; i < batch_size; ++i)
			cudaMemcpy(out + i * sig_len, &one, sizeof(T), cudaMemcpyHostToDevice);
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

	// Choose number of threads per block based on largest level size
	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	// Shared memory: increments + level_index (aligned)
	size_t smem_size = (dimension * sizeof(T) + 7) & ~size_t(7);
	smem_size += (degree + 2) * sizeof(uint64_t);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	const size_t aligned_li_bytes = (level_index_bytes + sizeof(T) - 1) / sizeof(T) * sizeof(T);

	// Use per-word kernel (fast path)
	if (horner) {
		signature_per_word_core_<T>(path, out, batch_size, dimension, length, degree);
		return;
	}
	else {
		// Single allocation: level_index + linear sig workspace
		const size_t workspace_bytes = batch_size * sig_len * sizeof(T);

		char* d_alloc;
		cudaMalloc(&d_alloc, aligned_li_bytes + workspace_bytes);
		uint64_t* d_level_index = reinterpret_cast<uint64_t*>(d_alloc);
		T* d_linear_sig = reinterpret_cast<T*>(d_alloc + aligned_li_bytes);
		cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);

		signature_naive_ker<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
			path, out, d_level_index,
			dimension, length, degree, sig_len, path_flat_len,
			d_linear_sig
		);

		cudaDeviceSynchronize();
		cudaFree(d_alloc);
	}

	check_cuda_error();
}

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
	// Given sig1, sig2 are two signatures, and sig_concat = sig1 * sig2.
	// sig_concat_deriv is dF/d(sig_concat).
	// Computes dF/d(sig1) into sig_concat_deriv and dF/d(sig2) into sig2_deriv.

	const int thread_id = threadIdx.x;
	const int nthreads = blockDim.x;

	// Use 32-bit integers for loop variables (faster on GPU than 64-bit)
	const unsigned int sig_len32 = static_cast<unsigned int>(sig_len);

	// Copy sig_concat_deriv to sig2_deriv
	for (unsigned int i = thread_id; i < sig_len32; i += nthreads)
		sig2_deriv[i] = sig_concat_deriv[i];
	__syncthreads();

	// First loop: accumulate sig2_deriv using sig1 values
	// For each (level, left_level, right_level):
	//   sig2_deriv[right_start + r] += sum_{l} sig_concat_deriv[level_start + l*right_size + r] * sig1[left_start + l]
	// Parallelize over r (the output dimension) — no atomics needed.
	// No __syncthreads needed: reads only from sig_concat_deriv and sig1 (both unmodified),
	// writes only to sig2_deriv at unique per-thread r positions.
	for (uint64_t level = degree; level > 0; --level) {
		for (uint64_t left_level = level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			const unsigned int left_size = static_cast<unsigned int>(level_index[left_level + 1] - level_index[left_level]);
			const unsigned int right_size = static_cast<unsigned int>(level_index[right_level + 1] - level_index[right_level]);
			const uint64_t level_start = level_index[level];
			const uint64_t left_start = level_index[left_level];
			const uint64_t right_start = level_index[right_level];

			for (unsigned int r = thread_id; r < right_size; r += nthreads) {
				T acc = static_cast<T>(0);
				for (unsigned int l = 0; l < left_size; ++l) {
					acc += sig_concat_deriv[level_start + l * right_size + r]
						* sig1[left_start + l];
				}
				sig2_deriv[right_start + r] += acc;
			}
		}
	}

	// Second loop: accumulate sig1 (sig_concat_deriv) derivatives using sig2 values
	// For each (left_level, level, right_level):
	//   sig_concat_deriv[left_start + l] += sum_{r} sig_concat_deriv[level_start + l*right_size + r] * sig2[right_start + r]
	// Parallelize over l (the output dimension) — no atomics needed.
	// Sync needed between left_level iterations (left_level=k writes to level k,
	// left_level=k-1 reads from level k), but not between inner iterations.
	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		__syncthreads();
		for (uint64_t level = left_level + 1, right_level = 1;
			level <= degree;
			++level, ++right_level) {

			const unsigned int left_size = static_cast<unsigned int>(level_index[left_level + 1] - level_index[left_level]);
			const unsigned int right_size = static_cast<unsigned int>(level_index[right_level + 1] - level_index[right_level]);
			const uint64_t level_start = level_index[level];
			const uint64_t left_start = level_index[left_level];
			const uint64_t right_start = level_index[right_level];

			for (unsigned int l = thread_id; l < left_size; l += nthreads) {
				T acc = static_cast<T>(0);
				for (unsigned int r = 0; r < right_size; ++r) {
					acc += sig_concat_deriv[level_start + static_cast<uint64_t>(l) * right_size + r]
						* sig2[right_start + r];
				}
				sig_concat_deriv[left_start + l] += acc;
			}
		}
	}
}

template<typename T>
__device__ void linear_sig_deriv_to_increment_deriv_device(
	const T* __restrict__ sig,
	T* __restrict__ sig_deriv,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* __restrict__ level_index
) {
	// Given sig is the signature of a line segment [a,b] and sig_deriv is
	// dF/d(sig), computes dF/d(b-a) and writes it into sig_deriv[1..dimension].

	const int thread_id = threadIdx.x;
	const int nthreads = blockDim.x;

	// Use 32-bit integers for loop variables (faster on GPU than 64-bit)
	const unsigned int dim32 = static_cast<unsigned int>(dimension);

	for (uint64_t level = degree; level > 1; --level) {
		const T one_over_level = static_cast<T>(1.) / static_cast<T>(level);
		const unsigned int level_size = static_cast<unsigned int>(level_index[level] - level_index[level - 1]);
		const uint64_t level_start = level_index[level];
		const uint64_t prev_start = level_index[level - 1];

		// Pass 1: parallelize over j (each j owns unique offs2)
		// Compute offs2_acc contribution (no atomics needed).
		for (unsigned int j = thread_id; j < level_size; j += nthreads) {
			const uint64_t offs1 = level_start + static_cast<uint64_t>(dim32) * j - 1;
			const uint64_t offs2 = prev_start + j;
			T offs2_acc = static_cast<T>(0);
			for (unsigned int dd = 1; dd <= dim32; ++dd) {
				offs2_acc += sig[dd] * sig_deriv[offs1 + dd] * one_over_level;
			}
			sig_deriv[offs2] += offs2_acc;
		}
		__syncthreads();

		// Pass 2: accumulate dd contributions locally per thread, then flush
		// with one atomicAdd per dd per thread (reduces atomics from
		// dimension*level_size to dimension*min(level_size, nthreads)).
		for (unsigned int dd = 1; dd <= dim32; ++dd) {
			T local_acc = static_cast<T>(0.);
			for (unsigned int j = thread_id; j < level_size; j += nthreads) {
				const uint64_t offs1 = level_start + static_cast<uint64_t>(dim32) * j - 1;
				const uint64_t offs2 = prev_start + j;
				local_acc += sig[offs2] * sig_deriv[offs1 + dd] * one_over_level;
			}
			if (local_acc != static_cast<T>(0.))
				myAtomicAdd(&sig_deriv[dd], local_acc);
		}
		__syncthreads();
	}
}

template<typename T>
__global__ void sig_backprop_ker(
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
	const int thread_id = threadIdx.x;
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

	// ---- Shared memory: increments + level_index ----
	extern __shared__ char smem[];
	T* increments = reinterpret_cast<T*>(smem);
	const size_t inc_bytes = dimension * sizeof(T);
	const size_t aligned_off = (inc_bytes + 7) & ~size_t(7);
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem + aligned_off);

	for (uint64_t i = thread_id; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// Zero the output
	for (uint64_t i = thread_id; i < path_flat_len; i += nthreads)
		my_out[i] = static_cast<T>(0);
	__syncthreads();

	// Iterate BACKWARDS through path segments
	// Segment indices: from (length-1) down to 1
	// For segment step, prev_pt = path[step-1], next_pt = path[step]
	for (int64_t step = static_cast<int64_t>(length) - 1; step >= 1; --step) {
		const T* prev_pt = my_path + (step - 1) * dimension;
		const T* next_pt = my_path + step * dimension;

		// Compute FORWARD increments = next_pt - prev_pt (for linear signature)
		for (uint64_t i = thread_id; i < dimension; i += nthreads)
			increments[i] = next_pt[i] - prev_pt[i];
		__syncthreads();

		// Compute linear signature of the FORWARD segment
		linear_signature_device(increments, linear_sig, dimension, degree, level_index_smem);
		__syncthreads();

		// Negate increments to get REVERSED = prev_pt - next_pt (for Horner uncombine)
		for (uint64_t i = thread_id; i < dimension; i += nthreads)
			increments[i] = -increments[i];
		__syncthreads();

		// Horner step to "uncombine": sig = sig uncombined with reversed segment
		signature_horner_step_device(my_sig, increments, dimension, degree,
			level_index_smem, horner_ws, horner_half_size);
		__syncthreads();

		// uncombine_sig_deriv: split derivatives
		// After this: my_sig_derivs contains dF/d(sig1), local_derivs contains dF/d(sig2)
		uncombine_sig_deriv_device(my_sig, linear_sig, my_sig_derivs, local_derivs,
			dimension, degree, sig_len, level_index_smem);
		__syncthreads();

		// linear_sig_deriv_to_increment_deriv: convert dF/d(linear_sig) to dF/d(increment)
		// Result is in local_derivs[1..dimension]
		linear_sig_deriv_to_increment_deriv_device(linear_sig, local_derivs,
			dimension, degree, level_index_smem);
		__syncthreads();

		// Accumulate into output: pos += s, neg -= s
		// pos = out + step * dimension, neg = out + (step-1) * dimension
		// s = local_derivs + 1
		T* pos = my_out + step * dimension;
		T* neg = my_out + (step - 1) * dimension;
		for (uint64_t d = thread_id; d < dimension; d += nthreads) {
			T s = local_derivs[1 + d];
			pos[d] += s;
			neg[d] -= s;
		}
		__syncthreads();
	}
}

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

	// Build level_index on host
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	// Workspace per batch element:
	//   local_derivs: sig_len
	//   linear_signature: sig_len
	//   horner_workspace: 2 * horner_half_size
	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	uint64_t horner_half_size = max_level_size;
	uint64_t workspace_per_batch = 2 * sig_len + 2 * horner_half_size;

	// Merge all device allocations into a single cudaMalloc
	// Layout: [level_index (aligned to 256)] [sig_derivs_copy] [sig_copy] [workspace]
	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	const size_t level_index_padded = (level_index_bytes + 255) & ~size_t(255);
	const size_t sig_derivs_bytes = batch_size * sig_len * sizeof(T);
	const size_t sig_copy_bytes = batch_size * sig_len * sizeof(T);
	const size_t workspace_bytes = batch_size * workspace_per_batch * sizeof(T);
	const size_t total_bytes = level_index_padded + sig_derivs_bytes + sig_copy_bytes + workspace_bytes;

	char* d_merged;
	cudaMalloc(&d_merged, total_bytes);

	uint64_t* d_level_index = reinterpret_cast<uint64_t*>(d_merged);
	T* d_sig_derivs_copy = reinterpret_cast<T*>(d_merged + level_index_padded);
	T* d_sig_copy = reinterpret_cast<T*>(d_merged + level_index_padded + sig_derivs_bytes);
	T* d_workspace = reinterpret_cast<T*>(d_merged + level_index_padded + sig_derivs_bytes + sig_copy_bytes);

	cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sig_derivs_copy, sig_derivs, sig_derivs_bytes, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_sig_copy, sig, sig_copy_bytes, cudaMemcpyDeviceToDevice);
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	// Shared memory: increments + level_index (aligned)
	size_t smem_size = (dimension * sizeof(T) + 7) & ~size_t(7);
	smem_size += (degree + 2) * sizeof(uint64_t);

	sig_backprop_ker<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		path, out, d_sig_derivs_copy, d_sig_copy,
		d_level_index,
		dimension, length, degree, sig_len, path_flat_len,
		d_workspace, workspace_per_batch, horner_half_size
	);

	cudaFree(d_merged);
	check_cuda_kernel_launch();
}

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


#include "cu_macros.h"


extern "C" {

	CUSIG_API int signature_cuda_f(
		const float* path, float* out,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time,
		bool horner
	) noexcept {
		CUSIG_SAFE_CALL(signature_cuda_<float>(path, out, 1, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CUSIG_API int signature_cuda_d(
		const double* path, double* out,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time,
		bool horner
	) noexcept {
		CUSIG_SAFE_CALL(signature_cuda_<double>(path, out, 1, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CUSIG_API int batch_signature_cuda_f(
		const float* path, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time,
		bool horner
	) noexcept {
		CUSIG_SAFE_CALL(signature_cuda_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	CUSIG_API int batch_signature_cuda_d(
		const double* path, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time,
		bool horner
	) noexcept {
		CUSIG_SAFE_CALL(signature_cuda_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner));
	}

	// =====================================================================
	// backprop
	// =====================================================================

	CUSIG_API int sig_backprop_cuda_f(
		const float* path, float* out,
		const float* sig_derivs, const float* sig,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time
	) noexcept {
		CUSIG_SAFE_CALL(sig_backprop_cuda_<float>(path, out, sig_derivs, sig, 1, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CUSIG_API int sig_backprop_cuda_d(
		const double* path, double* out,
		const double* sig_derivs, const double* sig,
		uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time
	) noexcept {
		CUSIG_SAFE_CALL(sig_backprop_cuda_<double>(path, out, sig_derivs, sig, 1, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_sig_backprop_cuda_f(
		const float* path, float* out,
		const float* sig_derivs, const float* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time
	) noexcept {
		CUSIG_SAFE_CALL(sig_backprop_cuda_<float>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_sig_backprop_cuda_d(
		const double* path, double* out,
		const double* sig_derivs, const double* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time
	) noexcept {
		CUSIG_SAFE_CALL(sig_backprop_cuda_<double>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time));
	}

}
