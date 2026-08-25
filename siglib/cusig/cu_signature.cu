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
#include "cu_runtime_utils.h"
#include "cu_signature.h"
#include "cu_sig_combine.h"
#include "cu_atomic.h"
#include "cu_path_transforms.h"
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>

// linear_signature_device is defined in cu_sig_combine.h

template<typename T>
__global__ void signature_naive_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	const uint64_t* __restrict__ d_level_index,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t sig_stride,
	uint64_t path_flat_len,
	T* __restrict__ linear_sig_workspace,  // [chunk_size * sig_stride]
	uint64_t workspace_stride,
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int thread_id = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * sig_stride;
	T* my_linear_sig = linear_sig_workspace + local_batch_idx * workspace_stride;

	// ---- Shared memory: increments + level_index ----
	extern __shared__ char smem[];
	T* increments = reinterpret_cast<T*>(smem);
	const size_t inc_bytes = dimension * sizeof(T);
	const size_t aligned_off = (inc_bytes + 7) & ~size_t(7);
	uint64_t* level_index_smem = reinterpret_cast<uint64_t*>(smem + aligned_off);

	for (uint64_t i = thread_id; i < degree + 2; i += nthreads)
		level_index_smem[i] = d_level_index[i];
	__syncthreads();

	// For scalar_term=false, shift offsets by -1 so that level k starts at
	// level_index[k]-1 (level 1 starts at 0).
	if (!scalar_term) {
		if (thread_id == 0) {
			for (uint64_t i = 1; i < degree + 2; ++i)
				level_index_smem[i] -= 1;
		}
		__syncthreads();
	}

	const T* prev_pt = my_path;
	const T* next_pt = my_path + dimension;

	// Increments for first segment
	for (uint64_t i = thread_id; i < dimension; i += nthreads)
		increments[i] = next_pt[i] - prev_pt[i];
	__syncthreads();

	// Linear signature of first segment
	linear_signature_device(increments, my_out, dimension, degree, level_index_smem, scalar_term);
	__syncthreads();

	if (length <= 2) return;

	for (uint64_t step = 2; step < length; ++step) {
		prev_pt = my_path + (step - 1) * dimension;
		next_pt = my_path + step * dimension;

		for (uint64_t i = thread_id; i < dimension; i += nthreads)
			increments[i] = next_pt[i] - prev_pt[i];
		__syncthreads();

		linear_signature_device(increments, my_linear_sig, dimension, degree, level_index_smem, scalar_term);
		__syncthreads();

		sig_combine_inplace_device(my_out, my_linear_sig, degree, level_index_smem, scalar_term);
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

// Runtime reciprocal for the generic (non-template) fallback kernels
template<typename T> __device__ __forceinline__ T d_recip_rt(int n) {
	return (n <= 12) ? d_recip<T>(n) : static_cast<T>(1) / static_cast<T>(n);
}

// Chunk sizes for batching time steps into shared memory (reduces sync overhead).
constexpr int SIG_CHUNK = 128;   // forward kernel
constexpr int BWD_CHUNK = 32;    // backward kernel (needs more shared mem for reduction)

template<bool use_shared_cache, typename T>
__device__ __forceinline__ T signature_increment_value_(
	const T* increment,
	const T* batch_path,
	int step,
	int dim,
	int letter
) {
	if constexpr (use_shared_cache)
		return increment[letter];
	return batch_path[(step + 1) * dim + letter]
		- batch_path[step * dim + letter];
}

template<typename T, int DEGREE, bool use_shared_cache>
__global__ void signature_per_word_ker(
	const T* __restrict__ path,       // [batch, length, dim]
	T* __restrict__ out,
	const int dim,
	const int steps,
	const int chunk_size,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride,       // length * dim
	const bool scalar_term,
	const uint64_t batch_offset,
	const uint64_t batch_chunk_size
) {
	static_assert(DEGREE >= 1 && DEGREE <= 13, "DEGREE must be 1-13");

	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
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
	for (int i = 0; i <= DEGREE; ++i) pref[i] = T(0);
	pref[0] = T(1);

	const T* batch_path = path + batch_idx * path_stride;

	for (int chunk_start = 0; chunk_start < steps; chunk_start += chunk_size) {
		const int chunk_end = (chunk_start + chunk_size < steps)
			? chunk_start + chunk_size : steps;
		const int chunk_len = chunk_end - chunk_start;

		__syncthreads();
		if constexpr (use_shared_cache) {
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
			const T* increment = shared_inc + t_local * dim;
			for (int sd = DEGREE; sd > 0; --sd) {
				T h = T(0);
				for (int k = 0; k < sd; ++k) {
					const T scale = signature_increment_value_<use_shared_cache>(
						increment, batch_path, chunk_start + t_local,
						dim, letters[k])
						* d_recip<T>(sd - k);
					h = scale * (pref[k] + h);
				}
				pref[sd] += h;
			}
		}
	}

	if (active) {
		out[batch_idx * sig_size + level_offset + word_idx] = pref[DEGREE];
		// Level-1 kernel also writes the constant term (level 0) when present.
		if constexpr (DEGREE == 1) {
			if (word_idx == 0 && scalar_term)
				out[batch_idx * sig_size] = T(1);
		}
	}
}

// ---------------------------------------------------------------------------
// Generic (non-template) per-word forward kernel for degree > 13.
// Same algorithm as the template version but with runtime degree parameter.
// ---------------------------------------------------------------------------

constexpr int MAX_GENERIC_DEGREE = 64;

template<typename T, bool use_shared_cache>
__global__ void signature_per_word_generic_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	const int dim,
	const int steps,
	const int chunk_size,
	const int degree,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride,
	const bool /*scalar_term*/,  // only relevant for the k==1 scalar write in the templated kernel
	const uint64_t batch_offset,
	const uint64_t batch_chunk_size
) {
	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const bool active = word_idx < level_size;

	extern __shared__ char smem[];
	T* shared_inc = reinterpret_cast<T*>(smem);

	int letters[MAX_GENERIC_DEGREE];
	if (active) {
		uint64_t w = word_idx;
		for (int i = degree - 1; i >= 0; --i) {
			letters[i] = static_cast<int>(w % dim);
			w /= dim;
		}
	}

	T pref[MAX_GENERIC_DEGREE + 1];
	for (int i = 0; i <= degree; ++i) pref[i] = T(0);
	pref[0] = T(1);

	const T* batch_path = path + batch_idx * path_stride;

	for (int chunk_start = 0; chunk_start < steps; chunk_start += chunk_size) {
		const int chunk_end = (chunk_start + chunk_size < steps)
			? chunk_start + chunk_size : steps;
		const int chunk_len = chunk_end - chunk_start;

		__syncthreads();
		if constexpr (use_shared_cache) {
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
			const T* increment = shared_inc + t_local * dim;
			for (int sd = degree; sd > 0; --sd) {
				T h = T(0);
				for (int k = 0; k < sd; ++k) {
					const T scale = signature_increment_value_<use_shared_cache>(
						increment, batch_path, chunk_start + t_local,
						dim, letters[k])
						* d_recip_rt<T>(sd - k);
					h = scale * (pref[k] + h);
				}
				pref[sd] += h;
			}
		}
	}

	if (active) {
		out[batch_idx * sig_size + level_offset + word_idx] = pref[degree];
	}
}

// ---------------------------------------------------------------------------
// Generic (non-template) per-word backward kernel for degree > 12.
// ---------------------------------------------------------------------------

template<typename T, bool use_shared_cache>
__global__ __launch_bounds__(128)
void sig_backprop_per_word_generic_ker(
	const T* __restrict__ path,
	const T* __restrict__ sig,
	const T* __restrict__ sig_grads,
	T* __restrict__ inc_grads,
	const int dim,
	const int steps,
	const int chunk_size,
	const int degree,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride,
	const bool scalar_term,
	const uint64_t batch_offset,
	const uint64_t batch_chunk_size
) {
	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const bool active = word_idx < level_size;

	extern __shared__ char smem[];
	T* shared_inc = reinterpret_cast<T*>(smem);
	const unsigned num_warps = blockDim.x >> 5;
	T* shared_letter_grads = shared_inc + chunk_size * dim;

	if constexpr (use_shared_cache) {
		for (int i = threadIdx.x; i < dim * (int)num_warps; i += blockDim.x)
			shared_letter_grads[i] = T(0);
	}

	int letters[MAX_GENERIC_DEGREE];
	if (active) {
		uint64_t w = word_idx;
		for (int i = degree - 1; i >= 0; --i) {
			letters[i] = static_cast<int>(w % dim);
			w /= dim;
		}
	}

	const T* batch_sig = sig + batch_idx * sig_size;
	const T grad_val = active ? sig_grads[batch_idx * sig_size + level_offset + word_idx] : T(0);

	T pref[MAX_GENERIC_DEGREE + 1];
	pref[0] = T(1);
	if (active) {
		// In scalar_term layout, level 1 starts at index 1 (skip the scalar at 0).
		// In no-scalar layout, level 1 starts at index 0.
		uint64_t off = scalar_term ? 1 : 0;
		uint64_t d_pow = static_cast<uint64_t>(dim);
		uint64_t pref_word = 0;
		for (int lvl = 1; lvl < degree; ++lvl) {
			pref_word = pref_word * dim + letters[lvl - 1];
			pref[lvl] = batch_sig[off + pref_word];
			off += d_pow;
			d_pow *= dim;
		}
		pref[degree] = batch_sig[level_offset + word_idx];
	} else {
		for (int i = 1; i <= degree; ++i) pref[i] = T(0);
	}

	T suf[MAX_GENERIC_DEGREE + 1];
	suf[0] = T(1);
	for (int i = 1; i <= degree; ++i) suf[i] = T(0);

	const T* batch_path = path + batch_idx * path_stride;
	T* batch_inc_grad = inc_grads
		+ local_batch_idx * static_cast<uint64_t>(steps) * dim;

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned lane = threadIdx.x & 31;
	for (int chunk_end = steps; chunk_end > 0; chunk_end -= chunk_size) {
		const int chunk_start = (chunk_end - chunk_size > 0)
			? chunk_end - chunk_size : 0;
		const int chunk_len = chunk_end - chunk_start;

		__syncthreads();
		if constexpr (use_shared_cache) {
			const int total_elems = chunk_len * dim;
			for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
				const int t_local = i / dim;
				const int d_idx = i - t_local * dim;
				const int t = chunk_start + t_local;
				shared_inc[i] = batch_path[(t + 1) * dim + d_idx] - batch_path[t * dim + d_idx];
			}
		}
		__syncthreads();

		for (int t_local = chunk_len - 1; t_local >= 0; --t_local) {
			const int t = chunk_start + t_local;
			const T* increment = shared_inc + t_local * dim;
			if (active) {
				for (int sd = degree; sd > 0; --sd) {
					T h = T(0);
					for (int k = 0; k < sd; ++k) {
						const T inc_value = signature_increment_value_<use_shared_cache>(
							increment, batch_path, t, dim, letters[k]);
						h = (-inc_value * d_recip_rt<T>(sd - k)) * (pref[k] + h);
					}
					pref[sd] += h;
				}
			}

			T letter_grads_local[MAX_GENERIC_DEGREE];
			for (int i = 0; i < degree; ++i) letter_grads_local[i] = T(0);

			if (active) {
				T left_prod[MAX_GENERIC_DEGREE];
				for (int pref_len = 0; pref_len < degree; ++pref_len) {
					const T pref_val = pref[pref_len];
					left_prod[pref_len] = T(1);
					for (int lp = pref_len + 1; lp < degree; ++lp) {
						const T inc_value = signature_increment_value_<use_shared_cache>(
							increment, batch_path, t, dim, letters[lp - 1]);
						left_prod[lp] = left_prod[lp - 1]
							* inc_value
							* d_recip_rt<T>(lp - pref_len + 1);
					}

					T right = suf[0];
					for (int lp = degree - 1; lp >= pref_len; --lp) {
						letter_grads_local[lp] += pref_val * left_prod[lp] * right;
						if (lp > pref_len) {
							const T inc_value = signature_increment_value_<use_shared_cache>(
								increment, batch_path, t, dim, letters[lp]);
							right = suf[degree - lp]
								+ inc_value
								* d_recip_rt<T>(lp - pref_len + 1) * right;
						}
					}
				}
			}

			if constexpr (use_shared_cache) {
				for (int letter = 0; letter < dim; ++letter) {
					T val = T(0);
					if (active) {
						for (int lp = 0; lp < degree; ++lp) {
							if (letters[lp] == letter)
								val += letter_grads_local[lp];
						}
					}
					val *= grad_val;
					val += __shfl_down_sync(0xffffffff, val, 16);
					val += __shfl_down_sync(0xffffffff, val, 8);
					val += __shfl_down_sync(0xffffffff, val, 4);
					val += __shfl_down_sync(0xffffffff, val, 2);
					val += __shfl_down_sync(0xffffffff, val, 1);
					if (lane == 0)
						shared_letter_grads[letter * num_warps + warp_id] = val;
				}
			}
			else if (active) {
				for (int lp = 0; lp < degree; ++lp) {
					myAtomicAdd(
						&batch_inc_grad[static_cast<uint64_t>(t) * dim + letters[lp]],
						letter_grads_local[lp] * grad_val);
				}
			}

			__syncthreads();

			if constexpr (use_shared_cache) {
				for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
					T sum = T(0);
					for (unsigned w = 0; w < num_warps; ++w)
						sum += shared_letter_grads[letter * num_warps + w];
					if (sum != T(0))
						myAtomicAdd(
							&batch_inc_grad[static_cast<uint64_t>(t) * dim + letter], sum);
				}
			}

			if (active) {
				for (int m = degree - 1; m > 0; --m) {
					const int base_pos = degree - m;
					T h = T(0);
					for (int p = m; p >= 1; --p) {
						int lp = base_pos + (p - 1);
						const T inc_value = signature_increment_value_<use_shared_cache>(
							increment, batch_path, t, dim, letters[lp]);
						h = (inc_value * d_recip_rt<T>(p)) * (suf[m - p] + h);
					}
					suf[m] += h;
				}
			}

			__syncthreads();
		}
	}
}

// ---------------------------------------------------------------------------
// Per-word backward kernel: computes dL/d(path_increments) for one level.
// Each thread handles one word, iterating backward through time.
// Accumulates into inc_grads via warp reduction + atomicAdd.
// ---------------------------------------------------------------------------

template<typename T, int DEGREE, bool use_shared_cache>
__global__ __launch_bounds__(128)
void sig_backprop_per_word_ker(
	const T* __restrict__ path,           // [batch, length, dim]
	const T* __restrict__ sig,            // [batch, sig_size] (forward signature)
	const T* __restrict__ sig_grads,      // [batch, sig_size] (incoming gradient)
	T* __restrict__ inc_grads,            // [batch, steps, dim] (output, accumulated via atomicAdd)
	const int dim,
	const int steps,
	const int chunk_size,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride,
	const bool scalar_term,
	const uint64_t batch_offset,
	const uint64_t batch_chunk_size
) {
	static_assert(DEGREE >= 1 && DEGREE <= 12, "DEGREE must be 1-12");

	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const bool active = word_idx < level_size;

	extern __shared__ char smem[];
	T* shared_inc = reinterpret_cast<T*>(smem);
	const unsigned num_warps = blockDim.x >> 5;
	T* shared_letter_grads = shared_inc + chunk_size * dim;

	// Zero the reduction workspace
	if constexpr (use_shared_cache) {
		for (int i = threadIdx.x; i < dim * (int)num_warps; i += blockDim.x)
			shared_letter_grads[i] = T(0);
	}

	// Unpack word letters
	int letters[DEGREE];
	if (active) {
		uint64_t w = word_idx;
		for (int i = DEGREE - 1; i >= 0; --i) {
			letters[i] = static_cast<int>(w % dim);
			w /= dim;
		}
	}

	// Incoming gradient for this word
	const T* batch_sig = sig + batch_idx * sig_size;
	const T grad_val = active ? sig_grads[batch_idx * sig_size + level_offset + word_idx] : T(0);

	T pref[DEGREE + 1];
	pref[0] = T(1);
	if (active) {
		// Load prefix signature values from the forward signature array
		// pref[k] = S(prefix word of length k)
		// With scalar_term: level 1 starts at index 1; without: it starts at 0.
		uint64_t off = scalar_term ? 1 : 0;
		uint64_t d_pow = static_cast<uint64_t>(dim);
		uint64_t pref_word = 0;
		for (int lvl = 1; lvl < DEGREE; ++lvl) {
			pref_word = pref_word * dim + letters[lvl - 1];
			pref[lvl] = batch_sig[off + pref_word];
			off += d_pow;
			d_pow *= dim;
		}
		pref[DEGREE] = batch_sig[level_offset + word_idx];
	} else {
		for (int i = 1; i <= DEGREE; ++i) pref[i] = T(0);
	}

	T suf[DEGREE + 1];
	suf[0] = T(1);
	for (int i = 1; i <= DEGREE; ++i) suf[i] = T(0);

	const T* batch_path = path + batch_idx * path_stride;
	T* batch_inc_grad = inc_grads
		+ local_batch_idx * static_cast<uint64_t>(steps) * dim;

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned lane = threadIdx.x & 31;

	for (int chunk_end = steps; chunk_end > 0; chunk_end -= chunk_size) {
		const int chunk_start = (chunk_end - chunk_size > 0)
			? chunk_end - chunk_size : 0;
		const int chunk_len = chunk_end - chunk_start;

		// Load all increments for this chunk into shared memory
		__syncthreads();
		if constexpr (use_shared_cache) {
			const int total_elems = chunk_len * dim;
			for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
				const int t_local = i / dim;
				const int d_idx = i - t_local * dim;
				const int t = chunk_start + t_local;
				shared_inc[i] = batch_path[(t + 1) * dim + d_idx] - batch_path[t * dim + d_idx];
			}
		}
		__syncthreads();

		// Process steps in this chunk backward (no increment-load syncs needed)
		for (int t_local = chunk_len - 1; t_local >= 0; --t_local) {
			const int t = chunk_start + t_local;
			const T* increment = shared_inc + t_local * dim;
			if (active) {
				for (int sd = DEGREE; sd > 0; --sd) {
					T h = T(0);
					for (int k = 0; k < sd; ++k) {
						const T inc_value = signature_increment_value_<use_shared_cache>(
							increment, batch_path, t, dim, letters[k]);
						h = (-inc_value * d_recip<T>(sd - k)) * (pref[k] + h);
					}
					pref[sd] += h;
				}
			}

			// Compute per-letter gradients
			T letter_grads_local[DEGREE];
			for (int i = 0; i < DEGREE; ++i) letter_grads_local[i] = T(0);

			if (active) {
				T left_prod[DEGREE];
				for (int pref_len = 0; pref_len < DEGREE; ++pref_len) {
					const T pref_val = pref[pref_len];
					left_prod[pref_len] = T(1);
					for (int lp = pref_len + 1; lp < DEGREE; ++lp) {
						const T inc_value = signature_increment_value_<use_shared_cache>(
							increment, batch_path, t, dim, letters[lp - 1]);
						left_prod[lp] = left_prod[lp - 1]
							* inc_value
							* d_recip<T>(lp - pref_len + 1);
					}

					T right = suf[0];
					for (int lp = DEGREE - 1; lp >= pref_len; --lp) {
						letter_grads_local[lp] += pref_val * left_prod[lp] * right;
						if (lp > pref_len) {
							const T inc_value = signature_increment_value_<use_shared_cache>(
								increment, batch_path, t, dim, letters[lp]);
							right = suf[DEGREE - lp]
								+ inc_value
								* d_recip<T>(lp - pref_len + 1) * right;
						}
					}
				}
			}

			if constexpr (!use_shared_cache) {
				if (active) {
					for (int lp = 0; lp < DEGREE; ++lp) {
						myAtomicAdd(
							&batch_inc_grad[static_cast<uint64_t>(t) * dim + letters[lp]],
							letter_grads_local[lp] * grad_val);
					}
				}
			}
			else if (dim <= 4) {
				for (int letter = 0; letter < dim; ++letter) {
					T val = T(0);
					if (active) {
						for (int lp = 0; lp < DEGREE; ++lp) {
							if (letters[lp] == letter) {
								val += letter_grads_local[lp];
							}
						}
					}
					val *= grad_val;
					val += __shfl_down_sync(0xffffffff, val, 16);
					val += __shfl_down_sync(0xffffffff, val, 8);
					val += __shfl_down_sync(0xffffffff, val, 4);
					val += __shfl_down_sync(0xffffffff, val, 2);
					val += __shfl_down_sync(0xffffffff, val, 1);
					if (lane == 0) {
						shared_letter_grads[letter * num_warps + warp_id] = val;
					}
				}
			} else if (active) {
				for (int lp = 0; lp < DEGREE; ++lp) {
					myAtomicAdd(
						&shared_letter_grads[letters[lp] * num_warps + warp_id],
						letter_grads_local[lp] * grad_val
					);
				}
			}

			__syncthreads();

			if constexpr (use_shared_cache) {
				for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
					T sum = T(0);
					for (unsigned w = 0; w < num_warps; ++w) {
						const uint64_t index = letter * num_warps + w;
						sum += shared_letter_grads[index];
						shared_letter_grads[index] = T(0);
					}
					if (sum != T(0))
						myAtomicAdd(
							&batch_inc_grad[static_cast<uint64_t>(t) * dim + letter], sum);
				}
			}

			// Forward suffix update
			if (active) {
				for (int m = DEGREE - 1; m > 0; --m) {
					const int base_pos = DEGREE - m;
					T h = T(0);
					for (int p = m; p >= 1; --p) {
						int lp = base_pos + (p - 1);
						const T inc_value = signature_increment_value_<use_shared_cache>(
							increment, batch_path, t, dim, letters[lp]);
						h = (inc_value * d_recip<T>(p)) * (suf[m - p] + h);
					}
					suf[m] += h;
				}
			}

			__syncthreads();
		}
	}
}

template<typename T>
__global__ void sig_dense_chen_prepare_step_ker(
	const T* __restrict__ path,
	T* __restrict__ state,
	T* __restrict__ exp_values,
	T* __restrict__ increments,
	int dim,
	int step,
	int degree,
	uint64_t state_size,
	uint64_t exp_size,
	uint64_t path_stride
) {
	__shared__ uint64_t powers[MAX_GENERIC_DEGREE + 1];
	__shared__ uint64_t level_starts[MAX_GENERIC_DEGREE + 1];
	extern __shared__ char smem[];
	T* increment = reinterpret_cast<T*>(smem);

	if (threadIdx.x == 0) {
		powers[0] = 1;
		level_starts[0] = 0;
		for (int level = 1; level <= degree; ++level) {
			powers[level] = powers[level - 1] * static_cast<uint64_t>(dim);
			level_starts[level] = level == 1 ? 0
				: level_starts[level - 1] + powers[level - 1];
		}
	}
	__syncthreads();

	const uint64_t batch_idx = blockIdx.x;
	const T* batch_path = path + batch_idx * path_stride;
	T* batch_state = state + batch_idx * state_size;
	T* batch_exp = exp_values + batch_idx * exp_size;
	T* batch_increment = increments + batch_idx * dim;
	for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
		const T value = batch_path[
			static_cast<uint64_t>(step + 1) * dim + letter]
			- batch_path[static_cast<uint64_t>(step) * dim + letter];
		increment[letter] = value;
		batch_increment[letter] = value;
	}
	__syncthreads();

	for (int output_level = degree; output_level >= 2; --output_level) {
		for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
			batch_exp[level_starts[1] + letter] =
				batch_state[level_starts[1] + letter]
				- increment[letter] * d_recip_rt<T>(output_level);
		}
		__syncthreads();
		for (int scratch_level = 2; scratch_level < output_level;
			++scratch_level) {
			const uint64_t level_size = powers[scratch_level];
			const T scale = -d_recip_rt<T>(
				output_level - scratch_level + 1);
			for (uint64_t word = threadIdx.x; word < level_size;
				word += blockDim.x) {
				batch_exp[level_starts[scratch_level] + word] =
					batch_state[level_starts[scratch_level] + word]
					+ batch_exp[level_starts[scratch_level - 1]
						+ word / dim] * increment[word % dim] * scale;
			}
			__syncthreads();
		}
		const uint64_t output_size = powers[output_level];
		for (uint64_t word = threadIdx.x; word < output_size;
			word += blockDim.x) {
			batch_state[level_starts[output_level] + word] -=
				batch_exp[level_starts[output_level - 1] + word / dim]
				* increment[word % dim];
		}
		__syncthreads();
	}
	for (int letter = threadIdx.x; letter < dim; letter += blockDim.x)
		batch_state[level_starts[1] + letter] -= increment[letter];
}

template<typename T>
__global__ void sig_dense_chen_pullback_step_ker(
	const T* __restrict__ state,
	T* __restrict__ scratch,
	T* __restrict__ grad_scratch,
	T* __restrict__ adjoint,
	const T* __restrict__ increments,
	T* __restrict__ out,
	int dim,
	int step,
	int degree,
	uint64_t state_size,
	uint64_t scratch_size,
	uint64_t path_stride
) {
	__shared__ uint64_t powers[MAX_GENERIC_DEGREE + 1];
	__shared__ uint64_t level_starts[MAX_GENERIC_DEGREE + 1];
	extern __shared__ char smem[];
	T* increment_grad = reinterpret_cast<T*>(smem);
	if (threadIdx.x == 0) {
		powers[0] = 1;
		level_starts[0] = 0;
		for (int level = 1; level <= degree; ++level) {
			powers[level] = powers[level - 1] * static_cast<uint64_t>(dim);
			level_starts[level] = level == 1 ? 0
				: level_starts[level - 1] + powers[level - 1];
		}
	}
	__syncthreads();

	const uint64_t batch_idx = blockIdx.x;
	const T* batch_state = state + batch_idx * state_size;
	T* batch_scratch = scratch + batch_idx * scratch_size;
	T* batch_grad_scratch = grad_scratch + batch_idx * scratch_size;
	T* batch_adjoint = adjoint + batch_idx * state_size;
	const T* increment = increments + batch_idx * dim;
	T* batch_out = out + batch_idx * path_stride;
	for (int letter = threadIdx.x; letter < dim; letter += blockDim.x)
		increment_grad[letter] = batch_adjoint[level_starts[1] + letter];
	__syncthreads();

	for (int output_level = 2; output_level <= degree; ++output_level) {
		for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
			batch_scratch[level_starts[1] + letter] =
				batch_state[level_starts[1] + letter]
				+ increment[letter] * d_recip_rt<T>(output_level);
		}
		__syncthreads();
		for (int scratch_level = 2; scratch_level < output_level;
			++scratch_level) {
			const uint64_t level_size = powers[scratch_level];
			const T scale = d_recip_rt<T>(
				output_level - scratch_level + 1);
			for (uint64_t word = threadIdx.x; word < level_size;
				word += blockDim.x) {
				batch_scratch[level_starts[scratch_level] + word] =
					batch_state[level_starts[scratch_level] + word]
					+ batch_scratch[level_starts[scratch_level - 1]
						+ word / dim] * increment[word % dim] * scale;
			}
			__syncthreads();
		}

		int lower_level = output_level - 1;
		uint64_t lower_size = powers[lower_level];
		int lower_group_size = 1;
		while (lower_group_size < 32
			&& static_cast<uint64_t>(lower_group_size * 2) * lower_size
				<= blockDim.x) {
			lower_group_size *= 2;
		}
		int lower_lane = threadIdx.x & (lower_group_size - 1);
		int lower_group = threadIdx.x / lower_group_size;
		int lower_group_count = blockDim.x / lower_group_size;
		for (uint64_t first_word = 0; first_word < lower_size;
			first_word += lower_group_count) {
			const uint64_t lower_word = first_word + lower_group;
			const bool active = lower_word < lower_size;
			T value = T(0);
			if (active) {
				for (int letter = lower_lane; letter < dim;
					letter += lower_group_size) {
					value += batch_adjoint[level_starts[output_level]
						+ lower_word * dim + letter] * increment[letter];
				}
			}
			for (int offset = lower_group_size / 2; offset > 0; offset /= 2) {
				value += __shfl_down_sync(
					0xffffffff, value, offset, lower_group_size);
			}
			if (active && lower_lane == 0) {
				batch_grad_scratch[level_starts[lower_level] + lower_word]
					= value;
			}
		}

		int letter_group_size = 1;
		while (letter_group_size < 32
			&& letter_group_size * 2 * dim <= blockDim.x) {
			letter_group_size *= 2;
		}
		const int letter_lane = threadIdx.x & (letter_group_size - 1);
		const int letter_group = threadIdx.x / letter_group_size;
		const int letter_group_count = blockDim.x / letter_group_size;
		for (int first_letter = 0; first_letter < dim;
			first_letter += letter_group_count) {
			const int target_letter = first_letter + letter_group;
			const bool active = target_letter < dim;
			T value = T(0);
			if (active) {
				for (uint64_t lower_word = letter_lane;
					lower_word < lower_size; lower_word += letter_group_size) {
					value += batch_scratch[level_starts[lower_level] + lower_word]
						* batch_adjoint[level_starts[output_level]
							+ lower_word * dim + target_letter];
				}
			}
			for (int offset = letter_group_size / 2; offset > 0; offset /= 2) {
				value += __shfl_down_sync(
					0xffffffff, value, offset, letter_group_size);
			}
			if (active && letter_lane == 0)
				increment_grad[target_letter] += value;
		}
		__syncthreads();

		for (int scratch_level = output_level - 1; scratch_level >= 2;
			--scratch_level) {
			const uint64_t level_size = powers[scratch_level];
			for (uint64_t word = threadIdx.x; word < level_size;
				word += blockDim.x) {
				batch_adjoint[level_starts[scratch_level] + word] +=
					batch_grad_scratch[level_starts[scratch_level] + word];
			}
			const T scale = d_recip_rt<T>(
				output_level - scratch_level + 1);
			lower_level = scratch_level - 1;
			lower_size = powers[lower_level];
			lower_group_size = 1;
			while (lower_group_size < 32
				&& static_cast<uint64_t>(lower_group_size * 2) * lower_size
					<= blockDim.x) {
				lower_group_size *= 2;
			}
			lower_lane = threadIdx.x & (lower_group_size - 1);
			lower_group = threadIdx.x / lower_group_size;
			lower_group_count = blockDim.x / lower_group_size;
			for (uint64_t first_word = 0; first_word < lower_size;
				first_word += lower_group_count) {
				const uint64_t lower_word = first_word + lower_group;
				const bool active = lower_word < lower_size;
				T value = T(0);
				if (active) {
					for (int letter = lower_lane; letter < dim;
						letter += lower_group_size) {
						value += batch_grad_scratch[
							level_starts[scratch_level]
							+ lower_word * dim + letter] * increment[letter]
							* scale;
					}
				}
				for (int offset = lower_group_size / 2; offset > 0;
					offset /= 2) {
					value += __shfl_down_sync(
						0xffffffff, value, offset, lower_group_size);
				}
				if (active && lower_lane == 0) {
					batch_grad_scratch[level_starts[lower_level] + lower_word]
						= value;
				}
			}

			for (int first_letter = 0; first_letter < dim;
				first_letter += letter_group_count) {
				const int target_letter = first_letter + letter_group;
				const bool active = target_letter < dim;
				T value = T(0);
				if (active) {
					for (uint64_t lower_word = letter_lane;
						lower_word < lower_size;
						lower_word += letter_group_size) {
						value += batch_scratch[
							level_starts[lower_level] + lower_word]
							* batch_grad_scratch[level_starts[scratch_level]
								+ lower_word * dim + target_letter] * scale;
					}
				}
				for (int offset = letter_group_size / 2; offset > 0;
					offset /= 2) {
					value += __shfl_down_sync(
						0xffffffff, value, offset, letter_group_size);
				}
				if (active && letter_lane == 0)
					increment_grad[target_letter] += value;
			}
			__syncthreads();
		}

		for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
			const T value = batch_grad_scratch[level_starts[1] + letter];
			batch_adjoint[level_starts[1] + letter] += value;
			increment_grad[letter] += value * d_recip_rt<T>(output_level);
		}
		__syncthreads();
	}

	for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
		batch_out[static_cast<uint64_t>(step) * dim + letter]
			= increment_grad[letter];
	}
}

// Convert increment gradients to path gradients
template<typename T>
__global__ void increment_to_path_grad_ker(
	const T* __restrict__ inc_grad,
	const T* __restrict__ level_one_grad,
	T* __restrict__ path_grad,        // [batch, length, dim]
	int batch_size,
	int length,
	int dim,
	uint64_t inc_time_stride,
	uint64_t inc_batch_stride,
	uint64_t context_time_stride,
	uint64_t context_batch_stride
) {
	const int total = batch_size * length * dim;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	const int time_stride = length * dim;
	const int b = idx / time_stride;
	const int rem = idx - b * time_stride;
	const int t = rem / dim;
	const int j = rem - t * dim;

	const int steps = length - 1;
	const uint64_t base = static_cast<uint64_t>(b) * inc_batch_stride + j;
	const T* batch_level_one = level_one_grad == nullptr ? nullptr
		: level_one_grad + static_cast<uint64_t>(b) * context_batch_stride + j;

	if (t == 0) {
		path_grad[idx] = -inc_grad[base]
			- (batch_level_one == nullptr ? T(0) : batch_level_one[0]);
	} else if (t == length - 1) {
		path_grad[idx] = inc_grad[base + (steps - 1) * inc_time_stride]
			+ (batch_level_one == nullptr ? T(0)
				: batch_level_one[(steps - 1) * context_time_stride]);
	} else {
		path_grad[idx] = inc_grad[base + (t - 1) * inc_time_stride]
			- inc_grad[base + t * inc_time_stride]
			+ (batch_level_one == nullptr ? T(0)
				: batch_level_one[(t - 1) * context_time_stride]
					- batch_level_one[t * context_time_stride]);
	}
}

template<typename T>
__global__ void increment_to_path_grad_inplace_ker(
	T* path_grad,
	int length,
	int dim
) {
	const uint64_t batch_idx = blockIdx.x;
	const int steps = length - 1;
	T* batch_grad = path_grad
		+ batch_idx * static_cast<uint64_t>(length) * dim;
	for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
		batch_grad[static_cast<uint64_t>(steps) * dim + letter]
			= batch_grad[static_cast<uint64_t>(steps - 1) * dim + letter];
		for (int step = steps - 1; step > 0; --step) {
			batch_grad[static_cast<uint64_t>(step) * dim + letter]
				= batch_grad[static_cast<uint64_t>(step - 1) * dim + letter]
				- batch_grad[static_cast<uint64_t>(step) * dim + letter];
		}
		batch_grad[letter] = -batch_grad[letter];
	}
}

template<typename T>
__global__ void increment_to_path_grad_batch_ker(
	const T* __restrict__ increment_grad,
	T* __restrict__ path_grad,
	int length,
	int dim,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int path_size = length * dim;
	if (idx >= path_size)
		return;
	const int step = idx / dim;
	const int letter = idx - step * dim;
	const int increment_steps = length - 1;
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const T* batch_increment_grad = increment_grad
		+ local_batch_idx * increment_steps * dim;
	T* batch_path_grad = path_grad
		+ batch_idx * path_size;
	if (step == 0) {
		batch_path_grad[idx] = -batch_increment_grad[letter];
	} else if (step == increment_steps) {
		batch_path_grad[idx] = batch_increment_grad[
			static_cast<uint64_t>(increment_steps - 1) * dim + letter];
	} else {
		batch_path_grad[idx] = batch_increment_grad[
			static_cast<uint64_t>(step - 1) * dim + letter]
			- batch_increment_grad[static_cast<uint64_t>(step) * dim + letter];
	}
}

template<typename T, int LEVEL, int BLOCK_SIZE>
__global__ void sig_context_scan_persistent_ker(
	const T* __restrict__ path,
	T* __restrict__ context,
	int dim,
	int steps,
	uint64_t padded_steps,
	uint64_t context_size,
	uint64_t level_offset,
	uint64_t path_stride,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	using BlockScan = cub::BlockScan<T, BLOCK_SIZE>;
	__shared__ typename BlockScan::TempStorage scan_storage;
	__shared__ T carry;

	const uint32_t word_idx = blockIdx.x;
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	uint32_t powers[LEVEL + 1];
	uint32_t level_starts[LEVEL + 1];
	powers[0] = 1;
	level_starts[0] = 0;
	if constexpr (LEVEL >= 1)
		level_starts[1] = 0;
	for (int level = 1; level <= LEVEL; ++level) {
		powers[level] = powers[level - 1] * static_cast<uint64_t>(dim);
		if (level > 1)
			level_starts[level] = level_starts[level - 1]
				+ powers[level - 1];
	}

	int letters[LEVEL];
	uint32_t word = word_idx;
	for (int pos = LEVEL - 1; pos >= 0; --pos) {
		letters[pos] = static_cast<int>(word % dim);
		word /= dim;
	}
	if (threadIdx.x == 0)
		carry = T(0);
	__syncthreads();

	const T* batch_path = path + batch_idx * path_stride;
	for (int first_step = 0; first_step < steps; first_step += BLOCK_SIZE) {
		const int step = first_step + static_cast<int>(threadIdx.x);
		const bool active = step < steps;
		T contribution = T(0);
		if (active) {
			const T* previous = batch_path + static_cast<uint64_t>(step) * dim;
			const T* next = previous + dim;
			const T* lower_context = context
				+ (batch_idx * padded_steps + step) * context_size;
			T exp_value = T(1);
			for (int exp_level = 1; exp_level <= LEVEL; ++exp_level) {
				const int lower_level = LEVEL - exp_level;
				const int letter_pos = LEVEL - exp_level;
				exp_value *= (next[letters[letter_pos]]
					- previous[letters[letter_pos]]) * d_recip<T>(exp_level);
				T lower_value = T(1);
				if (lower_level != 0) {
					const uint32_t lower_word = word_idx / powers[exp_level];
					lower_value = lower_level == 1
						? batch_path[static_cast<uint64_t>(step) * dim + lower_word]
							- batch_path[lower_word]
						: lower_context[level_starts[lower_level] + lower_word];
				}
				contribution += exp_value * lower_value;
			}
		}

		const T block_prefix = carry;
		T local_prefix;
		T block_total;
		BlockScan(scan_storage).ExclusiveSum(
			contribution, local_prefix, block_total);
		if (active) {
			context[(batch_idx * padded_steps + step) * context_size
				+ level_offset + word_idx] = block_prefix + local_prefix;
		}
		if (threadIdx.x == 0)
			carry += block_total;
		__syncthreads();
	}
}

template<typename T, int BLOCK_SIZE>
__global__ void sig_level_adjoint_scan_ker(
	T* __restrict__ adjoint_context,
	const T* __restrict__ sig_grads,
	int steps,
	uint64_t padded_steps,
	uint64_t context_size,
	uint64_t level_offset,
	uint64_t sig_stride,
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	using BlockScan = cub::BlockScan<T, BLOCK_SIZE>;
	__shared__ typename BlockScan::TempStorage scan_storage;
	__shared__ T carry;

	const uint64_t word_idx = blockIdx.x;
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	if (threadIdx.x == 0) {
		carry = sig_grads[batch_idx * sig_stride + (scalar_term ? 1 : 0)
			+ level_offset + word_idx];
	}
	__syncthreads();

	for (int last_step = steps; last_step > 0; last_step -= BLOCK_SIZE) {
		const int step = last_step - 1 - static_cast<int>(threadIdx.x);
		const bool active = step >= 0;
		const uint64_t context_idx = (batch_idx * padded_steps
			+ static_cast<uint64_t>(active ? step : 0)) * context_size
			+ level_offset + word_idx;
		const T contribution = active ? adjoint_context[context_idx] : T(0);
		const T block_prefix = carry;
		T local_prefix;
		T block_total;
		BlockScan(scan_storage).ExclusiveSum(
			contribution, local_prefix, block_total);
		if (active)
			adjoint_context[context_idx] = block_prefix + local_prefix;
		if (threadIdx.x == 0)
			carry += block_total;
		__syncthreads();
	}
}

template<typename T>
__global__ void sig_level_horner_backprop_ker(
	const T* __restrict__ path,
	const T* __restrict__ sig_grads,
	const T* __restrict__ prefix_context,
	T* __restrict__ adjoint_context,
	T* __restrict__ increment_grads,
	int dim,
	int steps,
	int current_level,
	int group_width,
	uint32_t current_size,
	uint32_t current_offset,
	uint64_t batch_size,
	uint64_t sig_stride,
	uint64_t context_size,
	uint64_t padded_steps,
	uint64_t path_stride,
	bool scalar_term,
	bool top_level
) {
	extern __shared__ char dynamic_shared[];
	T* shared_values = reinterpret_cast<T*>(dynamic_shared);
	const int groups_per_block = blockDim.x / group_width;
	T* shared_derivs = shared_values
		+ static_cast<uint64_t>(groups_per_block) * current_offset;
	const int group = threadIdx.x / group_width;
	const int lane = threadIdx.x - group * group_width;
	const uint64_t time_idx = static_cast<uint64_t>(blockIdx.x)
		* groups_per_block + group;
	const bool active = time_idx < batch_size * static_cast<uint64_t>(steps);
	const int step = active ? static_cast<int>(time_idx % steps) : 0;
	const uint64_t batch_idx = active ? time_idx / steps : 0;
	const bool active_letter = active && lane < dim;
	T* values = shared_values + static_cast<uint64_t>(group) * current_offset;
	T* derivs = shared_derivs + static_cast<uint64_t>(group) * current_offset;
	const T* batch_path = path + batch_idx * path_stride;
	const T* previous = batch_path + static_cast<uint64_t>(step) * dim;
	const T* next = previous + dim;
	const T* prefix = prefix_context
		+ (batch_idx * padded_steps + step) * context_size;
	T* lower_adjoint = adjoint_context
		+ (batch_idx * padded_steps + step) * context_size;
	T* batch_increment_grads = increment_grads + time_idx * dim;

	if (current_level == 1) {
		if (active_letter)
			batch_increment_grads[lane] += lower_adjoint[lane];
		return;
	}

	if (active_letter) {
		values[lane] = batch_path[static_cast<uint64_t>(step) * dim + lane]
			- batch_path[lane] + (next[lane] - previous[lane])
			* d_recip_rt<T>(current_level);
	}
	__syncthreads();

	uint32_t previous_size = static_cast<uint32_t>(dim);
	uint32_t previous_offset = 0;
	uint32_t rank_size = previous_size * static_cast<uint32_t>(dim);
	uint32_t rank_offset = previous_size;
	for (int rank = 2; rank < current_level; ++rank) {
		if (active) {
			const T inverse = d_recip_rt<T>(current_level - rank + 1);
			for (uint32_t word = lane; word < rank_size; word += group_width) {
				const uint32_t prefix_word = word / static_cast<uint32_t>(dim);
				const int last_letter = static_cast<int>(word % dim);
				values[rank_offset + word] = prefix[rank_offset + word]
					+ values[previous_offset + prefix_word]
						* (next[last_letter] - previous[last_letter]) * inverse;
			}
		}
		__syncthreads();
		previous_offset = rank_offset;
		previous_size = rank_size;
		rank_offset += rank_size;
		rank_size *= static_cast<uint32_t>(dim);
	}

	const T* current_adjoint = top_level
		? sig_grads + batch_idx * sig_stride + (scalar_term ? 1 : 0)
			+ current_offset
		: lower_adjoint + current_offset;
	if (active) {
		for (uint32_t word = lane; word < previous_size; word += group_width) {
			T value = T(0);
			for (int other = 0; other < dim; ++other) {
				value += current_adjoint[word * dim + other]
					* (next[other] - previous[other]);
			}
			derivs[previous_offset + word] = value;
		}
	}
	__syncthreads();

	T increment_deriv = T(0);
	if (active_letter) {
		for (uint32_t word = 0; word < previous_size; ++word) {
			increment_deriv += current_adjoint[word * dim + lane]
				* values[previous_offset + word];
		}
	}

	for (int rank = current_level - 1; rank >= 2; --rank) {
		const uint32_t lower_size = previous_size / static_cast<uint32_t>(dim);
		const uint32_t lower_offset = previous_offset - lower_size;
		const T inverse = d_recip_rt<T>(current_level - rank + 1);
		if (active) {
			for (uint32_t word = lane; word < previous_size; word += group_width)
				lower_adjoint[previous_offset + word] += derivs[previous_offset + word];
			for (uint32_t word = lane; word < lower_size; word += group_width) {
				T value = T(0);
				for (int other = 0; other < dim; ++other) {
					value += derivs[previous_offset + word * dim + other]
						* (next[other] - previous[other]) * inverse;
				}
				derivs[lower_offset + word] = value;
			}
		}
		if (active_letter) {
			for (uint32_t word = 0; word < lower_size; ++word) {
				increment_deriv += derivs[previous_offset + word * dim + lane]
					* values[lower_offset + word] * inverse;
			}
		}
		__syncthreads();
		previous_size = lower_size;
		previous_offset = lower_offset;
	}

	if (active_letter) {
		lower_adjoint[lane] += derivs[lane];
		increment_deriv += derivs[lane] * d_recip_rt<T>(current_level);
		batch_increment_grads[lane] += increment_deriv;
	}
}

template<typename T, int LEVEL>
__global__ void sig_suffix_from_prefix_ker(
	const T* __restrict__ path,
	const T* __restrict__ sig,
	const T* __restrict__ prefix_context,
	T* __restrict__ suffix_context,
	int dim,
	int steps,
	uint64_t batch_size,
	uint64_t sig_stride,
	uint64_t padded_steps,
	uint64_t context_size,
	uint64_t level_offset,
	uint64_t level_size,
	uint64_t path_stride,
	bool scalar_term
) {
	const uint32_t idx = blockIdx.x * blockDim.x
		+ threadIdx.x;
	const uint32_t total = static_cast<uint32_t>(batch_size) * steps
		* static_cast<uint32_t>(level_size);
	if (idx >= total)
		return;
	const uint32_t word_idx = idx % static_cast<uint32_t>(level_size);
	const uint32_t time_idx = idx / static_cast<uint32_t>(level_size);
	const int step = static_cast<int>(time_idx % steps);
	const uint64_t batch_idx = time_idx / steps;

	uint32_t powers[LEVEL + 1];
	uint32_t level_starts[LEVEL + 1];
	powers[0] = 1;
	level_starts[0] = 0;
	if constexpr (LEVEL >= 1)
		level_starts[1] = 0;
	for (int level = 1; level <= LEVEL; ++level) {
		powers[level] = powers[level - 1] * static_cast<uint64_t>(dim);
		if (level > 1)
			level_starts[level] = level_starts[level - 1]
				+ powers[level - 1];
	}
	int letters[LEVEL];
	uint32_t word = word_idx;
	for (int pos = LEVEL - 1; pos >= 0; --pos) {
		letters[pos] = static_cast<int>(word % dim);
		word /= dim;
	}

	const T* batch_path = path + batch_idx * path_stride;
	const T* previous = batch_path + static_cast<uint64_t>(step) * dim;
	const T* next = previous + dim;
	const T* prefix = prefix_context
		+ (batch_idx * padded_steps + step) * context_size;
	const T* suffix = suffix_context
		+ (batch_idx * padded_steps + step) * context_size;
	const uint64_t scalar_offset = scalar_term ? 1 : 0;
	T value = sig[batch_idx * sig_stride + scalar_offset
		+ level_starts[LEVEL] + word_idx];
	for (int left_level = 1; left_level <= LEVEL; ++left_level) {
		T through_step = T(0);
		for (int prefix_level = 0; prefix_level <= left_level;
			++prefix_level) {
			T prefix_value = T(1);
			if (prefix_level != 0) {
				const uint32_t prefix_word = word_idx
					/ powers[LEVEL - prefix_level];
				prefix_value = prefix_level == 1
					? batch_path[static_cast<uint64_t>(step) * dim + prefix_word]
						- batch_path[prefix_word]
					: prefix[level_starts[prefix_level] + prefix_word];
			}
			T exp_value = T(1);
			for (int pos = prefix_level; pos < left_level; ++pos) {
				exp_value *= (next[letters[pos]] - previous[letters[pos]])
					* d_recip<T>(pos - prefix_level + 1);
			}
			through_step += prefix_value * exp_value;
		}
		const int right_level = LEVEL - left_level;
		if (right_level != 0) {
			const uint32_t suffix_word = word_idx % powers[right_level];
			const T suffix_value = right_level == 1
				? batch_path[static_cast<uint64_t>(steps) * dim + suffix_word]
					- batch_path[static_cast<uint64_t>(step + 1) * dim + suffix_word]
				: suffix[level_starts[right_level] + suffix_word];
			through_step *= suffix_value;
		}
		value -= through_step;
	}
	suffix_context[(batch_idx * padded_steps + step) * context_size
		+ level_offset + word_idx] = value;
}

template<typename T, int DEGREE>
__global__ void sig_grouped_backprop_ker(
	const T* __restrict__ path,
	const T* __restrict__ sig_grads,
	const T* __restrict__ prefix_context,
	const T* __restrict__ suffix_context,
	T* __restrict__ increment_grads,
	int dim,
	int steps,
	uint64_t batch_size,
	uint64_t sig_stride,
	uint64_t context_size,
	uint64_t padded_steps,
	uint64_t path_stride,
	bool scalar_term
) {
	__shared__ uint32_t powers[DEGREE + 1];
	__shared__ uint32_t level_starts[DEGREE + 1];
	if (threadIdx.x == 0) {
		powers[0] = 1;
		level_starts[0] = 0;
		if constexpr (DEGREE >= 1)
			level_starts[1] = 0;
		for (int level = 1; level <= DEGREE; ++level) {
			powers[level] = powers[level - 1] * static_cast<uint32_t>(dim);
			if (level > 1)
				level_starts[level] = level_starts[level - 1]
					+ powers[level - 1];
		}
	}
	__syncthreads();

	extern __shared__ char dynamic_shared[];
	T* shared_derivs = reinterpret_cast<T*>(dynamic_shared);
	const int groups_per_block = blockDim.x / dim;
	T* shared_exp_values = shared_derivs
		+ static_cast<uint64_t>(groups_per_block) * context_size;
	T* shared_suffix_one = shared_exp_values
		+ static_cast<uint64_t>(groups_per_block) * context_size;
	const int group = threadIdx.x / dim;
	const int letter = threadIdx.x - group * dim;
	const bool participating = group < groups_per_block;
	const uint64_t time_idx = static_cast<uint64_t>(blockIdx.x)
		* groups_per_block + group;
	const bool active = participating
		&& time_idx < batch_size * static_cast<uint64_t>(steps);
	const int step = active ? static_cast<int>(time_idx % steps) : 0;
	const uint64_t batch_idx = active ? time_idx / steps : 0;
	const T* batch_path = path + batch_idx * path_stride;
	const T* previous = batch_path + static_cast<uint64_t>(step) * dim;
	const T* next = previous + dim;
	const T* batch_sig_grads = sig_grads + batch_idx * sig_stride;
	const T* prefix = prefix_context
		+ (batch_idx * padded_steps + step) * context_size;
	const T* suffix = suffix_context
		+ (batch_idx * padded_steps + step) * context_size;
	T* exp_derivs = shared_derivs + static_cast<uint64_t>(group) * context_size;
	T* exp_values = shared_exp_values
		+ static_cast<uint64_t>(group) * context_size;
	T* suffix_one = shared_suffix_one + static_cast<uint64_t>(group) * dim;
	const uint64_t scalar_offset = scalar_term ? 1 : 0;
	if (active) {
		exp_derivs[letter] = next[letter] - previous[letter];
		exp_values[letter] = batch_path[static_cast<uint64_t>(step) * dim + letter]
			- batch_path[letter];
		suffix_one[letter] = batch_path[static_cast<uint64_t>(steps) * dim + letter]
			- batch_path[static_cast<uint64_t>(step + 1) * dim + letter];
	}
	__syncthreads();

	for (int exp_level = DEGREE - 1; exp_level >= 2; --exp_level) {
		if (active) {
			const uint32_t exp_size = powers[exp_level];
			for (uint32_t exp_word = letter; exp_word < exp_size;
				exp_word += dim) {
				T exp_value = T(1);
				uint32_t word = exp_word;
				for (int pos = exp_level - 1; pos >= 0; --pos) {
					exp_value *= exp_derivs[word % dim];
					word /= dim;
				}
				for (int factor = 2; factor <= exp_level; ++factor)
					exp_value *= d_recip<T>(factor);
				exp_values[level_starts[exp_level] + exp_word] = exp_value;
				T exp_deriv = batch_sig_grads[
					scalar_offset + level_starts[exp_level] + exp_word];
				for (int left_level = 0;
					left_level <= DEGREE - exp_level; ++left_level) {
					for (int right_level = 0;
						right_level <= DEGREE - exp_level - left_level;
						++right_level) {
						if (left_level == 0 && right_level == 0)
							continue;
						const int output_level = left_level + exp_level
							+ right_level;
						for (uint32_t left_word = 0;
							left_word < powers[left_level]; ++left_word) {
							const T prefix_value = left_level == 0 ? T(1)
								: left_level == 1
									? exp_values[left_word]
									: prefix[level_starts[left_level] + left_word];
							for (uint32_t right_word = 0;
								right_word < powers[right_level]; ++right_word) {
								const T suffix_value = right_level == 0 ? T(1)
									: right_level == 1
										? suffix_one[right_word]
										: suffix[level_starts[right_level] + right_word];
								const uint32_t output_word =
									(left_word * exp_size + exp_word)
									* powers[right_level] + right_word;
								exp_deriv += prefix_value * batch_sig_grads[
									scalar_offset + level_starts[output_level]
									+ output_word] * suffix_value;
							}
						}
					}
				}
				const int upper_level = exp_level + 1;
				for (int upper_letter = 0; upper_letter < dim; ++upper_letter) {
					const uint32_t upper_word = exp_word * dim + upper_letter;
					const T upper_deriv = upper_level == DEGREE
						? batch_sig_grads[scalar_offset + level_starts[upper_level]
							+ upper_word]
						: exp_derivs[level_starts[upper_level] + upper_word];
					exp_deriv += upper_deriv * exp_derivs[upper_letter]
						* d_recip<T>(upper_level);
				}
				exp_derivs[level_starts[exp_level] + exp_word] = exp_deriv;
			}
		}
		__syncthreads();
	}

	T deriv = T(0);
	if (active) {
		deriv = batch_sig_grads[scalar_offset + letter];
		for (int left_level = 0; left_level < DEGREE; ++left_level) {
			for (int right_level = 0; right_level < DEGREE - left_level;
				++right_level) {
				if (left_level == 0 && right_level == 0)
					continue;
				const int output_level = left_level + right_level + 1;
				for (uint32_t left_word = 0; left_word < powers[left_level];
					++left_word) {
					const T prefix_value = left_level == 0 ? T(1)
						: left_level == 1
							? exp_values[left_word]
							: prefix[level_starts[left_level] + left_word];
					for (uint32_t right_word = 0;
						right_word < powers[right_level]; ++right_word) {
						const T suffix_value = right_level == 0 ? T(1)
							: right_level == 1
								? suffix_one[right_word]
								: suffix[level_starts[right_level] + right_word];
						const uint32_t output_word =
							(left_word * dim + letter) * powers[right_level]
							+ right_word;
						deriv += prefix_value * batch_sig_grads[
							scalar_offset + level_starts[output_level] + output_word]
							* suffix_value;
					}
				}
			}
		}

		const T* level_two_derivs = DEGREE == 2
			? batch_sig_grads + scalar_offset + level_starts[2]
			: exp_derivs + level_starts[2];
		for (int other = 0; other < dim; ++other) {
			deriv += (level_two_derivs[static_cast<uint32_t>(letter) * dim + other]
				+ level_two_derivs[static_cast<uint32_t>(other) * dim + letter])
				* exp_derivs[other] * T(0.5);
		}
	}

	if (active) {
		for (int level = 3; level <= DEGREE; ++level) {
			for (uint32_t previous_word = 0;
				previous_word < powers[level - 1]; ++previous_word) {
				const uint32_t word_idx = previous_word * dim + letter;
				const T upper_deriv = level == DEGREE
					? batch_sig_grads[
						scalar_offset + level_starts[level] + word_idx]
					: exp_derivs[level_starts[level] + word_idx];
				deriv += upper_deriv
					* exp_values[level_starts[level - 1] + previous_word]
					* d_recip<T>(level);
			}
		}
		increment_grads[time_idx * dim + letter] = deriv;
	}
}

template<typename T>
__global__ void set_sig_level0(T* out, uint64_t sig_size, uint64_t batch_size) {
	uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (b < batch_size) out[b * sig_size] = static_cast<T>(1);
}

static constexpr int MAX_PER_WORD_STREAMS = 12;
static cudaStream_t s_per_word_streams[MAX_PER_WORD_STREAMS] = {};
static bool s_streams_initialized = false;
static std::mutex s_streams_mu;

static void ensure_streams() {
	std::lock_guard<std::mutex> lock(s_streams_mu);
	if (!s_streams_initialized) {
		for (int i = 0; i < MAX_PER_WORD_STREAMS; ++i)
			cudaStreamCreate(&s_per_word_streams[i]);
		s_streams_initialized = true;
	}
}

static void* s_inc_grad_buf = nullptr;
static size_t s_inc_grad_buf_size = 0;
static std::mutex s_inc_grad_buf_mu;
static const void* s_sig_context_path = nullptr;
static const void* s_sig_context_sig = nullptr;
static uint64_t s_sig_context_batch_size = 0;
static uint64_t s_sig_context_dimension = 0;
static uint64_t s_sig_context_length = 0;
static uint64_t s_sig_context_degree = 0;
static size_t s_sig_context_value_size = 0;
static bool s_sig_context_scalar_term = false;
static int s_sig_context_kind = 0;
static std::atomic<bool> s_sig_context_valid = false;

static void* ensure_inc_grad_buf(size_t needed) {
	if (needed > s_inc_grad_buf_size) {
		if (s_inc_grad_buf) { cudaFree(s_inc_grad_buf); s_inc_grad_buf = nullptr; s_inc_grad_buf_size = 0; }
		CUDA_CHECK(cudaMalloc(&s_inc_grad_buf, needed));
		s_inc_grad_buf_size = needed;
		s_sig_context_valid = false;
	}
	return s_inc_grad_buf;
}

void release_signature_state() {
	{
		std::lock_guard<std::mutex> lock(s_streams_mu);
		if (s_streams_initialized) {
			for (int i = 0; i < MAX_PER_WORD_STREAMS; ++i) {
				if (s_per_word_streams[i]) {
					cudaStreamDestroy(s_per_word_streams[i]);
					s_per_word_streams[i] = nullptr;
				}
			}
			s_streams_initialized = false;
		}
	}
	{
		std::lock_guard<std::mutex> lock(s_inc_grad_buf_mu);
		if (s_inc_grad_buf) {
			cudaFree(s_inc_grad_buf);
			s_inc_grad_buf = nullptr;
			s_inc_grad_buf_size = 0;
		}
		s_sig_context_valid = false;
	}
}

template<typename T>
void signature_per_word_core_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool scalar_term = true
) {
	const uint64_t full_sig_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
	const uint64_t path_stride = length * dimension;
	const int steps = static_cast<int>(length - 1);
	const int dim = static_cast<int>(dimension);

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);

	const int requested_chunk = (steps < SIG_CHUNK) ? steps : SIG_CHUNK;
	const size_t bytes_per_step = checked_cuda_size_mul(
		static_cast<size_t>(dimension), sizeof(T), "CUDA signature");
	const size_t requested_smem = checked_cuda_size_mul(
		static_cast<size_t>(requested_chunk), bytes_per_step, "CUDA signature");
	CudaSharedMemoryLimits smem_limits = {
		CUDA_BASE_DYNAMIC_SMEM, CUDA_BASE_DYNAMIC_SMEM
	};
	if (requested_smem > CUDA_BASE_DYNAMIC_SMEM)
		smem_limits = cuda_shared_memory_limits();
	const size_t max_chunk = smem_limits.optin_bytes / bytes_per_step;
	const bool shared_cache_candidate = max_chunk != 0;
	const int chunk_size = shared_cache_candidate
		? static_cast<int>(std::min<size_t>(
			static_cast<size_t>(requested_chunk), max_chunk))
		: 1;
	const size_t smem = shared_cache_candidate
		? checked_cuda_size_mul(
			static_cast<size_t>(chunk_size), bytes_per_step, "CUDA signature")
		: 0;

	// For small total output sizes, skip streams and launch sequentially
	// on the default stream - avoids event create/destroy and stream sync overhead.
	const uint64_t top_level = host_power(dimension, degree);
	const bool use_streams = (top_level > 4096);

	if (use_streams)
		ensure_streams();

	for (uint64_t k = 1; k <= degree; ++k) {
		uint64_t level_size = host_power(dimension, k);
		// In scalar_term=false layout, level offsets shift down by 1.
		uint64_t level_offset = scalar_term ? li[k] : (li[k] - 1);
		unsigned int block = 128;
		if (level_size < 128) block = 32;
		unsigned int grid_x = (unsigned int)((level_size + block - 1) / block);
		cudaStream_t stream = (use_streams && k <= MAX_PER_WORD_STREAMS)
			? s_per_word_streams[k - 1] : nullptr;

		#define LAUNCH_DEGREE(D) \
			case D: { \
				const bool use_shared_cache = shared_cache_candidate \
					&& (smem <= CUDA_BASE_DYNAMIC_SMEM \
						|| try_configure_dynamic_smem( \
							signature_per_word_ker<T, D, true>, smem, smem_limits)); \
				if (use_shared_cache) \
					signature_per_word_ker<T, D, true><<<batch_chunk.grid, block, smem, stream>>>( \
						path, out, dim, steps, chunk_size, sig_stride, level_offset, \
						level_size, path_stride, scalar_term, \
						batch_chunk.offset, batch_chunk.size); \
				else \
					signature_per_word_ker<T, D, false><<<batch_chunk.grid, block, 0, stream>>>( \
						path, out, dim, steps, 1, sig_stride, level_offset, \
						level_size, path_stride, scalar_term, \
						batch_chunk.offset, batch_chunk.size); \
				break; \
			}

		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				grid_x, batch_size, batch_offset);
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
				LAUNCH_DEGREE(13)
				default:
					if (shared_cache_candidate
						&& (smem <= CUDA_BASE_DYNAMIC_SMEM
							|| try_configure_dynamic_smem(
								signature_per_word_generic_ker<T, true>, smem, smem_limits))) {
						signature_per_word_generic_ker<T, true><<<
							batch_chunk.grid, block, smem, stream>>>(
								path, out, dim, steps, chunk_size, static_cast<int>(k),
								sig_stride, level_offset, level_size, path_stride, scalar_term,
								batch_chunk.offset, batch_chunk.size);
					}
					else {
						signature_per_word_generic_ker<T, false><<<
							batch_chunk.grid, block, 0, stream>>>(
								path, out, dim, steps, 1, static_cast<int>(k),
								sig_stride, level_offset, level_size, path_stride, scalar_term,
								batch_chunk.offset, batch_chunk.size);
					}
					break;
			}
			batch_offset += batch_chunk.size;
		}
		#undef LAUNCH_DEGREE
	}

	if (use_streams) {
		cudaEvent_t done;
		cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
		const uint64_t n_custom = std::min<uint64_t>(degree, MAX_PER_WORD_STREAMS);
		for (uint64_t k = 1; k < n_custom; ++k) {
			cudaEventRecord(done, s_per_word_streams[k]);
			cudaStreamWaitEvent(s_per_word_streams[0], done, 0);
		}
		if (degree > MAX_PER_WORD_STREAMS) {
			cudaEventRecord(done, 0);
			cudaStreamWaitEvent(s_per_word_streams[0], done, 0);
		}
		cudaEventDestroy(done);
		cudaStreamSynchronize(s_per_word_streams[0]);
	} else {
		cudaDeviceSynchronize();
	}

	check_cuda_error();
}

inline void validate_signature_correction_args_cuda_(
	const void* correction,
	uint64_t correction_len,
	uint64_t dimension,
	uint64_t degree,
	bool lead_lag
) {
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");
	if (correction_len == 0)
		return;
	if (degree < 2)
		throw std::invalid_argument("correction must be empty when degree < 2");

	uint64_t offset = 0;
	uint64_t level_size = dimension;
	for (uint64_t level = 2; level <= degree; ++level) {
		level_size *= dimension;
		offset += level_size;
		if (offset == correction_len)
			return;
		if (offset > correction_len)
			break;
	}
	throw std::invalid_argument("correction length must be a prefix of tensor levels 2..degree");
}

template<typename T>
__device__ void build_correction_block_(
	const T* path,
	uint64_t step,
	T* local_log,
	const T* correction_segment,
	uint64_t correction_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;
	const uint64_t sig_len = level_index[degree + 1];

	for (uint64_t i = tid; i < sig_len; i += nthreads)
		local_log[i] = T(0);
	__syncthreads();

	const T* prev = path + step * dimension;
	const T* next = prev + dimension;
	for (uint64_t d = tid; d < dimension; d += nthreads)
		local_log[level_index[1] + d] = next[d] - prev[d];

	if (correction_segment == nullptr || correction_len == 0) {
		__syncthreads();
		return;
	}

	uint64_t offset = 0;
	uint64_t level_size = data_dimension;
	for (uint64_t level = 2; level <= degree; ++level) {
		level_size *= data_dimension;
		if (offset + level_size > correction_len)
			break;

		for (uint64_t word_idx = tid; word_idx < level_size; word_idx += nthreads) {
			const T value = correction_segment[offset + word_idx];
			if (value == T(0))
				continue;

			uint64_t tmp = word_idx;
			uint64_t aug_word_idx = 0;
			uint64_t pow = level_size / data_dimension;
			for (uint64_t pos = 0; pos < level; ++pos) {
				const uint64_t label = tmp / pow;
				tmp -= label * pow;
				if (pos + 1 < level)
					pow /= data_dimension;
				aug_word_idx = aug_word_idx * dimension + label;
			}
			local_log[level_index[level] + aug_word_idx] = value;
		}
		offset += level_size;
	}
	__syncthreads();
}

template<typename T>
__device__ void tensor_exp_block_(
	const T* local_log,
	T* out,
	T* power_prev,
	T* power_curr,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;
	const uint64_t sig_len = level_index[degree + 1];

	for (uint64_t i = tid; i < sig_len; i += nthreads) {
		out[i] = T(0);
		power_prev[i] = local_log[i];
	}
	if (tid == 0)
		out[0] = T(1);
	__syncthreads();

	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		out[i] += local_log[i];
	__syncthreads();

	for (uint64_t n = 2; n <= degree; ++n) {
		const T inv_n = T(1) / static_cast<T>(n);
		for (uint64_t target_level = n; target_level <= degree; ++target_level) {
			const uint64_t target_size = level_index[target_level + 1] - level_index[target_level];
			const uint64_t max_left = target_level - (n - 1);
			for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
				T sum = T(0);
				for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
					const uint64_t right_level = target_level - left_level;
					const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
					const uint64_t left_idx = idx / right_size;
					const uint64_t right_idx = idx - left_idx * right_size;
					sum += local_log[level_index[left_level] + left_idx]
						* power_prev[level_index[right_level] + right_idx] * inv_n;
				}
				const uint64_t out_idx = level_index[target_level] + idx;
				power_curr[out_idx] = sum;
				out[out_idx] += sum;
			}
		}
		__syncthreads();
		T* tmp = power_prev;
		power_prev = power_curr;
		power_curr = tmp;
		__syncthreads();
	}
}

template<typename T>
__device__ void sig_combine_block_(
	T* sig1,
	const T* sig2,
	uint64_t degree,
	const uint64_t* level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;
	if (tid == 0)
		sig1[0] = T(1);
	__syncthreads();

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {
		const uint64_t level = static_cast<uint64_t>(target_level);
		const uint64_t level_start = level_index[level];
		const uint64_t level_size = level_index[level + 1] - level_start;

		for (uint64_t idx = tid; idx < level_size; idx += nthreads) {
			T value = sig1[level_start + idx] + sig2[level_start + idx];
			for (uint64_t left_level = level - 1, right_level = 1;
				left_level > 0; --left_level, ++right_level) {
				const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
				const uint64_t left_idx = idx / right_size;
				const uint64_t right_idx = idx - left_idx * right_size;
				value += sig1[level_index[left_level] + left_idx]
					* sig2[level_index[right_level] + right_idx];
			}
			sig1[level_start + idx] = value;
		}
		__syncthreads();
	}
}

template<typename T>
__device__ void uncombine_sig_deriv_block_(
	const T* sig1,
	const T* sig2,
	T* sig_concat_deriv,
	T* sig2_deriv,
	uint64_t degree,
	const uint64_t* level_index,
	uint64_t sig_len
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	for (uint64_t i = tid; i < sig_len; i += nthreads)
		sig2_deriv[i] = sig_concat_deriv[i];
	__syncthreads();

	for (uint64_t level = degree; level > 0; --level) {
		for (uint64_t left_level = level - 1, right_level = 1;
			left_level > 0; --left_level, ++right_level) {
			const uint64_t target_start = level_index[level];
			const uint64_t left_start = level_index[left_level];
			const uint64_t right_start = level_index[right_level];
			const uint64_t left_size = level_index[left_level + 1] - left_start;
			const uint64_t right_size = level_index[right_level + 1] - right_start;

			for (uint64_t right_idx = tid; right_idx < right_size; right_idx += nthreads) {
				T accum = T(0);
				for (uint64_t left_idx = 0; left_idx < left_size; ++left_idx) {
					accum += sig_concat_deriv[target_start + left_idx * right_size + right_idx]
						* sig1[left_start + left_idx];
				}
				sig2_deriv[right_start + right_idx] += accum;
			}
			__syncthreads();
		}
	}

	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		for (uint64_t level = left_level + 1, right_level = 1;
			level <= degree; ++level, ++right_level) {
			const uint64_t target_start = level_index[level];
			const uint64_t left_start = level_index[left_level];
			const uint64_t right_start = level_index[right_level];
			const uint64_t left_size = level_index[left_level + 1] - left_start;
			const uint64_t right_size = level_index[right_level + 1] - right_start;

			for (uint64_t left_idx = tid; left_idx < left_size; left_idx += nthreads) {
				T accum = T(0);
				for (uint64_t right_idx = 0; right_idx < right_size; ++right_idx) {
					accum += sig_concat_deriv[target_start + left_idx * right_size + right_idx]
						* sig2[right_start + right_idx];
				}
				sig_concat_deriv[left_start + left_idx] += accum;
			}
			__syncthreads();
		}
	}
}

template<typename T>
__device__ void tensor_exp_backprop_block_(
	T* d_logsig,
	const T* d_sig,
	const T* log_sig,
	T* p_all,
	T* d_p,
	T* d_p_next,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index,
	uint64_t sig_len
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	for (uint64_t i = tid; i < sig_len; i += nthreads)
		d_logsig[i] = T(0);
	__syncthreads();

	if (degree <= 1) {
		for (uint64_t i = level_index[1] + tid; i < level_index[degree + 1]; i += nthreads)
			d_logsig[i] = d_sig[i];
		__syncthreads();
		return;
	}

	for (uint64_t i = tid; i < degree * sig_len; i += nthreads)
		p_all[i] = T(0);
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		p_all[i] = log_sig[i];
	__syncthreads();

	for (uint64_t n = 2; n <= degree; ++n) {
		const T inv_n = T(1) / static_cast<T>(n);
		T* p_curr = p_all + (n - 1) * sig_len;
		const T* p_prev = p_all + (n - 2) * sig_len;

		for (uint64_t target_level = n; target_level <= degree; ++target_level) {
			const uint64_t target_size = level_index[target_level + 1] - level_index[target_level];
			const uint64_t max_left = target_level - (n - 1);
			for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
				T sum = T(0);
				for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
					const uint64_t right_level = target_level - left_level;
					const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
					const uint64_t left_idx = idx / right_size;
					const uint64_t right_idx = idx - left_idx * right_size;
					sum += log_sig[level_index[left_level] + left_idx]
						* p_prev[level_index[right_level] + right_idx] * inv_n;
				}
				p_curr[level_index[target_level] + idx] = sum;
			}
		}
		__syncthreads();
	}

	for (uint64_t i = level_index[1] + tid; i < level_index[degree + 1]; i += nthreads)
		d_logsig[i] = d_sig[i];
	for (uint64_t i = tid; i < sig_len; i += nthreads)
		d_p[i] = T(0);
	__syncthreads();

	for (int64_t n_signed = static_cast<int64_t>(degree); n_signed >= 2; --n_signed) {
		const uint64_t n = static_cast<uint64_t>(n_signed);
		const T inv_n = T(1) / static_cast<T>(n);
		const T* p_prev = p_all + (n - 2) * sig_len;

		for (uint64_t i = tid; i < sig_len; i += nthreads)
			d_p_next[i] = T(0);
		__syncthreads();

		for (uint64_t target_level = n; target_level <= degree; ++target_level) {
			const uint64_t target_size = level_index[target_level + 1] - level_index[target_level];
			const uint64_t max_left = target_level - (n - 1);
			for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];

				for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
					const uint64_t left_idx = idx / right_size;
					const uint64_t right_idx = idx - left_idx * right_size;
					const uint64_t target_idx = level_index[target_level] + idx;
					const T upstream = (d_sig[target_idx] + d_p[target_idx]) * inv_n;
					myAtomicAdd(d_logsig + level_index[left_level] + left_idx,
						upstream * p_prev[level_index[right_level] + right_idx]);
					myAtomicAdd(d_p_next + level_index[right_level] + right_idx,
						upstream * log_sig[level_index[left_level] + left_idx]);
				}
				__syncthreads();
			}
		}

		T* tmp = d_p;
		d_p = d_p_next;
		d_p_next = tmp;
		__syncthreads();
	}

	for (uint64_t i = level_index[1] + tid; i < level_index[degree + 1]; i += nthreads)
		d_logsig[i] += d_p[i];
	__syncthreads();
}

template<typename T>
__global__ void signature_correction_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	const T* __restrict__ correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride,
	const uint64_t* __restrict__ level_index,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t full_sig_len,
	uint64_t sig_stride,
	uint64_t path_flat_len,
	T* __restrict__ workspace,
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	const T* my_correction = correction == nullptr ? nullptr
		: correction + batch_idx * correction_batch_stride;
	T* my_out = out + batch_idx * sig_stride;
	T* acc = workspace + local_batch_idx * 5 * full_sig_len;
	T* local = acc + full_sig_len;
	T* local_log = local + full_sig_len;
	T* power_prev = local_log + full_sig_len;
	T* power_curr = power_prev + full_sig_len;

	build_correction_block_(
		my_path, 0, local_log, my_correction, correction_len,
		data_dimension, dimension, degree, level_index);
	tensor_exp_block_(local_log, acc, power_prev, power_curr, dimension, degree, level_index);

	for (uint64_t step = 1; step + 1 < length; ++step) {
		const T* seg_corr = my_correction + step * correction_segment_stride;
		build_correction_block_(
			my_path, step, local_log, seg_corr, correction_len,
			data_dimension, dimension, degree, level_index);
		tensor_exp_block_(local_log, local, power_prev, power_curr, dimension, degree, level_index);
		sig_combine_block_(acc, local, degree, level_index);
	}

	if (scalar_term) {
		for (uint64_t i = tid; i < full_sig_len; i += nthreads)
			my_out[i] = acc[i];
	}
	else {
		for (uint64_t i = 1 + tid; i < full_sig_len; i += nthreads)
			my_out[i - 1] = acc[i];
	}
}

template<typename T>
void signature_cuda_core_(
	const T* path,          // GPU pointer, shape [batch_size, length, dimension] flattened
	T* out,                 // GPU pointer, shape [batch_size, sig_stride] flattened
	uint64_t batch_size,
	uint64_t dimension,     // transformed dimension
	uint64_t length,        // transformed length
	uint64_t degree,
	bool horner,
	bool scalar_term,
	uint64_t data_dimension,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	s_sig_context_valid = false;
	const uint64_t full_sig_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
	const uint64_t path_flat_len = dimension * length;

	// Handle trivial cases
	if (length <= 1) {
		// sig = (1, 0, 0, ..., 0) for each batch element. In scalar_term=false layout
		// the output is (0, 0, ...). Either way, zero the buffer then optionally write
		// the leading 1.
		const size_t out_bytes = checked_cuda_size_mul(
			checked_cuda_size_mul(
				static_cast<size_t>(batch_size), static_cast<size_t>(sig_stride),
				"CUDA signature trivial output"),
			sizeof(T), "CUDA signature trivial output");
		CUDA_CHECK(cudaMemset(out, 0, out_bytes));
		if (scalar_term) {
			T one = static_cast<T>(1);
			for (uint64_t i = 0; i < batch_size; ++i)
				cudaMemcpy(out + i * sig_stride, &one, sizeof(T), cudaMemcpyHostToDevice);
		}
		return;
	}

	if (degree == 0) {
		// degree-0 output is just the scalar 1 (nothing when scalar_term=false).
		if (scalar_term) {
			auto ones = std::make_unique<T[]>(batch_size);
			std::fill(ones.get(), ones.get() + batch_size, static_cast<T>(1));
			cudaMemcpy(out, ones.get(), batch_size * sizeof(T), cudaMemcpyHostToDevice);
		}
		return;
	}

	if (correction_len != 0) {
		auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
		host_populate_level_index(level_index_host.get(), dimension, degree + 2);
		const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
		CudaBuf<uint64_t> d_level_index(level_index_bytes);
		CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));
		const size_t workspace_row = checked_cuda_size_mul(
			5, static_cast<size_t>(full_sig_len),
			"CUDA signature correction workspace");
		CudaBatchWorkspace<T> workspace(
			batch_size, workspace_row,
			"CUDA signature correction workspace");
		const unsigned int threads_per_block = std::min(
			host_choose_threads_per_block(full_sig_len), 128u);
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset, workspace.capacity());
			signature_correction_ker<T><<<
				batch_chunk.grid, threads_per_block>>>(
					path, out, correction, correction_len,
					correction_batch_stride, correction_segment_stride,
					d_level_index.get(), data_dimension, dimension, length, degree,
					full_sig_len, sig_stride, path_flat_len, workspace.get(),
					scalar_term, batch_chunk.offset, batch_chunk.size);
			batch_offset += batch_chunk.size;
		}
		check_cuda_kernel_launch();
		cudaDeviceSynchronize();
		check_cuda_error();
		return;
	}

	if (horner) {
		signature_per_word_core_<T>(path, out, batch_size, dimension, length, degree, scalar_term);
		return;
	}

	// Naive (Chen's identity) fallback
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	const size_t increment_bytes = checked_cuda_size_mul(
		static_cast<size_t>(dimension), sizeof(T),
		"CUDA signature Chen fallback");
	size_t smem_size = checked_cuda_size_add(
		increment_bytes, 7, "CUDA signature Chen fallback") & ~size_t(7);
	smem_size = checked_cuda_size_add(
		smem_size,
		checked_cuda_size_mul(
			static_cast<size_t>(degree + 2), sizeof(uint64_t),
			"CUDA signature Chen fallback"),
		"CUDA signature Chen fallback");
	if (!try_configure_dynamic_smem(signature_naive_ker<T>, smem_size)) {
		signature_per_word_core_<T>(
			path, out, batch_size, dimension, length, degree, scalar_term);
		return;
	}

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	const size_t aligned_li_bytes = (level_index_bytes + sizeof(T) - 1) / sizeof(T) * sizeof(T);
	const size_t aligned_li_values = aligned_li_bytes / sizeof(T);
	const size_t workspace_stride = checked_cuda_size_add(
		aligned_li_values, static_cast<size_t>(sig_stride),
		"CUDA signature Chen fallback");
	CudaBatchWorkspace<T> workspace(
		batch_size, workspace_stride, "CUDA signature Chen fallback");
	uint64_t* d_level_index = reinterpret_cast<uint64_t*>(workspace.get());
	T* d_linear_sig = workspace.get() + aligned_li_values;
	CUDA_CHECK(cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));

	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, workspace.capacity());
		signature_naive_ker<T><<<
			batch_chunk.grid, threads_per_block, smem_size>>>(
				path, out, d_level_index,
				dimension, length, degree, sig_stride, path_flat_len,
				d_linear_sig, workspace_stride, scalar_term,
				batch_chunk.offset, batch_chunk.size
			);
		batch_offset += batch_chunk.size;
	}

	check_cuda_kernel_launch();
}

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
	bool horner,
	bool scalar_term = true,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	if (dimension == 0) throw std::invalid_argument("signature_cuda received path of dimension 0");
	validate_signature_correction_args_cuda_(correction, correction_len, dimension, degree, lead_lag);
	if (batch_size == 0)
		return;

	// Compute transformed dimensions
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		const size_t path_stride = checked_cuda_size_mul(
			static_cast<size_t>(length), static_cast<size_t>(dimension),
			"CUDA signature transformed path");
		const size_t transformed_stride = checked_cuda_size_mul(
			static_cast<size_t>(t_length), static_cast<size_t>(t_dimension),
			"CUDA signature transformed path");
		const uint64_t full_sig_len = host_sig_length(t_dimension, degree);
		const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
		CudaBatchWorkspace<T> transformed(
			batch_size, transformed_stride, "CUDA signature transformed path");
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset, transformed.capacity());
			cu_transform_path_<T>(
				path + chunk.offset * path_stride, transformed.get(), chunk.size,
				dimension, length, time_aug, lead_lag, end_time);
			const T* chunk_correction = correction_len == 0 ? nullptr
				: correction + chunk.offset * correction_batch_stride;
			signature_cuda_core_<T>(
				transformed.get(), out + chunk.offset * sig_stride, chunk.size,
				t_dimension, t_length, degree, horner, scalar_term, dimension,
				chunk_correction, correction_len, correction_batch_stride,
				correction_segment_stride);
			batch_offset += chunk.size;
		}
	}
	else {
		signature_cuda_core_<T>(
			path, out, batch_size, dimension, length, degree,
			horner, scalar_term, dimension, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
	}
}

template<typename T>
__global__ void sig_backprop_correction_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	const T* __restrict__ sig_derivs,
	const T* __restrict__ sig,
	const T* __restrict__ correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride,
	const uint64_t* __restrict__ level_index,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t full_sig_len,
	uint64_t sig_stride,
	uint64_t path_flat_len,
	T* __restrict__ workspace,
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	const T* my_correction = correction + batch_idx * correction_batch_stride;
	T* my_out = out + batch_idx * path_flat_len;
	const T* my_sig = sig + batch_idx * sig_stride;
	const T* my_derivs = sig_derivs + batch_idx * sig_stride;

	const uint64_t arrays_per_batch = degree + 12;
	T* base = workspace + local_batch_idx * arrays_per_batch * full_sig_len;
	T* sig_work = base;
	T* derivs_work = sig_work + full_sig_len;
	T* local_derivs = derivs_work + full_sig_len;
	T* local_log_derivs = local_derivs + full_sig_len;
	T* local_log = local_log_derivs + full_sig_len;
	T* inverse_correction = local_log + full_sig_len;
	T* local_sig = inverse_correction + full_sig_len;
	T* inverse_local_sig = local_sig + full_sig_len;
	T* power_prev = inverse_local_sig + full_sig_len;
	T* power_curr = power_prev + full_sig_len;
	T* p_all = power_curr + full_sig_len;
	T* d_p = p_all + degree * full_sig_len;
	T* d_p_next = d_p + full_sig_len;

	for (uint64_t i = tid; i < path_flat_len; i += nthreads)
		my_out[i] = T(0);
	__syncthreads();

	if (scalar_term) {
		for (uint64_t i = tid; i < full_sig_len; i += nthreads) {
			sig_work[i] = my_sig[i];
			derivs_work[i] = my_derivs[i];
		}
	}
	else {
		if (tid == 0) {
			sig_work[0] = T(1);
			derivs_work[0] = T(0);
		}
		for (uint64_t i = 1 + tid; i < full_sig_len; i += nthreads) {
			sig_work[i] = my_sig[i - 1];
			derivs_work[i] = my_derivs[i - 1];
		}
	}
	__syncthreads();

	for (int64_t seg = static_cast<int64_t>(length) - 2; seg >= 0; --seg) {
		const T* seg_corr = my_correction
			+ static_cast<uint64_t>(seg) * correction_segment_stride;
		build_correction_block_(
			my_path, static_cast<uint64_t>(seg), local_log, seg_corr, correction_len,
			data_dimension, dimension, degree, level_index);
		tensor_exp_block_(local_log, local_sig, power_prev, power_curr, dimension, degree, level_index);

		if (tid == 0)
			inverse_correction[0] = T(0);
		for (uint64_t i = 1 + tid; i < full_sig_len; i += nthreads)
			inverse_correction[i] = -local_log[i];
		__syncthreads();

		tensor_exp_block_(inverse_correction, inverse_local_sig, power_prev, power_curr, dimension, degree, level_index);
		sig_combine_block_(sig_work, inverse_local_sig, degree, level_index);

		uncombine_sig_deriv_block_(
			sig_work, local_sig, derivs_work, local_derivs,
			degree, level_index, full_sig_len);
		tensor_exp_backprop_block_(
			local_log_derivs, local_derivs, local_log, p_all, d_p, d_p_next,
			dimension, degree, level_index, full_sig_len);

		for (uint64_t d = tid; d < dimension; d += nthreads) {
			const T value = local_log_derivs[level_index[1] + d];
			my_out[(static_cast<uint64_t>(seg) + 1) * dimension + d] += value;
			my_out[static_cast<uint64_t>(seg) * dimension + d] -= value;
		}
		__syncthreads();
	}
}

template<typename T>
void sig_backprop_dense_chen_cuda_stream_(
	const T* path,
	T* out,
	const T* sig_derivs,
	const T* sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool scalar_term,
	cudaStream_t stream
) {
	const int dim = static_cast<int>(dimension);
	const int steps = static_cast<int>(length - 1);
	const uint64_t full_sig_len = host_sig_length(dimension, degree);
	const uint64_t state_size = full_sig_len - 1;
	const uint64_t sig_stride = scalar_term ? full_sig_len : state_size;
	const uint64_t exp_size = degree > 1
		? host_sig_length(dimension, degree - 1) - 1 : 0;
	const uint64_t path_stride = length * dimension;

	const size_t state_elements = static_cast<size_t>(batch_size) * state_size;
	const size_t exp_elements = static_cast<size_t>(batch_size) * exp_size;
	const size_t increment_elements = static_cast<size_t>(batch_size) * dimension;
	const size_t workspace_bytes = (
		2 * state_elements + 2 * exp_elements + increment_elements) * sizeof(T);
	T* workspace = static_cast<T*>(ensure_inc_grad_buf(workspace_bytes));
	T* state = workspace;
	T* adjoint = state + state_elements;
	T* exp_values = adjoint + state_elements;
	T* exp_adjoint = exp_values + exp_elements;
	T* increments = exp_adjoint + exp_elements;
	const uint64_t scalar_offset = scalar_term ? 1 : 0;
	CUDA_CHECK(cudaMemcpy2DAsync(
		state, state_size * sizeof(T), sig + scalar_offset,
		sig_stride * sizeof(T), state_size * sizeof(T), batch_size,
		cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpy2DAsync(
		adjoint, state_size * sizeof(T), sig_derivs + scalar_offset,
		sig_stride * sizeof(T), state_size * sizeof(T), batch_size,
		cudaMemcpyDeviceToDevice, stream));

	const int block = 256;
	for (int step = steps - 1; step >= 0; --step) {
		sig_dense_chen_prepare_step_ker<T><<<
			static_cast<unsigned int>(batch_size), block,
			dimension * sizeof(T), stream>>>(
				path, state, exp_values, increments, dim, step,
				static_cast<int>(degree), state_size, exp_size, path_stride);

		sig_dense_chen_pullback_step_ker<T><<<
			static_cast<unsigned int>(batch_size), block,
			dimension * sizeof(T), stream>>>(
				state, exp_values, exp_adjoint, adjoint, increments, out, dim, step,
				static_cast<int>(degree), state_size, exp_size, path_stride);
	}
	increment_to_path_grad_inplace_ker<T><<<
		static_cast<unsigned int>(batch_size), 64, 0, stream>>>(
			out, static_cast<int>(length), dim);
	check_cuda_kernel_launch();
}

template<typename T>
void sig_backprop_cuda_stream_(
	const T* path,
	T* out,
	const T* sig_derivs,
	const T* sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool scalar_term,
	cudaStream_t stream
);

template<typename T>
void sig_backprop_cuda_core_(
	const T* path,
	T* out,
	const T* sig_derivs,
	const T* sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool scalar_term,
	uint64_t data_dimension,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	const uint64_t full_sig_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
	const uint64_t path_flat_len = dimension * length;

	if (length <= 1 || degree == 0) {
		cudaMemset(out, 0, batch_size * path_flat_len * sizeof(T));
		return;
	}

	if (correction_len != 0) {
		auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
		host_populate_level_index(level_index_host.get(), dimension, degree + 2);
		const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
		CudaBuf<uint64_t> d_level_index(level_index_bytes);
		CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));
		const size_t workspace_row = checked_cuda_size_mul(
			static_cast<size_t>(degree + 12),
			static_cast<size_t>(full_sig_len),
			"CUDA signature backprop correction workspace");
		CudaBatchWorkspace<T> workspace(
			batch_size, workspace_row,
			"CUDA signature backprop correction workspace");
		const unsigned int threads_per_block = host_choose_threads_per_block(full_sig_len);
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset, workspace.capacity());
			sig_backprop_correction_ker<T><<<
				batch_chunk.grid, threads_per_block>>>(
					path, out, sig_derivs, sig, correction, correction_len,
					correction_batch_stride, correction_segment_stride,
					d_level_index.get(), data_dimension, dimension, length, degree,
					full_sig_len, sig_stride, path_flat_len, workspace.get(),
					scalar_term, batch_chunk.offset, batch_chunk.size);
			batch_offset += batch_chunk.size;
		}
		check_cuda_kernel_launch();
		cudaDeviceSynchronize();
		check_cuda_error();
		return;
	}

	{
		std::lock_guard<std::mutex> workspace_lock(s_inc_grad_buf_mu);
		sig_backprop_cuda_stream_<T>(
			path, out, sig_derivs, sig, batch_size, dimension, length, degree,
			scalar_term, nullptr);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	check_cuda_error();
}

template<typename T>
void sig_backprop_cuda_stream_(
	const T* path,
	T* out,
	const T* sig_derivs,
	const T* sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	bool scalar_term,
	cudaStream_t stream
) {
	if (dimension == 0)
		throw std::invalid_argument("sig_backprop_cuda received path of dimension 0");
	if (length <= 1 || degree == 0) {
		CUDA_CHECK(cudaMemsetAsync(
			out, 0, batch_size * length * dimension * sizeof(T), stream));
		return;
	}
	if (degree > MAX_GENERIC_DEGREE)
		throw std::invalid_argument(
			"sig_backprop on CUDA requires degree <= 64. "
			"Use CPU or reduce the truncation level.");

	const int steps = static_cast<int>(length - 1);
	const int dim = static_cast<int>(dimension);
	const uint64_t full_sig_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
	const uint64_t context_size = degree > 1
		? host_sig_length(dimension, degree - 1) - 1 : dimension;
	const uint64_t dense_state_size = full_sig_len - 1;
	const uint64_t dense_lower_state_size = degree > 1 ? context_size : 0;
	const size_t dense_static_shared_bytes =
		2 * (MAX_GENERIC_DEGREE + 1) * sizeof(uint64_t);
	const uint64_t dense_contraction_count = degree * (degree - 1);
	const size_t dense_workspace_bytes = static_cast<size_t>(batch_size)
		* (2 * dense_state_size + 2 * dense_lower_state_size + dimension)
		* sizeof(T);
	size_t dense_free_bytes = 0;
	size_t dense_total_bytes = 0;
	CUDA_CHECK(cudaMemGetInfo(&dense_free_bytes, &dense_total_bytes));
	const size_t dense_reusable_bytes = dense_free_bytes + s_inc_grad_buf_size;
	const size_t dense_reserve_bytes = std::max(
		size_t(512) * 1024 * 1024, dense_total_bytes / 16);
	const bool use_dense_chen = degree >= 2
		&& dense_state_size >= 256 * dense_contraction_count
		&& batch_size >= 32
		&& dimension * sizeof(T) + dense_static_shared_bytes <= CUDA_BASE_DYNAMIC_SMEM
		&& full_sig_len <= INT_MAX && batch_size <= INT_MAX
		&& dense_workspace_bytes <= dense_reusable_bytes
		&& dense_reusable_bytes - dense_workspace_bytes >= dense_reserve_bytes;
	if (use_dense_chen) {
		s_sig_context_valid = false;
		sig_backprop_dense_chen_cuda_stream_<T>(
			path, out, sig_derivs, sig, batch_size, dimension, length,
			degree, scalar_term, stream);
		return;
	}
	const size_t scan_context_elements = static_cast<size_t>(batch_size)
		* static_cast<size_t>(steps) * context_size;
	const size_t scan_increment_elements = static_cast<size_t>(batch_size)
		* static_cast<size_t>(steps) * dimension;
	const size_t scan_workspace_bytes =
		(2 * scan_context_elements + scan_increment_elements) * sizeof(T);
	int scan_group_width = 1;
	while (scan_group_width < dim)
		scan_group_width *= 2;
	const int scan_grouped_block = 128;
	const int scan_groups_per_block = scan_grouped_block / scan_group_width;
	const size_t scan_shared_bytes = static_cast<size_t>(scan_groups_per_block)
		* 2 * context_size * sizeof(T);
	const bool use_level_scan = degree >= 2 && degree <= 12
		&& dimension <= 32 && steps >= 512 && batch_size <= 65535
		&& context_size > 3 * dimension
		&& full_sig_len <= INT_MAX
		&& scan_shared_bytes <= CUDA_BASE_DYNAMIC_SMEM
		&& scan_workspace_bytes <= dense_reusable_bytes
		&& dense_reusable_bytes - scan_workspace_bytes >= dense_reserve_bytes;
	if (use_level_scan) {
		T* allocation = static_cast<T*>(ensure_inc_grad_buf(scan_workspace_bytes));
		T* prefix_context = allocation;
		T* adjoint_context = prefix_context + scan_context_elements;
		T* increment_grads = adjoint_context + scan_context_elements;
		CUDA_CHECK(cudaMemsetAsync(
			adjoint_context, 0, scan_context_elements * sizeof(T), stream));
		CUDA_CHECK(cudaMemsetAsync(
			increment_grads, 0, scan_increment_elements * sizeof(T), stream));
		const bool reuse_prefix_context = s_sig_context_valid
			&& s_sig_context_kind == 1
			&& s_sig_context_path == path && s_sig_context_sig == sig
			&& s_sig_context_batch_size == batch_size
			&& s_sig_context_dimension == dimension
			&& s_sig_context_length == length
			&& s_sig_context_degree == degree
			&& s_sig_context_value_size == sizeof(T)
			&& s_sig_context_scalar_term == scalar_term;

		uint64_t level_offset = dimension;
		for (uint64_t level = 2; !reuse_prefix_context && level < degree; ++level) {
			const uint64_t level_size = host_power(dimension, level);
			const auto scan_chunk = make_cuda_batch_grid_chunk(
				level_size, batch_size, 0);
			#define LAUNCH_LEVEL_PREFIX(D) \
				case D: sig_context_scan_persistent_ker<T, D, 256><<< \
					scan_chunk.grid, 256, 0, stream>>>( \
						path, prefix_context, dim, steps, steps, context_size, \
						level_offset, length * dimension, \
						scan_chunk.offset, scan_chunk.size); break;

			switch (level) {
				LAUNCH_LEVEL_PREFIX(2)  LAUNCH_LEVEL_PREFIX(3)
				LAUNCH_LEVEL_PREFIX(4)  LAUNCH_LEVEL_PREFIX(5)
				LAUNCH_LEVEL_PREFIX(6)  LAUNCH_LEVEL_PREFIX(7)
				LAUNCH_LEVEL_PREFIX(8)  LAUNCH_LEVEL_PREFIX(9)
				LAUNCH_LEVEL_PREFIX(10) LAUNCH_LEVEL_PREFIX(11)
				default: break;
			}
			#undef LAUNCH_LEVEL_PREFIX
			level_offset += level_size;
		}
		if (!reuse_prefix_context) {
			s_sig_context_path = path;
			s_sig_context_sig = sig;
			s_sig_context_batch_size = batch_size;
			s_sig_context_dimension = dimension;
			s_sig_context_length = length;
			s_sig_context_degree = degree;
			s_sig_context_value_size = sizeof(T);
			s_sig_context_scalar_term = scalar_term;
			s_sig_context_kind = 1;
			s_sig_context_valid = true;
		}

		const int block = 256;
		const uint64_t total_times = batch_size * static_cast<uint64_t>(steps);
		const int group_width = scan_group_width;
		const int grouped_block = scan_grouped_block;
		const int groups_per_block = scan_groups_per_block;
		for (uint64_t level = degree; level >= 2; --level) {
			const uint64_t level_size = host_power(dimension, level);
			const uint64_t current_offset = host_sig_length(dimension, level - 1) - 1;
			const bool top_level = level == degree;
			if (!top_level) {
				const auto adjoint_chunk = make_cuda_batch_grid_chunk(
					level_size, batch_size, 0);
				sig_level_adjoint_scan_ker<T, 256><<<
					adjoint_chunk.grid, 256, 0, stream>>>(
						adjoint_context, sig_derivs, steps, steps, context_size,
						current_offset, sig_stride, scalar_term,
						adjoint_chunk.offset, adjoint_chunk.size);
			}
			const unsigned int grouped_grid = static_cast<unsigned int>(
				(total_times + groups_per_block - 1) / groups_per_block);
			const size_t grouped_shared_bytes = static_cast<size_t>(
				groups_per_block) * 2 * current_offset * sizeof(T);
			sig_level_horner_backprop_ker<T><<<
				grouped_grid, grouped_block, grouped_shared_bytes, stream>>>(
					path, sig_derivs, prefix_context, adjoint_context,
					increment_grads, dim, steps, static_cast<int>(level),
					group_width, level_size, current_offset, batch_size, sig_stride,
					context_size, steps, length * dimension, scalar_term,
					top_level);
		}

		const auto level_one_chunk = make_cuda_batch_grid_chunk(
			dimension, batch_size, 0);
		sig_level_adjoint_scan_ker<T, 256><<<
			level_one_chunk.grid, 256, 0, stream>>>(
				adjoint_context, sig_derivs, steps, steps, context_size,
				0, sig_stride, scalar_term,
				level_one_chunk.offset, level_one_chunk.size);
		const unsigned int grouped_grid = static_cast<unsigned int>(
			(total_times + groups_per_block - 1) / groups_per_block);
		sig_level_horner_backprop_ker<T><<<
			grouped_grid, grouped_block, 0, stream>>>(
				path, sig_derivs, prefix_context, adjoint_context,
				increment_grads, dim, steps, 1, group_width, dimension, 0,
				batch_size, sig_stride, context_size, steps,
				length * dimension, scalar_term, false);
		const auto convert_chunk = make_cuda_batch_grid_chunk(
			(length * dimension + block - 1) / block,
			batch_size, 0);
		increment_to_path_grad_batch_ker<T><<<
			convert_chunk.grid, block, 0, stream>>>(
				increment_grads, out, static_cast<int>(length), dim,
				convert_chunk.offset, convert_chunk.size);
		check_cuda_kernel_launch();
		return;
	}
	const uint64_t padded_steps = static_cast<uint64_t>(steps);
	const size_t context_elements = static_cast<size_t>(batch_size)
		* padded_steps * context_size;
	const size_t increment_elements = static_cast<size_t>(batch_size)
		* steps * dimension;
	const uint64_t grouped_work_per_thread = degree > 1
		? host_power(dimension, degree - 2) : UINT64_MAX;
	const int grouped_block = 64;
	const int groups_per_block = dim <= grouped_block
		? grouped_block / dim : 0;
	const size_t grouped_shared_bytes = static_cast<size_t>(
		groups_per_block) * (2 * context_size + dimension) * sizeof(T);
	const size_t grouped_allocation_bytes =
		(2 * context_elements + increment_elements) * sizeof(T);
	const bool use_grouped_backprop = degree >= 2 && degree <= 12
		&& batch_size <= 65535 && groups_per_block != 0
		&& dim <= 32 && 32 % dim == 0
		&& grouped_work_per_thread <= 16
		&& grouped_shared_bytes <= CUDA_BASE_DYNAMIC_SMEM
		&& grouped_allocation_bytes <= dense_reusable_bytes
		&& dense_reusable_bytes - grouped_allocation_bytes >= dense_reserve_bytes
		&& context_elements <= UINT32_MAX
		&& full_sig_len <= UINT32_MAX;
	if (use_grouped_backprop) {
		T* allocation = static_cast<T*>(
			ensure_inc_grad_buf(grouped_allocation_bytes));
		T* prefix_context = allocation;
		T* suffix_context = prefix_context + context_elements;
		T* increment_grads = suffix_context + context_elements;
		const int block = 256;

		const bool reuse_grouped_context = degree > 2
			&& s_sig_context_valid && s_sig_context_kind == 2
			&& s_sig_context_path == path && s_sig_context_sig == sig
			&& s_sig_context_batch_size == batch_size
			&& s_sig_context_dimension == dimension
			&& s_sig_context_length == length
			&& s_sig_context_degree == degree
			&& s_sig_context_value_size == sizeof(T)
			&& s_sig_context_scalar_term == scalar_term;

		uint64_t level_offset = dimension;
		for (uint64_t level = 2; !reuse_grouped_context && level < degree;
			++level) {
			const uint64_t level_size = host_power(dimension, level);
			const int scan_block = 256;
			const auto scan_chunk = make_cuda_batch_grid_chunk(
				level_size, batch_size, 0);
			#define LAUNCH_PREFIX(D) \
				case D: \
					sig_context_scan_persistent_ker<T, D, 256><<< \
						scan_chunk.grid, scan_block, 0, stream>>>( \
							path, prefix_context, dim, steps, padded_steps, context_size, \
							level_offset, length * dimension, \
							scan_chunk.offset, scan_chunk.size); \
					sig_suffix_from_prefix_ker<T, D><<< \
					static_cast<unsigned int>((batch_size * static_cast<uint64_t>(steps) \
						* level_size + scan_block - 1) / scan_block), \
					scan_block, 0, stream>>>( \
						path, sig, prefix_context, suffix_context, dim, steps, batch_size, \
						sig_stride, padded_steps, context_size, level_offset, level_size, \
						length * dimension, scalar_term); \
					break;

			switch (level) {
				LAUNCH_PREFIX(1)  LAUNCH_PREFIX(2)  LAUNCH_PREFIX(3)
				LAUNCH_PREFIX(4)  LAUNCH_PREFIX(5)  LAUNCH_PREFIX(6)
				LAUNCH_PREFIX(7)  LAUNCH_PREFIX(8)  LAUNCH_PREFIX(9)
				LAUNCH_PREFIX(10) LAUNCH_PREFIX(11)
				default: break;
			}
			#undef LAUNCH_PREFIX
			level_offset += level_size;
		}
		if (degree > 2 && !reuse_grouped_context) {
			s_sig_context_path = path;
			s_sig_context_sig = sig;
			s_sig_context_batch_size = batch_size;
			s_sig_context_dimension = dimension;
			s_sig_context_length = length;
			s_sig_context_degree = degree;
			s_sig_context_value_size = sizeof(T);
			s_sig_context_scalar_term = scalar_term;
			s_sig_context_kind = 2;
			s_sig_context_valid = true;
		}
		const uint64_t total_times = batch_size
			* static_cast<uint64_t>(steps);
		const unsigned int grouped_grid = static_cast<unsigned int>(
			(total_times + groups_per_block - 1) / groups_per_block);
		#define LAUNCH_GROUPED_BWD(D) \
			case D: sig_grouped_backprop_ker<T, D><<< \
				grouped_grid, grouped_block, grouped_shared_bytes, stream>>>( \
					path, sig_derivs, prefix_context, suffix_context, increment_grads, \
					dim, steps, batch_size, sig_stride, context_size, padded_steps, \
					length * dimension, scalar_term); break;

		switch (degree) {
			LAUNCH_GROUPED_BWD(2)  LAUNCH_GROUPED_BWD(3)
			LAUNCH_GROUPED_BWD(4)  LAUNCH_GROUPED_BWD(5)
			LAUNCH_GROUPED_BWD(6)  LAUNCH_GROUPED_BWD(7)
			LAUNCH_GROUPED_BWD(8)  LAUNCH_GROUPED_BWD(9)
			LAUNCH_GROUPED_BWD(10) LAUNCH_GROUPED_BWD(11)
			LAUNCH_GROUPED_BWD(12)
			default: break;
		}
		#undef LAUNCH_GROUPED_BWD
		const uint64_t path_grad_values = batch_size * length * dimension;
		const unsigned int convert_grid = static_cast<unsigned int>(
			(path_grad_values + block - 1) / block);
		increment_to_path_grad_ker<T><<<convert_grid, block, 0, stream>>>(
			increment_grads, nullptr, out, static_cast<int>(batch_size),
			static_cast<int>(length), dim, dimension,
			static_cast<uint64_t>(steps) * dimension, 0, 0);
		check_cuda_kernel_launch();
		return;
	}

	const size_t inc_grad_row = checked_cuda_size_mul(
		static_cast<size_t>(steps), static_cast<size_t>(dimension),
		"CUDA signature backprop");
	CudaBatchWorkspace<T> inc_workspace(
		batch_size, inc_grad_row, "CUDA signature backprop");
	T* d_inc_grads = inc_workspace.get();

	auto level_index = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index.get(), dimension, degree + 2);
	unsigned int block = 128;
	const int requested_chunk = (steps < BWD_CHUNK) ? steps : BWD_CHUNK;
	const size_t bytes_per_vector = checked_cuda_size_mul(
		static_cast<size_t>(dimension), sizeof(T),
		"CUDA signature backprop");
	const size_t requested_vectors = checked_cuda_size_add(
		static_cast<size_t>(requested_chunk), block / 32,
		"CUDA signature backprop");
	const size_t requested_smem = checked_cuda_size_mul(
		requested_vectors, bytes_per_vector, "CUDA signature backprop");
	CudaSharedMemoryLimits smem_limits = {
		CUDA_BASE_DYNAMIC_SMEM, CUDA_BASE_DYNAMIC_SMEM
	};
	if (requested_smem > CUDA_BASE_DYNAMIC_SMEM)
		smem_limits = cuda_shared_memory_limits();
	size_t max_vectors = bytes_per_vector == 0
		? 0 : smem_limits.optin_bytes / bytes_per_vector;
	while (block > 32 && max_vectors <= block / 32)
		block /= 2;
	const unsigned int num_warps = block / 32;
	const bool shared_cache_candidate = max_vectors > num_warps;
	const int chunk_size = shared_cache_candidate
		? static_cast<int>(std::min<size_t>(
			static_cast<size_t>(requested_chunk), max_vectors - num_warps))
		: 1;
	const size_t smem_size = shared_cache_candidate
		? checked_cuda_size_mul(
			checked_cuda_size_add(
				static_cast<size_t>(chunk_size), num_warps,
				"CUDA signature backprop"),
			bytes_per_vector, "CUDA signature backprop")
		: 0;
	const size_t path_stride = checked_cuda_size_mul(
		static_cast<size_t>(length), static_cast<size_t>(dimension),
		"CUDA signature backprop");
	const int convert_block = 256;
	const unsigned int convert_grid_x = static_cast<unsigned int>(
		(path_stride + convert_block - 1) / convert_block);

	for (uint64_t outer_offset = 0; outer_offset < batch_size;) {
		const uint64_t outer_size = std::min<uint64_t>(
			inc_workspace.capacity(), batch_size - outer_offset);
		const size_t chunk_values = checked_cuda_size_mul(
			static_cast<size_t>(outer_size), inc_grad_row,
			"CUDA signature backprop");
		const size_t chunk_bytes = checked_cuda_size_mul(
			chunk_values, sizeof(T), "CUDA signature backprop");
		CUDA_CHECK(cudaMemsetAsync(d_inc_grads, 0, chunk_bytes, stream));

		for (uint64_t k = 1; k <= degree; ++k) {
			const uint64_t level_size = host_power(dimension, k);
			const uint64_t level_offset = scalar_term
				? level_index[k]
				: level_index[k] - 1;
			const unsigned int grid_x = static_cast<unsigned int>(
				(level_size + block - 1) / block);
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				grid_x, batch_size, outer_offset, outer_size);

			#define LAUNCH_BWD(D) \
				case D: \
					if (shared_cache_candidate \
						&& (smem_size <= CUDA_BASE_DYNAMIC_SMEM \
							|| try_configure_dynamic_smem( \
								sig_backprop_per_word_ker<T, D, true>, smem_size, \
								smem_limits))) { \
						sig_backprop_per_word_ker<T, D, true><<< \
							batch_chunk.grid, block, smem_size, stream>>>( \
							path, sig, sig_derivs, d_inc_grads, dim, steps, \
							chunk_size, sig_stride, level_offset, level_size, \
							path_stride, scalar_term, batch_chunk.offset, \
							batch_chunk.size); \
					} else { \
						sig_backprop_per_word_ker<T, D, false><<< \
							batch_chunk.grid, block, 0, stream>>>( \
							path, sig, sig_derivs, d_inc_grads, dim, steps, 1, \
							sig_stride, level_offset, level_size, path_stride, \
							scalar_term, batch_chunk.offset, batch_chunk.size); \
					} \
					break;

			switch (k) {
				LAUNCH_BWD(1)  LAUNCH_BWD(2)  LAUNCH_BWD(3)  LAUNCH_BWD(4)
				LAUNCH_BWD(5)  LAUNCH_BWD(6)  LAUNCH_BWD(7)  LAUNCH_BWD(8)
				LAUNCH_BWD(9)  LAUNCH_BWD(10) LAUNCH_BWD(11) LAUNCH_BWD(12)
				default:
					if (shared_cache_candidate
						&& (smem_size <= CUDA_BASE_DYNAMIC_SMEM
							|| try_configure_dynamic_smem(
								sig_backprop_per_word_generic_ker<T, true>, smem_size,
								smem_limits))) {
						sig_backprop_per_word_generic_ker<T, true><<<
							batch_chunk.grid, block, smem_size, stream>>>(
							path, sig, sig_derivs, d_inc_grads, dim, steps,
							chunk_size, static_cast<int>(k), sig_stride,
							level_offset, level_size, path_stride, scalar_term,
							batch_chunk.offset, batch_chunk.size);
					} else {
						sig_backprop_per_word_generic_ker<T, false><<<
							batch_chunk.grid, block, 0, stream>>>(
							path, sig, sig_derivs, d_inc_grads, dim, steps, 1,
							static_cast<int>(k), sig_stride, level_offset,
							level_size, path_stride, scalar_term,
							batch_chunk.offset, batch_chunk.size);
					}
					break;
			}
			#undef LAUNCH_BWD
		}

		const auto convert_chunk = make_cuda_batch_grid_chunk(
			convert_grid_x, batch_size, outer_offset, outer_size);
		increment_to_path_grad_batch_ker<T><<<
			convert_chunk.grid, convert_block, 0, stream>>>(
				d_inc_grads, out, static_cast<int>(length), dim,
				convert_chunk.offset, convert_chunk.size);
		outer_offset += outer_size;
	}
	check_cuda_kernel_launch();
}

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
	T end_time,
	bool scalar_term = true,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	if (dimension == 0) throw std::invalid_argument("sig_backprop_cuda received path of dimension 0");
	validate_signature_correction_args_cuda_(correction, correction_len, dimension, degree, lead_lag);
	if (batch_size == 0)
		return;

	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		const size_t path_stride = checked_cuda_size_mul(
			static_cast<size_t>(length), static_cast<size_t>(dimension),
			"CUDA signature backprop transformed path");
		const size_t transformed_stride = checked_cuda_size_mul(
			static_cast<size_t>(t_length), static_cast<size_t>(t_dimension),
			"CUDA signature backprop transformed path");
		const size_t workspace_row = checked_cuda_size_mul(
			2, transformed_stride,
			"CUDA signature backprop transformed path");
		const uint64_t full_sig_len = host_sig_length(t_dimension, degree);
		const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
		CudaBatchWorkspace<T> transformed(
			batch_size, workspace_row,
			"CUDA signature backprop transformed path");
		T* transformed_derivs = transformed.get()
			+ transformed.capacity() * transformed_stride;
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset, transformed.capacity());
			cu_transform_path_<T>(
				path + chunk.offset * path_stride, transformed.get(), chunk.size,
				dimension, length, time_aug, lead_lag, end_time);
			const T* chunk_correction = correction_len == 0 ? nullptr
				: correction + chunk.offset * correction_batch_stride;
			sig_backprop_cuda_core_<T>(
				transformed.get(), transformed_derivs,
				sig_derivs + chunk.offset * sig_stride,
				sig + chunk.offset * sig_stride, chunk.size,
				t_dimension, t_length, degree, scalar_term, dimension,
				chunk_correction, correction_len, correction_batch_stride,
				correction_segment_stride);
			cu_transform_path_backprop_<T>(
				transformed_derivs, out + chunk.offset * path_stride, chunk.size,
				dimension, length, time_aug, lead_lag, end_time);
			batch_offset += chunk.size;
		}
	}
	else {
		sig_backprop_cuda_core_<T>(
			path, out, sig_derivs, sig, batch_size, dimension, length, degree,
			scalar_term, dimension, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
	}
}


#include "cu_macros.h"


extern "C" {


	CUSIG_API int signature_cuda_f(
		const float* path, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time,
		bool horner, bool scalar_term,
		const float* correction, uint64_t correction_len,
		uint64_t correction_batch_stride, uint64_t correction_segment_stride
	) noexcept {
		CUDA_SAFE_CALL(signature_cuda_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int signature_cuda_d(
		const double* path, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time,
		bool horner, bool scalar_term,
		const double* correction, uint64_t correction_len,
		uint64_t correction_batch_stride, uint64_t correction_segment_stride
	) noexcept {
		CUDA_SAFE_CALL(signature_cuda_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	// =====================================================================
	// backprop
	// =====================================================================


	CUSIG_API int sig_backprop_cuda_f(
		const float* path, float* out,
		const float* sig_derivs, const float* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time, bool scalar_term,
		const float* correction, uint64_t correction_len,
		uint64_t correction_batch_stride, uint64_t correction_segment_stride
	) noexcept {
		CUDA_SAFE_CALL(sig_backprop_cuda_<float>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int sig_backprop_cuda_d(
		const double* path, double* out,
		const double* sig_derivs, const double* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time, bool scalar_term,
		const double* correction, uint64_t correction_len,
		uint64_t correction_batch_stride, uint64_t correction_segment_stride
	) noexcept {
		CUDA_SAFE_CALL(sig_backprop_cuda_<double>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}
}
