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

// Runtime reciprocal for the generic (non-template) fallback kernels
template<typename T> __device__ __forceinline__ T d_recip_rt(int n) {
	return (n <= 12) ? d_recip<T>(n) : static_cast<T>(1) / static_cast<T>(n);
}

// Chunk sizes for batching time steps into shared memory (reduces sync overhead).
constexpr int SIG_CHUNK = 128;   // forward kernel
constexpr int BWD_CHUNK = 32;    // backward kernel (needs more shared mem for reduction)

template<typename T, int DEGREE>
__global__ void signature_per_word_ker(
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

// ---------------------------------------------------------------------------
// Generic (non-template) per-word forward kernel for degree > 12.
// Same algorithm as the template version but with runtime degree parameter.
// ---------------------------------------------------------------------------

constexpr int MAX_GENERIC_DEGREE = 64;

template<typename T>
__global__ void signature_per_word_generic_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	const int dim,
	const int steps,
	const int degree,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride
) {
	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t batch_idx = blockIdx.y;
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
	T comp[MAX_GENERIC_DEGREE + 1];
	for (int i = 0; i <= degree; ++i) { pref[i] = T(0); comp[i] = T(0); }
	pref[0] = T(1);

	const T* batch_path = path + batch_idx * path_stride;

	for (int chunk_start = 0; chunk_start < steps; chunk_start += SIG_CHUNK) {
		const int chunk_end = (chunk_start + SIG_CHUNK < steps) ? chunk_start + SIG_CHUNK : steps;
		const int chunk_len = chunk_end - chunk_start;

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

			for (int sd = degree; sd > 0; --sd) {
				T h = T(0);
				for (int k = 0; k < sd; ++k) {
					const T scale = inc[letters[k]] * d_recip_rt<T>(sd - k);
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
		out[batch_idx * sig_size + level_offset + word_idx] = pref[degree];
	}
}

// ---------------------------------------------------------------------------
// Generic (non-template) per-word backward kernel for degree > 12.
// ---------------------------------------------------------------------------

template<typename T>
__global__ __launch_bounds__(128)
void sig_backprop_per_word_generic_ker(
	const T* __restrict__ path,
	const T* __restrict__ sig,
	const T* __restrict__ sig_grads,
	T* __restrict__ inc_grads,
	const int dim,
	const int steps,
	const int degree,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride
) {
	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t batch_idx = blockIdx.y;
	const bool active = word_idx < level_size;

	extern __shared__ char smem[];
	T* shared_inc = reinterpret_cast<T*>(smem);
	const unsigned num_warps = blockDim.x >> 5;
	T* shared_letter_grads = shared_inc + BWD_CHUNK * dim;

	for (int i = threadIdx.x; i < dim * (int)num_warps; i += blockDim.x)
		shared_letter_grads[i] = T(0);

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
		uint64_t off = 1;
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
	T* batch_inc_grad = inc_grads + batch_idx * static_cast<uint64_t>(steps) * dim;

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned lane = threadIdx.x & 31;

	for (int chunk_end = steps; chunk_end > 0; chunk_end -= BWD_CHUNK) {
		const int chunk_start = (chunk_end - BWD_CHUNK > 0) ? chunk_end - BWD_CHUNK : 0;
		const int chunk_len = chunk_end - chunk_start;

		__syncthreads();
		{
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
			const T* inc = shared_inc + t_local * dim;

			if (active) {
				for (int sd = degree; sd > 0; --sd) {
					T h = T(0);
					for (int k = 0; k < sd; ++k) {
						h = (-inc[letters[k]] * d_recip_rt<T>(sd - k)) * (pref[k] + h);
					}
					pref[sd] += h;
				}
			}

			T letter_grads_local[MAX_GENERIC_DEGREE];
			for (int i = 0; i < degree; ++i) letter_grads_local[i] = T(0);

			if (active) {
				for (int pref_len = 0; pref_len < degree; ++pref_len) {
					const T pref_val = pref[pref_len];
					T prev_prod = T(1);

					{
						T temp_prod = prev_prod;
						T temp_grad = temp_prod * suf[degree - pref_len - 1];
						int denom = 2;
						for (int pos = pref_len + 1; pos < degree; ++pos, ++denom) {
							temp_prod *= inc[letters[pos]] * d_recip_rt<T>(denom);
							temp_grad += temp_prod * suf[degree - pos - 1];
						}
						letter_grads_local[pref_len] += temp_grad * pref_val;
					}

					int denom_lp = 2;
					for (int lp = pref_len + 1; lp < degree; ++lp, ++denom_lp) {
						prev_prod *= inc[letters[lp - 1]] * d_recip_rt<T>(denom_lp);
						T temp_prod = prev_prod;
						T temp_grad = temp_prod * suf[degree - lp - 1];
						int denom = denom_lp + 1;
						for (int pos = lp + 1; pos < degree; ++pos, ++denom) {
							temp_prod *= inc[letters[pos]] * d_recip_rt<T>(denom);
							temp_grad += temp_prod * suf[degree - pos - 1];
						}
						letter_grads_local[lp] += temp_grad * pref_val;
					}
				}
			}

			for (int letter = 0; letter < dim; ++letter) {
				T val = T(0);
				if (active) {
					for (int lp = 0; lp < degree; ++lp) {
						if (letters[lp] == letter) val += letter_grads_local[lp];
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

			__syncthreads();

			for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
				T sum = T(0);
				for (unsigned w = 0; w < num_warps; ++w)
					sum += shared_letter_grads[letter * num_warps + w];
				if (sum != T(0))
					myAtomicAdd(&batch_inc_grad[static_cast<uint64_t>(t) * dim + letter], sum);
			}

			if (active) {
				for (int m = degree - 1; m > 0; --m) {
					const int base_pos = degree - m;
					T h = T(0);
					for (int p = m; p >= 1; --p) {
						int lp = base_pos + (p - 1);
						h = (inc[letters[lp]] * d_recip_rt<T>(p)) * (suf[m - p] + h);
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

template<typename T, int DEGREE>
__global__ __launch_bounds__(128)
void sig_backprop_per_word_ker(
	const T* __restrict__ path,           // [batch, length, dim]
	const T* __restrict__ sig,            // [batch, sig_size] (forward signature)
	const T* __restrict__ sig_grads,      // [batch, sig_size] (incoming gradient)
	T* __restrict__ inc_grads,            // [batch, steps, dim] (output, accumulated via atomicAdd)
	const int dim,
	const int steps,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride
) {
	static_assert(DEGREE >= 1 && DEGREE <= 12, "DEGREE must be 1-12");

	const uint64_t word_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t batch_idx = blockIdx.y;
	const bool active = word_idx < level_size;

	extern __shared__ char smem[];
	T* shared_inc = reinterpret_cast<T*>(smem);           // [BWD_CHUNK * dim]
	const unsigned num_warps = blockDim.x >> 5;
	T* shared_letter_grads = shared_inc + BWD_CHUNK * dim; // [dim * num_warps]

	// Zero the reduction workspace
	for (int i = threadIdx.x; i < dim * (int)num_warps; i += blockDim.x)
		shared_letter_grads[i] = T(0);

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

	// Prefix signature values: load from precomputed forward signature
	T pref[DEGREE + 1];
	pref[0] = T(1);
	if (active) {
		// Load prefix signature values from the forward signature array
		// pref[k] = S(prefix word of length k)
		uint64_t off = 1; // skip level 0 (scalar 1) in pysiglib's sig layout
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

	// Suffix signature values: start at identity (no suffix at end of path)
	T suf[DEGREE + 1];
	suf[0] = T(1);
	for (int i = 1; i <= DEGREE; ++i) suf[i] = T(0);

	const T* batch_path = path + batch_idx * path_stride;
	T* batch_inc_grad = inc_grads + batch_idx * static_cast<uint64_t>(steps) * dim;

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned lane = threadIdx.x & 31;

	for (int chunk_end = steps; chunk_end > 0; chunk_end -= BWD_CHUNK) {
		const int chunk_start = (chunk_end - BWD_CHUNK > 0) ? chunk_end - BWD_CHUNK : 0;
		const int chunk_len = chunk_end - chunk_start;

		// Load all increments for this chunk into shared memory
		__syncthreads();
		{
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
			const T* inc = shared_inc + t_local * dim;

			// Backward prefix update
			if (active) {
				for (int sd = DEGREE; sd > 0; --sd) {
					T h = T(0);
					for (int k = 0; k < sd; ++k) {
						h = (-inc[letters[k]] * d_recip<T>(sd - k)) * (pref[k] + h);
					}
					pref[sd] += h;
				}
			}

			// Compute per-letter gradients
			T letter_grads_local[DEGREE];
			for (int i = 0; i < DEGREE; ++i) letter_grads_local[i] = T(0);

			if (active) {
				for (int pref_len = 0; pref_len < DEGREE; ++pref_len) {
					const T pref_val = pref[pref_len];
					T prev_prod = T(1);

					{
						T temp_prod = prev_prod;
						T temp_grad = temp_prod * suf[DEGREE - pref_len - 1];
						int denom = 2;
						for (int pos = pref_len + 1; pos < DEGREE; ++pos, ++denom) {
							temp_prod *= inc[letters[pos]] * d_recip<T>(denom);
							temp_grad += temp_prod * suf[DEGREE - pos - 1];
						}
						letter_grads_local[pref_len] += temp_grad * pref_val;
					}

					int denom_lp = 2;
					for (int lp = pref_len + 1; lp < DEGREE; ++lp, ++denom_lp) {
						prev_prod *= inc[letters[lp - 1]] * d_recip<T>(denom_lp);
						T temp_prod = prev_prod;
						T temp_grad = temp_prod * suf[DEGREE - lp - 1];
						int denom = denom_lp + 1;
						for (int pos = lp + 1; pos < DEGREE; ++pos, ++denom) {
							temp_prod *= inc[letters[pos]] * d_recip<T>(denom);
							temp_grad += temp_prod * suf[DEGREE - pos - 1];
						}
						letter_grads_local[lp] += temp_grad * pref_val;
					}
				}
			}

			// Per-dimension warp reduction + global atomicAdd
			for (int letter = 0; letter < dim; ++letter) {
				T val = T(0);
				if (active) {
					for (int lp = 0; lp < DEGREE; ++lp) {
						if (letters[lp] == letter) val += letter_grads_local[lp];
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

			__syncthreads();

			for (int letter = threadIdx.x; letter < dim; letter += blockDim.x) {
				T sum = T(0);
				for (unsigned w = 0; w < num_warps; ++w)
					sum += shared_letter_grads[letter * num_warps + w];
				if (sum != T(0))
					myAtomicAdd(&batch_inc_grad[static_cast<uint64_t>(t) * dim + letter], sum);
			}

			// Forward suffix update
			if (active) {
				for (int m = DEGREE - 1; m > 0; --m) {
					const int base_pos = DEGREE - m;
					T h = T(0);
					for (int p = m; p >= 1; --p) {
						int lp = base_pos + (p - 1);
						h = (inc[letters[lp]] * d_recip<T>(p)) * (suf[m - p] + h);
					}
					suf[m] += h;
				}
			}

			__syncthreads();
		}
	}
}

// Convert increment gradients to path gradients
template<typename T>
__global__ void increment_to_path_grad_ker(
	const T* __restrict__ inc_grad,   // [batch, steps, dim]
	T* __restrict__ path_grad,        // [batch, length, dim]
	int batch_size, int length, int dim
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
	const int inc_stride = steps * dim;
	const int base = b * inc_stride + j;

	if (t == 0) {
		path_grad[idx] = -inc_grad[base];
	} else if (t == length - 1) {
		path_grad[idx] = inc_grad[base + (steps - 1) * dim];
	} else {
		path_grad[idx] = inc_grad[base + (t - 1) * dim] - inc_grad[base + t * dim];
	}
}

template<typename T>
__global__ void set_sig_level0(T* out, uint64_t sig_size, uint64_t batch_size) {
	uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (b < batch_size) out[b * sig_size] = static_cast<T>(1);
}

// Cached stream pool (created once, reused across calls)
static constexpr int MAX_PER_WORD_STREAMS = 12;
static cudaStream_t s_per_word_streams[MAX_PER_WORD_STREAMS] = {};
static bool s_streams_initialized = false;

static void ensure_streams() {
	if (!s_streams_initialized) {
		for (int i = 0; i < MAX_PER_WORD_STREAMS; ++i)
			cudaStreamCreate(&s_per_word_streams[i]);
		s_streams_initialized = true;
	}
}

// Cached workspace for backward kernel's increment gradients (grow-only)
static void* s_inc_grad_buf = nullptr;
static size_t s_inc_grad_buf_size = 0;

static void* ensure_inc_grad_buf(size_t needed) {
	if (needed > s_inc_grad_buf_size) {
		if (s_inc_grad_buf) cudaFree(s_inc_grad_buf);
		cudaMalloc(&s_inc_grad_buf, needed);
		s_inc_grad_buf_size = needed;
	}
	return s_inc_grad_buf;
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

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);

	size_t smem = SIG_CHUNK * dimension * sizeof(T);

	ensure_streams();

	for (uint64_t k = 1; k <= degree; ++k) {
		uint64_t level_size = host_power(dimension, k);
		uint64_t level_offset = li[k];
		unsigned int block = 128;
		if (level_size < 128) block = 32;
		unsigned int grid_x = (unsigned int)((level_size + block - 1) / block);
		dim3 grid(grid_x, (unsigned int)batch_size, 1);
		cudaStream_t stream = (k <= MAX_PER_WORD_STREAMS)
			? s_per_word_streams[k - 1] : nullptr;

		#define LAUNCH_DEGREE(D) \
			case D: signature_per_word_ker<T, D><<<grid, block, smem, stream>>>( \
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
			default:
				signature_per_word_generic_ker<T><<<grid, block, smem, stream>>>(
					path, out, dim, steps, static_cast<int>(k),
					sig_len, level_offset, level_size, path_stride);
				break;
		}
		#undef LAUNCH_DEGREE
	}

	for (uint64_t k = 0; k < degree && k < MAX_PER_WORD_STREAMS; ++k)
		cudaStreamSynchronize(s_per_word_streams[k]);
	cudaDeviceSynchronize();

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

	if (horner) {
		signature_per_word_core_<T>(path, out, batch_size, dimension, length, degree);
		return;
	}

	// Naive (Chen's identity) fallback
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	size_t smem_size = (dimension * sizeof(T) + 7) & ~size_t(7);
	smem_size += (degree + 2) * sizeof(uint64_t);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	const size_t aligned_li_bytes = (level_index_bytes + sizeof(T) - 1) / sizeof(T) * sizeof(T);
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

	if (length <= 1 || degree == 0) {
		cudaMemset(out, 0, batch_size * path_flat_len * sizeof(T));
		return;
	}

	if (degree > MAX_GENERIC_DEGREE)
		throw std::invalid_argument(
			"sig_backprop on CUDA requires degree <= 64. "
			"Use CPU or reduce the truncation level.");

	const int steps = static_cast<int>(length - 1);
	const int dim = static_cast<int>(dimension);

	// Allocate increment gradients buffer (zeroed)
	const size_t inc_grad_bytes = batch_size * steps * dimension * sizeof(T);
	T* d_inc_grads = static_cast<T*>(ensure_inc_grad_buf(inc_grad_bytes));
	cudaMemset(d_inc_grads, 0, inc_grad_bytes);

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);

	// Shared memory: BWD_CHUNK * dim (chunked increments) + dim * num_warps (reduction)
	const unsigned int block = 128;
	const unsigned int num_warps = block / 32;
	size_t smem_size = (BWD_CHUNK * dimension + dimension * num_warps) * sizeof(T);

	ensure_streams();

	// Launch per-word backward kernel for each level
	for (uint64_t k = 1; k <= degree; ++k) {
		uint64_t level_size = host_power(dimension, k);
		uint64_t level_offset = li[k];
		unsigned int grid_x = static_cast<unsigned int>((level_size + block - 1) / block);
		dim3 grid(grid_x, static_cast<unsigned int>(batch_size), 1);
		cudaStream_t stream = (k <= MAX_PER_WORD_STREAMS)
			? s_per_word_streams[k - 1] : nullptr;

		#define LAUNCH_BWD(D) \
			case D: sig_backprop_per_word_ker<T, D><<<grid, block, smem_size, stream>>>( \
				path, sig, sig_derivs, d_inc_grads, dim, steps, sig_len, \
				level_offset, level_size, length * dimension); break;

		switch (k) {
			LAUNCH_BWD(1)  LAUNCH_BWD(2)  LAUNCH_BWD(3)  LAUNCH_BWD(4)
			LAUNCH_BWD(5)  LAUNCH_BWD(6)  LAUNCH_BWD(7)  LAUNCH_BWD(8)
			LAUNCH_BWD(9)  LAUNCH_BWD(10) LAUNCH_BWD(11) LAUNCH_BWD(12)
			default:
				sig_backprop_per_word_generic_ker<T><<<grid, block, smem_size, stream>>>(
					path, sig, sig_derivs, d_inc_grads, dim, steps, static_cast<int>(k),
					sig_len, level_offset, level_size, length * dimension);
				break;
		}
		#undef LAUNCH_BWD
	}

	for (uint64_t k = 0; k < degree && k < MAX_PER_WORD_STREAMS; ++k)
		cudaStreamSynchronize(s_per_word_streams[k]);
	cudaDeviceSynchronize();

	// Convert increment gradients to path gradients
	{
		const int total = static_cast<int>(batch_size * length * dimension);
		const int block_cvt = 256;
		const int grid_cvt = (total + block_cvt - 1) / block_cvt;
		increment_to_path_grad_ker<T><<<grid_cvt, block_cvt>>>(
			d_inc_grads, out, static_cast<int>(batch_size),
			static_cast<int>(length), dim);
	}

	cudaDeviceSynchronize();
	check_cuda_error();
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
