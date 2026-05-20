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
#include "cu_sig_combine.h"
#include "cu_atomic.h"
#include "cu_path_transforms.h"

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
	T* __restrict__ linear_sig_workspace,  // [batch_size * sig_stride]
	bool scalar_term
) {
	const uint64_t batch_idx = blockIdx.x;
	const int thread_id = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * sig_stride;
	T* my_linear_sig = linear_sig_workspace + batch_idx * sig_stride;

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

template<typename T, int DEGREE>
__global__ void signature_per_word_ker(
	const T* __restrict__ path,       // [batch, length, dim]
	T* __restrict__ out,
	const int dim,
	const int steps,
	const uint64_t sig_size,
	const uint64_t level_offset,
	const uint64_t level_size,
	const uint64_t path_stride,       // length * dim
	const bool scalar_term
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
	for (int i = 0; i <= DEGREE; ++i) pref[i] = T(0);
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
	const uint64_t path_stride,
	const bool /*scalar_term*/  // only relevant for the k==1 scalar write in the templated kernel
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
	for (int i = 0; i <= degree; ++i) pref[i] = T(0);
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
	const uint64_t path_stride,
	const bool scalar_term
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
	const uint64_t path_stride,
	const bool scalar_term
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

static void* ensure_inc_grad_buf(size_t needed) {
	if (needed > s_inc_grad_buf_size) {
		if (s_inc_grad_buf) { cudaFree(s_inc_grad_buf); s_inc_grad_buf = nullptr; s_inc_grad_buf_size = 0; }
		CUDA_CHECK(cudaMalloc(&s_inc_grad_buf, needed));
		s_inc_grad_buf_size = needed;
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

	// Dynamic shared memory: only allocate what's needed for the actual path length
	const int actual_chunk = (steps < SIG_CHUNK) ? steps : SIG_CHUNK;
	size_t smem = actual_chunk * dimension * sizeof(T);

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
		dim3 grid(grid_x, (unsigned int)batch_size, 1);
		cudaStream_t stream = (use_streams && k <= MAX_PER_WORD_STREAMS)
			? s_per_word_streams[k - 1] : nullptr;

		#define LAUNCH_DEGREE(D) \
			case D: signature_per_word_ker<T, D><<<grid, block, smem, stream>>>( \
				path, out, dim, steps, sig_stride, level_offset, level_size, path_stride, scalar_term); break;

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
					sig_stride, level_offset, level_size, path_stride, scalar_term);
				break;
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

inline void validate_signature_primitives_args_cuda_(
	const void* primitives,
	uint64_t primitives_len,
	uint64_t dimension,
	uint64_t degree,
	bool lead_lag
) {
	if (primitives == nullptr && primitives_len != 0)
		throw std::invalid_argument("primitives pointer is null but primitives_len is nonzero");
	if (lead_lag && primitives_len != 0)
		throw std::invalid_argument("primitives cannot be used with lead_lag=true");
	if (primitives_len == 0)
		return;
	if (degree < 2)
		throw std::invalid_argument("primitives must be empty when degree < 2");

	uint64_t offset = 0;
	uint64_t level_size = dimension;
	for (uint64_t level = 2; level <= degree; ++level) {
		level_size *= dimension;
		offset += level_size;
		if (offset == primitives_len)
			return;
		if (offset > primitives_len)
			break;
	}
	throw std::invalid_argument("primitives length must be a prefix of tensor levels 2..degree");
}

template<typename T>
__device__ void build_primitive_block_(
	const T* path,
	uint64_t step,
	T* primitive,
	const T* primitives,
	uint64_t primitives_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;
	const uint64_t sig_len = level_index[degree + 1];

	for (uint64_t i = tid; i < sig_len; i += nthreads)
		primitive[i] = T(0);
	__syncthreads();

	const T* prev = path + step * dimension;
	const T* next = prev + dimension;
	for (uint64_t d = tid; d < dimension; d += nthreads)
		primitive[level_index[1] + d] = next[d] - prev[d];

	uint64_t offset = 0;
	uint64_t level_size = data_dimension;
	for (uint64_t level = 2; level <= degree; ++level) {
		level_size *= data_dimension;
		if (offset + level_size > primitives_len)
			break;

		for (uint64_t word_idx = tid; word_idx < level_size; word_idx += nthreads) {
			const T value = primitives[offset + word_idx];
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
			primitive[level_index[level] + aug_word_idx] = value;
		}
		offset += level_size;
	}
	__syncthreads();
}

template<typename T>
__device__ void tensor_exp_block_(
	const T* primitive,
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
		power_prev[i] = primitive[i];
	}
	if (tid == 0)
		out[0] = T(1);
	__syncthreads();

	for (uint64_t i = level_index[1] + tid; i < sig_len; i += nthreads)
		out[i] += primitive[i];
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
					sum += primitive[level_index[left_level] + left_idx]
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
__global__ void signature_primitives_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	const T* __restrict__ primitives,
	uint64_t primitives_len,
	const uint64_t* __restrict__ level_index,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t full_sig_len,
	uint64_t sig_stride,
	uint64_t path_flat_len,
	T* __restrict__ workspace,
	bool scalar_term
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * sig_stride;
	T* acc = workspace + batch_idx * 5 * full_sig_len;
	T* local = acc + full_sig_len;
	T* primitive = local + full_sig_len;
	T* power_prev = primitive + full_sig_len;
	T* power_curr = power_prev + full_sig_len;

	build_primitive_block_(
		my_path, 0, primitive, primitives, primitives_len,
		data_dimension, dimension, degree, level_index);
	tensor_exp_block_(primitive, acc, power_prev, power_curr, dimension, degree, level_index);

	for (uint64_t step = 1; step + 1 < length; ++step) {
		build_primitive_block_(
			my_path, step, primitive, primitives, primitives_len,
			data_dimension, dimension, degree, level_index);
		tensor_exp_block_(primitive, local, power_prev, power_curr, dimension, degree, level_index);
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
	const T* primitives,
	uint64_t primitives_len
) {
	const uint64_t full_sig_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
	const uint64_t path_flat_len = dimension * length;

	// Handle trivial cases
	if (length <= 1) {
		// sig = (1, 0, 0, ..., 0) for each batch element. In scalar_term=false layout
		// the output is (0, 0, ...). Either way, zero the buffer then optionally write
		// the leading 1.
		cudaMemset(out, 0, batch_size * sig_stride * sizeof(T));
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

	if (primitives_len != 0) {
		auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
		host_populate_level_index(level_index_host.get(), dimension, degree + 2);
		const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
		CudaBuf<uint64_t> d_level_index(level_index_bytes);
		CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));
		CudaBuf<T> d_workspace(batch_size * 5 * full_sig_len * sizeof(T));
		const unsigned int threads_per_block = host_choose_threads_per_block(full_sig_len);
		signature_primitives_ker<T><<<static_cast<unsigned int>(batch_size), threads_per_block>>>(
			path, out, primitives, primitives_len, d_level_index.get(),
			data_dimension, dimension, length, degree, full_sig_len, sig_stride,
			path_flat_len, d_workspace.get(), scalar_term);
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

	size_t smem_size = (dimension * sizeof(T) + 7) & ~size_t(7);
	smem_size += (degree + 2) * sizeof(uint64_t);

	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	const size_t aligned_li_bytes = (level_index_bytes + sizeof(T) - 1) / sizeof(T) * sizeof(T);
	const size_t workspace_bytes = batch_size * sig_stride * sizeof(T);

	CudaBuf<char> d_alloc(aligned_li_bytes + workspace_bytes);
	uint64_t* d_level_index = reinterpret_cast<uint64_t*>(d_alloc.get());
	T* d_linear_sig = reinterpret_cast<T*>(d_alloc.get() + aligned_li_bytes);
	CUDA_CHECK(cudaMemcpy(d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));

	signature_naive_ker<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		path, out, d_level_index,
		dimension, length, degree, sig_stride, path_flat_len,
		d_linear_sig, scalar_term
	);

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
	const T* primitives = nullptr,
	uint64_t primitives_len = 0
) {
	if (dimension == 0) throw std::invalid_argument("signature_cuda received path of dimension 0");
	validate_signature_primitives_args_cuda_(primitives, primitives_len, dimension, degree, lead_lag);

	// Compute transformed dimensions
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		// Transform path on GPU
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		CudaBuf<T> d_transformed(t_path_size * sizeof(T));

		cu_transform_path_<T>(path, d_transformed.get(), batch_size, dimension, length, time_aug, lead_lag, end_time);
		cudaDeviceSynchronize();

		signature_cuda_core_<T>(
			d_transformed.get(), out, batch_size, t_dimension, t_length, degree,
			horner, scalar_term, dimension, primitives, primitives_len);
	}
	else {
		signature_cuda_core_<T>(
			path, out, batch_size, dimension, length, degree,
			horner, scalar_term, dimension, primitives, primitives_len);
	}
}

template<typename T>
__global__ void sig_backprop_primitives_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	const T* __restrict__ sig_derivs,
	const T* __restrict__ sig,
	const T* __restrict__ primitives,
	uint64_t primitives_len,
	const uint64_t* __restrict__ level_index,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t length,
	uint64_t degree,
	uint64_t full_sig_len,
	uint64_t sig_stride,
	uint64_t path_flat_len,
	T* __restrict__ workspace,
	bool scalar_term
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_path = path + batch_idx * path_flat_len;
	T* my_out = out + batch_idx * path_flat_len;
	const T* my_sig = sig + batch_idx * sig_stride;
	const T* my_derivs = sig_derivs + batch_idx * sig_stride;

	const uint64_t arrays_per_batch = degree + 12;
	T* base = workspace + batch_idx * arrays_per_batch * full_sig_len;
	T* sig_work = base;
	T* derivs_work = sig_work + full_sig_len;
	T* local_derivs = derivs_work + full_sig_len;
	T* primitive_derivs = local_derivs + full_sig_len;
	T* primitive = primitive_derivs + full_sig_len;
	T* inverse_primitive = primitive + full_sig_len;
	T* local_sig = inverse_primitive + full_sig_len;
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
		build_primitive_block_(
			my_path, static_cast<uint64_t>(seg), primitive, primitives, primitives_len,
			data_dimension, dimension, degree, level_index);
		tensor_exp_block_(primitive, local_sig, power_prev, power_curr, dimension, degree, level_index);

		if (tid == 0)
			inverse_primitive[0] = T(0);
		for (uint64_t i = 1 + tid; i < full_sig_len; i += nthreads)
			inverse_primitive[i] = -primitive[i];
		__syncthreads();

		tensor_exp_block_(inverse_primitive, inverse_local_sig, power_prev, power_curr, dimension, degree, level_index);
		sig_combine_block_(sig_work, inverse_local_sig, degree, level_index);

		uncombine_sig_deriv_block_(
			sig_work, local_sig, derivs_work, local_derivs,
			degree, level_index, full_sig_len);
		tensor_exp_backprop_block_(
			primitive_derivs, local_derivs, primitive, p_all, d_p, d_p_next,
			dimension, degree, level_index, full_sig_len);

		for (uint64_t d = tid; d < dimension; d += nthreads) {
			const T value = primitive_derivs[level_index[1] + d];
			my_out[(static_cast<uint64_t>(seg) + 1) * dimension + d] += value;
			my_out[static_cast<uint64_t>(seg) * dimension + d] -= value;
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
	uint64_t degree,
	bool scalar_term,
	uint64_t data_dimension,
	const T* primitives,
	uint64_t primitives_len
) {
	const uint64_t full_sig_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_sig_len : full_sig_len - 1;
	const uint64_t path_flat_len = dimension * length;

	if (length <= 1 || degree == 0) {
		cudaMemset(out, 0, batch_size * path_flat_len * sizeof(T));
		return;
	}

	if (primitives_len != 0) {
		auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
		host_populate_level_index(level_index_host.get(), dimension, degree + 2);
		const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
		CudaBuf<uint64_t> d_level_index(level_index_bytes);
		CUDA_CHECK(cudaMemcpy(d_level_index.get(), level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));
		CudaBuf<T> d_workspace(batch_size * (degree + 12) * full_sig_len * sizeof(T));
		const unsigned int threads_per_block = host_choose_threads_per_block(full_sig_len);
		sig_backprop_primitives_ker<T><<<static_cast<unsigned int>(batch_size), threads_per_block>>>(
			path, out, sig_derivs, sig, primitives, primitives_len, d_level_index.get(),
			data_dimension, dimension, length, degree, full_sig_len, sig_stride,
			path_flat_len, d_workspace.get(), scalar_term);
		check_cuda_kernel_launch();
		cudaDeviceSynchronize();
		check_cuda_error();
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
	std::lock_guard<std::mutex> inc_lock(s_inc_grad_buf_mu);
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
		uint64_t level_offset = scalar_term ? li[k] : (li[k] - 1);
		unsigned int grid_x = static_cast<unsigned int>((level_size + block - 1) / block);
		dim3 grid(grid_x, static_cast<unsigned int>(batch_size), 1);
		cudaStream_t stream = (k <= MAX_PER_WORD_STREAMS)
			? s_per_word_streams[k - 1] : nullptr;

		#define LAUNCH_BWD(D) \
			case D: sig_backprop_per_word_ker<T, D><<<grid, block, smem_size, stream>>>( \
				path, sig, sig_derivs, d_inc_grads, dim, steps, sig_stride, \
				level_offset, level_size, length * dimension, scalar_term); break;

		switch (k) {
			LAUNCH_BWD(1)  LAUNCH_BWD(2)  LAUNCH_BWD(3)  LAUNCH_BWD(4)
			LAUNCH_BWD(5)  LAUNCH_BWD(6)  LAUNCH_BWD(7)  LAUNCH_BWD(8)
			LAUNCH_BWD(9)  LAUNCH_BWD(10) LAUNCH_BWD(11) LAUNCH_BWD(12)
			default:
				sig_backprop_per_word_generic_ker<T><<<grid, block, smem_size, stream>>>(
					path, sig, sig_derivs, d_inc_grads, dim, steps, static_cast<int>(k),
					sig_stride, level_offset, level_size, length * dimension, scalar_term);
				break;
		}
		#undef LAUNCH_BWD
	}

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
	const T* primitives = nullptr,
	uint64_t primitives_len = 0
) {
	if (dimension == 0) throw std::invalid_argument("sig_backprop_cuda received path of dimension 0");
	validate_signature_primitives_args_cuda_(primitives, primitives_len, dimension, degree, lead_lag);

	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		CudaBuf<T> d_transformed(t_path_size * sizeof(T));

		cu_transform_path_<T>(path, d_transformed.get(), batch_size, dimension, length, time_aug, lead_lag, end_time);

		CudaBuf<T> d_transformed_derivs(t_path_size * sizeof(T));

		sig_backprop_cuda_core_<T>(
			d_transformed.get(), d_transformed_derivs.get(), sig_derivs, sig,
			batch_size, t_dimension, t_length, degree, scalar_term,
			dimension, primitives, primitives_len);

		d_transformed.reset();

		cu_transform_path_backprop_<T>(d_transformed_derivs.get(), out, batch_size, dimension, length, time_aug, lead_lag, end_time);
	}
	else {
		sig_backprop_cuda_core_<T>(
			path, out, sig_derivs, sig, batch_size, dimension, length, degree,
			scalar_term, dimension, primitives, primitives_len);
	}
}


#include "cu_macros.h"


extern "C" {


	CUSIG_API int signature_cuda_f(
		const float* path, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time,
		bool horner, bool scalar_term,
		const float* primitives, uint64_t primitives_len
	) noexcept {
		CUSIG_SAFE_CALL(signature_cuda_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, scalar_term, primitives, primitives_len));
	}

	CUSIG_API int signature_cuda_d(
		const double* path, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time,
		bool horner, bool scalar_term,
		const double* primitives, uint64_t primitives_len
	) noexcept {
		CUSIG_SAFE_CALL(signature_cuda_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, scalar_term, primitives, primitives_len));
	}

	// =====================================================================
	// backprop
	// =====================================================================


	CUSIG_API int sig_backprop_cuda_f(
		const float* path, float* out,
		const float* sig_derivs, const float* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, float end_time, bool scalar_term,
		const float* primitives, uint64_t primitives_len
	) noexcept {
		CUSIG_SAFE_CALL(sig_backprop_cuda_<float>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, primitives, primitives_len));
	}

	CUSIG_API int sig_backprop_cuda_d(
		const double* path, double* out,
		const double* sig_derivs, const double* sig,
		uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
		bool time_aug, bool lead_lag, double end_time, bool scalar_term,
		const double* primitives, uint64_t primitives_len
	) noexcept {
		CUSIG_SAFE_CALL(sig_backprop_cuda_<double>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, primitives, primitives_len));
	}

}
