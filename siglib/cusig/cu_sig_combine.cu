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
#include "cu_sig_combine.h"

// =========================================================================
// CUDA sig_combine kernel
//
// Computes out = sig1 (x) sig2  (Chen's identity / tensor product)
// for a batch of signature pairs.  Each block handles one batch element.
// =========================================================================

template<typename T>
__global__ void sig_combine_kernel(
	const T* __restrict__ sig1,          // [batch_size * sig_stride]
	const T* __restrict__ sig2,          // [batch_size * sig_stride]
	T* __restrict__ out,                 // [batch_size * sig_stride]
	uint64_t dimension,
	uint64_t degree,
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// Compute level_index in shared memory (no device malloc needed).
	// For scalar_term=false, decrement entries 1..degree+1 so that offsets
	// address the layout without the leading scalar at index 0.
	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);

	if (tid == 0) {
		level_index[0] = 0;
		for (uint64_t i = 1; i < degree + 2; ++i)
			level_index[i] = level_index[i - 1] * dimension + 1;
		if (!scalar_term) {
			for (uint64_t i = 1; i < degree + 2; ++i)
				level_index[i] -= 1;
		}
	}
	__syncthreads();

	const uint64_t sig_stride = level_index[degree + 1];
	const T* my_sig1 = sig1 + batch_idx * sig_stride;
	const T* my_sig2 = sig2 + batch_idx * sig_stride;
	T* my_out = out + batch_idx * sig_stride;

	// Level 0: scalar component = 1 (only when scalar_term is present)
	if (tid == 0 && scalar_term)
		my_out[0] = static_cast<T>(1);

	// Each level reads only from the original sig1 and sig2,
	// so all levels are independent - no syncs needed between them.
	for (uint64_t target_level = 1; target_level <= degree; ++target_level) {
		const uint64_t target_start = level_index[target_level];
		const uint64_t target_size = level_index[target_level + 1] - target_start;

		for (uint64_t idx = tid; idx < target_size; idx += nthreads) {
			// S1^(k) + S2^(k)
			T val = my_sig1[target_start + idx] + my_sig2[target_start + idx];

			// + sum_{i=1}^{k-1} S1^(i) tensor S2^(k-i)
			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				const uint64_t right_level = target_level - left_level;
				const uint64_t left_start = level_index[left_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t right_size = level_index[right_level + 1] - right_start;

				const uint64_t l_idx = idx / right_size;
				const uint64_t r_idx = idx % right_size;
				val += my_sig1[left_start + l_idx] * my_sig2[right_start + r_idx];
			}

			my_out[target_start + idx] = val;
		}
	}
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
	uint64_t degree,
	bool scalar_term = true
) {
	const uint64_t sig_len = host_sig_length(dimension, degree);

	if (degree == 0) {
		// Signature of degree 0 is just the scalar 1 (length-1 layout when scalar_term=true).
		// When scalar_term=false, degree-0 output has zero-length - nothing to write.
		if (scalar_term) {
			auto ones = std::make_unique<T[]>(batch_size);
			std::fill(ones.get(), ones.get() + batch_size, static_cast<T>(1));
			cudaMemcpy(out, ones.get(), batch_size * sizeof(T), cudaMemcpyHostToDevice);
		}
		return;
	}

	// Choose threads per block based on largest level size
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	// Shared memory: level_index (computed inside kernel, no device malloc needed)
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	configure_dynamic_smem(
		sig_combine_kernel<T>, smem_size, "CUDA sig combine");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		sig_combine_kernel<T><<<
			batch_chunk.grid, threads_per_block, smem_size>>>(
				sig1, sig2, out, dimension, degree, scalar_term,
				batch_chunk.offset, batch_chunk.size
			);
		batch_offset += batch_chunk.size;
	}

	check_cuda_kernel_launch();
}

template<typename T>
void sig_combine_cuda_(
	const T* sig1,
	const T* sig2,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool scalar_term = true
) {
	if (dimension == 0) throw std::invalid_argument("sig_combine_cuda received dimension 0");
	sig_combine_cuda_core_<T>(sig1, sig2, out, batch_size, dimension, degree, scalar_term);
}

// =========================================================================
// CUDA sig_combine_backprop kernel
//
// Given d_out = dF/d(sig_combine(sig1, sig2)), computes:
//   sig1_deriv = dF/d(sig1)
//   sig2_deriv = dF/d(sig2)
//
// 2D grid: gridDim.y = batch_size, gridDim.x covers sig_len.
// Each thread processes exactly one (batch, element) pair.
// =========================================================================

template<typename T>
__global__ void __launch_bounds__(256)
sig_combine_backprop_kernel(
	const T* __restrict__ d_out,         // [batch_size * sig_stride]
	T* __restrict__ sig1_deriv,          // [batch_size * sig_stride]
	T* __restrict__ sig2_deriv,          // [batch_size * sig_stride]
	const T* __restrict__ sig1,          // [batch_size * sig_stride]
	const T* __restrict__ sig2,          // [batch_size * sig_stride]
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len,                    // elements to iterate per batch (sig_stride)
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);
	if (threadIdx.x == 0) {
		level_index[0] = 0;
		for (uint64_t i = 1; i < degree + 2; ++i)
			level_index[i] = level_index[i - 1] * dimension + 1;
		if (!scalar_term) {
			for (uint64_t i = 1; i < degree + 2; ++i)
				level_index[i] -= 1;
		}
	}
	__syncthreads();

	const uint64_t elem = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (elem >= sig_len) return;

	const uint64_t offset = batch_idx * sig_len;
	const T* my_d_out = d_out + offset;
	const T* my_sig2 = sig2 + offset;
	const T* my_sig1 = sig1 + offset;

	// Scalar-term slot (index 0) only exists when scalar_term=true.
	if (scalar_term && elem == 0) {
		sig1_deriv[offset] = static_cast<T>(0);
		sig2_deriv[offset] = static_cast<T>(0);
		return;
	}

	// Find level
	uint64_t k = 1;
	while (elem >= level_index[k + 1]) ++k;

	const T d_out_val = my_d_out[elem];

	// Top level: no inner loop work, just copy
	if (k == degree) {
		sig1_deriv[offset + elem] = d_out_val;
		sig2_deriv[offset + elem] = d_out_val;
		return;
	}

	const uint64_t k_start = level_index[k];
	const uint64_t k_size = level_index[k + 1] - k_start;
	const uint64_t idx = elem - k_start;

	// sig1_deriv
	T val1 = d_out_val;
	for (uint64_t rl = 1; rl <= degree - k; ++rl) {
		const uint64_t comb_start = level_index[k + rl];
		const uint64_t r_start = level_index[rl];
		const uint64_t rs = level_index[rl + 1] - r_start;
		const T* d_out_row = my_d_out + comb_start + idx * rs;
		for (uint64_t r = 0; r < rs; ++r) {
			val1 += d_out_row[r] * my_sig2[r_start + r];
		}
	}
	sig1_deriv[offset + elem] = val1;

	// sig2_deriv
	T val2 = d_out_val;
	for (uint64_t ll = 1; ll <= degree - k; ++ll) {
		const uint64_t comb_start = level_index[ll + k];
		const uint64_t l_start = level_index[ll];
		const uint64_t ls = level_index[ll + 1] - l_start;
		const T* d_out_col = my_d_out + comb_start + idx;
		for (uint64_t l = 0; l < ls; ++l) {
			val2 += d_out_col[l * k_size] * my_sig1[l_start + l];
		}
	}
	sig2_deriv[offset + elem] = val2;
}

// =========================================================================
// Host-side core launch for backprop
// =========================================================================

template<typename T>
void sig_combine_backprop_cuda_core_(
	const T* sig_combined_deriv,
	T* sig1_deriv,
	T* sig2_deriv,
	const T* sig1,
	const T* sig2,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool scalar_term = true
) {
	if (degree == 0) {
		// With scalar_term=true the output holds a single scalar = 1*1; derivs of index 0 are 0.
		// With scalar_term=false there is no element at all.
		if (scalar_term) {
			cudaMemset(sig1_deriv, 0, batch_size * sizeof(T));
			cudaMemset(sig2_deriv, 0, batch_size * sizeof(T));
		}
		return;
	}

	const uint64_t full_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_len : full_len - 1;

	unsigned int threads_per_block = 256;
	unsigned int grid_x = static_cast<unsigned int>((sig_stride + threads_per_block - 1) / threads_per_block);
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	configure_dynamic_smem(
		sig_combine_backprop_kernel<T>, smem_size,
		"CUDA sig combine backprop");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			grid_x, batch_size, batch_offset);
		sig_combine_backprop_kernel<T><<<
			batch_chunk.grid, threads_per_block, smem_size>>>(
				sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2,
				dimension, degree, sig_stride, scalar_term,
				batch_chunk.offset, batch_chunk.size
			);
		batch_offset += batch_chunk.size;
	}

	check_cuda_kernel_launch();
}

template<typename T>
void sig_combine_backprop_cuda_(
	const T* sig_combined_deriv,
	T* sig1_deriv,
	T* sig2_deriv,
	const T* sig1,
	const T* sig2,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool scalar_term = true
) {
	if (dimension == 0) throw std::invalid_argument("sig_combine_backprop_cuda received dimension 0");
	sig_combine_backprop_cuda_core_<T>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, scalar_term);
}

// =========================================================================
// CUDA linear_sig kernel
// =========================================================================

template<typename T>
__global__ void linear_sig_kernel(
	const T* __restrict__ displacement,
	T* __restrict__ out,
	const uint64_t* __restrict__ d_level_index,
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_stride,
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_disp = displacement + batch_idx * dimension;
	T* my_out = out + batch_idx * sig_stride;

	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index[i] = d_level_index[i];
	__syncthreads();

	// When scalar_term is false, shift offsets by -1 so that
	// level 1 starts at index 0 (no leading scalar slot in the output).
	if (!scalar_term) {
		if (tid == 0) {
			for (uint64_t i = 1; i < degree + 2; ++i)
				level_index[i] -= 1;
		}
		__syncthreads();
	}

	linear_signature_device<T>(my_disp, my_out, dimension, degree, level_index, scalar_term);
}

template<typename T>
void linear_sig_cuda_(
	const T* displacement, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	bool scalar_term = true
) {
	if (dimension == 0) throw std::invalid_argument("linear_sig_cuda received dimension 0");
	const uint64_t full_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_len : full_len - 1;

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);
	const size_t li_bytes = (degree + 2) * sizeof(uint64_t);

	CudaBuf<uint64_t> d_li(li_bytes);
	CUDA_CHECK(cudaMemcpy(d_li.get(), li.get(), li_bytes, cudaMemcpyHostToDevice));

	uint64_t max_level_size = li[degree + 1] - li[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem = li_bytes;

	configure_dynamic_smem(
		linear_sig_kernel<T>, smem, "CUDA linear signature");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		linear_sig_kernel<T><<<batch_chunk.grid, threads, smem>>>(
			displacement, out, d_li.get(), dimension, degree, sig_stride,
			scalar_term, batch_chunk.offset, batch_chunk.size
		);
		batch_offset += batch_chunk.size;
	}

	check_cuda_kernel_launch();
}

// =========================================================================
// CUDA sig_join kernel
// =========================================================================

template<typename T>
__global__ void sig_join_kernel(
	const T* __restrict__ sig,
	const T* __restrict__ displacement,
	T* __restrict__ out,
	T* __restrict__ lsig_buf,
	const uint64_t* __restrict__ d_level_index,
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_stride,
	bool prepend,
	bool scalar_term,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_sig = sig + batch_idx * sig_stride;
	const T* my_disp = displacement + batch_idx * dimension;
	T* my_out = out + batch_idx * sig_stride;
	// lsig_buf is allocated with sig_stride per batch element.
	T* my_lsig = lsig_buf + batch_idx * sig_stride;

	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index[i] = d_level_index[i];
	__syncthreads();

	// When scalar_term=false, shift offsets so that level k starts at
	// level_index[k]-1 (level 1 starts at 0).
	if (!scalar_term) {
		if (tid == 0) {
			for (uint64_t i = 1; i < degree + 2; ++i)
				level_index[i] -= 1;
		}
		__syncthreads();
	}

	linear_signature_device<T>(my_disp, my_lsig, dimension, degree, level_index, scalar_term);

	if (prepend) {
		for (uint64_t i = tid; i < sig_stride; i += nthreads)
			my_out[i] = my_lsig[i];
		__syncthreads();
		sig_combine_inplace_device<T>(my_out, my_sig, degree, level_index, scalar_term);
	} else {
		for (uint64_t i = tid; i < sig_stride; i += nthreads)
			my_out[i] = my_sig[i];
		__syncthreads();
		sig_combine_inplace_device<T>(my_out, my_lsig, degree, level_index, scalar_term);
	}
}

static void* g_sig_join_lsig_buf = nullptr;
static size_t g_sig_join_lsig_bytes = 0;
static std::mutex g_sig_join_lsig_mu;

void release_sig_combine_state() {
	std::lock_guard<std::mutex> lock(g_sig_join_lsig_mu);
	if (g_sig_join_lsig_buf) {
		cudaFree(g_sig_join_lsig_buf);
		g_sig_join_lsig_buf = nullptr;
		g_sig_join_lsig_bytes = 0;
	}
}

template<typename T>
void sig_join_cuda_(
	const T* sig, const T* displacement, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	bool prepend = false,
	bool scalar_term = true
) {
	if (dimension == 0) throw std::invalid_argument("sig_join_cuda received dimension 0");
	const uint64_t full_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_len : full_len - 1;

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);
	const size_t li_bytes = (degree + 2) * sizeof(uint64_t);

	CudaBuf<uint64_t> d_li(li_bytes);
	CUDA_CHECK(cudaMemcpy(d_li.get(), li.get(), li_bytes, cudaMemcpyHostToDevice));

	size_t need = sizeof(T) * batch_size * sig_stride;
	std::lock_guard<std::mutex> lock(g_sig_join_lsig_mu);
	if (need > g_sig_join_lsig_bytes) {
		if (g_sig_join_lsig_buf) {
			cudaFree(g_sig_join_lsig_buf);
			g_sig_join_lsig_buf = nullptr;
			g_sig_join_lsig_bytes = 0;
		}
		CUDA_CHECK(cudaMalloc(&g_sig_join_lsig_buf, need));
		g_sig_join_lsig_bytes = need;
	}

	uint64_t max_level_size = li[degree + 1] - li[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem = li_bytes;

	configure_dynamic_smem(
		sig_join_kernel<T>, smem, "CUDA sig join");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		sig_join_kernel<T><<<batch_chunk.grid, threads, smem>>>(
			sig, displacement, out, static_cast<T*>(g_sig_join_lsig_buf),
			d_li.get(), dimension, degree, sig_stride, prepend, scalar_term,
			batch_chunk.offset, batch_chunk.size
		);
		batch_offset += batch_chunk.size;
	}

	check_cuda_kernel_launch();
}

// =========================================================================
// CUDA sig_join_backprop
// =========================================================================

template<typename T>
void sig_join_backprop_cuda_(
	const T* d_out, T* d_sig, T* d_displacement,
	const T* sig, const T* displacement,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	bool prepend = false,
	bool scalar_term = true
) {
	if (dimension == 0) throw std::invalid_argument("sig_join_backprop_cuda received dimension 0");
	const uint64_t full_len = host_sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_len : full_len - 1;

	// Recompute linear_sig on device (same stride/layout as caller's sig).
	size_t lsig_bytes = sizeof(T) * batch_size * sig_stride;
	CudaBuf<T> d_lsig(lsig_bytes);
	linear_sig_cuda_<T>(displacement, d_lsig.get(), batch_size, dimension, degree, scalar_term);

	// Backprop through sig_combine: d_out -> d_sig, d_lsig_grad
	CudaBuf<T> d_lsig_grad(lsig_bytes);
	if (prepend) {
		// Forward was lsig \otimes sig
		sig_combine_backprop_cuda_core_<T>(d_out, d_lsig_grad.get(), d_sig, d_lsig.get(), sig, batch_size, dimension, degree, scalar_term);
	} else {
		// Forward was sig \otimes lsig
		sig_combine_backprop_cuda_core_<T>(d_out, d_sig, d_lsig_grad.get(), sig, d_lsig.get(), batch_size, dimension, degree, scalar_term);
	}

	// Backprop through linear_sig on the host (displacement is small).
	// The host-side loop below uses the scalar_term=true layout; for scalar_term=false
	// we stage into a full-length buffer with a synthetic scalar-0 slot, then strip it.
	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);

	auto h_ddisp = std::make_unique<T[]>(dimension * batch_size);

	if (scalar_term) {
		auto h_lsig = std::make_unique<T[]>(full_len * batch_size);
		auto h_dlsig = std::make_unique<T[]>(full_len * batch_size);
		CUDA_CHECK(cudaMemcpy(h_lsig.get(), d_lsig.get(), lsig_bytes, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_dlsig.get(), d_lsig_grad.get(), lsig_bytes, cudaMemcpyDeviceToHost));

		for (uint64_t b = 0; b < batch_size; ++b) {
			T* lsig_b = h_lsig.get() + b * full_len;
			T* dlsig_b = h_dlsig.get() + b * full_len;
			for (uint64_t level = degree; level > 1; --level) {
				const T one_over_level = static_cast<T>(1.) / static_cast<T>(level);
				const uint64_t level_size = li[level] - li[level - 1];
				for (uint64_t j = 0; j < level_size; ++j) {
					const uint64_t offs1 = li[level] + dimension * j - 1;
					const uint64_t offs2 = li[level - 1] + j;
					for (uint64_t dd = 1; dd <= dimension; ++dd) {
						const T ii = dlsig_b[offs1 + dd] * one_over_level;
						dlsig_b[offs2] += lsig_b[dd] * ii;
						dlsig_b[dd] += lsig_b[offs2] * ii;
					}
				}
			}
			std::memcpy(h_ddisp.get() + b * dimension, dlsig_b + 1, dimension * sizeof(T));
		}
	} else {
		// Stage into full-length buffer with scalar-0 slot prepended.
		auto h_lsig = std::make_unique<T[]>(full_len * batch_size);
		auto h_dlsig = std::make_unique<T[]>(full_len * batch_size);
		auto tmp_lsig = std::make_unique<T[]>(sig_stride);
		auto tmp_dlsig = std::make_unique<T[]>(sig_stride);
		for (uint64_t b = 0; b < batch_size; ++b) {
			CUDA_CHECK(cudaMemcpy(tmp_lsig.get(), d_lsig.get() + b * sig_stride, sig_stride * sizeof(T), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(tmp_dlsig.get(), d_lsig_grad.get() + b * sig_stride, sig_stride * sizeof(T), cudaMemcpyDeviceToHost));
			T* lsig_b = h_lsig.get() + b * full_len;
			T* dlsig_b = h_dlsig.get() + b * full_len;
			lsig_b[0] = static_cast<T>(1);
			dlsig_b[0] = static_cast<T>(0);
			std::memcpy(lsig_b + 1, tmp_lsig.get(), sig_stride * sizeof(T));
			std::memcpy(dlsig_b + 1, tmp_dlsig.get(), sig_stride * sizeof(T));
		}

		for (uint64_t b = 0; b < batch_size; ++b) {
			T* lsig_b = h_lsig.get() + b * full_len;
			T* dlsig_b = h_dlsig.get() + b * full_len;
			for (uint64_t level = degree; level > 1; --level) {
				const T one_over_level = static_cast<T>(1.) / static_cast<T>(level);
				const uint64_t level_size = li[level] - li[level - 1];
				for (uint64_t j = 0; j < level_size; ++j) {
					const uint64_t offs1 = li[level] + dimension * j - 1;
					const uint64_t offs2 = li[level - 1] + j;
					for (uint64_t dd = 1; dd <= dimension; ++dd) {
						const T ii = dlsig_b[offs1 + dd] * one_over_level;
						dlsig_b[offs2] += lsig_b[dd] * ii;
						dlsig_b[dd] += lsig_b[offs2] * ii;
					}
				}
			}
			std::memcpy(h_ddisp.get() + b * dimension, dlsig_b + 1, dimension * sizeof(T));
		}
	}
	CUDA_CHECK(cudaMemcpy(d_displacement, h_ddisp.get(), sizeof(T) * dimension * batch_size, cudaMemcpyHostToDevice));
}

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#include "cu_macros.h"

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {


	CUSIG_API int sig_combine_cuda_f(
		const float* sig1, const float* sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<float>(sig1, sig2, out, batch_size, dimension, degree, scalar_term));
	}

	CUSIG_API int sig_combine_cuda_d(
		const double* sig1, const double* sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<double>(sig1, sig2, out, batch_size, dimension, degree, scalar_term));
	}


	CUSIG_API int sig_combine_backprop_cuda_f(
		const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv,
		const float* sig1, const float* sig2,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<float>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, scalar_term));
	}

	CUSIG_API int sig_combine_backprop_cuda_d(
		const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv,
		const double* sig1, const double* sig2,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<double>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, scalar_term));
	}

	// linear_sig CUDA
	CUSIG_API int linear_sig_cuda_f(const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(linear_sig_cuda_<float>(displacement, out, batch_size, dimension, degree, scalar_term));
	}
	CUSIG_API int linear_sig_cuda_d(const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(linear_sig_cuda_<double>(displacement, out, batch_size, dimension, degree, scalar_term));
	}

	// sig_join CUDA
	CUSIG_API int sig_join_cuda_f(const float* sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(sig_join_cuda_<float>(sig, displacement, out, batch_size, dimension, degree, prepend, scalar_term));
	}
	CUSIG_API int sig_join_cuda_d(const double* sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(sig_join_cuda_<double>(sig, displacement, out, batch_size, dimension, degree, prepend, scalar_term));
	}

	// sig_join_backprop CUDA
	CUSIG_API int sig_join_backprop_cuda_f(const float* d_out, float* d_sig, float* d_displacement, const float* sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(sig_join_backprop_cuda_<float>(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend, scalar_term));
	}
	CUSIG_API int sig_join_backprop_cuda_d(const double* d_out, double* d_sig, double* d_displacement, const double* sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(sig_join_backprop_cuda_<double>(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend, scalar_term));
	}
}
