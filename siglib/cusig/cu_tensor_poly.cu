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
	uint64_t dimension,
	uint64_t degree
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	// Compute level_index in shared memory (no device malloc needed)
	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);

	if (tid == 0) {
		level_index[0] = 0;
		for (uint64_t i = 1; i < degree + 2; ++i)
			level_index[i] = level_index[i - 1] * dimension + 1;
	}
	__syncthreads();

	const uint64_t sig_len = level_index[degree + 1];
	const T* my_sig1 = sig1 + batch_idx * sig_len;
	const T* my_sig2 = sig2 + batch_idx * sig_len;
	T* my_out = out + batch_idx * sig_len;

	// Level 0: scalar component = 1
	if (tid == 0)
		my_out[0] = static_cast<T>(1);

	// Each level reads only from the original sig1 and sig2,
	// so all levels are independent — no syncs needed between them.
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

	// Choose threads per block based on largest level size
	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	unsigned int threads_per_block = host_choose_threads_per_block(max_level_size);

	// Shared memory: level_index (computed inside kernel, no device malloc needed)
	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_combine_kernel<T><<<static_cast<unsigned int>(batch_size), threads_per_block, smem_size>>>(
		sig1, sig2, out, dimension, degree
	);

	check_cuda_kernel_launch();
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
	const T* __restrict__ d_out,         // [batch_size * sig_len]
	T* __restrict__ sig1_deriv,          // [batch_size * sig_len]
	T* __restrict__ sig2_deriv,          // [batch_size * sig_len]
	const T* __restrict__ sig1,          // [batch_size * sig_len]
	const T* __restrict__ sig2,          // [batch_size * sig_len]
	uint64_t dimension,
	uint64_t degree,
	uint64_t sig_len
) {
	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);
	if (threadIdx.x == 0) {
		level_index[0] = 0;
		for (uint64_t i = 1; i < degree + 2; ++i)
			level_index[i] = level_index[i - 1] * dimension + 1;
	}
	__syncthreads();

	const uint64_t elem = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (elem >= sig_len) return;

	const uint64_t offset = (uint64_t)blockIdx.y * sig_len;
	const T* my_d_out = d_out + offset;
	const T* my_sig2 = sig2 + offset;
	const T* my_sig1 = sig1 + offset;

	if (elem == 0) {
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
	uint64_t degree
) {
	if (degree == 0) {
		cudaMemcpy(sig1_deriv, sig_combined_deriv, batch_size * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaMemcpy(sig2_deriv, sig_combined_deriv, batch_size * sizeof(T), cudaMemcpyDeviceToDevice);
		return;
	}

	uint64_t sig_len = host_sig_length(dimension, degree);

	unsigned int threads_per_block = 256;
	unsigned int grid_x = static_cast<unsigned int>((sig_len + threads_per_block - 1) / threads_per_block);
	dim3 grid(grid_x, static_cast<unsigned int>(batch_size));

	size_t smem_size = (degree + 2) * sizeof(uint64_t);

	sig_combine_backprop_kernel<T><<<grid, threads_per_block, smem_size>>>(
		sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2,
		dimension, degree, sig_len
	);

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
	uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("sig_combine_backprop_cuda received dimension 0");
	sig_combine_backprop_cuda_core_<T>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree);
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
	uint64_t sig_len
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_disp = displacement + batch_idx * dimension;
	T* my_out = out + batch_idx * sig_len;

	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index[i] = d_level_index[i];
	__syncthreads();

	linear_signature_device<T>(my_disp, my_out, dimension, degree, level_index);
}

template<typename T>
void linear_sig_cuda_(
	const T* displacement, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("linear_sig_cuda received dimension 0");
	const uint64_t sig_len = host_sig_length(dimension, degree);

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);
	const size_t li_bytes = (degree + 2) * sizeof(uint64_t);

	uint64_t* d_li = nullptr;
	cudaMalloc(&d_li, li_bytes);
	cudaMemcpy(d_li, li.get(), li_bytes, cudaMemcpyHostToDevice);

	uint64_t max_level_size = li[degree + 1] - li[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem = li_bytes;

	linear_sig_kernel<T><<<static_cast<unsigned int>(batch_size), threads, smem>>>(
		displacement, out, d_li, dimension, degree, sig_len
	);

	cudaFree(d_li);
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
	uint64_t sig_len,
	bool prepend
) {
	const uint64_t batch_idx = blockIdx.x;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;

	const T* my_sig = sig + batch_idx * sig_len;
	const T* my_disp = displacement + batch_idx * dimension;
	T* my_out = out + batch_idx * sig_len;
	T* my_lsig = lsig_buf + batch_idx * sig_len;

	extern __shared__ char smem[];
	uint64_t* level_index = reinterpret_cast<uint64_t*>(smem);
	for (uint64_t i = tid; i < degree + 2; i += nthreads)
		level_index[i] = d_level_index[i];
	__syncthreads();

	linear_signature_device<T>(my_disp, my_lsig, dimension, degree, level_index);

	if (prepend) {
		for (uint64_t i = tid; i < sig_len; i += nthreads)
			my_out[i] = my_lsig[i];
		__syncthreads();
		sig_combine_inplace_device<T>(my_out, my_sig, degree, level_index);
	} else {
		for (uint64_t i = tid; i < sig_len; i += nthreads)
			my_out[i] = my_sig[i];
		__syncthreads();
		sig_combine_inplace_device<T>(my_out, my_lsig, degree, level_index);
	}
}

static void* g_sig_join_lsig_buf = nullptr;
static size_t g_sig_join_lsig_bytes = 0;
static std::mutex g_sig_join_lsig_mu;

template<typename T>
void sig_join_cuda_(
	const T* sig, const T* displacement, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	bool prepend = false
) {
	if (dimension == 0) throw std::invalid_argument("sig_join_cuda received dimension 0");
	const uint64_t sig_len = host_sig_length(dimension, degree);

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);
	const size_t li_bytes = (degree + 2) * sizeof(uint64_t);

	uint64_t* d_li = nullptr;
	cudaMalloc(&d_li, li_bytes);
	cudaMemcpy(d_li, li.get(), li_bytes, cudaMemcpyHostToDevice);

	// Workspace for linear sig
	size_t need = sizeof(T) * batch_size * sig_len;
	std::lock_guard<std::mutex> lock(g_sig_join_lsig_mu);
	if (need > g_sig_join_lsig_bytes) {
		if (g_sig_join_lsig_buf) cudaFree(g_sig_join_lsig_buf);
		cudaMalloc(&g_sig_join_lsig_buf, need);
		g_sig_join_lsig_bytes = need;
	}

	uint64_t max_level_size = li[degree + 1] - li[degree];
	unsigned int threads = host_choose_threads_per_block(max_level_size);
	size_t smem = li_bytes;

	sig_join_kernel<T><<<static_cast<unsigned int>(batch_size), threads, smem>>>(
		sig, displacement, out,
		static_cast<T*>(g_sig_join_lsig_buf),
		d_li, dimension, degree, sig_len, prepend
	);

	cudaFree(d_li);
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
	bool prepend = false
) {
	if (dimension == 0) throw std::invalid_argument("sig_join_backprop_cuda received dimension 0");
	const uint64_t sig_len = host_sig_length(dimension, degree);

	// Recompute linear_sig on device
	size_t lsig_bytes = sizeof(T) * batch_size * sig_len;
	T* d_lsig = nullptr;
	cudaMalloc(&d_lsig, lsig_bytes);
	linear_sig_cuda_<T>(displacement, d_lsig, batch_size, dimension, degree);

	// Backprop through sig_combine: d_out -> d_sig, d_lsig_grad
	T* d_lsig_grad = nullptr;
	cudaMalloc(&d_lsig_grad, lsig_bytes);
	if (prepend) {
		// Forward was lsig ⊗ sig
		sig_combine_backprop_cuda_core_<T>(d_out, d_lsig_grad, d_sig, d_lsig, sig, batch_size, dimension, degree);
	} else {
		// Forward was sig ⊗ lsig
		sig_combine_backprop_cuda_core_<T>(d_out, d_sig, d_lsig_grad, sig, d_lsig, batch_size, dimension, degree);
	}

	// Backprop through linear_sig: d_displacement = d_lsig_grad at level 1
	// For linear_sig, the displacement gradient is just level 1 of d_lsig_grad
	// after applying the chain rule through the tensor product structure.
	// Since linear_sig(dx)[k] = dx^{tensor k} / k!, the gradient is:
	// d_displacement[i] = d_lsig[level1][i] + sum over higher levels
	// This is exactly what linear_sig_deriv_to_increment_deriv computes.
	// For simplicity, do this on CPU (the displacement is small).
	auto h_lsig = std::make_unique<T[]>(sig_len * batch_size);
	auto h_dlsig = std::make_unique<T[]>(sig_len * batch_size);
	cudaMemcpy(h_lsig.get(), d_lsig, lsig_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dlsig.get(), d_lsig_grad, lsig_bytes, cudaMemcpyDeviceToHost);

	auto li = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(li.get(), dimension, degree + 2);

	auto h_ddisp = std::make_unique<T[]>(dimension * batch_size);
	for (uint64_t b = 0; b < batch_size; ++b) {
		T* lsig_b = h_lsig.get() + b * sig_len;
		T* dlsig_b = h_dlsig.get() + b * sig_len;
		// Backprop through linear_sig: dF/d(displacement) from dF/d(linear_sig)
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
	cudaMemcpy(d_displacement, h_ddisp.get(), sizeof(T) * dimension * batch_size, cudaMemcpyHostToDevice);

	cudaFree(d_lsig);
	cudaFree(d_lsig_grad);
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
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<float>(sig1, sig2, out, batch_size, dimension, degree));
	}

	CUSIG_API int sig_combine_cuda_d(
		const double* sig1, const double* sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_cuda_<double>(sig1, sig2, out, batch_size, dimension, degree));
	}


	CUSIG_API int sig_combine_backprop_cuda_f(
		const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv,
		const float* sig1, const float* sig2,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<float>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree));
	}

	CUSIG_API int sig_combine_backprop_cuda_d(
		const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv,
		const double* sig1, const double* sig2,
		uint64_t batch_size, uint64_t dimension, uint64_t degree
	) noexcept {
		CUSIG_SAFE_CALL(sig_combine_backprop_cuda_<double>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree));
	}

	// linear_sig CUDA
	CUSIG_API int linear_sig_cuda_f(const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept {
		CUSIG_SAFE_CALL(linear_sig_cuda_<float>(displacement, out, batch_size, dimension, degree));
	}
	CUSIG_API int linear_sig_cuda_d(const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept {
		CUSIG_SAFE_CALL(linear_sig_cuda_<double>(displacement, out, batch_size, dimension, degree));
	}

	// sig_join CUDA
	CUSIG_API int sig_join_cuda_f(const float* sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend) noexcept {
		CUSIG_SAFE_CALL(sig_join_cuda_<float>(sig, displacement, out, batch_size, dimension, degree, prepend));
	}
	CUSIG_API int sig_join_cuda_d(const double* sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend) noexcept {
		CUSIG_SAFE_CALL(sig_join_cuda_<double>(sig, displacement, out, batch_size, dimension, degree, prepend));
	}

	// sig_join_backprop CUDA
	CUSIG_API int sig_join_backprop_cuda_f(const float* d_out, float* d_sig, float* d_displacement, const float* sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend) noexcept {
		CUSIG_SAFE_CALL(sig_join_backprop_cuda_<float>(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend));
	}
	CUSIG_API int sig_join_backprop_cuda_d(const double* d_out, double* d_sig, double* d_displacement, const double* sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend) noexcept {
		CUSIG_SAFE_CALL(sig_join_backprop_cuda_<double>(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend));
	}
}
