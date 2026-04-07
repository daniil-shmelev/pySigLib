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
#include "cu_macros.h"
#include "cu_atomic.h"

// We load the branched-sig cache from cpsig.dll at runtime via LoadLibrary /
// GetProcAddress.  This avoids any C++17 vs C++20 ABI issues between nvcc
// (C++17) and the cpsig MSVC build (C++20).

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// =========================================================================
// cpsig dynamic-loading helpers
// =========================================================================

typedef int (*fn_prepare_branched_sig)(uint64_t, uint64_t);
typedef int (*fn_get_branched_cache_sizes)(
	uint64_t, uint64_t,
	uint64_t*, uint64_t*, int*,
	uint64_t*, uint64_t*, uint64_t*, uint64_t*, uint64_t*);
typedef int (*fn_get_branched_cache_data)(
	uint64_t, uint64_t,
	double*, uint8_t*, uint64_t*, uint64_t*, uint64_t*, uint64_t*);

struct CpSigFuncs {
	fn_prepare_branched_sig prepare = nullptr;
	fn_get_branched_cache_sizes get_sizes = nullptr;
	fn_get_branched_cache_data get_data = nullptr;
};

static CpSigFuncs load_cpsig_funcs() {
	CpSigFuncs f;
#ifdef _WIN32
	// cpsig.dll is already loaded in the process (Python loaded it first)
	HMODULE h = GetModuleHandleA("cpsig.dll");
	if (!h)
		h = LoadLibraryA("cpsig.dll");
	if (!h)
		throw std::runtime_error("cu_branched_signature: cannot find cpsig.dll");
	f.prepare   = (fn_prepare_branched_sig)GetProcAddress(h, "prepare_branched_sig");
	f.get_sizes = (fn_get_branched_cache_sizes)GetProcAddress(h, "get_branched_cache_sizes");
	f.get_data  = (fn_get_branched_cache_data)GetProcAddress(h, "get_branched_cache_data");
#else
	void* h = dlopen("libcpsig.so", RTLD_NOW | RTLD_NOLOAD);
	if (!h) h = dlopen("libcpsig.so", RTLD_NOW);
	if (!h) throw std::runtime_error("cu_branched_signature: cannot find libcpsig.so");
	f.prepare   = (fn_prepare_branched_sig)dlsym(h, "prepare_branched_sig");
	f.get_sizes = (fn_get_branched_cache_sizes)dlsym(h, "get_branched_cache_sizes");
	f.get_data  = (fn_get_branched_cache_data)dlsym(h, "get_branched_cache_data");
#endif
	if (!f.prepare || !f.get_sizes || !f.get_data)
		throw std::runtime_error("cu_branched_signature: missing cpsig export functions");
	return f;
}

static CpSigFuncs& cpsig() {
	static CpSigFuncs f = load_cpsig_funcs();
	return f;
}

// =========================================================================
// GPU cache: mirrors BranchedSigCache on device memory
// =========================================================================

struct BranchedSigCacheGPU {
	// 32-bit GPU copies for fast index arithmetic
	uint32_t* d_coprod_data32 = nullptr;
	uint32_t* d_coprod_offsets32 = nullptr;
	uint32_t* d_labels_offsets32 = nullptr;
	uint32_t* d_order_index32 = nullptr;

	double* d_inv_factorial_f64 = nullptr;
	float* d_inv_factorial_f32 = nullptr;
	uint8_t* d_labels_data = nullptr;

	uint32_t total_length = 0;
	uint32_t num_trees = 0;
	uint32_t coprod_data_len = 0;
	int max_nodes = 0;

	BranchedSigCacheGPU() = default;
	BranchedSigCacheGPU(const BranchedSigCacheGPU&) = delete;
	BranchedSigCacheGPU& operator=(const BranchedSigCacheGPU&) = delete;

	~BranchedSigCacheGPU() {
		if (d_coprod_data32) cudaFree(d_coprod_data32);
		if (d_coprod_offsets32) cudaFree(d_coprod_offsets32);
		if (d_labels_offsets32) cudaFree(d_labels_offsets32);
		if (d_order_index32) cudaFree(d_order_index32);
		if (d_inv_factorial_f64) cudaFree(d_inv_factorial_f64);
		if (d_inv_factorial_f32) cudaFree(d_inv_factorial_f32);
		if (d_labels_data) cudaFree(d_labels_data);
	}
};

// Reuse the hash from cu_log_sig_cache.h
#include "cu_log_sig_cache.h"

static std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<BranchedSigCacheGPU>,
	CuPairHash
> s_gpu_cache_map;

template<typename T>
static void upload(T*& d_ptr, const T* h_data, size_t count) {
	cudaMalloc(&d_ptr, count * sizeof(T));
	cudaMemcpy(d_ptr, h_data, count * sizeof(T), cudaMemcpyHostToDevice);
}

static const BranchedSigCacheGPU& get_or_upload_gpu_cache(uint64_t dimension, uint64_t max_nodes) {
	auto key = std::make_pair(dimension, max_nodes);
	auto it = s_gpu_cache_map.find(key);
	if (it != s_gpu_cache_map.end())
		return *(it->second);

	auto& fn = cpsig();

	// Ensure the CPU cache is prepared
	if (fn.prepare(dimension, max_nodes) != 0)
		throw std::runtime_error("cu_branched_signature: prepare_branched_sig failed");

	// Query sizes
	uint64_t total_length, num_trees;
	int out_max_nodes;
	uint64_t order_index_len, labels_data_len, labels_offsets_len;
	uint64_t coprod_data_len, coprod_offsets_len;
	if (fn.get_sizes(dimension, max_nodes,
			&total_length, &num_trees, &out_max_nodes,
			&order_index_len, &labels_data_len, &labels_offsets_len,
			&coprod_data_len, &coprod_offsets_len) != 0)
		throw std::runtime_error("cu_branched_signature: get_branched_cache_sizes failed");

	// Allocate host buffers and fetch data
	std::vector<double> h_inv_factorial(num_trees);
	std::vector<uint8_t> h_labels_data(labels_data_len);
	std::vector<uint64_t> h_labels_offsets(labels_offsets_len);
	std::vector<uint64_t> h_coprod_data(coprod_data_len);
	std::vector<uint64_t> h_coprod_offsets(coprod_offsets_len);
	std::vector<uint64_t> h_order_index(order_index_len);

	if (fn.get_data(dimension, max_nodes,
			h_inv_factorial.data(), h_labels_data.data(), h_labels_offsets.data(),
			h_coprod_data.data(), h_coprod_offsets.data(), h_order_index.data()) != 0)
		throw std::runtime_error("cu_branched_signature: get_branched_cache_data failed");

	// Convert to 32-bit for GPU
	std::vector<uint32_t> coprod_data32(coprod_data_len);
	std::vector<uint32_t> coprod_offsets32(coprod_offsets_len);
	std::vector<uint32_t> labels_offsets32(labels_offsets_len);
	std::vector<uint32_t> order_index32(order_index_len);
	for (size_t i = 0; i < coprod_data_len; ++i) coprod_data32[i] = static_cast<uint32_t>(h_coprod_data[i]);
	for (size_t i = 0; i < coprod_offsets_len; ++i) coprod_offsets32[i] = static_cast<uint32_t>(h_coprod_offsets[i]);
	for (size_t i = 0; i < labels_offsets_len; ++i) labels_offsets32[i] = static_cast<uint32_t>(h_labels_offsets[i]);
	for (size_t i = 0; i < order_index_len; ++i) order_index32[i] = static_cast<uint32_t>(h_order_index[i]);

	std::vector<float> inv_factorial_f32(num_trees);
	for (size_t i = 0; i < num_trees; ++i) inv_factorial_f32[i] = static_cast<float>(h_inv_factorial[i]);

	// Upload to GPU
	auto gpu = std::make_unique<BranchedSigCacheGPU>();
	gpu->total_length = static_cast<uint32_t>(total_length);
	gpu->num_trees = static_cast<uint32_t>(num_trees);
	gpu->coprod_data_len = static_cast<uint32_t>(coprod_data_len);
	gpu->max_nodes = out_max_nodes;

	upload(gpu->d_coprod_data32, coprod_data32.data(), coprod_data32.size());
	upload(gpu->d_coprod_offsets32, coprod_offsets32.data(), coprod_offsets32.size());
	upload(gpu->d_labels_offsets32, labels_offsets32.data(), labels_offsets32.size());
	upload(gpu->d_order_index32, order_index32.data(), order_index32.size());
	upload(gpu->d_inv_factorial_f64, h_inv_factorial.data(), h_inv_factorial.size());
	upload(gpu->d_inv_factorial_f32, inv_factorial_f32.data(), inv_factorial_f32.size());
	upload(gpu->d_labels_data, h_labels_data.data(), h_labels_data.size());

	auto [ins, _] = s_gpu_cache_map.insert_or_assign(key, std::move(gpu));
	return *(ins->second);
}

// =========================================================================
// Fused segment kernel
// =========================================================================

// Shared memory layout:
// | temp[total_len] | inc[dim] | s_coprod_data[coprod_len] | s_coprod_off[num_trees+1] | s_order_idx[max_nodes+2] |

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	int dim,
	int steps,
	uint32_t total_len,
	uint64_t path_stride,
	const uint8_t* __restrict__ labels_data,
	const uint32_t* __restrict__ labels_offsets,
	const T* __restrict__ inv_factorial,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	const uint32_t* __restrict__ g_order_index,
	int max_nodes,
	uint32_t coprod_data_len
) {
	const uint32_t batch_idx = blockIdx.y;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	T* temp = reinterpret_cast<T*>(smem);
	T* inc = temp + total_len;
	uint32_t* s_coprod_data = reinterpret_cast<uint32_t*>(inc + dim);
	uint32_t* s_coprod_off = s_coprod_data + coprod_data_len;
	uint32_t* s_order_idx = s_coprod_off + num_trees + 1;

	// --- One-time cooperative load of coproduct table into shared memory ---
	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		s_coprod_data[i] = g_coprod_data[i];
	for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
		s_coprod_off[i] = g_coprod_offsets[i];
	for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2); i += blockDim.x)
		s_order_idx[i] = g_order_index[i];
	__syncthreads();

	const T* bp = path + static_cast<uint64_t>(batch_idx) * path_stride;
	T* X = out + static_cast<uint64_t>(batch_idx) * total_len;

	for (int seg = 0; seg < steps; ++seg) {
		// --- Cooperative increment load ---
		for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
			inc[d] = bp[(seg + 1) * dim + d] - bp[seg * dim + d];
		__syncthreads();

		// --- Linear branched sig ---
		T* tgt = (seg == 0) ? X : temp;
		if (tid == 0) tgt[0] = T(1);
		if (tid < num_trees) {
			T prod = T(1);
			uint32_t lstart = labels_offsets[tid];
			uint32_t lend = labels_offsets[tid + 1];
			#pragma unroll 8
			for (uint32_t j = lstart; j < lend; ++j)
				prod *= inc[labels_data[j]];
			tgt[tid + 1] = prod * inv_factorial[tid];
		}
		__syncthreads();

		// --- Butcher product (seg > 0 only) ---
		if (seg > 0) {
			for (int order = max_nodes; order >= 1; --order) {
				uint32_t ostart = s_order_idx[order];
				uint32_t oend = s_order_idx[order + 1];
				if (tid >= ostart && tid < oend) {
					uint32_t fi = tid + 1;
					T val = X[fi] + temp[fi];

					uint32_t pos = s_coprod_off[tid];
					uint32_t pend = s_coprod_off[tid + 1];
					while (pos < pend) {
						uint32_t nf = s_coprod_data[pos++];
						T term = temp[s_coprod_data[pos++]];
						#pragma unroll 4
						for (uint32_t j = 0; j < nf; ++j)
							term *= X[s_coprod_data[pos++]];
						val += term;
					}
					X[fi] = val;
				}
				__syncthreads();
			}
		}
	}
}

// =========================================================================
// Backprop kernel
// =========================================================================

// Shared memory layout:
// | s_bsig[total_len] | s_derivs[total_len] | temp_Y[total_len] | local_derivs[total_len] |
// | inc[dim] | inc_derivs[dim] | coprod tables (same as forward) |

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_backprop_ker(
	const T* __restrict__ path,
	T* __restrict__ path_derivs,
	const T* __restrict__ bsig_in,
	const T* __restrict__ bsig_derivs_in,
	int dim,
	int steps,
	uint32_t total_len,
	uint64_t path_stride,
	const uint8_t* __restrict__ labels_data,
	const uint32_t* __restrict__ labels_offsets,
	const T* __restrict__ inv_factorial,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	const uint32_t* __restrict__ g_order_index,
	int max_nodes,
	uint32_t coprod_data_len
) {
	const uint32_t batch_idx = blockIdx.y;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	T* s_bsig = reinterpret_cast<T*>(smem);
	T* s_derivs = s_bsig + total_len;
	T* temp_Y = s_derivs + total_len;
	T* local_derivs = temp_Y + total_len;
	T* inc = local_derivs + total_len;
	T* inc_derivs = inc + dim;
	uint32_t* s_coprod_data = reinterpret_cast<uint32_t*>(inc_derivs + dim);
	uint32_t* s_coprod_off = s_coprod_data + coprod_data_len;
	uint32_t* s_order_idx = s_coprod_off + num_trees + 1;

	// --- One-time loads ---
	const uint64_t batch_off = static_cast<uint64_t>(batch_idx) * total_len;
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		s_bsig[i] = bsig_in[batch_off + i];
		s_derivs[i] = bsig_derivs_in[batch_off + i];
	}
	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		s_coprod_data[i] = g_coprod_data[i];
	for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
		s_coprod_off[i] = g_coprod_offsets[i];
	for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2); i += blockDim.x)
		s_order_idx[i] = g_order_index[i];

	// Initialize path_derivs to zero
	const T* bp = path + static_cast<uint64_t>(batch_idx) * path_stride;
	T* pd = path_derivs + static_cast<uint64_t>(batch_idx) * path_stride;
	for (uint32_t i = tid; i < static_cast<uint32_t>(path_stride); i += blockDim.x)
		pd[i] = T(0);
	__syncthreads();

	for (int seg = steps - 1; seg >= 0; --seg) {
		// --- 1. Load increment ---
		for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
			inc[d] = bp[(seg + 1) * dim + d] - bp[seg * dim + d];
		__syncthreads();

		// --- 2. Linear branched sig → temp_Y ---
		if (tid == 0) temp_Y[0] = T(1);
		if (tid < num_trees) {
			T prod = T(1);
			uint32_t lstart = labels_offsets[tid];
			uint32_t lend = labels_offsets[tid + 1];
			#pragma unroll 8
			for (uint32_t j = lstart; j < lend; ++j)
				prod *= inc[labels_data[j]];
			temp_Y[tid + 1] = prod * inv_factorial[tid];
		}
		__syncthreads();

		// --- 3. Butcher uncombine (order 1 to max_nodes) ---
		if (seg > 0) {
			for (int order = 1; order <= max_nodes; ++order) {
				uint32_t ostart = s_order_idx[order];
				uint32_t oend = s_order_idx[order + 1];
				if (tid >= ostart && tid < oend) {
					uint32_t fi = tid + 1;
					T val = s_bsig[fi] - temp_Y[fi];
					uint32_t pos = s_coprod_off[tid];
					uint32_t pend = s_coprod_off[tid + 1];
					while (pos < pend) {
						uint32_t nf = s_coprod_data[pos++];
						T term = temp_Y[s_coprod_data[pos++]];
						#pragma unroll 4
						for (uint32_t j = 0; j < nf; ++j)
							term *= s_bsig[s_coprod_data[pos++]];
						val -= term;
					}
					s_bsig[fi] = val;
				}
				__syncthreads();
			}
		}

		// --- 4. Butcher product derivative ---
		if (seg > 0) {
			// Initialize local_derivs = s_derivs (dF/dY = dF/dX_combined)
			if (tid == 0) local_derivs[0] = T(0);
			for (uint32_t i = tid; i < num_trees; i += blockDim.x)
				local_derivs[i + 1] = s_derivs[i + 1];
			__syncthreads();

			// Differentiate through coproduct terms
			if (tid < num_trees) {
				uint32_t fi = tid + 1;
				T dF_tau = s_derivs[fi];
				if (dF_tau != T(0)) {
					uint32_t pos = s_coprod_off[tid];
					uint32_t pend = s_coprod_off[tid + 1];
					while (pos < pend) {
						uint32_t nf = s_coprod_data[pos++];
						uint32_t trunk_flat = s_coprod_data[pos++];
						uint32_t forest_start = pos;

						T forest_product = T(1);
						#pragma unroll 4
						for (uint32_t j = 0; j < nf; ++j)
							forest_product *= s_bsig[s_coprod_data[pos++]];

						myAtomicAdd(&local_derivs[trunk_flat], dF_tau * forest_product);

						if (nf > 0) {
							T base = dF_tau * temp_Y[trunk_flat];
							for (uint32_t k = 0; k < nf; ++k) {
								uint32_t fk = s_coprod_data[forest_start + k];
								T partial = base;
								for (uint32_t j = 0; j < nf; ++j) {
									if (j != k)
										partial *= s_bsig[s_coprod_data[forest_start + j]];
								}
								myAtomicAdd(&s_derivs[fk], partial);
							}
						}
					}
				}
			}
			__syncthreads();
		}
		else {
			// seg == 0: local_derivs = s_derivs
			for (uint32_t i = tid; i < total_len; i += blockDim.x)
				local_derivs[i] = s_derivs[i];
			__syncthreads();
		}

		// --- 5. Linear bsig deriv → increment derivs ---
		// Zero inc_derivs
		for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
			inc_derivs[d] = T(0);
		__syncthreads();

		if (tid < num_trees) {
			T dF_dYi = local_derivs[tid + 1];
			if (dF_dYi != T(0)) {
				T inv_gamma = inv_factorial[tid];
				uint32_t lstart = labels_offsets[tid];
				uint32_t lend = labels_offsets[tid + 1];
				uint32_t n_labels = lend - lstart;

				T base = inv_gamma * dF_dYi;
				T prefix = T(1);
				for (uint32_t j = 0; j < n_labels; ++j) {
					T suffix = T(1);
					for (uint32_t k = j + 1; k < n_labels; ++k)
						suffix *= inc[labels_data[lstart + k]];
					myAtomicAdd(&inc_derivs[labels_data[lstart + j]], base * prefix * suffix);
					prefix *= inc[labels_data[lstart + j]];
				}
			}
		}
		__syncthreads();

		// --- 6. Accumulate into path derivs ---
		for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x) {
			pd[(seg + 1) * dim + d] += inc_derivs[d];
			pd[seg * dim + d] -= inc_derivs[d];
		}
		__syncthreads();
	}
}

// =========================================================================
// Combine kernel: out = butcher_product(bsig1, bsig2)
// =========================================================================

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_combine_ker(
	const T* __restrict__ bsig1,
	const T* __restrict__ bsig2,
	T* __restrict__ out,
	uint32_t total_len,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	const uint32_t* __restrict__ g_order_index,
	int max_nodes,
	uint32_t coprod_data_len
) {
	const uint32_t batch_idx = blockIdx.y;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	uint32_t* s_coprod_data = reinterpret_cast<uint32_t*>(smem);
	uint32_t* s_coprod_off = s_coprod_data + coprod_data_len;
	uint32_t* s_order_idx = s_coprod_off + num_trees + 1;

	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		s_coprod_data[i] = g_coprod_data[i];
	for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
		s_coprod_off[i] = g_coprod_offsets[i];
	for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2); i += blockDim.x)
		s_order_idx[i] = g_order_index[i];
	__syncthreads();

	const uint64_t off = static_cast<uint64_t>(batch_idx) * total_len;
	const T* X = bsig1 + off;
	const T* Y = bsig2 + off;
	T* O = out + off;

	// Copy X to out
	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		O[i] = X[i];
	__syncthreads();

	// Butcher product: out = butcher_product(X, Y) — process high to low order
	for (int order = max_nodes; order >= 1; --order) {
		uint32_t ostart = s_order_idx[order];
		uint32_t oend = s_order_idx[order + 1];
		if (tid >= ostart && tid < oend) {
			uint32_t fi = tid + 1;
			T val = O[fi] + Y[fi];
			uint32_t pos = s_coprod_off[tid];
			uint32_t pend = s_coprod_off[tid + 1];
			while (pos < pend) {
				uint32_t nf = s_coprod_data[pos++];
				T term = Y[s_coprod_data[pos++]];
				#pragma unroll 4
				for (uint32_t j = 0; j < nf; ++j)
					term *= O[s_coprod_data[pos++]];
				val += term;
			}
			O[fi] = val;
		}
		__syncthreads();
	}
}

// Combine backprop kernel: given dF/d(out), compute dF/d(bsig1) and dF/d(bsig2)
template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_combine_backprop_ker(
	const T* __restrict__ bsig1,
	const T* __restrict__ bsig2,
	const T* __restrict__ derivs,
	T* __restrict__ out1,
	T* __restrict__ out2,
	uint32_t total_len,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	const uint32_t* __restrict__ g_order_index,
	int max_nodes,
	uint32_t coprod_data_len
) {
	const uint32_t batch_idx = blockIdx.y;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	uint32_t* s_coprod_data = reinterpret_cast<uint32_t*>(smem);
	uint32_t* s_coprod_off = s_coprod_data + coprod_data_len;
	uint32_t* s_order_idx = s_coprod_off + num_trees + 1;

	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		s_coprod_data[i] = g_coprod_data[i];
	for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
		s_coprod_off[i] = g_coprod_offsets[i];
	for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2); i += blockDim.x)
		s_order_idx[i] = g_order_index[i];
	__syncthreads();

	const uint64_t off = static_cast<uint64_t>(batch_idx) * total_len;
	const T* X = bsig1 + off;
	const T* Y = bsig2 + off;
	T* dX = out1 + off;
	T* dY = out2 + off;

	// Initialize dX = derivs, dY = derivs (direct pass-through terms)
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		dX[i] = derivs[off + i];
		dY[i] = derivs[off + i];
	}
	if (tid == 0) dY[0] = T(0);
	__syncthreads();

	// Differentiate coproduct terms
	if (tid < num_trees) {
		uint32_t fi = tid + 1;
		T dF_tau = derivs[off + fi];
		if (dF_tau != T(0)) {
			uint32_t pos = s_coprod_off[tid];
			uint32_t pend = s_coprod_off[tid + 1];
			while (pos < pend) {
				uint32_t nf = s_coprod_data[pos++];
				uint32_t trunk_flat = s_coprod_data[pos++];
				uint32_t forest_start = pos;

				T forest_product = T(1);
				#pragma unroll 4
				for (uint32_t j = 0; j < nf; ++j)
					forest_product *= X[s_coprod_data[pos++]];

				myAtomicAdd(&dY[trunk_flat], dF_tau * forest_product);

				if (nf > 0) {
					T base = dF_tau * Y[trunk_flat];
					for (uint32_t k = 0; k < nf; ++k) {
						uint32_t fk = s_coprod_data[forest_start + k];
						T partial = base;
						for (uint32_t j = 0; j < nf; ++j) {
							if (j != k)
								partial *= X[s_coprod_data[forest_start + j]];
						}
						myAtomicAdd(&dX[fk], partial);
					}
				}
			}
		}
	}
}

// =========================================================================
// Host-side launchers
// =========================================================================

template<typename T>
void branched_sig_combine_cuda_(
	const T* bsig1, const T* bsig2, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes
) {
	const auto& gc = get_or_upload_gpu_cache(dimension, max_nodes);
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) throw std::invalid_argument("CUDA branched sig combine: num_trees > 1024 not supported");

	size_t smem = gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);
	dim3 grid(1, static_cast<unsigned int>(batch_size));

	branched_sig_combine_ker<T><<<grid, block, smem>>>(
		bsig1, bsig2, out, gc.total_length,
		gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
		gc.max_nodes, gc.coprod_data_len);
	cudaDeviceSynchronize();
	check_cuda_error();
}

template<typename T>
void branched_sig_combine_backprop_cuda_(
	const T* bsig1, const T* bsig2, const T* derivs, T* out1, T* out2,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes
) {
	const auto& gc = get_or_upload_gpu_cache(dimension, max_nodes);
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) throw std::invalid_argument("CUDA branched sig combine backprop: num_trees > 1024 not supported");

	size_t smem = gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);
	dim3 grid(1, static_cast<unsigned int>(batch_size));

	branched_sig_combine_backprop_ker<T><<<grid, block, smem>>>(
		bsig1, bsig2, derivs, out1, out2, gc.total_length,
		gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
		gc.max_nodes, gc.coprod_data_len);
	cudaDeviceSynchronize();
	check_cuda_error();
}

// =========================================================================
// Host-side launcher (forward)
// =========================================================================

template<typename T>
void branched_sig_cuda_core_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes
) {
	const auto& gc = get_or_upload_gpu_cache(dimension, max_nodes);

	// Single-point paths have no increments => identity branched sig
	if (length <= 1) {
		for (uint64_t b = 0; b < batch_size; ++b) {
			T* dst = out + b * gc.total_length;
			cudaMemsetAsync(dst, 0, gc.total_length * sizeof(T));
			T one = T(1);
			cudaMemcpyAsync(dst, &one, sizeof(T), cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();
		check_cuda_error();
		return;
	}

	const int steps = static_cast<int>(length - 1);
	const uint64_t path_stride = length * dimension;

	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024)
		throw std::invalid_argument("CUDA branched sig: num_trees > 1024 not supported");

	// Shared memory: temp[total_len]*T + inc[dim]*T
	//              + coprod_data[coprod_data_len]*4 + coprod_offsets[num_trees+1]*4
	//              + order_index[max_nodes+2]*4
	size_t smem = (gc.total_length + dimension) * sizeof(T)
		+ gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);

	dim3 grid(1, static_cast<unsigned int>(batch_size));

	// Select float or double inv_factorial
	const T* d_inv_fact;
	if constexpr (std::is_same_v<T, float>)
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f32);
	else
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f64);

	branched_sig_ker<T><<<grid, block, smem>>>(
		path, out, static_cast<int>(dimension), steps,
		gc.total_length, path_stride,
		gc.d_labels_data, gc.d_labels_offsets32,
		d_inv_fact,
		gc.d_coprod_data32, gc.d_coprod_offsets32,
		gc.d_order_index32, gc.max_nodes,
		gc.coprod_data_len
	);
	cudaDeviceSynchronize();
	check_cuda_error();
}

// =========================================================================
// time_aug / lead_lag wrapper
// =========================================================================

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
void branched_sig_cuda_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	bool time_aug,
	bool lead_lag,
	T end_time
) {
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		T* d_transformed;
		cudaMalloc(&d_transformed, t_path_size * sizeof(T));

		transform_path_<T>(path, d_transformed, batch_size, dimension, length, time_aug, lead_lag, end_time);
		cudaDeviceSynchronize();

		branched_sig_cuda_core_<T>(d_transformed, out, batch_size, t_dimension, t_length, max_nodes);

		cudaFree(d_transformed);
	}
	else {
		branched_sig_cuda_core_<T>(path, out, batch_size, dimension, length, max_nodes);
	}
}

// =========================================================================
// Backprop host launcher
// =========================================================================

template<typename T>
void branched_sig_backprop_cuda_core_(
	const T* path,
	T* out,
	const T* bsig_derivs,
	const T* bsig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes
) {
	const auto& gc = get_or_upload_gpu_cache(dimension, max_nodes);

	// Single-point paths have no increments => zero path gradients
	if (length <= 1) {
		cudaMemset(out, 0, batch_size * length * dimension * sizeof(T));
		cudaDeviceSynchronize();
		check_cuda_error();
		return;
	}

	const int steps = static_cast<int>(length - 1);
	const uint64_t path_stride = length * dimension;

	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024)
		throw std::invalid_argument("CUDA branched sig backprop: num_trees > 1024 not supported");

	// Shared memory: 4*total_len*T + 2*dim*T + coprod tables
	size_t smem = (4 * gc.total_length + 2 * dimension) * sizeof(T)
		+ gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);

	dim3 grid(1, static_cast<unsigned int>(batch_size));

	const T* d_inv_fact;
	if constexpr (std::is_same_v<T, float>)
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f32);
	else
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f64);

	branched_sig_backprop_ker<T><<<grid, block, smem>>>(
		path, out, bsig, bsig_derivs,
		static_cast<int>(dimension), steps,
		gc.total_length, path_stride,
		gc.d_labels_data, gc.d_labels_offsets32,
		d_inv_fact,
		gc.d_coprod_data32, gc.d_coprod_offsets32,
		gc.d_order_index32, gc.max_nodes,
		gc.coprod_data_len
	);
	cudaDeviceSynchronize();
	check_cuda_error();
}

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
void branched_sig_backprop_cuda_(
	const T* path,
	T* out,
	const T* bsig_derivs,
	const T* bsig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	bool time_aug,
	bool lead_lag,
	T end_time
) {
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	if (time_aug || lead_lag) {
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		T* d_transformed = nullptr;
		T* d_transformed_derivs = nullptr;

		try {
			cudaMalloc(&d_transformed, t_path_size * sizeof(T));
			transform_path_<T>(path, d_transformed, batch_size, dimension, length, time_aug, lead_lag, end_time);

			cudaMalloc(&d_transformed_derivs, t_path_size * sizeof(T));
			branched_sig_backprop_cuda_core_<T>(d_transformed, d_transformed_derivs, bsig_derivs, bsig,
				batch_size, t_dimension, t_length, max_nodes);

			cudaFree(d_transformed);
			d_transformed = nullptr;

			transform_path_backprop_<T>(d_transformed_derivs, out, batch_size, dimension, length, time_aug, lead_lag, end_time);
			cudaFree(d_transformed_derivs);
		} catch (...) {
			if (d_transformed) cudaFree(d_transformed);
			if (d_transformed_derivs) cudaFree(d_transformed_derivs);
			throw;
		}
	}
	else {
		branched_sig_backprop_cuda_core_<T>(path, out, bsig_derivs, bsig,
			batch_size, dimension, length, max_nodes);
	}
}

// =========================================================================
// extern "C" wrappers
// =========================================================================

extern "C" {

	CUSIG_API int branched_sig_cuda_f(const float* path, float* out, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_cuda_<float>(path, out, 1, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

	CUSIG_API int branched_sig_cuda_d(const double* path, double* out, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_cuda_<double>(path, out, 1, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_branched_sig_cuda_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_cuda_<float>(path, out, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_branched_sig_cuda_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_cuda_<double>(path, out, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

	CUSIG_API int branched_sig_combine_cuda_f(const float* bsig1, const float* bsig2, float* out, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_cuda_<float>(bsig1, bsig2, out, 1, dimension, max_nodes));
	}
	CUSIG_API int branched_sig_combine_cuda_d(const double* bsig1, const double* bsig2, double* out, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_cuda_<double>(bsig1, bsig2, out, 1, dimension, max_nodes));
	}
	CUSIG_API int batch_branched_sig_combine_cuda_f(const float* bsig1, const float* bsig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_cuda_<float>(bsig1, bsig2, out, batch_size, dimension, max_nodes));
	}
	CUSIG_API int batch_branched_sig_combine_cuda_d(const double* bsig1, const double* bsig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_cuda_<double>(bsig1, bsig2, out, batch_size, dimension, max_nodes));
	}

	CUSIG_API int branched_sig_combine_backprop_cuda_f(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_backprop_cuda_<float>(bsig1, bsig2, derivs, out1, out2, 1, dimension, max_nodes));
	}
	CUSIG_API int branched_sig_combine_backprop_cuda_d(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_backprop_cuda_<double>(bsig1, bsig2, derivs, out1, out2, 1, dimension, max_nodes));
	}
	CUSIG_API int batch_branched_sig_combine_backprop_cuda_f(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_backprop_cuda_<float>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes));
	}
	CUSIG_API int batch_branched_sig_combine_backprop_cuda_d(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_backprop_cuda_<double>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes));
	}

	CUSIG_API int branched_sig_backprop_cuda_f(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_backprop_cuda_<float>(path, out, bsig_derivs, bsig, 1, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

	CUSIG_API int branched_sig_backprop_cuda_d(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_backprop_cuda_<double>(path, out, bsig_derivs, bsig, 1, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_branched_sig_backprop_cuda_f(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_backprop_cuda_<float>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_branched_sig_backprop_cuda_d(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time) noexcept {
		CUSIG_SAFE_CALL(branched_sig_backprop_cuda_<double>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time));
	}

}
