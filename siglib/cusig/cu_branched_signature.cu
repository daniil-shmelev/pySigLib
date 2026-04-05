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
	double* d_inv_tree_factorial = nullptr;
	uint8_t* d_node_labels_data = nullptr;
	uint64_t* d_node_labels_offsets = nullptr;
	uint64_t* d_coproduct_data = nullptr;
	uint64_t* d_coproduct_offsets = nullptr;
	uint64_t* d_order_index = nullptr;

	uint64_t total_length = 0;
	uint64_t num_trees = 0;
	int max_nodes = 0;

	BranchedSigCacheGPU() = default;
	BranchedSigCacheGPU(const BranchedSigCacheGPU&) = delete;
	BranchedSigCacheGPU& operator=(const BranchedSigCacheGPU&) = delete;

	~BranchedSigCacheGPU() {
		if (d_inv_tree_factorial) cudaFree(d_inv_tree_factorial);
		if (d_node_labels_data) cudaFree(d_node_labels_data);
		if (d_node_labels_offsets) cudaFree(d_node_labels_offsets);
		if (d_coproduct_data) cudaFree(d_coproduct_data);
		if (d_coproduct_offsets) cudaFree(d_coproduct_offsets);
		if (d_order_index) cudaFree(d_order_index);
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

	// Upload to GPU
	auto gpu = std::make_unique<BranchedSigCacheGPU>();
	gpu->total_length = total_length;
	gpu->num_trees = num_trees;
	gpu->max_nodes = out_max_nodes;

	upload(gpu->d_inv_tree_factorial, h_inv_factorial.data(), h_inv_factorial.size());
	upload(gpu->d_node_labels_data, h_labels_data.data(), h_labels_data.size());
	upload(gpu->d_node_labels_offsets, h_labels_offsets.data(), h_labels_offsets.size());
	upload(gpu->d_coproduct_data, h_coprod_data.data(), h_coprod_data.size());
	upload(gpu->d_coproduct_offsets, h_coprod_offsets.data(), h_coprod_offsets.size());
	upload(gpu->d_order_index, h_order_index.data(), h_order_index.size());

	auto [ins, _] = s_gpu_cache_map.insert_or_assign(key, std::move(gpu));
	return *(ins->second);
}

// =========================================================================
// Fused segment kernel
// =========================================================================

template<typename T>
__global__ void branched_sig_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	int dim,
	int steps,
	uint64_t total_len,
	uint64_t path_stride,
	const uint8_t* __restrict__ labels_data,
	const uint64_t* __restrict__ labels_offsets,
	const double* __restrict__ inv_factorial,
	const uint64_t* __restrict__ coprod_data,
	const uint64_t* __restrict__ coprod_offsets,
	const uint64_t* __restrict__ order_index,
	int max_nodes
) {
	const uint64_t batch_idx = blockIdx.y;
	const uint64_t tid = threadIdx.x;
	const uint64_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	T* temp = reinterpret_cast<T*>(smem);
	T* inc = temp + total_len;

	const T* bp = path + batch_idx * path_stride;
	T* X = out + batch_idx * total_len;

	for (int seg = 0; seg < steps; ++seg) {
		// --- Cooperative increment load ---
		for (uint64_t d = tid; d < static_cast<uint64_t>(dim); d += blockDim.x)
			inc[d] = bp[(seg + 1) * dim + d] - bp[seg * dim + d];
		__syncthreads();

		// --- Linear branched sig ---
		T* tgt = (seg == 0) ? X : temp;
		if (tid == 0) tgt[0] = T(1);
		if (tid < num_trees) {
			T prod = T(1);
			for (uint64_t j = labels_offsets[tid]; j < labels_offsets[tid + 1]; ++j)
				prod *= inc[labels_data[j]];
			tgt[tid + 1] = prod * static_cast<T>(inv_factorial[tid]);
		}
		__syncthreads();

		// --- Butcher product (seg > 0 only) ---
		if (seg > 0) {
			for (int order = max_nodes; order >= 1; --order) {
				uint64_t ostart = order_index[order];
				uint64_t oend = order_index[order + 1];
				if (tid >= ostart && tid < oend) {
					uint64_t fi = tid + 1;
					T val = X[fi] + temp[fi];

					uint64_t pos = coprod_offsets[tid];
					uint64_t pend = coprod_offsets[tid + 1];
					while (pos < pend) {
						uint64_t nf = coprod_data[pos++];
						T term = temp[coprod_data[pos++]];
						for (uint64_t j = 0; j < nf; ++j)
							term *= X[coprod_data[pos++]];
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
// Host-side launcher
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
	const int steps = static_cast<int>(length - 1);
	const uint64_t path_stride = length * dimension;

	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024)
		throw std::invalid_argument("CUDA branched sig: num_trees > 1024 not supported");

	size_t smem = (gc.total_length + dimension) * sizeof(T);
	dim3 grid(1, static_cast<unsigned int>(batch_size));

	branched_sig_ker<T><<<grid, block, smem>>>(
		path, out, static_cast<int>(dimension), steps,
		gc.total_length, path_stride,
		gc.d_node_labels_data, gc.d_node_labels_offsets,
		gc.d_inv_tree_factorial,
		gc.d_coproduct_data, gc.d_coproduct_offsets,
		gc.d_order_index, gc.max_nodes
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

}
