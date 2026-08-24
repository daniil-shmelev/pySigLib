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
#include "cu_macros.h"
#include "cu_atomic.h"
#include "cache_lifecycle/cu_branched_log_sig_cache.h"
#include "cache_lifecycle/cu_disk_cache.h"
#include "cu_path_transforms.h"
#include "cu_utils.h"
#include "../shared/preparation/branched_sig/branched_sig_cache.h"
#include "../shared/preparation/branched_sig/branched_sig_cache_io.h"
#include "../shared/preparation/branched_sig/branched_sig_coef_cache.h"

#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// =========================================================================
// GPU cache: mirrors BranchedSigCache on device memory
// =========================================================================

struct BranchedSigCacheGPU {
	// 32-bit GPU copies for fast index arithmetic
	uint32_t* d_coprod_data32 = nullptr;
	uint32_t* d_coprod_offsets32 = nullptr;
	uint32_t* d_labels_offsets32 = nullptr;
	uint32_t* d_order_index32 = nullptr;
	uint32_t* d_chain_index_offsets32 = nullptr;
	uint32_t* d_chain_indices32 = nullptr;

	double* d_inv_factorial_f64 = nullptr;
	float* d_inv_factorial_f32 = nullptr;
	uint8_t* d_labels_data = nullptr;

	uint32_t total_length = 0;
	uint32_t num_trees = 0;
	uint32_t coprod_data_len = 0;
	uint32_t deriv_target_count = 0;
	int max_nodes = 0;

	BranchedSigCacheGPU() = default;
	BranchedSigCacheGPU(const BranchedSigCacheGPU&) = delete;
	BranchedSigCacheGPU& operator=(const BranchedSigCacheGPU&) = delete;

	~BranchedSigCacheGPU() {
		if (d_coprod_data32) cudaFree(d_coprod_data32);
		if (d_coprod_offsets32) cudaFree(d_coprod_offsets32);
		if (d_labels_offsets32) cudaFree(d_labels_offsets32);
		if (d_order_index32) cudaFree(d_order_index32);
		if (d_chain_index_offsets32) cudaFree(d_chain_index_offsets32);
		if (d_chain_indices32) cudaFree(d_chain_indices32);
		if (d_inv_factorial_f64) cudaFree(d_inv_factorial_f64);
		if (d_inv_factorial_f32) cudaFree(d_inv_factorial_f32);
		if (d_labels_data) cudaFree(d_labels_data);
	}
};

struct BranchedSigCacheKey {
	int device = 0;
	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;

	bool operator==(const BranchedSigCacheKey& other) const noexcept {
		return device == other.device
			&& dimension == other.dimension
			&& max_nodes == other.max_nodes
			&& planar == other.planar;
	}
};

struct BranchedSigCacheKeyHash {
	size_t operator()(const BranchedSigCacheKey& key) const noexcept {
		size_t h = std::hash<int>{}(key.device);
		auto combine = [&h](uint64_t value) {
			h ^= std::hash<uint64_t>{}(value) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
		};
		combine(key.dimension);
		combine(key.max_nodes);
		combine(static_cast<uint64_t>(key.planar));
		return h;
	}
};

static std::unordered_map<
	BranchedSigCacheKey,
	std::unique_ptr<BranchedSigCacheGPU>,
	BranchedSigCacheKeyHash
> s_gpu_cache_map;
static std::mutex s_gpu_cache_map_mu;

struct BranchedSigCoefCacheGPU {
	uint32_t* d_target_indices = nullptr;
	uint32_t* d_coprod_data = nullptr;
	uint32_t* d_coprod_offsets = nullptr;
	uint32_t* d_labels_offsets = nullptr;
	uint32_t* d_order_index = nullptr;
	uint32_t* d_leaf_indices = nullptr;
	uint32_t* d_correction_offsets = nullptr;
	uint32_t* d_correction_locals = nullptr;
	std::vector<uint32_t> correction_offsets;
	double* d_inv_factorial_f64 = nullptr;
	float* d_inv_factorial_f32 = nullptr;
	uint8_t* d_labels_data = nullptr;
	uint32_t cache_size = 0;
	uint32_t num_targets = 0;
	uint32_t coprod_data_len = 0;
	uint32_t dimension = 0;
	uint32_t max_nodes = 0;

	BranchedSigCoefCacheGPU() = default;
	BranchedSigCoefCacheGPU(const BranchedSigCoefCacheGPU&) = delete;
	BranchedSigCoefCacheGPU& operator=(const BranchedSigCoefCacheGPU&) = delete;

	~BranchedSigCoefCacheGPU() {
		if (d_target_indices) cudaFree(d_target_indices);
		if (d_coprod_data) cudaFree(d_coprod_data);
		if (d_coprod_offsets) cudaFree(d_coprod_offsets);
		if (d_labels_offsets) cudaFree(d_labels_offsets);
		if (d_order_index) cudaFree(d_order_index);
		if (d_leaf_indices) cudaFree(d_leaf_indices);
		if (d_correction_offsets) cudaFree(d_correction_offsets);
		if (d_correction_locals) cudaFree(d_correction_locals);
		if (d_inv_factorial_f64) cudaFree(d_inv_factorial_f64);
		if (d_inv_factorial_f32) cudaFree(d_inv_factorial_f32);
		if (d_labels_data) cudaFree(d_labels_data);
	}
};

struct BranchedSigCoefCacheKey {
	int device = 0;
	uint64_t data_dimension = 0;
	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;
	std::vector<uint64_t> tree_data;

	bool operator==(const BranchedSigCoefCacheKey& other) const noexcept {
		return device == other.device
			&& data_dimension == other.data_dimension
			&& dimension == other.dimension
			&& max_nodes == other.max_nodes
			&& planar == other.planar
			&& tree_data == other.tree_data;
	}
};

struct BranchedSigCoefCacheKeyHash {
	size_t operator()(const BranchedSigCoefCacheKey& key) const noexcept {
		size_t h = std::hash<int>{}(key.device);
		auto combine = [&h](uint64_t value) {
			h ^= std::hash<uint64_t>{}(value) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
		};
		combine(key.data_dimension);
		combine(key.dimension);
		combine(key.max_nodes);
		combine(static_cast<uint64_t>(key.planar));
		for (uint64_t value : key.tree_data)
			combine(value);
		return h;
	}
};

static std::unordered_map<
	BranchedSigCoefCacheKey,
	std::unique_ptr<BranchedSigCoefCacheGPU>,
	BranchedSigCoefCacheKeyHash
> s_gpu_coef_cache_map;
static std::mutex s_gpu_coef_cache_map_mu;

void release_branched_sig_gpu_state() {
	{
		std::lock_guard<std::mutex> lock(s_gpu_cache_map_mu);
		s_gpu_cache_map.clear();
	}
	{
		std::lock_guard<std::mutex> lock(s_gpu_coef_cache_map_mu);
		s_gpu_coef_cache_map.clear();
	}
}

template<typename T>
static void upload(T*& d_ptr, const T* h_data, size_t count) {
	if (count == 0) {
		d_ptr = nullptr;
		return;
	}
	CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_ptr, h_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

static void prepare_branched_sig_gpu_cache_(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar = false,
	bool use_disk = false,
	BranchedSigCache* host_cache = nullptr
) {
	BranchedSigCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.dimension = dimension;
	key.max_nodes = max_nodes;
	key.planar = planar;
	if (host_cache == nullptr) {
		std::lock_guard<std::mutex> lock(s_gpu_cache_map_mu);
		auto it = s_gpu_cache_map.find(key);
		if (it != s_gpu_cache_map.end())
			return;
	}

	BranchedSigCache c;
	if (use_disk)
		ensure_cuda_cache_dir_();
	if (!use_disk || !read_branched_sig_cache(
		get_cuda_cache_dir_() / cu_cache_folder_name,
		dimension, max_nodes, planar, c)) {
		c = BranchedSigCache(dimension, max_nodes, planar);
		if (use_disk)
			write_branched_sig_cache(
				get_cuda_cache_dir_() / cu_cache_folder_name, c);
	}
	{
		std::lock_guard<std::mutex> lock(s_gpu_cache_map_mu);
		if (s_gpu_cache_map.find(key) != s_gpu_cache_map.end()) {
			if (host_cache != nullptr)
				*host_cache = std::move(c);
			return;
		}
	}

	uint64_t num_trees = c.total_length - 1;

	// Convert to 32-bit for GPU (with overflow check)
	auto safe_narrow = [](const uint64_t* src, uint32_t* dst, size_t n) {
		for (size_t i = 0; i < n; ++i) {
			if (src[i] > UINT32_MAX)
				throw std::overflow_error("Branched sig cache value exceeds uint32 range");
			dst[i] = static_cast<uint32_t>(src[i]);
		}
	};
	std::vector<uint32_t> coprod_data32(c.coproduct_data.size());
	std::vector<uint32_t> coprod_offsets32(c.coproduct_offsets.size());
	std::vector<uint32_t> labels_offsets32(c.node_labels_offsets.size());
	std::vector<uint32_t> order_index32(c.order_index.size());
	std::vector<uint32_t> chain_index_offsets32(c.chain_index_offsets.size());
	std::vector<uint32_t> chain_indices32(c.chain_indices.size());
	safe_narrow(c.coproduct_data.data(), coprod_data32.data(), c.coproduct_data.size());
	safe_narrow(c.coproduct_offsets.data(), coprod_offsets32.data(), c.coproduct_offsets.size());
	safe_narrow(c.node_labels_offsets.data(), labels_offsets32.data(), c.node_labels_offsets.size());
	safe_narrow(c.order_index.data(), order_index32.data(), c.order_index.size());
	safe_narrow(c.chain_index_offsets.data(), chain_index_offsets32.data(), c.chain_index_offsets.size());
	safe_narrow(c.chain_indices.data(), chain_indices32.data(), c.chain_indices.size());

	std::vector<float> inv_factorial_f32(num_trees);
	for (size_t i = 0; i < num_trees; ++i) inv_factorial_f32[i] = static_cast<float>(c.inv_tree_factorial[i]);

	// Upload to GPU
	auto gpu = std::make_unique<BranchedSigCacheGPU>();
	auto narrow32 = [](uint64_t v) -> uint32_t {
		if (v > UINT32_MAX) throw std::overflow_error("Branched sig cache value exceeds uint32 range");
		return static_cast<uint32_t>(v);
	};
	gpu->total_length = narrow32(c.total_length);
	gpu->num_trees = narrow32(num_trees);
	gpu->coprod_data_len = narrow32(c.coproduct_data.size());
	gpu->deriv_target_count = order_index32[static_cast<size_t>(max_nodes)] + 1;
	gpu->max_nodes = static_cast<int>(max_nodes);

	upload(gpu->d_coprod_data32, coprod_data32.data(), coprod_data32.size());
	upload(gpu->d_coprod_offsets32, coprod_offsets32.data(), coprod_offsets32.size());
	upload(gpu->d_labels_offsets32, labels_offsets32.data(), labels_offsets32.size());
	upload(gpu->d_order_index32, order_index32.data(), order_index32.size());
	upload(gpu->d_chain_index_offsets32, chain_index_offsets32.data(), chain_index_offsets32.size());
	upload(gpu->d_chain_indices32, chain_indices32.data(), chain_indices32.size());
	upload(gpu->d_inv_factorial_f64, c.inv_tree_factorial.data(), c.inv_tree_factorial.size());
	upload(gpu->d_inv_factorial_f32, inv_factorial_f32.data(), inv_factorial_f32.size());
	upload(gpu->d_labels_data, c.node_labels_data.data(), c.node_labels_data.size());

	std::lock_guard<std::mutex> lock(s_gpu_cache_map_mu);
	s_gpu_cache_map.try_emplace(key, std::move(gpu));
	if (host_cache != nullptr)
		*host_cache = std::move(c);
}

static const BranchedSigCacheGPU& get_gpu_cache(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar = false
) {
	BranchedSigCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.dimension = dimension;
	key.max_nodes = max_nodes;
	key.planar = planar;
	std::lock_guard<std::mutex> lock(s_gpu_cache_map_mu);
	auto it = s_gpu_cache_map.find(key);
	if (it != s_gpu_cache_map.end())
		return *(it->second);
	throw cache_not_found_error(
		"CUDA branched sig cache not found - call prepare_branched_sig with device='cuda' first");
}

static void prepare_branched_sig_coef_cache_cuda_(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	bool use_disk
) {
	if (data_dimension == 0)
		throw std::invalid_argument("prepare_branched_sig_coef_cuda received dimension 0");
	if (tree_data == nullptr && tree_data_len != 0)
		throw std::invalid_argument("prepare_branched_sig_coef_cuda received null tree_data");

	BranchedSigCoefCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.data_dimension = data_dimension;
	key.dimension = dimension;
	key.max_nodes = max_nodes;
	key.planar = planar;
	if (tree_data_len != 0)
		key.tree_data.assign(tree_data, tree_data + tree_data_len);

	{
		std::lock_guard<std::mutex> lock(s_gpu_coef_cache_map_mu);
		auto it = s_gpu_coef_cache_map.find(key);
		if (it != s_gpu_coef_cache_map.end())
			return;
	}

	BranchedSigCoefCache cache;
	if (use_disk)
		ensure_cuda_cache_dir_();
	if (!use_disk || !read_branched_sig_coef_cache(
		get_cuda_cache_dir_() / cu_cache_folder_name, data_dimension, dimension,
		max_nodes, planar, key.tree_data, cache)) {
		cache = BranchedSigCoefCache(
			key.tree_data.data(), key.tree_data.size(), data_dimension, dimension,
			max_nodes, planar);
		if (use_disk)
			write_branched_sig_coef_cache(
				get_cuda_cache_dir_() / cu_cache_folder_name, data_dimension,
				dimension, max_nodes, planar, key.tree_data, cache);
	}

	auto narrow = [](const std::vector<uint64_t>& source) {
		std::vector<uint32_t> result(source.size());
		for (size_t i = 0; i < source.size(); ++i) {
			if (source[i] > UINT32_MAX)
				throw std::overflow_error("Branched sig coef cache value exceeds uint32 range");
			result[i] = static_cast<uint32_t>(source[i]);
		}
		return result;
	};
	auto narrow_value = [](uint64_t value) {
		if (value > UINT32_MAX)
			throw std::overflow_error("Branched sig coef cache size exceeds uint32 range");
		return static_cast<uint32_t>(value);
	};

	const auto target_indices = narrow(cache.target_indices);
	const auto coprod_data = narrow(cache.coproduct_data);
	const auto coprod_offsets = narrow(cache.coproduct_offsets);
	const auto labels_offsets = narrow(cache.node_labels_offsets);
	const auto order_index = narrow(cache.order_index);
	const auto leaf_indices = narrow(cache.leaf_indices);
	std::vector<uint32_t> correction_offsets(cache.correction_indices.size());
	std::vector<uint32_t> correction_locals(cache.correction_indices.size());
	for (size_t i = 0; i < cache.correction_indices.size(); ++i) {
		correction_offsets[i] = narrow_value(cache.correction_indices[i].first);
		correction_locals[i] = narrow_value(cache.correction_indices[i].second);
	}
	std::vector<float> inv_factorial_f32(cache.inv_tree_factorial.size());
	for (size_t i = 0; i < cache.inv_tree_factorial.size(); ++i)
		inv_factorial_f32[i] = static_cast<float>(cache.inv_tree_factorial[i]);

	auto gpu = std::make_unique<BranchedSigCoefCacheGPU>();
	gpu->cache_size = narrow_value(cache.inv_tree_factorial.size());
	gpu->num_targets = narrow_value(cache.target_indices.size());
	gpu->coprod_data_len = narrow_value(cache.coproduct_data.size());
	gpu->correction_offsets = correction_offsets;
	gpu->dimension = narrow_value(dimension);
	gpu->max_nodes = narrow_value(cache.max_nodes);
	upload(gpu->d_target_indices, target_indices.data(), target_indices.size());
	upload(gpu->d_coprod_data, coprod_data.data(), coprod_data.size());
	upload(gpu->d_coprod_offsets, coprod_offsets.data(), coprod_offsets.size());
	upload(gpu->d_labels_offsets, labels_offsets.data(), labels_offsets.size());
	upload(gpu->d_order_index, order_index.data(), order_index.size());
	upload(gpu->d_leaf_indices, leaf_indices.data(), leaf_indices.size());
	upload(gpu->d_correction_offsets, correction_offsets.data(), correction_offsets.size());
	upload(gpu->d_correction_locals, correction_locals.data(), correction_locals.size());
	upload(gpu->d_inv_factorial_f64, cache.inv_tree_factorial.data(), cache.inv_tree_factorial.size());
	upload(gpu->d_inv_factorial_f32, inv_factorial_f32.data(), inv_factorial_f32.size());
	upload(gpu->d_labels_data, cache.node_labels_data.data(), cache.node_labels_data.size());

	std::lock_guard<std::mutex> lock(s_gpu_coef_cache_map_mu);
	s_gpu_coef_cache_map.try_emplace(std::move(key), std::move(gpu));
}

static const BranchedSigCoefCacheGPU& get_branched_sig_coef_cache_cuda_(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	if (tree_data == nullptr && tree_data_len != 0)
		throw std::invalid_argument("branched_sig_coef_cuda received null tree_data");
	BranchedSigCoefCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.data_dimension = data_dimension;
	key.dimension = dimension;
	key.max_nodes = max_nodes;
	key.planar = planar;
	if (tree_data_len != 0)
		key.tree_data.assign(tree_data, tree_data + tree_data_len);

	std::lock_guard<std::mutex> lock(s_gpu_coef_cache_map_mu);
	const auto it = s_gpu_coef_cache_map.find(key);
	if (it == s_gpu_coef_cache_map.end())
		throw cache_not_found_error(
			"CUDA branched signature coefficient cache not found - call prepare_branched_sig_coef with device='cuda' first");
	return *(it->second);
}

static uint32_t branched_sig_coef_num_corrections_(
	const BranchedSigCoefCacheGPU& cache,
	uint64_t correction_len
) {
	return static_cast<uint32_t>(std::lower_bound(
		cache.correction_offsets.begin(), cache.correction_offsets.end(), correction_len)
		- cache.correction_offsets.begin());
}

void clear_cuda_branched_sig_gpu_cache_() {
	release_branched_sig_gpu_state();
}

#include "cu_branched_sig_coef.cuh"

// =========================================================================
// Fused segment kernel
// =========================================================================

// Shared memory layout:
// | temp[total_len] | inc[dim] | s_coprod_data[coprod_len] | s_coprod_off[num_trees+1] | s_order_idx[max_nodes+2] |

template<typename T>
__device__ void branched_hopf_convolution_block_(
	const T* X,
	const T* Y,
	T* out,
	uint32_t total_len,
	const uint32_t* s_coprod_data,
	const uint32_t* s_coprod_off,
	uint32_t tid
) {
	const uint32_t num_trees = total_len - 1;
	if (tid == 0) out[0] = X[0] * Y[0];
	for (uint32_t tree = tid; tree < num_trees; tree += blockDim.x) {
		const uint32_t fi = tree + 1;
		T val = X[fi] * Y[0] + X[0] * Y[fi];
		uint32_t pos = s_coprod_off[tree];
		const uint32_t pend = s_coprod_off[tree + 1];
		while (pos < pend) {
			const uint32_t nf = s_coprod_data[pos++];
			T term = Y[s_coprod_data[pos++]];
			#pragma unroll 4
			for (uint32_t j = 0; j < nf; ++j)
				term *= X[s_coprod_data[pos++]];
			val += term;
		}
		out[fi] = val;
	}
	__syncthreads();
}

template<typename T>
__device__ void add_correction_block_(
	T* out,
	int dim,
	int data_dim,
	const T* correction,
	uint64_t correction_len,
	const uint32_t* chain_index_offsets,
	const uint32_t* chain_indices,
	int max_nodes,
	uint32_t tid
) {
	if (correction == nullptr || correction_len == 0)
		return;

	uint64_t offset = 0;
	uint64_t level_size = static_cast<uint64_t>(data_dim);
	for (int level = 2; level <= max_nodes; ++level) {
		level_size *= static_cast<uint64_t>(data_dim);
		if (offset + level_size > correction_len)
			break;

		for (uint64_t word = tid; word < level_size; word += blockDim.x) {
			const T value = correction[offset + word];
			if (value == T(0))
				continue;

			uint64_t tmp = word;
			uint64_t aug_word = 0;
			uint64_t pow = level_size / static_cast<uint64_t>(data_dim);
			for (int pos = 0; pos < level; ++pos) {
				const uint64_t label = tmp / pow;
				tmp -= label * pow;
				if (pos + 1 < level)
					pow /= static_cast<uint64_t>(data_dim);

				aug_word = aug_word * static_cast<uint64_t>(dim) + label;
			}

			const uint32_t flat = chain_indices[chain_index_offsets[level] + aug_word];
			if (flat != 0)
				out[flat] += value;
		}
		offset += level_size;
	}
	__syncthreads();
}

template<typename T>
__device__ void linear_branched_sig_block_(
	const T* inc,
	T* out,
	int dim,
	uint32_t total_len,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	uint32_t tid
) {
	const uint32_t num_trees = total_len - 1;
	if (tid == 0) out[0] = T(1);
	for (uint32_t tree = tid; tree < num_trees; tree += blockDim.x) {
		T prod = T(1);
		const uint32_t lstart = labels_offsets[tree];
		const uint32_t lend = labels_offsets[tree + 1];
		#pragma unroll 8
		for (uint32_t j = lstart; j < lend; ++j)
			prod *= inc[labels_data[j]];
		out[tree + 1] = prod * inv_factorial[tree];
	}
	__syncthreads();
}

template<typename T>
__device__ void local_branched_sig_block_(
	const T* inc,
	T* out,
	T* local_log,
	T* power,
	T* next_power,
	int dim,
	int data_dim,
	const T* correction,
	uint64_t correction_len,
	uint32_t total_len,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	const uint32_t* s_coprod_data,
	const uint32_t* s_coprod_off,
	const uint32_t* s_order_idx,
	const uint32_t* chain_index_offsets,
	const uint32_t* chain_indices,
	int max_nodes,
	uint32_t tid
) {
	if (correction_len == 0) {
		linear_branched_sig_block_(inc, out, dim, total_len, labels_data, labels_offsets, inv_factorial, tid);
		return;
	}

	if (max_nodes <= 2) {
		linear_branched_sig_block_(inc, out, dim, total_len, labels_data, labels_offsets, inv_factorial, tid);
		add_correction_block_(out, dim, data_dim, correction, correction_len,
			chain_index_offsets, chain_indices, max_nodes, tid);
		return;
	}

	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		local_log[i] = T(0);
	__syncthreads();
	for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
		local_log[s_order_idx[1] + d + 1] = inc[d];
	__syncthreads();
	add_correction_block_(local_log, dim, data_dim, correction, correction_len,
		chain_index_offsets, chain_indices, max_nodes, tid);

	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		out[i] = T(0);
		power[i] = local_log[i];
	}
	if (tid == 0) out[0] = T(1);
	__syncthreads();

	T inv_k_factorial = T(1);
	T* cur_power = power;
	T* cur_next = next_power;
	for (int k = 1; k <= max_nodes; ++k) {
		inv_k_factorial /= static_cast<T>(k);
		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			out[i] += inv_k_factorial * cur_power[i];
		__syncthreads();

		if (k < max_nodes) {
			branched_hopf_convolution_block_(cur_power, local_log, cur_next,
				total_len, s_coprod_data, s_coprod_off, tid);
			T* tmp = cur_power;
			cur_power = cur_next;
			cur_next = tmp;
		}
	}
}

template<typename T, bool use_shared_storage>
__global__ __launch_bounds__(use_shared_storage ? 1024 : 256)
void branched_sig_ker(
	const T* __restrict__ path,
	T* __restrict__ out,
	int dim,
	int data_dim,
	int steps,
	uint32_t total_len,
	uint64_t path_stride,
	const uint8_t* __restrict__ labels_data,
	const uint32_t* __restrict__ labels_offsets,
	const T* __restrict__ inv_factorial,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	const uint32_t* __restrict__ g_order_index,
	const uint32_t* __restrict__ chain_index_offsets,
	const uint32_t* __restrict__ chain_indices,
	int max_nodes,
	uint32_t coprod_data_len,
	const T* __restrict__ correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride,
	bool planar_fast_path,
	T* __restrict__ workspace,
	uint64_t workspace_stride,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	T* temp = use_shared_storage
		? reinterpret_cast<T*>(smem)
		: workspace + local_batch_idx * workspace_stride;
	const bool has_correction = correction_len != 0;
	const bool use_shared_state = planar_fast_path && !has_correction;
	T* local_log = has_correction ? temp + total_len : temp;
	T* power = has_correction ? local_log + total_len : temp;
	T* next_power = has_correction ? power + total_len : temp;
	T* state = use_shared_state ? temp + total_len : nullptr;
	T* next_state = use_shared_state ? state + total_len : nullptr;
	T* inc = has_correction ? next_power + total_len : (use_shared_state ? next_state + total_len : temp + total_len);
	uint32_t* shared_coprod_data = reinterpret_cast<uint32_t*>(inc + dim);
	uint32_t* shared_coprod_off = shared_coprod_data + coprod_data_len;
	uint32_t* shared_order_idx = shared_coprod_off + num_trees + 1;
	const uint32_t* s_coprod_data = g_coprod_data;
	const uint32_t* s_coprod_off = g_coprod_offsets;
	const uint32_t* s_order_idx = g_order_index;

	// --- One-time cooperative load of coproduct table into shared memory ---
	if constexpr (use_shared_storage) {
		for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
			shared_coprod_data[i] = g_coprod_data[i];
		for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
			shared_coprod_off[i] = g_coprod_offsets[i];
		for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2);
			i += blockDim.x)
			shared_order_idx[i] = g_order_index[i];
		s_coprod_data = shared_coprod_data;
		s_coprod_off = shared_coprod_off;
		s_order_idx = shared_order_idx;
	}
	__syncthreads();

	const T* bp = path + static_cast<uint64_t>(batch_idx) * path_stride;
	T* X = out + static_cast<uint64_t>(batch_idx) * total_len;
	const T* batch_correction = has_correction
		? correction + static_cast<uint64_t>(batch_idx) * correction_batch_stride
		: nullptr;

	for (int seg = 0; seg < steps; ++seg) {
		// --- Cooperative increment load ---
		for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
			inc[d] = bp[(seg + 1) * dim + d] - bp[seg * dim + d];
		__syncthreads();

		const T* seg_correction = has_correction
			? batch_correction + static_cast<uint64_t>(seg) * correction_segment_stride
			: nullptr;
		T* tgt = use_shared_state && seg == 0 ? state : ((seg == 0) ? X : temp);
		local_branched_sig_block_(inc, tgt, local_log, power, next_power,
			dim, data_dim, seg_correction, correction_len, total_len,
			labels_data, labels_offsets, inv_factorial,
			s_coprod_data, s_coprod_off, s_order_idx,
			chain_index_offsets, chain_indices, max_nodes, tid);

		if (use_shared_state) {
			if (seg > 0) {
				branched_hopf_convolution_block_(state, temp, next_state,
					total_len, s_coprod_data, s_coprod_off, tid);
				T* tmp = state;
				state = next_state;
				next_state = tmp;
			}
		}
		else if (seg > 0) {
			for (int order = max_nodes; order >= 1; --order) {
				const uint32_t ostart = s_order_idx[order];
				const uint32_t oend = s_order_idx[order + 1];
				for (uint32_t tree = ostart + tid; tree < oend;
					tree += blockDim.x) {
					uint32_t fi = tree + 1;
					T val = X[fi] + temp[fi];

					uint32_t pos = s_coprod_off[tree];
					uint32_t pend = s_coprod_off[tree + 1];
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

	if (use_shared_state) {
		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			X[i] = state[i];
	}
}

// =========================================================================
// Backprop kernel
// =========================================================================

// Shared memory layout:
// | s_bsig[total_len] | s_derivs[total_len] | temp_Y[total_len] | local_derivs[total_len] |
// | inc[dim] | inc_derivs[dim] | increment reduction | coprod tables (same as forward) |

template<typename T>
__device__ void branched_hopf_convolution_deriv_block_(
	const T* X,
	const T* Y,
	const T* d_out,
	T* d_X,
	T* d_Y,
	uint32_t total_len,
	const uint32_t* s_coprod_data,
	const uint32_t* s_coprod_off,
	uint32_t tid
) {
	const uint32_t num_trees = total_len - 1;
	if (tid == 0) {
		myAtomicAdd(&d_X[0], d_out[0] * Y[0]);
		myAtomicAdd(&d_Y[0], d_out[0] * X[0]);
	}
	for (uint32_t tree = tid; tree < num_trees; tree += blockDim.x) {
		const uint32_t fi = tree + 1;
		const T d = d_out[fi];
		if (d != T(0)) {
			myAtomicAdd(&d_X[fi], d * Y[0]);
			myAtomicAdd(&d_Y[0], d * X[fi]);
			myAtomicAdd(&d_X[0], d * Y[fi]);
			myAtomicAdd(&d_Y[fi], d * X[0]);

			uint32_t pos = s_coprod_off[tree];
			const uint32_t pend = s_coprod_off[tree + 1];
			while (pos < pend) {
				const uint32_t nf = s_coprod_data[pos++];
				const uint32_t trunk_flat = s_coprod_data[pos++];
				const uint32_t forest_start = pos;
				T forest_product = T(1);
				#pragma unroll 4
				for (uint32_t j = 0; j < nf; ++j)
					forest_product *= X[s_coprod_data[pos++]];

				myAtomicAdd(&d_Y[trunk_flat], d * forest_product);
				for (uint32_t k = 0; k < nf; ++k) {
					const uint32_t fk = s_coprod_data[forest_start + k];
					T partial = d * Y[trunk_flat];
					for (uint32_t j = 0; j < nf; ++j) {
						if (j != k)
							partial *= X[s_coprod_data[forest_start + j]];
					}
					myAtomicAdd(&d_X[fk], partial);
				}
			}
		}
	}
	__syncthreads();
}

template<typename T>
__device__ void linear_bsig_deriv_to_inc_block_(
	const T* local_derivs,
	const T* inc,
	T* inc_derivs,
	T* reduction,
	bool use_reduction,
	int dim,
	uint32_t total_len,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	uint32_t tid
) {
	const uint32_t num_trees = total_len - 1;
	if (!use_reduction) {
		for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
			inc_derivs[d] = T(0);
		__syncthreads();

		for (uint32_t tree = tid; tree < num_trees; tree += blockDim.x) {
			const T dF_dYi = local_derivs[tree + 1];
			if (dF_dYi != T(0)) {
				const uint32_t lstart = labels_offsets[tree];
				const uint32_t lend = labels_offsets[tree + 1];
				const T base = inv_factorial[tree] * dF_dYi;
				T prefix = T(1);
				for (uint32_t j = lstart; j < lend; ++j) {
					T suffix = T(1);
					for (uint32_t k = j + 1; k < lend; ++k)
						suffix *= inc[labels_data[k]];
					myAtomicAdd(&inc_derivs[labels_data[j]], base * prefix * suffix);
					prefix *= inc[labels_data[j]];
				}
			}
		}
		__syncthreads();
		return;
	}

	const uint32_t num_warps = blockDim.x >> 5;
	const uint32_t warp = tid >> 5;
	const uint32_t lane = tid & 31u;
	for (uint32_t d = 0; d < static_cast<uint32_t>(dim); ++d) {
		T value = T(0);
		for (uint32_t tree = tid; tree < num_trees; tree += blockDim.x) {
			const T dF_dYi = local_derivs[tree + 1];
			if (dF_dYi != T(0)) {
				const uint32_t lstart = labels_offsets[tree];
				const uint32_t lend = labels_offsets[tree + 1];
				const T base = inv_factorial[tree] * dF_dYi;
				T prefix = T(1);
				for (uint32_t j = lstart; j < lend; ++j) {
					T suffix = T(1);
					for (uint32_t k = j + 1; k < lend; ++k)
						suffix *= inc[labels_data[k]];
					if (labels_data[j] == d)
						value += base * prefix * suffix;
					prefix *= inc[labels_data[j]];
				}
			}
		}
		value += __shfl_down_sync(0xffffffff, value, 16);
		value += __shfl_down_sync(0xffffffff, value, 8);
		value += __shfl_down_sync(0xffffffff, value, 4);
		value += __shfl_down_sync(0xffffffff, value, 2);
		value += __shfl_down_sync(0xffffffff, value, 1);
		if (lane == 0)
			reduction[d * num_warps + warp] = value;
	}
	__syncthreads();
	for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x) {
		T value = T(0);
		for (uint32_t warp_idx = 0; warp_idx < num_warps; ++warp_idx)
			value += reduction[d * num_warps + warp_idx];
		inc_derivs[d] = value;
	}
	__syncthreads();
}

template<typename T>
__device__ void local_log_bsig_deriv_to_inc_rolling_block_(
	const T* local_derivs,
	const T* inc,
	T* inc_derivs,
	T* inc_reduction,
	bool use_inc_reduction,
	T* local_log,
	T* powers,
	T* power_derivs,
	T* d_correction,
	int dim,
	int data_dim,
	const T* correction,
	uint64_t correction_len,
	uint32_t total_len,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	const uint32_t* s_coprod_data,
	const uint32_t* s_coprod_off,
	const uint32_t* s_order_idx,
	const uint32_t* chain_index_offsets,
	const uint32_t* chain_indices,
	int max_nodes,
	uint32_t tid
) {
	if (max_nodes <= 2) {
		linear_bsig_deriv_to_inc_block_(local_derivs, inc, inc_derivs,
			inc_reduction, use_inc_reduction, dim,
			total_len, labels_data, labels_offsets, inv_factorial, tid);
		return;
	}

	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		local_log[i] = T(0);
		d_correction[i] = T(0);
	}
	__syncthreads();

	for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
		local_log[s_order_idx[1] + d + 1] = inc[d];
	__syncthreads();
	add_correction_block_(local_log, dim, data_dim, correction, correction_len,
		chain_index_offsets, chain_indices, max_nodes, tid);

	T* power_current = powers;
	T* power_next = powers + total_len;
	T* adjoint_current = power_derivs;
	T* adjoint_next = power_derivs + total_len;
	T inv_k_factorial = T(1);
	for (int k = 2; k <= max_nodes; ++k)
		inv_k_factorial /= static_cast<T>(k);
	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		adjoint_current[i] = i == 0
			? T(0) : inv_k_factorial * local_derivs[i];
	__syncthreads();

	for (int k = max_nodes; k > 1; --k) {
		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			power_current[i] = local_log[i];
		__syncthreads();
		T* current = power_current;
		T* next = power_next;
		for (int power_level = 2; power_level < k; ++power_level) {
			branched_hopf_convolution_block_(
				current, local_log, next, total_len,
				s_coprod_data, s_coprod_off, tid);
			T* swap = current;
			current = next;
			next = swap;
		}

		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			adjoint_next[i] = T(0);
		__syncthreads();
		branched_hopf_convolution_deriv_block_(
			current, local_log, adjoint_current, adjoint_next,
			d_correction, total_len, s_coprod_data, s_coprod_off, tid);
		inv_k_factorial *= static_cast<T>(k);
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			adjoint_next[i] += inv_k_factorial * local_derivs[i];
		__syncthreads();
		T* adjoint_swap = adjoint_current;
		adjoint_current = adjoint_next;
		adjoint_next = adjoint_swap;
	}

	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		d_correction[i] += adjoint_current[i];
	__syncthreads();

	for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
		inc_derivs[d] = d_correction[s_order_idx[1] + d + 1];
	__syncthreads();
}

template<typename T>
__device__ void local_log_bsig_deriv_to_inc_saved_block_(
	const T* local_derivs,
	const T* inc,
	T* inc_derivs,
	T* inc_reduction,
	bool use_inc_reduction,
	T* local_log,
	T* powers,
	T* power_derivs,
	T* d_correction,
	int dim,
	int data_dim,
	const T* correction,
	uint64_t correction_len,
	uint32_t total_len,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	const uint32_t* s_coprod_data,
	const uint32_t* s_coprod_off,
	const uint32_t* s_order_idx,
	const uint32_t* chain_index_offsets,
	const uint32_t* chain_indices,
	int max_nodes,
	uint32_t tid
) {
	if (max_nodes <= 2) {
		linear_bsig_deriv_to_inc_block_(local_derivs, inc, inc_derivs,
			inc_reduction, use_inc_reduction, dim,
			total_len, labels_data, labels_offsets, inv_factorial, tid);
		return;
	}

	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		local_log[i] = T(0);
		d_correction[i] = T(0);
	}
	for (uint32_t i = tid;
		i < static_cast<uint32_t>(max_nodes) * total_len;
		i += blockDim.x)
		power_derivs[i] = T(0);
	__syncthreads();

	for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
		local_log[s_order_idx[1] + d + 1] = inc[d];
	__syncthreads();
	add_correction_block_(local_log, dim, data_dim, correction, correction_len,
		chain_index_offsets, chain_indices, max_nodes, tid);

	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		powers[i] = local_log[i];
	__syncthreads();
	for (int k = 2; k <= max_nodes; ++k) {
		branched_hopf_convolution_block_(
			powers + static_cast<uint32_t>(k - 2) * total_len,
			local_log,
			powers + static_cast<uint32_t>(k - 1) * total_len,
			total_len, s_coprod_data, s_coprod_off, tid);
	}

	T inv_k_factorial = T(1);
	for (int k = 1; k <= max_nodes; ++k) {
		inv_k_factorial /= static_cast<T>(k);
		T* d_power = power_derivs
			+ static_cast<uint32_t>(k - 1) * total_len;
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			d_power[i] += inv_k_factorial * local_derivs[i];
		__syncthreads();
	}

	for (int k = max_nodes; k > 1; --k) {
		branched_hopf_convolution_deriv_block_(
			powers + static_cast<uint32_t>(k - 2) * total_len,
			local_log,
			power_derivs + static_cast<uint32_t>(k - 1) * total_len,
			power_derivs + static_cast<uint32_t>(k - 2) * total_len,
			d_correction, total_len, s_coprod_data, s_coprod_off, tid);
	}

	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		d_correction[i] += power_derivs[i];
	__syncthreads();

	for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
		inc_derivs[d] = d_correction[s_order_idx[1] + d + 1];
	__syncthreads();
}

template<typename T, bool use_shared_storage>
__global__ __launch_bounds__(use_shared_storage ? 1024 : 256)
void branched_sig_backprop_ker(
	const T* __restrict__ path,
	T* __restrict__ path_derivs,
	const T* __restrict__ bsig_in,
	const T* __restrict__ bsig_derivs_in,
	int dim,
	int data_dim,
	int steps,
	uint32_t total_len,
	uint64_t path_stride,
	const uint8_t* __restrict__ labels_data,
	const uint32_t* __restrict__ labels_offsets,
	const T* __restrict__ inv_factorial,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	const uint32_t* __restrict__ g_order_index,
	const uint32_t* __restrict__ chain_index_offsets,
	const uint32_t* __restrict__ chain_indices,
	int max_nodes,
	uint32_t coprod_data_len,
	uint32_t deriv_target_count,
	bool use_inc_reduction,
	bool use_product_reduction,
	const T* __restrict__ correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride,
	T* __restrict__ workspace,
	uint64_t workspace_stride,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	T* s_bsig = use_shared_storage
		? reinterpret_cast<T*>(smem)
		: workspace + local_batch_idx * workspace_stride;
	T* s_derivs = s_bsig + total_len;
	T* temp_Y = s_derivs + total_len;
	T* local_derivs = temp_Y + total_len;
	T* inc = local_derivs + total_len;
	T* inc_derivs = inc + dim;
	const uint32_t num_warps = blockDim.x >> 5;
	T* inc_reduction = inc_derivs + dim;
	const bool has_correction = correction_len != 0;
	T* product_x_reduction = inc_reduction
		+ (use_inc_reduction ? static_cast<uint32_t>(dim) * num_warps : 0);
	T* product_y_reduction = product_x_reduction + num_warps * deriv_target_count;
	T* local_log = use_product_reduction
		? product_y_reduction + num_warps * deriv_target_count
		: product_x_reduction;
	T* powers = has_correction ? local_log + total_len : local_log;
	const uint32_t power_count = use_shared_storage
		? static_cast<uint32_t>(max_nodes) : 2u;
	T* power_derivs = has_correction ? powers + power_count * total_len : powers;
	T* d_correction = has_correction
		? power_derivs + power_count * total_len : power_derivs;
	T* local_log_end = has_correction ? d_correction + total_len : local_log;
	uint32_t* shared_coprod_data = reinterpret_cast<uint32_t*>(local_log_end);
	uint32_t* shared_coprod_off = shared_coprod_data + coprod_data_len;
	uint32_t* shared_order_idx = shared_coprod_off + num_trees + 1;
	const uint32_t* s_coprod_data = g_coprod_data;
	const uint32_t* s_coprod_off = g_coprod_offsets;
	const uint32_t* s_order_idx = g_order_index;

	// --- One-time loads ---
	const uint64_t batch_off = static_cast<uint64_t>(batch_idx) * total_len;
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		s_bsig[i] = bsig_in[batch_off + i];
		s_derivs[i] = bsig_derivs_in[batch_off + i];
	}
	if constexpr (use_shared_storage) {
		for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
			shared_coprod_data[i] = g_coprod_data[i];
		for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
			shared_coprod_off[i] = g_coprod_offsets[i];
		for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2);
			i += blockDim.x)
			shared_order_idx[i] = g_order_index[i];
		s_coprod_data = shared_coprod_data;
		s_coprod_off = shared_coprod_off;
		s_order_idx = shared_order_idx;
	}

	// Initialize path_derivs to zero
	const T* bp = path + static_cast<uint64_t>(batch_idx) * path_stride;
	T* pd = path_derivs + static_cast<uint64_t>(batch_idx) * path_stride;
	const T* batch_correction = has_correction
		? correction + static_cast<uint64_t>(batch_idx) * correction_batch_stride
		: nullptr;
	for (uint32_t i = tid; i < static_cast<uint32_t>(path_stride); i += blockDim.x)
		pd[i] = T(0);
	__syncthreads();

	for (int seg = steps - 1; seg >= 0; --seg) {
		// --- 1. Load increment ---
		for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
			inc[d] = bp[(seg + 1) * dim + d] - bp[seg * dim + d];
		__syncthreads();

		const T* seg_correction = has_correction
			? batch_correction + static_cast<uint64_t>(seg) * correction_segment_stride
			: nullptr;
		local_branched_sig_block_(inc, temp_Y, local_log, powers,
			powers + total_len,
			dim, data_dim, seg_correction, correction_len, total_len,
			labels_data, labels_offsets, inv_factorial,
			s_coprod_data, s_coprod_off, s_order_idx,
			chain_index_offsets, chain_indices, max_nodes, tid);

		// --- 3. Butcher uncombine (order 1 to max_nodes) ---
		if (seg > 0) {
			for (int order = 1; order <= max_nodes; ++order) {
				const uint32_t ostart = s_order_idx[order];
				const uint32_t oend = s_order_idx[order + 1];
				for (uint32_t tree = ostart + tid; tree < oend;
					tree += blockDim.x) {
					uint32_t fi = tree + 1;
					T val = s_bsig[fi] - temp_Y[fi];
					uint32_t pos = s_coprod_off[tree];
					uint32_t pend = s_coprod_off[tree + 1];
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
			if (use_product_reduction) {
				const uint32_t reduction_len = 2 * num_warps * deriv_target_count;
				for (uint32_t i = tid; i < reduction_len; i += blockDim.x)
					product_x_reduction[i] = T(0);
			}
			__syncthreads();

			for (uint32_t tree = tid; tree < num_trees;
				tree += blockDim.x) {
				const T dF_tau = s_derivs[tree + 1];
				if (dF_tau == T(0))
					continue;
				uint32_t pos = s_coprod_off[tree];
				uint32_t pend = s_coprod_off[tree + 1];
				while (pos < pend) {
					uint32_t nf = s_coprod_data[pos++];
					uint32_t trunk_flat = s_coprod_data[pos++];
					uint32_t forest_start = pos;

					T forest_product = T(1);
					#pragma unroll 4
					for (uint32_t j = 0; j < nf; ++j)
						forest_product *= s_bsig[s_coprod_data[pos++]];

					T* d_Y = use_product_reduction
						? product_y_reduction + (threadIdx.x >> 5) * deriv_target_count
						: local_derivs;
					myAtomicAdd(&d_Y[trunk_flat], dF_tau * forest_product);

					if (nf > 0) {
						T base = dF_tau * temp_Y[trunk_flat];
						T* d_X = use_product_reduction
							? product_x_reduction + (threadIdx.x >> 5) * deriv_target_count
							: s_derivs;
						for (uint32_t k = 0; k < nf; ++k) {
							uint32_t fk = s_coprod_data[forest_start + k];
							T partial = base;
							for (uint32_t j = 0; j < nf; ++j) {
								if (j != k)
									partial *= s_bsig[s_coprod_data[forest_start + j]];
							}
							myAtomicAdd(&d_X[fk], partial);
						}
					}
				}
			}
			__syncthreads();
			if (use_product_reduction) {
				for (uint32_t target = tid; target < deriv_target_count; target += blockDim.x) {
					T d_X = T(0);
					T d_Y = T(0);
					for (uint32_t warp = 0; warp < num_warps; ++warp) {
						d_X += product_x_reduction[warp * deriv_target_count + target];
						d_Y += product_y_reduction[warp * deriv_target_count + target];
					}
					s_derivs[target] += d_X;
					local_derivs[target] += d_Y;
				}
				__syncthreads();
			}
		}
		else {
			// seg == 0: local_derivs = s_derivs
			for (uint32_t i = tid; i < total_len; i += blockDim.x)
				local_derivs[i] = s_derivs[i];
			__syncthreads();
		}

		if (!has_correction) {
			linear_bsig_deriv_to_inc_block_(local_derivs, inc, inc_derivs,
				inc_reduction, use_inc_reduction, dim,
				total_len, labels_data, labels_offsets, inv_factorial, tid);
		} else {
			if constexpr (use_shared_storage) {
				local_log_bsig_deriv_to_inc_saved_block_(
					local_derivs, inc, inc_derivs, inc_reduction,
					use_inc_reduction, local_log, powers, power_derivs,
					d_correction, dim, data_dim, seg_correction, correction_len,
					total_len, labels_data, labels_offsets, inv_factorial,
					s_coprod_data, s_coprod_off, s_order_idx,
					chain_index_offsets, chain_indices, max_nodes, tid);
			} else {
				local_log_bsig_deriv_to_inc_rolling_block_(
					local_derivs, inc, inc_derivs, inc_reduction,
					use_inc_reduction, local_log, powers, power_derivs,
					d_correction, dim, data_dim, seg_correction, correction_len,
					total_len, labels_data, labels_offsets, inv_factorial,
					s_coprod_data, s_coprod_off, s_order_idx,
					chain_index_offsets, chain_indices, max_nodes, tid);
			}
		}

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

template<typename T, bool use_shared_tables>
__global__ __launch_bounds__(use_shared_tables ? 1024 : 256)
void branched_sig_combine_ker(
	const T* __restrict__ bsig1,
	const T* __restrict__ bsig2,
	T* __restrict__ out,
	uint32_t total_len,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	const uint32_t* __restrict__ g_order_index,
	int max_nodes,
	uint32_t coprod_data_len,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	uint32_t* shared_coprod_data = reinterpret_cast<uint32_t*>(smem);
	uint32_t* shared_coprod_off = shared_coprod_data + coprod_data_len;
	uint32_t* shared_order_idx = shared_coprod_off + num_trees + 1;
	const uint32_t* coprod_data = g_coprod_data;
	const uint32_t* coprod_off = g_coprod_offsets;
	const uint32_t* order_idx = g_order_index;
	if constexpr (use_shared_tables) {
		for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
			shared_coprod_data[i] = g_coprod_data[i];
		for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
			shared_coprod_off[i] = g_coprod_offsets[i];
		for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2);
			i += blockDim.x)
			shared_order_idx[i] = g_order_index[i];
		coprod_data = shared_coprod_data;
		coprod_off = shared_coprod_off;
		order_idx = shared_order_idx;
	}
	__syncthreads();

	const uint64_t off = static_cast<uint64_t>(batch_idx) * total_len;
	const T* X = bsig1 + off;
	const T* Y = bsig2 + off;
	T* O = out + off;

	// Copy X to out
	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		O[i] = X[i];
	__syncthreads();

	// Butcher product: out = butcher_product(X, Y) - process high to low order
	for (int order = max_nodes; order >= 1; --order) {
		const uint32_t ostart = order_idx[order];
		const uint32_t oend = order_idx[order + 1];
		for (uint32_t tree = ostart + tid; tree < oend;
			tree += blockDim.x) {
			uint32_t fi = tree + 1;
			T val = O[fi] + Y[fi];
			uint32_t pos = coprod_off[tree];
			uint32_t pend = coprod_off[tree + 1];
			while (pos < pend) {
				uint32_t nf = coprod_data[pos++];
				T term = Y[coprod_data[pos++]];
				#pragma unroll 4
				for (uint32_t j = 0; j < nf; ++j)
					term *= O[coprod_data[pos++]];
				val += term;
			}
			O[fi] = val;
		}
		__syncthreads();
	}
}

// Combine backprop kernel: given dF/d(out), compute dF/d(bsig1) and dF/d(bsig2)
template<typename T, bool use_shared_tables>
__global__ __launch_bounds__(use_shared_tables ? 1024 : 256)
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
	uint32_t coprod_data_len,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	uint32_t* shared_coprod_data = reinterpret_cast<uint32_t*>(smem);
	uint32_t* shared_coprod_off = shared_coprod_data + coprod_data_len;
	uint32_t* shared_order_idx = shared_coprod_off + num_trees + 1;
	const uint32_t* coprod_data = g_coprod_data;
	const uint32_t* coprod_off = g_coprod_offsets;
	if constexpr (use_shared_tables) {
		for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
			shared_coprod_data[i] = g_coprod_data[i];
		for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
			shared_coprod_off[i] = g_coprod_offsets[i];
		for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes + 2);
			i += blockDim.x)
			shared_order_idx[i] = g_order_index[i];
		coprod_data = shared_coprod_data;
		coprod_off = shared_coprod_off;
	}
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
	for (uint32_t tree = tid; tree < num_trees; tree += blockDim.x) {
		uint32_t fi = tree + 1;
		T dF_tau = derivs[off + fi];
		if (dF_tau != T(0)) {
			uint32_t pos = coprod_off[tree];
			uint32_t pend = coprod_off[tree + 1];
			while (pos < pend) {
				uint32_t nf = coprod_data[pos++];
				uint32_t trunk_flat = coprod_data[pos++];
				uint32_t forest_start = pos;

				T forest_product = T(1);
				#pragma unroll 4
				for (uint32_t j = 0; j < nf; ++j)
					forest_product *= X[coprod_data[pos++]];

				myAtomicAdd(&dY[trunk_flat], dF_tau * forest_product);

				if (nf > 0) {
					T base = dF_tau * Y[trunk_flat];
					for (uint32_t k = 0; k < nf; ++k) {
						uint32_t fk = coprod_data[forest_start + k];
						T partial = base;
						for (uint32_t j = 0; j < nf; ++j) {
							if (j != k)
								partial *= X[coprod_data[forest_start + j]];
						}
						myAtomicAdd(&dX[fk], partial);
					}
				}
			}
		}
	}
}

// =========================================================================
// scalar_term=false staging helpers (zero-overhead when scalar_term=true)
// =========================================================================

template<typename T>
__global__ void bsig_prepend_one_kernel(
	const T* __restrict__ in_stripped,
	T* __restrict__ out_full,
	uint64_t full_len,
	uint64_t total
) {
	for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		flat < total; flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t b = flat / full_len;
		const uint64_t i = flat - b * full_len;
		if (i == 0) out_full[flat] = static_cast<T>(1);
		else out_full[flat] = in_stripped[b * (full_len - 1) + (i - 1)];
	}
}

template<typename T>
__global__ void bsig_prepend_zero_kernel(
	const T* __restrict__ in_stripped,
	T* __restrict__ out_full,
	uint64_t full_len,
	uint64_t total
) {
	for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		flat < total; flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t b = flat / full_len;
		const uint64_t i = flat - b * full_len;
		if (i == 0) out_full[flat] = static_cast<T>(0);
		else out_full[flat] = in_stripped[b * (full_len - 1) + (i - 1)];
	}
}

template<typename T>
__global__ void bsig_strip_kernel(
	const T* __restrict__ in_full,
	T* __restrict__ out_stripped,
	uint64_t full_len,
	uint64_t total
) {
	const uint64_t stripped_len = full_len - 1;
	for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		flat < total; flat += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t b = flat / stripped_len;
		const uint64_t i = flat - b * stripped_len;
		out_stripped[flat] = in_full[b * full_len + i + 1];
	}
}

template<typename T>
static void bsig_stage_prepend_one_(const T* in_stripped, T* out_full, uint64_t batch_size, uint64_t full_len) {
	const unsigned int block = 256;
	const uint64_t total = batch_size * full_len;
	const unsigned int grid = static_cast<unsigned int>(std::min<uint64_t>(
		(total + block - 1) / block, CUDA_BATCH_GRID_LIMIT));
	bsig_prepend_one_kernel<T><<<grid, block>>>(in_stripped, out_full, full_len, total);
	check_cuda_error();
}

template<typename T>
static void bsig_stage_prepend_zero_(const T* in_stripped, T* out_full, uint64_t batch_size, uint64_t full_len) {
	const unsigned int block = 256;
	const uint64_t total = batch_size * full_len;
	const unsigned int grid = static_cast<unsigned int>(std::min<uint64_t>(
		(total + block - 1) / block, CUDA_BATCH_GRID_LIMIT));
	bsig_prepend_zero_kernel<T><<<grid, block>>>(in_stripped, out_full, full_len, total);
	check_cuda_error();
}

template<typename T>
static void bsig_stage_strip_(const T* in_full, T* out_stripped, uint64_t batch_size, uint64_t full_len) {
	if (full_len == 1) return;
	const unsigned int block = 256;
	const uint64_t total = batch_size * (full_len - 1);
	const unsigned int grid = static_cast<unsigned int>(std::min<uint64_t>(
		(total + block - 1) / block, CUDA_BATCH_GRID_LIMIT));
	bsig_strip_kernel<T><<<grid, block>>>(in_full, out_stripped, full_len, total);
	check_cuda_error();
}

// =========================================================================
// Host-side launchers
// =========================================================================

// Forward declarations for the _core_ functions defined below (scalar_term=true bodies).
template<typename T>
void branched_sig_combine_cuda_core_(
	const T* bsig1, const T* bsig2, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes,
	bool planar
);
template<typename T>
void branched_sig_combine_backprop_cuda_core_(
	const T* bsig1, const T* bsig2, const T* derivs, T* out1, T* out2,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes,
	bool planar
);

template<typename T>
void branched_sig_combine_cuda_(
	const T* bsig1, const T* bsig2, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes,
	bool planar = false,
	bool scalar_term = true
) {
	if (batch_size == 0)
		return;
	if (scalar_term) {
		branched_sig_combine_cuda_core_<T>(bsig1, bsig2, out, batch_size, dimension, max_nodes, planar);
		return;
	}
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);
	const uint64_t full_len = gc.total_length;
	const size_t workspace_row = checked_cuda_size_mul(
		3, static_cast<size_t>(full_len),
		"CUDA branched sig combine scalar staging");
	CudaBatchWorkspace<T> workspace(
		batch_size, workspace_row,
		"CUDA branched sig combine scalar staging");
	T* d_b1 = workspace.get();
	T* d_b2 = d_b1 + workspace.capacity() * full_len;
	T* d_out = d_b2 + workspace.capacity() * full_len;
	const uint64_t stripped_len = full_len - 1;
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, workspace.capacity());
		bsig_stage_prepend_one_<T>(
			bsig1 + chunk.offset * stripped_len, d_b1, chunk.size, full_len);
		bsig_stage_prepend_one_<T>(
			bsig2 + chunk.offset * stripped_len, d_b2, chunk.size, full_len);
		branched_sig_combine_cuda_core_<T>(
			d_b1, d_b2, d_out, chunk.size, dimension, max_nodes, planar);
		bsig_stage_strip_<T>(
			d_out, out + chunk.offset * stripped_len, chunk.size, full_len);
		batch_offset += chunk.size;
	}
}

template<typename T>
void branched_sig_combine_backprop_cuda_(
	const T* bsig1, const T* bsig2, const T* derivs, T* out1, T* out2,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes,
	bool planar = false,
	bool scalar_term = true
) {
	if (batch_size == 0)
		return;
	if (scalar_term) {
		branched_sig_combine_backprop_cuda_core_<T>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, planar);
		return;
	}
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);
	const uint64_t full_len = gc.total_length;
	const size_t workspace_row = checked_cuda_size_mul(
		5, static_cast<size_t>(full_len),
		"CUDA branched sig combine backprop scalar staging");
	CudaBatchWorkspace<T> workspace(
		batch_size, workspace_row,
		"CUDA branched sig combine backprop scalar staging");
	T* d_b1 = workspace.get();
	T* d_b2 = d_b1 + workspace.capacity() * full_len;
	T* d_der = d_b2 + workspace.capacity() * full_len;
	T* d_o1 = d_der + workspace.capacity() * full_len;
	T* d_o2 = d_o1 + workspace.capacity() * full_len;
	const uint64_t stripped_len = full_len - 1;
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, workspace.capacity());
		bsig_stage_prepend_one_<T>(
			bsig1 + chunk.offset * stripped_len, d_b1, chunk.size, full_len);
		bsig_stage_prepend_one_<T>(
			bsig2 + chunk.offset * stripped_len, d_b2, chunk.size, full_len);
		bsig_stage_prepend_zero_<T>(
			derivs + chunk.offset * stripped_len, d_der, chunk.size, full_len);
		branched_sig_combine_backprop_cuda_core_<T>(
			d_b1, d_b2, d_der, d_o1, d_o2, chunk.size,
			dimension, max_nodes, planar);
		bsig_stage_strip_<T>(
			d_o1, out1 + chunk.offset * stripped_len, chunk.size, full_len);
		bsig_stage_strip_<T>(
			d_o2, out2 + chunk.offset * stripped_len, chunk.size, full_len);
		batch_offset += chunk.size;
	}
}

template<typename T>
void branched_sig_combine_cuda_core_(
	const T* bsig1, const T* bsig2, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes,
	bool planar
) {
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;
	const size_t table_entries = checked_cuda_size_add(
		static_cast<size_t>(gc.coprod_data_len),
		checked_cuda_size_add(
			static_cast<size_t>(gc.num_trees) + 1,
			static_cast<size_t>(gc.max_nodes) + 2,
			"CUDA branched sig combine"),
		"CUDA branched sig combine");
	const size_t smem = checked_cuda_size_mul(
		table_entries, sizeof(uint32_t), "CUDA branched sig combine");
	const bool use_shared_tables = try_configure_dynamic_smem(
		branched_sig_combine_ker<T, true>, smem);
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		if (use_shared_tables) {
			branched_sig_combine_ker<T, true><<<batch_chunk.grid, block, smem>>>(
				bsig1, bsig2, out, gc.total_length,
				gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
				gc.max_nodes, gc.coprod_data_len,
				batch_chunk.offset, batch_chunk.size);
		} else {
			branched_sig_combine_ker<T, false><<<
				batch_chunk.grid, std::min(block, 256u)>>>(
				bsig1, bsig2, out, gc.total_length,
				gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
				gc.max_nodes, gc.coprod_data_len,
				batch_chunk.offset, batch_chunk.size);
		}
		batch_offset += batch_chunk.size;
	}
	cudaDeviceSynchronize();
	check_cuda_error();
}

template<typename T>
void branched_sig_combine_backprop_cuda_core_(
	const T* bsig1, const T* bsig2, const T* derivs, T* out1, T* out2,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes,
	bool planar
) {
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;
	const size_t table_entries = checked_cuda_size_add(
		static_cast<size_t>(gc.coprod_data_len),
		checked_cuda_size_add(
			static_cast<size_t>(gc.num_trees) + 1,
			static_cast<size_t>(gc.max_nodes) + 2,
			"CUDA branched sig combine backprop"),
		"CUDA branched sig combine backprop");
	const size_t smem = checked_cuda_size_mul(
		table_entries, sizeof(uint32_t),
		"CUDA branched sig combine backprop");
	const bool use_shared_tables = try_configure_dynamic_smem(
		branched_sig_combine_backprop_ker<T, true>, smem);
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		if (use_shared_tables) {
			branched_sig_combine_backprop_ker<T, true><<<
				batch_chunk.grid, block, smem>>>(
				bsig1, bsig2, derivs, out1, out2, gc.total_length,
				gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
				gc.max_nodes, gc.coprod_data_len,
				batch_chunk.offset, batch_chunk.size);
		} else {
			branched_sig_combine_backprop_ker<T, false><<<
				batch_chunk.grid, std::min(block, 256u)>>>(
				bsig1, bsig2, derivs, out1, out2, gc.total_length,
				gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
				gc.max_nodes, gc.coprod_data_len,
				batch_chunk.offset, batch_chunk.size);
		}
		batch_offset += batch_chunk.size;
	}
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
	uint64_t max_nodes,
	bool planar = false,
	uint64_t data_dimension = 0,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	if (data_dimension == 0) data_dimension = dimension;
	validate_correction_len_(data_dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	const bool has_correction = correction_len != 0;
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);

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
	const size_t path_stride = checked_cuda_size_mul(
		static_cast<size_t>(length), static_cast<size_t>(dimension),
		"CUDA branched sig");

	const bool use_shared_state = !has_correction && planar;
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	const size_t state_arrays = has_correction ? 4 : (use_shared_state ? 3 : 1);
	const size_t t_arrays = checked_cuda_size_add(
		checked_cuda_size_mul(
			state_arrays, static_cast<size_t>(gc.total_length),
			"CUDA branched sig"),
		static_cast<size_t>(dimension), "CUDA branched sig");
	const size_t table_entries = checked_cuda_size_add(
		static_cast<size_t>(gc.coprod_data_len),
		checked_cuda_size_add(
			static_cast<size_t>(gc.num_trees) + 1,
			static_cast<size_t>(gc.max_nodes) + 2,
			"CUDA branched sig"),
		"CUDA branched sig");
	const size_t smem = checked_cuda_size_add(
		checked_cuda_size_mul(t_arrays, sizeof(T), "CUDA branched sig"),
		checked_cuda_size_mul(
			table_entries, sizeof(uint32_t), "CUDA branched sig"),
		"CUDA branched sig");

	// Select float or double inv_factorial
	const T* d_inv_fact;
	if constexpr (std::is_same_v<T, float>)
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f32);
	else
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f64);

	const bool use_shared_storage = try_configure_dynamic_smem(
		branched_sig_ker<T, true>, smem);
	std::unique_ptr<CudaBatchWorkspace<T>> workspace;
	if (!use_shared_storage) {
		workspace = std::make_unique<CudaBatchWorkspace<T>>(
			batch_size, t_arrays, "CUDA branched sig global workspace");
	}
	const uint64_t max_chunk_size = use_shared_storage
		? CUDA_BATCH_GRID_CAPACITY : workspace->capacity();
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, max_chunk_size);
		if (use_shared_storage) {
			branched_sig_ker<T, true><<<batch_chunk.grid, block, smem>>>(
				path, out, static_cast<int>(dimension),
				static_cast<int>(data_dimension), steps,
				gc.total_length, path_stride,
				gc.d_labels_data, gc.d_labels_offsets32, d_inv_fact,
				gc.d_coprod_data32, gc.d_coprod_offsets32,
				gc.d_order_index32, gc.d_chain_index_offsets32,
				gc.d_chain_indices32, gc.max_nodes, gc.coprod_data_len,
				correction, correction_len, correction_batch_stride,
				correction_segment_stride, use_shared_state, nullptr, 0,
				batch_chunk.offset, batch_chunk.size);
		} else {
			branched_sig_ker<T, false><<<
				batch_chunk.grid, std::min(block, 256u)>>>(
				path, out, static_cast<int>(dimension),
				static_cast<int>(data_dimension), steps,
				gc.total_length, path_stride,
				gc.d_labels_data, gc.d_labels_offsets32, d_inv_fact,
				gc.d_coprod_data32, gc.d_coprod_offsets32,
				gc.d_order_index32, gc.d_chain_index_offsets32,
				gc.d_chain_indices32, gc.max_nodes, gc.coprod_data_len,
				correction, correction_len, correction_batch_stride,
				correction_segment_stride, use_shared_state, workspace->get(),
				t_arrays, batch_chunk.offset, batch_chunk.size);
		}
		batch_offset += batch_chunk.size;
	}
	cudaDeviceSynchronize();
	check_cuda_error();
}

// =========================================================================
// time_aug / lead_lag wrapper
// =========================================================================

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
	T end_time,
	bool planar = false,
	bool scalar_term = true,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");
	if (batch_size == 0)
		return;
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;
	const auto& gc = get_gpu_cache(t_dimension, max_nodes, planar);
	const uint64_t full_len = gc.total_length;
	if (scalar_term && !time_aug && !lead_lag) {
		branched_sig_cuda_core_<T>(path, out, batch_size, dimension, length,
			max_nodes, planar, dimension, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
		return;
	}

	const size_t path_stride = checked_cuda_size_mul(
		static_cast<size_t>(length), static_cast<size_t>(dimension),
		"CUDA branched sig path staging");
	const size_t transformed_stride = time_aug || lead_lag
		? checked_cuda_size_mul(
			static_cast<size_t>(t_length), static_cast<size_t>(t_dimension),
			"CUDA branched sig path staging")
		: 0;
	const size_t output_staging = scalar_term ? 0 : static_cast<size_t>(full_len);
	const size_t workspace_row = checked_cuda_size_add(
		transformed_stride, output_staging,
		"CUDA branched sig path staging");
	CudaBatchWorkspace<T> workspace(
		batch_size, workspace_row, "CUDA branched sig path staging");
	T* transformed = workspace.get();
	T* staged_out = transformed
		+ workspace.capacity() * transformed_stride;
	const uint64_t output_stride = scalar_term ? full_len : full_len - 1;
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, workspace.capacity());
		const T* core_path = path + chunk.offset * path_stride;
		if (time_aug || lead_lag) {
			cu_transform_path_<T>(
				core_path, transformed, chunk.size, dimension, length,
				time_aug, lead_lag, end_time);
			core_path = transformed;
		}
		T* core_out = scalar_term
			? out + chunk.offset * output_stride : staged_out;
		const T* chunk_correction = correction_len == 0 ? nullptr
			: correction + chunk.offset * correction_batch_stride;
		branched_sig_cuda_core_<T>(
			core_path, core_out, chunk.size, t_dimension, t_length,
			max_nodes, planar, dimension, chunk_correction, correction_len,
			correction_batch_stride, correction_segment_stride);
		if (!scalar_term) {
			bsig_stage_strip_<T>(
				staged_out, out + chunk.offset * output_stride,
				chunk.size, full_len);
		}
		batch_offset += chunk.size;
	}
}

template<typename T>
void launch_branched_sig_coef_forward_(
	const T* path,
	T* state,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	const BranchedSigCoefCacheGPU& cache,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride,
	uint64_t batch_offset_start = 0,
	uint64_t batch_count = 0,
	uint64_t state_stride = 0,
	T* external_workspace = nullptr,
	uint64_t external_workspace_stride = 0
) {
	if (length > static_cast<uint64_t>(UINT32_MAX) + 1)
		throw std::invalid_argument("CUDA branched sig coef path length exceeds kernel range");
	if (batch_count == 0)
		batch_count = batch_size - batch_offset_start;
	if (state_stride == 0)
		state_stride = cache.cache_size;

	const uint32_t num_non_scalar = cache.cache_size - 1;
	unsigned int block = (num_non_scalar + 31u) & ~31u;
	if (block < 32)
		block = 32;
	if (block > 1024)
		block = 1024;

	const uint32_t num_corrections = branched_sig_coef_num_corrections_(cache, correction_len);
	const bool has_correction = num_corrections != 0;
	const size_t t_arrays = checked_cuda_size_add(
		checked_cuda_size_mul(
			has_correction ? 4 : 1,
			static_cast<size_t>(cache.cache_size),
			"CUDA branched sig coef"),
		static_cast<size_t>(dimension), "CUDA branched sig coef");
	const size_t state_arrays = state == nullptr ? cache.cache_size : 0;
	const size_t numeric_arrays = checked_cuda_size_add(
		t_arrays, state_arrays, "CUDA branched sig coef");
	const size_t table_entries = checked_cuda_size_add(
		static_cast<size_t>(cache.coprod_data_len),
		checked_cuda_size_add(
			static_cast<size_t>(cache.cache_size) + 1,
			static_cast<size_t>(cache.max_nodes) + 2,
			"CUDA branched sig coef"),
		"CUDA branched sig coef");
	const size_t smem = checked_cuda_size_add(
		checked_cuda_size_mul(
			numeric_arrays, sizeof(T), "CUDA branched sig coef"),
		checked_cuda_size_mul(
			table_entries, sizeof(uint32_t), "CUDA branched sig coef"),
		"CUDA branched sig coef");
	const T* d_inv_factorial;
	if constexpr (std::is_same_v<T, float>)
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f32);
	else
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f64);

	const bool use_shared_storage = try_configure_dynamic_smem(
		branched_sig_coef_forward_kernel_<T, true>, smem);
	std::unique_ptr<CudaBatchWorkspace<T>> owned_workspace;
	if (!use_shared_storage && external_workspace == nullptr) {
		owned_workspace = std::make_unique<CudaBatchWorkspace<T>>(
			batch_count, numeric_arrays,
			"CUDA branched sig coef global workspace");
		external_workspace = owned_workspace->get();
		external_workspace_stride = numeric_arrays;
	}
	const uint64_t max_chunk_size = use_shared_storage
		? CUDA_BATCH_GRID_CAPACITY
		: (owned_workspace == nullptr
			? batch_count : owned_workspace->capacity());
	uint64_t local_offset = 0;
	for (uint64_t batch_offset = batch_offset_start;
		batch_offset < batch_offset_start + batch_count;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset,
			std::min<uint64_t>(max_chunk_size,
				batch_offset_start + batch_count - batch_offset));
		T* chunk_state = state == nullptr
			? nullptr : state + local_offset * state_stride;
		T* chunk_workspace = external_workspace == nullptr
			? nullptr
			: external_workspace + (owned_workspace == nullptr
				? local_offset * external_workspace_stride : 0);
		if (use_shared_storage) {
			branched_sig_coef_forward_kernel_<T, true><<<
				batch_chunk.grid, block, smem>>>(
				path, chunk_state, out, static_cast<uint32_t>(dimension),
				length == 0 ? 0 : static_cast<uint32_t>(length - 1),
				length * dimension, cache.cache_size, cache.d_target_indices,
				cache.num_targets, cache.d_labels_data, cache.d_labels_offsets,
				d_inv_factorial, cache.d_coprod_data, cache.d_coprod_offsets,
				cache.d_order_index, cache.d_leaf_indices,
				cache.d_correction_offsets, cache.d_correction_locals,
				num_corrections, cache.max_nodes, cache.coprod_data_len,
				correction, correction_batch_stride, correction_segment_stride,
				state_stride, nullptr, 0, batch_chunk.offset, batch_chunk.size);
		} else {
			branched_sig_coef_forward_kernel_<T, false><<<
				batch_chunk.grid, std::min(block, 256u)>>>(
				path, chunk_state, out, static_cast<uint32_t>(dimension),
				length == 0 ? 0 : static_cast<uint32_t>(length - 1),
				length * dimension, cache.cache_size, cache.d_target_indices,
				cache.num_targets, cache.d_labels_data, cache.d_labels_offsets,
				d_inv_factorial, cache.d_coprod_data, cache.d_coprod_offsets,
				cache.d_order_index, cache.d_leaf_indices,
				cache.d_correction_offsets, cache.d_correction_locals,
				num_corrections, cache.max_nodes, cache.coprod_data_len,
				correction, correction_batch_stride, correction_segment_stride,
				state_stride, chunk_workspace, external_workspace_stride,
				batch_chunk.offset, batch_chunk.size);
		}
		batch_offset += batch_chunk.size;
		local_offset += batch_chunk.size;
	}
}

template<typename T>
void branched_sig_coef_cuda_core_(
	const T* path,
	T* out,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	uint64_t data_dimension,
	bool planar,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	validate_correction_len_(data_dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	const auto& cache = get_branched_sig_coef_cache_cuda_(
		tree_data, tree_data_len, data_dimension, dimension, max_nodes, planar);
	if (batch_size == 0)
		return;

	launch_branched_sig_coef_forward_(path, static_cast<T*>(nullptr), out, batch_size,
		dimension, length, cache, correction, correction_len, correction_batch_stride,
		correction_segment_stride);
	cudaDeviceSynchronize();
	check_cuda_error();
}

template<typename T>
void branched_sig_coef_cuda_(
	const T* path,
	T* out,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	bool time_aug,
	bool lead_lag,
	T end_time,
	bool planar,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	if (dimension == 0)
		throw std::invalid_argument("branched_sig_coef_cuda received path of dimension 0");
	if (lead_lag && length == 0)
		throw std::invalid_argument("lead_lag requires a path with at least one point");
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");

	const uint64_t transformed_dimension
		= (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t transformed_length = lead_lag ? 2 * length - 1 : length;
	if (batch_size == 0) {
		branched_sig_coef_cuda_core_(path, out, tree_data, tree_data_len, batch_size,
			transformed_dimension, transformed_length, max_nodes, dimension, planar,
			correction, correction_len, correction_batch_stride,
			correction_segment_stride);
		return;
	}

	if (time_aug || lead_lag) {
		const size_t path_stride = checked_cuda_size_mul(
			static_cast<size_t>(length), static_cast<size_t>(dimension),
			"CUDA branched sig coef transformed path");
		const size_t transformed_stride = checked_cuda_size_mul(
			static_cast<size_t>(transformed_length),
			static_cast<size_t>(transformed_dimension),
			"CUDA branched sig coef transformed path");
		const auto& cache = get_branched_sig_coef_cache_cuda_(
			tree_data, tree_data_len, dimension, transformed_dimension,
			max_nodes, planar);
		const uint64_t output_stride = cache.num_targets;
		CudaBatchWorkspace<T> transformed(
			batch_size, transformed_stride,
			"CUDA branched sig coef transformed path");
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto chunk = make_cuda_batch_grid_chunk(
				1, batch_size, batch_offset, transformed.capacity());
			cu_transform_path_<T>(
				path + chunk.offset * path_stride, transformed.get(), chunk.size,
				dimension, length, time_aug, lead_lag, end_time);
			const T* chunk_correction = correction_len == 0 ? nullptr
				: correction + chunk.offset * correction_batch_stride;
			branched_sig_coef_cuda_core_(
				transformed.get(), out + chunk.offset * output_stride,
				tree_data, tree_data_len, chunk.size, transformed_dimension,
				transformed_length, max_nodes, dimension, planar,
				chunk_correction, correction_len, correction_batch_stride,
				correction_segment_stride);
			batch_offset += chunk.size;
		}
	}
	else {
		branched_sig_coef_cuda_core_(path, out, tree_data, tree_data_len, batch_size,
			dimension, length, max_nodes, dimension, planar, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
	}
}

template<typename T>
void branched_sig_coef_backprop_cuda_core_(
	const T* path,
	T* out,
	const T* coefs,
	const T* derivs,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	uint64_t data_dimension,
	bool planar,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	validate_correction_len_(data_dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	const auto& cache = get_branched_sig_coef_cache_cuda_(
		tree_data, tree_data_len, data_dimension, dimension, max_nodes, planar);
	if (batch_size == 0)
		return;
	if (length <= 1) {
		CUDA_CHECK(cudaMemset(out, 0, batch_size * length * dimension * sizeof(T)));
		cudaDeviceSynchronize();
		check_cuda_error();
		return;
	}

	const uint32_t num_non_scalar = cache.cache_size - 1;
	unsigned int block = (num_non_scalar + 31u) & ~31u;
	if (block < 32)
		block = 32;
	if (block > 1024)
		block = 1024;
	const uint32_t num_corrections = branched_sig_coef_num_corrections_(cache, correction_len);
	const bool has_correction = num_corrections != 0;
	const size_t global_basis_arrays = has_correction ? 10 : 4;
	const size_t shared_basis_arrays = has_correction
		? checked_cuda_size_add(
			6, checked_cuda_size_mul(
				2, static_cast<size_t>(max_nodes),
				"CUDA branched sig coef backprop"),
			"CUDA branched sig coef backprop")
		: 4;
	const size_t global_backprop_arrays = checked_cuda_size_add(
		checked_cuda_size_mul(
			global_basis_arrays, static_cast<size_t>(cache.cache_size),
			"CUDA branched sig coef backprop"),
		checked_cuda_size_mul(
			2, static_cast<size_t>(dimension),
			"CUDA branched sig coef backprop"),
		"CUDA branched sig coef backprop");
	const size_t shared_backprop_arrays = checked_cuda_size_add(
		checked_cuda_size_mul(
			shared_basis_arrays, static_cast<size_t>(cache.cache_size),
			"CUDA branched sig coef backprop"),
		checked_cuda_size_mul(
			2, static_cast<size_t>(dimension),
			"CUDA branched sig coef backprop"),
		"CUDA branched sig coef backprop");
	const size_t forward_arrays = checked_cuda_size_add(
		checked_cuda_size_mul(
			has_correction ? 4 : 1,
			static_cast<size_t>(cache.cache_size),
			"CUDA branched sig coef backprop"),
		static_cast<size_t>(dimension),
		"CUDA branched sig coef backprop");
	const size_t table_entries = checked_cuda_size_add(
		static_cast<size_t>(cache.coprod_data_len),
		checked_cuda_size_add(
			static_cast<size_t>(cache.cache_size) + 1,
			static_cast<size_t>(cache.max_nodes) + 2,
			"CUDA branched sig coef backprop"),
		"CUDA branched sig coef backprop");
	const size_t table_bytes = checked_cuda_size_mul(
		table_entries, sizeof(uint32_t),
		"CUDA branched sig coef backprop");
	const size_t smem = checked_cuda_size_add(
		checked_cuda_size_mul(
			shared_backprop_arrays, sizeof(T),
			"CUDA branched sig coef backprop"),
		table_bytes, "CUDA branched sig coef backprop");
	const size_t forward_smem = checked_cuda_size_add(
		checked_cuda_size_mul(
			forward_arrays, sizeof(T),
			"CUDA branched sig coef backprop"),
		table_bytes, "CUDA branched sig coef backprop");
	const T* d_inv_factorial;
	if constexpr (std::is_same_v<T, float>)
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f32);
	else
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f64);

	const bool use_shared_forward = try_configure_dynamic_smem(
		branched_sig_coef_forward_kernel_<T, true>, forward_smem);
	const bool use_shared_backprop = try_configure_dynamic_smem(
		branched_sig_coef_backprop_kernel_<T, true>, smem);
	const size_t scratch_arrays = std::max(
		use_shared_forward ? size_t(0) : forward_arrays,
		use_shared_backprop ? size_t(0) : global_backprop_arrays);
	const size_t workspace_stride = checked_cuda_size_add(
		static_cast<size_t>(cache.cache_size), scratch_arrays,
		"CUDA branched sig coef backprop");
	CudaBatchWorkspace<T> workspace(
		batch_size, workspace_stride,
		"CUDA branched sig coef backprop workspace");
	T* state = workspace.get();
	T* scratch = state + cache.cache_size;
	const size_t path_stride = checked_cuda_size_mul(
		static_cast<size_t>(length), static_cast<size_t>(dimension),
		"CUDA branched sig coef backprop");

	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, workspace.capacity());
		launch_branched_sig_coef_forward_(
			path, state, static_cast<T*>(nullptr), batch_size, dimension, length,
			cache, correction, correction_len, correction_batch_stride,
			correction_segment_stride, batch_chunk.offset, batch_chunk.size,
			workspace_stride, use_shared_forward ? nullptr : scratch,
			workspace_stride);
		if (use_shared_backprop) {
			branched_sig_coef_backprop_kernel_<T, true><<<
				batch_chunk.grid, block, smem>>>(
				path, out, state, coefs, derivs,
				static_cast<uint32_t>(dimension),
				static_cast<uint32_t>(length - 1), path_stride,
				cache.cache_size, cache.d_target_indices, cache.num_targets,
				cache.d_labels_data, cache.d_labels_offsets, d_inv_factorial,
				cache.d_coprod_data, cache.d_coprod_offsets, cache.d_order_index,
				cache.d_leaf_indices, cache.d_correction_offsets,
				cache.d_correction_locals, num_corrections, cache.max_nodes,
				cache.coprod_data_len, correction, correction_batch_stride,
				correction_segment_stride, workspace_stride, nullptr, 0,
				batch_chunk.offset, batch_chunk.size);
		} else {
			branched_sig_coef_backprop_kernel_<T, false><<<
				batch_chunk.grid, std::min(block, 256u)>>>(
				path, out, state, coefs, derivs,
				static_cast<uint32_t>(dimension),
				static_cast<uint32_t>(length - 1), path_stride,
				cache.cache_size, cache.d_target_indices, cache.num_targets,
				cache.d_labels_data, cache.d_labels_offsets, d_inv_factorial,
				cache.d_coprod_data, cache.d_coprod_offsets, cache.d_order_index,
				cache.d_leaf_indices, cache.d_correction_offsets,
				cache.d_correction_locals, num_corrections, cache.max_nodes,
				cache.coprod_data_len, correction, correction_batch_stride,
				correction_segment_stride, workspace_stride, scratch,
				workspace_stride, batch_chunk.offset, batch_chunk.size);
		}
		batch_offset += batch_chunk.size;
	}
	cudaDeviceSynchronize();
	check_cuda_error();
}

template<typename T>
void branched_sig_coef_backprop_cuda_(
	const T* path,
	T* out,
	const T* coefs,
	const T* derivs,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	bool time_aug,
	bool lead_lag,
	T end_time,
	bool planar,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	if (dimension == 0)
		throw std::invalid_argument("branched_sig_coef_backprop_cuda received path of dimension 0");
	if (lead_lag && length == 0)
		throw std::invalid_argument("lead_lag requires a path with at least one point");
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");

	const uint64_t transformed_dimension
		= (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t transformed_length = lead_lag ? 2 * length - 1 : length;
	if (batch_size == 0) {
		branched_sig_coef_backprop_cuda_core_(path, out, coefs, derivs, tree_data,
			tree_data_len, batch_size, transformed_dimension, transformed_length,
			max_nodes, dimension, planar, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
		return;
	}

	if (time_aug || lead_lag) {
		const size_t path_stride = checked_cuda_size_mul(
			static_cast<size_t>(length), static_cast<size_t>(dimension),
			"CUDA branched sig coef backprop transformed path");
		const size_t transformed_stride = checked_cuda_size_mul(
			static_cast<size_t>(transformed_length),
			static_cast<size_t>(transformed_dimension),
			"CUDA branched sig coef backprop transformed path");
		const size_t workspace_row = checked_cuda_size_mul(
			2, transformed_stride,
			"CUDA branched sig coef backprop transformed path");
		const auto& cache = get_branched_sig_coef_cache_cuda_(
			tree_data, tree_data_len, dimension, transformed_dimension,
			max_nodes, planar);
		const uint64_t coef_stride = cache.num_targets;
		CudaBatchWorkspace<T> transformed(
			batch_size, workspace_row,
			"CUDA branched sig coef backprop transformed path");
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
			branched_sig_coef_backprop_cuda_core_(
				transformed.get(), transformed_derivs,
				coefs + chunk.offset * coef_stride,
				derivs + chunk.offset * coef_stride,
				tree_data, tree_data_len, chunk.size, transformed_dimension,
				transformed_length, max_nodes, dimension, planar,
				chunk_correction, correction_len, correction_batch_stride,
				correction_segment_stride);
			cu_transform_path_backprop_<T>(
				transformed_derivs, out + chunk.offset * path_stride, chunk.size,
				dimension, length, time_aug, lead_lag, end_time);
			batch_offset += chunk.size;
		}
		cudaDeviceSynchronize();
		check_cuda_error();
	}
	else {
		branched_sig_coef_backprop_cuda_core_(path, out, coefs, derivs, tree_data,
			tree_data_len, batch_size, dimension, length, max_nodes, dimension, planar,
			correction, correction_len, correction_batch_stride,
			correction_segment_stride);
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
	uint64_t max_nodes,
	bool planar = false,
	uint64_t data_dimension = 0,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	if (data_dimension == 0) data_dimension = dimension;
	validate_correction_len_(data_dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	const bool has_correction = correction_len != 0;
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);

	// Single-point paths have no increments => zero path gradients
	if (length <= 1) {
		cudaMemset(out, 0, batch_size * length * dimension * sizeof(T));
		cudaDeviceSynchronize();
		check_cuda_error();
		return;
	}

	const int steps = static_cast<int>(length - 1);
	const size_t path_stride = checked_cuda_size_mul(
		static_cast<size_t>(length), static_cast<size_t>(dimension),
		"CUDA branched sig backprop");

	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	const bool use_inc_reduction = dimension <= 4;
	const size_t reduction_length = use_inc_reduction
		? checked_cuda_size_mul(
			static_cast<size_t>(dimension), block >> 5,
			"CUDA branched sig backprop")
		: 0;
	const bool use_product_reduction = !has_correction && max_nodes >= 4
		&& (planar || std::is_same_v<T, double>);
	const size_t product_reduction_length = use_product_reduction
		? checked_cuda_size_mul(
			2 * static_cast<size_t>(block >> 5),
			static_cast<size_t>(gc.deriv_target_count),
			"CUDA branched sig backprop")
		: 0;
	const size_t global_basis_arrays = has_correction ? 10 : 4;
	const size_t shared_basis_arrays = has_correction
		? checked_cuda_size_add(
			6, checked_cuda_size_mul(
				2, static_cast<size_t>(max_nodes),
				"CUDA branched sig backprop"),
			"CUDA branched sig backprop")
		: 4;
	const size_t global_t_arrays = checked_cuda_size_add(
		checked_cuda_size_mul(
			global_basis_arrays, static_cast<size_t>(gc.total_length),
			"CUDA branched sig backprop"),
		checked_cuda_size_mul(
			2, static_cast<size_t>(dimension),
			"CUDA branched sig backprop"),
		"CUDA branched sig backprop");
	const size_t shared_t_arrays = checked_cuda_size_add(
		checked_cuda_size_add(
			checked_cuda_size_mul(
				shared_basis_arrays, static_cast<size_t>(gc.total_length),
				"CUDA branched sig backprop"),
			checked_cuda_size_mul(
				2, static_cast<size_t>(dimension),
				"CUDA branched sig backprop"),
			"CUDA branched sig backprop"),
		checked_cuda_size_add(
			reduction_length, product_reduction_length,
			"CUDA branched sig backprop"),
		"CUDA branched sig backprop");
	const size_t table_entries = checked_cuda_size_add(
		static_cast<size_t>(gc.coprod_data_len),
		checked_cuda_size_add(
			static_cast<size_t>(gc.num_trees) + 1,
			static_cast<size_t>(gc.max_nodes) + 2,
			"CUDA branched sig backprop"),
		"CUDA branched sig backprop");
	const size_t smem = checked_cuda_size_add(
		checked_cuda_size_mul(
			shared_t_arrays, sizeof(T), "CUDA branched sig backprop"),
		checked_cuda_size_mul(
			table_entries, sizeof(uint32_t),
			"CUDA branched sig backprop"),
		"CUDA branched sig backprop");

	const T* d_inv_fact;
	if constexpr (std::is_same_v<T, float>)
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f32);
	else
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f64);

	const bool use_shared_storage = try_configure_dynamic_smem(
		branched_sig_backprop_ker<T, true>, smem);
	std::unique_ptr<CudaBatchWorkspace<T>> workspace;
	if (!use_shared_storage) {
		workspace = std::make_unique<CudaBatchWorkspace<T>>(
			batch_size, global_t_arrays,
			"CUDA branched sig backprop global workspace");
	}
	const uint64_t max_chunk_size = use_shared_storage
		? CUDA_BATCH_GRID_CAPACITY : workspace->capacity();
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, max_chunk_size);
		if (use_shared_storage) {
			branched_sig_backprop_ker<T, true><<<
				batch_chunk.grid, block, smem>>>(
				path, out, bsig, bsig_derivs,
				static_cast<int>(dimension), static_cast<int>(data_dimension),
				steps, gc.total_length, path_stride,
				gc.d_labels_data, gc.d_labels_offsets32, d_inv_fact,
				gc.d_coprod_data32, gc.d_coprod_offsets32,
				gc.d_order_index32, gc.d_chain_index_offsets32,
				gc.d_chain_indices32, gc.max_nodes, gc.coprod_data_len,
				gc.deriv_target_count, use_inc_reduction,
				use_product_reduction, correction, correction_len,
				correction_batch_stride, correction_segment_stride,
				nullptr, 0, batch_chunk.offset, batch_chunk.size);
		} else {
			branched_sig_backprop_ker<T, false><<<
				batch_chunk.grid, std::min(block, 256u)>>>(
				path, out, bsig, bsig_derivs,
				static_cast<int>(dimension), static_cast<int>(data_dimension),
				steps, gc.total_length, path_stride,
				gc.d_labels_data, gc.d_labels_offsets32, d_inv_fact,
				gc.d_coprod_data32, gc.d_coprod_offsets32,
				gc.d_order_index32, gc.d_chain_index_offsets32,
				gc.d_chain_indices32, gc.max_nodes, gc.coprod_data_len,
				gc.deriv_target_count, false, false, correction, correction_len,
				correction_batch_stride, correction_segment_stride,
				workspace->get(), global_t_arrays,
				batch_chunk.offset, batch_chunk.size);
		}
		batch_offset += batch_chunk.size;
	}
	cudaDeviceSynchronize();
	check_cuda_error();
}

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
	T end_time,
	bool planar = false,
	bool scalar_term = true,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");
	if (batch_size == 0)
		return;
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;
	if (scalar_term && !time_aug && !lead_lag) {
		branched_sig_backprop_cuda_core_<T>(
			path, out, bsig_derivs, bsig, batch_size, dimension, length,
			max_nodes, planar, dimension, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
		return;
	}

	const auto& gc = get_gpu_cache(t_dimension, max_nodes, planar);
	const uint64_t full_len = gc.total_length;
	const uint64_t sig_stride = scalar_term ? full_len : full_len - 1;
	const size_t path_stride = checked_cuda_size_mul(
		static_cast<size_t>(length), static_cast<size_t>(dimension),
		"CUDA branched sig backprop staging");
	const size_t transformed_stride = time_aug || lead_lag
		? checked_cuda_size_mul(
			static_cast<size_t>(t_length), static_cast<size_t>(t_dimension),
			"CUDA branched sig backprop staging")
		: 0;
	const size_t transformed_arrays = checked_cuda_size_mul(
		2, transformed_stride, "CUDA branched sig backprop staging");
	const size_t scalar_arrays = scalar_term ? 0 : checked_cuda_size_mul(
		2, static_cast<size_t>(full_len),
		"CUDA branched sig backprop staging");
	const size_t workspace_row = checked_cuda_size_add(
		transformed_arrays, scalar_arrays,
		"CUDA branched sig backprop staging");
	CudaBatchWorkspace<T> workspace(
		batch_size, workspace_row, "CUDA branched sig backprop staging");
	T* transformed = workspace.get();
	T* transformed_derivs = transformed
		+ workspace.capacity() * transformed_stride;
	T* staged_bsig = transformed_derivs
		+ workspace.capacity() * transformed_stride;
	T* staged_derivs = staged_bsig
		+ workspace.capacity() * (scalar_term ? 0 : full_len);

	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset, workspace.capacity());
		const T* core_path = path + chunk.offset * path_stride;
		T* core_out = out + chunk.offset * path_stride;
		if (time_aug || lead_lag) {
			cu_transform_path_<T>(
				core_path, transformed, chunk.size, dimension, length,
				time_aug, lead_lag, end_time);
			core_path = transformed;
			core_out = transformed_derivs;
		}
		const T* core_bsig = bsig + chunk.offset * sig_stride;
		const T* core_derivs = bsig_derivs + chunk.offset * sig_stride;
		if (!scalar_term) {
			bsig_stage_prepend_one_<T>(
				core_bsig, staged_bsig, chunk.size, full_len);
			bsig_stage_prepend_zero_<T>(
				core_derivs, staged_derivs, chunk.size, full_len);
			core_bsig = staged_bsig;
			core_derivs = staged_derivs;
		}
		const T* chunk_correction = correction_len == 0 ? nullptr
			: correction + chunk.offset * correction_batch_stride;
		branched_sig_backprop_cuda_core_<T>(
			core_path, core_out, core_derivs, core_bsig, chunk.size,
			t_dimension, t_length, max_nodes, planar, dimension,
			chunk_correction, correction_len, correction_batch_stride,
			correction_segment_stride);
		if (time_aug || lead_lag) {
			cu_transform_path_backprop_<T>(
				transformed_derivs, out + chunk.offset * path_stride,
				chunk.size, dimension, length, time_aug, lead_lag, end_time);
		}
		batch_offset += chunk.size;
	}
}

static void prepare_branched_log_sig_cuda_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	bool planar,
	bool use_disk
) {
	if (method < 0 || method > 3)
		throw std::invalid_argument(
			"branched log signature method must be 0, 1, 2, or 3");
	if (method == 3 && max_nodes > 12)
		throw std::runtime_error(
			"CUDA MKW BCH method supports degree at most 12");
	if (method != 0 && !planar)
		throw std::invalid_argument(
			"compressed branched log signatures require planar=True");
	BranchedSigCache host_cache;
	prepare_branched_sig_gpu_cache_(
		dimension, max_nodes, planar, use_disk, &host_cache);
	const std::filesystem::path cache_directory = use_disk
		? get_cuda_cache_dir_() / cu_cache_folder_name
		: std::filesystem::path{};
	BranchedLogSigCache host_log_cache(
		host_cache, method, cache_directory, use_disk);
	prepare_cuda_branched_log_horner_plan_(
		host_cache, host_log_cache.horner_plan());
	if (method >= 1) {
		prepare_cuda_mkw_basis_cache_(
			host_cache,
			host_log_cache.basis_cache((std::min)(method, 2)));
		if (method == 3)
			prepare_cuda_branched_bch_cache_(
				host_cache, host_log_cache.bch_cache());
	}
}

// =========================================================================
// extern "C" wrappers
// =========================================================================

extern "C" {

	CUSIG_API int prepare_branched_sig_cuda(uint64_t dimension, uint64_t max_nodes, bool planar, bool use_disk) noexcept {
		CUDA_SAFE_CALL(prepare_branched_sig_gpu_cache_(dimension, max_nodes, planar, use_disk));
	}

	CUSIG_API int prepare_branched_log_sig_cuda(uint64_t dimension, uint64_t max_nodes, int method, bool planar, bool use_disk) noexcept {
		CUDA_SAFE_CALL(prepare_branched_log_sig_cuda_(dimension, max_nodes, method, planar, use_disk));
	}

	CUSIG_API int prepare_branched_sig_coef_cuda(const uint64_t* tree_data, uint64_t tree_data_len, uint64_t data_dimension, uint64_t dimension, uint64_t max_nodes, bool planar, bool use_disk) noexcept {
		CUDA_SAFE_CALL(prepare_branched_sig_coef_cache_cuda_(tree_data, tree_data_len, data_dimension, dimension, max_nodes, planar, use_disk));
	}

	CUSIG_API int branched_sig_coef_cuda_f(const float* path, float* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_coef_cuda_<float>(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_coef_cuda_d(const double* path, double* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_coef_cuda_<double>(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_coef_backprop_cuda_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_coef_backprop_cuda_<float>(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_coef_backprop_cuda_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_coef_backprop_cuda_<double>(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_cuda_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, bool scalar_term, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_cuda_<float>(path, out, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_cuda_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, bool scalar_term, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_cuda_<double>(path, out, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_combine_cuda_f(const float* bsig1, const float* bsig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUDA_SAFE_CALL(branched_sig_combine_cuda_<float>(bsig1, bsig2, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}
	CUSIG_API int branched_sig_combine_cuda_d(const double* bsig1, const double* bsig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUDA_SAFE_CALL(branched_sig_combine_cuda_<double>(bsig1, bsig2, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}

	CUSIG_API int branched_sig_combine_backprop_cuda_f(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUDA_SAFE_CALL(branched_sig_combine_backprop_cuda_<float>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, planar, scalar_term));
	}
	CUSIG_API int branched_sig_combine_backprop_cuda_d(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUDA_SAFE_CALL(branched_sig_combine_backprop_cuda_<double>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, planar, scalar_term));
	}


	CUSIG_API int branched_sig_backprop_cuda_f(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, bool scalar_term, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_backprop_cuda_<float>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_backprop_cuda_d(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, bool scalar_term, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUDA_SAFE_CALL(branched_sig_backprop_cuda_<double>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

}
