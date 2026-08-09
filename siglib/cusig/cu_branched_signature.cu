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
#include "cu_disk_cache.h"
#include "cu_path_transforms.h"
#include "cu_utils.h"
#include "../shared/branched_cache.h"
#include "../shared/branched_sig_coef_cache.h"

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>

void prepare_cuda_branched_log_sig_gpu_cache_(
	uint64_t dimension, uint64_t max_nodes, bool planar);

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

static constexpr const char* branched_cache_version = "v3";

static std::filesystem::path branched_cache_file_path_(uint64_t dimension, uint64_t max_nodes, bool planar) {
	const char* prefix = planar ? "planar_branched_" : "branched_";
	return get_cuda_cache_dir_() / cu_cache_folder_name /
		(prefix + std::to_string(dimension) + "_" + std::to_string(max_nodes) + "_" + branched_cache_version + ".bin");
}

static void write_branched_cache_(const BranchedSigCache& c) {
	std::ofstream out(branched_cache_file_path_(c.dimension, c.max_nodes, c.planar), std::ios::binary);
	if (!out)
		throw std::filesystem::filesystem_error(
			"Failed to open CUDA branched cache file for writing",
			branched_cache_file_path_(c.dimension, c.max_nodes, c.planar),
			std::make_error_code(std::errc::io_error));

	out.write(reinterpret_cast<const char*>(&cu_cache_magic_number), sizeof(cu_cache_magic_number));
	out.write(reinterpret_cast<const char*>(&c.dimension), sizeof(c.dimension));
	out.write(reinterpret_cast<const char*>(&c.max_nodes), sizeof(c.max_nodes));
	out.write(reinterpret_cast<const char*>(&c.total_length), sizeof(c.total_length));
	cu_serialize_vector_(out, c.order_index);

	uint64_t n = c.inv_tree_factorial.size();
	out.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (n > 0)
		out.write(reinterpret_cast<const char*>(c.inv_tree_factorial.data()), n * sizeof(double));

	n = c.node_labels_data.size();
	out.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (n > 0)
		out.write(reinterpret_cast<const char*>(c.node_labels_data.data()), n);

	cu_serialize_vector_(out, c.node_labels_offsets);
	cu_serialize_vector_(out, c.chain_index_offsets);
	cu_serialize_vector_(out, c.chain_indices);
	cu_serialize_vector_(out, c.coproduct_data);
	cu_serialize_vector_(out, c.coproduct_offsets);
}

static bool read_branched_cache_(uint64_t dimension, uint64_t max_nodes, bool planar, BranchedSigCache& c) {
	const auto path = branched_cache_file_path_(dimension, max_nodes, planar);
	if (!std::filesystem::exists(path))
		return false;

	std::ifstream in(path, std::ios::binary);
	if (!in)
		return false;

	BranchedSigCache tmp;
	uint64_t magic;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!in || magic != cu_cache_magic_number)
		throw corrupted_cache_error("Tried to read an invalid cache file. Cache may have been corrupted.");

	in.read(reinterpret_cast<char*>(&tmp.dimension), sizeof(tmp.dimension));
	in.read(reinterpret_cast<char*>(&tmp.max_nodes), sizeof(tmp.max_nodes));
	if (!in || tmp.dimension != dimension || tmp.max_nodes != max_nodes)
		return false;
	in.read(reinterpret_cast<char*>(&tmp.total_length), sizeof(tmp.total_length));
	if (!in || tmp.total_length > kCuMaxCacheVectorSize)
		throw std::runtime_error("Tried to read an invalid cache file: branched total_length exceeds limit");

	cu_deserialize_vector_(in, tmp.order_index);

	uint64_t n;
	in.read(reinterpret_cast<char*>(&n), sizeof(n));
	if (!in || n > kCuMaxCacheVectorSize || n + 1 > tmp.total_length)
		throw std::runtime_error("Tried to read an invalid cache file: branched inv_tree_factorial size invalid");
	cu_check_stream_has_bytes_(in, n * sizeof(double), "branched inv_tree_factorial body");
	tmp.inv_tree_factorial.resize(n);
	if (n > 0)
		in.read(reinterpret_cast<char*>(tmp.inv_tree_factorial.data()), n * sizeof(double));

	in.read(reinterpret_cast<char*>(&n), sizeof(n));
	if (!in || n > kCuMaxCacheVectorSize)
		throw std::runtime_error("Tried to read an invalid cache file: branched node_labels_data size invalid");
	cu_check_stream_has_bytes_(in, n, "branched node_labels_data body");
	tmp.node_labels_data.resize(n);
	if (n > 0)
		in.read(reinterpret_cast<char*>(tmp.node_labels_data.data()), n);

	cu_deserialize_vector_(in, tmp.node_labels_offsets);
	cu_deserialize_vector_(in, tmp.chain_index_offsets);
	cu_deserialize_vector_(in, tmp.chain_indices);
	cu_deserialize_vector_(in, tmp.coproduct_data);
	cu_deserialize_vector_(in, tmp.coproduct_offsets);

	if (!in.good())
		return false;

	tmp.planar = planar;
	c = std::move(tmp);
	return true;
}

static void prepare_branched_sig_gpu_cache_(uint64_t dimension, uint64_t max_nodes, bool planar = false, bool use_disk = false) {
	BranchedSigCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.dimension = dimension;
	key.max_nodes = max_nodes;
	key.planar = planar;
	{
		std::lock_guard<std::mutex> lock(s_gpu_cache_map_mu);
		auto it = s_gpu_cache_map.find(key);
		if (it != s_gpu_cache_map.end())
			return;
	}

	BranchedSigCache c;
	if (use_disk)
		ensure_cuda_cache_dir_();
	if (!use_disk || !read_branched_cache_(dimension, max_nodes, planar, c)) {
		c = build_branched_sig_cache(dimension, max_nodes, planar);
		if (use_disk)
			write_branched_cache_(c);
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
		cache = build_branched_sig_coef_cache(
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
	if (tid < num_trees) {
		const uint32_t fi = tid + 1;
		T val = X[fi] * Y[0] + X[0] * Y[fi];
		uint32_t pos = s_coprod_off[tid];
		const uint32_t pend = s_coprod_off[tid + 1];
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
	if (tid < num_trees) {
		T prod = T(1);
		const uint32_t lstart = labels_offsets[tid];
		const uint32_t lend = labels_offsets[tid + 1];
		#pragma unroll 8
		for (uint32_t j = lstart; j < lend; ++j)
			prod *= inc[labels_data[j]];
		out[tid + 1] = prod * inv_factorial[tid];
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

template<typename T>
__global__ __launch_bounds__(1024)
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
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	T* temp = reinterpret_cast<T*>(smem);
	const bool has_correction = correction_len != 0;
	const bool use_shared_state = planar_fast_path && !has_correction;
	T* local_log = has_correction ? temp + total_len : temp;
	T* power = has_correction ? local_log + total_len : temp;
	T* next_power = has_correction ? power + total_len : temp;
	T* state = use_shared_state ? temp + total_len : nullptr;
	T* next_state = use_shared_state ? state + total_len : nullptr;
	T* inc = has_correction ? next_power + total_len : (use_shared_state ? next_state + total_len : temp + total_len);
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
// | inc[dim] | inc_derivs[dim] | coprod tables (same as forward) |

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
	if (tid < num_trees) {
		const uint32_t fi = tid + 1;
		const T d = d_out[fi];
		if (d != T(0)) {
			myAtomicAdd(&d_X[fi], d * Y[0]);
			myAtomicAdd(&d_Y[0], d * X[fi]);
			myAtomicAdd(&d_X[0], d * Y[fi]);
			myAtomicAdd(&d_Y[fi], d * X[0]);

			uint32_t pos = s_coprod_off[tid];
			const uint32_t pend = s_coprod_off[tid + 1];
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
	int dim,
	uint32_t total_len,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	uint32_t tid
) {
	const uint32_t num_trees = total_len - 1;
	for (uint32_t d = tid; d < static_cast<uint32_t>(dim); d += blockDim.x)
		inc_derivs[d] = T(0);
	__syncthreads();

	if (tid < num_trees) {
		const T dF_dYi = local_derivs[tid + 1];
		if (dF_dYi != T(0)) {
			const T inv_gamma = inv_factorial[tid];
			const uint32_t lstart = labels_offsets[tid];
			const uint32_t lend = labels_offsets[tid + 1];
			const uint32_t n_labels = lend - lstart;

			const T base = inv_gamma * dF_dYi;
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
}

template<typename T>
__device__ void local_log_bsig_deriv_to_inc_block_(
	const T* local_derivs,
	const T* inc,
	T* inc_derivs,
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
		linear_bsig_deriv_to_inc_block_(local_derivs, inc, inc_derivs, dim,
			total_len, labels_data, labels_offsets, inv_factorial, tid);
		return;
	}

	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		local_log[i] = T(0);
		d_correction[i] = T(0);
	}
	for (uint32_t i = tid; i < static_cast<uint32_t>(max_nodes) * total_len; i += blockDim.x)
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
		branched_hopf_convolution_block_(powers + static_cast<uint32_t>(k - 2) * total_len, local_log,
			powers + static_cast<uint32_t>(k - 1) * total_len, total_len, s_coprod_data, s_coprod_off, tid);
	}

	T inv_k_factorial = T(1);
	for (int k = 1; k <= max_nodes; ++k) {
		inv_k_factorial /= static_cast<T>(k);
		T* d_power = power_derivs + static_cast<uint32_t>(k - 1) * total_len;
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			d_power[i] += inv_k_factorial * local_derivs[i];
		__syncthreads();
	}

	for (int k = max_nodes; k > 1; --k) {
		branched_hopf_convolution_deriv_block_(powers + static_cast<uint32_t>(k - 2) * total_len, local_log,
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

template<typename T>
__global__ __launch_bounds__(1024)
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
	const T* __restrict__ correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;

	extern __shared__ char smem[];
	T* s_bsig = reinterpret_cast<T*>(smem);
	T* s_derivs = s_bsig + total_len;
	T* temp_Y = s_derivs + total_len;
	T* local_derivs = temp_Y + total_len;
	T* inc = local_derivs + total_len;
	T* inc_derivs = inc + dim;
	T* local_log = inc_derivs + dim;
	const bool has_correction = correction_len != 0;
	T* powers = has_correction ? local_log + total_len : local_log;
	T* power_derivs = has_correction ? powers + static_cast<uint32_t>(max_nodes) * total_len : powers;
	T* d_correction = has_correction ? power_derivs + static_cast<uint32_t>(max_nodes) * total_len : power_derivs;
	T* local_log_end = has_correction ? d_correction + total_len : inc_derivs + dim;
	uint32_t* s_coprod_data = reinterpret_cast<uint32_t*>(local_log_end);
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
		local_branched_sig_block_(inc, temp_Y, local_log, powers, power_derivs,
			dim, data_dim, seg_correction, correction_len, total_len,
			labels_data, labels_offsets, inv_factorial,
			s_coprod_data, s_coprod_off, s_order_idx,
			chain_index_offsets, chain_indices, max_nodes, tid);

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

			const T dF_tau = (tid < num_trees) ? s_derivs[tid + 1] : T(0);
			__syncthreads();

			if (tid < num_trees && dF_tau != T(0)) {
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
			__syncthreads();
		}
		else {
			// seg == 0: local_derivs = s_derivs
			for (uint32_t i = tid; i < total_len; i += blockDim.x)
				local_derivs[i] = s_derivs[i];
			__syncthreads();
		}

		if (!has_correction) {
			linear_bsig_deriv_to_inc_block_(local_derivs, inc, inc_derivs, dim,
				total_len, labels_data, labels_offsets, inv_factorial, tid);
		} else {
			local_log_bsig_deriv_to_inc_block_(local_derivs, inc, inc_derivs,
				local_log, powers, power_derivs, d_correction,
				dim, data_dim, seg_correction, correction_len, total_len,
				labels_data, labels_offsets, inv_factorial,
				s_coprod_data, s_coprod_off, s_order_idx,
				chain_index_offsets, chain_indices, max_nodes, tid);
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

	// Butcher product: out = butcher_product(X, Y) - process high to low order
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
	if (scalar_term) {
		branched_sig_combine_cuda_core_<T>(bsig1, bsig2, out, batch_size, dimension, max_nodes, planar);
		return;
	}
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);
	const uint64_t full_len = gc.total_length;
	CudaBuf<T> d_b1(batch_size * full_len * sizeof(T));
	CudaBuf<T> d_b2(batch_size * full_len * sizeof(T));
	CudaBuf<T> d_out(batch_size * full_len * sizeof(T));
	bsig_stage_prepend_one_<T>(bsig1, d_b1.get(), batch_size, full_len);
	bsig_stage_prepend_one_<T>(bsig2, d_b2.get(), batch_size, full_len);
	branched_sig_combine_cuda_core_<T>(d_b1.get(), d_b2.get(), d_out.get(), batch_size, dimension, max_nodes, planar);
	bsig_stage_strip_<T>(d_out.get(), out, batch_size, full_len);
}

template<typename T>
void branched_sig_combine_backprop_cuda_(
	const T* bsig1, const T* bsig2, const T* derivs, T* out1, T* out2,
	uint64_t batch_size, uint64_t dimension, uint64_t max_nodes,
	bool planar = false,
	bool scalar_term = true
) {
	if (scalar_term) {
		branched_sig_combine_backprop_cuda_core_<T>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, planar);
		return;
	}
	const auto& gc = get_gpu_cache(dimension, max_nodes, planar);
	const uint64_t full_len = gc.total_length;
	CudaBuf<T> d_b1(batch_size * full_len * sizeof(T));
	CudaBuf<T> d_b2(batch_size * full_len * sizeof(T));
	CudaBuf<T> d_der(batch_size * full_len * sizeof(T));
	CudaBuf<T> d_o1(batch_size * full_len * sizeof(T));
	CudaBuf<T> d_o2(batch_size * full_len * sizeof(T));
	bsig_stage_prepend_one_<T>(bsig1, d_b1.get(), batch_size, full_len);
	bsig_stage_prepend_one_<T>(bsig2, d_b2.get(), batch_size, full_len);
	bsig_stage_prepend_zero_<T>(derivs, d_der.get(), batch_size, full_len);
	branched_sig_combine_backprop_cuda_core_<T>(d_b1.get(), d_b2.get(), d_der.get(), d_o1.get(), d_o2.get(),
		batch_size, dimension, max_nodes, planar);
	bsig_stage_strip_<T>(d_o1.get(), out1, batch_size, full_len);
	bsig_stage_strip_<T>(d_o2.get(), out2, batch_size, full_len);
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
	if (block > 1024) throw std::invalid_argument("CUDA branched sig combine: num_trees > 1024 not supported");

	size_t smem = gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);
	configure_dynamic_smem(
		branched_sig_combine_ker<T>, smem, "CUDA branched sig combine");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_combine_ker<T><<<batch_chunk.grid, block, smem>>>(
			bsig1, bsig2, out, gc.total_length,
			gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
			gc.max_nodes, gc.coprod_data_len,
			batch_chunk.offset, batch_chunk.size);
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
	if (block > 1024) throw std::invalid_argument("CUDA branched sig combine backprop: num_trees > 1024 not supported");

	size_t smem = gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);
	configure_dynamic_smem(
		branched_sig_combine_backprop_ker<T>, smem,
		"CUDA branched sig combine backprop");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_combine_backprop_ker<T><<<batch_chunk.grid, block, smem>>>(
			bsig1, bsig2, derivs, out1, out2, gc.total_length,
			gc.d_coprod_data32, gc.d_coprod_offsets32, gc.d_order_index32,
			gc.max_nodes, gc.coprod_data_len,
			batch_chunk.offset, batch_chunk.size);
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
	const uint64_t path_stride = length * dimension;

	const bool use_shared_state = !has_correction && planar;
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024)
		throw std::invalid_argument("CUDA branched sig: num_trees > 1024 not supported");

	const uint64_t t_arrays = has_correction
		? (4 * gc.total_length + dimension)
		: ((use_shared_state ? 3 * gc.total_length : gc.total_length) + dimension);
	size_t smem = t_arrays * sizeof(T)
		+ gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);

	// Select float or double inv_factorial
	const T* d_inv_fact;
	if constexpr (std::is_same_v<T, float>)
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f32);
	else
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f64);

	configure_dynamic_smem(
		branched_sig_ker<T>, smem, "CUDA branched sig");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_ker<T><<<batch_chunk.grid, block, smem>>>(
			path, out, static_cast<int>(dimension), static_cast<int>(data_dimension), steps,
			gc.total_length, path_stride,
			gc.d_labels_data, gc.d_labels_offsets32,
			d_inv_fact,
			gc.d_coprod_data32, gc.d_coprod_offsets32,
			gc.d_order_index32, gc.d_chain_index_offsets32, gc.d_chain_indices32, gc.max_nodes,
			gc.coprod_data_len, correction, correction_len,
			correction_batch_stride, correction_segment_stride, use_shared_state,
			batch_chunk.offset, batch_chunk.size
		);
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
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	// For scalar_term=false, stage output through a full-sized buffer and strip the scalar.
	T* core_out = out;
	CudaBuf<T> d_out_full;
	const auto& gc = get_gpu_cache(t_dimension, max_nodes, planar);
	const uint64_t full_len = gc.total_length;
	if (!scalar_term) {
		d_out_full = CudaBuf<T>(batch_size * full_len * sizeof(T));
		core_out = d_out_full.get();
	}

	if (time_aug || lead_lag) {
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		CudaBuf<T> d_transformed(t_path_size * sizeof(T));

		cu_transform_path_<T>(path, d_transformed.get(), batch_size, dimension, length, time_aug, lead_lag, end_time);
		cudaDeviceSynchronize();

		branched_sig_cuda_core_<T>(d_transformed.get(), core_out, batch_size, t_dimension, t_length,
			max_nodes, planar, dimension, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
	}
	else {
		branched_sig_cuda_core_<T>(path, core_out, batch_size, dimension, length,
			max_nodes, planar, dimension, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
	}

	if (!scalar_term) {
		bsig_stage_strip_<T>(core_out, out, batch_size, full_len);
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
	uint64_t correction_segment_stride
) {
	if (length > static_cast<uint64_t>(UINT32_MAX) + 1)
		throw std::invalid_argument("CUDA branched sig coef path length exceeds kernel range");

	const uint32_t num_non_scalar = cache.cache_size - 1;
	unsigned int block = (num_non_scalar + 31u) & ~31u;
	if (block < 32)
		block = 32;
	if (block > 1024)
		throw std::invalid_argument("CUDA branched sig coef sparse closure exceeds 1024 trees");

	const uint32_t num_corrections = branched_sig_coef_num_corrections_(cache, correction_len);
	const bool has_correction = num_corrections != 0;
	const uint64_t t_arrays = has_correction
		? 4 * static_cast<uint64_t>(cache.cache_size) + dimension
		: static_cast<uint64_t>(cache.cache_size) + dimension;
	const uint64_t state_arrays = state == nullptr ? cache.cache_size : 0;
	const size_t smem = (t_arrays + state_arrays) * sizeof(T)
		+ cache.coprod_data_len * sizeof(uint32_t)
		+ (static_cast<uint64_t>(cache.cache_size) + 1) * sizeof(uint32_t)
		+ (static_cast<uint64_t>(cache.max_nodes) + 2) * sizeof(uint32_t);
	const T* d_inv_factorial;
	if constexpr (std::is_same_v<T, float>)
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f32);
	else
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f64);

	configure_dynamic_smem(
		branched_sig_coef_forward_kernel_<T>, smem, "CUDA branched sig coef");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_coef_forward_kernel_<T><<<batch_chunk.grid, block, smem>>>(
			path, state, out, static_cast<uint32_t>(dimension),
			length == 0 ? 0 : static_cast<uint32_t>(length - 1), length * dimension,
			cache.cache_size, cache.d_target_indices, cache.num_targets,
			cache.d_labels_data, cache.d_labels_offsets, d_inv_factorial,
			cache.d_coprod_data, cache.d_coprod_offsets, cache.d_order_index,
			cache.d_leaf_indices, cache.d_correction_offsets, cache.d_correction_locals,
			num_corrections, cache.max_nodes, cache.coprod_data_len, correction,
			correction_batch_stride, correction_segment_stride,
			batch_chunk.offset, batch_chunk.size);
		batch_offset += batch_chunk.size;
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
		CudaBuf<T> transformed(
			batch_size * transformed_length * transformed_dimension * sizeof(T));
		cu_transform_path_<T>(path, transformed.get(), batch_size, dimension, length,
			time_aug, lead_lag, end_time);
		branched_sig_coef_cuda_core_(transformed.get(), out, tree_data, tree_data_len,
			batch_size, transformed_dimension, transformed_length, max_nodes, dimension,
			planar, correction, correction_len, correction_batch_stride,
			correction_segment_stride);
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

	CudaBuf<T> state(batch_size * static_cast<uint64_t>(cache.cache_size) * sizeof(T));
	launch_branched_sig_coef_forward_(path, state.get(), static_cast<T*>(nullptr),
		batch_size, dimension, length, cache, correction, correction_len,
		correction_batch_stride, correction_segment_stride);

	const uint32_t num_non_scalar = cache.cache_size - 1;
	unsigned int block = (num_non_scalar + 31u) & ~31u;
	if (block < 32)
		block = 32;
	if (block > 1024)
		throw std::invalid_argument("CUDA branched sig coef sparse closure exceeds 1024 trees");
	const uint32_t num_corrections = branched_sig_coef_num_corrections_(cache, correction_len);
	const bool has_correction = num_corrections != 0;
	const uint64_t t_arrays = has_correction
		? (6 + 2 * static_cast<uint64_t>(cache.max_nodes)) * cache.cache_size
			+ 2 * dimension
		: 4 * static_cast<uint64_t>(cache.cache_size) + 2 * dimension;
	const size_t smem = t_arrays * sizeof(T)
		+ cache.coprod_data_len * sizeof(uint32_t)
		+ (static_cast<uint64_t>(cache.cache_size) + 1) * sizeof(uint32_t)
		+ (static_cast<uint64_t>(cache.max_nodes) + 2) * sizeof(uint32_t);
	const T* d_inv_factorial;
	if constexpr (std::is_same_v<T, float>)
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f32);
	else
		d_inv_factorial = reinterpret_cast<const T*>(cache.d_inv_factorial_f64);

	configure_dynamic_smem(
		branched_sig_coef_backprop_kernel_<T>, smem,
		"CUDA branched sig coef backprop");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_coef_backprop_kernel_<T><<<batch_chunk.grid, block, smem>>>(
			path, out, state.get(), coefs, derivs, static_cast<uint32_t>(dimension),
			static_cast<uint32_t>(length - 1), length * dimension, cache.cache_size,
			cache.d_target_indices, cache.num_targets, cache.d_labels_data,
			cache.d_labels_offsets, d_inv_factorial, cache.d_coprod_data,
			cache.d_coprod_offsets, cache.d_order_index, cache.d_leaf_indices,
			cache.d_correction_offsets, cache.d_correction_locals, num_corrections,
			cache.max_nodes, cache.coprod_data_len, correction, correction_batch_stride,
			correction_segment_stride, batch_chunk.offset, batch_chunk.size);
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
		CudaBuf<T> transformed(
			batch_size * transformed_length * transformed_dimension * sizeof(T));
		CudaBuf<T> transformed_derivs(
			batch_size * transformed_length * transformed_dimension * sizeof(T));
		cu_transform_path_<T>(path, transformed.get(), batch_size, dimension, length,
			time_aug, lead_lag, end_time);
		branched_sig_coef_backprop_cuda_core_(transformed.get(),
			transformed_derivs.get(), coefs, derivs, tree_data, tree_data_len,
			batch_size, transformed_dimension, transformed_length, max_nodes, dimension,
			planar, correction, correction_len, correction_batch_stride,
			correction_segment_stride);
		cu_transform_path_backprop_<T>(transformed_derivs.get(), out, batch_size,
			dimension, length, time_aug, lead_lag, end_time);
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
	const uint64_t path_stride = length * dimension;

	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024)
		throw std::invalid_argument("CUDA branched sig backprop: num_trees > 1024 not supported");

	const uint64_t t_arrays = has_correction
		? ((6 + 2 * static_cast<uint64_t>(gc.max_nodes)) * gc.total_length + 2 * dimension)
		: (4 * gc.total_length + 2 * dimension);
	size_t smem = t_arrays * sizeof(T)
		+ gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t)
		+ (gc.max_nodes + 2) * sizeof(uint32_t);

	const T* d_inv_fact;
	if constexpr (std::is_same_v<T, float>)
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f32);
	else
		d_inv_fact = reinterpret_cast<const T*>(gc.d_inv_factorial_f64);

	configure_dynamic_smem(
		branched_sig_backprop_ker<T>, smem, "CUDA branched sig backprop");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_backprop_ker<T><<<batch_chunk.grid, block, smem>>>(
			path, out, bsig, bsig_derivs,
			static_cast<int>(dimension), static_cast<int>(data_dimension), steps,
			gc.total_length, path_stride,
			gc.d_labels_data, gc.d_labels_offsets32,
			d_inv_fact,
			gc.d_coprod_data32, gc.d_coprod_offsets32,
			gc.d_order_index32, gc.d_chain_index_offsets32, gc.d_chain_indices32, gc.max_nodes,
			gc.coprod_data_len, correction, correction_len,
			correction_batch_stride, correction_segment_stride,
			batch_chunk.offset, batch_chunk.size
		);
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
	const uint64_t t_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t t_length = lead_lag ? 2 * length - 1 : length;

	// Stage bsig and bsig_derivs to full-size buffers when scalar_term=false.
	const T* core_bsig = bsig;
	const T* core_derivs = bsig_derivs;
	CudaBuf<T> d_bsig_full;
	CudaBuf<T> d_derivs_full;
	if (!scalar_term) {
		const auto& gc = get_gpu_cache(t_dimension, max_nodes, planar);
		const uint64_t full_len = gc.total_length;
		d_bsig_full = CudaBuf<T>(batch_size * full_len * sizeof(T));
		d_derivs_full = CudaBuf<T>(batch_size * full_len * sizeof(T));
		bsig_stage_prepend_one_<T>(bsig, d_bsig_full.get(), batch_size, full_len);
		bsig_stage_prepend_zero_<T>(bsig_derivs, d_derivs_full.get(), batch_size, full_len);
		core_bsig = d_bsig_full.get();
		core_derivs = d_derivs_full.get();
	}

	if (time_aug || lead_lag) {
		const uint64_t t_path_size = batch_size * t_length * t_dimension;
		T* d_transformed = nullptr;
		T* d_transformed_derivs = nullptr;

		try {
			cudaMalloc(&d_transformed, t_path_size * sizeof(T));
			cu_transform_path_<T>(path, d_transformed, batch_size, dimension, length, time_aug, lead_lag, end_time);

			cudaMalloc(&d_transformed_derivs, t_path_size * sizeof(T));
			branched_sig_backprop_cuda_core_<T>(d_transformed, d_transformed_derivs, core_derivs, core_bsig,
				batch_size, t_dimension, t_length, max_nodes, planar, dimension, correction, correction_len,
				correction_batch_stride, correction_segment_stride);

			cudaFree(d_transformed);
			d_transformed = nullptr;

			cu_transform_path_backprop_<T>(d_transformed_derivs, out, batch_size, dimension, length, time_aug, lead_lag, end_time);
			cudaFree(d_transformed_derivs);
		} catch (...) {
			if (d_transformed) cudaFree(d_transformed);
			if (d_transformed_derivs) cudaFree(d_transformed_derivs);
			throw;
		}
	}
	else {
		branched_sig_backprop_cuda_core_<T>(path, out, core_derivs, core_bsig,
			batch_size, dimension, length, max_nodes, planar, dimension, correction, correction_len,
			correction_batch_stride, correction_segment_stride);
	}
}

// =========================================================================
// extern "C" wrappers
// =========================================================================

extern "C" {

	CUSIG_API int prepare_branched_sig_cuda(uint64_t dimension, uint64_t max_nodes, bool planar, bool use_disk) noexcept {
		CUSIG_SAFE_CALL(prepare_branched_sig_gpu_cache_(dimension, max_nodes, planar, use_disk));
	}

	CUSIG_API int prepare_branched_log_sig_cuda(
		uint64_t dimension, uint64_t max_nodes, bool planar, bool use_disk
	) noexcept {
		CUSIG_SAFE_CALL(
			prepare_branched_sig_gpu_cache_(dimension, max_nodes, planar, use_disk);
			prepare_cuda_branched_log_sig_gpu_cache_(dimension, max_nodes, planar)
		);
	}

	CUSIG_API int prepare_branched_sig_coef_cuda(const uint64_t* tree_data, uint64_t tree_data_len, uint64_t data_dimension, uint64_t dimension, uint64_t max_nodes, bool planar, bool use_disk) noexcept {
		CUSIG_SAFE_CALL(prepare_branched_sig_coef_cache_cuda_(tree_data, tree_data_len, data_dimension, dimension, max_nodes, planar, use_disk));
	}

	CUSIG_API int branched_sig_coef_cuda_f(const float* path, float* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_coef_cuda_<float>(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_coef_cuda_d(const double* path, double* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_coef_cuda_<double>(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_coef_backprop_cuda_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_coef_backprop_cuda_<float>(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_coef_backprop_cuda_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_coef_backprop_cuda_<double>(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_cuda_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, bool scalar_term, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_cuda_<float>(path, out, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_cuda_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, bool scalar_term, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_cuda_<double>(path, out, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_combine_cuda_f(const float* bsig1, const float* bsig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_cuda_<float>(bsig1, bsig2, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}
	CUSIG_API int branched_sig_combine_cuda_d(const double* bsig1, const double* bsig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_cuda_<double>(bsig1, bsig2, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}

	CUSIG_API int branched_sig_combine_backprop_cuda_f(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_backprop_cuda_<float>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, planar, scalar_term));
	}
	CUSIG_API int branched_sig_combine_backprop_cuda_d(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_combine_backprop_cuda_<double>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, planar, scalar_term));
	}


	CUSIG_API int branched_sig_backprop_cuda_f(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, float end_time, bool planar, bool scalar_term, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_backprop_cuda_<float>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CUSIG_API int branched_sig_backprop_cuda_d(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug, bool lead_lag, double end_time, bool planar, bool scalar_term, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		CUSIG_SAFE_CALL(branched_sig_backprop_cuda_<double>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, time_aug, lead_lag, end_time, planar, scalar_term, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

}
