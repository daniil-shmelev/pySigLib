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
#include "cu_atomic.h"
#include "cu_branched_log_sig_cache.h"
#include "cu_log_sig_cache.h"
#include "cu_macros.h"
#include "cu_utils.h"
#include "../shared/branched_cache.h"
#include "../shared/branched_log_cache.h"

#include <cstdint>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct BranchedLogSigCacheGPU {
	// All arrays are uint32 CSR data uploaded from BranchedLogProductCache.
	// The kernels use this layout for expanded method 0 and as scratch for
	// compact methods 1 and 2.
	uint32_t* d_forest_offsets32 = nullptr;
	uint32_t* d_forest_trees32 = nullptr;
	uint32_t* d_forest_coprod_offsets32 = nullptr;
	uint32_t* d_forest_coprod_data32 = nullptr;
	uint32_t* d_single_tree_forest32 = nullptr;

	uint32_t total_length = 0;
	uint32_t num_trees = 0;
	uint32_t num_forests = 0;
	uint32_t forest_trees_len = 0;
	uint32_t forest_coprod_data_len = 0;
	int max_nodes = 0;

	BranchedLogSigCacheGPU() = default;
	BranchedLogSigCacheGPU(const BranchedLogSigCacheGPU&) = delete;
	BranchedLogSigCacheGPU& operator=(const BranchedLogSigCacheGPU&) = delete;

	~BranchedLogSigCacheGPU() {
		if (d_forest_offsets32) cudaFree(d_forest_offsets32);
		if (d_forest_trees32) cudaFree(d_forest_trees32);
		if (d_forest_coprod_offsets32) cudaFree(d_forest_coprod_offsets32);
		if (d_forest_coprod_data32) cudaFree(d_forest_coprod_data32);
		if (d_single_tree_forest32) cudaFree(d_single_tree_forest32);
	}
};

struct CuBranchedLogCacheKey {
	// CUDA allocations are local to the active device.
	// The same mathematical cache therefore has one entry per device.
	int device = 0;
	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;

	bool operator==(const CuBranchedLogCacheKey& other) const noexcept {
		return device == other.device
			&& dimension == other.dimension
			&& max_nodes == other.max_nodes
			&& planar == other.planar;
	}
};

struct CuBranchedLogCacheKeyHash {
	size_t operator()(const CuBranchedLogCacheKey& key) const noexcept {
		size_t h = std::hash<int>{}(key.device);
		auto combine = [&h](uint64_t value) {
			h ^= std::hash<uint64_t>{}(value) + 0x9e3779b9ULL
				+ (h << 6) + (h >> 2);
		};
		combine(key.dimension);
		combine(key.max_nodes);
		combine(static_cast<uint64_t>(key.planar));
		return h;
	}
};

static CuBranchedLogCacheKey make_cu_branched_log_key(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	CuBranchedLogCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.dimension = dimension;
	key.max_nodes = max_nodes;
	key.planar = planar;
	return key;
}

static std::unordered_map<
	CuBranchedLogCacheKey,
	std::unique_ptr<BranchedLogSigCacheGPU>,
	CuBranchedLogCacheKeyHash
> s_branched_log_gpu_cache_map;
static std::mutex s_branched_log_gpu_cache_mu;

static constexpr const char* cu_mkw_basis_cache_prefix_ = "mkw_lyndon_";

static std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<CuMkwHostBasisData>,
	CuPairHash
> s_mkw_host_basis_cache_map;
static std::mutex s_mkw_host_basis_cache_mu;

static std::unordered_map<
	CuBranchedLogCacheKey,
	std::unique_ptr<CuMkwBasisGpuCache>,
	CuBranchedLogCacheKeyHash
> s_mkw_gpu_basis_cache_map;
static std::mutex s_mkw_gpu_basis_cache_mu;

template<typename T>
static void upload_branched_log(T*& d_ptr, const T* h_data, size_t count);

CuMkwBasisGpuCache::~CuMkwBasisGpuCache() {
	if (d_lyndon_idx) cudaFree(d_lyndon_idx);
	if (d_sparse_vals) cudaFree(d_sparse_vals);
	if (d_sparse_cols) cudaFree(d_sparse_cols);
	if (d_sparse_row_ptr) cudaFree(d_sparse_row_ptr);
	if (d_sparse_vals_t) cudaFree(d_sparse_vals_t);
	if (d_sparse_cols_t) cudaFree(d_sparse_cols_t);
	if (d_sparse_row_ptr_t) cudaFree(d_sparse_row_ptr_t);
}

static std::unique_ptr<CuMkwHostBasisData> build_mkw_word_data_(
	const BranchedSigCache& cache
) {
	auto data = std::make_unique<CuMkwHostBasisData>();
	// Reconstruct forest words from the portable flattened branched cache.
	// The CPU and CUDA builders must select the same Lyndon word ordering.
	const uint64_t lyndon_count = compute_branched_log_sig_length(
		cache.dimension, cache.max_nodes, true);
	data->flat_words.resize(cache.total_length);
	data->flat_idx.reserve(cache.total_length);
	data->lyndon_words.reserve(lyndon_count);
	data->lyndon_idx.reserve(lyndon_count);
	data->lyndon_weights.reserve(lyndon_count);
	for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
		const uint64_t start = cache.basis_forest_offsets[basis_idx];
		const uint64_t end = cache.basis_forest_offsets[basis_idx + 1];
		cu_word forest(
			cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(start),
			cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(end));
		data->flat_words[basis_idx + 1] = forest;
		if (cu_is_lyndon(forest)) {
			if (forest.size() == 1) {
				if (data->lyndon_words.size() > UINT32_MAX)
					throw std::overflow_error("MKW Lyndon letter index exceeds uint32 range");
				data->letter_log_idx.push_back(
					static_cast<uint32_t>(data->lyndon_words.size()));
				data->letter_basis_idx.push_back(basis_idx);
			}
			data->lyndon_words.push_back(forest);
			data->lyndon_idx.push_back(basis_idx + 1);
			data->lyndon_weights.push_back(
				cache.node_labels_offsets[basis_idx + 1]
				- cache.node_labels_offsets[basis_idx]);
		}
	}
	if (data->lyndon_idx.size() != lyndon_count)
		throw std::runtime_error("MKW Lyndon cache length mismatch");
	for (uint64_t i = 0; i < data->flat_words.size(); ++i)
		data->flat_idx[data->flat_words[i]] = i;

	std::unordered_set<cu_word, CuWordHash> lyndon_set(
		data->lyndon_words.begin(), data->lyndon_words.end());
	std::unordered_map<cu_word, uint64_t, CuWordHash> lyndon_map;
	lyndon_map.reserve(lyndon_count);
	for (uint64_t i = 0; i < lyndon_count; ++i)
		lyndon_map[data->lyndon_words[i]] = i;
	data->left_factor.assign(lyndon_count, UINT64_MAX);
	data->right_factor.assign(lyndon_count, UINT64_MAX);
	for (uint64_t i = 0; i < lyndon_count; ++i) {
		const cu_word& w = data->lyndon_words[i];
		if (w.size() == 1)
			continue;
		const cu_word right = cu_longest_lyndon_suffix_(w, lyndon_set);
		const cu_word left(w.begin(), w.end() - right.size());
		data->left_factor[i] = lyndon_map.at(left);
		data->right_factor[i] = lyndon_map.at(right);
	}
	return data;
}

static void build_mkw_projection_(
	CuMkwHostBasisData& data,
	uint64_t flat_word_count
) {
	// Build once on the host, then upload the inverse and its transpose as CSR.
	// The unit triangular matrix maps bracket coefficients to Lyndon coordinates.
	CuSparseIntMatrix projection;
	cu_lyndon_proj_matrix_from_words(
		projection,
		data.lyndon_words,
		flat_word_count,
		[&data](const cu_word& w) {
			return data.flat_idx.at(w);
		},
		[&data](uint64_t i, uint64_t j, uint64_t) {
			cu_word product = data.flat_words.at(i);
			const cu_word& right = data.flat_words.at(j);
			product.insert(product.end(), right.begin(), right.end());
			return data.flat_idx.at(product);
		});
	projection.inverse(data.inv_proj_mat);
	data.inv_proj_mat.transpose(data.inv_proj_mat_t);
}

static void prepare_cuda_mkw_host_basis_cache_(
	const BranchedSigCache& cache,
	int method,
	bool use_disk
) {
	const int basis_method = std::min(method, 2);
	const std::pair<uint64_t, uint64_t> key(cache.dimension, cache.max_nodes);
	{
		std::lock_guard<std::mutex> lock(s_mkw_host_basis_cache_mu);
		auto found = s_mkw_host_basis_cache_map.find(key);
		if (found != s_mkw_host_basis_cache_map.end()
			&& found->second->method >= basis_method)
			return;
	}

	auto data = build_mkw_word_data_(cache);
	bool loaded = false;
	std::unique_ptr<CuCacheFile> file;
	if (use_disk) {
		ensure_cuda_cache_dir_();
		file = std::make_unique<CuCacheFile>(
			cache.dimension, cache.max_nodes, cu_mkw_basis_cache_prefix_);
	}
	if (file && file->exists()) {
		int disk_method = 0;
		std::vector<uint64_t> disk_lyndon_idx;
		CuSparseIntMatrix disk_inverse;
		CuSparseIntMatrix disk_inverse_t;
		file->read(
			disk_method, disk_lyndon_idx, disk_inverse, disk_inverse_t);
		if (disk_lyndon_idx != data->lyndon_idx)
			throw corrupted_cache_error(
				"MKW Lyndon cache indices do not match the branched basis");
		if (disk_method >= basis_method) {
			if (disk_method >= 2
				&& (disk_inverse.n != data->lyndon_idx.size()
					|| disk_inverse.m != data->lyndon_idx.size()
					|| disk_inverse_t.n != data->lyndon_idx.size()
					|| disk_inverse_t.m != data->lyndon_idx.size()))
				throw corrupted_cache_error(
					"MKW Lyndon cache matrix has an invalid shape");
			data->method = std::min(disk_method, 2);
			data->inv_proj_mat = std::move(disk_inverse);
			data->inv_proj_mat_t = std::move(disk_inverse_t);
			loaded = true;
		}
	}
	if (!loaded) {
		// Method 1 has only Lyndon indices. Method 2 adds the inverse projection.
		// A cached method 2 entry remains usable for later method 1 preparation.
		data->method = basis_method;
		if (basis_method == 2)
			build_mkw_projection_(*data, cache.total_length);
		if (file)
			file->write(
				data->method, data->lyndon_idx,
				data->inv_proj_mat, data->inv_proj_mat_t);
	}

	std::lock_guard<std::mutex> lock(s_mkw_host_basis_cache_mu);
	auto found = s_mkw_host_basis_cache_map.find(key);
	if (found == s_mkw_host_basis_cache_map.end()
		|| found->second->method < data->method)
		s_mkw_host_basis_cache_map.insert_or_assign(key, std::move(data));
}

const CuMkwHostBasisData& get_cuda_mkw_host_basis_data_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method
) {
	const int basis_method = std::min(method, 2);
	const std::pair<uint64_t, uint64_t> key(dimension, max_nodes);
	std::lock_guard<std::mutex> lock(s_mkw_host_basis_cache_mu);
	auto found = s_mkw_host_basis_cache_map.find(key);
	if (found == s_mkw_host_basis_cache_map.end()
		|| found->second->method < basis_method)
		throw cache_not_found_error(
			"CUDA MKW basis cache not found - call prepare_branched_log_sig first");
	return *found->second;
}

static uint32_t narrow_mkw_u32_(uint64_t value, const char* label) {
	if (value > UINT32_MAX)
		throw std::overflow_error(std::string(label) + " exceeds uint32 range");
	return static_cast<uint32_t>(value);
}

static void upload_mkw_csr_(
	const CuSparseIntMatrix& matrix,
	int*& d_values,
	uint32_t*& d_columns,
	uint32_t*& d_row_offsets
) {
	const uint32_t row_count = narrow_mkw_u32_(matrix.n, "MKW matrix row count");
	uint64_t nnz64 = 0;
	for (const auto& row : matrix.rows) {
		if (row.size() > UINT32_MAX
			|| nnz64 > UINT32_MAX - static_cast<uint64_t>(row.size()))
			throw std::overflow_error("MKW matrix entry count exceeds uint32 range");
		nnz64 += row.size();
	}
	const uint32_t nnz = static_cast<uint32_t>(nnz64);
	std::vector<int> values(nnz);
	std::vector<uint32_t> columns(nnz);
	std::vector<uint32_t> row_offsets(static_cast<uint64_t>(row_count) + 1);
	uint32_t entry = 0;
	for (uint32_t row = 0; row < row_count; ++row) {
		row_offsets[row] = entry;
		for (const CuEntry& item : matrix.rows[row]) {
			values[entry] = item.val;
			columns[entry] = narrow_mkw_u32_(
				item.col, "MKW matrix column index");
			++entry;
		}
	}
	row_offsets[row_count] = entry;

	CudaBuf<int> value_buffer;
	CudaBuf<uint32_t> column_buffer;
	if (nnz != 0) {
		value_buffer = CudaBuf<int>(static_cast<size_t>(nnz) * sizeof(int));
		column_buffer = CudaBuf<uint32_t>(
			static_cast<size_t>(nnz) * sizeof(uint32_t));
		CUDA_CHECK(cudaMemcpy(
			value_buffer.get(), values.data(),
			static_cast<size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(
			column_buffer.get(), columns.data(),
			static_cast<size_t>(nnz) * sizeof(uint32_t), cudaMemcpyHostToDevice));
	}
	CudaBuf<uint32_t> row_buffer(
		(static_cast<size_t>(row_count) + 1) * sizeof(uint32_t));
	CUDA_CHECK(cudaMemcpy(
		row_buffer.get(), row_offsets.data(),
		(static_cast<size_t>(row_count) + 1) * sizeof(uint32_t),
		cudaMemcpyHostToDevice));
	d_values = value_buffer.release();
	d_columns = column_buffer.release();
	d_row_offsets = row_buffer.release();
}

static void upload_mkw_projection_(
	CuMkwBasisGpuCache& gpu,
	const CuMkwHostBasisData& host
) {
	// Keep both orientations because forward and backward use opposite products.
	// Values stay signed integers because the projection is integral.
	int* values = nullptr;
	uint32_t* columns = nullptr;
	uint32_t* row_offsets = nullptr;
	int* values_t = nullptr;
	uint32_t* columns_t = nullptr;
	uint32_t* row_offsets_t = nullptr;
	try {
		upload_mkw_csr_(
			host.inv_proj_mat, values, columns, row_offsets);
		upload_mkw_csr_(
			host.inv_proj_mat_t, values_t, columns_t, row_offsets_t);
	} catch (...) {
		if (values) cudaFree(values);
		if (columns) cudaFree(columns);
		if (row_offsets) cudaFree(row_offsets);
		if (values_t) cudaFree(values_t);
		if (columns_t) cudaFree(columns_t);
		if (row_offsets_t) cudaFree(row_offsets_t);
		throw;
	}
	gpu.d_sparse_vals = values;
	gpu.d_sparse_cols = columns;
	gpu.d_sparse_row_ptr = row_offsets;
	gpu.d_sparse_vals_t = values_t;
	gpu.d_sparse_cols_t = columns_t;
	gpu.d_sparse_row_ptr_t = row_offsets_t;
	gpu.method = 2;
}

void prepare_cuda_mkw_basis_cache_(
	const BranchedSigCache& cache,
	int method,
	bool use_disk
) {
	if (method < 1 || method > 3)
		throw std::invalid_argument(
			"branched log signature method must be 0, 1, 2, or 3");
	if (!cache.planar)
		throw std::invalid_argument(
			"compressed branched log signatures require planar=True");
	const int basis_method = std::min(method, 2);
	prepare_cuda_mkw_host_basis_cache_(cache, basis_method, use_disk);
	const CuMkwHostBasisData& host = get_cuda_mkw_host_basis_data_(
		cache.dimension, cache.max_nodes, basis_method);
	const auto key = make_cu_branched_log_key(
		cache.dimension, cache.max_nodes, true);
	std::lock_guard<std::mutex> lock(s_mkw_gpu_basis_cache_mu);
	auto found = s_mkw_gpu_basis_cache_map.find(key);
	if (found != s_mkw_gpu_basis_cache_map.end()) {
		if (host.method >= 2 && found->second->method < 2)
			upload_mkw_projection_(*found->second, host);
		return;
	}

	auto gpu = std::make_unique<CuMkwBasisGpuCache>();
	gpu->compact_length = narrow_mkw_u32_(
		host.lyndon_idx.size(), "MKW basis length");
	if (gpu->compact_length != 0) {
		std::vector<uint32_t> indices(gpu->compact_length);
		for (uint32_t i = 0; i < gpu->compact_length; ++i)
			indices[i] = narrow_mkw_u32_(
				host.lyndon_idx[i], "MKW Lyndon flat index");
		upload_branched_log(
			gpu->d_lyndon_idx, indices.data(), indices.size());
	}
	gpu->method = 1;
	if (host.method >= 2)
		upload_mkw_projection_(*gpu, host);
	s_mkw_gpu_basis_cache_map.try_emplace(key, std::move(gpu));
}

const CuMkwBasisGpuCache& get_cuda_mkw_basis_gpu_cache_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method
) {
	const int basis_method = std::min(method, 2);
	const auto key = make_cu_branched_log_key(dimension, max_nodes, true);
	std::lock_guard<std::mutex> lock(s_mkw_gpu_basis_cache_mu);
	auto found = s_mkw_gpu_basis_cache_map.find(key);
	if (found == s_mkw_gpu_basis_cache_map.end()
		|| found->second->method < basis_method)
		throw cache_not_found_error(
			"CUDA MKW basis cache not found - call prepare_branched_log_sig first");
	return *found->second;
}

void release_branched_log_sig_gpu_state() {
	{
		std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
		s_branched_log_gpu_cache_map.clear();
	}
	{
		std::lock_guard<std::mutex> lock(s_mkw_gpu_basis_cache_mu);
		s_mkw_gpu_basis_cache_map.clear();
	}
	std::lock_guard<std::mutex> lock(s_mkw_host_basis_cache_mu);
	s_mkw_host_basis_cache_map.clear();
}

void clear_cuda_branched_log_sig_gpu_cache_() {
	release_branched_log_sig_gpu_state();
}

bool is_cuda_branched_log_sig_gpu_cache_prepared_(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	const auto key = make_cu_branched_log_key(dimension, max_nodes, planar);
	std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
	return s_branched_log_gpu_cache_map.find(key)
		!= s_branched_log_gpu_cache_map.end();
}

template<typename T>
static void upload_branched_log(T*& d_ptr, const T* h_data, size_t count) {
	CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_ptr, h_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

void prepare_cuda_branched_log_sig_gpu_cache_(
	const BranchedSigCache& c
) {
	const uint64_t dimension = c.dimension;
	const uint64_t max_nodes = c.max_nodes;
	const bool planar = c.planar;
	const auto key = make_cu_branched_log_key(dimension, max_nodes, planar);
	{
		std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
		auto it = s_branched_log_gpu_cache_map.find(key);
		if (it != s_branched_log_gpu_cache_map.end())
			return;
	}

	BranchedLogProductCache product_cache = build_branched_log_product_cache(c);
	// Kernels use uint32 indices, so validate every host offset before upload.
	// The host cache remains uint64 so CPU and CUDA use the same disk format.
	uint64_t num_trees = c.total_length - 1;

	auto safe_narrow = [](const uint64_t* src, uint32_t* dst, size_t n) {
		for (size_t i = 0; i < n; ++i) {
			if (src[i] > UINT32_MAX)
				throw std::overflow_error("Branched log sig cache value exceeds uint32 range");
			dst[i] = static_cast<uint32_t>(src[i]);
		}
	};

	std::vector<uint32_t> forest_offsets32(product_cache.product_offsets.size());
	std::vector<uint32_t> forest_trees32(product_cache.product_factors.size());
	std::vector<uint32_t> forest_coprod_offsets32(product_cache.coproduct_offsets.size());
	std::vector<uint32_t> forest_coprod_data32(product_cache.coproduct_pairs.size());
	std::vector<uint32_t> single_tree_forest32(product_cache.flat_to_product.size());
	safe_narrow(
		product_cache.product_offsets.data(), forest_offsets32.data(),
		product_cache.product_offsets.size());
	safe_narrow(
		product_cache.product_factors.data(), forest_trees32.data(),
		product_cache.product_factors.size());
	safe_narrow(
		product_cache.coproduct_offsets.data(), forest_coprod_offsets32.data(),
		product_cache.coproduct_offsets.size());
	safe_narrow(
		product_cache.coproduct_pairs.data(), forest_coprod_data32.data(),
		product_cache.coproduct_pairs.size());
	safe_narrow(
		product_cache.flat_to_product.data(), single_tree_forest32.data(),
		product_cache.flat_to_product.size());

	auto narrow32 = [](uint64_t v) -> uint32_t {
		if (v > UINT32_MAX) throw std::overflow_error("Branched log sig cache value exceeds uint32 range");
		return static_cast<uint32_t>(v);
	};

	auto gpu = std::make_unique<BranchedLogSigCacheGPU>();
	gpu->total_length = narrow32(c.total_length);
	gpu->num_trees = narrow32(num_trees);
	gpu->num_forests = narrow32(product_cache.product_offsets.size() - 1);
	gpu->forest_trees_len = narrow32(product_cache.product_factors.size());
	gpu->forest_coprod_data_len = narrow32(product_cache.coproduct_pairs.size());
	gpu->max_nodes = static_cast<int>(max_nodes);
	upload_branched_log(gpu->d_forest_offsets32, forest_offsets32.data(), forest_offsets32.size());
	upload_branched_log(gpu->d_forest_trees32, forest_trees32.data(), forest_trees32.size());
	upload_branched_log(gpu->d_forest_coprod_offsets32, forest_coprod_offsets32.data(), forest_coprod_offsets32.size());
	upload_branched_log(gpu->d_forest_coprod_data32, forest_coprod_data32.data(), forest_coprod_data32.size());
	upload_branched_log(gpu->d_single_tree_forest32, single_tree_forest32.data(), single_tree_forest32.size());

	std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
	s_branched_log_gpu_cache_map.try_emplace(key, std::move(gpu));
}

static const BranchedLogSigCacheGPU& get_branched_log_gpu_cache(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	const auto key = make_cu_branched_log_key(dimension, max_nodes, planar);
	std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
	auto it = s_branched_log_gpu_cache_map.find(key);
	if (it != s_branched_log_gpu_cache_map.end())
		return *(it->second);
	throw cache_not_found_error(
		"CUDA branched log sig cache not found - call prepare_branched_log_sig with device='cuda' first");
}

template<typename T, int Method>
__global__ __launch_bounds__(1024)
void branched_sig_to_log_sig_ker(
	const T* __restrict__ bsig,
	T* __restrict__ out,
	uint32_t total_len,
	uint32_t num_forests,
	const uint32_t* __restrict__ g_forest_offsets,
	const uint32_t* __restrict__ g_forest_trees,
	const uint32_t* __restrict__ g_forest_coprod_offsets,
	const uint32_t* __restrict__ g_forest_coprod_data,
	const uint32_t* __restrict__ g_single_tree_forest,
	int max_nodes,
	bool scalar_term,
	const uint32_t* __restrict__ g_lyndon_idx,
	uint32_t compact_len,
	const int* __restrict__ g_sparse_vals,
	const uint32_t* __restrict__ g_sparse_cols,
	const uint32_t* __restrict__ g_sparse_row_ptr,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;
	const uint64_t input_stride = scalar_term ? total_len : total_len - 1;
	const uint64_t output_stride = Method == 0 ? input_stride : compact_len;

	extern __shared__ char smem[];
	T* h = reinterpret_cast<T*>(smem);
	T* h_forest = h + total_len;
	T* power = h_forest + num_forests;
	T* next_power = power + num_forests;
	T* full_out = next_power + num_forests;
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
		full_out[i] = T(0);
	}
	for (uint32_t i = tid; i < num_forests; i += blockDim.x) {
		h_forest[i] = T(0);
		power[i] = T(0);
		next_power[i] = T(0);
	}
	__syncthreads();

	const T* src = bsig + static_cast<uint64_t>(batch_idx) * input_stride;
	T* dst = out + static_cast<uint64_t>(batch_idx) * output_stride;
	for (uint32_t i = tid; i < num_trees; i += blockDim.x) {
		const T v = scalar_term ? src[i + 1] : src[i];
		h[i + 1] = v;
	}
	__syncthreads();

	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x) {
		T val = T(1);
		const uint32_t start = g_forest_offsets[forest_idx];
		const uint32_t end = g_forest_offsets[forest_idx + 1];
		for (uint32_t pos = start; pos < end; ++pos)
			val *= h[g_forest_trees[pos]];
		h_forest[forest_idx] = val;
		power[forest_idx] = val;
	}
	__syncthreads();

	for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
		full_out[i] = power[g_single_tree_forest[i]];
	__syncthreads();

	T* cur = power;
	T* next = next_power;
	for (int k = 2; k <= max_nodes; ++k) {
		for (uint32_t forest_idx = tid; forest_idx < num_forests; forest_idx += blockDim.x) {
			T val = T(0);
			const uint32_t start = g_forest_coprod_offsets[forest_idx];
			const uint32_t end = g_forest_coprod_offsets[forest_idx + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = g_forest_coprod_data[pos];
				const uint32_t right = g_forest_coprod_data[pos + 1];
				val += cur[left] * h_forest[right];
			}
			next[forest_idx] = val;
		}
		__syncthreads();

		const T coeff = (k % 2 == 0) ? T(-1) / T(k) : T(1) / T(k);
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			full_out[i] += coeff * next[g_single_tree_forest[i]];
		__syncthreads();

		T* tmp = cur;
		cur = next;
		next = tmp;
	}

	if constexpr (Method == 0) {
		if (scalar_term) {
			for (uint32_t i = tid; i < total_len; i += blockDim.x)
				dst[i] = (i == 0) ? T(0) : full_out[i];
		} else {
			for (uint32_t i = tid; i < num_trees; i += blockDim.x)
				dst[i] = full_out[i + 1];
		}
	} else if constexpr (Method == 1) {
		for (uint32_t i = tid; i < compact_len; i += blockDim.x)
			dst[i] = full_out[g_lyndon_idx[i]];
	} else {
		for (uint32_t i = tid; i < compact_len; i += blockDim.x)
			h[i] = full_out[g_lyndon_idx[i]];
		__syncthreads();
		for (uint32_t i = tid; i < compact_len; i += blockDim.x) {
			T value = h[i];
			const uint32_t start = g_sparse_row_ptr[i];
			const uint32_t end = g_sparse_row_ptr[i + 1];
			for (uint32_t pos = start; pos < end; ++pos)
				value += static_cast<T>(g_sparse_vals[pos])
					* h[g_sparse_cols[pos]];
			dst[i] = value;
		}
	}
}

template<typename T, int Method>
__global__ __launch_bounds__(1024)
void branched_sig_to_log_sig_backprop_ker(
	const T* __restrict__ bsig,
	const T* __restrict__ derivs,
	T* __restrict__ out,
	uint32_t total_len,
	uint32_t num_forests,
	const uint32_t* __restrict__ g_forest_offsets,
	const uint32_t* __restrict__ g_forest_trees,
	const uint32_t* __restrict__ g_forest_coprod_offsets,
	const uint32_t* __restrict__ g_forest_coprod_data,
	const uint32_t* __restrict__ g_single_tree_forest,
	T* __restrict__ workspace,
	int max_nodes,
	bool scalar_term,
	const uint32_t* __restrict__ g_lyndon_idx,
	uint32_t compact_len,
	const int* __restrict__ g_sparse_vals_t,
	const uint32_t* __restrict__ g_sparse_cols_t,
	const uint32_t* __restrict__ g_sparse_row_ptr_t,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t batch_idx = batch_offset + local_batch_idx;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;
	const uint32_t levels = static_cast<uint32_t>(max_nodes + 1);
	const uint64_t input_stride = scalar_term ? total_len : total_len - 1;
	const uint64_t deriv_stride = Method == 0 ? input_stride : compact_len;

	extern __shared__ char smem[];
	const uint64_t powers_size = static_cast<uint64_t>(levels) * num_forests;
	T* powers = workspace + local_batch_idx * 2 * powers_size;
	T* d_powers = powers + powers_size;
	T* h = reinterpret_cast<T*>(smem);
	T* h_forest = h + total_len;
	T* d_h_forest = h_forest + num_forests;
	T* d_h_tree = d_h_forest + num_forests;
	T* full_derivs = d_h_tree + total_len;

	for (uint64_t i = tid; i < powers_size; i += blockDim.x) {
		powers[i] = T(0);
		d_powers[i] = T(0);
	}
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
		d_h_tree[i] = T(0);
		full_derivs[i] = T(0);
	}
	for (uint32_t i = tid; i < num_forests; i += blockDim.x) {
		h_forest[i] = T(0);
		d_h_forest[i] = T(0);
	}
	__syncthreads();

	const T* src = bsig + static_cast<uint64_t>(batch_idx) * input_stride;
	const T* dsrc = derivs + static_cast<uint64_t>(batch_idx) * deriv_stride;
	T* dst = out + static_cast<uint64_t>(batch_idx) * input_stride;

	for (uint32_t i = tid; i < num_trees; i += blockDim.x) {
		const T v = scalar_term ? src[i + 1] : src[i];
		h[i + 1] = v;
	}
	if constexpr (Method == 0) {
		for (uint32_t i = tid; i < num_trees; i += blockDim.x)
			full_derivs[i + 1] = scalar_term ? dsrc[i + 1] : dsrc[i];
	} else if constexpr (Method == 1) {
		for (uint32_t i = tid; i < compact_len; i += blockDim.x)
			full_derivs[g_lyndon_idx[i]] = dsrc[i];
	} else {
		for (uint32_t i = tid; i < compact_len; i += blockDim.x) {
			T value = dsrc[i];
			const uint32_t start = g_sparse_row_ptr_t[i];
			const uint32_t end = g_sparse_row_ptr_t[i + 1];
			for (uint32_t pos = start; pos < end; ++pos)
				value += static_cast<T>(g_sparse_vals_t[pos])
					* dsrc[g_sparse_cols_t[pos]];
			full_derivs[g_lyndon_idx[i]] = value;
		}
	}
	__syncthreads();

	T* p1 = powers + num_forests;
	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x) {
		T val = T(1);
		const uint32_t start = g_forest_offsets[forest_idx];
		const uint32_t end = g_forest_offsets[forest_idx + 1];
		for (uint32_t pos = start; pos < end; ++pos)
			val *= h[g_forest_trees[pos]];
		h_forest[forest_idx] = val;
		p1[forest_idx] = val;
	}
	__syncthreads();

	for (int k = 2; k <= max_nodes; ++k) {
		T* prev = powers + static_cast<uint64_t>(k - 1) * num_forests;
		T* next = powers + static_cast<uint64_t>(k) * num_forests;
		for (uint32_t forest_idx = tid; forest_idx < num_forests; forest_idx += blockDim.x) {
			T val = T(0);
			const uint32_t start = g_forest_coprod_offsets[forest_idx];
			const uint32_t end = g_forest_coprod_offsets[forest_idx + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = g_forest_coprod_data[pos];
				const uint32_t right = g_forest_coprod_data[pos + 1];
				val += prev[left] * h_forest[right];
			}
			next[forest_idx] = val;
		}
		__syncthreads();
	}

	for (int k = 1; k <= max_nodes; ++k) {
		const T coeff = (k == 1) ? T(1) : ((k % 2 == 0) ? T(-1) / T(k) : T(1) / T(k));
		T* dk = d_powers + static_cast<uint64_t>(k) * num_forests;
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			dk[g_single_tree_forest[i]] += coeff * full_derivs[i];
	}
	__syncthreads();

	for (int k = max_nodes; k >= 2; --k) {
		const T* prev = powers + static_cast<uint64_t>(k - 1) * num_forests;
		const T* d_cur = d_powers + static_cast<uint64_t>(k) * num_forests;
		T* d_prev = d_powers + static_cast<uint64_t>(k - 1) * num_forests;
		for (uint32_t forest_idx = tid; forest_idx < num_forests; forest_idx += blockDim.x) {
			const T d = d_cur[forest_idx];
			if (d == T(0))
				continue;
			const uint32_t start = g_forest_coprod_offsets[forest_idx];
			const uint32_t end = g_forest_coprod_offsets[forest_idx + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = g_forest_coprod_data[pos];
				const uint32_t right = g_forest_coprod_data[pos + 1];
				myAtomicAdd(&d_prev[left], d * h_forest[right]);
				myAtomicAdd(&d_h_forest[right], d * prev[left]);
			}
		}
		__syncthreads();
	}

	T* d1 = d_powers + num_forests;
	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x)
		d_h_forest[forest_idx] += d1[forest_idx];
	__syncthreads();

	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x) {
		const T d = d_h_forest[forest_idx];
		if (d == T(0))
			continue;
		const uint32_t start = g_forest_offsets[forest_idx];
		const uint32_t end = g_forest_offsets[forest_idx + 1];
		for (uint32_t pos = start; pos < end; ++pos) {
			T partial = d;
			for (uint32_t other = start; other < end; ++other) {
				if (other != pos)
					partial *= h[g_forest_trees[other]];
			}
			myAtomicAdd(&d_h_tree[g_forest_trees[pos]], partial);
		}
	}
	__syncthreads();
	if (tid == 0) d_h_tree[0] = T(0);
	__syncthreads();

	if (scalar_term) {
		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			dst[i] = d_h_tree[i];
	} else {
		for (uint32_t i = tid; i < num_trees; i += blockDim.x)
			dst[i] = d_h_tree[i + 1];
	}
}

template<typename T, int Method>
static void branched_sig_to_log_sig_cuda_method_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	const BranchedLogSigCacheGPU& gc,
	const CuMkwBasisGpuCache* basis,
	bool scalar_term
) {
	const uint32_t compact_len = basis == nullptr ? 0 : basis->compact_length;
	if (batch_size == 0 || (Method != 0 && compact_len == 0))
		return;
	const uint32_t work_items = std::max(
		std::max(gc.num_trees, gc.num_forests), compact_len);
	unsigned int block = static_cast<unsigned int>((work_items + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	size_t smem = (2 * static_cast<uint64_t>(gc.total_length)
		+ 3 * static_cast<uint64_t>(gc.num_forests)) * sizeof(T);
	configure_dynamic_smem(
		branched_sig_to_log_sig_ker<T, Method>, smem, "CUDA branched log sig");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_to_log_sig_ker<T, Method><<<batch_chunk.grid, block, smem>>>(
			bsig, out, gc.total_length, gc.num_forests,
			gc.d_forest_offsets32, gc.d_forest_trees32,
			gc.d_forest_coprod_offsets32, gc.d_forest_coprod_data32,
			gc.d_single_tree_forest32,
			gc.max_nodes, scalar_term,
			basis == nullptr ? nullptr : basis->d_lyndon_idx,
			compact_len,
			basis == nullptr ? nullptr : basis->d_sparse_vals,
			basis == nullptr ? nullptr : basis->d_sparse_cols,
			basis == nullptr ? nullptr : basis->d_sparse_row_ptr,
			batch_chunk.offset, batch_chunk.size);
		batch_offset += batch_chunk.size;
	}
	cudaDeviceSynchronize();
	check_cuda_error();
}

template<typename T>
void branched_sig_to_log_sig_cuda_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	bool planar,
	bool scalar_term
) {
	if (method == 3)
		throw std::invalid_argument(
			"method=3 is not supported in branched_sig_to_log_sig; use branched_log_sig instead");
	if (method < 0 || method > 2)
		throw std::invalid_argument(
			"branched log signature method must be 0, 1, 2, or 3");
	if (method != 0 && !planar)
		throw std::invalid_argument(
			"compressed branched log signatures require planar=True");
	const auto& gc = get_branched_log_gpu_cache(
		dimension, max_nodes, planar);
	if (method == 0) {
		branched_sig_to_log_sig_cuda_method_<T, 0>(
			bsig, out, batch_size, gc, nullptr, scalar_term);
		return;
	}
	const auto& basis = get_cuda_mkw_basis_gpu_cache_(
		dimension, max_nodes, method);
	if (method == 1)
		branched_sig_to_log_sig_cuda_method_<T, 1>(
			bsig, out, batch_size, gc, &basis, scalar_term);
	else
		branched_sig_to_log_sig_cuda_method_<T, 2>(
			bsig, out, batch_size, gc, &basis, scalar_term);
}

template<typename T, int Method>
static void branched_sig_to_log_sig_backprop_cuda_method_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	const BranchedLogSigCacheGPU& gc,
	const CuMkwBasisGpuCache* basis,
	bool scalar_term
) {
	const uint32_t compact_len = basis == nullptr ? 0 : basis->compact_length;
	if (batch_size == 0)
		return;
	if (Method != 0 && compact_len == 0) {
		const uint64_t input_stride = scalar_term
			? gc.total_length : gc.total_length - 1;
		if (input_stride != 0) {
			CUDA_CHECK(cudaMemset(
				out, 0, batch_size * input_stride * sizeof(T)));
			check_cuda_kernel_launch();
		}
		return;
	}
	const uint32_t work_items = std::max(
		std::max(gc.num_trees, gc.num_forests), compact_len);
	unsigned int block = static_cast<unsigned int>((work_items + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	const uint64_t levels = static_cast<uint64_t>(gc.max_nodes + 1);
	const uint64_t power_elements = checked_branched_mul_(
		checked_branched_mul_(2, levels,
			"CUDA branched log sig backprop workspace size overflow"),
		gc.num_forests,
		"CUDA branched log sig backprop workspace size overflow");
	const uint64_t workspace_bytes = checked_branched_mul_(
		power_elements, sizeof(T),
		"CUDA branched log sig backprop workspace byte size overflow");
	size_t free_memory = 0;
	size_t total_memory = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
	const uint64_t reservation_bytes = checked_branched_mul_(
		2, workspace_bytes,
		"CUDA branched log sig backprop memory reservation overflow");
	uint64_t workspace_chunk_size = workspace_bytes == 0
		? batch_size
		: free_memory / reservation_bytes;
	workspace_chunk_size = std::max<uint64_t>(1, workspace_chunk_size);
	workspace_chunk_size = std::min<uint64_t>(
		std::min<uint64_t>(batch_size, workspace_chunk_size),
		CUDA_BATCH_GRID_CAPACITY);
	CudaBuf<T> workspace(checked_branched_mul_(
		workspace_chunk_size, workspace_bytes,
		"CUDA branched log sig backprop workspace allocation overflow"));

	size_t smem = (2 * static_cast<uint64_t>(gc.num_forests)
		+ 3 * static_cast<uint64_t>(gc.total_length)) * sizeof(T);
	configure_dynamic_smem(
		branched_sig_to_log_sig_backprop_ker<T, Method>, smem,
		"CUDA branched log sig backprop");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;
		batch_offset += workspace_chunk_size) {
		const uint64_t current_batch = std::min<uint64_t>(
			workspace_chunk_size, batch_size - batch_offset);
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, current_batch, 0);
		branched_sig_to_log_sig_backprop_ker<T, Method><<<batch_chunk.grid, block, smem>>>(
			bsig, derivs, out, gc.total_length, gc.num_forests,
			gc.d_forest_offsets32, gc.d_forest_trees32,
			gc.d_forest_coprod_offsets32, gc.d_forest_coprod_data32,
			gc.d_single_tree_forest32,
			workspace.get(),
			gc.max_nodes, scalar_term,
			basis == nullptr ? nullptr : basis->d_lyndon_idx,
			compact_len,
			basis == nullptr ? nullptr : basis->d_sparse_vals_t,
			basis == nullptr ? nullptr : basis->d_sparse_cols_t,
			basis == nullptr ? nullptr : basis->d_sparse_row_ptr_t,
			batch_offset, current_batch);
	}
	cudaDeviceSynchronize();
	check_cuda_error();
}

template<typename T>
void branched_sig_to_log_sig_backprop_cuda_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	bool planar,
	bool scalar_term
) {
	if (method == 3)
		throw std::invalid_argument(
			"method=3 is not supported in branched_sig_to_log_sig_backprop");
	if (method < 0 || method > 2)
		throw std::invalid_argument(
			"branched log signature method must be 0, 1, 2, or 3");
	if (method != 0 && !planar)
		throw std::invalid_argument(
			"compressed branched log signatures require planar=True");
	const auto& gc = get_branched_log_gpu_cache(
		dimension, max_nodes, planar);
	if (method == 0) {
		branched_sig_to_log_sig_backprop_cuda_method_<T, 0>(
			bsig, derivs, out, batch_size, gc, nullptr, scalar_term);
		return;
	}
	const auto& basis = get_cuda_mkw_basis_gpu_cache_(
		dimension, max_nodes, method);
	if (method == 1)
		branched_sig_to_log_sig_backprop_cuda_method_<T, 1>(
			bsig, derivs, out, batch_size, gc, &basis, scalar_term);
	else
		branched_sig_to_log_sig_backprop_cuda_method_<T, 2>(
			bsig, derivs, out, batch_size, gc, &basis, scalar_term);
}

extern "C" {

	CUSIG_API int branched_sig_to_log_sig_cuda_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_cuda_<float>(bsig, out, batch_size, dimension, max_nodes, method, planar, scalar_term));
	}

	CUSIG_API int branched_sig_to_log_sig_cuda_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_cuda_<double>(bsig, out, batch_size, dimension, max_nodes, method, planar, scalar_term));
	}

	CUSIG_API int branched_sig_to_log_sig_backprop_cuda_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_backprop_cuda_<float>(bsig, derivs, out, batch_size, dimension, max_nodes, method, planar, scalar_term));
	}

	CUSIG_API int branched_sig_to_log_sig_backprop_cuda_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_backprop_cuda_<double>(bsig, derivs, out, batch_size, dimension, max_nodes, method, planar, scalar_term));
	}

}
