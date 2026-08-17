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
#include "cache_lifecycle/cu_branched_log_sig_cache.h"
#include "cache_lifecycle/cu_log_sig_cache.h"
#include "cu_macros.h"
#include "cu_utils.h"
#include "../shared/preparation/branched_sig_cache.h"
#include "../shared/preparation/branched_log_plan.h"
#include "../shared/trees/basis_counts.h"

#include <cstdint>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<
	CuBranchedLogCacheKey,
	CudaBranchedLogSigCache,
	CuBranchedLogCacheKeyHash
>& get_cuda_branched_log_sig_cache_map_() {
	static std::unordered_map<
		CuBranchedLogCacheKey,
		CudaBranchedLogSigCache,
		CuBranchedLogCacheKeyHash
	> cache;
	return cache;
}

std::mutex& get_cuda_branched_log_sig_cache_mu_() {
	static std::mutex mu;
	return mu;
}

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

static uint32_t narrow_mkw_u32_(uint64_t value, const char* label) {
	if (value > UINT32_MAX)
		throw std::overflow_error(std::string(label) + " exceeds uint32 range");
	return static_cast<uint32_t>(value);
}

static void upload_mkw_csr_(
	const SparseIntMatrix& matrix,
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
		for (const Entry& item : matrix.rows[row]) {
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
	const BasisCache& host
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
			host.inv_proj_mat_transpose, values_t, columns_t, row_offsets_t);
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
	const BasisCache& host
) {
	if (!cache.planar)
		throw std::invalid_argument(
			"compressed branched log signatures require planar=True");
	const auto key = make_cuda_branched_log_cache_key_(
		cache.dimension, cache.max_nodes, true);
	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	auto [found, inserted] = get_cuda_branched_log_sig_cache_map_().try_emplace(key);
	if (found->second.basis) {
		if (host.method >= 2 && found->second.basis->method < 2)
			upload_mkw_projection_(*found->second.basis, host);
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
	found->second.basis = std::move(gpu);
}

const CuMkwBasisGpuCache& get_cuda_mkw_basis_gpu_cache_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method
) {
	const int basis_method = std::min(method, 2);
	const auto key = make_cuda_branched_log_cache_key_(dimension, max_nodes, true);
	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	auto found = get_cuda_branched_log_sig_cache_map_().find(key);
	if (found == get_cuda_branched_log_sig_cache_map_().end()
		|| !found->second.basis
		|| found->second.basis->method < basis_method)
		throw cache_not_found_error(
			"CUDA MKW basis cache not found - call prepare_branched_log_sig first");
	return *found->second.basis;
}

void release_branched_log_sig_gpu_state() {
	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	get_cuda_branched_log_sig_cache_map_().clear();
}

void clear_cuda_branched_log_sig_gpu_cache_() {
	release_branched_log_sig_gpu_state();
}

bool is_cuda_branched_log_horner_plan_prepared_(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	const auto key = make_cuda_branched_log_cache_key_(
		dimension, max_nodes, planar);
	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	const auto found = get_cuda_branched_log_sig_cache_map_().find(key);
	return found != get_cuda_branched_log_sig_cache_map_().end()
		&& found->second.horner;
}

template<typename T>
static void upload_branched_log(T*& d_ptr, const T* h_data, size_t count) {
	if (count == 0) {
		d_ptr = nullptr;
		return;
	}
	CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_ptr, h_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

void prepare_cuda_branched_log_horner_plan_(
	const BranchedSigCache& c,
	const BranchedLogHornerPlan& plan
) {
	const uint64_t dimension = c.dimension;
	const uint64_t max_nodes = c.max_nodes;
	const bool planar = c.planar;
	const auto key = make_cuda_branched_log_cache_key_(
		dimension, max_nodes, planar);
	{
		std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
		auto found = get_cuda_branched_log_sig_cache_map_().find(key);
		if (found != get_cuda_branched_log_sig_cache_map_().end()
			&& found->second.horner)
			return;
	}

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

	std::vector<uint32_t> product_offsets32(plan.cuda_products.offsets.size());
	std::vector<uint32_t> product_factors32(plan.cuda_products.factors.size());
	std::vector<uint32_t> coproduct_offsets32(plan.coproduct_offsets.size());
	std::vector<uint32_t> coproduct_pairs32(plan.coproduct_pairs.size());
	std::vector<uint32_t> flat_to_product32(plan.flat_to_product.size());
	safe_narrow(
		plan.cuda_products.offsets.data(), product_offsets32.data(),
		plan.cuda_products.offsets.size());
	safe_narrow(
		plan.cuda_products.factors.data(), product_factors32.data(),
		plan.cuda_products.factors.size());
	safe_narrow(
		plan.coproduct_offsets.data(), coproduct_offsets32.data(),
		plan.coproduct_offsets.size());
	safe_narrow(
		plan.coproduct_pairs.data(), coproduct_pairs32.data(),
		plan.coproduct_pairs.size());
	safe_narrow(
		plan.flat_to_product.data(), flat_to_product32.data(),
		plan.flat_to_product.size());

	auto narrow32 = [](uint64_t v) -> uint32_t {
		if (v > UINT32_MAX) throw std::overflow_error("Branched log sig cache value exceeds uint32 range");
		return static_cast<uint32_t>(v);
	};

	auto gpu = std::make_unique<BranchedLogHornerPlanGPU>();
	gpu->total_length = narrow32(c.total_length);
	gpu->num_trees = narrow32(num_trees);
	gpu->product_count = narrow32(plan.product_count);
	gpu->max_nodes = static_cast<int>(max_nodes);
	upload_branched_log(
		gpu->d_product_offsets, product_offsets32.data(), product_offsets32.size());
	upload_branched_log(
		gpu->d_product_factors, product_factors32.data(), product_factors32.size());
	upload_branched_log(
		gpu->d_coproduct_offsets, coproduct_offsets32.data(), coproduct_offsets32.size());
	upload_branched_log(
		gpu->d_coproduct_pairs, coproduct_pairs32.data(), coproduct_pairs32.size());
	upload_branched_log(
		gpu->d_flat_to_product, flat_to_product32.data(), flat_to_product32.size());

	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	auto [found, inserted] = get_cuda_branched_log_sig_cache_map_().try_emplace(key);
	if (!found->second.horner)
		found->second.horner = std::move(gpu);
}

static const BranchedLogHornerPlanGPU& get_branched_log_horner_plan_gpu_(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	const auto key = make_cuda_branched_log_cache_key_(
		dimension, max_nodes, planar);
	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	auto found = get_cuda_branched_log_sig_cache_map_().find(key);
	if (found != get_cuda_branched_log_sig_cache_map_().end()
		&& found->second.horner)
		return *found->second.horner;
	throw cache_not_found_error(
		"CUDA branched log sig cache not found - call prepare_branched_log_sig with device='cuda' first");
}

template<typename T, int Method, bool Planar>
__global__ __launch_bounds__(1024)
void branched_sig_to_log_sig_horner_ker(
	const T* __restrict__ bsig,
	T* __restrict__ out,
	uint32_t total_len,
	uint32_t product_count,
	const uint32_t* __restrict__ g_product_offsets,
	const uint32_t* __restrict__ g_product_factors,
	const uint32_t* __restrict__ g_coproduct_offsets,
	const uint32_t* __restrict__ g_coproduct_pairs,
	const uint32_t* __restrict__ g_flat_to_product,
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
	T* h_products = h + total_len;
	T* horner_current = h_products + product_count;
	T* horner_next = horner_current + product_count;
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
	}
	for (uint32_t i = tid; i < product_count; i += blockDim.x) {
		h_products[i] = T(0);
		horner_current[i] = T(0);
		horner_next[i] = T(0);
	}
	__syncthreads();

	const T* src = bsig + static_cast<uint64_t>(batch_idx) * input_stride;
	T* dst = out + static_cast<uint64_t>(batch_idx) * output_stride;
	for (uint32_t i = tid; i < num_trees; i += blockDim.x) {
		const T v = scalar_term ? src[i + 1] : src[i];
		h[i + 1] = v;
	}
	__syncthreads();

	for (uint32_t product = tid + 1; product < product_count; product += blockDim.x) {
		T val;
		if constexpr (Planar) {
			val = h[product];
		} else {
			val = T(1);
			const uint32_t start = g_product_offsets[product];
			const uint32_t end = g_product_offsets[product + 1];
			for (uint32_t pos = start; pos < end; ++pos)
				val *= h[g_product_factors[pos]];
		}
		h_products[product] = val;
		if (max_nodes > 1)
			horner_current[product] = val / T(max_nodes);
	}
	__syncthreads();

	T* cur = horner_current;
	T* next = horner_next;
	for (int k = max_nodes - 1; k > 1; --k) {
		for (uint32_t product = tid; product < product_count; product += blockDim.x) {
			T val = T(0);
			const uint32_t start = g_coproduct_offsets[product];
			const uint32_t end = g_coproduct_offsets[product + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = g_coproduct_pairs[pos];
				const uint32_t right = g_coproduct_pairs[pos + 1];
				val += cur[left] * h_products[right];
			}
			next[product] = h_products[product] / T(k) - val;
		}
		__syncthreads();

		T* tmp = cur;
		cur = next;
		next = tmp;
	}
	if (max_nodes > 1) {
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x) {
			const uint32_t product = Planar ? i : g_flat_to_product[i];
			T val = T(0);
			const uint32_t start = g_coproduct_offsets[product];
			const uint32_t end = g_coproduct_offsets[product + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = g_coproduct_pairs[pos];
				const uint32_t right = g_coproduct_pairs[pos + 1];
				val += cur[left] * h_products[right];
			}
			h[i] -= val;
		}
		__syncthreads();
	}

	if constexpr (Method == 0) {
		if (scalar_term) {
			for (uint32_t i = tid; i < total_len; i += blockDim.x)
				dst[i] = h[i];
		} else {
			for (uint32_t i = tid; i < num_trees; i += blockDim.x)
				dst[i] = h[i + 1];
		}
	} else if constexpr (Method == 1) {
		for (uint32_t i = tid; i < compact_len; i += blockDim.x)
			dst[i] = h[g_lyndon_idx[i]];
	} else {
		for (uint32_t i = tid; i < compact_len; i += blockDim.x)
			horner_current[i] = h[g_lyndon_idx[i]];
		__syncthreads();
		for (uint32_t i = tid; i < compact_len; i += blockDim.x) {
			T value = horner_current[i];
			const uint32_t start = g_sparse_row_ptr[i];
			const uint32_t end = g_sparse_row_ptr[i + 1];
			for (uint32_t pos = start; pos < end; ++pos)
				value += static_cast<T>(g_sparse_vals[pos])
					* horner_current[g_sparse_cols[pos]];
			dst[i] = value;
		}
	}
}

template<typename T, int Method, bool Planar>
__global__ __launch_bounds__(1024)
void branched_sig_to_log_sig_backprop_horner_ker(
	const T* __restrict__ bsig,
	const T* __restrict__ derivs,
	T* __restrict__ out,
	uint32_t total_len,
	uint32_t product_count,
	const uint32_t* __restrict__ g_product_offsets,
	const uint32_t* __restrict__ g_product_factors,
	const uint32_t* __restrict__ g_coproduct_offsets,
	const uint32_t* __restrict__ g_coproduct_pairs,
	const uint32_t* __restrict__ g_flat_to_product,
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
	const uint64_t input_stride = scalar_term ? total_len : total_len - 1;
	const uint64_t deriv_stride = Method == 0 ? input_stride : compact_len;

	extern __shared__ char smem[];
	const uint64_t state_levels = max_nodes > 1
		? static_cast<uint64_t>(max_nodes - 1) : 0;
	const uint64_t states_size = state_levels * product_count;
	const uint64_t workspace_size = states_size + 2 * product_count;
	T* states = workspace + local_batch_idx * workspace_size;
	T* d_current = states + states_size;
	T* d_next = d_current + product_count;
	T* h = reinterpret_cast<T*>(smem);
	T* h_products = h + total_len;
	T* d_h_products = h_products + product_count;
	T* d_h_tree = d_h_products + product_count;
	T* full_derivs = d_h_tree + total_len;

	for (uint64_t i = tid; i < workspace_size; i += blockDim.x)
		states[i] = T(0);
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
		d_h_tree[i] = T(0);
		full_derivs[i] = T(0);
	}
	for (uint32_t i = tid; i < product_count; i += blockDim.x) {
		h_products[i] = T(0);
		d_h_products[i] = T(0);
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

	for (uint32_t product = tid + 1; product < product_count; product += blockDim.x) {
		T val;
		if constexpr (Planar) {
			val = h[product];
		} else {
			val = T(1);
			const uint32_t start = g_product_offsets[product];
			const uint32_t end = g_product_offsets[product + 1];
			for (uint32_t pos = start; pos < end; ++pos)
				val *= h[g_product_factors[pos]];
		}
		h_products[product] = val;
	}
	for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
		d_h_tree[i] = full_derivs[i];
	__syncthreads();

	if (max_nodes <= 1) {
		if (scalar_term) {
			for (uint32_t i = tid; i < total_len; i += blockDim.x)
				dst[i] = d_h_tree[i];
		} else {
			for (uint32_t i = tid; i < num_trees; i += blockDim.x)
				dst[i] = d_h_tree[i + 1];
		}
		return;
	}

	T* last = states + static_cast<uint64_t>(max_nodes - 2) * product_count;
	for (uint32_t product = tid; product < product_count; product += blockDim.x)
		last[product] = h_products[product] / T(max_nodes);
	__syncthreads();

	for (int k = max_nodes - 1; k > 1; --k) {
		T* current = states + static_cast<uint64_t>(k - 2) * product_count;
		const T* next = current + product_count;
		for (uint32_t product = tid; product < product_count; product += blockDim.x) {
			T val = T(0);
			const uint32_t start = g_coproduct_offsets[product];
			const uint32_t end = g_coproduct_offsets[product + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = g_coproduct_pairs[pos];
				const uint32_t right = g_coproduct_pairs[pos + 1];
				val += next[left] * h_products[right];
			}
			current[product] = h_products[product] / T(k) - val;
		}
		__syncthreads();
	}

	const T* b2 = states;
	for (uint32_t i = tid + 1; i < total_len; i += blockDim.x) {
		const T d = full_derivs[i];
		const uint32_t product = Planar ? i : g_flat_to_product[i];
		const uint32_t start = g_coproduct_offsets[product];
		const uint32_t end = g_coproduct_offsets[product + 1];
		for (uint32_t pos = start; pos < end; pos += 2) {
			const uint32_t left = g_coproduct_pairs[pos];
			const uint32_t right = g_coproduct_pairs[pos + 1];
			myAtomicAdd(&d_current[left], -d * h_products[right]);
			myAtomicAdd(&d_h_products[right], -d * b2[left]);
		}
	}
	__syncthreads();

	for (int k = 2; k < max_nodes; ++k) {
		const T* next_state = states + static_cast<uint64_t>(k - 1) * product_count;
		for (uint32_t product = tid; product < product_count; product += blockDim.x) {
			const T d = d_current[product];
			myAtomicAdd(&d_h_products[product], d / T(k));
			const uint32_t start = g_coproduct_offsets[product];
			const uint32_t end = g_coproduct_offsets[product + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = g_coproduct_pairs[pos];
				const uint32_t right = g_coproduct_pairs[pos + 1];
				myAtomicAdd(&d_next[left], -d * h_products[right]);
				myAtomicAdd(&d_h_products[right], -d * next_state[left]);
			}
		}
		__syncthreads();
		T* tmp = d_current;
		d_current = d_next;
		d_next = tmp;
		for (uint32_t product = tid; product < product_count; product += blockDim.x)
			d_next[product] = T(0);
		__syncthreads();
	}

	for (uint32_t product = tid + 1; product < product_count; product += blockDim.x)
		d_h_products[product] += d_current[product] / T(max_nodes);
	__syncthreads();

	if constexpr (Planar) {
		for (uint32_t product = tid + 1; product < product_count; product += blockDim.x)
			d_h_tree[product] += d_h_products[product];
	} else {
		for (uint32_t product = tid + 1; product < product_count; product += blockDim.x) {
			const T d = d_h_products[product];
			if (d == T(0))
				continue;
			const uint32_t start = g_product_offsets[product];
			const uint32_t end = g_product_offsets[product + 1];
			for (uint32_t pos = start; pos < end; ++pos) {
				T partial = d;
				for (uint32_t other = start; other < end; ++other) {
					if (other != pos)
						partial *= h[g_product_factors[other]];
				}
				myAtomicAdd(&d_h_tree[g_product_factors[pos]], partial);
			}
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

template<typename T, int Method, bool Planar>
static void branched_sig_to_log_sig_cuda_method_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	const BranchedLogHornerPlanGPU& plan,
	const CuMkwBasisGpuCache* basis,
	bool scalar_term
) {
	const uint32_t compact_len = basis == nullptr ? 0 : basis->compact_length;
	if (batch_size == 0 || (Method != 0 && compact_len == 0))
		return;
	const uint32_t work_items = std::max(
		std::max(plan.num_trees, plan.product_count), compact_len);
	unsigned int block = static_cast<unsigned int>((work_items + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	size_t smem = (static_cast<uint64_t>(plan.total_length)
		+ 3 * static_cast<uint64_t>(plan.product_count)) * sizeof(T);
	configure_dynamic_smem(
		branched_sig_to_log_sig_horner_ker<T, Method, Planar>, smem,
		"CUDA branched log sig");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size, batch_offset);
		branched_sig_to_log_sig_horner_ker<T, Method, Planar>
			<<<batch_chunk.grid, block, smem>>>(
			bsig, out, plan.total_length, plan.product_count,
			plan.d_product_offsets, plan.d_product_factors,
			plan.d_coproduct_offsets, plan.d_coproduct_pairs,
			plan.d_flat_to_product,
			plan.max_nodes, scalar_term,
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
	const auto& plan = get_branched_log_horner_plan_gpu_(
		dimension, max_nodes, planar);
	if (method == 0) {
		if (planar)
			branched_sig_to_log_sig_cuda_method_<T, 0, true>(
				bsig, out, batch_size, plan, nullptr, scalar_term);
		else
			branched_sig_to_log_sig_cuda_method_<T, 0, false>(
				bsig, out, batch_size, plan, nullptr, scalar_term);
		return;
	}
	const auto& basis = get_cuda_mkw_basis_gpu_cache_(
		dimension, max_nodes, method);
	if (method == 1)
		branched_sig_to_log_sig_cuda_method_<T, 1, true>(
			bsig, out, batch_size, plan, &basis, scalar_term);
	else
		branched_sig_to_log_sig_cuda_method_<T, 2, true>(
			bsig, out, batch_size, plan, &basis, scalar_term);
}

template<typename T, int Method, bool Planar>
static void branched_sig_to_log_sig_backprop_cuda_method_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	const BranchedLogHornerPlanGPU& plan,
	const CuMkwBasisGpuCache* basis,
	bool scalar_term
) {
	const uint32_t compact_len = basis == nullptr ? 0 : basis->compact_length;
	if (batch_size == 0)
		return;
	if (Method != 0 && compact_len == 0) {
		const uint64_t input_stride = scalar_term
			? plan.total_length : plan.total_length - 1;
		if (input_stride != 0) {
			CUDA_CHECK(cudaMemset(
				out, 0, batch_size * input_stride * sizeof(T)));
			check_cuda_kernel_launch();
		}
		return;
	}
	const uint32_t work_items = std::max(
		std::max(plan.num_trees, plan.product_count), compact_len);
	unsigned int block = static_cast<unsigned int>((work_items + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	const uint64_t state_levels = plan.max_nodes > 1
		? static_cast<uint64_t>(plan.max_nodes - 1) : 0;
	const uint64_t workspace_levels = state_levels + 2;
	const uint64_t workspace_elements = checked_branched_mul_(
		workspace_levels,
		plan.product_count,
		"CUDA branched log sig backprop workspace size overflow");
	const uint64_t workspace_bytes = checked_branched_mul_(
		workspace_elements, sizeof(T),
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

	size_t smem = (2 * static_cast<uint64_t>(plan.product_count)
		+ 3 * static_cast<uint64_t>(plan.total_length)) * sizeof(T);
	configure_dynamic_smem(
		branched_sig_to_log_sig_backprop_horner_ker<T, Method, Planar>, smem,
		"CUDA branched log sig backprop");
	for (uint64_t batch_offset = 0; batch_offset < batch_size;
		batch_offset += workspace_chunk_size) {
		const uint64_t current_batch = std::min<uint64_t>(
			workspace_chunk_size, batch_size - batch_offset);
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, current_batch, 0);
		branched_sig_to_log_sig_backprop_horner_ker<T, Method, Planar>
			<<<batch_chunk.grid, block, smem>>>(
			bsig, derivs, out, plan.total_length, plan.product_count,
			plan.d_product_offsets, plan.d_product_factors,
			plan.d_coproduct_offsets, plan.d_coproduct_pairs,
			plan.d_flat_to_product,
			workspace.get(),
			plan.max_nodes, scalar_term,
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
	const auto& plan = get_branched_log_horner_plan_gpu_(
		dimension, max_nodes, planar);
	if (method == 0) {
		if (planar)
			branched_sig_to_log_sig_backprop_cuda_method_<T, 0, true>(
				bsig, derivs, out, batch_size, plan, nullptr, scalar_term);
		else
			branched_sig_to_log_sig_backprop_cuda_method_<T, 0, false>(
				bsig, derivs, out, batch_size, plan, nullptr, scalar_term);
		return;
	}
	const auto& basis = get_cuda_mkw_basis_gpu_cache_(
		dimension, max_nodes, method);
	if (method == 1)
		branched_sig_to_log_sig_backprop_cuda_method_<T, 1, true>(
			bsig, derivs, out, batch_size, plan, &basis, scalar_term);
	else
		branched_sig_to_log_sig_backprop_cuda_method_<T, 2, true>(
			bsig, derivs, out, batch_size, plan, &basis, scalar_term);
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
