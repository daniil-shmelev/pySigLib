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

#pragma once

#include "cu_disk_cache.h"
#include "cu_utils.h"
#include "preparation/log_sig/log_sig_cache.h"
#include "preparation/log_sig/lyndon_words.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

struct CUDABchPlan {
	// A null node list is the dense all-nodes-live plan.
	uint32_t* nodes = nullptr;
	uint64_t size = 0;

#ifdef __CUDACC__
	__host__ __device__ __forceinline__
#endif
	uint32_t operator[](uint64_t index) const {
		return nodes ? nodes[index] : static_cast<uint32_t>(index + 2);
	}
};

struct CUDACommutatorView {
	const uint32_t* row_output = nullptr;
	const uint32_t* ptr = nullptr;
	const uint32_t* i = nullptr;
	const uint32_t* j = nullptr;
	const int* value = nullptr;

	bool balanced() const noexcept { return row_output != nullptr; }

#ifdef __CUDACC__
	template<bool use_balanced_rows>
	__device__ __forceinline__ uint32_t output(uint64_t row) const {
		if constexpr (use_balanced_rows)
			return row_output[row];
		return static_cast<uint32_t>(row);
	}
#endif
};

struct CUDACommutatorPlan {
	CudaBuf<uint32_t> storage;
	CUDACommutatorView balanced_view;
	bool use_for_dense = false;
	bool use_for_linear = false;

	explicit operator bool() const noexcept { return storage.get() != nullptr; }
};

struct CUDABchCache {
	double* d_bch_coefficients = nullptr;
	uint64_t* d_bch_left_factor = nullptr;
	uint64_t* d_bch_right_factor = nullptr;
	CUDABchPlan bch_plan;
	CUDACommutatorPlan commutator_plan;
	CudaBuf<uint32_t> linear_forward_storage;
	CUDACommutatorView linear_forward;
	uint64_t* d_linear_range = nullptr;
	uint64_t m2 = 0;
	uint64_t m = 0;
	uint64_t linear_dense_forward_work = 0;
	uint64_t linear_active_forward_work = 0;
	uint64_t linear_zero_work = 0;
	uint32_t* d_comm_k_ptr = nullptr;
	uint32_t* d_comm_k_i = nullptr;
	uint32_t* d_comm_k_j = nullptr;
	int* d_comm_k_val = nullptr;
	uint32_t* d_comm_a_ptr = nullptr;
	uint32_t* d_comm_a_k = nullptr;
	uint32_t* d_comm_a_partner = nullptr;
	int* d_comm_a_signed_c = nullptr;
	uint64_t* d_linear_a_ptr = nullptr;
	uint32_t* d_linear_a_ptr32 = nullptr;
	uint32_t* d_linear_a_idx = nullptr;

	CUDABchCache() = default;
	CUDABchCache(const CUDABchCache&) = delete;
	CUDABchCache& operator=(const CUDABchCache&) = delete;
	CUDABchCache(CUDABchCache&& other) noexcept
		: d_bch_coefficients(std::exchange(other.d_bch_coefficients, nullptr)),
		d_bch_left_factor(std::exchange(other.d_bch_left_factor, nullptr)),
		d_bch_right_factor(std::exchange(other.d_bch_right_factor, nullptr)),
		bch_plan{ std::exchange(other.bch_plan.nodes, nullptr),
			std::exchange(other.bch_plan.size, 0) },
		commutator_plan(std::move(other.commutator_plan)),
		linear_forward_storage(std::move(other.linear_forward_storage)),
		linear_forward(other.linear_forward),
		d_linear_range(std::exchange(other.d_linear_range, nullptr)),
		m2(std::exchange(other.m2, 0)),
		m(std::exchange(other.m, 0)),
		linear_dense_forward_work(other.linear_dense_forward_work),
		linear_active_forward_work(other.linear_active_forward_work),
		linear_zero_work(other.linear_zero_work),
		d_comm_k_ptr(std::exchange(other.d_comm_k_ptr, nullptr)),
		d_comm_k_i(std::exchange(other.d_comm_k_i, nullptr)),
		d_comm_k_j(std::exchange(other.d_comm_k_j, nullptr)),
		d_comm_k_val(std::exchange(other.d_comm_k_val, nullptr)),
		d_comm_a_ptr(std::exchange(other.d_comm_a_ptr, nullptr)),
		d_comm_a_k(std::exchange(other.d_comm_a_k, nullptr)),
		d_comm_a_partner(std::exchange(other.d_comm_a_partner, nullptr)),
		d_comm_a_signed_c(std::exchange(other.d_comm_a_signed_c, nullptr)),
		d_linear_a_ptr(std::exchange(other.d_linear_a_ptr, nullptr)),
		d_linear_a_ptr32(std::exchange(other.d_linear_a_ptr32, nullptr)),
		d_linear_a_idx(std::exchange(other.d_linear_a_idx, nullptr)) {}

	~CUDABchCache() {
		if (d_bch_coefficients) cudaFree(d_bch_coefficients);
		if (d_bch_left_factor) cudaFree(d_bch_left_factor);
		if (d_bch_right_factor) cudaFree(d_bch_right_factor);
		if (bch_plan.nodes) cudaFree(bch_plan.nodes);
		if (d_linear_range) cudaFree(d_linear_range);
		if (d_comm_k_ptr) cudaFree(d_comm_k_ptr);
		if (d_comm_k_i) cudaFree(d_comm_k_i);
		if (d_comm_k_j) cudaFree(d_comm_k_j);
		if (d_comm_k_val) cudaFree(d_comm_k_val);
		if (d_comm_a_ptr) cudaFree(d_comm_a_ptr);
		if (d_comm_a_k) cudaFree(d_comm_a_k);
		if (d_comm_a_partner) cudaFree(d_comm_a_partner);
		if (d_comm_a_signed_c) cudaFree(d_comm_a_signed_c);
		if (d_linear_a_ptr) cudaFree(d_linear_a_ptr);
		if (d_linear_a_ptr32) cudaFree(d_linear_a_ptr32);
		if (d_linear_a_idx) cudaFree(d_linear_a_idx);
	}

	CUDACommutatorView commutator_view() const noexcept {
		return { nullptr, d_comm_k_ptr, d_comm_k_i, d_comm_k_j,
			d_comm_k_val };
	}

	CUDACommutatorView commutator_view_(bool use_balanced) const noexcept {
		if (use_balanced)
			return commutator_plan.balanced_view;
		return commutator_view();
	}

	CUDACommutatorView dense_commutator_view() const noexcept {
		return commutator_view_(
			commutator_plan && commutator_plan.use_for_dense);
	}

	CUDACommutatorView linear_commutator_view() const noexcept {
		return commutator_view_(
			commutator_plan && commutator_plan.use_for_linear);
	}
};

struct CUDALogSigCache {
	uint64_t* d_lyndon_idx = nullptr;
	uint64_t log_sig_len = 0;
	uint64_t* d_level_index = nullptr;
	uint64_t sig_len = 0;
	uint64_t buff1_len = 0;
	unsigned int threads_per_block = 32;
	int method = 0;

	int* d_sparse_vals = nullptr;
	uint64_t* d_sparse_cols = nullptr;
	uint64_t* d_sparse_row_ptr = nullptr;
	int* d_sparse_vals_t = nullptr;
	uint64_t* d_sparse_cols_t = nullptr;
	uint64_t* d_sparse_row_ptr_t = nullptr;
	std::unique_ptr<CUDABchCache> bch;

	CUDALogSigCache() = default;
	CUDALogSigCache(const CUDALogSigCache&) = delete;
	CUDALogSigCache& operator=(const CUDALogSigCache&) = delete;

	CUDALogSigCache(CUDALogSigCache&& other) noexcept
		: d_lyndon_idx(std::exchange(other.d_lyndon_idx, nullptr)),
		log_sig_len(std::exchange(other.log_sig_len, 0)),
		d_level_index(std::exchange(other.d_level_index, nullptr)),
		sig_len(std::exchange(other.sig_len, 0)),
		buff1_len(std::exchange(other.buff1_len, 0)),
		threads_per_block(other.threads_per_block),
		method(std::exchange(other.method, 0)),
		d_sparse_vals(std::exchange(other.d_sparse_vals, nullptr)),
		d_sparse_cols(std::exchange(other.d_sparse_cols, nullptr)),
		d_sparse_row_ptr(std::exchange(other.d_sparse_row_ptr, nullptr)),
		d_sparse_vals_t(std::exchange(other.d_sparse_vals_t, nullptr)),
		d_sparse_cols_t(std::exchange(other.d_sparse_cols_t, nullptr)),
		d_sparse_row_ptr_t(std::exchange(other.d_sparse_row_ptr_t, nullptr)),
		bch(std::move(other.bch)) {}

	~CUDALogSigCache() {
		if (d_lyndon_idx) cudaFree(d_lyndon_idx);
		if (d_level_index) cudaFree(d_level_index);
		if (d_sparse_vals) cudaFree(d_sparse_vals);
		if (d_sparse_cols) cudaFree(d_sparse_cols);
		if (d_sparse_row_ptr) cudaFree(d_sparse_row_ptr);
		if (d_sparse_vals_t) cudaFree(d_sparse_vals_t);
		if (d_sparse_cols_t) cudaFree(d_sparse_cols_t);
		if (d_sparse_row_ptr_t) cudaFree(d_sparse_row_ptr_t);
	}
};

struct CuLogSigCacheKey {
	int device = 0;
	uint64_t dimension = 0;
	uint64_t degree = 0;

	bool operator==(const CuLogSigCacheKey& other) const noexcept {
		return device == other.device
			&& dimension == other.dimension
			&& degree == other.degree;
	}
};

struct CuLogSigCacheKeyHash {
	size_t operator()(const CuLogSigCacheKey& key) const noexcept {
		size_t value = std::hash<int>{}(key.device);
		auto combine = [&value](uint64_t item) {
			value ^= std::hash<uint64_t>{}(item) + kFibHashConst
				+ (value << 6) + (value >> 2);
		};
		combine(key.dimension);
		combine(key.degree);
		return value;
	}
};

inline CuLogSigCacheKey make_cuda_log_sig_cache_key_(
	uint64_t dimension,
	uint64_t degree
) {
	CuLogSigCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.dimension = dimension;
	key.degree = degree;
	return key;
}

std::unordered_map<
	CuLogSigCacheKey, CUDALogSigCache, CuLogSigCacheKeyHash
>& get_cuda_log_sig_cache_map_();
std::mutex& get_cuda_log_sig_cache_mu_();

inline void upload_csr_to_gpu_(
	const SparseIntMatrix& matrix,
	int*& d_values,
	uint64_t*& d_columns,
	uint64_t*& d_row_offsets
) {
	uint64_t nnz = 0;
	for (const auto& row : matrix.rows)
		nnz += row.size();
	std::vector<int> values(nnz);
	std::vector<uint64_t> columns(nnz);
	std::vector<uint64_t> row_offsets(matrix.n + 1);
	uint64_t entry_index = 0;
	for (uint64_t row = 0; row < matrix.n; ++row) {
		row_offsets[row] = entry_index;
		for (const Entry& entry : matrix.rows[row]) {
			values[entry_index] = entry.val;
			columns[entry_index] = entry.col;
			++entry_index;
		}
	}
	row_offsets[matrix.n] = entry_index;

	CudaBuf<int> value_buffer;
	CudaBuf<uint64_t> column_buffer;
	if (nnz != 0) {
		value_buffer = CudaBuf<int>(nnz * sizeof(int));
		column_buffer = CudaBuf<uint64_t>(nnz * sizeof(uint64_t));
		CUDA_CHECK(cudaMemcpy(
			value_buffer.get(), values.data(), nnz * sizeof(int),
			cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(
			column_buffer.get(), columns.data(), nnz * sizeof(uint64_t),
			cudaMemcpyHostToDevice));
	}
	CudaBuf<uint64_t> offset_buffer(
		(matrix.n + 1) * sizeof(uint64_t));
	CUDA_CHECK(cudaMemcpy(
		offset_buffer.get(), row_offsets.data(),
		(matrix.n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	d_values = value_buffer.release();
	d_columns = column_buffer.release();
	d_row_offsets = offset_buffer.release();
}

inline void upload_log_sig_basis_(
	CUDALogSigCache& device_cache,
	const BasisCache& host_cache,
	uint64_t dimension,
	uint64_t degree
) {
	if (device_cache.d_lyndon_idx == nullptr) {
		device_cache.log_sig_len = host_cache.lyndon_idx.size();
		if (!host_cache.lyndon_idx.empty()) {
			CUDA_CHECK(cudaMalloc(
				&device_cache.d_lyndon_idx,
				host_cache.lyndon_idx.size() * sizeof(uint64_t)));
			CUDA_CHECK(cudaMemcpy(
				device_cache.d_lyndon_idx, host_cache.lyndon_idx.data(),
				host_cache.lyndon_idx.size() * sizeof(uint64_t),
				cudaMemcpyHostToDevice));
		}
		auto level_index = std::make_unique<uint64_t[]>(degree + 2);
		host_populate_level_index(level_index.get(), dimension, degree + 2);
		CUDA_CHECK(cudaMalloc(
			&device_cache.d_level_index,
			(degree + 2) * sizeof(uint64_t)));
		CUDA_CHECK(cudaMemcpy(
			device_cache.d_level_index, level_index.get(),
			(degree + 2) * sizeof(uint64_t), cudaMemcpyHostToDevice));
		device_cache.sig_len = host_sig_length(dimension, degree);
		device_cache.buff1_len = degree >= 2
			? host_sig_length(dimension, degree - 1) : 1;
		const uint64_t max_level_size =
			level_index[degree + 1] - level_index[degree];
		device_cache.threads_per_block =
			host_choose_threads_per_block(max_level_size);
		device_cache.method = 1;
	}
	if (host_cache.method >= 2 && device_cache.method < 2) {
		upload_csr_to_gpu_(
			host_cache.inv_proj_mat,
			device_cache.d_sparse_vals,
			device_cache.d_sparse_cols,
			device_cache.d_sparse_row_ptr);
		upload_csr_to_gpu_(
			host_cache.inv_proj_mat_transpose,
			device_cache.d_sparse_vals_t,
			device_cache.d_sparse_cols_t,
			device_cache.d_sparse_row_ptr_t);
		device_cache.method = 2;
	}
}

inline void prepare_log_sig_cuda_(
	uint64_t dimension,
	uint64_t degree,
	int method,
	bool use_disk
) {
	const int basis_method = (std::min)(method, 2);
	const auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	{
		std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
		const auto found = get_cuda_log_sig_cache_map_().find(key);
		if (found != get_cuda_log_sig_cache_map_().end()
			&& found->second.method >= basis_method)
			return;
	}
	if (use_disk)
		ensure_cuda_cache_dir_();
	const auto cache_directory = get_cuda_cache_dir_() / cu_cache_folder_name;
	LogSigCache host_cache(
		dimension, degree, basis_method, cache_directory, use_disk);
	std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
	auto [found, inserted] = get_cuda_log_sig_cache_map_().try_emplace(key);
	if (found->second.method < basis_method)
		upload_log_sig_basis_(
			found->second, host_cache.basis(basis_method), dimension, degree);
}

inline const CUDALogSigCache& get_cuda_log_sig_cache(
	uint64_t dimension,
	uint64_t degree,
	int method = 1
) {
	const auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
	const auto found = get_cuda_log_sig_cache_map_().find(key);
	if (found == get_cuda_log_sig_cache_map_().end()
		|| found->second.method < (std::min)(method, 2)) {
		throw cache_not_found_error(
			"CUDA log sig cache not found - call prepare_log_sig with device='cuda' first");
	}
	return found->second;
}

void clear_cuda_branched_sig_gpu_cache_();
void clear_cuda_branched_log_sig_gpu_cache_();

inline void clear_cache_cuda_(bool use_disk) {
	{
		std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
		get_cuda_log_sig_cache_map_().clear();
	}
	clear_cuda_branched_sig_gpu_cache_();
	clear_cuda_branched_log_sig_gpu_cache_();
	if (use_disk) {
		ensure_cuda_cache_dir_();
		std::filesystem::remove_all(
			get_cuda_cache_dir_() / cu_cache_folder_name);
	}
}
