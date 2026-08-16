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

#include "cu_log_sig_cache.h"
#include "preparation/branched_sig/branched_log_sig_cache.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

struct BranchedLogHornerPlanGPU {
	uint32_t* d_product_offsets = nullptr;
	uint32_t* d_product_factors = nullptr;
	uint32_t* d_coproduct_offsets = nullptr;
	uint32_t* d_coproduct_pairs = nullptr;
	uint32_t* d_flat_to_product = nullptr;

	uint32_t total_length = 0;
	uint32_t num_trees = 0;
	uint32_t product_count = 0;
	int max_nodes = 0;

	BranchedLogHornerPlanGPU() = default;
	BranchedLogHornerPlanGPU(const BranchedLogHornerPlanGPU&) = delete;
	BranchedLogHornerPlanGPU& operator=(const BranchedLogHornerPlanGPU&) = delete;

	~BranchedLogHornerPlanGPU() {
		if (d_product_offsets) cudaFree(d_product_offsets);
		if (d_product_factors) cudaFree(d_product_factors);
		if (d_coproduct_offsets) cudaFree(d_coproduct_offsets);
		if (d_coproduct_pairs) cudaFree(d_coproduct_pairs);
		if (d_flat_to_product) cudaFree(d_flat_to_product);
	}
};

struct CuMkwBasisGpuCache {
	// Method 1 uses indices. Method 2 additionally uses these CSR matrices.
	// Both matrix orientations are needed because the backward pass is adjoint.
	uint32_t* d_lyndon_idx = nullptr;
	int* d_sparse_vals = nullptr;
	uint32_t* d_sparse_cols = nullptr;
	uint32_t* d_sparse_row_ptr = nullptr;
	int* d_sparse_vals_t = nullptr;
	uint32_t* d_sparse_cols_t = nullptr;
	uint32_t* d_sparse_row_ptr_t = nullptr;
	uint32_t compact_length = 0;
	int method = 0;

	CuMkwBasisGpuCache() = default;
	CuMkwBasisGpuCache(const CuMkwBasisGpuCache&) = delete;
	CuMkwBasisGpuCache& operator=(const CuMkwBasisGpuCache&) = delete;
	~CuMkwBasisGpuCache();
};

struct CuBranchedLogCacheKey {
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
		size_t value = std::hash<int>{}(key.device);
		auto combine = [&value](uint64_t item) {
			value ^= std::hash<uint64_t>{}(item) + kFibHashConst
				+ (value << 6) + (value >> 2);
		};
		combine(key.dimension);
		combine(key.max_nodes);
		combine(static_cast<uint64_t>(key.planar));
		return value;
	}
};

inline CuBranchedLogCacheKey make_cuda_branched_log_cache_key_(
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

struct CudaBranchedLogSigCache {
	std::unique_ptr<BranchedLogHornerPlanGPU> horner;
	std::unique_ptr<CuMkwBasisGpuCache> basis;
	std::shared_ptr<void> bch;
};

std::unordered_map<
	CuBranchedLogCacheKey,
	CudaBranchedLogSigCache,
	CuBranchedLogCacheKeyHash
>& get_cuda_branched_log_sig_cache_map_();

std::mutex& get_cuda_branched_log_sig_cache_mu_();

void prepare_cuda_branched_log_horner_plan_(
	const BranchedSigCache& cache,
	const BranchedLogHornerPlan& plan);

bool is_cuda_branched_log_horner_plan_prepared_(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar);

void prepare_cuda_mkw_basis_cache_(
	const BranchedSigCache& cache,
	const BasisCache& basis);

const CuMkwBasisGpuCache& get_cuda_mkw_basis_gpu_cache_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method);

void prepare_cuda_branched_bch_cache_(
	const BranchedSigCache& cache,
	const BranchedBchCache& host_cache);
