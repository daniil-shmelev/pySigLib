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
#include "../shared/branched_cache.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Host data builds the MKW Lyndon basis before its compact GPU representation.
// It mirrors the CPU basis cache but keeps the data needed to upload CUDA CSR
// buffers and to build the method 3 BCH cache.
struct CuMkwHostBasisData {
	int method = 0;
	// Flat forests and their reverse lookup, represented as words of tree IDs.
	std::vector<cu_word> flat_words;
	std::unordered_map<cu_word, uint64_t, CuWordHash> flat_idx;
	// Compact Lyndon coordinates and their weighted degrees.
	std::vector<cu_word> lyndon_words;
	std::vector<uint64_t> lyndon_idx;
	std::vector<uint64_t> lyndon_weights;
	std::vector<uint32_t> letter_log_idx;
	std::vector<uint64_t> letter_basis_idx;
	std::vector<uint64_t> left_factor;
	std::vector<uint64_t> right_factor;
	CuSparseIntMatrix inv_proj_mat;
	CuSparseIntMatrix inv_proj_mat_t;
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

void prepare_cuda_branched_log_sig_gpu_cache_(
	const BranchedSigCache& cache);

bool is_cuda_branched_log_sig_gpu_cache_prepared_(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar);

void prepare_cuda_mkw_basis_cache_(
	const BranchedSigCache& cache,
	int method,
	bool use_disk);

const CuMkwHostBasisData& get_cuda_mkw_host_basis_data_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method);

const CuMkwBasisGpuCache& get_cuda_mkw_basis_gpu_cache_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method);

void prepare_cuda_branched_bch_cache_(
	const BranchedSigCache& cache,
	bool use_disk);

void clear_cuda_branched_bch_cache_();
