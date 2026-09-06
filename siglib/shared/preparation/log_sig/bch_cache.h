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

#include "lyndon_words.h"
#include "log_sig_cache.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

using SparseVec = std::vector<std::pair<uint64_t, int>>;
using TensorElem = std::unordered_map<uint64_t, int>;

struct BchForwardEntry {
	uint32_t i, j;
	int coefficient;
	uint8_t orientation;
};

struct BchForwardNode {
	std::vector<uint32_t> row_ptr;
	std::vector<BchForwardEntry> entries;
};

struct BchCache {
	uint64_t dimension = 0;
	uint64_t degree = 0;
	uint64_t m = 0;
	std::vector<SparseVec> commutator_table;
	std::vector<uint32_t> comm_k_ptr;
	std::vector<uint32_t> comm_k_i;
	std::vector<uint32_t> comm_k_j;
	std::vector<int> comm_k_val;
	std::vector<double> comm_k_val_d;
	std::vector<uint32_t> comm_k_sparse_end;
	std::vector<uint32_t> comm_k_sparse_ptr;
	std::vector<uint32_t> comm_k_sparse_idx;
	std::vector<uint32_t> comm_a_ptr;
	std::vector<uint32_t> comm_a_k;
	std::vector<uint32_t> comm_a_partner;
	std::vector<int> comm_a_signed_c;
	uint32_t n_pairs = 0;
	std::vector<uint32_t> comm_ij_i;
	std::vector<uint32_t> comm_ij_j;
	std::vector<uint32_t> comm_ij_ptr;
	std::vector<uint32_t> comm_ij_k;
	std::vector<double> comm_ij_c;
	std::vector<std::pair<uint64_t, uint64_t>> linear_range;
	std::vector<uint64_t> linear_pair_ptr;
	std::vector<uint32_t> linear_pair_idx;
	std::vector<BchForwardNode> linear_forward;
	std::vector<uint64_t> coordinate_weights;
	std::vector<uint32_t> linear_input_idx;
	std::vector<uint8_t> linear_input_mask;
	bool linear_input_prefix = true;
	bool prune_linear_backprop = false;
	std::vector<uint64_t> left_factor;
	std::vector<uint64_t> right_factor;
	std::vector<double> bch_coefficients;
	std::vector<uint64_t> bch_left_factor;
	std::vector<uint64_t> bch_right_factor;
	std::vector<uint32_t> live_bch_nodes;
	bool all_bch_nodes_live = true;
};

void remove_zero_entries(TensorElem& element);
TensorElem tensor_product(
	const TensorElem& left,
	uint64_t left_degree,
	const TensorElem& right,
	uint64_t right_degree,
	uint64_t dimension,
	uint64_t max_degree);

std::vector<word> compute_factorization_indices(
	uint64_t dimension,
	uint64_t degree,
	std::vector<uint64_t>& left_factor,
	std::vector<uint64_t>& right_factor);

void build_commutator_views(BchCache& cache);
void build_standard_commutator_table(
	BchCache& cache,
	const BasisCache& basis);
void configure_linear_bch_input(
	BchCache& cache,
	std::vector<uint32_t> input_indices,
	bool prefix);
bool load_hardcoded_bch_formula(BchCache& cache);
void build_bch_formula_data(BchCache& cache);
void build_live_bch_nodes(BchCache& cache);
std::unique_ptr<BchCache> make_standard_bch_cache(
	uint64_t dimension,
	uint64_t degree,
	const BasisCache& basis);
