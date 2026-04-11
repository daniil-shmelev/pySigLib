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
#include "cppch.h"
#include "cp_branched_trees.h"
#include "words.h"  // for PairHash

struct BranchedSigCache {
	uint64_t dimension;
	uint64_t max_nodes;
	uint64_t total_length;  // 1 + num_trees (includes empty tree at index 0)

	// order_index[n] = index in the trees vector where order-n trees start.
	// Flat sig index = tree_vector_index + 1 (offset by 1 for empty tree at index 0).
	std::vector<uint64_t> order_index;

	// Per-tree data (indexed 0..num_trees-1, flat sig index = tree_index + 1)
	std::vector<double> inv_tree_factorial;  // 1.0 / gamma(tau), precomputed for hot-path multiply

	// Flattened node labels in CSR format for cache-friendly access.
	// Tree i's labels: node_labels_data[node_labels_offsets[i] .. node_labels_offsets[i+1])
	std::vector<uint8_t> node_labels_data;
	std::vector<uint64_t> node_labels_offsets;

	// Flattened Connes-Kreimer coproduct table (non-trivial cuts only).
	// Tree i's terms: coproduct_data[coproduct_offsets[i] .. coproduct_offsets[i+1])
	// Each term packed as: [num_forest_trees, trunk_flat_idx, forest_flat_idx_0, ...]
	// where flat_idx = tree_vector_index + 1.
	std::vector<uint64_t> coproduct_offsets;
	std::vector<uint64_t> coproduct_data;
};

extern std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<BranchedSigCache>,
	PairHash
> branched_sig_cache_map;
extern std::shared_mutex branched_sig_cache_mu;

void prepare_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool use_disk = false);
const BranchedSigCache& get_branched_sig_cache(uint64_t dimension, uint64_t max_nodes);
uint64_t branched_sig_length_(uint64_t dimension, uint64_t max_nodes);
void clear_branched_sig_cache();
