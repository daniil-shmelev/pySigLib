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
#include "branched_trees.h"
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <memory>

struct BranchedSigCache {
	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;
	uint64_t total_length = 0;  // 1 + num_trees (includes empty tree at index 0)

	// order_index[n] = index in the trees vector where order-n trees start.
	// Flat sig index = tree_vector_index + 1 (offset by 1 for empty tree at index 0).
	std::vector<uint64_t> order_index;

	// Per-tree data (indexed 0..num_trees-1, flat sig index = tree_index + 1)
	std::vector<double> inv_tree_factorial;  // 1.0 / gamma(tau), precomputed for hot-path multiply

	// Flattened node labels in CSR format for cache-friendly access.
	// Tree i's labels: node_labels_data[node_labels_offsets[i] .. node_labels_offsets[i+1])
	std::vector<uint8_t> node_labels_data;
	std::vector<uint64_t> node_labels_offsets;

	// chain_indices[chain_index_offsets[n] + word_idx] is the flat signature
	// index of the length-n chain whose labels are encoded by word_idx in
	// base dimension, read from root to leaf.
	std::vector<uint64_t> chain_index_offsets;
	std::vector<uint64_t> chain_indices;

	// Flattened Connes-Kreimer coproduct table (non-trivial cuts only).
	// Tree i's terms: coproduct_data[coproduct_offsets[i] .. coproduct_offsets[i+1])
	// Each term packed as: [num_forest_trees, trunk_flat_idx, forest_flat_idx_0, ...]
	// where flat_idx = tree_vector_index + 1.
	std::vector<uint64_t> coproduct_offsets;
	std::vector<uint64_t> coproduct_data;
};

// Result of a single admissible cut
struct CutResult {
	std::vector<uint64_t> forest;  // tree indices of pruned subtrees
	uint64_t trunk = 0;            // tree index of the remaining trunk
};

using TreeIndexMap = std::unordered_map<CanonicalTree, uint64_t, CanonicalTreeHash>;

inline uint64_t checked_mul_add_(uint64_t a, uint64_t b, uint64_t c, const char* msg) {
	if (b != 0 && a > (UINT64_MAX - c) / b)
		throw std::overflow_error(msg);
	return a * b + c;
}

inline uint64_t validate_correction_len_(uint64_t data_dimension, uint64_t max_nodes, uint64_t correction_len) {
	if (correction_len == 0)
		return 1;
	if (max_nodes < 2)
		throw std::invalid_argument("correction must be empty when degree < 2");

	uint64_t offset = 0;
	uint64_t level_size = data_dimension;
	for (uint64_t level = 2; level <= max_nodes; ++level) {
		if (data_dimension != 0 && level_size > UINT64_MAX / data_dimension)
			throw std::overflow_error("correction level size overflow");
		level_size *= data_dimension;
		if (offset > UINT64_MAX - level_size)
			throw std::overflow_error("correction length overflow");
		offset += level_size;
		if (offset == correction_len)
			return level;
		if (offset > correction_len)
			break;
	}
	throw std::invalid_argument("correction length must be a prefix of tensor levels 2..degree");
}

inline bool chain_word_index_(
	uint64_t tree_idx,
	const std::vector<DecoratedTreeInfo>& trees,
	uint64_t dimension,
	uint64_t& word_idx
) {
	const auto& tree = trees[tree_idx].canonical;
	word_idx = tree.root_label;
	uint64_t cur = tree_idx;
	while (true) {
		const auto& node = trees[cur].canonical;
		if (node.child_ids.empty())
			return true;
		if (node.child_ids.size() != 1)
			return false;
		cur = node.child_ids[0];
		word_idx = checked_mul_add_(word_idx, dimension,
			trees[cur].canonical.root_label, "chain word index overflow");
	}
}

// Enumerate all admissible cuts for a tree using memoized child cuts.
// When planar=true, preserve child ordering (no sort) so distinct left-to-right
// arrangements produce distinct canonical trees and forests. The planar
// coproduct stored here is the NCK (Foissy) one: all admissible cuts with the
// concatenation product on forests. This is the convolution that satisfies
// Chen's identity for branched path signatures under path concatenation, and
// differs from the MKW (Loday-Ronco) coproduct (left-admissible cuts with
// shuffle-extended forests) used in Munthe-Kaas-Wright B-series.
inline void enumerate_admissible_cuts(
	uint64_t tree_idx,
	const std::vector<DecoratedTreeInfo>& trees,
	const std::vector<uint64_t>& order_index,
	const TreeIndexMap& tree_map,
	const std::vector<std::vector<CutResult>>& memo,
	std::vector<CutResult>& results,
	bool planar = false
) {
	const auto& tree = trees[tree_idx];
	const auto& children = tree.canonical.child_ids;

	if (children.empty()) return;

	struct ChildOption {
		const std::vector<uint64_t>* forest_ptr;
		uint64_t single_forest;
		int64_t trunk_child;
		bool is_cut;
	};

	std::vector<std::vector<ChildOption>> all_child_options;
	all_child_options.reserve(children.size());

	for (uint64_t child_idx : children) {
		std::vector<ChildOption> options;
		options.reserve(2 + memo[child_idx].size());
		options.push_back({ nullptr, child_idx, -1, true });
		options.push_back({ nullptr, 0, static_cast<int64_t>(child_idx), false });
		for (const auto& sub_cut : memo[child_idx]) {
			options.push_back({ &sub_cut.forest, 0, static_cast<int64_t>(sub_cut.trunk), false });
		}
		all_child_options.push_back(std::move(options));
	}

	uint64_t num_children = children.size();
	std::vector<uint64_t> indices(num_children, 0);

	std::vector<uint64_t> forest;
	CanonicalTree trunk_canonical;
	forest.reserve(num_children);
	trunk_canonical.child_ids.reserve(num_children);

	while (true) {
		forest.clear();
		trunk_canonical.child_ids.clear();
		bool all_kept_empty = true;

		for (uint64_t c = 0; c < num_children; ++c) {
			const auto& opt = all_child_options[c][indices[c]];
			if (opt.is_cut) {
				forest.push_back(opt.single_forest);
			} else {
				if (opt.forest_ptr) {
					for (uint64_t f : *opt.forest_ptr)
						forest.push_back(f);
				}
				if (opt.trunk_child >= 0)
					trunk_canonical.child_ids.push_back(static_cast<uint64_t>(opt.trunk_child));
			}
			if (indices[c] != 1)
				all_kept_empty = false;
		}

		if (!all_kept_empty) {
			if (!planar)
				std::sort(trunk_canonical.child_ids.begin(), trunk_canonical.child_ids.end());
			trunk_canonical.root_label = tree.canonical.root_label;
			trunk_canonical.num_nodes = 1;
			for (uint64_t tc : trunk_canonical.child_ids)
				trunk_canonical.num_nodes += trees[tc].canonical.num_nodes;

			uint64_t trunk_idx;
			if (trunk_canonical.num_nodes == 1) {
				trunk_idx = order_index[1] + trunk_canonical.root_label;
			}
			else {
				auto it = tree_map.find(trunk_canonical);
				if (it == tree_map.end())
					throw std::runtime_error("Tree not found in enumeration");
				trunk_idx = it->second;
			}

			if (!planar)
				std::sort(forest.begin(), forest.end());
			results.push_back(CutResult{ {forest.begin(), forest.end()}, trunk_idx });
		}

		int64_t pos = static_cast<int64_t>(num_children) - 1;
		while (pos >= 0) {
			indices[pos]++;
			if (indices[pos] < all_child_options[pos].size()) break;
			indices[pos] = 0;
			--pos;
		}
		if (pos < 0) break;
	}
}

// Build a BranchedSigCache from scratch (pure computation, no disk cache or threading).
inline BranchedSigCache build_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool planar = false) {
	BranchedSigCache cache;
	cache.dimension = dimension;
	cache.max_nodes = max_nodes;
	cache.planar = planar;

	std::vector<DecoratedTreeInfo> trees;
	enumerate_all_decorated_trees(dimension, max_nodes, trees, cache.order_index, planar);

	uint64_t num_trees = trees.size();
	cache.total_length = 1 + num_trees;

	cache.inv_tree_factorial.resize(num_trees);
	for (uint64_t i = 0; i < num_trees; ++i) {
		cache.inv_tree_factorial[i] = 1.0 / trees[i].tree_factorial;
	}

	cache.node_labels_offsets.resize(num_trees + 1);
	cache.node_labels_offsets[0] = 0;
	for (uint64_t i = 0; i < num_trees; ++i) {
		cache.node_labels_offsets[i + 1] = cache.node_labels_offsets[i] + trees[i].node_labels.size();
	}
	cache.node_labels_data.resize(cache.node_labels_offsets[num_trees]);
	for (uint64_t i = 0; i < num_trees; ++i) {
		std::memcpy(cache.node_labels_data.data() + cache.node_labels_offsets[i],
			trees[i].node_labels.data(), trees[i].node_labels.size());
	}

	cache.chain_index_offsets.assign(max_nodes + 2, 0);
	uint64_t chain_size = 0;
	uint64_t level_size = 1;
	for (uint64_t level = 1; level <= max_nodes; ++level) {
		cache.chain_index_offsets[level] = chain_size;
		if (dimension != 0 && level_size > UINT64_MAX / dimension)
			throw std::overflow_error("chain index size overflow");
		level_size *= dimension;
		if (chain_size > UINT64_MAX - level_size)
			throw std::overflow_error("chain index size overflow");
		chain_size += level_size;
	}
	cache.chain_index_offsets[max_nodes + 1] = chain_size;
	cache.chain_indices.assign(chain_size, 0);
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		uint64_t start = cache.order_index[order];
		uint64_t end = cache.order_index[order + 1];
		for (uint64_t i = start; i < end; ++i) {
			uint64_t word_idx = 0;
			if (chain_word_index_(i, trees, dimension, word_idx))
				cache.chain_indices[cache.chain_index_offsets[order] + word_idx] = i + 1;
		}
	}

	TreeIndexMap tree_map;
	tree_map.reserve(num_trees);
	for (uint64_t i = 0; i < num_trees; ++i)
		tree_map[trees[i].canonical] = i;

	std::vector<std::vector<CutResult>> all_cuts(num_trees);
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		uint64_t ostart = cache.order_index[order];
		uint64_t oend = cache.order_index[order + 1];
		for (uint64_t i = ostart; i < oend; i += dimension) {
			enumerate_admissible_cuts(i, trees, cache.order_index, tree_map, all_cuts, all_cuts[i], planar);
			for (uint64_t L = 1; L < dimension && i + L < oend; ++L) {
				auto& dest = all_cuts[i + L];
				dest.resize(all_cuts[i].size());
				for (size_t k = 0; k < all_cuts[i].size(); ++k) {
					dest[k].forest = all_cuts[i][k].forest;
					dest[k].trunk = all_cuts[i][k].trunk + L;
				}
			}
		}
	}

	uint64_t total_coprod = 0;
	for (uint64_t i = 0; i < num_trees; ++i)
		for (const auto& cut : all_cuts[i])
			total_coprod += 2 + cut.forest.size();
	cache.coproduct_data.reserve(total_coprod);

	cache.coproduct_offsets.resize(num_trees + 1, 0);
	for (uint64_t i = 0; i < num_trees; ++i) {
		cache.coproduct_offsets[i] = cache.coproduct_data.size();
		for (const auto& cut : all_cuts[i]) {
			cache.coproduct_data.push_back(cut.forest.size());
			cache.coproduct_data.push_back(cut.trunk + 1);
			for (uint64_t f : cut.forest)
				cache.coproduct_data.push_back(f + 1);
		}
	}
	cache.coproduct_offsets[num_trees] = cache.coproduct_data.size();

	return cache;
}
