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

struct CanonicalTree {
	uint64_t num_nodes;
	uint8_t root_label;
	std::vector<uint64_t> child_ids; // indices into the global decorated tree list, sorted

	bool operator<(const CanonicalTree& other) const {
		if (num_nodes != other.num_nodes) return num_nodes < other.num_nodes;
		if (root_label != other.root_label) return root_label < other.root_label;
		return child_ids < other.child_ids;
	}

	bool operator==(const CanonicalTree& other) const {
		return num_nodes == other.num_nodes
			&& root_label == other.root_label
			&& child_ids == other.child_ids;
	}
};

struct CanonicalTreeHash {
	size_t operator()(const CanonicalTree& t) const {
		size_t h = std::hash<uint64_t>()(t.num_nodes);
		h ^= std::hash<uint8_t>()(t.root_label) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
		for (uint64_t c : t.child_ids)
			h ^= std::hash<uint64_t>()(c) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
		return h;
	}
};

struct DecoratedTreeInfo {
	CanonicalTree canonical;
	std::vector<uint8_t> node_labels;  // all node labels in pre-order traversal
	double tree_factorial;             // gamma(tau)
};

inline double compute_tree_factorial(
	const CanonicalTree& tree,
	const std::vector<DecoratedTreeInfo>& all_trees
) {
	double gamma = static_cast<double>(tree.num_nodes);
	for (uint64_t child_id : tree.child_ids) {
		gamma *= all_trees[child_id].tree_factorial;
	}
	return gamma;
}

inline void collect_labels(
	const CanonicalTree& tree,
	const std::vector<DecoratedTreeInfo>& all_trees,
	std::vector<uint8_t>& out
) {
	out.push_back(tree.root_label);
	for (uint64_t child_id : tree.child_ids) {
		collect_labels(all_trees[child_id].canonical, all_trees, out);
	}
}

// Enumerate multisets of tree indices whose total node count equals target_nodes.
// Trees are appended in non-decreasing index order (min_idx enforces multiset property).
// Relies on trees being ordered by num_nodes (non-decreasing) so the break is safe.
inline void enumerate_child_multisets(
	uint64_t target_nodes,
	uint64_t min_idx,
	const std::vector<DecoratedTreeInfo>& all_trees,
	uint64_t total_count,
	std::vector<uint64_t>& current,
	std::vector<std::vector<uint64_t>>& results
) {
	if (target_nodes == 0) {
		results.push_back(current);
		return;
	}

	for (uint64_t idx = min_idx; idx < total_count; ++idx) {
		uint64_t n = all_trees[idx].canonical.num_nodes;
		if (n > target_nodes) break;
		current.push_back(idx);
		enumerate_child_multisets(target_nodes - n, idx, all_trees, total_count, current, results);
		current.pop_back();
	}
}

inline void enumerate_all_decorated_trees(
	uint64_t dimension,
	uint64_t max_nodes,
	std::vector<DecoratedTreeInfo>& trees,
	std::vector<uint64_t>& order_index
) {
	if (dimension > 255)
		throw std::invalid_argument("branched signature dimension must be <= 255");

	trees.clear();
	order_index.clear();
	order_index.resize(max_nodes + 2, 0);

	for (uint64_t order = 1; order <= max_nodes; ++order) {
		order_index[order] = trees.size();

		if (order == 1) {
			for (uint8_t label = 0; label < static_cast<uint8_t>(dimension); ++label) {
				DecoratedTreeInfo info;
				info.canonical.num_nodes = 1;
				info.canonical.root_label = label;
				info.tree_factorial = 1.0;
				info.node_labels.push_back(label);
				trees.push_back(std::move(info));
			}
		}
		else {
			uint64_t current_count = trees.size();

			std::vector<std::vector<uint64_t>> child_multisets;
			std::vector<uint64_t> current_multiset;
			enumerate_child_multisets(order - 1, 0, trees, current_count,
				current_multiset, child_multisets);

			for (const auto& children : child_multisets) {
				if (children.empty()) continue;

				for (uint8_t label = 0; label < static_cast<uint8_t>(dimension); ++label) {
					DecoratedTreeInfo info;
					info.canonical.num_nodes = order;
					info.canonical.root_label = label;
					info.canonical.child_ids = children;

					info.tree_factorial = compute_tree_factorial(info.canonical, trees);
					collect_labels(info.canonical, trees, info.node_labels);

					trees.push_back(std::move(info));
				}
			}
		}
	}

	order_index[max_nodes + 1] = trees.size();
}

inline uint64_t compute_branched_sig_length(uint64_t dimension, uint64_t max_nodes) {
	std::vector<DecoratedTreeInfo> trees;
	std::vector<uint64_t> order_index;
	enumerate_all_decorated_trees(dimension, max_nodes, trees, order_index);
	return 1 + trees.size();
}
