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
#include <utility>

struct BranchedSigCache {
	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;
	uint64_t total_length = 0;  // 1 + basis size (includes empty element at index 0)

	// order_index[n] = index where order-n basis elements start.
	// Flat sig index = basis index + 1 (offset by 1 for the empty element).
	std::vector<uint64_t> order_index;

	// Per-basis data (indexed 0..basis_size-1, flat sig index = basis_index + 1)
	std::vector<double> inv_tree_factorial;  // inverse tree or forest factorial

	// Flattened node labels in CSR format for cache-friendly access.
	// Basis element i's labels: node_labels_data[node_labels_offsets[i] .. node_labels_offsets[i+1])
	std::vector<uint8_t> node_labels_data;
	std::vector<uint64_t> node_labels_offsets;
	std::vector<uint64_t> basis_forest_data;
	std::vector<uint64_t> basis_forest_offsets;

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

// max_nodes is combinatorially bounded, so bit 63 is available for planar.
inline std::pair<uint64_t, uint64_t> make_branched_sig_cache_key(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	return {
		dimension,
		max_nodes | (static_cast<uint64_t>(planar) << 63)
	};
}

// Result of a single admissible cut
struct CutResult {
	std::vector<uint64_t> forest;  // tree indices of pruned subtrees
	uint64_t trunk = 0;            // tree index of the remaining trunk
};

using TreeIndexMap = std::unordered_map<CanonicalTree, uint64_t, CanonicalTreeHash>;

struct OrderedForestHash {
	size_t operator()(const std::vector<uint64_t>& forest) const noexcept {
		size_t h = forest.size();
		for (uint64_t v : forest)
			h ^= std::hash<uint64_t>()(v) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
		return h;
	}
};

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

inline void append_cut_result_(
	const DecoratedTreeInfo& tree,
	const std::vector<DecoratedTreeInfo>& trees,
	const std::vector<uint64_t>& order_index,
	const TreeIndexMap& tree_map,
	const std::vector<uint64_t>& forest,
	const std::vector<uint64_t>& trunk_children,
	std::vector<CutResult>& results,
	bool planar
) {
	CanonicalTree trunk_canonical;
	trunk_canonical.root_label = tree.canonical.root_label;
	trunk_canonical.child_ids = trunk_children;
	if (!planar)
		std::sort(trunk_canonical.child_ids.begin(), trunk_canonical.child_ids.end());
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

	std::vector<uint64_t> forest_out = forest;
	if (!planar)
		std::sort(forest_out.begin(), forest_out.end());
	results.push_back(CutResult{ std::move(forest_out), trunk_idx });
}

inline void append_forest_shuffles_(
	const std::vector<uint64_t>& a,
	const std::vector<uint64_t>& b,
	uint64_t ai,
	uint64_t bi,
	std::vector<uint64_t>& current,
	std::vector<std::vector<uint64_t>>& out
) {
	if (ai == a.size() && bi == b.size()) {
		out.push_back(current);
		return;
	}
	if (ai < a.size()) {
		current.push_back(a[ai]);
		append_forest_shuffles_(a, b, ai + 1, bi, current, out);
		current.pop_back();
	}
	if (bi < b.size()) {
		current.push_back(b[bi]);
		append_forest_shuffles_(a, b, ai, bi + 1, current, out);
		current.pop_back();
	}
}

inline void shuffle_forest_groups_(
	const std::vector<std::vector<uint64_t>>& groups,
	std::vector<std::vector<uint64_t>>& out
) {
	out.assign(1, {});
	for (const auto& group : groups) {
		if (group.empty())
			continue;
		std::vector<std::vector<uint64_t>> next;
		for (const auto& base : out) {
			std::vector<uint64_t> current;
			current.reserve(base.size() + group.size());
			append_forest_shuffles_(base, group, 0, 0, current, next);
		}
		out.swap(next);
	}
}

inline void enumerate_mkw_admissible_cuts_(
	uint64_t tree_idx,
	const std::vector<DecoratedTreeInfo>& trees,
	const std::vector<uint64_t>& order_index,
	const TreeIndexMap& tree_map,
	const std::vector<std::vector<CutResult>>& memo,
	std::vector<CutResult>& results
) {
	const auto& tree = trees[tree_idx];
	const auto& children = tree.canonical.child_ids;

	if (children.empty()) return;

	struct ChildOption {
		const std::vector<uint64_t>* forest_ptr;
		uint64_t trunk_child;
	};

	std::vector<std::vector<ChildOption>> child_internal;
	child_internal.reserve(children.size());
	for (uint64_t child_idx : children) {
		std::vector<ChildOption> options;
		options.reserve(1 + memo[child_idx].size());
		options.push_back({ nullptr, child_idx });
		for (const auto& sub_cut : memo[child_idx])
			options.push_back({ &sub_cut.forest, sub_cut.trunk });
		child_internal.push_back(std::move(options));
	}

	const uint64_t num_children = children.size();
	for (uint64_t prefix_len = 0; prefix_len <= num_children; ++prefix_len) {
		std::vector<uint64_t> prefix_forest(
			children.begin(), children.begin() + static_cast<int64_t>(prefix_len));

		if (prefix_len == num_children) {
			append_cut_result_(tree, trees, order_index, tree_map,
				prefix_forest, {}, results, true);
			continue;
		}

		std::vector<uint64_t> indices(num_children - prefix_len, 0);
		while (true) {
			std::vector<std::vector<uint64_t>> forest_groups;
			std::vector<uint64_t> trunk_children;
			bool all_kept_empty = prefix_len == 0;
			forest_groups.reserve(num_children - prefix_len + 1);
			trunk_children.reserve(num_children - prefix_len);

			for (uint64_t c = prefix_len; c < num_children; ++c) {
				const auto& opt = child_internal[c][indices[c - prefix_len]];
				if (opt.forest_ptr) {
					forest_groups.push_back(*opt.forest_ptr);
					all_kept_empty = false;
				}
				trunk_children.push_back(opt.trunk_child);
			}
			forest_groups.push_back(prefix_forest);
			if (prefix_len != 0)
				all_kept_empty = false;

			if (!all_kept_empty) {
				std::vector<std::vector<uint64_t>> forests;
				shuffle_forest_groups_(forest_groups, forests);
				for (const auto& forest : forests)
					append_cut_result_(tree, trees, order_index, tree_map,
						forest, trunk_children, results, true);
			}

			int64_t pos = static_cast<int64_t>(indices.size()) - 1;
			while (pos >= 0) {
				const uint64_t child_pos = prefix_len + static_cast<uint64_t>(pos);
				indices[pos]++;
				if (indices[pos] < child_internal[child_pos].size()) break;
				indices[pos] = 0;
				--pos;
			}
			if (pos < 0) break;
		}
	}
}

// Enumerate all admissible cuts for a tree using memoized child cuts.
// For planar trees this uses MKW left-admissible cuts.
inline void enumerate_admissible_cuts(
	uint64_t tree_idx,
	const std::vector<DecoratedTreeInfo>& trees,
	const std::vector<uint64_t>& order_index,
	const TreeIndexMap& tree_map,
	const std::vector<std::vector<CutResult>>& memo,
	std::vector<CutResult>& results,
	bool planar = false
) {
	if (planar) {
		enumerate_mkw_admissible_cuts_(tree_idx, trees, order_index, tree_map, memo, results);
		return;
	}

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
			append_cut_result_(tree, trees, order_index, tree_map,
				forest, trunk_canonical.child_ids, results, false);
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

inline uint64_t forest_node_count_(
	const std::vector<uint64_t>& forest,
	const std::vector<DecoratedTreeInfo>& trees
) {
	uint64_t out = 0;
	for (uint64_t tree_idx : forest)
		out += trees[tree_idx].canonical.num_nodes;
	return out;
}

inline void enumerate_ordered_forest_basis_(
	const std::vector<DecoratedTreeInfo>& trees,
	uint64_t max_nodes,
	std::vector<std::vector<uint64_t>>& forests,
	std::vector<uint64_t>& order_index
) {
	forests.clear();
	order_index.assign(max_nodes + 2, 0);

	std::vector<uint64_t> current;
	auto enumerate = [&](auto&& self, uint64_t remaining) -> void {
		if (remaining == 0) {
			forests.push_back(current);
			return;
		}
		for (uint64_t tree_idx = 0; tree_idx < trees.size(); ++tree_idx) {
			const uint64_t nodes = trees[tree_idx].canonical.num_nodes;
			if (nodes > remaining)
				break;
			current.push_back(tree_idx);
			self(self, remaining - nodes);
			current.pop_back();
		}
	};

	for (uint64_t order = 1; order <= max_nodes; ++order) {
		order_index[order] = forests.size();
		enumerate(enumerate, order);
	}
	order_index[max_nodes + 1] = forests.size();
}

struct ForestCutTerm_ {
	std::vector<uint64_t> left;
	std::vector<uint64_t> right;
};

inline void enumerate_mkw_forest_coproduct_terms_(
	const std::vector<uint64_t>& forest,
	const std::vector<std::vector<CutResult>>& tree_cuts,
	std::vector<ForestCutTerm_>& results
) {
	if (forest.empty())
		return;

	struct ChildOption {
		const std::vector<uint64_t>* forest_ptr;
		uint64_t trunk_child;
	};

	std::vector<std::vector<ChildOption>> child_internal;
	child_internal.reserve(forest.size());
	for (uint64_t child_idx : forest) {
		std::vector<ChildOption> options;
		options.reserve(1 + tree_cuts[child_idx].size());
		options.push_back({ nullptr, child_idx });
		for (const auto& sub_cut : tree_cuts[child_idx])
			options.push_back({ &sub_cut.forest, sub_cut.trunk });
		child_internal.push_back(std::move(options));
	}

	const uint64_t num_children = forest.size();
	for (uint64_t prefix_len = 0; prefix_len <= num_children; ++prefix_len) {
		std::vector<uint64_t> prefix_forest(
			forest.begin(), forest.begin() + static_cast<int64_t>(prefix_len));

		if (prefix_len == num_children) {
			results.push_back({ prefix_forest, {} });
			continue;
		}

		std::vector<uint64_t> indices(num_children - prefix_len, 0);
		while (true) {
			std::vector<std::vector<uint64_t>> forest_groups;
			std::vector<uint64_t> right_forest;
			bool all_kept_empty = prefix_len == 0;
			forest_groups.reserve(num_children - prefix_len + 1);
			right_forest.reserve(num_children - prefix_len);

			for (uint64_t c = prefix_len; c < num_children; ++c) {
				const auto& opt = child_internal[c][indices[c - prefix_len]];
				if (opt.forest_ptr) {
					forest_groups.push_back(*opt.forest_ptr);
					all_kept_empty = false;
				}
				right_forest.push_back(opt.trunk_child);
			}
			forest_groups.push_back(prefix_forest);
			if (prefix_len != 0)
				all_kept_empty = false;

			if (!all_kept_empty) {
				std::vector<std::vector<uint64_t>> left_forests;
				shuffle_forest_groups_(forest_groups, left_forests);
				for (const auto& left : left_forests)
					results.push_back({ left, right_forest });
			}

			int64_t pos = static_cast<int64_t>(indices.size()) - 1;
			while (pos >= 0) {
				const uint64_t child_pos = prefix_len + static_cast<uint64_t>(pos);
				indices[pos]++;
				if (indices[pos] < child_internal[child_pos].size()) break;
				indices[pos] = 0;
				--pos;
			}
			if (pos < 0) break;
		}
	}
}

inline BranchedSigCache build_mkw_branched_sig_cache_(uint64_t dimension, uint64_t max_nodes) {
	BranchedSigCache cache;
	cache.dimension = dimension;
	cache.max_nodes = max_nodes;
	cache.planar = true;

	std::vector<DecoratedTreeInfo> trees;
	std::vector<uint64_t> tree_order_index;
	enumerate_all_decorated_trees(dimension, max_nodes, trees, tree_order_index, true);

	std::vector<std::vector<uint64_t>> basis_forests;
	enumerate_ordered_forest_basis_(trees, max_nodes, basis_forests, cache.order_index);

	const uint64_t num_trees = trees.size();
	const uint64_t num_basis = basis_forests.size();
	cache.total_length = 1 + num_basis;

	std::unordered_map<std::vector<uint64_t>, uint64_t, OrderedForestHash> basis_map;
	basis_map.reserve(num_basis);
	for (uint64_t i = 0; i < num_basis; ++i)
		basis_map[basis_forests[i]] = i;

	auto basis_flat = [&](const std::vector<uint64_t>& forest) -> uint64_t {
		if (forest.empty())
			return 0;
		auto it = basis_map.find(forest);
		if (it == basis_map.end())
			throw std::runtime_error("Forest not found in MKW basis");
		return it->second + 1;
	};

	cache.inv_tree_factorial.resize(num_basis);
	cache.node_labels_offsets.resize(num_basis + 1);
	cache.basis_forest_offsets.resize(num_basis + 1);
	for (uint64_t i = 0; i < num_basis; ++i) {
		const auto& forest = basis_forests[i];
		double inv_factor = 1.0;
		double forest_factorial = 1.0;
		for (uint64_t k = 2; k <= forest.size(); ++k)
			forest_factorial *= static_cast<double>(k);
		for (uint64_t tree_idx : forest)
			inv_factor *= 1.0 / trees[tree_idx].tree_factorial;
		cache.inv_tree_factorial[i] = inv_factor / forest_factorial;

		cache.node_labels_offsets[i] = cache.node_labels_data.size();
		cache.basis_forest_offsets[i] = cache.basis_forest_data.size();
		for (uint64_t tree_idx : forest) {
			const auto& labels = trees[tree_idx].node_labels;
			cache.node_labels_data.insert(cache.node_labels_data.end(), labels.begin(), labels.end());
			cache.basis_forest_data.push_back(tree_idx);
		}
	}
	cache.node_labels_offsets[num_basis] = cache.node_labels_data.size();
	cache.basis_forest_offsets[num_basis] = cache.basis_forest_data.size();

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
	for (uint64_t i = 0; i < num_basis; ++i) {
		if (basis_forests[i].size() != 1)
			continue;
		const uint64_t tree_idx = basis_forests[i][0];
		const uint64_t order = trees[tree_idx].canonical.num_nodes;
		uint64_t word_idx = 0;
		if (chain_word_index_(tree_idx, trees, dimension, word_idx))
			cache.chain_indices[cache.chain_index_offsets[order] + word_idx] = i + 1;
	}

	TreeIndexMap tree_map;
	tree_map.reserve(num_trees);
	for (uint64_t i = 0; i < num_trees; ++i)
		tree_map[trees[i].canonical] = i;

	std::vector<std::vector<CutResult>> tree_cuts(num_trees);
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		uint64_t ostart = tree_order_index[order];
		uint64_t oend = tree_order_index[order + 1];
		for (uint64_t i = ostart; i < oend; i += dimension) {
			enumerate_mkw_admissible_cuts_(i, trees, tree_order_index, tree_map, tree_cuts, tree_cuts[i]);
			for (uint64_t L = 1; L < dimension && i + L < oend; ++L) {
				auto& dest = tree_cuts[i + L];
				dest.resize(tree_cuts[i].size());
				for (size_t k = 0; k < tree_cuts[i].size(); ++k) {
					dest[k].forest = tree_cuts[i][k].forest;
					dest[k].trunk = tree_cuts[i][k].trunk + L;
				}
			}
		}
	}

	cache.coproduct_offsets.resize(num_basis + 1, 0);
	for (uint64_t i = 0; i < num_basis; ++i) {
		cache.coproduct_offsets[i] = cache.coproduct_data.size();
		std::vector<ForestCutTerm_> terms;
		if (basis_forests[i].size() == 1) {
			const uint64_t tree_idx = basis_forests[i][0];
			for (const auto& cut : tree_cuts[tree_idx])
				terms.push_back({ cut.forest, { cut.trunk } });
		}
		else {
			enumerate_mkw_forest_coproduct_terms_(basis_forests[i], tree_cuts, terms);
		}

		const uint64_t self_flat = i + 1;
		for (const auto& term : terms) {
			const uint64_t left_flat = basis_flat(term.left);
			const uint64_t right_flat = basis_flat(term.right);
			if ((left_flat == 0 && right_flat == self_flat)
				|| (left_flat == self_flat && right_flat == 0))
				continue;
			cache.coproduct_data.push_back(left_flat == 0 ? 0 : 1);
			cache.coproduct_data.push_back(right_flat);
			if (left_flat != 0)
				cache.coproduct_data.push_back(left_flat);
		}
	}
	cache.coproduct_offsets[num_basis] = cache.coproduct_data.size();

	return cache;
}

// Build a BranchedSigCache from scratch (pure computation, no disk cache or threading).
inline BranchedSigCache build_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool planar = false) {
	if (planar)
		return build_mkw_branched_sig_cache_(dimension, max_nodes);

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
