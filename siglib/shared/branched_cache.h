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

#include "trees/coproduct.h"
#include "trees/basis_counts.h"
#include "trees/tree.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// Flattened basis data for execution kernels and stable disk serialization.
struct BranchedSigCache {
	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;
	uint64_t total_length = 0;
	std::vector<uint64_t> order_index;
	std::vector<double> inv_tree_factorial;
	std::vector<uint8_t> node_labels_data;
	std::vector<uint64_t> node_labels_offsets;
	std::vector<uint64_t> basis_forest_data;
	std::vector<uint64_t> basis_forest_offsets;
	std::vector<uint64_t> chain_index_offsets;
	std::vector<uint64_t> chain_indices;
	std::vector<uint64_t> coproduct_offsets;
	std::vector<uint64_t> coproduct_data;

	uint64_t basis_size() const noexcept {
		return total_length == 0 ? 0 : total_length - 1;
	}

	std::span<const uint8_t> basis_labels(uint64_t index) const {
		if (index >= basis_size() || index + 1 >= node_labels_offsets.size())
			throw std::out_of_range("branched basis index is out of range");
		return std::span<const uint8_t>(node_labels_data).subspan(
			static_cast<size_t>(node_labels_offsets[index]),
			static_cast<size_t>(node_labels_offsets[index + 1] - node_labels_offsets[index]));
	}

	uint64_t basis_node_count(uint64_t index) const {
		if (index >= basis_size() || index + 1 >= node_labels_offsets.size())
			throw std::out_of_range("branched basis index is out of range");
		return node_labels_offsets[index + 1] - node_labels_offsets[index];
	}

	std::span<const uint64_t> basis_forest(uint64_t index) const {
		if (index >= basis_size() || index + 1 >= basis_forest_offsets.size())
			throw std::out_of_range("branched forest basis index is out of range");
		return std::span<const uint64_t>(basis_forest_data).subspan(
			static_cast<size_t>(basis_forest_offsets[index]),
			static_cast<size_t>(basis_forest_offsets[index + 1]
				- basis_forest_offsets[index]));
	}
};

inline std::pair<uint64_t, uint64_t> make_branched_sig_cache_key(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	return { dimension, max_nodes | (static_cast<uint64_t>(planar) << 63) };
}

inline uint64_t checked_mul_add_(
	uint64_t a,
	uint64_t b,
	uint64_t c,
	const char* message
) {
	if (b != 0 && a > (UINT64_MAX - c) / b)
		throw std::overflow_error(message);
	return a * b + c;
}

inline uint64_t validate_correction_len_(
	uint64_t data_dimension,
	uint64_t max_nodes,
	uint64_t correction_len
) {
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
	TreeId tree_id,
	const TreeTable& trees,
	uint64_t dimension,
	uint64_t& word_index
) {
	word_index = trees.tree(tree_id).root_label();
	while (true) {
		const Tree& tree = trees.tree(tree_id);
		if (tree.children().empty())
			return true;
		if (tree.children().size() != 1)
			return false;
		tree_id = tree.children()[0];
		word_index = checked_mul_add_(
			word_index, dimension, trees.tree(tree_id).root_label(),
			"chain word index overflow");
	}
}

inline void enumerate_ordered_forest_basis_(
	const TreeTable& trees,
	uint64_t max_nodes,
	std::vector<Forest>& forests,
	std::vector<uint64_t>& order_offsets
) {
	// Forests are words of planar trees, ordered by total node count.
	forests.clear();
	order_offsets.assign(max_nodes + 2, 0);
	Forest current;
	const auto enumerate = [&](auto&& self, uint64_t remaining) -> void {
		if (remaining == 0) {
			forests.push_back(current);
			return;
		}
		for (TreeId tree_id = 0; tree_id < trees.size(); ++tree_id) {
			const uint64_t nodes = trees.tree(tree_id).node_count();
			if (nodes > remaining)
				break;
			current.push_back(tree_id);
			self(self, remaining - nodes);
			current.pop_back();
		}
	};

	for (uint64_t order = 1; order <= max_nodes; ++order) {
		order_offsets[order] = forests.size();
		enumerate(enumerate, order);
	}
	order_offsets[max_nodes + 1] = forests.size();
}

inline void build_chain_indices_(
	BranchedSigCache& cache,
	const TreeTable& trees,
	const std::vector<Forest>* basis_forests = nullptr
) {
	// Map single-tree chains to tensor-word indices for correction handling.
	cache.chain_index_offsets.assign(cache.max_nodes + 2, 0);
	uint64_t chain_size = 0;
	uint64_t level_size = 1;
	for (uint64_t level = 1; level <= cache.max_nodes; ++level) {
		cache.chain_index_offsets[level] = chain_size;
		if (cache.dimension != 0 && level_size > UINT64_MAX / cache.dimension)
			throw std::overflow_error("chain index size overflow");
		level_size *= cache.dimension;
		if (chain_size > UINT64_MAX - level_size)
			throw std::overflow_error("chain index size overflow");
		chain_size += level_size;
	}
	cache.chain_index_offsets[cache.max_nodes + 1] = chain_size;
	cache.chain_indices.assign(chain_size, 0);

	if (basis_forests != nullptr) {
		for (uint64_t basis_index = 0; basis_index < basis_forests->size(); ++basis_index) {
			const Forest& forest = (*basis_forests)[basis_index];
			if (forest.size() != 1)
				continue;
			const TreeId tree_id = forest[0];
			const uint64_t order = trees.tree(tree_id).node_count();
			uint64_t word_index = 0;
			if (chain_word_index_(tree_id, trees, cache.dimension, word_index)) {
				cache.chain_indices[cache.chain_index_offsets[order] + word_index]
					= basis_index + 1;
			}
		}
		return;
	}

	for (uint64_t order = 1; order <= cache.max_nodes; ++order) {
		for (TreeId tree_id = cache.order_index[order];
			tree_id < cache.order_index[order + 1]; ++tree_id) {
			uint64_t word_index = 0;
			if (chain_word_index_(tree_id, trees, cache.dimension, word_index)) {
				cache.chain_indices[cache.chain_index_offsets[order] + word_index]
					= tree_id + 1;
			}
		}
	}
}

inline void build_tree_cuts_(
	TreeTable& trees,
	const std::vector<uint64_t>& order_offsets,
	uint64_t max_nodes,
	std::vector<std::vector<TreeCut>>& cuts
) {
	// Trees with the same shape differ only by their root label in this basis.
	cuts.resize(static_cast<size_t>(trees.size()));
	std::vector<uint8_t> ready(static_cast<size_t>(trees.size()), 0);
	const uint64_t dimension = trees.dimension();
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		const TreeId start = order_offsets[order];
		const TreeId end = order_offsets[order + 1];
		for (TreeId tree_id = start; tree_id < end; tree_id += dimension) {
			ensure_tree_cuts(tree_id, trees, cuts, ready);
			for (uint64_t label = 1; label < dimension && tree_id + label < end; ++label) {
				auto& destination = cuts[tree_id + label];
				destination.resize(cuts[tree_id].size());
				for (size_t cut_index = 0;
					cut_index < cuts[tree_id].size(); ++cut_index) {
					destination[cut_index].pruned = cuts[tree_id][cut_index].pruned;
					destination[cut_index].trunk =
						cuts[tree_id][cut_index].trunk + label;
				}
				ready[tree_id + label] = 1;
			}
		}
	}
}

inline BranchedSigCache build_mkw_branched_sig_cache_(
	uint64_t dimension,
	uint64_t max_nodes
) {
	BranchedSigCache cache;
	cache.dimension = dimension;
	cache.max_nodes = max_nodes;
	cache.planar = true;

	// Build the planar domain basis first, then flatten it for execution.
	TreeTable trees(dimension, TreeKind::Planar);
	std::vector<uint64_t> tree_order_offsets;
	enumerate_trees(trees, max_nodes, tree_order_offsets);
	std::vector<Forest> basis_forests;
	enumerate_ordered_forest_basis_(
		trees, max_nodes, basis_forests, cache.order_index);
	const uint64_t basis_size = static_cast<uint64_t>(basis_forests.size());
	cache.total_length = basis_size + 1;

	std::unordered_map<Forest, uint64_t, Forest::Hash> basis_indices;
	basis_indices.reserve(static_cast<size_t>(basis_size));
	for (uint64_t index = 0; index < basis_size; ++index)
		basis_indices.emplace(basis_forests[index], index);
	const auto flat_index = [&](const Forest& forest) -> uint64_t {
		if (forest.empty())
			return 0;
		const auto existing = basis_indices.find(forest);
		if (existing == basis_indices.end())
			throw std::runtime_error("forest not found in MKW basis");
		return existing->second + 1;
	};

	// Flatten the domain objects into the stable execution and disk format.
	cache.inv_tree_factorial.resize(basis_size);
	cache.node_labels_offsets.resize(basis_size + 1);
	cache.basis_forest_offsets.resize(basis_size + 1);
	for (uint64_t index = 0; index < basis_size; ++index) {
		const Forest& forest = basis_forests[index];
		double inverse_factorial = 1.0;
		double forest_factorial = 1.0;
		for (uint64_t k = 2; k <= forest.size(); ++k)
			forest_factorial *= static_cast<double>(k);
		for (TreeId tree_id : forest)
			inverse_factorial /= trees.tree(tree_id).tree_factorial();
		cache.inv_tree_factorial[index] = inverse_factorial / forest_factorial;

		cache.node_labels_offsets[index] = cache.node_labels_data.size();
		cache.basis_forest_offsets[index] = cache.basis_forest_data.size();
		const auto labels = forest.node_labels(trees);
		cache.node_labels_data.insert(
			cache.node_labels_data.end(), labels.begin(), labels.end());
		for (TreeId tree_id : forest)
			cache.basis_forest_data.push_back(tree_id);
	}
	cache.node_labels_offsets[basis_size] = cache.node_labels_data.size();
	cache.basis_forest_offsets[basis_size] = cache.basis_forest_data.size();
	build_chain_indices_(cache, trees, &basis_forests);

	std::vector<std::vector<TreeCut>> tree_cuts;
	build_tree_cuts_(trees, tree_order_offsets, max_nodes, tree_cuts);
	cache.coproduct_offsets.resize(basis_size + 1, 0);
	for (uint64_t index = 0; index < basis_size; ++index) {
		cache.coproduct_offsets[index] = cache.coproduct_data.size();
		std::vector<CoproductTerm> terms;
		if (basis_forests[index].size() == 1) {
			const TreeId tree_id = basis_forests[index][0];
			for (const TreeCut& cut : tree_cuts[tree_id])
				terms.push_back({ cut.pruned, { cut.trunk } });
		}
		else {
			enumerate_mkw_forest_coproduct_terms(
				basis_forests[index], tree_cuts, terms);
		}

		const uint64_t self_flat = index + 1;
		for (const CoproductTerm& term : terms) {
			const uint64_t left_flat = flat_index(term.left);
			const uint64_t right_flat = flat_index(term.right);
			if ((left_flat == 0 && right_flat == self_flat)
				|| (left_flat == self_flat && right_flat == 0))
				continue;
			cache.coproduct_data.push_back(left_flat == 0 ? 0 : 1);
			cache.coproduct_data.push_back(right_flat);
			if (left_flat != 0)
				cache.coproduct_data.push_back(left_flat);
		}
	}
	cache.coproduct_offsets[basis_size] = cache.coproduct_data.size();
	return cache;
}

inline BranchedSigCache build_branched_sig_cache(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar = false
) {
	if (planar)
		return build_mkw_branched_sig_cache_(dimension, max_nodes);

	BranchedSigCache cache;
	cache.dimension = dimension;
	cache.max_nodes = max_nodes;
	cache.planar = false;

	// The BCK basis contains one non-planar tree at each non-scalar coordinate.
	TreeTable trees(dimension, TreeKind::NonPlanar);
	enumerate_trees(trees, max_nodes, cache.order_index);
	const uint64_t tree_count = trees.size();
	cache.total_length = tree_count + 1;
	cache.inv_tree_factorial.resize(tree_count);
	cache.node_labels_offsets.resize(tree_count + 1);
	for (TreeId tree_id = 0; tree_id < tree_count; ++tree_id) {
		const Tree& tree = trees.tree(tree_id);
		cache.inv_tree_factorial[tree_id] = 1.0 / tree.tree_factorial();
		cache.node_labels_offsets[tree_id] = cache.node_labels_data.size();
		const auto labels = tree.node_labels();
		cache.node_labels_data.insert(
			cache.node_labels_data.end(), labels.begin(), labels.end());
	}
	cache.node_labels_offsets[tree_count] = cache.node_labels_data.size();
	build_chain_indices_(cache, trees);

	std::vector<std::vector<TreeCut>> tree_cuts;
	build_tree_cuts_(trees, cache.order_index, max_nodes, tree_cuts);
	uint64_t coproduct_size = 0;
	for (const auto& cuts : tree_cuts) {
		for (const TreeCut& cut : cuts)
			coproduct_size += 2 + cut.pruned.size();
	}
	cache.coproduct_data.reserve(coproduct_size);
	cache.coproduct_offsets.resize(tree_count + 1, 0);
	for (TreeId tree_id = 0; tree_id < tree_count; ++tree_id) {
		cache.coproduct_offsets[tree_id] = cache.coproduct_data.size();
		for (const TreeCut& cut : tree_cuts[tree_id]) {
			cache.coproduct_data.push_back(cut.pruned.size());
			cache.coproduct_data.push_back(cut.trunk + 1);
			for (TreeId pruned : cut.pruned)
				cache.coproduct_data.push_back(pruned + 1);
		}
	}
	cache.coproduct_offsets[tree_count] = cache.coproduct_data.size();
	return cache;
}
