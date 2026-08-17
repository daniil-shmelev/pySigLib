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

#include "branched_sig_cache_builders.h"
#include "branched_sig_cache_common.h"

#include <unordered_map>

namespace {
void enumerate_ordered_forest_basis_(
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
}  // namespace

namespace branched_sig_cache_detail {
BranchedSigCache build_mkw_branched_sig_cache_(
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
}  // namespace branched_sig_cache_detail
