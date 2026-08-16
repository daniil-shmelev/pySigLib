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

#include "branched_sig_cache_common.h"

namespace {
bool chain_word_index_(
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
}  // namespace

namespace branched_sig_cache_detail {
void build_chain_indices_(
	BranchedSigCache& cache,
	const TreeTable& trees,
	const std::vector<Forest>* basis_forests
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

void build_tree_cuts_(
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
}  // namespace branched_sig_cache_detail
