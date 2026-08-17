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

namespace branched_sig_cache_detail {
BranchedSigCache build_bck_branched_sig_cache_(
	uint64_t dimension,
	uint64_t max_nodes
) {
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
}  // namespace branched_sig_cache_detail
