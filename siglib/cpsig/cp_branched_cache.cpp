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

#include "cppch.h"
#include "cpsig.h"
#include "cp_branched_cache.h"
#include "macros.h"

// Global cache map
std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<BranchedSigCache>,
	PairHash
> branched_sig_cache_map;

// ---------------------------------------------------------------------------
// Admissible cut enumeration and coproduct computation
// ---------------------------------------------------------------------------

// An admissible cut result: (forest_tree_indices, trunk_tree_index)
// where indices are into the decorated trees vector.
struct CutResult {
	std::vector<uint64_t> forest;  // tree indices of pruned components
	uint64_t trunk;                // tree index of the trunk
};

// Linear scan to find tree index within its order. Only used during prepare.
static uint64_t find_tree_index(
	const CanonicalTree& target,
	const std::vector<DecoratedTreeInfo>& trees,
	const std::vector<uint64_t>& order_index
) {
	uint64_t order = target.num_nodes;
	uint64_t start = order_index[order];
	uint64_t end = order_index[order + 1];

	for (uint64_t i = start; i < end; ++i) {
		if (trees[i].canonical == target) {
			return i;
		}
	}

	throw std::runtime_error("Tree not found in enumeration");
}

// Enumerate all admissible cuts for a tree and produce CutResults.
// An admissible cut on tree tau with children t1,...,tk:
// For each child edge, either CUT (subtree goes to forest) or KEEP (recurse).
// This produces all (forest, trunk) pairs.
//
// We only produce NON-TRIVIAL cuts:
// - Exclude the empty cut (forest=empty, trunk=whole_tree)
// - Include cuts where ALL root edges are cut (trunk=root_only)
static void enumerate_admissible_cuts(
	uint64_t tree_idx,
	const std::vector<DecoratedTreeInfo>& trees,
	const std::vector<uint64_t>& order_index,
	std::vector<CutResult>& results
) {
	const auto& tree = trees[tree_idx];
	const auto& children = tree.canonical.child_ids;

	if (children.empty()) {
		// Leaf: no edges to cut, no non-trivial cuts
		return;
	}

	// For each child, generate options:
	// Option A: CUT this edge -> child subtree goes to forest
	// Option B: KEEP this edge -> recurse into child (get sub-cuts)
	//
	// Each child's options are stored as a list of (forest_contribution, trunk_child_or_none)

	struct ChildOption {
		std::vector<uint64_t> forest_trees;   // trees added to forest
		int64_t trunk_child;                  // -1 if child is cut (no trunk child), else trunk subtree index
	};

	std::vector<std::vector<ChildOption>> all_child_options;

	for (uint64_t child_idx : children) {
		std::vector<ChildOption> options;

		// Option A: CUT - entire child subtree goes to forest
		{
			ChildOption opt;
			opt.forest_trees.push_back(child_idx);
			opt.trunk_child = -1;
			options.push_back(std::move(opt));
		}

		// Option B: KEEP - recurse into child for sub-cuts
		// Sub-option B0: empty sub-cut on child (no edges cut within child)
		{
			ChildOption opt;
			// no forest contribution
			opt.trunk_child = static_cast<int64_t>(child_idx);
			options.push_back(std::move(opt));
		}

		// Sub-options B1..Bn: non-trivial sub-cuts on child
		std::vector<CutResult> child_cuts;
		enumerate_admissible_cuts(child_idx, trees, order_index, child_cuts);
		for (const auto& sub_cut : child_cuts) {
			ChildOption opt;
			opt.forest_trees = sub_cut.forest;
			opt.trunk_child = static_cast<int64_t>(sub_cut.trunk);
			options.push_back(std::move(opt));
		}

		all_child_options.push_back(std::move(options));
	}

	// Take Cartesian product of all child options
	// Each combination produces a (forest, trunk_children) pair.
	// The trunk is reconstructed from root_label + trunk_children.

	uint64_t num_children = children.size();
	std::vector<uint64_t> indices(num_children, 0);

	while (true) {
		// Build forest and trunk children from current combination
		std::vector<uint64_t> forest;
		std::vector<uint64_t> trunk_children;
		bool all_kept_empty = true;  // track if this is the trivial empty cut

		for (uint64_t c = 0; c < num_children; ++c) {
			const auto& opt = all_child_options[c][indices[c]];
			for (uint64_t f : opt.forest_trees) {
				forest.push_back(f);
			}
			if (opt.trunk_child >= 0) {
				trunk_children.push_back(static_cast<uint64_t>(opt.trunk_child));
			}
			if (indices[c] != 1) {
				// index 1 = "keep with empty sub-cut" (the no-op)
				all_kept_empty = false;
			}
		}

		// Skip the trivial empty cut (all children kept with empty sub-cuts)
		if (!all_kept_empty) {
			// Build the trunk tree's canonical form
			std::sort(trunk_children.begin(), trunk_children.end(),
				[&trees](uint64_t a, uint64_t b) {
					return trees[a].canonical < trees[b].canonical;
				});

			CanonicalTree trunk_canonical;
			trunk_canonical.root_label = tree.canonical.root_label;
			trunk_canonical.child_ids = trunk_children;
			trunk_canonical.num_nodes = 1;
			for (uint64_t tc : trunk_children) {
				trunk_canonical.num_nodes += trees[tc].canonical.num_nodes;
			}

			// Find trunk index
			uint64_t trunk_idx;
			if (trunk_canonical.num_nodes == 1) {
				// Trunk is just the root vertex (a leaf)
				trunk_idx = order_index[1] + trunk_canonical.root_label;
			}
			else {
				trunk_idx = find_tree_index(trunk_canonical, trees, order_index);
			}

			// Sort forest for deterministic output
			std::sort(forest.begin(), forest.end());

			CutResult result;
			result.forest = std::move(forest);
			result.trunk = trunk_idx;
			results.push_back(std::move(result));
		}

		// Advance to next combination (odometer-style)
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

// ---------------------------------------------------------------------------
// Cache construction
// ---------------------------------------------------------------------------

void prepare_branched_sig_cache(uint64_t dimension, uint64_t max_nodes) {
	std::pair<uint64_t, uint64_t> key(dimension, max_nodes);

	if (branched_sig_cache_map.find(key) != branched_sig_cache_map.end()) {
		return;  // already cached
	}

	auto cache = std::make_unique<BranchedSigCache>();
	cache->dimension = dimension;
	cache->max_nodes = max_nodes;

	// Enumerate all decorated trees
	std::vector<DecoratedTreeInfo> trees;
	enumerate_all_decorated_trees(dimension, max_nodes, trees, cache->order_index);

	uint64_t num_trees = trees.size();
	cache->total_length = 1 + num_trees;

	// Extract per-tree data: precompute reciprocal factorials for hot-path multiply
	cache->inv_tree_factorial.resize(num_trees);
	for (uint64_t i = 0; i < num_trees; ++i) {
		cache->inv_tree_factorial[i] = 1.0 / trees[i].tree_factorial;
	}

	// Flatten node labels into CSR format for cache-friendly access
	cache->node_labels_offsets.resize(num_trees + 1);
	cache->node_labels_offsets[0] = 0;
	for (uint64_t i = 0; i < num_trees; ++i) {
		cache->node_labels_offsets[i + 1] = cache->node_labels_offsets[i] + trees[i].node_labels.size();
	}
	cache->node_labels_data.resize(cache->node_labels_offsets[num_trees]);
	for (uint64_t i = 0; i < num_trees; ++i) {
		std::memcpy(cache->node_labels_data.data() + cache->node_labels_offsets[i],
			trees[i].node_labels.data(), trees[i].node_labels.size());
	}

	// Build coproduct table
	cache->coproduct_offsets.resize(num_trees + 1, 0);

	for (uint64_t i = 0; i < num_trees; ++i) {
		cache->coproduct_offsets[i] = cache->coproduct_data.size();

		std::vector<CutResult> cuts;
		enumerate_admissible_cuts(i, trees, cache->order_index, cuts);

		for (const auto& cut : cuts) {
			// Pack: [num_forest_trees, trunk_flat_idx, forest_flat_idx_0, ...]
			cache->coproduct_data.push_back(cut.forest.size());
			cache->coproduct_data.push_back(cut.trunk + 1);  // +1 for flat sig index
			for (uint64_t f : cut.forest) {
				cache->coproduct_data.push_back(f + 1);  // +1 for flat sig index
			}
		}
	}
	cache->coproduct_offsets[num_trees] = cache->coproduct_data.size();

	branched_sig_cache_map.insert_or_assign(key, std::move(cache));
}

const BranchedSigCache& get_branched_sig_cache(uint64_t dimension, uint64_t max_nodes) {
	std::pair<uint64_t, uint64_t> key(dimension, max_nodes);
	auto it = branched_sig_cache_map.find(key);
	if (it == branched_sig_cache_map.end()) {
		throw std::runtime_error("Branched signature cache not found. Call prepare_branched_sig first.");
	}
	return *(it->second);
}

uint64_t branched_sig_length_(uint64_t dimension, uint64_t max_nodes) {
	std::pair<uint64_t, uint64_t> key(dimension, max_nodes);
	auto it = branched_sig_cache_map.find(key);
	if (it != branched_sig_cache_map.end()) {
		return it->second->total_length;
	}
	return compute_branched_sig_length(dimension, max_nodes);
}

// ---------------------------------------------------------------------------
// extern "C" wrappers
// ---------------------------------------------------------------------------

extern "C" {

	CPSIG_API int prepare_branched_sig(uint64_t dimension, uint64_t max_nodes) noexcept {
		SAFE_CALL(prepare_branched_sig_cache(dimension, max_nodes));
	}

	CPSIG_API uint64_t branched_sig_length(uint64_t dimension, uint64_t max_nodes) noexcept {
		try {
			return branched_sig_length_(dimension, max_nodes);
		}
		catch (...) {
			return 0;
		}
	}

}
