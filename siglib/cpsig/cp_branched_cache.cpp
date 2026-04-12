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
#include "disk_cache.h"
#include "macros.h"

// Global cache map
std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<BranchedSigCache>,
	PairHash
> branched_sig_cache_map;
std::shared_mutex branched_sig_cache_mu;

// ---------------------------------------------------------------------------
// Admissible cut enumeration and coproduct computation
// ---------------------------------------------------------------------------

// An admissible cut result: (forest_tree_indices, trunk_tree_index)
// where indices are into the decorated trees vector.
struct CutResult {
	std::vector<uint64_t> forest;  // tree indices of pruned components
	uint64_t trunk;                // tree index of the trunk
};

using TreeIndexMap = std::unordered_map<CanonicalTree, uint64_t, CanonicalTreeHash>;

// Enumerate all admissible cuts for a tree using memoized child cuts and hash-map tree lookup.
// Minimizes allocations by reusing buffers and storing forest pointers instead of copies.
static void enumerate_admissible_cuts(
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

	// ChildOption avoids vector copies: stores a pointer to the memoized forest
	// or a single index for the CUT case.
	struct ChildOption {
		const std::vector<uint64_t>* forest_ptr;  // non-null for KEEP with sub-cut
		uint64_t single_forest;                    // used when is_cut=true
		int64_t trunk_child;
		bool is_cut;
	};

	std::vector<std::vector<ChildOption>> all_child_options;
	all_child_options.reserve(children.size());

	for (uint64_t child_idx : children) {
		std::vector<ChildOption> options;
		options.reserve(2 + memo[child_idx].size());

		// Option A: CUT
		options.push_back({ nullptr, child_idx, -1, true });

		// Option B0: KEEP with empty sub-cut
		options.push_back({ nullptr, 0, static_cast<int64_t>(child_idx), false });

		// Option B1..Bn: KEEP with non-trivial sub-cuts (pointer to memoized forest)
		for (const auto& sub_cut : memo[child_idx]) {
			options.push_back({ &sub_cut.forest, 0, static_cast<int64_t>(sub_cut.trunk), false });
		}

		all_child_options.push_back(std::move(options));
	}

	uint64_t num_children = children.size();
	std::vector<uint64_t> indices(num_children, 0);

	// Pre-allocate reusable buffers
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

			std::sort(forest.begin(), forest.end());

			// Copy forest into result (can't avoid this — result owns its data)
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

// ---------------------------------------------------------------------------
// Disk cache serialization
// ---------------------------------------------------------------------------

static std::filesystem::path branched_cache_file_path(uint64_t dimension, uint64_t max_nodes) {
	return cache_dir / cache_folder_name /
		("branched_" + std::to_string(dimension) + "_" + std::to_string(max_nodes) + "_v1.bin");
}

static void write_branched_cache(const BranchedSigCache& c) {
	if (cache_dir.empty()) set_default_cache_dir();
	auto dir = cache_dir / cache_folder_name;
	if (!std::filesystem::exists(dir))
		std::filesystem::create_directory(dir);

	std::ofstream out(branched_cache_file_path(c.dimension, c.max_nodes), std::ios::binary);
	if (!out) return;

	out.write(reinterpret_cast<const char*>(&cache_magic_number), sizeof(cache_magic_number));
	out.write(reinterpret_cast<const char*>(&c.dimension), sizeof(c.dimension));
	out.write(reinterpret_cast<const char*>(&c.max_nodes), sizeof(c.max_nodes));
	out.write(reinterpret_cast<const char*>(&c.total_length), sizeof(c.total_length));
	serialize_vector(out, c.order_index);

	uint64_t n = c.inv_tree_factorial.size();
	out.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (n > 0) out.write(reinterpret_cast<const char*>(c.inv_tree_factorial.data()), n * sizeof(double));

	n = c.node_labels_data.size();
	out.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (n > 0) out.write(reinterpret_cast<const char*>(c.node_labels_data.data()), n);

	serialize_vector(out, c.node_labels_offsets);
	serialize_vector(out, c.coproduct_data);
	serialize_vector(out, c.coproduct_offsets);
}

static bool read_branched_cache(uint64_t dimension, uint64_t max_nodes, BranchedSigCache& c) {
	if (cache_dir.empty()) set_default_cache_dir();
	auto path = branched_cache_file_path(dimension, max_nodes);
	if (!std::filesystem::exists(path)) return false;

	std::ifstream in(path, std::ios::binary);
	if (!in) return false;

	// Read into a fresh local first; only move into `c` on full success.
	BranchedSigCache tmp;

	uint64_t magic;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!in || magic != cache_magic_number)
		throw corrupted_cache_error("Tried to read an invalid cache file. Cache may have been corrupted.");

	in.read(reinterpret_cast<char*>(&tmp.dimension), sizeof(tmp.dimension));
	in.read(reinterpret_cast<char*>(&tmp.max_nodes), sizeof(tmp.max_nodes));
	if (!in || tmp.dimension != dimension || tmp.max_nodes != max_nodes)
		return false;
	in.read(reinterpret_cast<char*>(&tmp.total_length), sizeof(tmp.total_length));
	if (!in || tmp.total_length > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error("Tried to read an invalid cache file: branched total_length exceeds limit");

	deserialize_vector(in, tmp.order_index);

	// inv_tree_factorial: bounded by total_length - 1 (num_trees).
	uint64_t n;
	in.read(reinterpret_cast<char*>(&n), sizeof(n));
	if (!in || n > MAX_CACHE_VECTOR_SIZE || n + 1 > tmp.total_length)
		throw std::runtime_error("Tried to read an invalid cache file: branched inv_tree_factorial size invalid");
	check_stream_has_bytes(in, n * sizeof(double), "branched inv_tree_factorial body");
	tmp.inv_tree_factorial.resize(n);
	if (n > 0) in.read(reinterpret_cast<char*>(tmp.inv_tree_factorial.data()), n * sizeof(double));

	// node_labels_data: raw bytes, also bounded.
	in.read(reinterpret_cast<char*>(&n), sizeof(n));
	if (!in || n > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error("Tried to read an invalid cache file: branched node_labels_data size invalid");
	check_stream_has_bytes(in, n, "branched node_labels_data body");
	tmp.node_labels_data.resize(n);
	if (n > 0) in.read(reinterpret_cast<char*>(tmp.node_labels_data.data()), n);

	deserialize_vector(in, tmp.node_labels_offsets);
	deserialize_vector(in, tmp.coproduct_data);
	deserialize_vector(in, tmp.coproduct_offsets);

	if (!in.good()) return false;

	c = std::move(tmp);
	return true;
}

// ---------------------------------------------------------------------------
// Cache construction
// ---------------------------------------------------------------------------

void prepare_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool use_disk) {
	std::pair<uint64_t, uint64_t> key(dimension, max_nodes);

	{
		std::shared_lock rlock(branched_sig_cache_mu);
		if (branched_sig_cache_map.find(key) != branched_sig_cache_map.end())
			return;
	}

	// Try loading from disk
	if (use_disk) {
		auto cache = std::make_unique<BranchedSigCache>();
		if (read_branched_cache(dimension, max_nodes, *cache)) {
			std::unique_lock wlock(branched_sig_cache_mu);
			branched_sig_cache_map.insert_or_assign(key, std::move(cache));
			return;
		}
	}

	// Compute from scratch
	auto cache = std::make_unique<BranchedSigCache>();
	cache->dimension = dimension;
	cache->max_nodes = max_nodes;

	std::vector<DecoratedTreeInfo> trees;
	enumerate_all_decorated_trees(dimension, max_nodes, trees, cache->order_index);

	uint64_t num_trees = trees.size();
	cache->total_length = 1 + num_trees;

	cache->inv_tree_factorial.resize(num_trees);
	for (uint64_t i = 0; i < num_trees; ++i) {
		cache->inv_tree_factorial[i] = 1.0 / trees[i].tree_factorial;
	}

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

	// Build hash map for O(1) tree lookup during coproduct construction
	TreeIndexMap tree_map;
	tree_map.reserve(num_trees);
	for (uint64_t i = 0; i < num_trees; ++i)
		tree_map[trees[i].canonical] = i;

	// Compute admissible cuts bottom-up with memoization + label symmetry.
	// Compute for label=0, derive label=1..d-1 with trunk offset (needed for memo).
	std::vector<std::vector<CutResult>> all_cuts(num_trees);
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		uint64_t ostart = cache->order_index[order];
		uint64_t oend = cache->order_index[order + 1];
		for (uint64_t i = ostart; i < oend; i += dimension) {
			enumerate_admissible_cuts(i, trees, cache->order_index, tree_map, all_cuts, all_cuts[i]);

			// Populate memo for labels 1..d-1 (share forest refs, just offset trunk)
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

	// Pre-compute total packed size for single reserve
	uint64_t total_coprod = 0;
	for (uint64_t i = 0; i < num_trees; ++i)
		for (const auto& cut : all_cuts[i])
			total_coprod += 2 + cut.forest.size();
	cache->coproduct_data.reserve(total_coprod);

	// Pack coproduct table
	cache->coproduct_offsets.resize(num_trees + 1, 0);
	for (uint64_t i = 0; i < num_trees; ++i) {
		cache->coproduct_offsets[i] = cache->coproduct_data.size();
		for (const auto& cut : all_cuts[i]) {
			cache->coproduct_data.push_back(cut.forest.size());
			cache->coproduct_data.push_back(cut.trunk + 1);
			for (uint64_t f : cut.forest)
				cache->coproduct_data.push_back(f + 1);
		}
	}
	cache->coproduct_offsets[num_trees] = cache->coproduct_data.size();

	if (use_disk) {
		write_branched_cache(*cache);
	}

	std::unique_lock wlock(branched_sig_cache_mu);
	branched_sig_cache_map.insert_or_assign(key, std::move(cache));
}

const BranchedSigCache& get_branched_sig_cache(uint64_t dimension, uint64_t max_nodes) {
	std::pair<uint64_t, uint64_t> key(dimension, max_nodes);
	std::shared_lock rlock(branched_sig_cache_mu);
	auto it = branched_sig_cache_map.find(key);
	if (it == branched_sig_cache_map.end()) {
		throw std::runtime_error("Branched signature cache not found. Call prepare_branched_sig first.");
	}
	return *(it->second);
}

uint64_t branched_sig_length_(uint64_t dimension, uint64_t max_nodes) {
	std::pair<uint64_t, uint64_t> key(dimension, max_nodes);
	{
		std::shared_lock rlock(branched_sig_cache_mu);
		auto it = branched_sig_cache_map.find(key);
		if (it != branched_sig_cache_map.end())
			return it->second->total_length;
	}
	return compute_branched_sig_length(dimension, max_nodes);
}

void clear_branched_sig_cache() {
	std::unique_lock wlock(branched_sig_cache_mu);
	branched_sig_cache_map.clear();
}

// ---------------------------------------------------------------------------
// extern "C" wrappers
// ---------------------------------------------------------------------------

extern "C" {

	CPSIG_API int prepare_branched_sig(uint64_t dimension, uint64_t max_nodes, bool use_disk) noexcept {
		SAFE_CALL(prepare_branched_sig_cache(dimension, max_nodes, use_disk));
	}

	CPSIG_API uint64_t branched_sig_length(uint64_t dimension, uint64_t max_nodes) noexcept {
		try {
			return branched_sig_length_(dimension, max_nodes);
		}
		catch (...) {
			return 0;
		}
	}

	// Returns the cache arrays' sizes so cusig can allocate before calling
	// get_branched_cache_data.  Returns 0 on success, 1 on error.
	CPSIG_API int get_branched_cache_sizes(
		uint64_t dimension, uint64_t max_nodes,
		uint64_t* total_length,
		uint64_t* num_trees,
		int* out_max_nodes,
		uint64_t* order_index_len,
		uint64_t* labels_data_len,
		uint64_t* labels_offsets_len,
		uint64_t* coprod_data_len,
		uint64_t* coprod_offsets_len
	) noexcept {
		try {
			const auto& c = get_branched_sig_cache(dimension, max_nodes);
			*total_length = c.total_length;
			*num_trees = c.total_length - 1;
			*out_max_nodes = static_cast<int>(c.max_nodes);
			*order_index_len = c.order_index.size();
			*labels_data_len = c.node_labels_data.size();
			*labels_offsets_len = c.node_labels_offsets.size();
			*coprod_data_len = c.coproduct_data.size();
			*coprod_offsets_len = c.coproduct_offsets.size();
			return 0;
		}
		catch (...) { return 1; }
	}

	// Copies cache arrays into caller-provided buffers.
	// Caller must allocate buffers of the sizes returned by get_branched_cache_sizes.
	CPSIG_API int get_branched_cache_data(
		uint64_t dimension, uint64_t max_nodes,
		double* inv_tree_factorial,
		uint8_t* node_labels_data,
		uint64_t* node_labels_offsets,
		uint64_t* coproduct_data,
		uint64_t* coproduct_offsets,
		uint64_t* order_index
	) noexcept {
		try {
			const auto& c = get_branched_sig_cache(dimension, max_nodes);
			memcpy(inv_tree_factorial, c.inv_tree_factorial.data(), c.inv_tree_factorial.size() * sizeof(double));
			memcpy(node_labels_data, c.node_labels_data.data(), c.node_labels_data.size() * sizeof(uint8_t));
			memcpy(node_labels_offsets, c.node_labels_offsets.data(), c.node_labels_offsets.size() * sizeof(uint64_t));
			memcpy(coproduct_data, c.coproduct_data.data(), c.coproduct_data.size() * sizeof(uint64_t));
			memcpy(coproduct_offsets, c.coproduct_offsets.data(), c.coproduct_offsets.size() * sizeof(uint64_t));
			memcpy(order_index, c.order_index.data(), c.order_index.size() * sizeof(uint64_t));
			return 0;
		}
		catch (...) { return 1; }
	}

}
