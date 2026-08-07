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
#include "errors.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

struct BranchedSigCoefCache {
	uint64_t max_nodes = 0;
	std::vector<uint64_t> target_indices;
	std::vector<double> inv_tree_factorial;
	std::vector<uint64_t> node_labels_offsets;
	std::vector<uint8_t> node_labels_data;
	std::vector<uint64_t> coproduct_offsets;
	std::vector<uint64_t> coproduct_data;
	std::vector<uint64_t> order_index;
	std::vector<uint64_t> leaf_indices;
	std::vector<std::pair<uint64_t, uint64_t>> correction_indices;
};

struct BranchedSigCoefCut_ {
	std::vector<uint64_t> forest;
	uint64_t trunk = 0;
};

struct BranchedSigCoefTreeStore_ {
	uint64_t dimension = 0;
	bool planar = false;
	std::vector<DecoratedTreeInfo> trees;
	std::unordered_map<CanonicalTree, uint64_t, CanonicalTreeHash> tree_map;
	std::vector<std::vector<BranchedSigCoefCut_>> cuts;
	std::vector<uint8_t> cuts_ready;

	uint64_t intern(uint64_t label, std::vector<uint64_t> children) {
		if (label >= dimension)
			throw std::invalid_argument("branched_sig_coef tree label out of range");
		if (!planar)
			std::sort(children.begin(), children.end());

		CanonicalTree canonical;
		canonical.root_label = static_cast<uint8_t>(label);
		canonical.child_ids = std::move(children);
		canonical.num_nodes = 1;
		for (uint64_t child : canonical.child_ids) {
			if (canonical.num_nodes > UINT64_MAX - trees[child].canonical.num_nodes)
				throw std::overflow_error("branched_sig_coef tree size overflow");
			canonical.num_nodes += trees[child].canonical.num_nodes;
		}

		const auto existing = tree_map.find(canonical);
		if (existing != tree_map.end())
			return existing->second;

		DecoratedTreeInfo info;
		info.canonical = canonical;
		info.tree_factorial = static_cast<double>(canonical.num_nodes);
		for (uint64_t child : canonical.child_ids)
			info.tree_factorial *= trees[child].tree_factorial;
		collect_labels(info.canonical, trees, info.node_labels);

		const uint64_t index = trees.size();
		trees.push_back(std::move(info));
		tree_map.emplace(trees.back().canonical, index);
		cuts.emplace_back();
		cuts_ready.push_back(0);
		return index;
	}
};

inline uint64_t parse_branched_sig_coef_tree_(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t& position,
	BranchedSigCoefTreeStore_& store
) {
	if (position > tree_data_len || tree_data_len - position < 2)
		throw std::invalid_argument("branched_sig_coef tree data is truncated");
	const uint64_t label = tree_data[position++];
	const uint64_t num_children = tree_data[position++];
	if (num_children > (tree_data_len - position) / 2)
		throw std::invalid_argument("branched_sig_coef child count is invalid");
	std::vector<uint64_t> children;
	children.reserve(num_children);
	for (uint64_t i = 0; i < num_children; ++i)
		children.push_back(parse_branched_sig_coef_tree_(
			tree_data, tree_data_len, position, store));
	return store.intern(label, std::move(children));
}

inline void append_branched_sig_coef_shuffles_(
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
		append_branched_sig_coef_shuffles_(a, b, ai + 1, bi, current, out);
		current.pop_back();
	}
	if (bi < b.size()) {
		current.push_back(b[bi]);
		append_branched_sig_coef_shuffles_(a, b, ai, bi + 1, current, out);
		current.pop_back();
	}
}

inline void shuffle_branched_sig_coef_groups_(
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
			append_branched_sig_coef_shuffles_(
				base, group, 0, 0, current, next);
		}
		out.swap(next);
	}
}

inline void ensure_branched_sig_coef_cuts_(
	uint64_t tree_index,
	BranchedSigCoefTreeStore_& store
) {
	if (store.cuts_ready[tree_index])
		return;

	const CanonicalTree tree = store.trees[tree_index].canonical;
	for (uint64_t child : tree.child_ids)
		ensure_branched_sig_coef_cuts_(child, store);

	std::vector<BranchedSigCoefCut_> results;
	if (tree.child_ids.empty()) {
		store.cuts_ready[tree_index] = 1;
		return;
	}

	if (store.planar) {
		struct ChildOption {
			std::vector<uint64_t> forest;
			uint64_t trunk = 0;
		};
		std::vector<std::vector<ChildOption>> child_options;
		child_options.reserve(tree.child_ids.size());
		for (uint64_t child : tree.child_ids) {
			std::vector<ChildOption> options;
			options.push_back({ {}, child });
			for (const auto& cut : store.cuts[child])
				options.push_back({ cut.forest, cut.trunk });
			child_options.push_back(std::move(options));
		}

		for (uint64_t prefix_len = 0; prefix_len <= tree.child_ids.size(); ++prefix_len) {
			std::vector<uint64_t> prefix(
				tree.child_ids.begin(), tree.child_ids.begin() + prefix_len);
			if (prefix_len == tree.child_ids.size()) {
				results.push_back({ std::move(prefix), store.intern(tree.root_label, {}) });
				continue;
			}

			std::vector<uint64_t> indices(tree.child_ids.size() - prefix_len, 0);
			while (true) {
				std::vector<std::vector<uint64_t>> forest_groups;
				std::vector<uint64_t> trunk_children;
				bool nontrivial = prefix_len != 0;
				for (uint64_t c = prefix_len; c < tree.child_ids.size(); ++c) {
					const auto& option = child_options[c][indices[c - prefix_len]];
					if (!option.forest.empty()) {
						forest_groups.push_back(option.forest);
						nontrivial = true;
					}
					trunk_children.push_back(option.trunk);
				}
				forest_groups.push_back(prefix);
				if (nontrivial) {
					std::vector<std::vector<uint64_t>> forests;
					shuffle_branched_sig_coef_groups_(forest_groups, forests);
					const uint64_t trunk = store.intern(
						tree.root_label, trunk_children);
					for (auto& forest : forests)
						results.push_back({ std::move(forest), trunk });
				}

				int64_t pos = static_cast<int64_t>(indices.size()) - 1;
				while (pos >= 0) {
					const uint64_t child_pos = prefix_len + static_cast<uint64_t>(pos);
					indices[pos]++;
					if (indices[pos] < child_options[child_pos].size())
						break;
					indices[pos] = 0;
					--pos;
				}
				if (pos < 0)
					break;
			}
		}
	}
	else {
		struct ChildOption {
			std::vector<uint64_t> forest;
			uint64_t trunk = 0;
			bool has_trunk = false;
		};
		std::vector<std::vector<ChildOption>> child_options;
		child_options.reserve(tree.child_ids.size());
		for (uint64_t child : tree.child_ids) {
			std::vector<ChildOption> options;
			options.push_back({ { child }, 0, false });
			options.push_back({ {}, child, true });
			for (const auto& cut : store.cuts[child])
				options.push_back({ cut.forest, cut.trunk, true });
			child_options.push_back(std::move(options));
		}

		std::vector<uint64_t> indices(tree.child_ids.size(), 0);
		while (true) {
			std::vector<uint64_t> forest;
			std::vector<uint64_t> trunk_children;
			bool all_kept = true;
			for (uint64_t c = 0; c < tree.child_ids.size(); ++c) {
				const auto& option = child_options[c][indices[c]];
				forest.insert(forest.end(), option.forest.begin(), option.forest.end());
				if (option.has_trunk)
					trunk_children.push_back(option.trunk);
				if (indices[c] != 1)
					all_kept = false;
			}
			if (!all_kept) {
				std::sort(forest.begin(), forest.end());
				results.push_back({ std::move(forest), store.intern(
					tree.root_label, std::move(trunk_children)) });
			}

			int64_t pos = static_cast<int64_t>(indices.size()) - 1;
			while (pos >= 0) {
				indices[pos]++;
				if (indices[pos] < child_options[pos].size())
					break;
				indices[pos] = 0;
				--pos;
			}
			if (pos < 0)
				break;
		}
	}

	store.cuts[tree_index] = std::move(results);
	store.cuts_ready[tree_index] = 1;
}

struct BranchedSigCoefForestTerm_ {
	std::vector<uint64_t> left;
	std::vector<uint64_t> right;
};

inline void enumerate_branched_sig_coef_forest_terms_(
	const std::vector<uint64_t>& forest,
	const BranchedSigCoefTreeStore_& store,
	std::vector<BranchedSigCoefForestTerm_>& results
) {
	if (forest.empty())
		return;

	struct ChildOption {
		std::vector<uint64_t> forest;
		uint64_t trunk = 0;
	};
	std::vector<std::vector<ChildOption>> child_options;
	child_options.reserve(forest.size());
	for (uint64_t tree : forest) {
		std::vector<ChildOption> options;
		options.push_back({ {}, tree });
		for (const auto& cut : store.cuts[tree])
			options.push_back({ cut.forest, cut.trunk });
		child_options.push_back(std::move(options));
	}

	for (uint64_t prefix_len = 0; prefix_len <= forest.size(); ++prefix_len) {
		std::vector<uint64_t> prefix(forest.begin(), forest.begin() + prefix_len);
		if (prefix_len == forest.size()) {
			results.push_back({ std::move(prefix), {} });
			continue;
		}

		std::vector<uint64_t> indices(forest.size() - prefix_len, 0);
		while (true) {
			std::vector<std::vector<uint64_t>> forest_groups;
			std::vector<uint64_t> right;
			bool nontrivial = prefix_len != 0;
			for (uint64_t c = prefix_len; c < forest.size(); ++c) {
				const auto& option = child_options[c][indices[c - prefix_len]];
				if (!option.forest.empty()) {
					forest_groups.push_back(option.forest);
					nontrivial = true;
				}
				right.push_back(option.trunk);
			}
			forest_groups.push_back(prefix);
			if (nontrivial) {
				std::vector<std::vector<uint64_t>> lefts;
				shuffle_branched_sig_coef_groups_(forest_groups, lefts);
				for (auto& left : lefts)
					results.push_back({ std::move(left), right });
			}

			int64_t pos = static_cast<int64_t>(indices.size()) - 1;
			while (pos >= 0) {
				const uint64_t child_pos = prefix_len + static_cast<uint64_t>(pos);
				indices[pos]++;
				if (indices[pos] < child_options[child_pos].size())
					break;
				indices[pos] = 0;
				--pos;
			}
			if (pos < 0)
				break;
		}
	}
}

struct BranchedSigCoefCoordinateHash_ {
	size_t operator()(const std::vector<uint64_t>& coordinate) const noexcept {
		size_t h = coordinate.size();
		for (uint64_t tree : coordinate)
			h ^= std::hash<uint64_t>{}(tree) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
		return h;
	}
};

struct BranchedSigCoefLocalTerm_ {
	std::vector<uint64_t> left;
	uint64_t right = 0;
};

inline uint64_t branched_sig_coef_coordinate_nodes_(
	const std::vector<uint64_t>& coordinate,
	const BranchedSigCoefTreeStore_& store
) {
	uint64_t nodes = 0;
	for (uint64_t tree : coordinate) {
		if (nodes > UINT64_MAX - store.trees[tree].canonical.num_nodes)
			throw std::overflow_error("branched_sig_coef forest size overflow");
		nodes += store.trees[tree].canonical.num_nodes;
	}
	return nodes;
}

inline BranchedSigCoefCache build_branched_sig_coef_cache(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	if (tree_data == nullptr || tree_data_len == 0)
		throw std::invalid_argument("branched_sig_coef tree data is empty");
	if (dimension == 0 || dimension > 255)
		throw std::invalid_argument("branched_sig_coef dimension must be in [1, 255]");
	const uint64_t num_indices = tree_data[0];
	if (num_indices == 0)
		throw std::invalid_argument("branched_sig_coef requires at least one tree");

	BranchedSigCoefTreeStore_ store;
	store.dimension = dimension;
	store.planar = planar;
	std::vector<std::vector<uint64_t>> requested;
	requested.reserve(num_indices);
	uint64_t position = 1;
	for (uint64_t i = 0; i < num_indices; ++i) {
		if (position >= tree_data_len)
			throw std::invalid_argument("branched_sig_coef tree data is truncated");
		const uint64_t num_roots = tree_data[position++];
		if (!planar && num_roots > 1)
			throw std::invalid_argument("non-planar coefficient must contain one tree");
		std::vector<uint64_t> coordinate;
		coordinate.reserve(num_roots);
		for (uint64_t root = 0; root < num_roots; ++root)
			coordinate.push_back(parse_branched_sig_coef_tree_(
				tree_data, tree_data_len, position, store));
		if (branched_sig_coef_coordinate_nodes_(coordinate, store) > max_nodes)
			throw std::invalid_argument("branched_sig_coef tree exceeds max_nodes");
		requested.push_back(std::move(coordinate));
	}
	if (position != tree_data_len)
		throw std::invalid_argument("branched_sig_coef tree data has trailing values");

	std::vector<std::vector<uint64_t>> coordinates(1);
	std::vector<std::vector<BranchedSigCoefLocalTerm_>> coordinate_terms(1);
	std::unordered_map<
		std::vector<uint64_t>, uint64_t, BranchedSigCoefCoordinateHash_
	> coordinate_map;
	coordinate_map.emplace(std::vector<uint64_t>{}, 0);
	auto add_coordinate = [&](const std::vector<uint64_t>& coordinate) {
		const auto existing = coordinate_map.find(coordinate);
		if (existing != coordinate_map.end())
			return existing->second;
		const uint64_t index = coordinates.size();
		coordinates.push_back(coordinate);
		coordinate_terms.emplace_back();
		coordinate_map.emplace(coordinates.back(), index);
		return index;
	};

	std::vector<uint64_t> target_provisional;
	target_provisional.reserve(num_indices);
	for (const auto& coordinate : requested)
		target_provisional.push_back(add_coordinate(coordinate));

	for (uint64_t current = 1; current < coordinates.size(); ++current) {
		const auto coordinate = coordinates[current];
		std::vector<BranchedSigCoefLocalTerm_> terms;
		if (planar) {
			for (uint64_t tree : coordinate)
				ensure_branched_sig_coef_cuts_(tree, store);
			std::vector<BranchedSigCoefForestTerm_> forest_terms;
			enumerate_branched_sig_coef_forest_terms_(
				coordinate, store, forest_terms);
			for (const auto& term : forest_terms) {
				if ((term.left.empty() && term.right == coordinate)
					|| (term.left == coordinate && term.right.empty()))
					continue;
				BranchedSigCoefLocalTerm_ local_term;
				local_term.right = add_coordinate(term.right);
				if (!term.left.empty())
					local_term.left.push_back(add_coordinate(term.left));
				terms.push_back(std::move(local_term));
			}
		}
		else {
			if (coordinate.size() != 1)
				throw std::runtime_error("invalid non-planar sparse coordinate");
			ensure_branched_sig_coef_cuts_(coordinate[0], store);
			const auto cuts = store.cuts[coordinate[0]];
			for (const auto& cut : cuts) {
				BranchedSigCoefLocalTerm_ term;
				term.right = add_coordinate({ cut.trunk });
				for (uint64_t tree : cut.forest)
					term.left.push_back(add_coordinate({ tree }));
				terms.push_back(std::move(term));
			}
		}
		coordinate_terms[current] = std::move(terms);
	}

	std::vector<uint64_t> ordered;
	ordered.reserve(coordinates.size() - 1);
	for (uint64_t i = 1; i < coordinates.size(); ++i)
		ordered.push_back(i);
	std::sort(ordered.begin(), ordered.end(), [&](uint64_t a, uint64_t b) {
		const uint64_t a_nodes = branched_sig_coef_coordinate_nodes_(coordinates[a], store);
		const uint64_t b_nodes = branched_sig_coef_coordinate_nodes_(coordinates[b], store);
		if (a_nodes != b_nodes)
			return a_nodes < b_nodes;
		return coordinates[a] < coordinates[b];
	});

	std::vector<uint64_t> local_index(coordinates.size(), 0);
	for (uint64_t i = 0; i < ordered.size(); ++i)
		local_index[ordered[i]] = i + 1;

	BranchedSigCoefCache cache;
	const uint64_t cache_size = coordinates.size();
	cache.target_indices.reserve(num_indices);
	for (uint64_t target : target_provisional)
		cache.target_indices.push_back(local_index[target]);
	cache.inv_tree_factorial.assign(cache_size, 1.0);
	cache.node_labels_offsets.assign(cache_size + 1, 0);
	cache.coproduct_offsets.assign(cache_size + 1, 0);

	for (uint64_t local = 1; local < cache_size; ++local) {
		const uint64_t provisional = ordered[local - 1];
		const auto& coordinate = coordinates[provisional];
		const uint64_t nodes = branched_sig_coef_coordinate_nodes_(coordinate, store);
		cache.max_nodes = std::max(cache.max_nodes, nodes);

		double inv_factorial = 1.0;
		for (uint64_t tree : coordinate) {
			inv_factorial /= store.trees[tree].tree_factorial;
			const auto& labels = store.trees[tree].node_labels;
			cache.node_labels_data.insert(
				cache.node_labels_data.end(), labels.begin(), labels.end());
		}
		if (planar) {
			for (uint64_t k = 2; k <= coordinate.size(); ++k)
				inv_factorial /= static_cast<double>(k);
		}
		cache.inv_tree_factorial[local] = inv_factorial;
		cache.node_labels_offsets[local + 1] = cache.node_labels_data.size();

		for (const auto& term : coordinate_terms[provisional]) {
			cache.coproduct_data.push_back(term.left.size());
			cache.coproduct_data.push_back(local_index[term.right]);
			for (uint64_t left : term.left)
				cache.coproduct_data.push_back(local_index[left]);
		}
		cache.coproduct_offsets[local + 1] = cache.coproduct_data.size();
	}

	cache.order_index.assign(cache.max_nodes + 2, cache_size);
	uint64_t local = 1;
	for (uint64_t order = 1; order <= cache.max_nodes; ++order) {
		while (local < cache_size
			&& cache.node_labels_offsets[local + 1]
				- cache.node_labels_offsets[local] < order)
			++local;
		cache.order_index[order] = local;
	}

	cache.leaf_indices.assign(dimension, 0);
	for (uint64_t label = 0; label < dimension; ++label) {
		CanonicalTree leaf;
		leaf.num_nodes = 1;
		leaf.root_label = static_cast<uint8_t>(label);
		const auto tree = store.tree_map.find(leaf);
		if (tree == store.tree_map.end())
			continue;
		const auto coordinate = coordinate_map.find({ tree->second });
		if (coordinate != coordinate_map.end())
			cache.leaf_indices[label] = local_index[coordinate->second];
	}

	for (uint64_t local_idx = 1; local_idx < cache_size; ++local_idx) {
		const auto& coordinate = coordinates[ordered[local_idx - 1]];
		if (coordinate.size() != 1)
			continue;
		uint64_t tree = coordinate[0];
		const uint64_t level = store.trees[tree].canonical.num_nodes;
		if (level < 2 || level > max_nodes)
			continue;

		uint64_t word_index = 0;
		bool is_data_chain = true;
		while (true) {
			const auto& node = store.trees[tree].canonical;
			if (node.root_label >= data_dimension) {
				is_data_chain = false;
				break;
			}
			if (word_index > (UINT64_MAX - node.root_label) / data_dimension)
				throw std::overflow_error("branched_sig_coef correction index overflow");
			word_index = word_index * data_dimension + node.root_label;
			if (node.child_ids.empty())
				break;
			if (node.child_ids.size() != 1) {
				is_data_chain = false;
				break;
			}
			tree = node.child_ids[0];
		}
		if (!is_data_chain)
			continue;

		uint64_t correction_offset = 0;
		uint64_t level_size = data_dimension;
		for (uint64_t order = 2; order < level; ++order) {
			if (level_size > UINT64_MAX / data_dimension)
				throw std::overflow_error("branched_sig_coef correction index overflow");
			level_size *= data_dimension;
			if (correction_offset > UINT64_MAX - level_size)
				throw std::overflow_error("branched_sig_coef correction index overflow");
			correction_offset += level_size;
		}
		cache.correction_indices.emplace_back(
			correction_offset + word_index, local_idx);
	}
	std::sort(cache.correction_indices.begin(), cache.correction_indices.end());

	return cache;
}

inline constexpr uint64_t branched_sig_coef_cache_magic_number = 0x70797369676C6962;
inline constexpr uint64_t branched_sig_coef_max_disk_vector_size = 1'000'000'000ULL;
inline constexpr const char* branched_sig_coef_cache_version = "v1";

inline void branched_sig_coef_cache_hash_value_(uint64_t value, uint64_t& hash) {
	for (int i = 0; i < 8; ++i) {
		hash ^= static_cast<uint8_t>(value);
		hash *= 1099511628211ULL;
		value >>= 8;
	}
}

inline uint64_t branched_sig_coef_cache_hash_(
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	const std::vector<uint64_t>& tree_data
) {
	uint64_t hash = 14695981039346656037ULL;
	branched_sig_coef_cache_hash_value_(data_dimension, hash);
	branched_sig_coef_cache_hash_value_(dimension, hash);
	branched_sig_coef_cache_hash_value_(max_nodes, hash);
	branched_sig_coef_cache_hash_value_(static_cast<uint64_t>(planar), hash);
	branched_sig_coef_cache_hash_value_(tree_data.size(), hash);
	for (uint64_t value : tree_data)
		branched_sig_coef_cache_hash_value_(value, hash);
	return hash;
}

inline std::filesystem::path branched_sig_coef_cache_file_path(
	const std::filesystem::path& cache_dir,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	const std::vector<uint64_t>& tree_data
) {
	return cache_dir / (
		"branched_coef_" + std::to_string(data_dimension) + "_" +
		std::to_string(dimension) + "_" + std::to_string(max_nodes) + "_" +
		std::to_string(static_cast<uint64_t>(planar)) + "_" +
		std::to_string(branched_sig_coef_cache_hash_(
			data_dimension, dimension, max_nodes, planar, tree_data)) + "_" +
		branched_sig_coef_cache_version + ".bin");
}

template<typename T>
inline void serialize_branched_sig_coef_cache_vector_(
	std::ostream& out,
	const std::vector<T>& values
) {
	static_assert(std::is_trivially_copyable_v<T>);
	const uint64_t size = values.size();
	out.write(reinterpret_cast<const char*>(&size), sizeof(size));
	if (size > 0)
		out.write(reinterpret_cast<const char*>(values.data()), size * sizeof(T));
}

inline void check_branched_sig_coef_cache_bytes_(
	std::istream& in,
	uint64_t need,
	const char* label
) {
	const std::streampos here = in.tellg();
	in.seekg(0, std::ios::end);
	const std::streampos end = in.tellg();
	in.seekg(here);
	if (here < 0 || end < 0 || static_cast<uint64_t>(end - here) < need)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
}

template<typename T>
inline void deserialize_branched_sig_coef_cache_vector_(
	std::istream& in,
	std::vector<T>& values,
	const char* label
) {
	static_assert(std::is_trivially_copyable_v<T>);
	uint64_t size;
	in.read(reinterpret_cast<char*>(&size), sizeof(size));
	if (!in || size > branched_sig_coef_max_disk_vector_size)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
	if (size == 0) {
		values.clear();
		return;
	}
	check_branched_sig_coef_cache_bytes_(in, size * sizeof(T), label);
	values.resize(size);
	in.read(reinterpret_cast<char*>(values.data()), size * sizeof(T));
	if (!in)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
}

inline void write_branched_sig_coef_cache(
	const std::filesystem::path& cache_dir,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	const std::vector<uint64_t>& tree_data,
	const BranchedSigCoefCache& cache
) {
	const auto path = branched_sig_coef_cache_file_path(
		cache_dir, data_dimension, dimension, max_nodes, planar, tree_data);
	std::ofstream out(path, std::ios::binary);
	if (!out)
		throw std::filesystem::filesystem_error(
			"Failed to open branched coefficient cache file for writing", path,
			std::make_error_code(std::errc::io_error));

	out.write(reinterpret_cast<const char*>(&branched_sig_coef_cache_magic_number), sizeof(branched_sig_coef_cache_magic_number));
	out.write(reinterpret_cast<const char*>(&data_dimension), sizeof(data_dimension));
	out.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
	out.write(reinterpret_cast<const char*>(&max_nodes), sizeof(max_nodes));
	const uint64_t planar_value = planar;
	out.write(reinterpret_cast<const char*>(&planar_value), sizeof(planar_value));
	serialize_branched_sig_coef_cache_vector_(out, tree_data);
	out.write(reinterpret_cast<const char*>(&cache.max_nodes), sizeof(cache.max_nodes));
	serialize_branched_sig_coef_cache_vector_(out, cache.target_indices);
	serialize_branched_sig_coef_cache_vector_(out, cache.inv_tree_factorial);
	serialize_branched_sig_coef_cache_vector_(out, cache.node_labels_offsets);
	serialize_branched_sig_coef_cache_vector_(out, cache.node_labels_data);
	serialize_branched_sig_coef_cache_vector_(out, cache.coproduct_offsets);
	serialize_branched_sig_coef_cache_vector_(out, cache.coproduct_data);
	serialize_branched_sig_coef_cache_vector_(out, cache.order_index);
	serialize_branched_sig_coef_cache_vector_(out, cache.leaf_indices);

	std::vector<uint64_t> correction_offsets(cache.correction_indices.size());
	std::vector<uint64_t> correction_locals(cache.correction_indices.size());
	for (size_t i = 0; i < cache.correction_indices.size(); ++i) {
		correction_offsets[i] = cache.correction_indices[i].first;
		correction_locals[i] = cache.correction_indices[i].second;
	}
	serialize_branched_sig_coef_cache_vector_(out, correction_offsets);
	serialize_branched_sig_coef_cache_vector_(out, correction_locals);
}

inline bool read_branched_sig_coef_cache(
	const std::filesystem::path& cache_dir,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	const std::vector<uint64_t>& tree_data,
	BranchedSigCoefCache& cache
) {
	const auto path = branched_sig_coef_cache_file_path(
		cache_dir, data_dimension, dimension, max_nodes, planar, tree_data);
	if (!std::filesystem::exists(path))
		return false;

	std::ifstream in(path, std::ios::binary);
	if (!in)
		return false;

	uint64_t magic;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!in || magic != branched_sig_coef_cache_magic_number)
		throw corrupted_cache_error("Tried to read an invalid cache file. Cache may have been corrupted.");

	uint64_t disk_data_dimension;
	uint64_t disk_dimension;
	uint64_t disk_max_nodes;
	uint64_t disk_planar;
	in.read(reinterpret_cast<char*>(&disk_data_dimension), sizeof(disk_data_dimension));
	in.read(reinterpret_cast<char*>(&disk_dimension), sizeof(disk_dimension));
	in.read(reinterpret_cast<char*>(&disk_max_nodes), sizeof(disk_max_nodes));
	in.read(reinterpret_cast<char*>(&disk_planar), sizeof(disk_planar));
	if (!in || disk_data_dimension != data_dimension || disk_dimension != dimension ||
		disk_max_nodes != max_nodes || disk_planar != static_cast<uint64_t>(planar))
		return false;

	std::vector<uint64_t> disk_tree_data;
	deserialize_branched_sig_coef_cache_vector_(in, disk_tree_data, "branched coefficient tree data");
	if (disk_tree_data != tree_data)
		return false;

	BranchedSigCoefCache tmp;
	in.read(reinterpret_cast<char*>(&tmp.max_nodes), sizeof(tmp.max_nodes));
	if (!in || tmp.max_nodes != max_nodes)
		return false;
	deserialize_branched_sig_coef_cache_vector_(in, tmp.target_indices, "branched coefficient target indices");
	deserialize_branched_sig_coef_cache_vector_(in, tmp.inv_tree_factorial, "branched coefficient inverse factorials");
	deserialize_branched_sig_coef_cache_vector_(in, tmp.node_labels_offsets, "branched coefficient label offsets");
	deserialize_branched_sig_coef_cache_vector_(in, tmp.node_labels_data, "branched coefficient label data");
	deserialize_branched_sig_coef_cache_vector_(in, tmp.coproduct_offsets, "branched coefficient coproduct offsets");
	deserialize_branched_sig_coef_cache_vector_(in, tmp.coproduct_data, "branched coefficient coproduct data");
	deserialize_branched_sig_coef_cache_vector_(in, tmp.order_index, "branched coefficient order index");
	deserialize_branched_sig_coef_cache_vector_(in, tmp.leaf_indices, "branched coefficient leaf indices");

	std::vector<uint64_t> correction_offsets;
	std::vector<uint64_t> correction_locals;
	deserialize_branched_sig_coef_cache_vector_(in, correction_offsets, "branched coefficient correction offsets");
	deserialize_branched_sig_coef_cache_vector_(in, correction_locals, "branched coefficient correction locals");
	if (!in.good() || correction_offsets.size() != correction_locals.size())
		return false;
	tmp.correction_indices.resize(correction_offsets.size());
	for (size_t i = 0; i < correction_offsets.size(); ++i)
		tmp.correction_indices[i] = { correction_offsets[i], correction_locals[i] };

	cache = std::move(tmp);
	return true;
}
