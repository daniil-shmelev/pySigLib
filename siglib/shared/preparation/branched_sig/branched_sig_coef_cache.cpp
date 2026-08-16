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

#include "branched_sig_coef_cache.h"
#include "../cache_io.h"
#include "../../errors.h"
#include "../../trees/coproduct.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>

TreeId parse_branched_sig_coef_tree_(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t& position,
	TreeTable& trees
) {
	// Input trees use preorder: root label, child count, then each child.
	if (position > tree_data_len || tree_data_len - position < 2)
		throw std::invalid_argument("branched_sig_coef tree data is truncated");
	const uint64_t label = tree_data[position++];
	const uint64_t num_children = tree_data[position++];
	if (num_children > (tree_data_len - position) / 2)
		throw std::invalid_argument("branched_sig_coef child count is invalid");
	Forest children;
	children.reserve(num_children);
	for (uint64_t i = 0; i < num_children; ++i)
		children.push_back(parse_branched_sig_coef_tree_(
			tree_data, tree_data_len, position, trees));
	return trees.intern(label, std::move(children));
}

struct BranchedSigCoefLocalTerm_ {
	std::vector<uint64_t> left;
	uint64_t right = 0;
};

BranchedSigCoefCache build_branched_sig_coef_cache_data_(
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

	// Keep only the requested coordinates and their coproduct dependencies.
	TreeTable trees(
		dimension, planar ? TreeKind::Planar : TreeKind::NonPlanar);
	std::vector<std::vector<TreeCut>> tree_cuts;
	std::vector<uint8_t> cuts_ready;
	std::vector<Forest> requested;
	requested.reserve(num_indices);
	uint64_t position = 1;
	for (uint64_t i = 0; i < num_indices; ++i) {
		if (position >= tree_data_len)
			throw std::invalid_argument("branched_sig_coef tree data is truncated");
		const uint64_t num_roots = tree_data[position++];
		if (!planar && num_roots > 1)
			throw std::invalid_argument("non-planar coefficient must contain one tree");
		Forest coordinate;
		coordinate.reserve(num_roots);
		for (uint64_t root = 0; root < num_roots; ++root)
			coordinate.push_back(parse_branched_sig_coef_tree_(
				tree_data, tree_data_len, position, trees));
		if (coordinate.node_count(trees) > max_nodes)
			throw std::invalid_argument("branched_sig_coef tree exceeds max_nodes");
		requested.push_back(std::move(coordinate));
	}
	if (position != tree_data_len)
		throw std::invalid_argument("branched_sig_coef tree data has trailing values");

	std::vector<Forest> coordinates(1);
	std::vector<std::vector<BranchedSigCoefLocalTerm_>> coordinate_terms(1);
	std::unordered_map<Forest, uint64_t, Forest::Hash> coordinate_map;
	coordinate_map.emplace(Forest{}, 0);
	auto add_coordinate = [&](const Forest& coordinate) {
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

	// Close the requested coordinates under all coproduct dependencies.
	for (uint64_t current = 1; current < coordinates.size(); ++current) {
		const auto coordinate = coordinates[current];
		std::vector<BranchedSigCoefLocalTerm_> terms;
		if (planar) {
			for (TreeId tree : coordinate)
				ensure_tree_cuts(tree, trees, tree_cuts, cuts_ready);
			std::vector<CoproductTerm> forest_terms;
			enumerate_mkw_forest_coproduct_terms(
				coordinate, tree_cuts, forest_terms);
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
			ensure_tree_cuts(coordinate[0], trees, tree_cuts, cuts_ready);
			const auto cuts = tree_cuts[coordinate[0]];
			for (const auto& cut : cuts) {
				BranchedSigCoefLocalTerm_ term;
				term.right = add_coordinate(Forest{ cut.trunk });
				for (TreeId tree : cut.pruned)
					term.left.push_back(add_coordinate(Forest{ tree }));
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
		const uint64_t a_nodes = coordinates[a].node_count(trees);
		const uint64_t b_nodes = coordinates[b].node_count(trees);
		if (a_nodes != b_nodes)
			return a_nodes < b_nodes;
		return coordinates[a] < coordinates[b];
	});

	std::vector<uint64_t> local_index(coordinates.size(), 0);
	for (uint64_t i = 0; i < ordered.size(); ++i)
		local_index[ordered[i]] = i + 1;

	// Reindex by degree and flatten to the existing sparse cache format.
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
		const uint64_t nodes = coordinate.node_count(trees);
		cache.max_nodes = std::max(cache.max_nodes, nodes);

		double inv_factorial = 1.0;
		for (TreeId tree : coordinate) {
			const Tree& tree_data = trees.tree(tree);
			inv_factorial /= tree_data.tree_factorial();
			const auto& labels = tree_data.node_labels();
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
		const auto tree = trees.find(label, Forest{});
		if (!tree)
			continue;
		const auto coordinate = coordinate_map.find(Forest{ *tree });
		if (coordinate != coordinate_map.end())
			cache.leaf_indices[label] = local_index[coordinate->second];
	}

	for (uint64_t local_idx = 1; local_idx < cache_size; ++local_idx) {
		const auto& coordinate = coordinates[ordered[local_idx - 1]];
		if (coordinate.size() != 1)
			continue;
		TreeId tree = coordinate[0];
		const uint64_t level = trees.tree(tree).node_count();
		if (level < 2 || level > max_nodes)
			continue;

		uint64_t word_index = 0;
		bool is_data_chain = true;
		while (true) {
			const Tree& node = trees.tree(tree);
			if (node.root_label() >= data_dimension) {
				is_data_chain = false;
				break;
			}
			if (word_index > (UINT64_MAX - node.root_label()) / data_dimension)
				throw std::overflow_error("branched_sig_coef correction index overflow");
			word_index = word_index * data_dimension + node.root_label();
			if (node.children().empty())
				break;
			if (node.children().size() != 1) {
				is_data_chain = false;
				break;
			}
			tree = node.children()[0];
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

constexpr const char* branched_sig_coef_cache_version = "v1";

void branched_sig_coef_cache_hash_value_(uint64_t value, uint64_t& hash) {
	for (int i = 0; i < 8; ++i) {
		hash ^= static_cast<uint8_t>(value);
		hash *= 1099511628211ULL;
		value >>= 8;
	}
}

uint64_t branched_sig_coef_cache_hash_(
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

std::filesystem::path branched_sig_coef_cache_file_path(
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

void write_branched_sig_coef_cache(
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

	out.write(reinterpret_cast<const char*>(&cache_magic_number), sizeof(cache_magic_number));
	out.write(reinterpret_cast<const char*>(&data_dimension), sizeof(data_dimension));
	out.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
	out.write(reinterpret_cast<const char*>(&max_nodes), sizeof(max_nodes));
	const uint64_t planar_value = planar;
	out.write(reinterpret_cast<const char*>(&planar_value), sizeof(planar_value));
	serialize_cache_vector(out, tree_data);
	out.write(reinterpret_cast<const char*>(&cache.max_nodes), sizeof(cache.max_nodes));
	serialize_cache_vector(out, cache.target_indices);
	serialize_cache_vector(out, cache.inv_tree_factorial);
	serialize_cache_vector(out, cache.node_labels_offsets);
	serialize_cache_vector(out, cache.node_labels_data);
	serialize_cache_vector(out, cache.coproduct_offsets);
	serialize_cache_vector(out, cache.coproduct_data);
	serialize_cache_vector(out, cache.order_index);
	serialize_cache_vector(out, cache.leaf_indices);

	std::vector<uint64_t> correction_offsets(cache.correction_indices.size());
	std::vector<uint64_t> correction_locals(cache.correction_indices.size());
	for (size_t i = 0; i < cache.correction_indices.size(); ++i) {
		correction_offsets[i] = cache.correction_indices[i].first;
		correction_locals[i] = cache.correction_indices[i].second;
	}
	serialize_cache_vector(out, correction_offsets);
	serialize_cache_vector(out, correction_locals);
}

bool read_branched_sig_coef_cache(
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
	if (!in || magic != cache_magic_number)
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
	deserialize_cache_vector(in, disk_tree_data, "branched coefficient tree data");
	if (disk_tree_data != tree_data)
		return false;

	BranchedSigCoefCache tmp;
	in.read(reinterpret_cast<char*>(&tmp.max_nodes), sizeof(tmp.max_nodes));
	if (!in || tmp.max_nodes != max_nodes)
		return false;
	deserialize_cache_vector(in, tmp.target_indices, "branched coefficient target indices");
	deserialize_cache_vector(in, tmp.inv_tree_factorial, "branched coefficient inverse factorials");
	deserialize_cache_vector(in, tmp.node_labels_offsets, "branched coefficient label offsets");
	deserialize_cache_vector(in, tmp.node_labels_data, "branched coefficient label data");
	deserialize_cache_vector(in, tmp.coproduct_offsets, "branched coefficient coproduct offsets");
	deserialize_cache_vector(in, tmp.coproduct_data, "branched coefficient coproduct data");
	deserialize_cache_vector(in, tmp.order_index, "branched coefficient order index");
	deserialize_cache_vector(in, tmp.leaf_indices, "branched coefficient leaf indices");

	std::vector<uint64_t> correction_offsets;
	std::vector<uint64_t> correction_locals;
	deserialize_cache_vector(in, correction_offsets, "branched coefficient correction offsets");
	deserialize_cache_vector(in, correction_locals, "branched coefficient correction locals");
	if (!in.good() || correction_offsets.size() != correction_locals.size())
		return false;
	tmp.correction_indices.resize(correction_offsets.size());
	for (size_t i = 0; i < correction_offsets.size(); ++i)
		tmp.correction_indices[i] = { correction_offsets[i], correction_locals[i] };

	cache = std::move(tmp);
	return true;
}

BranchedSigCoefCache::BranchedSigCoefCache(
	const uint64_t* tree_data_value,
	uint64_t tree_data_len_value,
	uint64_t data_dimension_value,
	uint64_t dimension_value,
	uint64_t max_nodes_value,
	bool planar_value
) : BranchedSigCoefCache(build_branched_sig_coef_cache_data_(
	tree_data_value,
	tree_data_len_value,
	data_dimension_value,
	dimension_value,
	max_nodes_value,
	planar_value)) {}
