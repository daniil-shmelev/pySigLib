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

#include "branched_sig_cache_io.h"

#include "../cache_io.h"
#include "../../errors.h"

#include <fstream>
#include <stdexcept>
#include <system_error>

namespace {
constexpr const char* branched_cache_version_ = "v4";
}

std::filesystem::path branched_sig_cache_file_path(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	const char* prefix = planar ? "planar_branched_" : "branched_";
	return cache_directory / (
		prefix + std::to_string(dimension) + "_"
		+ std::to_string(max_nodes) + "_" + branched_cache_version_ + ".bin");
}

void write_branched_sig_cache(
	const std::filesystem::path& cache_directory,
	const BranchedSigCache& cache
) {
	std::filesystem::create_directories(cache_directory);
	const auto path = branched_sig_cache_file_path(
		cache_directory, cache.dimension, cache.max_nodes, cache.planar);
	std::ofstream out(path, std::ios::binary);
	if (!out)
		throw std::filesystem::filesystem_error(
			"Failed to open branched cache file for writing", path,
			std::make_error_code(std::errc::io_error));
	out.write(reinterpret_cast<const char*>(&cache_magic_number), sizeof(cache_magic_number));
	out.write(reinterpret_cast<const char*>(&cache.dimension), sizeof(cache.dimension));
	out.write(reinterpret_cast<const char*>(&cache.max_nodes), sizeof(cache.max_nodes));
	out.write(reinterpret_cast<const char*>(&cache.total_length), sizeof(cache.total_length));
	serialize_cache_vector(out, cache.order_index);
	serialize_cache_vector(out, cache.inv_tree_factorial);
	serialize_cache_vector(out, cache.node_labels_data);
	serialize_cache_vector(out, cache.node_labels_offsets);
	serialize_cache_vector(out, cache.chain_index_offsets);
	serialize_cache_vector(out, cache.chain_indices);
	serialize_cache_vector(out, cache.coproduct_data);
	serialize_cache_vector(out, cache.coproduct_offsets);
	serialize_cache_vector(out, cache.basis_forest_data);
	serialize_cache_vector(out, cache.basis_forest_offsets);
	out.write(reinterpret_cast<const char*>(&cache.horner.product_count),
		sizeof(cache.horner.product_count));
	serialize_cache_vector(out, cache.horner.product_build);
	serialize_cache_vector(out, cache.horner.product_build_parent);
	serialize_cache_vector(out, cache.horner.product_build_factor);
	serialize_cache_vector(out, cache.horner.flat_to_product);
	serialize_cache_vector(out, cache.horner.product_parent);
	serialize_cache_vector(out, cache.horner.product_factor);
	serialize_cache_vector(out, cache.horner.product_node_counts);
	serialize_cache_vector(out, cache.horner.coproduct_offsets);
	serialize_cache_vector(out, cache.horner.coproduct_pairs);
	serialize_cache_vector(out, cache.horner.correction_horner_node_offsets);
	serialize_cache_vector(out, cache.horner.correction_horner_variables);
	serialize_cache_vector(out, cache.horner.correction_horner_children);
	serialize_cache_vector(out, cache.horner.correction_horner_constants);
	serialize_cache_vector(out, cache.horner.correction_horner_roots);
	serialize_cache_vector(out, cache.horner.stage_offsets);
	serialize_cache_vector(out, cache.horner.stage_products);
	serialize_cache_vector(out, cache.horner.derivative_offsets);
	serialize_cache_vector(out, cache.horner.derivative_left);
	serialize_cache_vector(out, cache.horner.derivative_label);
	serialize_cache_vector(out, cache.horner.planar_coproduct_offsets);
	serialize_cache_vector(out, cache.horner.planar_coproduct_left);
	serialize_cache_vector(out, cache.horner.planar_coproduct_right);
	serialize_cache_vector(out, cache.horner.planar_log_coefficients);
	serialize_cache_vector(out, cache.horner.planar_log_flats);
	serialize_cache_vector(out, cache.horner.planar_log_flat_monomial);
	serialize_cache_vector(out, cache.horner.planar_log_monomial_parent);
	serialize_cache_vector(out, cache.horner.planar_log_monomial_label);
}

bool read_branched_sig_cache(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	BranchedSigCache& cache
) {
	const auto path = branched_sig_cache_file_path(
		cache_directory, dimension, max_nodes, planar);
	if (!std::filesystem::exists(path))
		return false;
	std::ifstream in(path, std::ios::binary);
	if (!in)
		return false;

	BranchedSigCache loaded;
	uint64_t magic = 0;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!in || magic != cache_magic_number)
		throw corrupted_cache_error(
			"Tried to read an invalid cache file. Cache may have been corrupted.");
	in.read(reinterpret_cast<char*>(&loaded.dimension), sizeof(loaded.dimension));
	in.read(reinterpret_cast<char*>(&loaded.max_nodes), sizeof(loaded.max_nodes));
	if (!in || loaded.dimension != dimension || loaded.max_nodes != max_nodes)
		return false;
	in.read(reinterpret_cast<char*>(&loaded.total_length), sizeof(loaded.total_length));
	if (!in || loaded.total_length > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error(
			"Tried to read an invalid cache file: branched total_length exceeds limit");
	deserialize_cache_vector(in, loaded.order_index, "branched order index");
	deserialize_cache_vector(
		in, loaded.inv_tree_factorial, "branched inverse factorials");
	if (loaded.inv_tree_factorial.size() + 1 > loaded.total_length)
		throw std::runtime_error(
			"Tried to read an invalid cache file: branched inverse factorial size invalid");
	deserialize_cache_vector(in, loaded.node_labels_data, "branched label data");
	deserialize_cache_vector(in, loaded.node_labels_offsets, "branched label offsets");
	deserialize_cache_vector(in, loaded.chain_index_offsets, "branched chain offsets");
	deserialize_cache_vector(in, loaded.chain_indices, "branched chain indices");
	deserialize_cache_vector(in, loaded.coproduct_data, "branched coproduct data");
	deserialize_cache_vector(in, loaded.coproduct_offsets, "branched coproduct offsets");
	deserialize_cache_vector(
		in, loaded.basis_forest_data, "branched forest data");
	deserialize_cache_vector(
		in, loaded.basis_forest_offsets, "branched forest offsets");
	in.read(reinterpret_cast<char*>(&loaded.horner.product_count),
		sizeof(loaded.horner.product_count));
	if (!in || loaded.horner.product_count > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error(
			"Tried to read an invalid cache file: branched Horner product count");
	deserialize_cache_vector(
		in, loaded.horner.product_build, "branched Horner product build");
	deserialize_cache_vector(
		in, loaded.horner.product_build_parent, "branched Horner product parents");
	deserialize_cache_vector(
		in, loaded.horner.product_build_factor, "branched Horner product factors");
	deserialize_cache_vector(
		in, loaded.horner.flat_to_product, "branched Horner flat products");
	deserialize_cache_vector(
		in, loaded.horner.product_parent,
		"branched Horner product parents");
	deserialize_cache_vector(
		in, loaded.horner.product_factor,
		"branched Horner product factors");
	deserialize_cache_vector(
		in, loaded.horner.product_node_counts,
		"branched Horner product node counts");
	deserialize_cache_vector(
		in, loaded.horner.coproduct_offsets,
		"branched Horner coproduct offsets");
	deserialize_cache_vector(
		in, loaded.horner.coproduct_pairs,
		"branched Horner coproduct pairs");
	deserialize_cache_vector(
		in, loaded.horner.correction_horner_node_offsets,
		"branched correction Horner node offsets");
	deserialize_cache_vector(
		in, loaded.horner.correction_horner_variables,
		"branched correction Horner variables");
	deserialize_cache_vector(
		in, loaded.horner.correction_horner_children,
		"branched correction Horner children");
	deserialize_cache_vector(
		in, loaded.horner.correction_horner_constants,
		"branched correction Horner constants");
	deserialize_cache_vector(
		in, loaded.horner.correction_horner_roots,
		"branched correction Horner roots");
	deserialize_cache_vector(
		in, loaded.horner.stage_offsets, "branched Horner stage offsets");
	deserialize_cache_vector(
		in, loaded.horner.stage_products, "branched Horner stage products");
	deserialize_cache_vector(
		in, loaded.horner.derivative_offsets, "branched Horner derivative offsets");
	deserialize_cache_vector(
		in, loaded.horner.derivative_left, "branched Horner derivative left");
	deserialize_cache_vector(
		in, loaded.horner.derivative_label, "branched Horner derivative labels");
	deserialize_cache_vector(
		in, loaded.horner.planar_coproduct_offsets,
		"branched Horner planar coproduct offsets");
	deserialize_cache_vector(
		in, loaded.horner.planar_coproduct_left,
		"branched Horner planar coproduct left");
	deserialize_cache_vector(
		in, loaded.horner.planar_coproduct_right,
		"branched Horner planar coproduct right");
	deserialize_cache_vector(
		in, loaded.horner.planar_log_coefficients,
		"branched Horner planar log coefficients");
	deserialize_cache_vector(
		in, loaded.horner.planar_log_flats,
		"branched Horner planar log flats");
	deserialize_cache_vector(
		in, loaded.horner.planar_log_flat_monomial,
		"branched Horner planar flat monomials");
	deserialize_cache_vector(
		in, loaded.horner.planar_log_monomial_parent,
		"branched Horner planar monomial parents");
	deserialize_cache_vector(
		in, loaded.horner.planar_log_monomial_label,
		"branched Horner planar monomial labels");
	if (!in.good())
		return false;
	if (loaded.horner.product_count == 0
		|| loaded.horner.product_build.size()
			!= loaded.horner.product_build_parent.size()
		|| loaded.horner.product_build.size()
			!= loaded.horner.product_build_factor.size()
		|| loaded.horner.product_parent.size()
			!= loaded.horner.product_factor.size()
		|| loaded.horner.product_node_counts.size()
			!= loaded.horner.product_count
		|| loaded.horner.coproduct_offsets.size()
			!= loaded.horner.product_count + 1
		|| (loaded.horner.coproduct_pairs.size() & 1) != 0
		|| loaded.horner.correction_horner_node_offsets.size()
			!= loaded.horner.correction_horner_constants.size() + 1
		|| loaded.horner.correction_horner_variables.size()
			!= loaded.horner.correction_horner_children.size()
		|| loaded.horner.correction_horner_node_offsets.empty()
		|| loaded.horner.correction_horner_node_offsets.back()
			!= loaded.horner.correction_horner_variables.size()
		|| loaded.horner.correction_horner_roots.size()
			!= loaded.total_length)
		return false;
	for (uint64_t node = 0;
		node < loaded.horner.correction_horner_constants.size(); ++node) {
		const uint64_t start
			= loaded.horner.correction_horner_node_offsets[node];
		const uint64_t end
			= loaded.horner.correction_horner_node_offsets[node + 1];
		if (start > end
			|| end > loaded.horner.correction_horner_variables.size())
			return false;
		for (uint64_t pos = start; pos < end; ++pos) {
			if (loaded.horner.correction_horner_variables[pos]
					>= loaded.total_length
				|| loaded.horner.correction_horner_children[pos] >= node)
				return false;
		}
	}
	for (const uint64_t root : loaded.horner.correction_horner_roots) {
		if (root != UINT64_MAX
			&& root >= loaded.horner.correction_horner_constants.size())
			return false;
	}
	if (planar) {
		if (loaded.basis_forest_offsets.size() != loaded.total_length
			|| loaded.basis_forest_offsets.empty()
			|| loaded.basis_forest_offsets.front() != 0
			|| loaded.basis_forest_offsets.back() != loaded.basis_forest_data.size())
			return false;
		for (uint64_t i = 1; i < loaded.basis_forest_offsets.size(); ++i) {
			if (loaded.basis_forest_offsets[i] < loaded.basis_forest_offsets[i - 1]
				|| loaded.basis_forest_offsets[i] > loaded.basis_forest_data.size())
				return false;
		}
		if (loaded.horner.planar_coproduct_offsets.size()
				!= loaded.horner.product_count + 1
			|| loaded.horner.planar_coproduct_left.size()
				!= loaded.horner.planar_coproduct_right.size()
			|| loaded.horner.planar_log_coefficients.size()
				!= loaded.total_length
			|| loaded.horner.planar_log_flats.size()
				!= loaded.horner.planar_log_flat_monomial.size()
			|| loaded.horner.planar_log_monomial_parent.empty()
			|| loaded.horner.planar_log_monomial_parent.size()
				!= loaded.horner.planar_log_monomial_label.size())
			return false;
	}
	else if (loaded.horner.flat_to_product.size() != loaded.total_length
		|| loaded.horner.product_parent.size()
			!= loaded.horner.product_count
		|| loaded.horner.derivative_offsets.size()
			!= loaded.horner.product_count + 1
		|| loaded.horner.derivative_left.size()
			!= loaded.horner.derivative_label.size()) {
		return false;
	}
	loaded.planar = planar;
	cache = std::move(loaded);
	return true;
}
