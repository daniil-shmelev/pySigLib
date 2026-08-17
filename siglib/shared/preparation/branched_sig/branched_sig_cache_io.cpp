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
constexpr const char* branched_cache_version_ = "v3";
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
	if (!in.good())
		return false;
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
	}
	loaded.planar = planar;
	cache = std::move(loaded);
	return true;
}
