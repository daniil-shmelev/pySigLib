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
#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

struct BranchedSigCoefCache {
	BranchedSigCoefCache() = default;
	BranchedSigCoefCache(
		const uint64_t* tree_data,
		uint64_t tree_data_len,
		uint64_t data_dimension,
		uint64_t dimension,
		uint64_t max_nodes,
		bool planar);

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

std::filesystem::path branched_sig_coef_cache_file_path(
	const std::filesystem::path& cache_directory,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	const std::vector<uint64_t>& tree_data);

void write_branched_sig_coef_cache(
	const std::filesystem::path& cache_directory,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	const std::vector<uint64_t>& tree_data,
	const BranchedSigCoefCache& cache);

bool read_branched_sig_coef_cache(
	const std::filesystem::path& cache_directory,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	const std::vector<uint64_t>& tree_data,
	BranchedSigCoefCache& cache);
