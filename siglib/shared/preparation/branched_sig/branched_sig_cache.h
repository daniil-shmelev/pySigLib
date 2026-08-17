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
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Flattened basis data for execution kernels and stable disk serialization.
struct BranchedSigCache {
	BranchedSigCache() = default;
	BranchedSigCache(uint64_t dimension, uint64_t max_nodes, bool planar = false);

	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;
	uint64_t total_length = 0;
	std::vector<uint64_t> order_index;
	std::vector<double> inv_tree_factorial;
	std::vector<uint8_t> node_labels_data;
	std::vector<uint64_t> node_labels_offsets;
	std::vector<uint64_t> basis_forest_data;
	std::vector<uint64_t> basis_forest_offsets;
	std::vector<uint64_t> chain_index_offsets;
	std::vector<uint64_t> chain_indices;
	std::vector<uint64_t> coproduct_offsets;
	std::vector<uint64_t> coproduct_data;

	uint64_t basis_size() const noexcept {
		return total_length == 0 ? 0 : total_length - 1;
	}

	std::span<const uint8_t> basis_labels(uint64_t index) const {
		if (index >= basis_size() || index + 1 >= node_labels_offsets.size())
			throw std::out_of_range("branched basis index is out of range");
		return std::span<const uint8_t>(node_labels_data).subspan(
			static_cast<size_t>(node_labels_offsets[index]),
			static_cast<size_t>(node_labels_offsets[index + 1] - node_labels_offsets[index]));
	}

	uint64_t basis_node_count(uint64_t index) const {
		if (index >= basis_size() || index + 1 >= node_labels_offsets.size())
			throw std::out_of_range("branched basis index is out of range");
		return node_labels_offsets[index + 1] - node_labels_offsets[index];
	}

	std::span<const uint64_t> basis_forest(uint64_t index) const {
		if (index >= basis_size() || index + 1 >= basis_forest_offsets.size())
			throw std::out_of_range("branched forest basis index is out of range");
		return std::span<const uint64_t>(basis_forest_data).subspan(
			static_cast<size_t>(basis_forest_offsets[index]),
			static_cast<size_t>(basis_forest_offsets[index + 1]
				- basis_forest_offsets[index]));
	}
};

inline std::pair<uint64_t, uint64_t> make_branched_sig_cache_key(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	return { dimension, max_nodes | (static_cast<uint64_t>(planar) << 63) };
}

inline uint64_t checked_mul_add_(
	uint64_t a,
	uint64_t b,
	uint64_t c,
	const char* message
) {
	if (b != 0 && a > (UINT64_MAX - c) / b)
		throw std::overflow_error(message);
	return a * b + c;
}

inline uint64_t validate_correction_len_(
	uint64_t data_dimension,
	uint64_t max_nodes,
	uint64_t correction_len
) {
	if (correction_len == 0)
		return 1;
	if (max_nodes < 2)
		throw std::invalid_argument("correction must be empty when degree < 2");

	uint64_t offset = 0;
	uint64_t level_size = data_dimension;
	for (uint64_t level = 2; level <= max_nodes; ++level) {
		if (data_dimension != 0 && level_size > UINT64_MAX / data_dimension)
			throw std::overflow_error("correction level size overflow");
		level_size *= data_dimension;
		if (offset > UINT64_MAX - level_size)
			throw std::overflow_error("correction length overflow");
		offset += level_size;
		if (offset == correction_len)
			return level;
		if (offset > correction_len)
			break;
	}
	throw std::invalid_argument("correction length must be a prefix of tensor levels 2..degree");
}
