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
#include <stdexcept>
#include <vector>

inline uint64_t checked_branched_add_(uint64_t a, uint64_t b, const char* message) {
	if (a > UINT64_MAX - b)
		throw std::overflow_error(message);
	return a + b;
}

inline uint64_t checked_branched_mul_(uint64_t a, uint64_t b, const char* message) {
	if (a != 0 && b > UINT64_MAX / a)
		throw std::overflow_error(message);
	return a * b;
}

inline void compute_planar_branched_counts_(
	uint64_t dimension,
	uint64_t max_nodes,
	std::vector<uint64_t>& trees_per_order,
	std::vector<uint64_t>& forests_per_order
) {
	// Count planar trees and forests by degree without enumerating the basis.
	trees_per_order.assign(max_nodes + 1, 0);
	forests_per_order.assign(max_nodes + 1, 0);
	forests_per_order[0] = 1;
	if (dimension == 0 || max_nodes == 0)
		return;

	// A planar tree is a root label followed by an ordered child forest.
	trees_per_order[1] = dimension;
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		uint64_t sum = 0;
		for (uint64_t tree_order = 1; tree_order <= order; ++tree_order) {
			const uint64_t term = checked_branched_mul_(
				trees_per_order[tree_order],
				forests_per_order[order - tree_order],
				"planar branched count overflow");
			sum = checked_branched_add_(sum, term, "planar branched count overflow");
		}
		forests_per_order[order] = sum;
		if (order < max_nodes) {
			trees_per_order[order + 1] = checked_branched_mul_(
				dimension, forests_per_order[order], "planar tree count overflow");
		}
	}
}

inline uint64_t compute_branched_sig_length(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar = false
) {
	if (dimension > 255)
		throw std::invalid_argument("branched signature dimension must be <= 255");
	if (dimension == 0 || max_nodes == 0)
		return 1;
	if (max_nodes == UINT64_MAX)
		throw std::overflow_error("branched signature degree overflow");

	if (planar) {
		std::vector<uint64_t> trees_per_order;
		std::vector<uint64_t> forests_per_order;
		compute_planar_branched_counts_(
			dimension, max_nodes, trees_per_order, forests_per_order);
		uint64_t total_forests = 0;
		for (uint64_t order = 1; order <= max_nodes; ++order) {
			total_forests = checked_branched_add_(
				total_forests, forests_per_order[order],
				"planar branched length overflow");
		}
		return checked_branched_add_(
			1, total_forests, "planar branched length overflow");
	}

	std::vector<uint64_t> trees_per_order(max_nodes + 1, 0);
	trees_per_order[1] = dimension;
	uint64_t total_trees = dimension;
	std::vector<uint64_t> multiset_partitions(max_nodes + 1, 0);
	std::vector<uint64_t> divisor_weighted_sum(max_nodes + 1, 0);
	multiset_partitions[0] = 1;
	for (uint64_t order = 1; order < max_nodes; ++order) {
		uint64_t weighted_sum = 0;
		for (uint64_t divisor = 1; divisor <= order / divisor; ++divisor) {
			if (order % divisor != 0)
				continue;
			const uint64_t term = checked_branched_mul_(
				divisor, trees_per_order[divisor], "nonplanar tree count overflow");
			weighted_sum = checked_branched_add_(
				weighted_sum, term, "nonplanar tree count overflow");
			const uint64_t complementary_divisor = order / divisor;
			if (complementary_divisor != divisor) {
				const uint64_t complementary_term = checked_branched_mul_(
					complementary_divisor,
					trees_per_order[complementary_divisor],
					"nonplanar tree count overflow");
				weighted_sum = checked_branched_add_(
					weighted_sum, complementary_term,
					"nonplanar tree count overflow");
			}
		}
		divisor_weighted_sum[order] = weighted_sum;

		uint64_t sum = 0;
		for (uint64_t term = 1; term <= order; ++term) {
			const uint64_t product = checked_branched_mul_(
				divisor_weighted_sum[term],
				multiset_partitions[order - term],
				"nonplanar tree count overflow");
			sum = checked_branched_add_(sum, product, "nonplanar tree count overflow");
		}
		multiset_partitions[order] = sum / order;

		trees_per_order[order + 1] = checked_branched_mul_(
			dimension, multiset_partitions[order], "nonplanar tree count overflow");
		total_trees = checked_branched_add_(
			total_trees, trees_per_order[order + 1],
			"nonplanar branched length overflow");
	}
	return checked_branched_add_(1, total_trees, "nonplanar branched length overflow");
}

inline uint64_t compute_branched_log_sig_length(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar = false
) {
	if (dimension > 255)
		throw std::invalid_argument("branched log signature dimension must be <= 255");
	if (dimension == 0 || max_nodes == 0)
		return 0;
	if (max_nodes == UINT64_MAX)
		throw std::overflow_error("branched log signature degree overflow");
	if (!planar)
		return compute_branched_sig_length(dimension, max_nodes, false) - 1;

	std::vector<uint64_t> trees_per_order;
	std::vector<uint64_t> words_per_order;
	compute_planar_branched_counts_(
		dimension, max_nodes, trees_per_order, words_per_order);

	// The Euler transform recovers weighted Lyndon dimensions from word counts.
	std::vector<uint64_t> logarithmic_counts(max_nodes + 1, 0);
	std::vector<uint64_t> lyndon_counts(max_nodes + 1, 0);
	uint64_t result = 0;
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		const uint64_t leading = checked_branched_mul_(
			order, words_per_order[order], "branched log signature length overflow");
		uint64_t convolution = 0;
		for (uint64_t k = 1; k < order; ++k) {
			const uint64_t term = checked_branched_mul_(
				logarithmic_counts[k], words_per_order[order - k],
				"branched log signature length overflow");
			convolution = checked_branched_add_(
				convolution, term, "branched log signature length overflow");
		}
		if (convolution > leading)
			throw std::overflow_error("branched log signature length recurrence underflow");
		logarithmic_counts[order] = leading - convolution;

		uint64_t proper_divisor_sum = 0;
		for (uint64_t divisor = 1; divisor < order; ++divisor) {
			if (order % divisor != 0)
				continue;
			const uint64_t term = checked_branched_mul_(
				divisor, lyndon_counts[divisor],
				"branched log signature length overflow");
			proper_divisor_sum = checked_branched_add_(
				proper_divisor_sum, term, "branched log signature length overflow");
		}
		if (proper_divisor_sum > logarithmic_counts[order])
			throw std::overflow_error("branched log signature length recurrence underflow");
		const uint64_t numerator = logarithmic_counts[order] - proper_divisor_sum;
		if (numerator % order != 0)
			throw std::runtime_error("branched log signature length recurrence is not integral");
		lyndon_counts[order] = numerator / order;
		result = checked_branched_add_(
			result, lyndon_counts[order], "branched log signature length overflow");
	}
	return result;
}
