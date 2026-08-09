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

#include "branched_cache.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

struct BranchedLogForestCache {
	std::vector<uint64_t> forest_offsets;
	std::vector<uint64_t> forest_trees;
	std::vector<uint64_t> forest_coprod_offsets;
	std::vector<uint64_t> forest_coprod_data;
	std::vector<uint64_t> single_tree_forest;
};

struct BranchedLogForestHash {
	static constexpr std::size_t kFibHashConst = 0x9e3779b97f4a7c15ULL;
	size_t operator()(const std::vector<uint64_t>& forest) const noexcept {
		size_t seed = forest.size();
		for (uint64_t v : forest) {
			seed ^= std::hash<uint64_t>{}(v + kFibHashConst + (seed << 6) + (seed >> 2));
		}
		return seed;
	}
};

inline uint64_t branched_log_tree_nodes(const BranchedSigCache& cache, uint64_t flat_idx) {
	const uint64_t tree_idx = flat_idx - 1;
	return cache.node_labels_offsets[tree_idx + 1] - cache.node_labels_offsets[tree_idx];
}

inline BranchedLogForestCache build_branched_log_forest_cache(const BranchedSigCache& cache) {
	using ForestMap = std::unordered_map<std::vector<uint64_t>, uint64_t, BranchedLogForestHash>;

	BranchedLogForestCache out;
	if (cache.planar) {
		out.forest_offsets.resize(cache.total_length + 1, 0);
		for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
			out.forest_offsets[flat] = out.forest_trees.size();
			out.forest_trees.push_back(flat);
		}
		out.forest_offsets[cache.total_length] = out.forest_trees.size();

		out.single_tree_forest.resize(cache.total_length, 0);
		for (uint64_t flat = 1; flat < cache.total_length; ++flat)
			out.single_tree_forest[flat] = flat;

		out.forest_coprod_offsets.resize(cache.total_length + 1, 0);
		for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
			out.forest_coprod_offsets[flat] = out.forest_coprod_data.size();
			out.forest_coprod_data.push_back(flat);
			out.forest_coprod_data.push_back(0);
			out.forest_coprod_data.push_back(0);
			out.forest_coprod_data.push_back(flat);

			const uint64_t basis_idx = flat - 1;
			uint64_t pos = cache.coproduct_offsets[basis_idx];
			const uint64_t pos_end = cache.coproduct_offsets[basis_idx + 1];
			while (pos < pos_end) {
				const uint64_t num_forest = cache.coproduct_data[pos++];
				const uint64_t right_flat = cache.coproduct_data[pos++];
				uint64_t left_flat = 0;
				if (num_forest == 1)
					left_flat = cache.coproduct_data[pos++];
				else if (num_forest != 0)
					throw std::runtime_error("Invalid MKW coproduct term");
				out.forest_coprod_data.push_back(left_flat);
				out.forest_coprod_data.push_back(right_flat);
			}
		}
		out.forest_coprod_offsets[cache.total_length] = out.forest_coprod_data.size();
		return out;
	}

	std::vector<std::vector<uint64_t>> forests;
	ForestMap forest_index;

	auto normalize_forest = [&](std::vector<uint64_t>& forest) {
		if (!cache.planar)
			std::sort(forest.begin(), forest.end());
	};

	auto add_forest = [&](const std::vector<uint64_t>& forest) -> uint64_t {
		auto [it, inserted] = forest_index.try_emplace(forest, forests.size());
		if (inserted)
			forests.push_back(forest);
		return it->second;
	};

	auto find_forest = [&](const std::vector<uint64_t>& forest) -> uint64_t {
		auto it = forest_index.find(forest);
		if (it == forest_index.end())
			throw std::runtime_error("Branched log forest not found");
		return it->second;
	};

	add_forest({});

	const uint64_t num_trees = cache.total_length - 1;
	std::vector<uint64_t> current;
	auto enumerate = [&](auto&& self, uint64_t min_flat, uint64_t remaining) -> void {
		for (uint64_t flat = min_flat; flat <= num_trees; ++flat) {
			const uint64_t nodes = branched_log_tree_nodes(cache, flat);
			if (nodes > remaining)
				continue;
			current.push_back(flat);
			add_forest(current);
			self(self, cache.planar ? 1 : flat, remaining - nodes);
			current.pop_back();
		}
	};
	enumerate(enumerate, 1, cache.max_nodes);

	out.single_tree_forest.assign(cache.total_length, 0);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		std::vector<uint64_t> singleton{ flat };
		out.single_tree_forest[flat] = find_forest(singleton);
	}

	auto combine_forests = [&](uint64_t a, uint64_t b) -> uint64_t {
		std::vector<uint64_t> combined;
		combined.reserve(forests[a].size() + forests[b].size());
		combined.insert(combined.end(), forests[a].begin(), forests[a].end());
		combined.insert(combined.end(), forests[b].begin(), forests[b].end());
		normalize_forest(combined);
		return find_forest(combined);
	};

	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> tree_coprod(cache.total_length);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		tree_coprod[flat].push_back({ out.single_tree_forest[flat], 0 });
		tree_coprod[flat].push_back({ 0, out.single_tree_forest[flat] });

		const uint64_t tree_idx = flat - 1;
		uint64_t pos = cache.coproduct_offsets[tree_idx];
		const uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];
		while (pos < pos_end) {
			const uint64_t num_forest = cache.coproduct_data[pos++];
			const uint64_t trunk_flat = cache.coproduct_data[pos++];
			std::vector<uint64_t> forest;
			forest.reserve(num_forest);
			for (uint64_t j = 0; j < num_forest; ++j)
				forest.push_back(cache.coproduct_data[pos++]);
			normalize_forest(forest);
			tree_coprod[flat].push_back({ find_forest(forest), out.single_tree_forest[trunk_flat] });
		}
	}

	out.forest_coprod_offsets.resize(forests.size() + 1, 0);
	for (uint64_t forest_idx = 0; forest_idx < forests.size(); ++forest_idx) {
		out.forest_coprod_offsets[forest_idx] = out.forest_coprod_data.size();
		std::vector<std::pair<uint64_t, uint64_t>> terms{ {0, 0} };
		for (uint64_t flat : forests[forest_idx]) {
			std::vector<std::pair<uint64_t, uint64_t>> next_terms;
			next_terms.reserve(terms.size() * tree_coprod[flat].size());
			for (const auto& term : terms) {
				for (const auto& tree_term : tree_coprod[flat]) {
					next_terms.push_back({
						combine_forests(term.first, tree_term.first),
						combine_forests(term.second, tree_term.second)
					});
				}
			}
			terms.swap(next_terms);
		}
		for (const auto& term : terms) {
			out.forest_coprod_data.push_back(term.first);
			out.forest_coprod_data.push_back(term.second);
		}
	}
	out.forest_coprod_offsets[forests.size()] = out.forest_coprod_data.size();

	out.forest_offsets.resize(forests.size() + 1, 0);
	for (uint64_t forest_idx = 0; forest_idx < forests.size(); ++forest_idx) {
		out.forest_offsets[forest_idx] = out.forest_trees.size();
		out.forest_trees.insert(out.forest_trees.end(), forests[forest_idx].begin(), forests[forest_idx].end());
	}
	out.forest_offsets[forests.size()] = out.forest_trees.size();

	return out;
}

struct BranchedLogPolyTerm {
	double coeff;
	std::vector<uint64_t> factors;
};

using BranchedLogPolynomial = std::vector<BranchedLogPolyTerm>;
using BranchedLogPolynomials = std::vector<BranchedLogPolynomial>;

namespace branched_log_poly_detail {

using Terms = std::map<std::vector<uint64_t>, double>;

inline void add(Terms& terms, double coeff, std::vector<uint64_t> factors) {
	if (coeff == 0.0)
		return;
	std::sort(factors.begin(), factors.end());
	terms[std::move(factors)] += coeff;
}

inline BranchedLogPolynomial flatten(Terms&& terms) {
	BranchedLogPolynomial out;
	out.reserve(terms.size());
	for (auto& [factors, coeff] : terms) {
		if (coeff != 0.0)
			out.push_back({ coeff, std::move(factors) });
	}
	return out;
}

inline void multiply(
	const BranchedLogPolynomial& lhs,
	const BranchedLogPolynomial& rhs,
	Terms& out
) {
	for (const auto& lterm : lhs) {
		for (const auto& rterm : rhs) {
			std::vector<uint64_t> factors = lterm.factors;
			factors.insert(factors.end(), rterm.factors.begin(), rterm.factors.end());
			add(out, lterm.coeff * rterm.coeff, std::move(factors));
		}
	}
}

}  // namespace branched_log_poly_detail

inline BranchedLogPolynomials build_branched_log_polynomials(
	const BranchedSigCache& cache,
	const BranchedLogForestCache& forests
) {
	using namespace branched_log_poly_detail;
	const uint64_t num_forests = forests.forest_offsets.size() - 1;
	BranchedLogPolynomials h(num_forests);
	for (uint64_t i = 1; i < num_forests; ++i) {
		auto first = forests.forest_trees.begin() + forests.forest_offsets[i];
		auto last = forests.forest_trees.begin() + forests.forest_offsets[i + 1];
		h[i].push_back({ 1.0, { first, last } });
		std::sort(h[i][0].factors.begin(), h[i][0].factors.end());
	}

	std::vector<BranchedLogPolynomials> powers(
		cache.max_nodes + 1, BranchedLogPolynomials(num_forests));
	if (cache.max_nodes >= 1)
		powers[1] = h;
	for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
		for (uint64_t i = 0; i < num_forests; ++i) {
			Terms terms;
			for (uint64_t pos = forests.forest_coprod_offsets[i];
				 pos < forests.forest_coprod_offsets[i + 1]; pos += 2) {
				multiply(
					powers[k - 1][forests.forest_coprod_data[pos]],
					h[forests.forest_coprod_data[pos + 1]], terms);
			}
			powers[k][i] = flatten(std::move(terms));
		}
	}

	BranchedLogPolynomials out(cache.total_length);
	for (uint64_t flat_idx = 1; flat_idx < cache.total_length; ++flat_idx) {
		Terms terms;
		const uint64_t forest = forests.single_tree_forest[flat_idx];
		for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
			const double coeff = (k % 2 ? 1.0 : -1.0) / static_cast<double>(k);
			for (const auto& term : powers[k][forest])
				add(terms, coeff * term.coeff, term.factors);
		}
		out[flat_idx] = flatten(std::move(terms));
	}
	return out;
}
