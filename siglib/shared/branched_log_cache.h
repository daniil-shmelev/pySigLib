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
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct BranchedLogProductChain {
	std::vector<uint64_t> parent;
	std::vector<uint64_t> last_factor;
};

struct BranchedLogProductCsr {
	std::vector<uint64_t> offsets;
	std::vector<uint64_t> factors;
};

// A product is a forest of flat branched-signature coordinates. The Horner
// plan stores the reduced coproduct and the backend-specific product layouts.
struct BranchedLogHornerPlan {
	uint64_t product_count = 0;
	BranchedLogProductChain cpu_products;
	BranchedLogProductCsr cuda_products;
	// CSR list of (left product, right product) coproduct pairs.
	std::vector<uint64_t> coproduct_offsets;
	std::vector<uint64_t> coproduct_pairs;
	// Empty for planar signatures, where the mapping is the identity.
	std::vector<uint64_t> flat_to_product;
};

struct BranchedLogProductHash {
	static constexpr std::size_t kFibHashConst = 0x9e3779b97f4a7c15ULL;
	size_t operator()(const std::vector<uint64_t>& forest) const noexcept {
		size_t seed = forest.size();
		for (uint64_t v : forest) {
			seed ^= std::hash<uint64_t>{}(v + kFibHashConst + (seed << 6) + (seed >> 2));
		}
		return seed;
	}
};

namespace branched_log_detail {

using Product = std::vector<uint64_t>;
using ProductMap = std::unordered_map<Product, uint64_t, BranchedLogProductHash>;
using ProductPair = std::pair<uint64_t, uint64_t>;
using TreeCoproduct = std::vector<std::vector<ProductPair>>;

inline uint64_t find_product_(const ProductMap& products, const Product& product) {
	const auto found = products.find(product);
	if (found == products.end())
		throw std::runtime_error("Branched log forest not found");
	return found->second;
}

inline std::vector<Product> enumerate_nonplanar_products_(
	const BranchedSigCache& cache,
	ProductMap& product_index
) {
	std::vector<Product> products;
	product_index.reserve(cache.total_length);
	product_index.emplace(Product{}, 0);
	products.push_back({});

	// Enumerate nondecreasing tree-coordinate sequences by total node count.
	const uint64_t num_trees = cache.total_length - 1;
	Product current;
	const auto enumerate = [&](auto&& self, uint64_t min_flat, uint64_t remaining) -> void {
		for (uint64_t flat = min_flat; flat <= num_trees; ++flat) {
			const uint64_t nodes = cache.basis_node_count(flat - 1);
			if (nodes > remaining)
				continue;
			current.push_back(flat);
			const bool inserted = product_index.try_emplace(current, products.size()).second;
			if (inserted)
				products.push_back(current);
			self(self, flat, remaining - nodes);
			current.pop_back();
		}
	};
	enumerate(enumerate, 1, cache.max_nodes);
	return products;
}

inline BranchedLogHornerPlan build_planar_horner_plan_(
	const BranchedSigCache& cache
) {
	BranchedLogHornerPlan out;
	// MKW coordinates already are ordered forests, so each flat coordinate is
	// one product. All product mappings are therefore implicit.
	out.product_count = cache.total_length;

	out.coproduct_offsets.resize(cache.total_length + 1, 0);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		out.coproduct_offsets[flat] = out.coproduct_pairs.size();

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
			if (left_flat != 0 && right_flat != 0) {
				out.coproduct_pairs.push_back(left_flat);
				out.coproduct_pairs.push_back(right_flat);
			}
		}
	}
	out.coproduct_offsets[cache.total_length] = out.coproduct_pairs.size();
	return out;
}

inline std::vector<uint64_t> build_flat_to_product_(
	const BranchedSigCache& cache,
	const ProductMap& product_index
) {
	std::vector<uint64_t> flat_to_product(cache.total_length, 0);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat)
		flat_to_product[flat] = find_product_(product_index, { flat });
	return flat_to_product;
}

inline TreeCoproduct build_tree_product_coproducts_(
	const BranchedSigCache& cache,
	const ProductMap& product_index,
	const std::vector<uint64_t>& flat_to_product
) {
	TreeCoproduct tree_coproduct(cache.total_length);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		// Include tree tensor scalar and scalar tensor tree explicitly.
		tree_coproduct[flat].push_back({ flat_to_product[flat], 0 });
		tree_coproduct[flat].push_back({ 0, flat_to_product[flat] });

		const uint64_t tree_idx = flat - 1;
		uint64_t pos = cache.coproduct_offsets[tree_idx];
		const uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];
		while (pos < pos_end) {
			const uint64_t num_forest = cache.coproduct_data[pos++];
			const uint64_t trunk_flat = cache.coproduct_data[pos++];
			Product forest;
			forest.reserve(num_forest);
			for (uint64_t j = 0; j < num_forest; ++j)
				forest.push_back(cache.coproduct_data[pos++]);
			std::sort(forest.begin(), forest.end());
			tree_coproduct[flat].push_back({
				find_product_(product_index, forest), flat_to_product[trunk_flat] });
		}
	}
	return tree_coproduct;
}

inline uint64_t combine_products_(
	const std::vector<Product>& products,
	const ProductMap& product_index,
	uint64_t left,
	uint64_t right
) {
	Product combined;
	combined.reserve(products[left].size() + products[right].size());
	combined.insert(combined.end(), products[left].begin(), products[left].end());
	combined.insert(combined.end(), products[right].begin(), products[right].end());
	std::sort(combined.begin(), combined.end());
	return find_product_(product_index, combined);
}

inline std::vector<std::vector<ProductPair>> expand_product_coproducts_(
	const std::vector<Product>& products,
	const ProductMap& product_index,
	const TreeCoproduct& tree_coproduct
) {
	std::vector<std::vector<ProductPair>> coproducts(products.size());
	for (uint64_t product = 0; product < products.size(); ++product) {
		// The coproduct is multiplicative over the factors of a product.
		std::vector<ProductPair> terms{ { 0, 0 } };
		for (uint64_t flat : products[product]) {
			std::vector<ProductPair> next_terms;
			next_terms.reserve(terms.size() * tree_coproduct[flat].size());
			for (const ProductPair& term : terms) {
				for (const ProductPair& tree_term : tree_coproduct[flat]) {
					next_terms.push_back({
						combine_products_(products, product_index, term.first, tree_term.first),
						combine_products_(products, product_index, term.second, tree_term.second)
					});
				}
			}
			terms.swap(next_terms);
		}
		coproducts[product] = std::move(terms);
	}
	return coproducts;
}

inline BranchedLogHornerPlan build_nonplanar_horner_plan_(
	const BranchedSigCache& cache
) {
	ProductMap product_index;
	const std::vector<Product> products = enumerate_nonplanar_products_(
		cache, product_index);
	BranchedLogHornerPlan out;
	out.product_count = products.size();
	out.flat_to_product = build_flat_to_product_(cache, product_index);
	out.cpu_products.parent.resize(products.size(), 0);
	out.cpu_products.last_factor.resize(products.size(), 0);
	for (uint64_t product = 1; product < products.size(); ++product) {
		Product prefix = products[product];
		out.cpu_products.last_factor[product] = prefix.back();
		prefix.pop_back();
		out.cpu_products.parent[product] = find_product_(product_index, prefix);
	}
	const TreeCoproduct tree_coproduct = build_tree_product_coproducts_(
		cache, product_index, out.flat_to_product);
	const auto coproducts = expand_product_coproducts_(
		products, product_index, tree_coproduct);

	// Flatten temporary products and coproducts into the kernel CSR layout.
	out.cuda_products.offsets.resize(products.size() + 1, 0);
	out.coproduct_offsets.resize(products.size() + 1, 0);
	for (uint64_t product = 0; product < products.size(); ++product) {
		out.cuda_products.offsets[product] = out.cuda_products.factors.size();
		out.cuda_products.factors.insert(
			out.cuda_products.factors.end(),
			products[product].begin(), products[product].end());
		out.coproduct_offsets[product] = out.coproduct_pairs.size();
		for (const ProductPair& term : coproducts[product]) {
			if (term.first == 0 || term.second == 0)
				continue;
			out.coproduct_pairs.push_back(term.first);
			out.coproduct_pairs.push_back(term.second);
		}
	}
	out.cuda_products.offsets.back() = out.cuda_products.factors.size();
	out.coproduct_offsets.back() = out.coproduct_pairs.size();
	return out;
}

} // namespace branched_log_detail

inline BranchedLogHornerPlan build_branched_log_horner_plan(
	const BranchedSigCache& cache
) {
	if (cache.planar)
		return branched_log_detail::build_planar_horner_plan_(cache);
	return branched_log_detail::build_nonplanar_horner_plan_(cache);
}
