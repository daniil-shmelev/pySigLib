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

#include "bch_cache.h"

#include "bch_data.h"
#include "tensor_basis.h"
#include "lyndon_words.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

void remove_zero_entries(TensorElem& element) {
	for (auto item = element.begin(); item != element.end();) {
		if (item->second == 0)
			item = element.erase(item);
		else
			++item;
	}
}

TensorElem tensor_product(
	const TensorElem& left,
	uint64_t left_degree,
	const TensorElem& right,
	uint64_t right_degree,
	uint64_t dimension,
	uint64_t max_degree
) {
	if (left_degree + right_degree > max_degree)
		return {};
	TensorElem result;
	for (const auto& [left_index, left_coefficient] : left) {
		for (const auto& [right_index, right_coefficient] : right) {
			const uint64_t index = concatenate_idx(
				left_index, right_index, right_degree, dimension);
			result[index] += left_coefficient * right_coefficient;
		}
	}
	remove_zero_entries(result);
	return result;
}

std::vector<word> compute_factorization_indices(
	uint64_t dimension,
	uint64_t degree,
	std::vector<uint64_t>& left_factor,
	std::vector<uint64_t>& right_factor
) {
	std::vector<word> words = all_lyndon_words(dimension, degree);
	const std::unordered_set<word, WordHash> lyndon_set(
		words.begin(), words.end());
	std::unordered_map<word, uint64_t, WordHash> indices;
	indices.reserve(words.size());
	for (uint64_t index = 0; index < words.size(); ++index)
		indices[words[index]] = index;
	left_factor.assign(words.size(), UINT64_MAX);
	right_factor.assign(words.size(), UINT64_MAX);
	for (uint64_t index = 0; index < words.size(); ++index) {
		if (words[index].size() <= 1)
			continue;
		const auto [left, right] = standard_factorization(
			words[index], lyndon_set);
		left_factor[index] = indices.at(left);
		right_factor[index] = indices.at(right);
	}
	return words;
}

namespace {
void build_tensor_representations_(
	const std::vector<word>& words,
	const std::vector<uint64_t>& left_factor,
	const std::vector<uint64_t>& right_factor,
	uint64_t dimension,
	uint64_t degree,
	std::vector<TensorElem>& representations,
	std::vector<uint64_t>& weights
) {
	representations.resize(words.size());
	weights.resize(words.size());
	for (uint64_t index = 0; index < words.size(); ++index) {
		weights[index] = words[index].size();
		if (weights[index] == 1) {
			representations[index] = { { words[index][0] + 1, 1 } };
			continue;
		}
		TensorElem value = tensor_product(
			representations[left_factor[index]], weights[left_factor[index]],
			representations[right_factor[index]], weights[right_factor[index]],
			dimension, degree);
		const TensorElem reverse = tensor_product(
			representations[right_factor[index]], weights[right_factor[index]],
			representations[left_factor[index]], weights[left_factor[index]],
			dimension, degree);
		for (const auto& [flat, coefficient] : reverse)
			value[flat] -= coefficient;
		remove_zero_entries(value);
		representations[index] = std::move(value);
	}
}

void build_linear_bch_ranges_(BchCache& cache) {
	const uint64_t formula_size = cache.bch_coefficients.size();
	if (cache.coordinate_weights.size() != cache.m
		|| cache.linear_input_mask.size() != cache.m)
		throw std::runtime_error("BCH cache support data has an invalid size");
	std::vector<uint64_t> min_degree(formula_size, 1);
	std::vector<uint64_t> max_degree(formula_size, cache.degree);
	if (formula_size > 1) {
		min_degree[1] = cache.degree + 1;
		max_degree[1] = 0;
		for (uint32_t index : cache.linear_input_idx) {
			if (index >= cache.m)
				throw std::out_of_range("BCH linear input index out of range");
			min_degree[1] = (std::min)(
				min_degree[1], cache.coordinate_weights[index]);
			max_degree[1] = (std::max)(
				max_degree[1], cache.coordinate_weights[index]);
		}
		if (cache.linear_input_idx.empty())
			min_degree[1] = 1;
	}
	for (uint64_t node = 2; node < formula_size; ++node) {
		const uint64_t left = cache.bch_left_factor[node];
		const uint64_t right = cache.bch_right_factor[node];
		min_degree[node] = min_degree[left] + min_degree[right];
		max_degree[node] = (std::min)(
			cache.degree, max_degree[left] + max_degree[right]);
	}
	cache.linear_range.resize(formula_size);
	for (uint64_t node = 0; node < formula_size; ++node) {
		const auto begin = std::lower_bound(
			cache.coordinate_weights.begin(), cache.coordinate_weights.end(),
			min_degree[node]);
		const auto end = std::upper_bound(
			cache.coordinate_weights.begin(), cache.coordinate_weights.end(),
			max_degree[node]);
		cache.linear_range[node] = {
			static_cast<uint64_t>(begin - cache.coordinate_weights.begin()),
			static_cast<uint64_t>(end - cache.coordinate_weights.begin()) };
	}
	cache.linear_pair_ptr.assign(formula_size + 1, 0);
	cache.linear_pair_idx.clear();
	for (uint64_t node = 2; node < formula_size; ++node) {
		const uint64_t left = cache.bch_left_factor[node];
		const uint64_t right = cache.bch_right_factor[node];
		const auto [left_begin, left_end] = cache.linear_range[left];
		const auto [right_begin, right_end] = cache.linear_range[right];
		for (uint32_t pair = 0; pair < cache.n_pairs; ++pair) {
			const uint32_t i = cache.comm_ij_i[pair];
			const uint32_t j = cache.comm_ij_j[pair];
			const bool i_left = left == 1 ? cache.linear_input_mask[i]
				: i >= left_begin && i < left_end;
			const bool j_left = left == 1 ? cache.linear_input_mask[j]
				: j >= left_begin && j < left_end;
			const bool i_right = right == 1 ? cache.linear_input_mask[i]
				: i >= right_begin && i < right_end;
			const bool j_right = right == 1 ? cache.linear_input_mask[j]
				: j >= right_begin && j < right_end;
			if ((i_left && j_right) || (j_left && i_right))
				cache.linear_pair_idx.push_back(pair);
		}
		cache.linear_pair_ptr[node + 1] = cache.linear_pair_idx.size();
	}
	const uint64_t dense_count = formula_size > 2
		? (formula_size - 2) * cache.n_pairs : 0;
	cache.prune_linear_backprop = dense_count > 0
		&& cache.linear_pair_idx.size() <= dense_count - dense_count / 3;
	cache.linear_forward.clear();
	if (cache.degree < 5 || !cache.linear_input_prefix)
		return;
	std::vector<std::vector<uint8_t>> support(formula_size, std::vector<uint8_t>(cache.m));
	std::fill(support[0].begin(), support[0].end(), 1);
	for (uint32_t index : cache.linear_input_idx) support[1][index] = 1;
	cache.linear_forward.resize(formula_size);
	uint64_t full_entries = 0, active_entries = 0;
	for (uint64_t node = 2; node < formula_size; ++node) {
		const uint64_t left = cache.bch_left_factor[node], right = cache.bch_right_factor[node];
		const auto [begin, end] = cache.linear_range[node];
		auto& plan = cache.linear_forward[node];
		plan.row_ptr.reserve(end - begin + 1);
		plan.row_ptr.push_back(0);
		for (uint64_t k = begin; k < end; ++k) {
			const uint32_t stop = left == 1 || right == 1
				? cache.comm_k_sparse_end[k] : cache.comm_k_ptr[k + 1];
			full_entries += stop - cache.comm_k_ptr[k];
			for (uint32_t q = cache.comm_k_ptr[k]; q < stop; ++q) {
				const uint32_t i = cache.comm_k_i[q], j = cache.comm_k_j[q];
				const uint8_t orientation = (support[left][i] && support[right][j] ? 1 : 0)
					| (support[left][j] && support[right][i] ? 2 : 0);
				if (orientation) {
					plan.entries.push_back({ i, j, cache.comm_k_val[q], orientation });
					support[node][k] = 1;
				}
			}
			plan.row_ptr.push_back(static_cast<uint32_t>(plan.entries.size()));
		}
		active_entries += plan.entries.size();
		plan.entries.shrink_to_fit();
	}
	if (active_entries > full_entries - full_entries / 5)
		std::vector<BchForwardNode>().swap(cache.linear_forward);
}
}  // namespace

void build_commutator_views(BchCache& cache) {
	const uint64_t m = cache.m;
	if (!cache.commutator_table.empty()) {
		cache.comm_ij_i.clear();
		cache.comm_ij_j.clear();
		cache.comm_ij_ptr.assign(1, 0);
		cache.comm_ij_k.clear();
		cache.comm_ij_c.clear();
		for (uint64_t i = 0; i < m; ++i) {
			for (uint64_t j = i + 1; j < m; ++j) {
				const auto& entries = cache.commutator_table[i * m + j];
				if (entries.empty()) continue;
				cache.comm_ij_i.push_back(static_cast<uint32_t>(i));
				cache.comm_ij_j.push_back(static_cast<uint32_t>(j));
				for (const auto& [k, coefficient] : entries) {
					cache.comm_ij_k.push_back(static_cast<uint32_t>(k));
					cache.comm_ij_c.push_back(static_cast<double>(coefficient));
				}
				if (cache.comm_ij_k.size() > UINT32_MAX)
					throw std::overflow_error("BCH commutator table is too large");
				cache.comm_ij_ptr.push_back(static_cast<uint32_t>(cache.comm_ij_k.size()));
			}
		}
		std::vector<SparseVec>().swap(cache.commutator_table);
	}
	if (cache.comm_ij_i.size() > UINT32_MAX || cache.comm_ij_k.size() > UINT32_MAX / 2)
		throw std::overflow_error("BCH commutator table is too large");
	cache.n_pairs = static_cast<uint32_t>(cache.comm_ij_i.size());
	std::vector<uint32_t> counts(m, 0);
	for (uint32_t k : cache.comm_ij_k)
		++counts[k];
	cache.comm_k_ptr.assign(m + 1, 0);
	for (uint64_t k = 0; k < m; ++k)
		cache.comm_k_ptr[k + 1] = cache.comm_k_ptr[k] + counts[k];
	const uint32_t nnz = cache.comm_k_ptr[m];
	cache.comm_k_i.resize(nnz);
	cache.comm_k_j.resize(nnz);
	cache.comm_k_val.resize(nnz);
	std::fill(counts.begin(), counts.end(), 0);
	for (uint32_t pair = 0; pair < cache.n_pairs; ++pair) {
		const uint32_t i = cache.comm_ij_i[pair], j = cache.comm_ij_j[pair];
		for (uint32_t index = cache.comm_ij_ptr[pair]; index < cache.comm_ij_ptr[pair + 1]; ++index) {
			const uint32_t k = cache.comm_ij_k[index];
			const int coefficient = static_cast<int>(cache.comm_ij_c[index]);
			const uint32_t position = cache.comm_k_ptr[k] + counts[k]++;
			cache.comm_k_i[position] = i;
			cache.comm_k_j[position] = j;
			cache.comm_k_val[position] = coefficient;
		}
	}
	cache.comm_k_val_d.resize(nnz);
	for (uint32_t index = 0; index < nnz; ++index)
		cache.comm_k_val_d[index] = static_cast<double>(cache.comm_k_val[index]);
	cache.comm_k_sparse_end.resize(m);
	for (uint64_t k = 0; k < m; ++k) {
		uint32_t end = cache.comm_k_ptr[k];
		for (uint32_t index = cache.comm_k_ptr[k];
			index < cache.comm_k_ptr[k + 1]; ++index) {
			if (cache.comm_k_i[index] >= cache.dimension)
				break;
			end = index + 1;
		}
		cache.comm_k_sparse_end[k] = end;
	}
	std::vector<uint32_t> input_counts(m, 0);
	for (uint32_t pair = 0; pair < cache.n_pairs; ++pair) {
		const uint32_t size = cache.comm_ij_ptr[pair + 1] - cache.comm_ij_ptr[pair];
		input_counts[cache.comm_ij_i[pair]] += size;
		input_counts[cache.comm_ij_j[pair]] += size;
	}
	cache.comm_a_ptr.assign(m + 1, 0);
	for (uint64_t index = 0; index < m; ++index)
		cache.comm_a_ptr[index + 1] = cache.comm_a_ptr[index] + input_counts[index];
	const uint32_t input_nnz = cache.comm_a_ptr[m];
	cache.comm_a_k.resize(input_nnz);
	cache.comm_a_partner.resize(input_nnz);
	cache.comm_a_signed_c.resize(input_nnz);
	std::fill(input_counts.begin(), input_counts.end(), 0);
	for (uint32_t pair = 0; pair < cache.n_pairs; ++pair) {
		const uint32_t i = cache.comm_ij_i[pair], j = cache.comm_ij_j[pair];
		for (uint32_t index = cache.comm_ij_ptr[pair]; index < cache.comm_ij_ptr[pair + 1]; ++index) {
			const uint32_t k = cache.comm_ij_k[index];
			const int coefficient = static_cast<int>(cache.comm_ij_c[index]);
			const uint32_t left = cache.comm_a_ptr[i] + input_counts[i]++;
			cache.comm_a_k[left] = k;
			cache.comm_a_partner[left] = j;
			cache.comm_a_signed_c[left] = coefficient;
			const uint32_t right = cache.comm_a_ptr[j] + input_counts[j]++;
			cache.comm_a_k[right] = k;
			cache.comm_a_partner[right] = i;
			cache.comm_a_signed_c[right] = -coefficient;
		}
	}
}

void build_standard_commutator_table(
	BchCache& cache,
	const BasisCache& basis
) {
	std::vector<word> words = compute_factorization_indices(
		cache.dimension, cache.degree, cache.left_factor, cache.right_factor);
	cache.m = words.size();
	if (basis.lyndon_idx.size() != cache.m)
		throw std::runtime_error("BCH basis length mismatch");
	std::vector<TensorElem> representations;
	build_tensor_representations_(
		words, cache.left_factor, cache.right_factor,
		cache.dimension, cache.degree,
		representations, cache.coordinate_weights);
	cache.comm_ij_ptr.assign(1, 0);
	std::vector<double> coordinates(cache.m, 0.0);
	for (uint64_t i = 0; i < cache.m; ++i) {
		for (uint64_t j = i + 1; j < cache.m; ++j) {
			const uint64_t weight = cache.coordinate_weights[i]
				+ cache.coordinate_weights[j];
			if (weight > cache.degree)
				continue;
			TensorElem value = tensor_product(
				representations[i], cache.coordinate_weights[i],
				representations[j], cache.coordinate_weights[j],
				cache.dimension, cache.degree);
			const TensorElem reverse = tensor_product(
				representations[j], cache.coordinate_weights[j],
				representations[i], cache.coordinate_weights[i],
				cache.dimension, cache.degree);
			for (const auto& [flat, coefficient] : reverse)
				value[flat] -= coefficient;
			std::fill(coordinates.begin(), coordinates.end(), 0.0);
			for (uint64_t k = 0; k < cache.m; ++k) {
				if (cache.coordinate_weights[k] != weight)
					continue;
				const auto found = value.find(basis.lyndon_idx[k]);
				if (found != value.end())
					coordinates[k] = static_cast<double>(found->second);
			}
			basis.inv_proj_mat.mul_vec_inplace_lower(coordinates.data());
			const auto start = cache.comm_ij_k.size();
			for (uint64_t k = 0; k < cache.m; ++k) {
				const int coefficient = static_cast<int>(std::round(coordinates[k]));
				if (coefficient != 0) {
					cache.comm_ij_k.push_back(static_cast<uint32_t>(k));
					cache.comm_ij_c.push_back(static_cast<double>(coefficient));
				}
			}
			if (cache.comm_ij_k.size() != start) {
				if (cache.comm_ij_k.size() > UINT32_MAX)
					throw std::overflow_error("BCH commutator table is too large");
				cache.comm_ij_i.push_back(static_cast<uint32_t>(i));
				cache.comm_ij_j.push_back(static_cast<uint32_t>(j));
				cache.comm_ij_ptr.push_back(static_cast<uint32_t>(cache.comm_ij_k.size()));
			}
		}
	}
	build_commutator_views(cache);
}

void configure_linear_bch_input(
	BchCache& cache,
	std::vector<uint32_t> input_indices,
	bool prefix
) {
	cache.linear_input_idx = std::move(input_indices);
	cache.linear_input_prefix = prefix;
	cache.linear_input_mask.assign(cache.m, 0);
	for (uint32_t index : cache.linear_input_idx) {
		if (index >= cache.m)
			throw std::out_of_range("BCH linear input index out of range");
		cache.linear_input_mask[index] = 1;
	}
	cache.comm_k_sparse_ptr.assign(cache.m + 1, 0);
	cache.comm_k_sparse_idx.clear();
	for (uint64_t k = 0; k < cache.m; ++k) {
		for (uint32_t index = cache.comm_k_ptr[k];
			index < cache.comm_k_ptr[k + 1]; ++index) {
			if (cache.linear_input_mask[cache.comm_k_i[index]]
				|| cache.linear_input_mask[cache.comm_k_j[index]])
				cache.comm_k_sparse_idx.push_back(index);
		}
		cache.comm_k_sparse_ptr[k + 1] = cache.comm_k_sparse_idx.size();
	}
	build_linear_bch_ranges_(cache);
}

bool load_hardcoded_bch_formula(BchCache& cache) {
	const BchHardcodedData* data = get_hardcoded_bch_data(cache.degree);
	if (data == nullptr)
		return false;
	cache.bch_coefficients.assign(
		data->coefficients, data->coefficients + data->size);
	cache.bch_left_factor.assign(
		data->left_factor, data->left_factor + data->size);
	cache.bch_right_factor.assign(
		data->right_factor, data->right_factor + data->size);
	return true;
}

void build_bch_formula_data(BchCache& cache) {
	if (load_hardcoded_bch_formula(cache))
		return;
	throw std::invalid_argument(
		"BCH methods support truncation degrees at most 20");
}

void build_live_bch_nodes(BchCache& cache) {
	const uint64_t formula_size = cache.bch_coefficients.size();
	std::vector<uint8_t> live(formula_size, 0);
	for (uint64_t node = 2; node < formula_size; ++node)
		live[node] = cache.bch_coefficients[node] != 0.0;

	for (uint64_t node = formula_size; node-- > 2;) {
		const uint64_t left = cache.bch_left_factor[node];
		const uint64_t right = cache.bch_right_factor[node];
		if (!live[node])
			continue;
		if (left >= 2)
			live[left] = 1;
		if (right >= 2)
			live[right] = 1;
	}

	cache.live_bch_nodes.clear();
	cache.live_bch_nodes.reserve(formula_size > 2 ? formula_size - 2 : 0);
	for (uint32_t node = 2; node < formula_size; ++node) {
		if (live[node])
			cache.live_bch_nodes.push_back(node);
	}
	cache.all_bch_nodes_live = cache.live_bch_nodes.size()
		== (formula_size > 2 ? formula_size - 2 : 0);
}

std::unique_ptr<BchCache> make_standard_bch_cache(
	uint64_t dimension,
	uint64_t degree,
	const BasisCache& basis
) {
	auto cache = std::make_unique<BchCache>();
	cache->dimension = dimension;
	cache->degree = degree;
	build_bch_formula_data(*cache);
	build_live_bch_nodes(*cache);
	build_standard_commutator_table(*cache, basis);
	if (dimension > UINT32_MAX)
		throw std::overflow_error("BCH linear input is too large");
	const uint64_t input_size = (std::min)(dimension, cache->m);
	std::vector<uint32_t> input_indices(input_size);
	for (uint32_t index = 0; index < input_size; ++index)
		input_indices[index] = index;
	configure_linear_bch_input(*cache, std::move(input_indices), true);
	return cache;
}
