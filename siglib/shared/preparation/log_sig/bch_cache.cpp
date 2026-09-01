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
#ifdef PYSIGLIB_USE_UPSTREAM_BCH
#include "bch_formula.h"
#endif
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
}
}  // namespace

void build_commutator_views(BchCache& cache) {
	const uint64_t m = cache.m;
	std::vector<uint32_t> counts(m, 0);
	for (uint64_t i = 0; i < m; ++i)
		for (uint64_t j = i + 1; j < m; ++j)
			for (const auto& [k, coefficient] : cache.commutator_table[i * m + j])
				++counts[k];
	cache.comm_k_ptr.assign(m + 1, 0);
	for (uint64_t k = 0; k < m; ++k)
		cache.comm_k_ptr[k + 1] = cache.comm_k_ptr[k] + counts[k];
	const uint32_t nnz = cache.comm_k_ptr[m];
	cache.comm_k_i.resize(nnz);
	cache.comm_k_j.resize(nnz);
	cache.comm_k_val.resize(nnz);
	std::fill(counts.begin(), counts.end(), 0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			for (const auto& [k, coefficient] : cache.commutator_table[i * m + j]) {
				const uint32_t position = cache.comm_k_ptr[k] + counts[k]++;
				cache.comm_k_i[position] = static_cast<uint32_t>(i);
				cache.comm_k_j[position] = static_cast<uint32_t>(j);
				cache.comm_k_val[position] = coefficient;
			}
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
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			for (const auto& entry : cache.commutator_table[i * m + j]) {
				++input_counts[i];
				++input_counts[j];
			}
		}
	}
	cache.comm_a_ptr.assign(m + 1, 0);
	for (uint64_t index = 0; index < m; ++index)
		cache.comm_a_ptr[index + 1] = cache.comm_a_ptr[index] + input_counts[index];
	const uint32_t input_nnz = cache.comm_a_ptr[m];
	cache.comm_a_k.resize(input_nnz);
	cache.comm_a_partner.resize(input_nnz);
	cache.comm_a_signed_c.resize(input_nnz);
	std::fill(input_counts.begin(), input_counts.end(), 0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			for (const auto& [k, coefficient] : cache.commutator_table[i * m + j]) {
				const uint32_t left = cache.comm_a_ptr[i] + input_counts[i]++;
				cache.comm_a_k[left] = static_cast<uint32_t>(k);
				cache.comm_a_partner[left] = static_cast<uint32_t>(j);
				cache.comm_a_signed_c[left] = coefficient;
				const uint32_t right = cache.comm_a_ptr[j] + input_counts[j]++;
				cache.comm_a_k[right] = static_cast<uint32_t>(k);
				cache.comm_a_partner[right] = static_cast<uint32_t>(i);
				cache.comm_a_signed_c[right] = -coefficient;
			}
		}
	}
	cache.n_pairs = 0;
	for (uint64_t i = 0; i < m; ++i)
		for (uint64_t j = i + 1; j < m; ++j)
			cache.n_pairs += !cache.commutator_table[i * m + j].empty();
	cache.comm_ij_i.resize(cache.n_pairs);
	cache.comm_ij_j.resize(cache.n_pairs);
	cache.comm_ij_ptr.assign(cache.n_pairs + 1, 0);
	cache.comm_ij_k.resize(nnz);
	cache.comm_ij_c.resize(nnz);
	uint32_t pair_index = 0;
	uint32_t entry_index = 0;
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			const SparseVec& entries = cache.commutator_table[i * m + j];
			if (entries.empty())
				continue;
			cache.comm_ij_i[pair_index] = static_cast<uint32_t>(i);
			cache.comm_ij_j[pair_index] = static_cast<uint32_t>(j);
			for (const auto& [k, coefficient] : entries) {
				cache.comm_ij_k[entry_index] = static_cast<uint32_t>(k);
				cache.comm_ij_c[entry_index++] = static_cast<double>(coefficient);
			}
			cache.comm_ij_ptr[++pair_index] = entry_index;
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
	cache.commutator_table.assign(cache.m * cache.m, {});
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
			SparseVec& output = cache.commutator_table[i * cache.m + j];
			for (uint64_t k = 0; k < cache.m; ++k) {
				const int coefficient = static_cast<int>(std::round(coordinates[k]));
				if (coefficient != 0)
					output.push_back({ k, coefficient });
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
#ifdef PYSIGLIB_USE_UPSTREAM_BCH
	LyndonBchFormula formula = compute_lyndon_bch_formula(cache.degree);
	cache.bch_coefficients = std::move(formula.coefficients);
	cache.bch_left_factor = std::move(formula.left_factor);
	cache.bch_right_factor = std::move(formula.right_factor);
#else
	const uint64_t degree = cache.degree;
	std::vector<uint64_t> offsets(degree + 2, 0);
	for (uint64_t level = 0; level <= degree; ++level)
		offsets[level + 1] = offsets[level] + tensor_power(2, level);
	std::vector<double> signature(offsets[degree + 1], 0.0);
	signature[0] = 1.0;
	std::vector<double> inverse_factorial(degree + 1, 1.0);
	for (uint64_t level = 1; level <= degree; ++level)
		inverse_factorial[level] = inverse_factorial[level - 1]
			/ static_cast<double>(level);
	for (uint64_t level = 1; level <= degree; ++level) {
		for (uint64_t left_degree = 0; left_degree <= level; ++left_degree) {
			const uint64_t right_degree = level - left_degree;
			const uint64_t word_index = tensor_power(2, right_degree) - 1;
			signature[offsets[level] + word_index]
				= inverse_factorial[left_degree]
				* inverse_factorial[right_degree];
		}
	}

	std::vector<double> x = signature;
	x[0] = 0.0;
	std::vector<double> power = x;
	std::vector<double> tensor_log(signature.size(), 0.0);
	for (uint64_t term = 1; term <= degree; ++term) {
		const double coefficient = (term % 2 == 0 ? -1.0 : 1.0)
			/ static_cast<double>(term);
		for (uint64_t index = 1; index < tensor_log.size(); ++index)
			tensor_log[index] += coefficient * power[index];
		if (term == degree)
			break;
		std::vector<double> next(signature.size(), 0.0);
		for (uint64_t level = 2; level <= degree; ++level) {
			for (uint64_t left_degree = 1; left_degree < level; ++left_degree) {
				const uint64_t right_degree = level - left_degree;
				const uint64_t left_size = tensor_power(2, left_degree);
				const uint64_t right_size = tensor_power(2, right_degree);
				for (uint64_t left = 0; left < left_size; ++left) {
					const double left_value = power[offsets[left_degree] + left];
					if (left_value == 0.0)
						continue;
					for (uint64_t right = 0; right < right_size; ++right) {
						next[offsets[level] + left * right_size + right]
							+= left_value * x[offsets[right_degree] + right];
					}
				}
			}
		}
		power = std::move(next);
	}

	LogSigCache basis_cache(2, degree, 2);
	const BasisCache& basis = basis_cache.basis(2);
	cache.bch_coefficients.resize(basis.lyndon_idx.size());
	for (uint64_t i = 0; i < basis.lyndon_idx.size(); ++i)
		cache.bch_coefficients[i] = tensor_log[basis.lyndon_idx[i]];
	if (!cache.bch_coefficients.empty())
		basis.inv_proj_mat.mul_vec_inplace_lower(cache.bch_coefficients.data());
	compute_factorization_indices(
		2, degree, cache.bch_left_factor, cache.bch_right_factor);
#endif
}

void build_live_bch_nodes(BchCache& cache) {
	const uint64_t formula_size = cache.bch_coefficients.size();
	std::vector<uint8_t> live(formula_size, 0);
	// Hardcoded and upstream coefficients preserve exact rational zeroes. The
	// legacy floating-point generator does not, so keep its nodes conservatively.
	bool exact_zero_metadata = cache.degree <= BCH_MAX_HARDCODED_DEGREE;
#ifdef PYSIGLIB_USE_UPSTREAM_BCH
	exact_zero_metadata = true;
#endif
	for (uint64_t node = 2; node < formula_size; ++node)
		live[node] = !exact_zero_metadata
			|| cache.bch_coefficients[node] != 0.0;

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
