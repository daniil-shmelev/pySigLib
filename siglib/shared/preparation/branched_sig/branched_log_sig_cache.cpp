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

#include "branched_log_sig_cache.h"

#include "../log_sig/bch_data.h"
#include "../log_sig/lyndon_words.h"
#include "../../branched_log_horner.h"
#include "../../errors.h"
#include "../../trees/basis_counts.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>


namespace {
constexpr const char* mkw_basis_cache_prefix_ = "mkw_lyndon_";

struct MkwWordData_ {
	// Every flat MKW forest reconstructed as a word of tree IDs.
	std::vector<word> flat_words;
	std::unordered_map<word, uint64_t, WordHash> flat_idx;
	// Lyndon words identify compact coordinates in methods 1 and 2.
	std::vector<word> lyndon_words;
	std::vector<uint64_t> lyndon_idx;
	std::vector<uint64_t> lyndon_weights;
};


word mkw_forest_word_(const BranchedSigCache& cache, uint64_t basis_idx) {
	const uint64_t start = cache.basis_forest_offsets[basis_idx];
	const uint64_t end = cache.basis_forest_offsets[basis_idx + 1];
	return word(
		cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(start),
		cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(end));
}


MkwWordData_ build_mkw_word_data_(const BranchedSigCache& cache) {
	MkwWordData_ data;
	// Treat each decorated planar tree as a letter in the forest-word alphabet.
	// cache.basis_forest_data holds these words in the same order as the output.
	const uint64_t lyndon_count = compute_branched_log_sig_length(
		cache.dimension, cache.max_nodes, true);
	data.flat_words.resize(cache.total_length);
	data.flat_idx.reserve(cache.total_length);
	data.lyndon_words.reserve(lyndon_count);
	data.lyndon_idx.reserve(lyndon_count);
	data.lyndon_weights.reserve(lyndon_count);
	for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
		word forest = mkw_forest_word_(cache, basis_idx);
		data.flat_words[basis_idx + 1] = forest;
		if (is_lyndon(forest)) {
			data.lyndon_words.push_back(forest);
			data.lyndon_idx.push_back(basis_idx + 1);
			data.lyndon_weights.push_back(
				cache.node_labels_offsets[basis_idx + 1]
				- cache.node_labels_offsets[basis_idx]);
		}
	}
	if (data.lyndon_idx.size() != lyndon_count)
		throw std::runtime_error("MKW Lyndon cache length mismatch");
	for (uint64_t i = 0; i < data.flat_words.size(); ++i)
		data.flat_idx[data.flat_words[i]] = i;
	return data;
}

void build_mkw_projection_(
	uint64_t total_length,
	const MkwWordData_& word_data,
	BasisCache& basis
) {
	if (basis.lyndon_idx != word_data.lyndon_idx)
		throw std::runtime_error("MKW Lyndon cache index mismatch");
	SparseIntMatrix projection;
	lyndon_proj_matrix_from_words(
		projection,
		word_data.lyndon_words,
		total_length,
		[&word_data](const word& value) {
			return word_data.flat_idx.at(value);
		},
		[&word_data](uint64_t i, uint64_t j, uint64_t) {
			return word_data.flat_idx.at(concatenate_words(
				word_data.flat_words.at(i), word_data.flat_words.at(j)));
		});
	projection.inverse(basis.inv_proj_mat);
	basis.inv_proj_mat.transpose(basis.inv_proj_mat_transpose);
	basis.method = 2;
}


std::vector<uint64_t> build_mkw_lyndon_indices_(
	const BranchedSigCache& cache
) {
	std::vector<uint64_t> lyndon_idx;
	const uint64_t lyndon_count = compute_branched_log_sig_length(
		cache.dimension, cache.max_nodes, true);
	lyndon_idx.reserve(lyndon_count);
	for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
		if (is_lyndon(mkw_forest_word_(cache, basis_idx)))
			lyndon_idx.push_back(basis_idx + 1);
	}
	if (lyndon_idx.size() != lyndon_count)
		throw std::runtime_error("MKW Lyndon cache length mismatch");
	return lyndon_idx;
}


BasisCache build_branched_log_basis_cache_(
	const BranchedSigCache& cache,
	int method
) {
	if (!cache.planar)
		throw std::invalid_argument("compressed branched log signatures require planar=True");

	if (method == 2) {
		MkwWordData_ word_data = build_mkw_word_data_(cache);
		BasisCache result(
			method,
			word_data.lyndon_idx,
			{},
			{});
		build_mkw_projection_(cache.total_length, word_data, result);
		return result;
	}

	return BasisCache(
		method,
		build_mkw_lyndon_indices_(cache),
		SparseIntMatrix{},
		SparseIntMatrix{});
}


BasisCache load_or_build_branched_log_basis_cache_(
	const BranchedSigCache& cache,
	int method,
	const std::filesystem::path& cache_directory,
	bool use_disk
) {
	if (use_disk && !cache_directory.empty()) {
		// A method 2 disk entry is also valid for a method 1 request.
		BasisCache basis;
		if (read_log_sig_basis_cache(
			cache_directory,
			cache.dimension,
			cache.max_nodes,
			basis,
			mkw_basis_cache_prefix_)
			&& basis.method >= method)
			return basis;
	}

	auto basis = build_branched_log_basis_cache_(cache, method);
	if (use_disk && !cache_directory.empty()) {
		write_log_sig_basis_cache(
			cache_directory,
			cache.dimension,
			cache.max_nodes,
			basis,
			mkw_basis_cache_prefix_);
	}
	return basis;
}


TensorElem mkw_tensor_product_(
	const TensorElem& left,
	const TensorElem& right,
	const MkwWordData_& words
) {
	TensorElem result;
	for (const auto& [left_idx, left_coefficient] : left) {
		for (const auto& [right_idx, right_coefficient] : right) {
			const word product = concatenate_words(
				words.flat_words.at(left_idx), words.flat_words.at(right_idx));
			const auto flat = words.flat_idx.find(product);
			if (flat != words.flat_idx.end())
				result[flat->second] += left_coefficient * right_coefficient;
		}
	}
	remove_zero_entries(result);
	return result;
}


using MkwInfinitesimalProduct_ = std::unordered_map<
	std::pair<uint64_t, uint64_t>, TensorElem, PairHash>;


MkwInfinitesimalProduct_ build_mkw_infinitesimal_product_(
	const BranchedSigCache& cache
) {
	// Retain only one-branch cuts, which define the infinitesimal product.
	// The keys are the two input flat coordinates and the values are outputs.
	MkwInfinitesimalProduct_ product;
	for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
		uint64_t pos = cache.coproduct_offsets[basis_idx];
		const uint64_t end = cache.coproduct_offsets[basis_idx + 1];
		while (pos < end) {
			const uint64_t forest_size = cache.coproduct_data[pos++];
			const uint64_t trunk = cache.coproduct_data[pos++];
			if (forest_size == 1) {
				const uint64_t branch = cache.coproduct_data[pos++];
				product[{ branch, trunk }][basis_idx + 1] += 1;
			} else {
				pos += forest_size;
			}
		}
	}
	return product;
}


TensorElem mkw_infinitesimal_product_(
	const TensorElem& left,
	const TensorElem& right,
	const MkwInfinitesimalProduct_& product
) {
	TensorElem result;
	for (const auto& [left_idx, left_coefficient] : left) {
		for (const auto& [right_idx, right_coefficient] : right) {
			const auto found = product.find({ left_idx, right_idx });
			if (found == product.end())
				continue;
			for (const auto& [out_idx, coefficient] : found->second) {
				result[out_idx] += left_coefficient
					* right_coefficient * coefficient;
			}
		}
	}
	remove_zero_entries(result);
	return result;
}


}  // namespace


BranchedBchCache::BranchedBchCache(
	const BranchedSigCache& branched_cache,
	const BranchedLogHornerPlan& horner_plan,
	const BasisCache& basis
) {
	// BCH operates in method 2 coordinates, not in the expanded forest basis.
	MkwWordData_ words = build_mkw_word_data_(branched_cache);
	const uint64_t m = words.lyndon_words.size();
	if (m > UINT32_MAX)
		throw std::overflow_error("MKW BCH basis is too large");

	bch.dimension = branched_cache.dimension;
	bch.degree = branched_cache.max_nodes;
	bch.m = m;
	bch.coordinate_weights = words.lyndon_weights;
	linear_basis_idx.resize(m);
	for (uint64_t i = 0; i < m; ++i)
		linear_basis_idx[i] = basis.lyndon_idx[i] - 1;

	std::unordered_set<word, WordHash> lyndon_set(
		words.lyndon_words.begin(), words.lyndon_words.end());
	std::unordered_map<word, uint64_t, WordHash> lyndon_map;
	lyndon_map.reserve(m);
	for (uint64_t i = 0; i < m; ++i)
		lyndon_map[words.lyndon_words[i]] = i;
	bch.left_factor.assign(m, UINT64_MAX);
	bch.right_factor.assign(m, UINT64_MAX);
	for (uint64_t i = 0; i < m; ++i) {
		if (words.lyndon_words[i].size() <= 1)
			continue;
		auto [left, right] = standard_factorization(words.lyndon_words[i], lyndon_set);
		bch.left_factor[i] = lyndon_map.at(left);
		bch.right_factor[i] = lyndon_map.at(right);
	}

	std::vector<TensorElem> tensor_reps(m);
	// Expand standard Lyndon bracketings in the forest concatenation algebra.
	// Each representation is then used to derive the infinitesimal commutator.
	for (uint64_t i = 0; i < m; ++i) {
		if (words.lyndon_words[i].size() == 1) {
			tensor_reps[i] = { { words.flat_idx.at(words.lyndon_words[i]), 1 } };
			continue;
		}
		TensorElem left_right = mkw_tensor_product_(
			tensor_reps[bch.left_factor[i]], tensor_reps[bch.right_factor[i]], words);
		TensorElem right_left = mkw_tensor_product_(
			tensor_reps[bch.right_factor[i]], tensor_reps[bch.left_factor[i]], words);
		for (const auto& [index, coefficient] : right_left)
			left_right[index] -= coefficient;
		remove_zero_entries(left_right);
		tensor_reps[i] = std::move(left_right);
	}

	const MkwInfinitesimalProduct_ infinitesimal_product
		= build_mkw_infinitesimal_product_(branched_cache);
	bch.commutator_table.resize(m * m);
	std::vector<double> coordinates(m, 0.0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			const uint64_t weight = bch.coordinate_weights[i] + bch.coordinate_weights[j];
			if (weight > bch.degree)
				continue;
			TensorElem commutator = mkw_infinitesimal_product_(
				tensor_reps[i], tensor_reps[j], infinitesimal_product);
			TensorElem reverse = mkw_infinitesimal_product_(
				tensor_reps[j], tensor_reps[i], infinitesimal_product);
			for (const auto& [index, coefficient] : reverse)
				commutator[index] -= coefficient;
			std::fill(coordinates.begin(), coordinates.end(), 0.0);
			for (uint64_t k = 0; k < m; ++k) {
				if (bch.coordinate_weights[k] != weight)
					continue;
				const auto entry = commutator.find(basis.lyndon_idx[k]);
				if (entry != commutator.end())
					coordinates[k] = static_cast<double>(entry->second);
			}
			basis.inv_proj_mat.mul_vec_inplace_lower(coordinates.data());
			SparseVec& entry = bch.commutator_table[i * m + j];
			for (uint64_t k = 0; k < m; ++k) {
				const int coefficient = static_cast<int>(std::round(coordinates[k]));
				if (coefficient != 0)
					entry.push_back({ k, coefficient });
			}
		}
	}

	// Reuse the ordinary BCH formula and its coefficient-pruning plans. Only
	// the MKW commutator table and segment lift are specific to branched paths.
	build_commutator_views(bch);
	build_bch_formula_data(bch);
	build_live_bch_nodes(bch);

	std::vector<double> linear_sig(branched_cache.total_length);
	linear_coefficients.resize(m);
	linear_sig[0] = 1.0;
	for (uint64_t flat = 1; flat < branched_cache.total_length; ++flat)
		linear_sig[flat] = branched_cache.inv_tree_factorial[flat - 1];
	std::vector<double> expanded(branched_cache.total_length, 0.0);
	BranchedLogHornerWorkspace<double> workspace(horner_plan.product_count);
	branched_log_horner_forward<double>(
		branched_cache.total_length,
		branched_cache.max_nodes,
		true,
		horner_plan,
		[&linear_sig](uint64_t flat) { return linear_sig[flat]; },
		[&expanded](uint64_t flat, double value) { expanded[flat] = value; },
		workspace);
	for (uint64_t i = 0; i < m; ++i)
		linear_coefficients[i] = expanded[basis.lyndon_idx[i]];
	if (m != 0)
		basis.inv_proj_mat.mul_vec_inplace_lower(linear_coefficients.data());
	std::vector<uint32_t> linear_input_idx;
	linear_input_idx.reserve(m);
	for (uint64_t i = 0; i < m; ++i) {
		if (linear_coefficients[i] != 0.0)
			linear_input_idx.push_back(static_cast<uint32_t>(i));
	}
	configure_linear_bch_input(bch, std::move(linear_input_idx), false);
}


BranchedLogSigCache::BranchedLogSigCache(
	const BranchedSigCache& branched_cache,
	int method,
	const std::filesystem::path& cache_directory,
	bool use_disk
) : method_{ -1 } {
	upgrade(branched_cache, method, cache_directory, use_disk);
}


void BranchedLogSigCache::upgrade(
	const BranchedSigCache& branched_cache,
	int method,
	const std::filesystem::path& cache_directory,
	bool use_disk
) {
	if (method < 0 || method > 3)
		throw std::invalid_argument(
			"branched log signature method must be 0, 1, 2, or 3");
	if (method != 0 && !branched_cache.planar)
		throw std::invalid_argument(
			"compressed branched log signatures require planar=True");

	if (method_ < 0)
		horner_plan_ = build_branched_log_horner_plan(branched_cache);
	if (method >= 1) {
		const int basis_method = (std::min)(method, 2);
		if (!basis_cache_ || !basis_cache_->supports(basis_method)) {
			if (basis_cache_ && basis_cache_->method == 1
				&& basis_method == 2) {
				BasisCache loaded;
				if (use_disk && !cache_directory.empty()
					&& read_log_sig_basis_cache(
						cache_directory,
						branched_cache.dimension,
						branched_cache.max_nodes,
						loaded,
						mkw_basis_cache_prefix_)
					&& loaded.supports(2)) {
					basis_cache_ = std::move(loaded);
				}
				else {
					const MkwWordData_ word_data =
						build_mkw_word_data_(branched_cache);
					build_mkw_projection_(
						branched_cache.total_length, word_data, *basis_cache_);
					if (use_disk && !cache_directory.empty()) {
						write_log_sig_basis_cache(
							cache_directory,
							branched_cache.dimension,
							branched_cache.max_nodes,
							*basis_cache_,
							mkw_basis_cache_prefix_);
					}
				}
			}
			else {
				basis_cache_ = load_or_build_branched_log_basis_cache_(
					branched_cache, basis_method, cache_directory, use_disk);
			}
		}
	}
	if (method == 3 && !bch_cache_) {
		bch_cache_.emplace(
			branched_cache, horner_plan_, *basis_cache_);
	}
	method_ = (std::max)(method_, method);
}


bool BranchedLogSigCache::supports(int method) const noexcept {
	return method_ >= method;
}


const BranchedLogHornerPlan& BranchedLogSigCache::horner_plan() const noexcept {
	return horner_plan_;
}


const BasisCache& BranchedLogSigCache::basis_cache(int method) const {
	if (!basis_cache_ || basis_cache_->method < method)
		throw cache_not_found_error(
			"MKW branched log basis cache not found - call prepare_branched_log_sig first");
	return *basis_cache_;
}


const BranchedBchCache& BranchedLogSigCache::bch_cache() const {
	if (!bch_cache_)
		throw cache_not_found_error(
			"MKW BCH cache not found - call prepare_branched_log_sig with method=3 first");
	return *bch_cache_;
}
