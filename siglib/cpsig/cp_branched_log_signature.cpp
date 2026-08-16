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

#include "cppch.h"
#include "cpsig.h"
#include "cp_branched_log_signature.h"
#include "cp_branched_cache.h"
#include "cp_branched_signature.h"
#include "cp_bch.h"
#include "log_sig_cache.h"
#include "words.h"
#include "../shared/branched_log_cache.h"
#include "multithreading.h"
#include "macros.h"

namespace {
constexpr const char* mkw_basis_cache_prefix_ = "mkw_lyndon_";

// Method 2 data also provides the coordinate indices needed by method 1.
// The registry stores only the strongest prepared representation per key.
struct BranchedLogBasisCacheRegistry_ {
	std::unordered_map<
		std::pair<uint64_t, uint64_t>,
		std::unique_ptr<BasisCache>,
		PairHash
	> map;
	std::shared_mutex mu;
};

BranchedLogBasisCacheRegistry_& branched_log_basis_cache_registry_() {
	static BranchedLogBasisCacheRegistry_ registry;
	return registry;
}

struct MkwWordData_ {
	// Every flat MKW forest reconstructed as a word of tree IDs.
	std::vector<word> flat_words;
	std::unordered_map<word, uint64_t, WordHash> flat_idx;
	// Lyndon words identify compact coordinates in methods 1 and 2.
	std::vector<word> lyndon_words;
	std::vector<uint64_t> lyndon_idx;
	std::vector<uint64_t> lyndon_weights;
	// One-letter Lyndon words provide the path increment generators for BCH.
	std::vector<uint32_t> letter_log_idx;
	std::vector<uint64_t> letter_basis_idx;
};

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
		const uint64_t start = cache.basis_forest_offsets[basis_idx];
		const uint64_t end = cache.basis_forest_offsets[basis_idx + 1];
		word forest(
			cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(start),
			cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(end));
		data.flat_words[basis_idx + 1] = forest;
		if (is_lyndon(forest)) {
			if (forest.size() == 1) {
				data.letter_log_idx.push_back(
					static_cast<uint32_t>(data.lyndon_words.size()));
				data.letter_basis_idx.push_back(basis_idx);
			}
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

void clear_branched_log_basis_cache_() {
	auto& registry = branched_log_basis_cache_registry_();
	std::unique_lock lock(registry.mu);
	registry.map.clear();
}

std::unique_ptr<BasisCache> build_branched_log_basis_cache_(
	const BranchedSigCache& cache,
	int method
) {
	if (!cache.planar)
		throw std::invalid_argument("compressed branched log signatures require planar=True");

	std::vector<uint64_t> lyndon_idx;
	const uint64_t lyndon_count = compute_branched_log_sig_length(
		cache.dimension, cache.max_nodes, true);
	lyndon_idx.reserve(lyndon_count);
	MkwWordData_ word_data;
	if (method == 2)
		word_data = build_mkw_word_data_(cache);
	else {
		for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
			const uint64_t start = cache.basis_forest_offsets[basis_idx];
			const uint64_t end = cache.basis_forest_offsets[basis_idx + 1];
			word forest(
				cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(start),
				cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(end));
			if (is_lyndon(forest))
				lyndon_idx.push_back(basis_idx + 1);
		}
	}
	if (method == 2)
		lyndon_idx = word_data.lyndon_idx;

	if (lyndon_idx.size() != lyndon_count)
		throw std::runtime_error("MKW Lyndon cache length mismatch");

	SparseIntMatrix projection;
	SparseIntMatrix inverse;
	SparseIntMatrix inverse_transpose;
	if (method == 2) {
		// The inverse projection converts Lyndon coordinates to bracket coordinates.
		// Its transpose is cached separately for the reverse pass.
		lyndon_proj_matrix_from_words(
			projection,
			word_data.lyndon_words,
			cache.total_length,
			[&word_data](const word& w) {
				return word_data.flat_idx.at(w);
			},
			[&word_data](uint64_t i, uint64_t j, uint64_t) {
				return word_data.flat_idx.at(concatenate_words(
					word_data.flat_words.at(i), word_data.flat_words.at(j)));
			});
		projection.inverse(inverse);
		inverse.transpose(inverse_transpose);
	}

	return std::make_unique<BasisCache>(
		method,
		std::move(lyndon_idx),
		std::move(inverse),
		std::move(inverse_transpose));
}

void prepare_branched_log_basis_cache_(
	const BranchedSigCache& cache,
	int method,
	bool use_disk
) {
	if (method < 1)
		return;
	if (method > 3)
		throw std::invalid_argument("branched log signature method must be 0, 1, 2, or 3");
	const int basis_method = std::min(method, 2);
	// Method 3 also uses the method 2 coordinate system for its BCH state.

	const std::pair<uint64_t, uint64_t> key(cache.dimension, cache.max_nodes);
	auto& registry = branched_log_basis_cache_registry_();
	{
		std::shared_lock lock(registry.mu);
		auto it = registry.map.find(key);
		if (it != registry.map.end() && it->second->method >= basis_method)
			return;
	}

	std::unique_ptr<CacheFile> file;
	if (use_disk) {
		auto dir = get_cache_dir();
		if (!std::filesystem::exists(dir / cache_folder_name))
			std::filesystem::create_directory(dir / cache_folder_name);
		file = std::make_unique<CacheFile>(
			cache.dimension, cache.max_nodes, mkw_basis_cache_prefix_);
	}
	if (file && file->exists()) {
		// A method 2 disk entry is also valid for a method 1 request.
		auto basis = std::make_unique<BasisCache>();
		file->read(basis);
		if (basis->method >= basis_method) {
			std::unique_lock lock(registry.mu);
			registry.map.insert_or_assign(key, std::move(basis));
			return;
		}
	}

	auto basis = build_branched_log_basis_cache_(cache, basis_method);
	if (file)
		file->write(basis);

	std::unique_lock lock(registry.mu);
	registry.map.insert_or_assign(key, std::move(basis));
}

const BasisCache& get_branched_log_basis_cache_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method
) {
	const std::pair<uint64_t, uint64_t> key(dimension, max_nodes);
	auto& registry = branched_log_basis_cache_registry_();
	{
		std::shared_lock lock(registry.mu);
		auto it = registry.map.find(key);
		if (it != registry.map.end() && it->second->method >= method)
			return *(it->second);
	}

	auto dir = get_cache_dir();
	if (!std::filesystem::exists(dir / cache_folder_name))
		throw cache_not_found_error(
			"MKW branched log basis cache not found - call prepare_branched_log_sig first");
	CacheFile file(dimension, max_nodes, mkw_basis_cache_prefix_);
	if (!file.exists())
		throw cache_not_found_error(
			"MKW branched log basis cache not found - call prepare_branched_log_sig first");
	auto basis = std::make_unique<BasisCache>();
	file.read(basis);
	if (basis->method < method)
		throw cache_not_found_error(
			"MKW branched log basis cache does not support the requested method");

	std::unique_lock lock(registry.mu);
	auto inserted = registry.map.insert_or_assign(key, std::move(basis));
	return *(inserted.first->second);
}

struct BranchedBchCache_ {
	// Ordinary BCH data plus the sparse lift of one path increment.
	BchCache bch;
	// Multipliers and flat coordinates for the nonzero segment lift entries.
	std::vector<double> linear_coefficients;
	std::vector<uint64_t> linear_basis_idx;
};

template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_compressed_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	int n_jobs
);

struct BranchedBchCacheRegistry_ {
	std::unordered_map<
		std::pair<uint64_t, uint64_t>,
		std::unique_ptr<BranchedBchCache_>,
		PairHash
	> map;
	std::shared_mutex mu;
};

BranchedBchCacheRegistry_& branched_bch_cache_registry_() {
	static BranchedBchCacheRegistry_ registry;
	return registry;
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

std::unique_ptr<BranchedBchCache_> build_branched_bch_cache_(
	const BranchedSigCache& branched_cache,
	bool use_disk
) {
	// BCH operates in method 2 coordinates, not in the expanded forest basis.
	const BasisCache& basis = get_branched_log_basis_cache_(
		branched_cache.dimension, branched_cache.max_nodes, 2);
	MkwWordData_ words = build_mkw_word_data_(branched_cache);
	const uint64_t m = words.lyndon_words.size();
	if (m > UINT32_MAX)
		throw std::overflow_error("MKW BCH basis is too large");

	auto result = std::make_unique<BranchedBchCache_>();
	BchCache& bch = result->bch;
	bch.dimension = branched_cache.dimension;
	bch.degree = branched_cache.max_nodes;
	bch.m = m;
	bch.coordinate_weights = words.lyndon_weights;
	result->linear_basis_idx.resize(m);
	for (uint64_t i = 0; i < m; ++i)
		result->linear_basis_idx[i] = basis.lyndon_idx[i] - 1;

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
	build_bch_formula_data(bch, use_disk);

	std::vector<double> unit_increment(branched_cache.dimension, 1.0);
	std::vector<double> linear_sig(branched_cache.total_length);
	result->linear_coefficients.resize(m);
	linear_branched_sig_(
		unit_increment.data(), linear_sig.data(), branched_cache);
	branched_sig_to_log_sig_compressed_<double, true>(
		linear_sig.data(), result->linear_coefficients.data(), 1,
		branched_cache.dimension, branched_cache.max_nodes, 2, 1);
	std::vector<uint32_t> linear_input_idx;
	linear_input_idx.reserve(m);
	for (uint64_t i = 0; i < m; ++i) {
		if (result->linear_coefficients[i] != 0.0)
			linear_input_idx.push_back(static_cast<uint32_t>(i));
	}
	configure_linear_bch_input(bch, std::move(linear_input_idx), false);
	return result;
}

void prepare_branched_bch_cache_(const BranchedSigCache& cache, bool use_disk) {
	const std::pair<uint64_t, uint64_t> key(cache.dimension, cache.max_nodes);
	auto& registry = branched_bch_cache_registry_();
	{
		std::shared_lock lock(registry.mu);
		if (registry.map.find(key) != registry.map.end())
			return;
	}
	auto bch = build_branched_bch_cache_(cache, use_disk);
	std::unique_lock lock(registry.mu);
	registry.map.try_emplace(key, std::move(bch));
}

const BranchedBchCache_& get_branched_bch_cache_(
	uint64_t dimension,
	uint64_t max_nodes
) {
	const std::pair<uint64_t, uint64_t> key(dimension, max_nodes);
	auto& registry = branched_bch_cache_registry_();
	std::shared_lock lock(registry.mu);
	const auto found = registry.map.find(key);
	if (found == registry.map.end())
		throw cache_not_found_error(
			"MKW BCH cache not found - call prepare_branched_log_sig with method=3 first");
	return *found->second;
}

void clear_branched_bch_cache_() {
	auto& registry = branched_bch_cache_registry_();
	std::unique_lock lock(registry.mu);
	registry.map.clear();
}

std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<BranchedLogProductCache>,
	PairHash
> branched_log_product_cache_registry_;
std::shared_mutex branched_log_product_cache_mu_;

template<std::floating_point T, bool ScalarTerm>
FORCE_INLINE T sig_tree_value_(const T* bsig, uint64_t flat_idx) {
	if constexpr (ScalarTerm) {
		return bsig[flat_idx];
	} else {
		return bsig[flat_idx - 1];
	}
}


template<bool ScalarTerm>
FORCE_INLINE uint64_t log_output_idx_(uint64_t flat_idx) {
	if constexpr (ScalarTerm) {
		return flat_idx;
	} else {
		return flat_idx - 1;
	}
}


struct BranchedLogPolyTermBuild_ {
	double coeff;
	std::vector<uint64_t> factors;
};


struct BranchedLogConstTerm_ {
	uint64_t out;
	double coeff;
};


struct BranchedLogTerm1_ {
	uint64_t out;
	uint64_t f0;
	double coeff;
};


struct BranchedLogTerm2_ {
	uint64_t out;
	uint64_t f0;
	uint64_t f1;
	double coeff;
};


struct BranchedLogTerm3_ {
	uint64_t out;
	uint64_t f0;
	uint64_t f1;
	uint64_t f2;
	double coeff;
};


struct BranchedLogTerm4_ {
	uint64_t out;
	uint64_t f0;
	uint64_t f1;
	uint64_t f2;
	uint64_t f3;
	double coeff;
};


struct BranchedLogTermN_ {
	uint64_t out;
	uint64_t factor_start;
	uint64_t factor_count;
	double coeff;
};


struct BranchedLogPolyCache_ {
	std::vector<BranchedLogConstTerm_> const_terms;
	std::vector<BranchedLogTerm1_> terms1;
	std::vector<BranchedLogTerm2_> terms2;
	std::vector<BranchedLogTerm3_> terms3;
	std::vector<BranchedLogTerm4_> terms4;
	std::vector<BranchedLogTermN_> terms_n;
	std::vector<uint64_t> factors;
};


std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<BranchedLogPolyCache_>,
	PairHash
> branched_log_poly_cache_registry_;
std::shared_mutex branched_log_poly_cache_mu_;


using BranchedLogPolyBuild_ = std::vector<BranchedLogPolyTermBuild_>;
using BranchedLogPolyMap_ = std::map<std::vector<uint64_t>, double>;


void add_branched_log_poly_term_(
	BranchedLogPolyMap_& terms,
	double coeff,
	std::vector<uint64_t> factors
) {
	if (coeff == 0.0)
		return;
	std::sort(factors.begin(), factors.end());
	terms[std::move(factors)] += coeff;
}


BranchedLogPolyBuild_ flatten_branched_log_poly_(BranchedLogPolyMap_&& terms) {
	BranchedLogPolyBuild_ out;
	out.reserve(terms.size());
	for (auto& [factors, coeff] : terms) {
		if (coeff != 0.0)
			out.push_back({ coeff, std::move(factors) });
	}
	return out;
}


void multiply_branched_log_poly_into_(
	const BranchedLogPolyBuild_& lhs,
	const BranchedLogPolyBuild_& rhs,
	BranchedLogPolyMap_& out
) {
	for (const auto& lterm : lhs) {
		for (const auto& rterm : rhs) {
			std::vector<uint64_t> factors;
			factors.reserve(lterm.factors.size() + rterm.factors.size());
			factors.insert(factors.end(), lterm.factors.begin(), lterm.factors.end());
			factors.insert(factors.end(), rterm.factors.begin(), rterm.factors.end());
			add_branched_log_poly_term_(out, lterm.coeff * rterm.coeff, std::move(factors));
		}
	}
}


BranchedLogPolyCache_ build_branched_log_poly_cache_(
	const BranchedSigCache& cache,
	const BranchedLogProductCache& product_cache
) {
	const uint64_t total_len = cache.total_length;
	const uint64_t num_products = product_cache.product_offsets.size() - 1;

	std::vector<BranchedLogPolyBuild_> h_poly(num_products);
	for (uint64_t product_idx = 1; product_idx < num_products; ++product_idx) {
		const uint64_t start = product_cache.product_offsets[product_idx];
		const uint64_t end = product_cache.product_offsets[product_idx + 1];
		std::vector<uint64_t> factors(
			product_cache.product_factors.begin() + start,
			product_cache.product_factors.begin() + end);
		std::sort(factors.begin(), factors.end());
		h_poly[product_idx].push_back({ 1.0, std::move(factors) });
	}

	std::vector<std::vector<BranchedLogPolyBuild_>> powers(
		cache.max_nodes + 1,
		std::vector<BranchedLogPolyBuild_>(num_products));
	if (cache.max_nodes >= 1)
		powers[1] = h_poly;

	for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
		for (uint64_t product_idx = 0; product_idx < num_products; ++product_idx) {
			BranchedLogPolyMap_ terms;
			const uint64_t start = product_cache.coproduct_offsets[product_idx];
			const uint64_t end = product_cache.coproduct_offsets[product_idx + 1];
			for (uint64_t pos = start; pos < end; pos += 2) {
				const uint64_t left = product_cache.coproduct_pairs[pos];
				const uint64_t right = product_cache.coproduct_pairs[pos + 1];
				multiply_branched_log_poly_into_(powers[k - 1][left], h_poly[right], terms);
			}
			powers[k][product_idx] = flatten_branched_log_poly_(std::move(terms));
		}
	}

	BranchedLogPolyCache_ out;
	for (uint64_t flat_idx = 1; flat_idx < total_len; ++flat_idx) {
		BranchedLogPolyMap_ terms;
		const uint64_t product_idx = product_cache.flat_to_product[flat_idx];
		for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
			const double coeff = (k % 2 == 0) ? -1.0 / static_cast<double>(k) : 1.0 / static_cast<double>(k);
			for (const auto& term : powers[k][product_idx])
				add_branched_log_poly_term_(terms, coeff * term.coeff, term.factors);
		}
		const BranchedLogPolyBuild_ flat_terms = flatten_branched_log_poly_(std::move(terms));
		for (const auto& term : flat_terms) {
			switch (term.factors.size()) {
			case 0:
				out.const_terms.push_back({ flat_idx, term.coeff });
				break;
			case 1:
				out.terms1.push_back({ flat_idx, term.factors[0], term.coeff });
				break;
			case 2:
				out.terms2.push_back({ flat_idx, term.factors[0], term.factors[1], term.coeff });
				break;
			case 3:
				out.terms3.push_back({ flat_idx, term.factors[0], term.factors[1], term.factors[2], term.coeff });
				break;
			case 4:
				out.terms4.push_back({ flat_idx, term.factors[0], term.factors[1], term.factors[2], term.factors[3], term.coeff });
				break;
			default: {
				const uint64_t factor_start = out.factors.size();
				out.factors.insert(out.factors.end(), term.factors.begin(), term.factors.end());
				out.terms_n.push_back({ flat_idx, factor_start, term.factors.size(), term.coeff });
				break;
			}
			}
		}
	}
	return out;
}


const BranchedLogPolyCache_& get_cached_branched_log_poly_cache_(
	const BranchedSigCache& cache
) {
	const auto key = make_branched_sig_cache_key(
		cache.dimension, cache.max_nodes, cache.planar);
	std::shared_lock rlock(branched_log_poly_cache_mu_);
	auto it = branched_log_poly_cache_registry_.find(key);
	if (it != branched_log_poly_cache_registry_.end())
		return *(it->second);
	throw cache_not_found_error(
		"Branched log sig cache not found - call prepare_branched_log_sig first");
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_poly_range_(
	const T* bsig,
	T* out,
	uint64_t start,
	uint64_t end,
	uint64_t stride,
	const BranchedLogPolyCache_& poly_cache
) {
	if (start == end)
		return;
	if (stride == 0)
		return;

	const uint64_t row_count = end - start;
	const T* bsig_start = bsig + start * stride;
	T* out_start = out + start * stride;
	std::memcpy(out_start, bsig_start, row_count * stride * sizeof(T));
	if constexpr (ScalarTerm) {
		for (uint64_t row = 0; row < row_count; ++row)
			out_start[row * stride] = static_cast<T>(0);
	}

	for (const auto& term : poly_cache.const_terms) {
		T* out_i = out_start;
		const uint64_t out_idx = log_output_idx_<ScalarTerm>(term.out);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, out_i += stride)
			out_i[out_idx] += coeff;
	}
	for (const auto& term : poly_cache.terms1) {
		const T* bsig_i = bsig_start;
		T* out_i = out_start;
		const uint64_t out_idx = log_output_idx_<ScalarTerm>(term.out);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, bsig_i += stride, out_i += stride) {
			out_i[out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0);
		}
	}
	for (const auto& term : poly_cache.terms2) {
		const T* bsig_i = bsig_start;
		T* out_i = out_start;
		const uint64_t out_idx = log_output_idx_<ScalarTerm>(term.out);
		const T coeff = static_cast<T>(term.coeff);
		uint64_t row = 0;
		for (; row + 4 <= row_count; row += 4, bsig_i += 4 * stride, out_i += 4 * stride) {
			out_i[out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0)
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1);
			out_i[stride + out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i + stride, term.f0)
				* sig_tree_value_<T, ScalarTerm>(bsig_i + stride, term.f1);
			out_i[2 * stride + out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i + 2 * stride, term.f0)
				* sig_tree_value_<T, ScalarTerm>(bsig_i + 2 * stride, term.f1);
			out_i[3 * stride + out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i + 3 * stride, term.f0)
				* sig_tree_value_<T, ScalarTerm>(bsig_i + 3 * stride, term.f1);
		}
		for (; row < row_count; ++row, bsig_i += stride, out_i += stride) {
			out_i[out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0)
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1);
		}
	}
	for (const auto& term : poly_cache.terms3) {
		const T* bsig_i = bsig_start;
		T* out_i = out_start;
		const uint64_t out_idx = log_output_idx_<ScalarTerm>(term.out);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, bsig_i += stride, out_i += stride) {
			out_i[out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0)
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1)
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f2);
		}
	}
	for (const auto& term : poly_cache.terms4) {
		const T* bsig_i = bsig_start;
		T* out_i = out_start;
		const uint64_t out_idx = log_output_idx_<ScalarTerm>(term.out);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, bsig_i += stride, out_i += stride) {
			out_i[out_idx] += coeff
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0)
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1)
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f2)
				* sig_tree_value_<T, ScalarTerm>(bsig_i, term.f3);
		}
	}
	for (const auto& term : poly_cache.terms_n) {
		const T* bsig_i = bsig_start;
		T* out_i = out_start;
		const uint64_t out_idx = log_output_idx_<ScalarTerm>(term.out);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, bsig_i += stride, out_i += stride) {
			T val = coeff;
			for (uint64_t pos = 0; pos < term.factor_count; ++pos)
				val *= sig_tree_value_<T, ScalarTerm>(bsig_i, poly_cache.factors[term.factor_start + pos]);
			out_i[out_idx] += val;
		}
	}
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_backprop_poly_range_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t start,
	uint64_t end,
	uint64_t stride,
	const BranchedLogPolyCache_& poly_cache
) {
	if (start == end)
		return;
	if (stride == 0)
		return;

	const uint64_t row_count = end - start;
	const T* bsig_start = bsig + start * stride;
	const T* derivs_start = derivs + start * stride;
	T* out_start = out + start * stride;
	std::memcpy(out_start, derivs_start, row_count * stride * sizeof(T));
	if constexpr (ScalarTerm) {
		for (uint64_t row = 0; row < row_count; ++row)
			out_start[row * stride] = static_cast<T>(0);
	}

	for (const auto& term : poly_cache.terms1) {
		const T* derivs_i = derivs_start;
		T* out_i = out_start;
		const uint64_t deriv_idx = log_output_idx_<ScalarTerm>(term.out);
		const uint64_t out0 = log_output_idx_<ScalarTerm>(term.f0);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, derivs_i += stride, out_i += stride) {
			const T d = coeff * derivs_i[deriv_idx];
			out_i[out0] += d;
		}
	}
	for (const auto& term : poly_cache.terms2) {
		const T* bsig_i = bsig_start;
		const T* derivs_i = derivs_start;
		T* out_i = out_start;
		const uint64_t deriv_idx = log_output_idx_<ScalarTerm>(term.out);
		const uint64_t out0 = log_output_idx_<ScalarTerm>(term.f0);
		const uint64_t out1 = log_output_idx_<ScalarTerm>(term.f1);
		const T coeff = static_cast<T>(term.coeff);
		uint64_t row = 0;
		for (; row + 4 <= row_count; row += 4, bsig_i += 4 * stride, derivs_i += 4 * stride, out_i += 4 * stride) {
			const T d0 = coeff * derivs_i[deriv_idx];
			const T v00 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0);
			const T v01 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1);
			out_i[out0] += d0 * v01;
			out_i[out1] += d0 * v00;

			const T* bsig1 = bsig_i + stride;
			const T d1 = coeff * derivs_i[stride + deriv_idx];
			const T v10 = sig_tree_value_<T, ScalarTerm>(bsig1, term.f0);
			const T v11 = sig_tree_value_<T, ScalarTerm>(bsig1, term.f1);
			out_i[stride + out0] += d1 * v11;
			out_i[stride + out1] += d1 * v10;

			const T* bsig2 = bsig_i + 2 * stride;
			const T d2 = coeff * derivs_i[2 * stride + deriv_idx];
			const T v20 = sig_tree_value_<T, ScalarTerm>(bsig2, term.f0);
			const T v21 = sig_tree_value_<T, ScalarTerm>(bsig2, term.f1);
			out_i[2 * stride + out0] += d2 * v21;
			out_i[2 * stride + out1] += d2 * v20;

			const T* bsig3 = bsig_i + 3 * stride;
			const T d3 = coeff * derivs_i[3 * stride + deriv_idx];
			const T v30 = sig_tree_value_<T, ScalarTerm>(bsig3, term.f0);
			const T v31 = sig_tree_value_<T, ScalarTerm>(bsig3, term.f1);
			out_i[3 * stride + out0] += d3 * v31;
			out_i[3 * stride + out1] += d3 * v30;
		}
		for (; row < row_count; ++row, bsig_i += stride, derivs_i += stride, out_i += stride) {
			const T d = coeff * derivs_i[deriv_idx];
			const T v0 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0);
			const T v1 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1);
			out_i[out0] += d * v1;
			out_i[out1] += d * v0;
		}
	}
	for (const auto& term : poly_cache.terms3) {
		const T* bsig_i = bsig_start;
		const T* derivs_i = derivs_start;
		T* out_i = out_start;
		const uint64_t deriv_idx = log_output_idx_<ScalarTerm>(term.out);
		const uint64_t out0 = log_output_idx_<ScalarTerm>(term.f0);
		const uint64_t out1 = log_output_idx_<ScalarTerm>(term.f1);
		const uint64_t out2 = log_output_idx_<ScalarTerm>(term.f2);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, bsig_i += stride, derivs_i += stride, out_i += stride) {
			const T d = coeff * derivs_i[deriv_idx];
			const T v0 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0);
			const T v1 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1);
			const T v2 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f2);
			out_i[out0] += d * v1 * v2;
			out_i[out1] += d * v0 * v2;
			out_i[out2] += d * v0 * v1;
		}
	}
	for (const auto& term : poly_cache.terms4) {
		const T* bsig_i = bsig_start;
		const T* derivs_i = derivs_start;
		T* out_i = out_start;
		const uint64_t deriv_idx = log_output_idx_<ScalarTerm>(term.out);
		const uint64_t out0 = log_output_idx_<ScalarTerm>(term.f0);
		const uint64_t out1 = log_output_idx_<ScalarTerm>(term.f1);
		const uint64_t out2 = log_output_idx_<ScalarTerm>(term.f2);
		const uint64_t out3 = log_output_idx_<ScalarTerm>(term.f3);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, bsig_i += stride, derivs_i += stride, out_i += stride) {
			const T d = coeff * derivs_i[deriv_idx];
			const T v0 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f0);
			const T v1 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f1);
			const T v2 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f2);
			const T v3 = sig_tree_value_<T, ScalarTerm>(bsig_i, term.f3);
			out_i[out0] += d * v1 * v2 * v3;
			out_i[out1] += d * v0 * v2 * v3;
			out_i[out2] += d * v0 * v1 * v3;
			out_i[out3] += d * v0 * v1 * v2;
		}
	}
	for (const auto& term : poly_cache.terms_n) {
		const T* bsig_i = bsig_start;
		const T* derivs_i = derivs_start;
		T* out_i = out_start;
		const uint64_t deriv_idx = log_output_idx_<ScalarTerm>(term.out);
		const T coeff = static_cast<T>(term.coeff);
		for (uint64_t row = 0; row < row_count; ++row, bsig_i += stride, derivs_i += stride, out_i += stride) {
			const T d = coeff * derivs_i[deriv_idx];
			for (uint64_t pos = 0; pos < term.factor_count; ++pos) {
				T partial = d;
				for (uint64_t other = 0; other < term.factor_count; ++other) {
					if (other != pos)
						partial *= sig_tree_value_<T, ScalarTerm>(bsig_i, poly_cache.factors[term.factor_start + other]);
				}
				out_i[log_output_idx_<ScalarTerm>(poly_cache.factors[term.factor_start + pos])] += partial;
			}
		}
	}
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs,
	bool planar
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t stride = ScalarTerm ? total_len : total_len - 1;

	const auto& poly_cache = get_cached_branched_log_poly_cache_(cache);
	auto work_range = [&](uint64_t start, uint64_t end) {
		branched_sig_to_log_sig_poly_range_<T, ScalarTerm>(
			bsig, out, start, end, stride, poly_cache);
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) {
		work_range(0, batch_size);
		return;
	}
	spawn_batch_threads(batch_size, n_jobs, work_range);
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_backprop_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs,
	bool planar
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t stride = ScalarTerm ? total_len : total_len - 1;

	const auto& poly_cache = get_cached_branched_log_poly_cache_(cache);
	auto work_range = [&](uint64_t start, uint64_t end) {
		branched_sig_to_log_sig_backprop_poly_range_<T, ScalarTerm>(
			bsig, derivs, out, start, end, stride, poly_cache);
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) {
		work_range(0, batch_size);
		return;
	}
	spawn_batch_threads(batch_size, n_jobs, work_range);
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_compressed_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	int n_jobs
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, true);
	const uint64_t input_stride = ScalarTerm
		? cache.total_length
		: cache.total_length - 1;
	const auto& poly_cache = get_cached_branched_log_poly_cache_(cache);
	const auto& basis_cache = get_branched_log_basis_cache_(dimension, max_nodes, method);
	const uint64_t output_stride = basis_cache.lyndon_idx.size();
	if (output_stride == 0)
		return;

	auto work_range = [&](uint64_t start, uint64_t end) {
		std::vector<T> expanded(input_stride);
		for (uint64_t row = start; row < end; ++row) {
			const T* bsig_row = bsig + row * input_stride;
			branched_sig_to_log_sig_poly_range_<T, ScalarTerm>(
				bsig_row, expanded.data(), 0, 1, input_stride, poly_cache);
			T* out_row = out + row * output_stride;
			for (uint64_t i = 0; i < output_stride; ++i) {
				const uint64_t flat_idx = basis_cache.lyndon_idx[i];
				out_row[i] = expanded[ScalarTerm ? flat_idx : flat_idx - 1];
			}
			if (method == 2)
				basis_cache.inv_proj_mat.mul_vec_inplace_lower(out_row);
		}
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) {
		work_range(0, batch_size);
		return;
	}
	spawn_batch_threads(batch_size, n_jobs, work_range);
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_backprop_compressed_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	int n_jobs
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, true);
	const uint64_t input_stride = ScalarTerm
		? cache.total_length
		: cache.total_length - 1;
	const auto& poly_cache = get_cached_branched_log_poly_cache_(cache);
	const auto& basis_cache = get_branched_log_basis_cache_(dimension, max_nodes, method);
	const uint64_t deriv_stride = basis_cache.lyndon_idx.size();
	if (input_stride == 0)
		return;

	auto work_range = [&](uint64_t start, uint64_t end) {
		std::vector<T> compact(deriv_stride);
		std::vector<T> expanded_derivs(input_stride);
		for (uint64_t row = start; row < end; ++row) {
			if (deriv_stride != 0)
				std::copy_n(derivs + row * deriv_stride, deriv_stride, compact.begin());
			if (method == 2)
				basis_cache.inv_proj_mat_transpose.mul_vec_inplace_upper(compact.data());
			std::fill(expanded_derivs.begin(), expanded_derivs.end(), static_cast<T>(0));
			for (uint64_t i = 0; i < deriv_stride; ++i) {
				const uint64_t flat_idx = basis_cache.lyndon_idx[i];
				expanded_derivs[ScalarTerm ? flat_idx : flat_idx - 1] = compact[i];
			}
			branched_sig_to_log_sig_backprop_poly_range_<T, ScalarTerm>(
				bsig + row * input_stride,
				expanded_derivs.data(),
				out + row * input_stride,
				0, 1, input_stride, poly_cache);
		}
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) {
		work_range(0, batch_size);
		return;
	}
	spawn_batch_threads(batch_size, n_jobs, work_range);
}

template<std::floating_point T>
void linear_mkw_log_sig_(
	const T* increment,
	T* out,
	const BranchedSigCache& branched_cache,
	const BranchedBchCache_& branched_bch
) {
	const BchCache& bch = branched_bch.bch;
	std::fill(out, out + bch.m, T(0));
	for (uint32_t coordinate : bch.linear_input_idx) {
		const uint64_t basis_idx = branched_bch.linear_basis_idx[coordinate];
		T value = static_cast<T>(
			branched_bch.linear_coefficients[coordinate]);
		for (uint64_t j = branched_cache.node_labels_offsets[basis_idx];
			j < branched_cache.node_labels_offsets[basis_idx + 1]; ++j)
			value *= increment[branched_cache.node_labels_data[j]];
		out[coordinate] = value;
	}
}

template<std::floating_point T>
void linear_mkw_log_sig_backprop_(
	const T* derivs,
	const T* increment,
	T* increment_derivs,
	const BranchedSigCache& branched_cache,
	const BranchedBchCache_& branched_bch
) {
	std::fill(
		increment_derivs,
		increment_derivs + branched_cache.dimension,
		T(0));
	for (uint32_t coordinate : branched_bch.bch.linear_input_idx) {
		const uint64_t basis_idx = branched_bch.linear_basis_idx[coordinate];
		const T base = derivs[coordinate]
			* static_cast<T>(branched_bch.linear_coefficients[coordinate]);
		const uint64_t start = branched_cache.node_labels_offsets[basis_idx];
		const uint64_t end = branched_cache.node_labels_offsets[basis_idx + 1];
		T prefix = T(1);
		for (uint64_t j = start; j < end; ++j) {
			T suffix = T(1);
			for (uint64_t k = j + 1; k < end; ++k)
				suffix *= increment[branched_cache.node_labels_data[k]];
			increment_derivs[branched_cache.node_labels_data[j]]
				+= base * prefix * suffix;
			prefix *= increment[branched_cache.node_labels_data[j]];
		}
	}
}

template<std::floating_point T>
void branched_log_sig_from_path_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs
) {
	if (length == 0)
		throw std::invalid_argument("branched_log_sig method 3 received an empty path");
	const BranchedSigCache& branched_cache = get_branched_sig_cache(
		dimension, max_nodes, true);
	const BranchedBchCache_& branched_bch = get_branched_bch_cache_(
		dimension, max_nodes);
	const BchCache& bch = branched_bch.bch;
	if (bch.m == 0)
		return;
	const uint64_t path_stride = length * dimension;
	const uint64_t m2 = bch.bch_coefficients.size();

	auto work_range = [&](uint64_t start, uint64_t end) {
		std::vector<T> increment(dimension);
		std::vector<T> segment(bch.m);
		std::vector<T> temporary(bch.m);
		std::vector<T> memo(m2 * bch.m);
		for (uint64_t row = start; row < end; ++row) {
			const T* path_row = path + row * path_stride;
			T* out_row = out + row * bch.m;
			if (length == 1) {
				std::fill(out_row, out_row + bch.m, T(0));
				continue;
			}
			for (uint64_t k = 0; k < dimension; ++k)
				increment[k] = path_row[dimension + k] - path_row[k];
			linear_mkw_log_sig_(
				increment.data(), out_row, branched_cache, branched_bch);
			T* accumulator = out_row;
			T* target = temporary.data();
			for (uint64_t segment_idx = 1; segment_idx + 1 < length; ++segment_idx) {
				const T* left = path_row + segment_idx * dimension;
				const T* right = left + dimension;
				for (uint64_t k = 0; k < dimension; ++k)
					increment[k] = right[k] - left[k];
				linear_mkw_log_sig_(
					increment.data(), segment.data(), branched_cache, branched_bch);
				std::swap(accumulator, target);
				bch_combine_linear_impl_<T>(
					target, segment.data(), accumulator, bch, memo.data());
			}
			if (accumulator != out_row)
				std::copy_n(accumulator, bch.m, out_row);
		}
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) {
		work_range(0, batch_size);
		return;
	}
	spawn_batch_threads(batch_size, n_jobs, work_range);
}

template<std::floating_point T>
void branched_log_sig_from_path_backprop_(
	const T* derivs,
	T* path_derivs,
	const T* path,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs
) {
	if (length == 0)
		throw std::invalid_argument("branched_log_sig method 3 received an empty path");
	const BranchedSigCache& branched_cache = get_branched_sig_cache(
		dimension, max_nodes, true);
	const BranchedBchCache_& branched_bch = get_branched_bch_cache_(
		dimension, max_nodes);
	const BchCache& bch = branched_bch.bch;
	const uint64_t path_stride = length * dimension;
	if (bch.m == 0) {
		std::fill(path_derivs, path_derivs + batch_size * path_stride, T(0));
		return;
	}
	const uint64_t m2 = bch.bch_coefficients.size();
	const uint64_t workspace_size = 7 * bch.m + 2 * m2 * bch.m;

	auto work_range = [&](uint64_t start, uint64_t end) {
		std::vector<T> workspace(workspace_size);
		std::vector<T> increment(dimension);
		std::vector<T> increment_derivs(dimension);
		T* current = workspace.data();
		T* previous = current + bch.m;
		T* segment = previous + bch.m;
		T* negative_segment = segment + bch.m;
		T* bch_workspace = negative_segment + bch.m;
		T* accumulated_derivs = bch_workspace + 2 * m2 * bch.m;
		T* left_derivs = accumulated_derivs + bch.m;
		T* segment_derivs = left_derivs + bch.m;
		for (uint64_t row = start; row < end; ++row) {
			const T* path_row = path + row * path_stride;
			T* path_derivs_row = path_derivs + row * path_stride;
			std::fill(path_derivs_row, path_derivs_row + path_stride, T(0));
			if (length == 1)
				continue;

			for (uint64_t k = 0; k < dimension; ++k)
				increment[k] = path_row[dimension + k] - path_row[k];
			linear_mkw_log_sig_(
				increment.data(), current, branched_cache, branched_bch);
			const uint64_t num_segments = length - 1;
			for (uint64_t segment_idx = 1; segment_idx < num_segments; ++segment_idx) {
				const T* left = path_row + segment_idx * dimension;
				const T* right = left + dimension;
				for (uint64_t k = 0; k < dimension; ++k)
					increment[k] = right[k] - left[k];
				linear_mkw_log_sig_(
					increment.data(), segment, branched_cache, branched_bch);
				bch_combine_linear_impl_<T>(
					current, segment, previous, bch, bch_workspace);
				std::swap(current, previous);
			}
			std::copy_n(derivs + row * bch.m, bch.m, accumulated_derivs);

			for (uint64_t segment_idx = num_segments - 1;
				segment_idx >= 1; --segment_idx) {
				const T* left = path_row + segment_idx * dimension;
				const T* right = left + dimension;
				for (uint64_t k = 0; k < dimension; ++k)
					increment[k] = right[k] - left[k];
				linear_mkw_log_sig_(
					increment.data(), segment, branched_cache, branched_bch);
				for (uint64_t k = 0; k < bch.m; ++k)
					negative_segment[k] = -segment[k];
				bch_combine_linear_impl_<T>(
					current, negative_segment, previous, bch, bch_workspace);
				if (bch.prune_linear_backprop)
					bch_combine_backprop_impl_<T, true, true>(
						accumulated_derivs, left_derivs, segment_derivs,
						previous, segment, bch, bch_workspace);
				else
					bch_combine_backprop_impl_<T, true, false>(
						accumulated_derivs, left_derivs, segment_derivs,
						previous, segment, bch, bch_workspace);
				linear_mkw_log_sig_backprop_(
					segment_derivs, increment.data(), increment_derivs.data(),
					branched_cache, branched_bch);
				for (uint64_t k = 0; k < dimension; ++k) {
					path_derivs_row[(segment_idx + 1) * dimension + k]
						+= increment_derivs[k];
					path_derivs_row[segment_idx * dimension + k]
						-= increment_derivs[k];
				}
				std::copy_n(left_derivs, bch.m, accumulated_derivs);
				std::swap(current, previous);
			}

			for (uint64_t k = 0; k < dimension; ++k)
				increment[k] = path_row[dimension + k] - path_row[k];
			linear_mkw_log_sig_backprop_(
				accumulated_derivs, increment.data(), increment_derivs.data(),
				branched_cache, branched_bch);
			for (uint64_t k = 0; k < dimension; ++k) {
				path_derivs_row[dimension + k] += increment_derivs[k];
				path_derivs_row[k] -= increment_derivs[k];
			}
		}
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) {
		work_range(0, batch_size);
		return;
	}
	spawn_batch_threads(batch_size, n_jobs, work_range);
}
}  // namespace


void prepare_branched_log_sig_cache(const BranchedSigCache& cache) {
	const auto key = make_branched_sig_cache_key(cache.dimension, cache.max_nodes, cache.planar);

	const BranchedLogProductCache* product_cache = nullptr;
	{
		std::shared_lock rlock(branched_log_product_cache_mu_);
		auto it = branched_log_product_cache_registry_.find(key);
		if (it != branched_log_product_cache_registry_.end())
			product_cache = it->second.get();
	}
	if (product_cache == nullptr) {
		auto built = std::make_unique<BranchedLogProductCache>(
			build_branched_log_product_cache(cache));
		std::unique_lock wlock(branched_log_product_cache_mu_);
		auto [it, _] = branched_log_product_cache_registry_.try_emplace(
			key, std::move(built));
		product_cache = it->second.get();
	}

	{
		std::shared_lock rlock(branched_log_poly_cache_mu_);
		if (branched_log_poly_cache_registry_.find(key)
			!= branched_log_poly_cache_registry_.end())
			return;
	}
	auto pc = std::make_unique<BranchedLogPolyCache_>(
		build_branched_log_poly_cache_(cache, *product_cache));
	std::unique_lock wlock(branched_log_poly_cache_mu_);
	branched_log_poly_cache_registry_.try_emplace(key, std::move(pc));
}


void clear_branched_log_sig_cache() {
	clear_branched_bch_cache_();
	clear_branched_log_basis_cache_();
	{
		std::unique_lock lock(branched_log_product_cache_mu_);
		branched_log_product_cache_registry_.clear();
	}
	std::unique_lock lock(branched_log_poly_cache_mu_);
	branched_log_poly_cache_registry_.clear();
}


template<std::floating_point T>
void branched_sig_to_log_sig_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	int n_jobs,
	bool planar,
	bool scalar_term
) {
	if (method == 3)
		throw std::invalid_argument(
			"method=3 is not supported in branched_sig_to_log_sig; use branched_log_sig instead");
	if (method < 0 || method > 2)
		throw std::invalid_argument("branched log signature method must be 0, 1, 2, or 3");
	if (method != 0 && !planar)
		throw std::invalid_argument("compressed branched log signatures require planar=True");
	if (method == 0) {
		if (scalar_term) {
			branched_sig_to_log_sig_<T, true>(
				bsig, out, batch_size, dimension, max_nodes, n_jobs, planar);
		} else {
			branched_sig_to_log_sig_<T, false>(
				bsig, out, batch_size, dimension, max_nodes, n_jobs, planar);
		}
	} else if (scalar_term) {
		branched_sig_to_log_sig_compressed_<T, true>(
			bsig, out, batch_size, dimension, max_nodes, method, n_jobs);
	} else {
		branched_sig_to_log_sig_compressed_<T, false>(
			bsig, out, batch_size, dimension, max_nodes, method, n_jobs);
	}
}


template<std::floating_point T>
void branched_sig_to_log_sig_backprop_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	int n_jobs,
	bool planar,
	bool scalar_term
) {
	if (method == 3)
		throw std::invalid_argument(
			"method=3 is not supported in branched_sig_to_log_sig_backprop");
	if (method < 0 || method > 2)
		throw std::invalid_argument("branched log signature method must be 0, 1, 2, or 3");
	if (method != 0 && !planar)
		throw std::invalid_argument("compressed branched log signatures require planar=True");
	if (method == 0) {
		if (scalar_term) {
			branched_sig_to_log_sig_backprop_<T, true>(
				bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar);
		} else {
			branched_sig_to_log_sig_backprop_<T, false>(
				bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar);
		}
	} else if (scalar_term) {
		branched_sig_to_log_sig_backprop_compressed_<T, true>(
			bsig, derivs, out, batch_size, dimension, max_nodes, method, n_jobs);
	} else {
		branched_sig_to_log_sig_backprop_compressed_<T, false>(
			bsig, derivs, out, batch_size, dimension, max_nodes, method, n_jobs);
	}
}

static void prepare_branched_log_sig_(
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	bool use_disk,
	bool planar
) {
	if (method < 0 || method > 3)
		throw std::invalid_argument("branched log signature method must be 0, 1, 2, or 3");
	if (method != 0 && !planar)
		throw std::invalid_argument("compressed branched log signatures are not available for planar=False");
	prepare_branched_sig_cache(dimension, max_nodes, use_disk, planar);
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	prepare_branched_log_sig_cache(cache);
	if (method >= 1)
		prepare_branched_log_basis_cache_(
			cache, (std::min)(method, 2), use_disk);
	if (method == 3)
		prepare_branched_bch_cache_(cache, use_disk);
}



extern "C" {

	CPSIG_API int prepare_branched_log_sig(uint64_t dimension, uint64_t max_nodes, int method, bool use_disk, bool planar) noexcept {
		SAFE_CALL(prepare_branched_log_sig_(dimension, max_nodes, method, use_disk, planar));
	}

	CPSIG_API int branched_sig_to_log_sig_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_<float>(bsig, out, batch_size, dimension, max_nodes, method, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_<double>(bsig, out, batch_size, dimension, max_nodes, method, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_backprop_<float>(bsig, derivs, out, batch_size, dimension, max_nodes, method, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_backprop_<double>(bsig, derivs, out, batch_size, dimension, max_nodes, method, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_log_sig_from_path_f(const float* path, float* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_log_sig_from_path_<float>(path, out, batch_size, length, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_log_sig_from_path_d(const double* path, double* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_log_sig_from_path_<double>(
			path, out, batch_size, length, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_log_sig_from_path_backprop_f(const float* derivs, float* path_derivs, const float* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_log_sig_from_path_backprop_<float>(derivs, path_derivs, path, batch_size, length, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_log_sig_from_path_backprop_d(const double* derivs, double* path_derivs, const double* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_log_sig_from_path_backprop_<double>(derivs, path_derivs, path, batch_size, length, dimension, max_nodes, n_jobs));
	}

}
