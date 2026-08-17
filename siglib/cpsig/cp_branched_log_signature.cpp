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
#include "../shared/branched_log_horner.h"
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
	std::unique_ptr<BranchedLogHornerPlan>,
	PairHash
> branched_log_horner_plan_registry_;
std::shared_mutex branched_log_horner_plan_mu_;

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


const BranchedLogHornerPlan& get_cached_branched_log_horner_plan_(
	const BranchedSigCache& cache
) {
	const auto key = make_branched_sig_cache_key(
		cache.dimension, cache.max_nodes, cache.planar);
	std::shared_lock rlock(branched_log_horner_plan_mu_);
	auto it = branched_log_horner_plan_registry_.find(key);
	if (it != branched_log_horner_plan_registry_.end())
		return *(it->second);
	throw cache_not_found_error(
		"Branched log Horner plan not found - call prepare_branched_log_sig first");
}


template<std::floating_point T>
struct BranchedLogHornerBackpropWorkspace_ {
	std::vector<T> h;
	std::vector<T> states;
	std::vector<T> d_h;
	std::vector<T> d_current;
	std::vector<T> d_next;

	BranchedLogHornerBackpropWorkspace_(uint64_t size, uint64_t max_nodes)
		: h(size),
		states(size * (max_nodes > 1 ? max_nodes - 1 : 0)),
		d_h(size),
		d_current(size),
		d_next(size) {}
};


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_horner_row_(
	const T* bsig,
	T* out,
	const BranchedSigCache& cache,
	const BranchedLogHornerPlan& plan,
	BranchedLogHornerWorkspace<T>& workspace
) {
	if constexpr (ScalarTerm)
		out[0] = T(0);
	auto flat_value = [bsig](uint64_t flat) {
		return sig_tree_value_<T, ScalarTerm>(bsig, flat);
	};
	auto set_output = [out](uint64_t flat, T value) {
		out[log_output_idx_<ScalarTerm>(flat)] = value;
	};
	branched_log_horner_forward<T>(
		cache.total_length, cache.max_nodes, cache.planar,
		plan, flat_value, set_output, workspace);
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_backprop_horner_row_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t stride,
	const BranchedSigCache& cache,
	const BranchedLogHornerPlan& plan,
	BranchedLogHornerBackpropWorkspace_<T>& workspace
) {
	std::fill(out, out + stride, T(0));
	if (cache.max_nodes == 0)
		return;

	const uint64_t total_len = cache.total_length;
	const uint64_t num_products = plan.product_count;
	T* const h = workspace.h.data();
	auto flat_value = [bsig](uint64_t flat) {
		return sig_tree_value_<T, ScalarTerm>(bsig, flat);
	};
	fill_branched_log_horner_products<T>(
		plan, cache.planar, flat_value, h);

	for (uint64_t flat = 1; flat < total_len; ++flat) {
		const uint64_t idx = log_output_idx_<ScalarTerm>(flat);
		out[idx] = derivs[idx];
	}
	if (cache.max_nodes == 1)
		return;

	std::fill(workspace.states.begin(), workspace.states.end(), T(0));
	T* const last = workspace.states.data()
		+ (cache.max_nodes - 2) * num_products;
	const T initial_scale = T(1) / static_cast<T>(cache.max_nodes);
	for (uint64_t product = 1; product < num_products; ++product) {
		if (plan.product_node_counts[product] == 1)
			last[product] = initial_scale * h[product];
	}

	for (uint64_t k = cache.max_nodes - 1; k > 1; --k) {
		T* const current = workspace.states.data() + (k - 2) * num_products;
		const T* const next = current + num_products;
		const T scale = T(1) / static_cast<T>(k);
		const uint64_t max_product_nodes = cache.max_nodes - k + 1;
		for (uint64_t product = 1; product < num_products; ++product) {
			if (plan.product_node_counts[product] > max_product_nodes)
				continue;
			T value = T(0);
			const uint64_t start = plan.coproduct_offsets[product];
			const uint64_t end = plan.coproduct_offsets[product + 1];
			for (uint64_t pos = start; pos < end; pos += 2) {
				value += next[plan.coproduct_pairs[pos]]
					* h[plan.coproduct_pairs[pos + 1]];
			}
			current[product] = scale * h[product] - value;
		}
	}

	std::fill(workspace.d_h.begin(), workspace.d_h.end(), T(0));
	std::fill(workspace.d_current.begin(), workspace.d_current.end(), T(0));
	std::fill(workspace.d_next.begin(), workspace.d_next.end(), T(0));
	T* const d_h = workspace.d_h.data();
	T* d_current = workspace.d_current.data();
	T* d_next = workspace.d_next.data();
	const T* const b2 = workspace.states.data();
	for (uint64_t flat = 1; flat < total_len; ++flat) {
		const T d = derivs[log_output_idx_<ScalarTerm>(flat)];
		const uint64_t product = branched_log_product_for_flat(
			plan, cache.planar, flat);
		const uint64_t start = plan.coproduct_offsets[product];
		const uint64_t end = plan.coproduct_offsets[product + 1];
		for (uint64_t pos = start; pos < end; pos += 2) {
			const uint64_t left = plan.coproduct_pairs[pos];
			const uint64_t right = plan.coproduct_pairs[pos + 1];
			d_current[left] -= d * h[right];
			d_h[right] -= d * b2[left];
		}
	}

	for (uint64_t k = 2; k < cache.max_nodes; ++k) {
		const T* const next_state = workspace.states.data()
			+ (k - 1) * num_products;
		const T scale = T(1) / static_cast<T>(k);
		const uint64_t max_product_nodes = cache.max_nodes - k + 1;
		for (uint64_t product = 1; product < num_products; ++product) {
			if (plan.product_node_counts[product] > max_product_nodes)
				continue;
			const T d = d_current[product];
			d_h[product] += scale * d;
			const uint64_t start = plan.coproduct_offsets[product];
			const uint64_t end = plan.coproduct_offsets[product + 1];
			for (uint64_t pos = start; pos < end; pos += 2) {
				const uint64_t left = plan.coproduct_pairs[pos];
				const uint64_t right = plan.coproduct_pairs[pos + 1];
				d_next[left] -= d * h[right];
				d_h[right] -= d * next_state[left];
			}
		}
		std::swap(d_current, d_next);
		std::fill(d_next, d_next + num_products, T(0));
	}
	const T last_scale = T(1) / static_cast<T>(cache.max_nodes);
	for (uint64_t product = 1; product < num_products; ++product) {
		if (plan.product_node_counts[product] == 1)
			d_h[product] += last_scale * d_current[product];
	}

	if (cache.planar) {
		for (uint64_t product = 1; product < num_products; ++product) {
			if (plan.product_node_counts[product] < cache.max_nodes)
				out[log_output_idx_<ScalarTerm>(product)] += d_h[product];
		}
		return;
	}
	for (uint64_t product = num_products - 1; product > 0; --product) {
		if (plan.product_node_counts[product] >= cache.max_nodes)
			continue;
		const uint64_t factor = plan.cpu_products.last_factor[product];
		const uint64_t parent = plan.cpu_products.parent[product];
		const T d = d_h[product];
		if (parent == 0) {
			out[log_output_idx_<ScalarTerm>(factor)] += d;
		} else {
			d_h[parent] += d * sig_tree_value_<T, ScalarTerm>(bsig, factor);
			out[log_output_idx_<ScalarTerm>(factor)] += d * h[parent];
		}
	}
}


template<std::floating_point T, bool ScalarTerm>
void fill_branched_log_horner_products_range_(
	const T* bsig,
	uint64_t row_count,
	uint64_t stride,
	const BranchedSigCache& cache,
	const BranchedLogHornerPlan& plan,
	T* h
) {
	const uint64_t product_count = plan.product_count;
	if (cache.planar) {
		for (uint64_t product = 1; product < product_count; ++product) {
			T* const h_product = h + product * row_count;
			for (uint64_t row = 0; row < row_count; ++row) {
				h_product[row] = sig_tree_value_<T, ScalarTerm>(
					bsig + row * stride, product);
			}
		}
		return;
	}

	for (uint64_t product = 1; product < product_count; ++product) {
		T* const h_product = h + product * row_count;
		const uint64_t parent = plan.cpu_products.parent[product];
		const uint64_t factor = plan.cpu_products.last_factor[product];
		const T* const h_parent = h + parent * row_count;
		for (uint64_t row = 0; row < row_count; ++row) {
			const T factor_value = sig_tree_value_<T, ScalarTerm>(
				bsig + row * stride, factor);
			h_product[row] = parent == 0
				? factor_value
				: h_parent[row] * factor_value;
		}
	}
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_horner_range_(
	const T* bsig,
	T* out,
	uint64_t start,
	uint64_t end,
	uint64_t stride,
	const BranchedSigCache& cache,
	const BranchedLogHornerPlan& plan
) {
	if (start == end || stride == 0)
		return;
	const uint64_t row_count = end - start;
	constexpr uint64_t tile_rows = 64;
	if (row_count > tile_rows) {
		for (uint64_t tile_start = start; tile_start < end; tile_start += tile_rows) {
			branched_sig_to_log_sig_horner_range_<T, ScalarTerm>(
				bsig, out, tile_start, (std::min)(tile_start + tile_rows, end),
				stride, cache, plan);
		}
		return;
	}
	if (row_count < 4 || cache.max_nodes <= 1) {
		thread_local BranchedLogHornerWorkspace<T> workspace(0);
		workspace.h.resize(plan.product_count);
		workspace.current.resize(plan.product_count);
		workspace.next.resize(plan.product_count);
		for (uint64_t row = start; row < end; ++row) {
			branched_sig_to_log_sig_horner_row_<T, ScalarTerm>(
				bsig + row * stride, out + row * stride,
				cache, plan, workspace);
		}
		return;
	}

	const T* const bsig_start = bsig + start * stride;
	T* const out_start = out + start * stride;
	const uint64_t product_count = plan.product_count;
	const uint64_t workspace_size = product_count * row_count;
	thread_local std::vector<T> workspace;
	workspace.resize(3 * workspace_size + row_count);
	T* const h = workspace.data();
	T* current = h + workspace_size;
	T* next = current + workspace_size;
	T* const values = next + workspace_size;
	std::fill(h, h + row_count, T(0));
	std::fill(current, current + row_count, T(0));
	std::fill(next, next + row_count, T(0));
	fill_branched_log_horner_products_range_<T, ScalarTerm>(
		bsig_start, row_count, stride, cache, plan, h);

	const T initial_scale = T(1) / static_cast<T>(cache.max_nodes);
	for (uint64_t product = 1; product < product_count; ++product) {
		if (plan.product_node_counts[product] != 1)
			continue;
		T* const current_product = current + product * row_count;
		const T* const h_product = h + product * row_count;
		for (uint64_t row = 0; row < row_count; ++row)
			current_product[row] = initial_scale * h_product[row];
	}

	for (uint64_t k = cache.max_nodes - 1; k > 1; --k) {
		const T scale = T(1) / static_cast<T>(k);
		const uint64_t max_product_nodes = cache.max_nodes - k + 1;
		for (uint64_t product = 1; product < product_count; ++product) {
			if (plan.product_node_counts[product] > max_product_nodes)
				continue;
			T* const next_product = next + product * row_count;
			const T* const h_product = h + product * row_count;
			for (uint64_t row = 0; row < row_count; ++row)
				next_product[row] = scale * h_product[row];
			const uint64_t coprod_start = plan.coproduct_offsets[product];
			const uint64_t coprod_end = plan.coproduct_offsets[product + 1];
			for (uint64_t pos = coprod_start; pos < coprod_end; pos += 2) {
				const T* const left = current
					+ plan.coproduct_pairs[pos] * row_count;
				const T* const right = h
					+ plan.coproduct_pairs[pos + 1] * row_count;
				for (uint64_t row = 0; row < row_count; ++row)
					next_product[row] -= left[row] * right[row];
			}
		}
		std::swap(current, next);
	}

	if constexpr (ScalarTerm) {
		for (uint64_t row = 0; row < row_count; ++row)
			out_start[row * stride] = T(0);
	}
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		const uint64_t product = branched_log_product_for_flat(
			plan, cache.planar, flat);
		const uint64_t coprod_start = plan.coproduct_offsets[product];
		const uint64_t coprod_end = plan.coproduct_offsets[product + 1];
		std::fill(values, values + row_count, T(0));
		for (uint64_t pos = coprod_start; pos < coprod_end; pos += 2) {
			const T* const left = current
				+ plan.coproduct_pairs[pos] * row_count;
			const T* const right = h
				+ plan.coproduct_pairs[pos + 1] * row_count;
			for (uint64_t row = 0; row < row_count; ++row)
				values[row] += left[row] * right[row];
		}
		for (uint64_t row = 0; row < row_count; ++row) {
			out_start[row * stride + log_output_idx_<ScalarTerm>(flat)]
				= sig_tree_value_<T, ScalarTerm>(
					bsig_start + row * stride, flat) - values[row];
		}
	}
}


template<std::floating_point T, bool ScalarTerm>
void branched_sig_to_log_sig_backprop_horner_range_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t start,
	uint64_t end,
	uint64_t stride,
	const BranchedSigCache& cache,
	const BranchedLogHornerPlan& plan
) {
	if (start == end || stride == 0)
		return;
	const uint64_t row_count = end - start;
	constexpr uint64_t tile_rows = 64;
	if (row_count > tile_rows) {
		for (uint64_t tile_start = start; tile_start < end; tile_start += tile_rows) {
			branched_sig_to_log_sig_backprop_horner_range_<T, ScalarTerm>(
				bsig, derivs, out, tile_start,
				(std::min)(tile_start + tile_rows, end), stride, cache, plan);
		}
		return;
	}
	if (row_count < 4 || cache.max_nodes <= 1) {
		thread_local BranchedLogHornerBackpropWorkspace_<T> workspace(0, 0);
		workspace.h.resize(plan.product_count);
		workspace.states.resize(
			plan.product_count * (cache.max_nodes > 1 ? cache.max_nodes - 1 : 0));
		workspace.d_h.resize(plan.product_count);
		workspace.d_current.resize(plan.product_count);
		workspace.d_next.resize(plan.product_count);
		for (uint64_t row = start; row < end; ++row) {
			branched_sig_to_log_sig_backprop_horner_row_<T, ScalarTerm>(
				bsig + row * stride, derivs + row * stride, out + row * stride,
				stride, cache, plan, workspace);
		}
		return;
	}

	const T* const bsig_start = bsig + start * stride;
	const T* const derivs_start = derivs + start * stride;
	T* const out_start = out + start * stride;
	const uint64_t product_count = plan.product_count;
	const uint64_t workspace_size = product_count * row_count;
	thread_local std::vector<T> workspace;
	workspace.resize((cache.max_nodes + 3) * workspace_size);
	T* const h = workspace.data();
	T* const states = h + workspace_size;
	T* const d_h = states + (cache.max_nodes - 1) * workspace_size;
	T* d_current = d_h + workspace_size;
	T* d_next = d_current + workspace_size;
	std::fill(h, h + row_count, T(0));
	for (uint64_t state = 0; state < cache.max_nodes - 1; ++state) {
		std::fill(
			states + state * workspace_size,
			states + state * workspace_size + row_count,
			T(0));
	}
	for (uint64_t product = 1; product < product_count; ++product) {
		const uint64_t product_nodes = plan.product_node_counts[product];
		if (product_nodes < cache.max_nodes) {
			std::fill(
				d_h + product * row_count,
				d_h + (product + 1) * row_count, T(0));
			std::fill(
				d_current + product * row_count,
				d_current + (product + 1) * row_count, T(0));
		}
		if (product_nodes + 1 < cache.max_nodes) {
			std::fill(
				d_next + product * row_count,
				d_next + (product + 1) * row_count, T(0));
		}
	}
	fill_branched_log_horner_products_range_<T, ScalarTerm>(
		bsig_start, row_count, stride, cache, plan, h);

	for (uint64_t row = 0; row < row_count; ++row) {
		std::copy_n(
			derivs_start + row * stride, stride, out_start + row * stride);
		if constexpr (ScalarTerm)
			out_start[row * stride] = T(0);
	}

	T* const last = states
		+ (cache.max_nodes - 2) * workspace_size;
	const T initial_scale = T(1) / static_cast<T>(cache.max_nodes);
	for (uint64_t product = 1; product < product_count; ++product) {
		if (plan.product_node_counts[product] != 1)
			continue;
		T* const last_product = last + product * row_count;
		const T* const h_product = h + product * row_count;
		for (uint64_t row = 0; row < row_count; ++row)
			last_product[row] = initial_scale * h_product[row];
	}

	for (uint64_t k = cache.max_nodes - 1; k > 1; --k) {
		T* const current_state = states + (k - 2) * workspace_size;
		const T* const next_state = current_state + workspace_size;
		const T scale = T(1) / static_cast<T>(k);
		const uint64_t max_product_nodes = cache.max_nodes - k + 1;
		for (uint64_t product = 1; product < product_count; ++product) {
			if (plan.product_node_counts[product] > max_product_nodes)
				continue;
			T* const current_product = current_state + product * row_count;
			const T* const h_product = h + product * row_count;
			for (uint64_t row = 0; row < row_count; ++row)
				current_product[row] = scale * h_product[row];
			const uint64_t coprod_start = plan.coproduct_offsets[product];
			const uint64_t coprod_end = plan.coproduct_offsets[product + 1];
			for (uint64_t pos = coprod_start; pos < coprod_end; pos += 2) {
				const T* const left = next_state
					+ plan.coproduct_pairs[pos] * row_count;
				const T* const right = h
					+ plan.coproduct_pairs[pos + 1] * row_count;
				for (uint64_t row = 0; row < row_count; ++row)
					current_product[row] -= left[row] * right[row];
			}
		}
	}

	const T* const b2 = states;
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		const uint64_t product = branched_log_product_for_flat(
			plan, cache.planar, flat);
		const uint64_t coprod_start = plan.coproduct_offsets[product];
		const uint64_t coprod_end = plan.coproduct_offsets[product + 1];
		for (uint64_t pos = coprod_start; pos < coprod_end; pos += 2) {
			const uint64_t left = plan.coproduct_pairs[pos];
			const uint64_t right = plan.coproduct_pairs[pos + 1];
			T* const d_current_left = d_current + left * row_count;
			T* const d_h_right = d_h + right * row_count;
			const T* const h_right = h + right * row_count;
			const T* const b2_left = b2 + left * row_count;
			for (uint64_t row = 0; row < row_count; ++row) {
				const T d = derivs_start[
					row * stride + log_output_idx_<ScalarTerm>(flat)];
				d_current_left[row] -= d * h_right[row];
				d_h_right[row] -= d * b2_left[row];
			}
		}
	}

	for (uint64_t k = 2; k < cache.max_nodes; ++k) {
		const T* const next_state = states + (k - 1) * workspace_size;
		const T scale = T(1) / static_cast<T>(k);
		const uint64_t max_product_nodes = cache.max_nodes - k + 1;
		for (uint64_t product = 1; product < product_count; ++product) {
			if (plan.product_node_counts[product] > max_product_nodes)
				continue;
			T* const d_h_product = d_h + product * row_count;
			const T* const d_current_product
				= d_current + product * row_count;
			for (uint64_t row = 0; row < row_count; ++row)
				d_h_product[row] += scale * d_current_product[row];
			const uint64_t coprod_start = plan.coproduct_offsets[product];
			const uint64_t coprod_end = plan.coproduct_offsets[product + 1];
			for (uint64_t pos = coprod_start; pos < coprod_end; pos += 2) {
				const uint64_t left = plan.coproduct_pairs[pos];
				const uint64_t right = plan.coproduct_pairs[pos + 1];
				T* const d_next_left = d_next + left * row_count;
				T* const d_h_right = d_h + right * row_count;
				const T* const h_right = h + right * row_count;
				const T* const next_state_left
					= next_state + left * row_count;
				for (uint64_t row = 0; row < row_count; ++row) {
					const T d = d_current_product[row];
					d_next_left[row] -= d * h_right[row];
					d_h_right[row] -= d * next_state_left[row];
				}
			}
		}
		std::swap(d_current, d_next);
		if (k + 1 < cache.max_nodes) {
			const uint64_t max_next_nodes = cache.max_nodes - k - 1;
			for (uint64_t product = 1; product < product_count; ++product) {
				if (plan.product_node_counts[product] <= max_next_nodes) {
					std::fill(
						d_next + product * row_count,
						d_next + (product + 1) * row_count, T(0));
				}
			}
		}
	}
	const T last_scale = T(1) / static_cast<T>(cache.max_nodes);
	for (uint64_t product = 1; product < product_count; ++product) {
		if (plan.product_node_counts[product] != 1)
			continue;
		T* const d_h_product = d_h + product * row_count;
		const T* const d_current_product = d_current + product * row_count;
		for (uint64_t row = 0; row < row_count; ++row)
			d_h_product[row] += last_scale * d_current_product[row];
	}

	if (cache.planar) {
		for (uint64_t product = 1; product < product_count; ++product) {
			if (plan.product_node_counts[product] >= cache.max_nodes)
				continue;
			const T* const d_h_product = d_h + product * row_count;
			for (uint64_t row = 0; row < row_count; ++row) {
				out_start[row * stride + log_output_idx_<ScalarTerm>(product)]
					+= d_h_product[row];
			}
		}
		return;
	}
	for (uint64_t product = product_count - 1; product > 0; --product) {
		if (plan.product_node_counts[product] >= cache.max_nodes)
			continue;
		const uint64_t factor = plan.cpu_products.last_factor[product];
		const uint64_t parent = plan.cpu_products.parent[product];
		const T* const h_parent = h + parent * row_count;
		T* const d_h_product = d_h + product * row_count;
		T* const d_h_parent = d_h + parent * row_count;
		for (uint64_t row = 0; row < row_count; ++row) {
			const T d = d_h_product[row];
			T& out_factor = out_start[
				row * stride + log_output_idx_<ScalarTerm>(factor)];
			if (parent == 0) {
				out_factor += d;
			} else {
				d_h_parent[row] += d * sig_tree_value_<T, ScalarTerm>(
					bsig_start + row * stride, factor);
				out_factor += d * h_parent[row];
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

	const auto& plan = get_cached_branched_log_horner_plan_(cache);
	auto work_range = [&](uint64_t start, uint64_t end) {
		branched_sig_to_log_sig_horner_range_<T, ScalarTerm>(
			bsig, out, start, end, stride, cache, plan);
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

	const auto& plan = get_cached_branched_log_horner_plan_(cache);
	auto work_range = [&](uint64_t start, uint64_t end) {
		branched_sig_to_log_sig_backprop_horner_range_<T, ScalarTerm>(
			bsig, derivs, out, start, end, stride, cache, plan);
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
	const auto& plan = get_cached_branched_log_horner_plan_(cache);
	const auto& basis_cache = get_branched_log_basis_cache_(dimension, max_nodes, method);
	const uint64_t output_stride = basis_cache.lyndon_idx.size();
	if (output_stride == 0)
		return;

	auto work_range = [&](uint64_t start, uint64_t end) {
		thread_local std::vector<T> expanded;
		thread_local BranchedLogHornerWorkspace<T> workspace(0);
		expanded.resize(input_stride);
		workspace.h.resize(plan.product_count);
		workspace.current.resize(plan.product_count);
		workspace.next.resize(plan.product_count);
		for (uint64_t row = start; row < end; ++row) {
			const T* bsig_row = bsig + row * input_stride;
			branched_sig_to_log_sig_horner_row_<T, ScalarTerm>(
				bsig_row, expanded.data(), cache, plan, workspace);
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
	const auto& plan = get_cached_branched_log_horner_plan_(cache);
	const auto& basis_cache = get_branched_log_basis_cache_(dimension, max_nodes, method);
	const uint64_t deriv_stride = basis_cache.lyndon_idx.size();
	if (input_stride == 0)
		return;

	auto work_range = [&](uint64_t start, uint64_t end) {
		thread_local std::vector<T> compact;
		thread_local std::vector<T> expanded_derivs;
		thread_local BranchedLogHornerBackpropWorkspace_<T> workspace(0, 0);
		compact.resize(deriv_stride);
		expanded_derivs.resize(input_stride);
		workspace.h.resize(plan.product_count);
		workspace.states.resize(
			plan.product_count * (cache.max_nodes > 1 ? cache.max_nodes - 1 : 0));
		workspace.d_h.resize(plan.product_count);
		workspace.d_current.resize(plan.product_count);
		workspace.d_next.resize(plan.product_count);
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
			branched_sig_to_log_sig_backprop_horner_row_<T, ScalarTerm>(
				bsig + row * input_stride,
				expanded_derivs.data(),
				out + row * input_stride,
				input_stride, cache, plan, workspace);
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
	{
		std::shared_lock rlock(branched_log_horner_plan_mu_);
		if (branched_log_horner_plan_registry_.find(key)
			!= branched_log_horner_plan_registry_.end())
			return;
	}
	auto built = std::make_unique<BranchedLogHornerPlan>(
		build_branched_log_horner_plan(cache));
	std::unique_lock wlock(branched_log_horner_plan_mu_);
	branched_log_horner_plan_registry_.try_emplace(key, std::move(built));
}


void clear_branched_log_sig_cache() {
	clear_branched_bch_cache_();
	clear_branched_log_basis_cache_();
	std::unique_lock lock(branched_log_horner_plan_mu_);
	branched_log_horner_plan_registry_.clear();
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
