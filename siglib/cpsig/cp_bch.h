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
#include "cppch.h"
#include "cp_tensor_poly.h"
#include "cp_log_signature.h"
#include "cp_bch_data.h"
#include "words.h"
#include "macros.h"

// Sparse vector: list of (index, coefficient) pairs
using SparseVec = std::vector<std::pair<uint64_t, int>>;

struct BchCache {
	uint64_t dimension;
	uint64_t degree;
	uint64_t m; // = log_sig_length(dimension, degree)

	// commutator_table[i * m + j] = [e_i, e_j] for i < j
	std::vector<SparseVec> commutator_table;

	// Standard factorization of each d-letter Lyndon word (as index pairs)
	// For degree-1 words: left_factor[i] = right_factor[i] = UINT64_MAX
	std::vector<uint64_t> left_factor;
	std::vector<uint64_t> right_factor;

	// 2-letter BCH data
	std::vector<double> bch_coefficients;
	std::vector<uint64_t> bch_left_factor;
	std::vector<uint64_t> bch_right_factor;
};

extern std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<BchCache>, PairHash> bch_cache;

// ========================================================================
// BCH coefficients via 2-letter tensor algebra
// ========================================================================

inline std::vector<double> compute_bch_coefficients(uint64_t degree) {
	// Compute BCH coefficients by evaluating log(exp(e0) * exp(e1))
	// in the 2-letter tensor algebra truncated to degree N.

	const uint64_t dim2 = 2;
	uint64_t slen = ::sig_length(dim2, degree);

	// Build exp(e0): component at level k, index 0^k = 1/k!
	std::vector<double> exp0(slen, 0.0);
	exp0[0] = 1.0; // level 0

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dim2, degree + 2);

	{
		double factorial_inv = 1.0;
		for (uint64_t k = 1; k <= degree; ++k) {
			factorial_inv /= static_cast<double>(k);
			// The all-zeros word is the first element at each level (offset 0)
			exp0[level_index[k]] = factorial_inv;
		}
	}

	// Build exp(e1): component at level k, index 1^k = 1/k!
	std::vector<double> exp1(slen, 0.0);
	exp1[0] = 1.0;
	{
		double factorial_inv = 1.0;
		for (uint64_t k = 1; k <= degree; ++k) {
			factorial_inv /= static_cast<double>(k);
			// The all-ones word is the last element at each level
			exp1[level_index[k + 1] - 1] = factorial_inv;
		}
	}

	// Multiply: result = exp(e0) * exp(e1) using sig_combine_inplace_
	// Copy exp0 to result first, then combine with exp1 in-place
	std::vector<double> result(slen);
	std::memcpy(result.data(), exp0.data(), slen * sizeof(double));
	sig_combine_inplace_<double>(result.data(), exp1.data(), degree, level_index);

	// Project to 2-letter Lyndon basis via log_sig_lyndon_basis
	// (log_sig_lyndon_basis internally applies tensor_log_)
	uint64_t lslen = ::log_sig_length(dim2, degree);
	std::vector<double> bch_coefs(lslen, 0.0);

	// Use log_sig_lyndon_basis: first get Lyndon word indices, then project
	set_basis_cache(dim2, degree, 2, false);
	log_sig_lyndon_basis<double>(result.data(), bch_coefs.data(), dim2, degree);

	return bch_coefs;
}

// ========================================================================
// Commutator table via tensor algebra
// ========================================================================

// Sparse tensor algebra element: (tensor_monomial_index -> coefficient)
using TensorElem = std::unordered_map<uint64_t, int>;

inline void remove_zero_entries(TensorElem& m) {
	for (auto it = m.begin(); it != m.end(); ) {
		if (it->second == 0) it = m.erase(it);
		else ++it;
	}
}

// Tensor product of two TensorElems (concatenation of monomials), truncated to max_degree
inline TensorElem tensor_product(
	const TensorElem& a, uint64_t deg_a,
	const TensorElem& b, uint64_t deg_b,
	uint64_t dim, uint64_t max_degree
) {
	if (deg_a + deg_b > max_degree) return {};
	TensorElem result;
	for (const auto& [ia, ca] : a) {
		for (const auto& [ib, cb] : b) {
			uint64_t ic = concatenate_idx(ia, ib, deg_b, dim);
			result[ic] += ca * cb;
		}
	}
	remove_zero_entries(result);
	return result;
}

// Compute tensor algebra representations of each Lyndon basis element
// tensor_reps[i] maps tensor monomial index -> integer coefficient for basis element e_i
// tensor_degs[i] = degree (word length) of basis element i
inline void build_tensor_representations(
	const std::vector<word>& lyndon_words,
	const std::vector<uint64_t>& left_factor,
	const std::vector<uint64_t>& right_factor,
	uint64_t dim, uint64_t deg,
	std::vector<TensorElem>& tensor_reps,
	std::vector<uint64_t>& tensor_degs
) {
	uint64_t m = lyndon_words.size();
	tensor_reps.resize(m);
	tensor_degs.resize(m);

	for (uint64_t i = 0; i < m; ++i) {
		tensor_degs[i] = lyndon_words[i].size();

		if (tensor_degs[i] == 1) {
			// Generator e_g: tensor index = word_to_idx({g}, dim) = g + 1
			uint64_t g = lyndon_words[i][0];
			tensor_reps[i] = { {g + 1, 1} };
		}
		else {
			// e_i = [e_l, e_r] via standard factorization
			uint64_t l = left_factor[i];
			uint64_t r = right_factor[i];
			// [e_l, e_r] = e_l * e_r - e_r * e_l in tensor algebra
			TensorElem prod_lr = tensor_product(tensor_reps[l], tensor_degs[l],
				tensor_reps[r], tensor_degs[r], dim, deg);
			TensorElem prod_rl = tensor_product(tensor_reps[r], tensor_degs[r],
				tensor_reps[l], tensor_degs[l], dim, deg);
			// Subtract
			for (const auto& [idx, coef] : prod_rl) {
				prod_lr[idx] -= coef;
			}
			remove_zero_entries(prod_lr);
			tensor_reps[i] = std::move(prod_lr);
		}
	}
}

// Enumerate Lyndon words and compute standard factorization index pairs.
// Returns the Lyndon word list; fills left_factor/right_factor with index pairs.
inline std::vector<word> compute_factorization_indices(
	uint64_t dimension, uint64_t degree,
	std::vector<uint64_t>& left_factor,
	std::vector<uint64_t>& right_factor
) {
	std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
	std::unordered_set<word, WordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());
	std::unordered_map<word, uint64_t, WordHash> word_to_index;
	uint64_t m = lyndon_words.size();
	for (uint64_t i = 0; i < m; ++i) {
		word_to_index[lyndon_words[i]] = i;
	}
	left_factor.assign(m, UINT64_MAX);
	right_factor.assign(m, UINT64_MAX);
	for (uint64_t i = 0; i < m; ++i) {
		if (lyndon_words[i].size() > 1) {
			auto [l, r] = standard_factorization(lyndon_words[i], lyndon_set);
			left_factor[i] = word_to_index.at(l);
			right_factor[i] = word_to_index.at(r);
		}
	}
	return lyndon_words;
}

inline void build_commutator_table(BchCache& cache) {
	uint64_t m = cache.m;
	uint64_t dim = cache.dimension;
	uint64_t deg = cache.degree;

	std::vector<word> lyndon_words = compute_factorization_indices(dim, deg,
		cache.left_factor, cache.right_factor);

	// Build tensor algebra representations of each Lyndon basis element
	std::vector<TensorElem> tensor_reps;
	std::vector<uint64_t> tensor_degs;
	build_tensor_representations(lyndon_words, cache.left_factor, cache.right_factor,
		dim, deg, tensor_reps, tensor_degs);

	// Get the Lyndon word tensor indices and P^{-1} matrix from BasisCache
	// (set_basis_cache has already been called before build_commutator_table)
	const BasisCache& bc = get_basis_cache(dim, deg, 2);
	uint64_t n_lyndon = bc.lyndon_idx.size(); // = m

	// Initialize commutator table
	cache.commutator_table.resize(m * m);

	// For each pair (i, j) with i < j and size(i) + size(j) <= deg,
	// compute [e_i, e_j] = e_i * e_j - e_j * e_i in tensor algebra,
	// then project to Lyndon basis via P^{-1}
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			uint64_t total_deg = tensor_degs[i] + tensor_degs[j];
			if (total_deg > deg) continue;

			// Commutator in tensor algebra: e_i * e_j - e_j * e_i
			TensorElem comm = tensor_product(tensor_reps[i], tensor_degs[i],
				tensor_reps[j], tensor_degs[j], dim, deg);
			TensorElem prod_ji = tensor_product(tensor_reps[j], tensor_degs[j],
				tensor_reps[i], tensor_degs[i], dim, deg);
			for (const auto& [idx, coef] : prod_ji) {
				comm[idx] -= coef;
			}

			// Extract Lyndon-word coordinates from tensor algebra element
			// (only consider Lyndon words of degree = total_deg)
			// Note: lyndon_idx[k] is the tensor algebra index of the k-th Lyndon word
			std::vector<int> lyndon_coords(n_lyndon, 0);
			for (uint64_t k = 0; k < n_lyndon; ++k) {
				if (lyndon_words[k].size() != total_deg) continue;
				auto it = comm.find(bc.lyndon_idx[k]);
				if (it != comm.end()) {
					lyndon_coords[k] = it->second;
				}
			}

			// Apply P^{-1} (inv_proj_mat) to convert from Lyndon-word to Lyndon-basis coords
			// inv_proj_mat.mul_vec_inplace_lower operates in-place on a floating-point array
			// We need to do this with integers. Use double as intermediary.
			std::vector<double> coords_d(n_lyndon, 0.0);
			for (uint64_t k = 0; k < n_lyndon; ++k) {
				coords_d[k] = static_cast<double>(lyndon_coords[k]);
			}
			bc.inv_proj_mat.mul_vec_inplace_lower(coords_d.data());

			// Store non-zero entries in commutator table
			SparseVec& entry = cache.commutator_table[i * m + j];
			entry.clear();
			for (uint64_t k = 0; k < n_lyndon; ++k) {
				if (coords_d[k] != 0.0) {
					// Round to nearest integer (should be exact integers)
					int val = static_cast<int>(std::round(coords_d[k]));
					if (val != 0) {
						entry.push_back({ k, val });
					}
				}
			}
		}
	}
}

// ========================================================================
// BCH cache management
// ========================================================================

inline void set_bch_cache(uint64_t dimension, uint64_t degree) {
	std::pair<uint64_t, uint64_t> key(dimension, degree);
	if (bch_cache.find(key) != bch_cache.end()) return;

	// Ensure the d-letter basis cache is set (needed for commutator table)
	set_basis_cache(dimension, degree, 2, false);

	auto cache = std::make_unique<BchCache>();
	cache->dimension = dimension;
	cache->degree = degree;
	cache->m = ::log_sig_length(dimension, degree);

	// Use hardcoded BCH data when available (degrees 1-12)
	const BchHardcodedData* hc = get_hardcoded_bch_data(degree);
	if (hc) {
		cache->bch_coefficients.assign(hc->coefficients, hc->coefficients + hc->size);
		cache->bch_left_factor.assign(hc->left_factor, hc->left_factor + hc->size);
		cache->bch_right_factor.assign(hc->right_factor, hc->right_factor + hc->size);
	}
	else {
		// Fallback for degree > 12: compute at runtime
		// This requires a 2-letter basis cache for the Lyndon projection
		set_basis_cache(2, degree, 2, false);
		cache->bch_coefficients = compute_bch_coefficients(degree);
		compute_factorization_indices(2, degree, cache->bch_left_factor, cache->bch_right_factor);
	}

	// Build commutator table for d-letter Lyndon basis
	build_commutator_table(*cache);

	bch_cache.insert_or_assign(key, std::move(cache));
}

inline const BchCache& get_bch_cache(uint64_t dimension, uint64_t degree) {
	std::pair<uint64_t, uint64_t> key(dimension, degree);
	auto it = bch_cache.find(key);
	if (it == bch_cache.end()) {
		set_bch_cache(dimension, degree);
		it = bch_cache.find(key);
	}
	return *(it->second);
}

inline void clear_bch_cache() {
	bch_cache.clear();
}

// ========================================================================
// Lie bracket of two dense vectors using the commutator table
// ========================================================================

template<std::floating_point T>
void lie_bracket(
	const T* v1, const T* v2, T* result, uint64_t m,
	const std::vector<SparseVec>& commutator_table
) {
	std::fill(result, result + m, static_cast<T>(0));

	for (uint64_t i = 0; i < m; ++i) {
		if (v1[i] == static_cast<T>(0)) continue;
		for (uint64_t j = 0; j < m; ++j) {
			if (i == j) continue;
			if (v2[j] == static_cast<T>(0)) continue;

			T prod = v1[i] * v2[j];
			if (i < j) {
				for (const auto& [k, c] : commutator_table[i * m + j]) {
					result[k] += prod * static_cast<T>(c);
				}
			}
			else {
				// [e_i, e_j] = -[e_j, e_i]
				for (const auto& [k, c] : commutator_table[j * m + i]) {
					result[k] -= prod * static_cast<T>(c);
				}
			}
		}
	}
}

// ========================================================================
// Evaluate bracket tree of a 2-letter Lyndon word w on (L1, L2)
// ========================================================================

// workspace must have at least 2 * depth_remaining * m elements.
// Each recursion level uses 2 slots (left_buf, right_buf) at its depth.
template<std::floating_point T>
void evaluate_bracket_tree(
	uint64_t bch_idx,
	const T* log_sig1, const T* log_sig2,
	T* result, uint64_t m,
	const BchCache& cache,
	T* workspace, uint64_t depth
) {
	if (cache.bch_left_factor[bch_idx] == UINT64_MAX) {
		// Leaf: index 0 = letter A -> log_sig1, index 1 = letter B -> log_sig2
		std::memcpy(result, bch_idx == 0 ? log_sig1 : log_sig2, m * sizeof(T));
		return;
	}

	T* left_buf = workspace + 2 * depth * m;
	T* right_buf = workspace + (2 * depth + 1) * m;

	evaluate_bracket_tree(cache.bch_left_factor[bch_idx], log_sig1, log_sig2,
		left_buf, m, cache, workspace, depth + 1);
	evaluate_bracket_tree(cache.bch_right_factor[bch_idx], log_sig1, log_sig2,
		right_buf, m, cache, workspace, depth + 1);

	lie_bracket(left_buf, right_buf, result, m, cache.commutator_table);
}

// ========================================================================
// log_sig_combine_: BCH evaluation
// ========================================================================

// Internal implementation using pre-allocated buffers.
// bracket_buf: m elements. workspace: 2 * (degree - 1) * m elements.
template<std::floating_point T>
void log_sig_combine_impl_(
	const T* log_sig1, const T* log_sig2, T* out,
	const BchCache& cache, T* bracket_buf, T* workspace
) {
	uint64_t m = cache.m;
	for (uint64_t i = 0; i < m; ++i) {
		out[i] = log_sig1[i] + log_sig2[i];
	}

	uint64_t m2 = cache.bch_coefficients.size();
	// The first 2 entries (index 0 and 1) are the degree-1 words {A} and {B}
	// which correspond to the L1 + L2 terms already added above.
	for (uint64_t w = 2; w < m2; ++w) {
		T c_w = static_cast<T>(cache.bch_coefficients[w]);
		if (c_w == static_cast<T>(0)) continue;

		evaluate_bracket_tree(w, log_sig1, log_sig2, bracket_buf, m, cache,
			workspace, 0);

		for (uint64_t k = 0; k < m; ++k) {
			out[k] += c_w * bracket_buf[k];
		}
	}
}

template<std::floating_point T>
void log_sig_combine_(
	const T* log_sig1, const T* log_sig2, T* out,
	uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t i = 0; i < m; ++i) out[i] = log_sig1[i] + log_sig2[i];
		return;
	}

	std::vector<T> bracket_buf(m);
	std::vector<T> workspace(2 * (degree - 1) * m);
	log_sig_combine_impl_<T>(log_sig1, log_sig2, out, cache, bracket_buf.data(), workspace.data());
}

template<std::floating_point T>
void batch_log_sig_combine_(
	const T* log_sig1, const T* log_sig2, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine received degree 0");

	// Resolve cache once before any parallel region (avoids data race on cache init)
	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t i = 0; i < batch_size; ++i) {
			for (uint64_t j = 0; j < m; ++j)
				out[i * m + j] = log_sig1[i * m + j] + log_sig2[i * m + j];
		}
		return;
	}

	if (n_jobs != 1) {
		auto func = [&](const T* ls1, const T* ls2, T* o) {
			thread_local std::vector<T> tl_bracket_buf;
			thread_local std::vector<T> tl_workspace;
			tl_bracket_buf.resize(m);
			tl_workspace.resize(2 * (degree - 1) * m);
			log_sig_combine_impl_<T>(ls1, ls2, o, cache, tl_bracket_buf.data(), tl_workspace.data());
		};
		multi_threaded_batch_2<const T, const T, T>(func, log_sig1, log_sig2, out, batch_size, m, m, m, n_jobs);
	}
	else {
		std::vector<T> bracket_buf(m);
		std::vector<T> workspace(2 * (degree - 1) * m);
		for (uint64_t i = 0; i < batch_size; ++i) {
			log_sig_combine_impl_<T>(log_sig1 + i * m, log_sig2 + i * m, out + i * m,
				cache, bracket_buf.data(), workspace.data());
		}
	}
}
