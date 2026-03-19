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

	// Transposed commutator table (CSR format, grouped by output index k)
	// For output element k, entries from comm_k_ptr[k] to comm_k_ptr[k+1]-1
	// store (i, j, coefficient) triples where i < j and [e_i, e_j] has a
	// non-zero k-th component.
	std::vector<uint32_t> comm_k_ptr;
	std::vector<uint32_t> comm_k_i;
	std::vector<uint32_t> comm_k_j;
	std::vector<int> comm_k_val;
	std::vector<double> comm_k_val_d; // pre-cast to double

	// Input-grouped commutator table (CSR format, grouped by input index a)
	// For each input index a, stores (k, partner, signed_c) triples where
	// signed_c = +c when a==i, signed_c = -c when a==j, partner is the other index.
	std::vector<uint32_t> comm_a_ptr;
	std::vector<uint32_t> comm_a_k;
	std::vector<uint32_t> comm_a_partner;
	std::vector<int> comm_a_signed_c;

	// Pair-grouped commutator table (CSR format, grouped by (i,j) pair)
	// For the backward pass: allows factoring out v1/v2 loads per pair.
	uint32_t n_pairs = 0;
	std::vector<uint32_t> comm_ij_i;    // [n_pairs] left index
	std::vector<uint32_t> comm_ij_j;    // [n_pairs] right index
	std::vector<uint32_t> comm_ij_ptr;  // [n_pairs+1] CSR row pointers
	std::vector<uint32_t> comm_ij_k;    // [nnz] output index
	std::vector<double>   comm_ij_c;    // [nnz] coefficient (pre-cast)

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

	// Build transposed commutator table (CSR format, grouped by output k)
	std::vector<uint32_t> counts(m, 0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			for (const auto& [k, c] : cache.commutator_table[i * m + j]) {
				++counts[k];
			}
		}
	}

	cache.comm_k_ptr.resize(m + 1);
	cache.comm_k_ptr[0] = 0;
	for (uint64_t k = 0; k < m; ++k) {
		cache.comm_k_ptr[k + 1] = cache.comm_k_ptr[k] + counts[k];
	}

	uint32_t nnz = cache.comm_k_ptr[m];
	cache.comm_k_i.resize(nnz);
	cache.comm_k_j.resize(nnz);
	cache.comm_k_val.resize(nnz);

	std::fill(counts.begin(), counts.end(), 0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			for (const auto& [k, c] : cache.commutator_table[i * m + j]) {
				uint32_t pos = cache.comm_k_ptr[k] + counts[k]++;
				cache.comm_k_i[pos] = static_cast<uint32_t>(i);
				cache.comm_k_j[pos] = static_cast<uint32_t>(j);
				cache.comm_k_val[pos] = c;
			}
		}
	}

	// Pre-cast k-grouped coefficients to double
	cache.comm_k_val_d.resize(nnz);
	for (uint32_t idx = 0; idx < nnz; ++idx) {
		cache.comm_k_val_d[idx] = static_cast<double>(cache.comm_k_val[idx]);
	}

	// Build input-grouped commutator table (CSR format, grouped by input index a)
	std::vector<uint32_t> a_counts(m, 0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			for (const auto& [k, c] : cache.commutator_table[i * m + j]) {
				a_counts[i]++;
				a_counts[j]++;
			}
		}
	}

	cache.comm_a_ptr.resize(m + 1);
	cache.comm_a_ptr[0] = 0;
	for (uint64_t a = 0; a < m; ++a) {
		cache.comm_a_ptr[a + 1] = cache.comm_a_ptr[a] + a_counts[a];
	}

	uint32_t a_nnz = cache.comm_a_ptr[m];
	cache.comm_a_k.resize(a_nnz);
	cache.comm_a_partner.resize(a_nnz);
	cache.comm_a_signed_c.resize(a_nnz);

	std::fill(a_counts.begin(), a_counts.end(), 0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			for (const auto& [k, c] : cache.commutator_table[i * m + j]) {
				uint32_t pos_i = cache.comm_a_ptr[i] + a_counts[i]++;
				cache.comm_a_k[pos_i] = static_cast<uint32_t>(k);
				cache.comm_a_partner[pos_i] = static_cast<uint32_t>(j);
				cache.comm_a_signed_c[pos_i] = c;

				uint32_t pos_j = cache.comm_a_ptr[j] + a_counts[j]++;
				cache.comm_a_k[pos_j] = static_cast<uint32_t>(k);
				cache.comm_a_partner[pos_j] = static_cast<uint32_t>(i);
				cache.comm_a_signed_c[pos_j] = -c;
			}
		}
	}

	// Build pair-grouped commutator table (CSR format, grouped by (i,j) pair)
	// Iterate non-empty pairs and flatten their (k, c) entries
	cache.n_pairs = 0;
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			if (!cache.commutator_table[i * m + j].empty()) {
				cache.n_pairs++;
			}
		}
	}

	cache.comm_ij_i.resize(cache.n_pairs);
	cache.comm_ij_j.resize(cache.n_pairs);
	cache.comm_ij_ptr.resize(cache.n_pairs + 1);
	cache.comm_ij_k.resize(nnz);
	cache.comm_ij_c.resize(nnz);

	uint32_t pair_idx = 0;
	uint32_t entry_idx = 0;
	cache.comm_ij_ptr[0] = 0;
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			const auto& entries = cache.commutator_table[i * m + j];
			if (entries.empty()) continue;
			cache.comm_ij_i[pair_idx] = static_cast<uint32_t>(i);
			cache.comm_ij_j[pair_idx] = static_cast<uint32_t>(j);
			for (const auto& [k, c] : entries) {
				cache.comm_ij_k[entry_idx] = static_cast<uint32_t>(k);
				cache.comm_ij_c[entry_idx] = static_cast<double>(c);
				entry_idx++;
			}
			cache.comm_ij_ptr[pair_idx + 1] = entry_idx;
			pair_idx++;
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
	const T* __restrict v1, const T* __restrict v2, T* __restrict result, uint64_t m,
	const std::vector<SparseVec>& commutator_table
) {
	std::memset(result, 0, m * sizeof(T));

	// Gather non-zero indices to avoid O(m^2) scan over zero pairs
	std::vector<uint64_t> nz1_buf, nz2_buf;
	nz1_buf.reserve(m);
	nz2_buf.reserve(m);
	for (uint64_t i = 0; i < m; ++i) {
		if (v1[i] != static_cast<T>(0)) nz1_buf.push_back(i);
	}
	for (uint64_t j = 0; j < m; ++j) {
		if (v2[j] != static_cast<T>(0)) nz2_buf.push_back(j);
	}

	for (uint64_t ii = 0; ii < nz1_buf.size(); ++ii) {
		uint64_t i = nz1_buf[ii];
		T vi = v1[i];
		for (uint64_t jj = 0; jj < nz2_buf.size(); ++jj) {
			uint64_t j = nz2_buf[jj];
			if (i == j) continue;
			T prod = vi * v2[j];
			if (i < j) {
				for (const auto& [k, c] : commutator_table[i * m + j]) {
					result[k] += prod * static_cast<T>(c);
				}
			}
			else {
				for (const auto& [k, c] : commutator_table[j * m + i]) {
					result[k] -= prod * static_cast<T>(c);
				}
			}
		}
	}
}

// ========================================================================
// log_sig_combine_: BCH evaluation with memoized bracket subtrees
// ========================================================================

// Internal implementation using transposed commutator table (CSR format).
// memo: m2 * m elements (one m-length slot per BCH index).
// Uses output-grouped iteration: for each output element k, gather all
// contributing (i,j) pairs from the CSR table. This eliminates empty-pair
// overhead, removes per-bracket vector allocations, and produces sequential
// output writes.
template<std::floating_point T>
void log_sig_combine_impl_(
	const T* __restrict log_sig1, const T* __restrict log_sig2, T* __restrict out,
	const BchCache& cache, T* memo
) {
	uint64_t m = cache.m;
	uint64_t m2 = cache.bch_coefficients.size();

	for (uint64_t i = 0; i < m; ++i) {
		out[i] = log_sig1[i] + log_sig2[i];
	}

	if (m2 <= 2) return;

	// Initialize leaves: memo[0] = log_sig1, memo[1] = log_sig2
	std::memcpy(memo, log_sig1, m * sizeof(T));
	std::memcpy(memo + m, log_sig2, m * sizeof(T));

	const uint32_t* k_ptr = cache.comm_k_ptr.data();
	const uint32_t* k_i = cache.comm_k_i.data();
	const uint32_t* k_j = cache.comm_k_j.data();
	const double* k_val_d = cache.comm_k_val_d.data();

	// Forward pass: BCH indices are topologically sorted (children < parent),
	// so memo[lf] and memo[rf] are always already computed when we reach w.
	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const T* v1 = memo + lf * m;
		const T* v2 = memo + rf * m;
		T* result = memo + w * m;
		const T c_w = static_cast<T>(cache.bch_coefficients[w]);

		// Fused lie_bracket + output accumulation via k-grouped commutator table
		for (uint64_t k = 0; k < m; ++k) {
			T sum = T(0);
			const uint32_t start = k_ptr[k];
			const uint32_t end = k_ptr[k + 1];
			for (uint32_t idx = start; idx < end; ++idx) {
				sum += static_cast<T>(k_val_d[idx]) * (v1[k_i[idx]] * v2[k_j[idx]] - v1[k_j[idx]] * v2[k_i[idx]]);
			}
			result[k] = sum;
			if (c_w != T(0)) out[k] += c_w * sum;
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

	uint64_t m2 = cache.bch_coefficients.size();
	std::vector<T> memo(m2 * m);
	log_sig_combine_impl_<T>(log_sig1, log_sig2, out, cache, memo.data());
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

	uint64_t m2 = cache.bch_coefficients.size();

	if (n_jobs != 1) {
		auto func = [&](const T* ls1, const T* ls2, T* o) {
			thread_local std::vector<T> tl_memo;
			tl_memo.resize(m2 * m);
			log_sig_combine_impl_<T>(ls1, ls2, o, cache, tl_memo.data());
		};
		multi_threaded_batch_2<const T, const T, T>(func, log_sig1, log_sig2, out, batch_size, m, m, m, n_jobs);
	}
	else {
		std::vector<T> memo(m2 * m);
		for (uint64_t i = 0; i < batch_size; ++i) {
			log_sig_combine_impl_<T>(log_sig1 + i * m, log_sig2 + i * m, out + i * m,
				cache, memo.data());
		}
	}
}

// ========================================================================
// log_sig_combine_backprop_: backward pass through BCH
// ========================================================================

template<std::floating_point T>
void log_sig_combine_backprop_impl_(
	const T* __restrict d_out, T* __restrict d_ls1, T* __restrict d_ls2,
	const T* __restrict ls1, const T* __restrict ls2,
	const BchCache& cache, T* workspace
) {
	uint64_t m = cache.m;
	uint64_t m2 = cache.bch_coefficients.size();

	// d_ls1 = d_out, d_ls2 = d_out (from the addition)
	std::memcpy(d_ls1, d_out, m * sizeof(T));
	std::memcpy(d_ls2, d_out, m * sizeof(T));

	if (m2 <= 2) return;

	// workspace layout: memo[m2 * m] then d_memo[m2 * m]
	T* memo = workspace;
	T* d_memo = workspace + m2 * m;

	// Recompute forward memo using pair-grouped table
	std::memcpy(memo, ls1, m * sizeof(T));
	std::memcpy(memo + m, ls2, m * sizeof(T));

	const uint32_t* ij_i = cache.comm_ij_i.data();
	const uint32_t* ij_j = cache.comm_ij_j.data();
	const uint32_t* ij_ptr = cache.comm_ij_ptr.data();
	const uint32_t* ij_k = cache.comm_ij_k.data();
	const double* ij_c = cache.comm_ij_c.data();
	const uint32_t n_pairs = cache.n_pairs;

	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const T* v1 = memo + lf * m;
		const T* v2 = memo + rf * m;
		T* result = memo + w * m;
		std::memset(result, 0, m * sizeof(T));

		for (uint32_t p = 0; p < n_pairs; ++p) {
			const uint32_t i = ij_i[p];
			const uint32_t j = ij_j[p];
			const T prod = v1[i] * v2[j] - v1[j] * v2[i];
			if (prod == T(0)) continue;
			const uint32_t start = ij_ptr[p];
			const uint32_t end = ij_ptr[p + 1];
			for (uint32_t idx = start; idx < end; ++idx) {
				result[ij_k[idx]] += static_cast<T>(ij_c[idx]) * prod;
			}
		}
	}

	// Initialize d_memo: d_memo[w] = c_w * d_out for w >= 2, zero for leaves
	std::memset(d_memo, 0, 2 * m * sizeof(T));
	for (uint64_t w = 2; w < m2; ++w) {
		const T c_w = static_cast<T>(cache.bch_coefficients[w]);
		T* __restrict dm = d_memo + w * m;
		if (c_w != T(0)) {
			for (uint64_t k = 0; k < m; ++k) {
				dm[k] = c_w * d_out[k];
			}
		}
		else {
			std::memset(dm, 0, m * sizeof(T));
		}
	}

	// Reverse BCH loop using pair-grouped table
	// For each (i,j) pair, compute S = sum_k c * dm_w[k], then scatter 4 updates
	for (uint64_t w = m2 - 1; w >= 2; --w) {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const T* v1 = memo + lf * m;
		const T* v2 = memo + rf * m;
		const T* dm_w = d_memo + w * m;
		T* dm_lf = d_memo + lf * m;
		T* dm_rf = d_memo + rf * m;

		for (uint32_t p = 0; p < n_pairs; ++p) {
			// Dot product: S = sum_k c_{ijk} * dm_w[k]
			T S = T(0);
			const uint32_t start = ij_ptr[p];
			const uint32_t end = ij_ptr[p + 1];
			for (uint32_t idx = start; idx < end; ++idx) {
				S += static_cast<T>(ij_c[idx]) * dm_w[ij_k[idx]];
			}
			if (S == T(0)) continue;

			const uint32_t i = ij_i[p];
			const uint32_t j = ij_j[p];
			dm_lf[i] += S * v2[j];
			dm_lf[j] -= S * v2[i];
			dm_rf[j] += S * v1[i];
			dm_rf[i] -= S * v1[j];
		}
	}

	// Accumulate leaf gradients
	for (uint64_t i = 0; i < m; ++i) {
		d_ls1[i] += d_memo[i];
		d_ls2[i] += d_memo[m + i];
	}
}

template<std::floating_point T>
void log_sig_combine_backprop_(
	const T* d_out, T* d_ls1, T* d_ls2,
	const T* ls1, const T* ls2,
	uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine_backprop received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		std::memcpy(d_ls1, d_out, m * sizeof(T));
		std::memcpy(d_ls2, d_out, m * sizeof(T));
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();
	std::vector<T> workspace(2 * m2 * m);
	log_sig_combine_backprop_impl_<T>(d_out, d_ls1, d_ls2, ls1, ls2, cache, workspace.data());
}

template<std::floating_point T>
void batch_log_sig_combine_backprop_(
	const T* d_out, T* d_ls1, T* d_ls2,
	const T* ls1, const T* ls2,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine_backprop received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t i = 0; i < batch_size * m; ++i) {
			d_ls1[i] = d_out[i];
			d_ls2[i] = d_out[i];
		}
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();

	if (n_jobs != 1) {
		auto func = [&](const T* dout, T* dls1, T* dls2, const T* l1, const T* l2) {
			thread_local std::vector<T> tl_workspace;
			tl_workspace.resize(2 * m2 * m);
			log_sig_combine_backprop_impl_<T>(dout, dls1, dls2, l1, l2, cache, tl_workspace.data());
		};
		multi_threaded_batch_4<T>(func, d_out, d_ls1, d_ls2, ls1, ls2, batch_size, m, m, m, m, m, n_jobs);
	}
	else {
		std::vector<T> workspace(2 * m2 * m);
		for (uint64_t i = 0; i < batch_size; ++i) {
			log_sig_combine_backprop_impl_<T>(
				d_out + i * m, d_ls1 + i * m, d_ls2 + i * m,
				ls1 + i * m, ls2 + i * m, cache, workspace.data());
		}
	}
}
