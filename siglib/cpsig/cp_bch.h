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
#include "cp_sig_combine.h"
#include "cp_tensor_log.h"
#include "preparation/log_sig/bch_data.h"
#include "../shared/preparation/log_sig/lyndon_words.h"
#include "../shared/preparation/log_sig/bch_cache.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

// ========================================================================
// Commutator table via tensor algebra
// ========================================================================

// ========================================================================
// BCH cache management
// ========================================================================

inline void prepare_bch_cache(
	uint64_t dimension, uint64_t degree, bool use_disk = false
) {
	if (degree > BCH_MAX_HARDCODED_DEGREE)
		throw std::invalid_argument(
			"BCH methods support truncation degrees at most 20");
	prepare_basis_cache(dimension, degree, 2, use_disk);
	LogSigCache& log_cache = get_log_sig_cache_mutable(
		dimension, degree, 2);
	if (log_cache.has_bch())
		return;

	log_cache.set_bch(make_standard_bch_cache(
		dimension, degree, log_cache.basis(2)));
}

inline const BchCache& get_bch_cache(uint64_t dimension, uint64_t degree) {
	return get_log_sig_cache(dimension, degree, 3).bch();
}

inline void clear_bch_cache() {
	// BCH data is owned by the LogSigCache registry.
}

template<typename Function>
FORCE_INLINE void for_each_bch_node_(
	const BchCache& cache, Function&& function
) {
	if (cache.all_bch_nodes_live) {
		for (uint64_t node = 2; node < cache.bch_coefficients.size(); ++node)
			function(node);
	}
	else {
		for (uint32_t node : cache.live_bch_nodes)
			function(node);
	}
}

template<typename Function>
FORCE_INLINE void for_each_bch_node_reverse_(
	const BchCache& cache, Function&& function
) {
	if (cache.all_bch_nodes_live) {
		for (uint64_t node = cache.bch_coefficients.size(); node-- > 2;)
			function(node);
	}
	else {
		for (auto node = cache.live_bch_nodes.rbegin();
			node != cache.live_bch_nodes.rend(); ++node)
			function(*node);
	}
}

// ========================================================================
// Lie bracket of two dense vectors using the commutator table
// ========================================================================

template<std::floating_point T>
void lie_bracket(
	const T* RESTRICT v1, const T* RESTRICT v2, T* RESTRICT result, uint64_t m,
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
// bch_combine_impl_: BCH evaluation with memoized bracket subtrees
// ========================================================================

// Internal implementation using transposed commutator table (CSR format).
// memo: m2 * m elements (one m-length slot per BCH index).
// Uses output-grouped iteration: for each output element k, gather all
// contributing (i,j) pairs from the CSR table. This eliminates empty-pair
// overhead, removes per-bracket vector allocations, and produces sequential
// output writes.
template<std::floating_point T>
void bch_combine_impl_(
	const T* RESTRICT ls1, const T* RESTRICT ls2, T* RESTRICT out,
	const BchCache& cache, T* memo
) {
	uint64_t m = cache.m;
	uint64_t m2 = cache.bch_coefficients.size();

	for (uint64_t i = 0; i < m; ++i) {
		out[i] = ls1[i] + ls2[i];
	}

	if (m2 <= 2) return;

	std::memcpy(memo, ls1, m * sizeof(T));
	std::memcpy(memo + m, ls2, m * sizeof(T));

	const uint32_t* k_ptr = cache.comm_k_ptr.data();
	const uint32_t* k_i = cache.comm_k_i.data();
	const uint32_t* k_j = cache.comm_k_j.data();
	const double* k_val_d = cache.comm_k_val_d.data();

	// Forward pass: BCH indices are topologically sorted (children < parent),
	// so memo[lf] and memo[rf] are always already computed when we reach w.
	for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
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
	});
}

// ========================================================================
// bch_combine_linear_impl_: BCH where the second operand is a
// linear log-sig (only first dim entries nonzero). Skips commutator pairs
// where both indices >= dim when either bracket operand is node 1.
// ========================================================================

template<std::floating_point T>
void bch_combine_linear_impl_(
	const T* RESTRICT ls1, const T* RESTRICT ls2, T* RESTRICT out,
	const BchCache& cache, T* memo
) {
	uint64_t m = cache.m;
	uint64_t dim = cache.dimension;
	uint64_t m2 = cache.bch_coefficients.size();

	std::memcpy(out, ls1, m * sizeof(T));
	if (cache.linear_input_prefix) {
		for (uint64_t i = 0; i < dim; ++i)
			out[i] += ls2[i];
	}
	else {
		for (uint32_t i : cache.linear_input_idx)
			out[i] += ls2[i];
	}

	if (m2 <= 2) return;

	std::memcpy(memo, ls1, m * sizeof(T));
	if (ls2 != memo + m)
		std::memcpy(memo + m, ls2, m * sizeof(T));

	const uint32_t* k_ptr = cache.comm_k_ptr.data();
	const uint32_t* k_i = cache.comm_k_i.data();
	const uint32_t* k_j = cache.comm_k_j.data();
	const double* k_val_d = cache.comm_k_val_d.data();

	for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const T* v1 = memo + lf * m;
		const T* v2 = memo + rf * m;
		T* result = memo + w * m;
		const T c_w = static_cast<T>(cache.bch_coefficients[w]);
		const auto [begin, end] = cache.linear_range[w];
		std::fill(result, result + begin, T(0));
		std::fill(result + end, result + m, T(0));

		// When one operand is node 1 (the sparse displacement), use
		// precomputed sparse_end bounds - no per-entry branch needed.
		const bool sparse = (rf == 1 || lf == 1);

		for (uint64_t k = begin; k < end; ++k) {
			T sum = T(0);
			if (sparse && !cache.linear_input_prefix) {
				for (uint32_t q = cache.comm_k_sparse_ptr[k];
					q < cache.comm_k_sparse_ptr[k + 1]; ++q) {
					const uint32_t idx = cache.comm_k_sparse_idx[q];
					sum += static_cast<T>(k_val_d[idx])
						* (v1[k_i[idx]] * v2[k_j[idx]]
							- v1[k_j[idx]] * v2[k_i[idx]]);
				}
			}
			else {
				const uint32_t start = k_ptr[k];
				const uint32_t stop = sparse
					? cache.comm_k_sparse_end[k] : k_ptr[k + 1];
				for (uint32_t idx = start; idx < stop; ++idx) {
					sum += static_cast<T>(k_val_d[idx])
						* (v1[k_i[idx]] * v2[k_j[idx]]
							- v1[k_j[idx]] * v2[k_i[idx]]);
				}
			}
			result[k] = sum;
			if (c_w != T(0)) out[k] += c_w * sum;
		}
	});
}

// 4-wide BCH combination (works on both AVX2 and NEON).
#ifdef VEC
inline void bch_combine_impl_x4_(
	const double* RESTRICT ls1, const double* RESTRICT ls2, double* RESTRICT out,
	const BchCache& cache, double* memo
) {
	uint64_t m = cache.m;
	uint64_t m2 = cache.bch_coefficients.size();

	vec4_add(out, ls1, ls2, m);
	if (m2 <= 2) return;

	std::memcpy(memo, ls1, m * 4 * sizeof(double));
	std::memcpy(memo + m * 4, ls2, m * 4 * sizeof(double));

	const uint32_t* k_ptr = cache.comm_k_ptr.data();
	const uint32_t* k_i = cache.comm_k_i.data();
	const uint32_t* k_j = cache.comm_k_j.data();
	const double* k_val_d = cache.comm_k_val_d.data();

	for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const double* v1 = memo + lf * m * 4;
		const double* v2 = memo + rf * m * 4;
		double* result = memo + w * m * 4;
		const double c_w = cache.bch_coefficients[w];

		for (uint64_t k = 0; k < m; ++k) {
			vec4_commutator_accum(&result[k * 4], v1, v2, k_i, k_j, k_val_d, k_ptr[k], k_ptr[k + 1]);
			if (c_w != 0.0)
				vec4_fmadd(&out[k * 4], &result[k * 4], c_w, 1);
		}
	});
}
// 4-wide BCH with a degree-one second input.
inline void bch_combine_linear_impl_x4_(
	const double* RESTRICT ls1, const double* RESTRICT ls2, double* RESTRICT out,
	const BchCache& cache, double* memo
) {
	uint64_t m = cache.m;
	uint64_t dim = cache.dimension;
	uint64_t m2 = cache.bch_coefficients.size();

	vec4_add(out, ls1, ls2, dim);
	std::memcpy(&out[dim * 4], &ls1[dim * 4], (m - dim) * 4 * sizeof(double));
	if (m2 <= 2) return;

	std::memcpy(memo, ls1, m * 4 * sizeof(double));
	std::memcpy(memo + m * 4, ls2, m * 4 * sizeof(double));

	const uint32_t* k_ptr = cache.comm_k_ptr.data();
	const uint32_t* k_i = cache.comm_k_i.data();
	const uint32_t* k_j = cache.comm_k_j.data();
	const double* k_val_d = cache.comm_k_val_d.data();

	for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const double* v1 = memo + lf * m * 4;
		const double* v2 = memo + rf * m * 4;
		double* result = memo + w * m * 4;
		const double c_w = cache.bch_coefficients[w];
		const auto [begin, end] = cache.linear_range[w];
		std::memset(result, 0, begin * 4 * sizeof(double));
		std::memset(result + end * 4, 0, (m - end) * 4 * sizeof(double));

		for (uint64_t k = begin; k < end; ++k) {
			vec4_commutator_accum(&result[k * 4], v1, v2, k_i, k_j, k_val_d,
				k_ptr[k], k_ptr[k + 1]);
			if (c_w != 0.0)
				vec4_fmadd(&out[k * 4], &result[k * 4], c_w, 1);
		}
	});
}
// 4-wide BCH backprop for a degree-one second input.
template<bool prune_linear>
inline void bch_combine_linear_backprop_impl_x4_(
	const double* RESTRICT d_out, double* RESTRICT d_ls1, double* RESTRICT d_ls2,
	const double* RESTRICT ls1, const double* RESTRICT ls2,
	const BchCache& cache, double* workspace
) {
	uint64_t m = cache.m;
	uint64_t m2 = cache.bch_coefficients.size();

	if (m2 <= 2) {
		std::memcpy(d_ls1, d_out, m * 4 * sizeof(double));
		std::memcpy(d_ls2, d_out, m * 4 * sizeof(double));
		return;
	}

	double* memo = workspace;
	double* d_memo = workspace + m2 * m * 4;

	std::memcpy(memo, ls1, m * 4 * sizeof(double));
	std::memcpy(memo + m * 4, ls2, m * 4 * sizeof(double));

	const uint32_t* k_ptr = cache.comm_k_ptr.data();
	const uint32_t* k_i = cache.comm_k_i.data();
	const uint32_t* k_j = cache.comm_k_j.data();
	const double* k_c = cache.comm_k_val_d.data();
	const uint32_t* ij_i = cache.comm_ij_i.data();
	const uint32_t* ij_j = cache.comm_ij_j.data();
	const uint32_t* ij_ptr = cache.comm_ij_ptr.data();
	const uint32_t* ij_k = cache.comm_ij_k.data();
	const double* ij_c = cache.comm_ij_c.data();
	const uint32_t n_pairs = cache.n_pairs;

	// Forward recompute (output-grouped, 4-wide)
	for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const double* v1 = memo + lf * m * 4;
		const double* v2 = memo + rf * m * 4;
		double* result = memo + w * m * 4;
		const auto [begin, end] = cache.linear_range[w];
		std::memset(result, 0, begin * 4 * sizeof(double));
		std::memset(result + end * 4, 0, (m - end) * 4 * sizeof(double));
		for (uint64_t k = begin; k < end; ++k)
			vec4_commutator_accum(&result[k * 4], v1, v2, k_i, k_j, k_c,
				k_ptr[k], k_ptr[k + 1]);
	});

	// d_memo init
	std::memset(d_memo, 0, 2 * m * 4 * sizeof(double));
	for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const double c_w = cache.bch_coefficients[w];
		double* dm = d_memo + w * m * 4;
		if constexpr (prune_linear) {
			const auto [begin, end] = cache.linear_range[w];
			std::memset(dm, 0, begin * 4 * sizeof(double));
			if (c_w != 0.0)
				vec4_scale(dm + begin * 4, d_out + begin * 4, c_w, end - begin);
			else
				std::memset(dm + begin * 4, 0, (end - begin) * 4 * sizeof(double));
			std::memset(dm + end * 4, 0, (m - end) * 4 * sizeof(double));
		}
		else if (c_w != 0.0) {
			vec4_scale(dm, d_out, c_w, m);
		}
		else {
			std::memset(dm, 0, m * 4 * sizeof(double));
		}
	});

	// Reverse BCH (pair-grouped, 4-wide)
	for_each_bch_node_reverse_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const double* v1 = memo + lf * m * 4;
		const double* v2 = memo + rf * m * 4;
		const double* dm_w = d_memo + w * m * 4;
		double* dm_lf = d_memo + lf * m * 4;
		double* dm_rf = d_memo + rf * m * 4;

		if constexpr (prune_linear) {
			for (uint64_t q = cache.linear_pair_ptr[w]; q < cache.linear_pair_ptr[w + 1]; ++q) {
				const uint32_t p = cache.linear_pair_idx[q];
				vec4_bracket_grad(dm_lf, dm_rf, dm_w, v1, v2, ij_i[p], ij_j[p],
					ij_k, ij_c, ij_ptr[p], ij_ptr[p + 1]);
			}
		}
		else {
			for (uint32_t p = 0; p < n_pairs; ++p)
				vec4_bracket_grad(dm_lf, dm_rf, dm_w, v1, v2, ij_i[p], ij_j[p],
					ij_k, ij_c, ij_ptr[p], ij_ptr[p + 1]);
		}
	});

	// Add the direct gradient to the accumulated BCH leaf gradients.
	vec4_add(d_ls1, d_out, d_memo, m);
	vec4_add(d_ls2, d_out, d_memo + m * 4, m);
}
#endif // VEC

// ========================================================================
// bch_combine_backprop_impl_: backward pass through BCH
// ========================================================================

template<std::floating_point T, bool linear_input = false, bool prune_linear = false>
void bch_combine_backprop_impl_(
	const T* RESTRICT d_out, T* RESTRICT d_ls1, T* RESTRICT d_ls2,
	const T* RESTRICT ls1, const T* RESTRICT ls2,
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

	// Linear inputs use the range-aware combiner; otherwise recompute below.
	std::memcpy(memo, ls1, m * sizeof(T));
	std::memcpy(memo + m, ls2, m * sizeof(T));

	const uint32_t* ij_i = cache.comm_ij_i.data();
	const uint32_t* ij_j = cache.comm_ij_j.data();
	const uint32_t* ij_ptr = cache.comm_ij_ptr.data();
	const uint32_t* ij_k = cache.comm_ij_k.data();
	const double* ij_c = cache.comm_ij_c.data();
	const uint32_t n_pairs = cache.n_pairs;

	if constexpr (linear_input)
		bch_combine_linear_impl_(ls1, ls2, d_memo, cache, memo);
	if constexpr (!linear_input) {
		for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
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
		});
	}

	// Initialize d_memo: d_memo[w] = c_w * d_out for w >= 2, zero for leaves
	std::memset(d_memo, 0, 2 * m * sizeof(T));
	for_each_bch_node_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const T c_w = static_cast<T>(cache.bch_coefficients[w]);
		T* RESTRICT dm = d_memo + w * m;
		if constexpr (linear_input && prune_linear) {
			const auto [begin, end] = cache.linear_range[w];
			std::memset(dm, 0, begin * sizeof(T));
			if (c_w != T(0)) {
				for (uint64_t k = begin; k < end; ++k)
					dm[k] = c_w * d_out[k];
			}
			else {
				std::memset(dm + begin, 0, (end - begin) * sizeof(T));
			}
			std::memset(dm + end, 0, (m - end) * sizeof(T));
		}
		else {
			if (c_w != T(0)) {
				for (uint64_t k = 0; k < m; ++k)
					dm[k] = c_w * d_out[k];
			}
			else {
				std::memset(dm, 0, m * sizeof(T));
			}
		}
	});

	// Reverse BCH loop using pair-grouped table
	// For each (i,j) pair, compute S = sum_k c * dm_w[k], then scatter 4 updates
	for_each_bch_node_reverse_(cache, [&](uint64_t w) FORCE_INLINE_LAMBDA {
		const uint64_t lf = cache.bch_left_factor[w];
		const uint64_t rf = cache.bch_right_factor[w];
		const T* v1 = memo + lf * m;
		const T* v2 = memo + rf * m;
		const T* dm_w = d_memo + w * m;
		T* dm_lf = d_memo + lf * m;
		T* dm_rf = d_memo + rf * m;

		if constexpr (linear_input && prune_linear) {
			for (uint64_t q = cache.linear_pair_ptr[w]; q < cache.linear_pair_ptr[w + 1]; ++q) {
				const uint32_t p = cache.linear_pair_idx[q];
				const uint32_t i = ij_i[p];
				const uint32_t j = ij_j[p];
				T S = T(0);
				for (uint32_t idx = ij_ptr[p]; idx < ij_ptr[p + 1]; ++idx)
					S += static_cast<T>(ij_c[idx]) * dm_w[ij_k[idx]];
				if (S == T(0)) continue;
				dm_lf[i] += S * v2[j];
				dm_lf[j] -= S * v2[i];
				dm_rf[j] += S * v1[i];
				dm_rf[i] -= S * v1[j];
			}
		}
		else {
			for (uint32_t p = 0; p < n_pairs; ++p) {
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
	});

	// Accumulate leaf gradients
	for (uint64_t i = 0; i < m; ++i) {
		d_ls1[i] += d_memo[i];
		d_ls2[i] += d_memo[m + i];
	}
}
