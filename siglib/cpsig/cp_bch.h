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

	if (!cache.linear_forward.empty()) {
		for (uint64_t w = 2; w < m2; ++w) {
			const auto& plan = cache.linear_forward[w];
			const auto [begin, end] = cache.linear_range[w];
			const T* v1 = memo + cache.bch_left_factor[w] * m;
			const T* v2 = memo + cache.bch_right_factor[w] * m;
			T* result = memo + w * m;
			std::fill(result, result + begin, T(0));
			std::fill(result + end, result + m, T(0));
			const T weight = static_cast<T>(cache.bch_coefficients[w]);
			for (uint64_t k = begin; k < end; ++k) {
				T sum = 0;
				for (uint32_t q = plan.row_ptr[k - begin]; q < plan.row_ptr[k - begin + 1]; ++q) {
					const auto& entry = plan.entries[q];
					T term = 0;
					if (entry.orientation & 1) term = v1[entry.i] * v2[entry.j];
					if (entry.orientation & 2) term -= v1[entry.j] * v2[entry.i];
					sum += static_cast<T>(entry.coefficient) * term;
				}
				result[k] = sum;
				if (weight != T(0)) out[k] += weight * sum;
			}
		}
		return;
	}

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

#ifdef VEC
template<std::floating_point T>
struct alignas(vec_batch_bytes) BchBatchValue {
	static constexpr uint64_t width = vec_batch_bytes / sizeof(T);
	T data[width]{};
};

template<std::floating_point T>
void bch_combine_linear_batch_(
	const BchBatchValue<T>* ls1, const BchBatchValue<T>* ls2,
	BchBatchValue<T>* out, const BchCache& cache, BchBatchValue<T>* memo
) {
	const uint64_t m = cache.m;
	std::copy_n(ls1, m, out);
	for (uint64_t k = 0; k < cache.dimension; ++k)
		vec_batch_add_inplace(out[k].data, ls2[k].data);
	std::copy_n(ls1, m, memo);
	std::copy_n(ls2, m, memo + m);
	for (uint64_t w = 2; w < cache.bch_coefficients.size(); ++w) {
		const uint64_t left = cache.bch_left_factor[w], right = cache.bch_right_factor[w];
		const auto* v1 = memo + left * m;
		const auto* v2 = memo + right * m;
		auto* result = memo + w * m;
		const auto [begin, end] = cache.linear_range[w];
		std::fill(result, result + begin, BchBatchValue<T>{});
		std::fill(result + end, result + m, BchBatchValue<T>{});
		const T weight = static_cast<T>(cache.bch_coefficients[w]);
		for (uint64_t k = begin; k < end; ++k) {
			BchBatchValue<T> sum, term, reverse;
			const uint32_t stop = left == 1 || right == 1
				? cache.comm_k_sparse_end[k] : cache.comm_k_ptr[k + 1];
			for (uint32_t q = cache.comm_k_ptr[k]; q < stop; ++q) {
				const uint32_t i = cache.comm_k_i[q], j = cache.comm_k_j[q];
				vec_batch_multiply(term.data, v1[i].data, v2[j].data);
				vec_batch_multiply(reverse.data, v1[j].data, v2[i].data);
				vec_batch_subtract_inplace(term.data, reverse.data);
				vec_batch_scaled_add(sum.data, term.data, static_cast<T>(cache.comm_k_val_d[q]));
			}
			result[k] = sum;
			if (weight != T(0)) vec_batch_scaled_add(out[k].data, sum.data, weight);
		}
	}
}

template<std::floating_point T>
void bch_linear_batch_backprop_(
	const BchBatchValue<T>* d_out, BchBatchValue<T>* d_ls1, BchBatchValue<T>* d_ls2,
	const BchBatchValue<T>* ls1, const BchBatchValue<T>* ls2,
	const BchCache& cache, BchBatchValue<T>* workspace
) {
	const uint64_t m = cache.m, count = cache.bch_coefficients.size();
	auto* memo = workspace;
	auto* d_memo = memo + count * m;
	bch_combine_linear_batch_(ls1, ls2, d_ls1, cache, memo);
	std::fill(d_memo, d_memo + 2 * m, BchBatchValue<T>{});
	for (uint64_t w = 2; w < count; ++w) {
		const T weight = static_cast<T>(cache.bch_coefficients[w]);
		const auto [begin, end] = cache.linear_range[w];
		auto* dm = d_memo + w * m;
		for (uint64_t k = 0; k < m; ++k) {
			if (weight != T(0) && (!cache.prune_linear_backprop || (k >= begin && k < end)))
				vec_batch_scale(dm[k].data, d_out[k].data, weight);
			else dm[k] = {};
		}
	}
	for (uint64_t w = count - 1; w >= 2; --w) {
		const uint64_t left = cache.bch_left_factor[w], right = cache.bch_right_factor[w];
		const auto* v1 = memo + left * m;
		const auto* v2 = memo + right * m;
		const auto* dm = d_memo + w * m;
		auto* d1 = d_memo + left * m;
		auto* d2 = d_memo + right * m;
		const uint64_t begin = cache.prune_linear_backprop ? cache.linear_pair_ptr[w] : 0;
		const uint64_t end = cache.prune_linear_backprop ? cache.linear_pair_ptr[w + 1] : cache.n_pairs;
		for (uint64_t q = begin; q < end; ++q) {
			const uint32_t p = cache.prune_linear_backprop ? cache.linear_pair_idx[q] : static_cast<uint32_t>(q);
			const uint32_t i = cache.comm_ij_i[p], j = cache.comm_ij_j[p];
			BchBatchValue<T> sum;
			for (uint32_t t = cache.comm_ij_ptr[p]; t < cache.comm_ij_ptr[p + 1]; ++t)
				vec_batch_scaled_add(sum.data, dm[cache.comm_ij_k[t]].data, static_cast<T>(cache.comm_ij_c[t]));
			vec_batch_multiply_add(d1[i].data, sum.data, v2[j].data);
			vec_batch_subtract_product(d1[j].data, sum.data, v2[i].data);
			vec_batch_multiply_add(d2[j].data, sum.data, v1[i].data);
			vec_batch_subtract_product(d2[i].data, sum.data, v1[j].data);
		}
	}
	for (uint64_t k = 0; k < m; ++k) {
		vec_batch_add(d_ls1[k].data, d_out[k].data, d_memo[k].data);
		vec_batch_add(d_ls2[k].data, d_out[k].data, d_memo[m + k].data);
	}
}

template<std::floating_point T, bool backward>
void log_sig_from_path_batch_vec_(
	const T* path, const T* cotangent, T* out,
	uint64_t length, uint64_t dimension, const BchCache& cache
) {
	using Value = BchBatchValue<T>;
	constexpr uint64_t width = Value::width;
	const uint64_t m = cache.m, count = cache.bch_coefficients.size();
	thread_local std::vector<Value> scratch;
	scratch.resize((backward ? 2 * count + 7 : count + 3) * m);
	auto* curr = scratch.data();
	auto* prev = curr + m;
	auto* seg = prev + m;
	auto* workspace = seg + m;
	std::fill(curr, curr + m, Value{});
	std::fill(seg, seg + m, Value{});
	for (uint64_t k = 0; k < dimension; ++k)
		for (uint64_t lane = 0; lane < width; ++lane)
			curr[k].data[lane] = path[lane * length * dimension + dimension + k] - path[lane * length * dimension + k];
	for (uint64_t s = 1; s < length - 1; ++s) {
		for (uint64_t k = 0; k < dimension; ++k)
			for (uint64_t lane = 0; lane < width; ++lane)
				seg[k].data[lane] = path[(lane * length + s + 1) * dimension + k] - path[(lane * length + s) * dimension + k];
		bch_combine_linear_batch_(curr, seg, prev, cache, workspace);
		std::swap(curr, prev);
	}
	if constexpr (!backward) {
		for (uint64_t k = 0; k < m; ++k)
			for (uint64_t lane = 0; lane < width; ++lane)
				out[lane * m + k] = curr[k].data[lane];
	}
	else {
		auto* d_acc = workspace + 2 * count * m;
		auto* d1 = d_acc + m;
		auto* d2 = d1 + m;
		auto* neg_seg = d2 + m;
		std::fill(neg_seg, neg_seg + m, Value{});
		for (uint64_t k = 0; k < m; ++k)
			for (uint64_t lane = 0; lane < width; ++lane)
				d_acc[k].data[lane] = cotangent[lane * m + k];
		std::fill(out, out + width * length * dimension, T(0));
		for (uint64_t s = length - 2; s >= 1; --s) {
			for (uint64_t k = 0; k < dimension; ++k) {
				for (uint64_t lane = 0; lane < width; ++lane) {
					const T diff = path[(lane * length + s + 1) * dimension + k] - path[(lane * length + s) * dimension + k];
					seg[k].data[lane] = diff;
					neg_seg[k].data[lane] = -diff;
				}
			}
			bch_combine_linear_batch_(curr, neg_seg, prev, cache, workspace);
			bch_linear_batch_backprop_(d_acc, d1, d2, prev, seg, cache, workspace);
			for (uint64_t k = 0; k < dimension; ++k)
				for (uint64_t lane = 0; lane < width; ++lane) {
					out[(lane * length + s + 1) * dimension + k] += d2[k].data[lane];
					out[(lane * length + s) * dimension + k] -= d2[k].data[lane];
				}
			std::copy_n(d1, m, d_acc);
			std::swap(curr, prev);
		}
		for (uint64_t k = 0; k < dimension; ++k)
			for (uint64_t lane = 0; lane < width; ++lane) {
				out[(lane * length + 1) * dimension + k] += d_acc[k].data[lane];
				out[lane * length * dimension + k] -= d_acc[k].data[lane];
			}
	}
}
#endif
