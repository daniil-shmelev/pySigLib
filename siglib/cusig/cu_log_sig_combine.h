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
#include "cupch.h"
#include "cu_log_sig_cache.h"
#include "bch_data.h"

// =========================================================================
// Host-side tensor algebra for commutator table construction
// (ported from cpsig/cp_bch.h)
// =========================================================================

using CuTensorElem = std::unordered_map<uint64_t, int>;

inline void cu_remove_zero_entries_(CuTensorElem& m) {
	for (auto it = m.begin(); it != m.end(); ) {
		if (it->second == 0) it = m.erase(it);
		else ++it;
	}
}

inline CuTensorElem cu_tensor_product_(
	const CuTensorElem& a, uint64_t deg_a,
	const CuTensorElem& b, uint64_t deg_b,
	uint64_t dim, uint64_t max_degree
) {
	if (deg_a + deg_b > max_degree) return {};
	CuTensorElem result;
	for (const auto& [ia, ca] : a) {
		for (const auto& [ib, cb] : b) {
			uint64_t ic = cu_concatenate_idx(ia, ib, deg_b, dim);
			result[ic] += ca * cb;
		}
	}
	cu_remove_zero_entries_(result);
	return result;
}

inline void cu_build_tensor_representations_(
	const std::vector<cu_word>& lyndon_words,
	const std::vector<uint64_t>& left_factor,
	const std::vector<uint64_t>& right_factor,
	uint64_t dim, uint64_t deg,
	std::vector<CuTensorElem>& tensor_reps,
	std::vector<uint64_t>& tensor_degs
) {
	uint64_t m = lyndon_words.size();
	tensor_reps.resize(m);
	tensor_degs.resize(m);

	for (uint64_t i = 0; i < m; ++i) {
		tensor_degs[i] = lyndon_words[i].size();
		if (tensor_degs[i] == 1) {
			uint64_t g = lyndon_words[i][0];
			tensor_reps[i] = { {g + 1, 1} };
		}
		else {
			uint64_t l = left_factor[i];
			uint64_t r = right_factor[i];
			CuTensorElem prod_lr = cu_tensor_product_(tensor_reps[l], tensor_degs[l],
				tensor_reps[r], tensor_degs[r], dim, deg);
			CuTensorElem prod_rl = cu_tensor_product_(tensor_reps[r], tensor_degs[r],
				tensor_reps[l], tensor_degs[l], dim, deg);
			for (const auto& [idx, coef] : prod_rl) {
				prod_lr[idx] -= coef;
			}
			cu_remove_zero_entries_(prod_lr);
			tensor_reps[i] = std::move(prod_lr);
		}
	}
}

inline void cu_compute_factorization_indices_(
	const std::vector<cu_word>& lyndon_words,
	std::vector<uint64_t>& left_factor,
	std::vector<uint64_t>& right_factor,
	uint64_t dimension
) {
	std::unordered_set<cu_word, CuWordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());
	std::unordered_map<cu_word, uint64_t, CuWordHash> word_to_index;
	uint64_t m = lyndon_words.size();
	for (uint64_t i = 0; i < m; ++i) {
		word_to_index[lyndon_words[i]] = i;
	}
	left_factor.assign(m, UINT64_MAX);
	right_factor.assign(m, UINT64_MAX);
	for (uint64_t i = 0; i < m; ++i) {
		if (lyndon_words[i].size() > 1) {
			cu_word v = cu_longest_lyndon_suffix_(lyndon_words[i], lyndon_set);
			cu_word u(lyndon_words[i].begin(), lyndon_words[i].end() - v.size());
			left_factor[i] = word_to_index.at(u);
			right_factor[i] = word_to_index.at(v);
		}
	}
}

// =========================================================================
// Transposed commutator table: grouped by output index k
// For output index k, stores list of (i, j, coefficient) triples where
// i < j and [e_i, e_j] has a non-zero k-th component.
// =========================================================================

inline void cu_build_transposed_commutator_table_(
	uint64_t dimension, uint64_t degree,
	std::vector<uint32_t>& k_ptr,
	std::vector<uint32_t>& k_i,
	std::vector<uint32_t>& k_j,
	std::vector<int>& k_val,
	std::vector<uint64_t>& degree_offsets,
	uint64_t& out_m
) {
	std::vector<cu_word> lyndon_words = cu_all_lyndon_words(dimension, degree);
	uint64_t m = lyndon_words.size();
	out_m = m;
	degree_offsets.assign(degree + 1, 0);
	for (const auto& word : lyndon_words)
		++degree_offsets[word.size()];
	for (uint64_t n = 1; n <= degree; ++n)
		degree_offsets[n] += degree_offsets[n - 1];

	std::vector<uint64_t> left_factor, right_factor;
	cu_compute_factorization_indices_(lyndon_words, left_factor, right_factor, dimension);

	std::vector<CuTensorElem> tensor_reps;
	std::vector<uint64_t> tensor_degs;
	cu_build_tensor_representations_(lyndon_words, left_factor, right_factor,
		dimension, degree, tensor_reps, tensor_degs);

	// Compute Lyndon indices directly from words we already have
	std::vector<uint64_t> lyndon_idx(m);
	for (uint64_t i = 0; i < m; ++i) {
		lyndon_idx[i] = cu_word_to_idx(lyndon_words[i], dimension);
	}

	CuSparseIntMatrix proj_mat;
	cu_lyndon_proj_matrix(proj_mat, lyndon_words, lyndon_idx, dimension, degree);
	CuSparseIntMatrix inv_proj_mat;
	proj_mat.inverse(inv_proj_mat);

	// Build per-output-k entry lists
	struct CommEntry { uint32_t i; uint32_t j; int val; };
	std::vector<std::vector<CommEntry>> entries_by_k(m);

	// Hoist coords allocation out of O(m^2) loop
	std::vector<double> coords(m);

	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			uint64_t total_deg = tensor_degs[i] + tensor_degs[j];
			if (total_deg > degree) continue;

			CuTensorElem comm = cu_tensor_product_(tensor_reps[i], tensor_degs[i],
				tensor_reps[j], tensor_degs[j], dimension, degree);
			CuTensorElem prod_ji = cu_tensor_product_(tensor_reps[j], tensor_degs[j],
				tensor_reps[i], tensor_degs[i], dimension, degree);
			for (const auto& [idx, coef] : prod_ji) {
				comm[idx] -= coef;
			}

			// Extract Lyndon-word coordinates
			std::fill(coords.begin(), coords.end(), 0.0);
			for (uint64_t k = 0; k < m; ++k) {
				if (lyndon_words[k].size() != total_deg) continue;
				auto it = comm.find(lyndon_idx[k]);
				if (it != comm.end()) {
					coords[k] = static_cast<double>(it->second);
				}
			}

			// Apply P^{-1} in-place (lower triangular, diagonal implicit)
			// Must iterate backwards to respect dependencies
			for (uint64_t k_ = 0; k_ < m; ++k_) {
				uint64_t k = m - k_ - 1;
				for (const auto& e : inv_proj_mat.rows[k]) {
					coords[k] += e.val * coords[e.col];
				}
			}

			for (uint64_t k = 0; k < m; ++k) {
				if (coords[k] != 0.0) {
					int val = static_cast<int>(std::round(coords[k]));
					if (val != 0) {
						entries_by_k[k].push_back({
							static_cast<uint32_t>(i),
							static_cast<uint32_t>(j),
							val
						});
					}
				}
			}
		}
	}

	// Flatten to CSR
	k_ptr.resize(m + 1);
	uint32_t offset = 0;
	for (uint64_t k = 0; k < m; ++k) {
		k_ptr[k] = offset;
		offset += static_cast<uint32_t>(entries_by_k[k].size());
	}
	k_ptr[m] = offset;

	uint32_t nnz = offset;
	k_i.resize(nnz);
	k_j.resize(nnz);
	k_val.resize(nnz);

	uint32_t idx = 0;
	for (uint64_t k = 0; k < m; ++k) {
		for (const auto& e : entries_by_k[k]) {
			k_i[idx] = e.i;
			k_j[idx] = e.j;
			k_val[idx] = e.val;
			++idx;
		}
	}
}

// =========================================================================
// GPU BCH cache
// =========================================================================

struct CUDABchCache {
	double* d_bch_coefficients = nullptr;
	uint64_t* d_bch_left_factor = nullptr;
	uint64_t* d_bch_right_factor = nullptr;
	uint64_t* d_linear_range = nullptr;
	uint64_t m2 = 0;
	uint64_t m = 0;
	uint64_t linear_dense_forward_work = 0;
	uint64_t linear_active_forward_work = 0;
	uint64_t linear_zero_work = 0;

	uint32_t* d_comm_k_ptr = nullptr;
	uint32_t* d_comm_k_i = nullptr;
	uint32_t* d_comm_k_j = nullptr;
	int* d_comm_k_val = nullptr;

	uint32_t* d_comm_a_ptr = nullptr;
	uint32_t* d_comm_a_k = nullptr;
	uint32_t* d_comm_a_partner = nullptr;
	int* d_comm_a_signed_c = nullptr;
	uint64_t* d_linear_a_ptr = nullptr;
	uint32_t* d_linear_a_idx = nullptr;

	~CUDABchCache() {
		if (d_bch_coefficients) cudaFree(d_bch_coefficients);
		if (d_bch_left_factor) cudaFree(d_bch_left_factor);
		if (d_bch_right_factor) cudaFree(d_bch_right_factor);
		if (d_linear_range) cudaFree(d_linear_range);
		if (d_comm_k_ptr) cudaFree(d_comm_k_ptr);
		if (d_comm_k_i) cudaFree(d_comm_k_i);
		if (d_comm_k_j) cudaFree(d_comm_k_j);
		if (d_comm_k_val) cudaFree(d_comm_k_val);
		if (d_comm_a_ptr) cudaFree(d_comm_a_ptr);
		if (d_comm_a_k) cudaFree(d_comm_a_k);
		if (d_comm_a_partner) cudaFree(d_comm_a_partner);
		if (d_comm_a_signed_c) cudaFree(d_comm_a_signed_c);
		if (d_linear_a_ptr) cudaFree(d_linear_a_ptr);
		if (d_linear_a_idx) cudaFree(d_linear_a_idx);
	}

	CUDABchCache(const CUDABchCache&) = delete;
	CUDABchCache& operator=(const CUDABchCache&) = delete;
	CUDABchCache() = default;
	CUDABchCache(CUDABchCache&& o) noexcept
		: d_bch_coefficients(std::exchange(o.d_bch_coefficients, nullptr)),
		  d_bch_left_factor(std::exchange(o.d_bch_left_factor, nullptr)),
		  d_bch_right_factor(std::exchange(o.d_bch_right_factor, nullptr)),
		  d_linear_range(std::exchange(o.d_linear_range, nullptr)),
		  m2(std::exchange(o.m2, 0)), m(std::exchange(o.m, 0)),
		  linear_dense_forward_work(std::exchange(o.linear_dense_forward_work, 0)),
		  linear_active_forward_work(std::exchange(o.linear_active_forward_work, 0)),
		  linear_zero_work(std::exchange(o.linear_zero_work, 0)),
		  d_comm_k_ptr(std::exchange(o.d_comm_k_ptr, nullptr)),
		  d_comm_k_i(std::exchange(o.d_comm_k_i, nullptr)),
		  d_comm_k_j(std::exchange(o.d_comm_k_j, nullptr)),
		  d_comm_k_val(std::exchange(o.d_comm_k_val, nullptr)),
		  d_comm_a_ptr(std::exchange(o.d_comm_a_ptr, nullptr)),
		  d_comm_a_k(std::exchange(o.d_comm_a_k, nullptr)),
		  d_comm_a_partner(std::exchange(o.d_comm_a_partner, nullptr)),
		  d_comm_a_signed_c(std::exchange(o.d_comm_a_signed_c, nullptr)),
		  d_linear_a_ptr(std::exchange(o.d_linear_a_ptr, nullptr)),
		  d_linear_a_idx(std::exchange(o.d_linear_a_idx, nullptr))
	{}
};

// =========================================================================
// Cache management
// =========================================================================

std::unordered_map<
	CuLogSigCacheKey, CUDABchCache, CuLogSigCacheKeyHash
>& get_cuda_bch_cache_map_();
std::mutex& get_cuda_bch_cache_mu_();

inline void prepare_cuda_bch_cache_(uint64_t dimension, uint64_t degree) {
	auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	auto& cache_map = get_cuda_bch_cache_map_();
	std::lock_guard<std::mutex> lock(get_cuda_bch_cache_mu_());
	if (cache_map.find(key) != cache_map.end()) return;

	const BchHardcodedData* hc = get_hardcoded_bch_data(degree);
	if (!hc) {
		throw std::runtime_error("log_sig_combine_cuda: degree > 12 not supported");
	}

	CUDABchCache cache;
	cache.m2 = hc->size;

	// Build transposed commutator table on host
	std::vector<uint32_t> h_k_ptr, h_k_i, h_k_j;
	std::vector<int> h_k_val;
	std::vector<uint64_t> h_degree_offsets;
	cu_build_transposed_commutator_table_(
		dimension, degree, h_k_ptr, h_k_i, h_k_j, h_k_val,
		h_degree_offsets, cache.m);

	CUDA_CHECK(cudaMalloc(&cache.d_bch_coefficients, cache.m2 * sizeof(double)));
	CUDA_CHECK(cudaMemcpy(cache.d_bch_coefficients, hc->coefficients, cache.m2 * sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMalloc(&cache.d_bch_left_factor, cache.m2 * sizeof(uint64_t)));
	CUDA_CHECK(cudaMemcpy(cache.d_bch_left_factor, hc->left_factor, cache.m2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMalloc(&cache.d_bch_right_factor, cache.m2 * sizeof(uint64_t)));
	CUDA_CHECK(cudaMemcpy(cache.d_bch_right_factor, hc->right_factor, cache.m2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

	std::vector<uint64_t> min_degree(cache.m2, 1);
	std::vector<uint64_t> max_degree(cache.m2, degree);
	if (cache.m2 > 1) max_degree[1] = 1;
	for (uint64_t w = 2; w < cache.m2; ++w) {
		const uint64_t lf = hc->left_factor[w];
		const uint64_t rf = hc->right_factor[w];
		min_degree[w] = min_degree[lf] + min_degree[rf];
		max_degree[w] = std::min(degree, max_degree[lf] + max_degree[rf]);
	}
	std::vector<uint64_t> h_linear_range(2 * cache.m2);
	for (uint64_t w = 0; w < cache.m2; ++w) {
		h_linear_range[2 * w] = h_degree_offsets[min_degree[w] - 1];
		h_linear_range[2 * w + 1] = h_degree_offsets[max_degree[w]];
	}
	uint64_t linear_threads = std::min<uint64_t>(64, cache.m);
	linear_threads = std::max<uint64_t>(32, ((linear_threads + 31) / 32) * 32);
	const uint64_t linear_warps = linear_threads / 32;
	uint64_t dense_node_work = 0;
	for (uint64_t warp = 0; warp < linear_warps; ++warp) {
		uint64_t warp_work = 0;
		for (uint64_t start = 32 * warp; start < cache.m; start += linear_threads) {
			uint64_t round_work = 0;
			for (uint64_t k = start; k < std::min(start + 32, cache.m); ++k)
				round_work = std::max<uint64_t>(
					round_work, 1 + h_k_ptr[k + 1] - h_k_ptr[k]);
			warp_work += round_work;
		}
		dense_node_work += warp_work;
	}
	for (uint64_t w = 2; w < cache.m2; ++w) {
		const uint64_t begin = h_linear_range[2 * w];
		const uint64_t end = h_linear_range[2 * w + 1];
		uint64_t active_node_work = 0;
		for (uint64_t warp = 0; warp < linear_warps; ++warp) {
			uint64_t warp_work = 0;
			for (uint64_t start = begin + 32 * warp; start < end; start += linear_threads) {
				uint64_t round_work = 0;
				for (uint64_t k = start; k < std::min(start + 32, end); ++k)
					round_work = std::max<uint64_t>(
						round_work, 1 + h_k_ptr[k + 1] - h_k_ptr[k]);
				warp_work += round_work;
			}
			active_node_work += warp_work;
		}
		cache.linear_active_forward_work += active_node_work;
		cache.linear_zero_work += (begin + 31) / 32 + (cache.m - end + 31) / 32;
	}
	const uint64_t node_count = cache.m2 > 2 ? cache.m2 - 2 : 0;
	cache.linear_dense_forward_work = node_count * dense_node_work;
	CUDA_CHECK(cudaMalloc(&cache.d_linear_range, h_linear_range.size() * sizeof(uint64_t)));
	CUDA_CHECK(cudaMemcpy(cache.d_linear_range, h_linear_range.data(),
		h_linear_range.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

	uint32_t nnz = h_k_ptr.back();

	CUDA_CHECK(cudaMalloc(&cache.d_comm_k_ptr, (cache.m + 1) * sizeof(uint32_t)));
	CUDA_CHECK(cudaMemcpy(cache.d_comm_k_ptr, h_k_ptr.data(), (cache.m + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	if (nnz > 0) {
		CUDA_CHECK(cudaMalloc(&cache.d_comm_k_i, nnz * sizeof(uint32_t)));
		CUDA_CHECK(cudaMemcpy(cache.d_comm_k_i, h_k_i.data(), nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&cache.d_comm_k_j, nnz * sizeof(uint32_t)));
		CUDA_CHECK(cudaMemcpy(cache.d_comm_k_j, h_k_j.data(), nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&cache.d_comm_k_val, nnz * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(cache.d_comm_k_val, h_k_val.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
	}

	struct AEntry { uint32_t k; uint32_t partner; int signed_c; };
	std::vector<std::vector<AEntry>> entries_by_a(cache.m);
	for (uint64_t k = 0; k < cache.m; ++k) {
		for (uint32_t idx = h_k_ptr[k]; idx < h_k_ptr[k + 1]; ++idx) {
			uint32_t i = h_k_i[idx];
			uint32_t j = h_k_j[idx];
			int c = h_k_val[idx];
			entries_by_a[i].push_back({static_cast<uint32_t>(k), j, c});
			entries_by_a[j].push_back({static_cast<uint32_t>(k), i, -c});
		}
	}

	std::vector<uint32_t> h_a_ptr(cache.m + 1);
	uint32_t a_offset = 0;
	for (uint64_t a = 0; a < cache.m; ++a) {
		h_a_ptr[a] = a_offset;
		a_offset += static_cast<uint32_t>(entries_by_a[a].size());
	}
	h_a_ptr[cache.m] = a_offset;

	uint32_t a_nnz = a_offset;
	std::vector<uint32_t> h_a_k(a_nnz), h_a_partner(a_nnz);
	std::vector<int> h_a_signed_c(a_nnz);
	uint32_t a_idx = 0;
	for (uint64_t a = 0; a < cache.m; ++a) {
		for (const auto& e : entries_by_a[a]) {
			h_a_k[a_idx] = e.k;
			h_a_partner[a_idx] = e.partner;
			h_a_signed_c[a_idx] = e.signed_c;
			++a_idx;
		}
	}

	if (a_nnz > (std::numeric_limits<uint32_t>::max() >> 1))
		throw std::overflow_error("CUDA BCH linear reverse plan exceeds uint32_t packing");
	std::vector<uint64_t> h_linear_a_ptr(cache.m2 * cache.m + 1);
	std::vector<uint32_t> h_linear_a_idx;
	for (uint64_t w = 0; w < cache.m2; ++w) {
		for (uint64_t a = 0; a < cache.m; ++a) {
			const uint64_t row = w * cache.m + a;
			h_linear_a_ptr[row] = h_linear_a_idx.size();
			if (w < 2) continue;
			const uint64_t lf = hc->left_factor[w];
			const uint64_t rf = hc->right_factor[w];
			const uint64_t lf_begin = h_linear_range[2 * lf];
			const uint64_t lf_end = h_linear_range[2 * lf + 1];
			const uint64_t rf_begin = h_linear_range[2 * rf];
			const uint64_t rf_end = h_linear_range[2 * rf + 1];
			const bool active_dv1 = a >= lf_begin && a < lf_end;
			const bool active_dv2 = a >= rf_begin && a < rf_end;
			for (uint32_t idx = h_a_ptr[a]; idx < h_a_ptr[a + 1]; ++idx) {
				const uint32_t partner = h_a_partner[idx];
				if (active_dv1 && partner >= rf_begin && partner < rf_end)
					h_linear_a_idx.push_back(idx << 1);
				if (active_dv2 && partner >= lf_begin && partner < lf_end)
					h_linear_a_idx.push_back((idx << 1) | 1);
			}
		}
	}
	h_linear_a_ptr.back() = h_linear_a_idx.size();
	CUDA_CHECK(cudaMalloc(&cache.d_linear_a_ptr, h_linear_a_ptr.size() * sizeof(uint64_t)));
	CUDA_CHECK(cudaMemcpy(cache.d_linear_a_ptr, h_linear_a_ptr.data(),
		h_linear_a_ptr.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
	if (!h_linear_a_idx.empty()) {
		CUDA_CHECK(cudaMalloc(&cache.d_linear_a_idx, h_linear_a_idx.size() * sizeof(uint32_t)));
		CUDA_CHECK(cudaMemcpy(cache.d_linear_a_idx, h_linear_a_idx.data(),
			h_linear_a_idx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	CUDA_CHECK(cudaMalloc(&cache.d_comm_a_ptr, (cache.m + 1) * sizeof(uint32_t)));
	CUDA_CHECK(cudaMemcpy(cache.d_comm_a_ptr, h_a_ptr.data(), (cache.m + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	if (a_nnz > 0) {
		CUDA_CHECK(cudaMalloc(&cache.d_comm_a_k, a_nnz * sizeof(uint32_t)));
		CUDA_CHECK(cudaMemcpy(cache.d_comm_a_k, h_a_k.data(), a_nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&cache.d_comm_a_partner, a_nnz * sizeof(uint32_t)));
		CUDA_CHECK(cudaMemcpy(cache.d_comm_a_partner, h_a_partner.data(), a_nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&cache.d_comm_a_signed_c, a_nnz * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(cache.d_comm_a_signed_c, h_a_signed_c.data(), a_nnz * sizeof(int), cudaMemcpyHostToDevice));
	}

	cache_map.emplace(key, std::move(cache));
}

inline const CUDABchCache& get_cuda_bch_cache_(uint64_t dimension, uint64_t degree) {
	auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	auto& cache_map = get_cuda_bch_cache_map_();
	std::lock_guard<std::mutex> lock(get_cuda_bch_cache_mu_());
	auto it = cache_map.find(key);
	if (it != cache_map.end()) return it->second;
	throw cache_not_found_error(
		"CUDA BCH cache not found - call prepare_log_sig with method=3 and device='cuda' first");
}

inline void clear_cuda_bch_cache_() {
	std::lock_guard<std::mutex> lock(get_cuda_bch_cache_mu_());
	get_cuda_bch_cache_map_().clear();
}
