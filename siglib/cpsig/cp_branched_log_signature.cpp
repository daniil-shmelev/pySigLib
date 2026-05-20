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
#include "../shared/branched_log_cache.h"
#include "multithreading.h"
#include "macros.h"

namespace {
const BranchedLogForestCache& get_cached_branched_log_forest_cache(const BranchedSigCache& cache) {
	static std::unordered_map<
		std::pair<uint64_t, uint64_t>,
		std::unique_ptr<BranchedLogForestCache>,
		PairHash
	> registry;
	static std::shared_mutex mu;

	const std::pair<uint64_t, uint64_t> key{
		cache.dimension,
		cache.max_nodes | (static_cast<uint64_t>(cache.planar) << 63)
	};
	{
		std::shared_lock rlock(mu);
		auto it = registry.find(key);
		if (it != registry.end())
			return *(it->second);
	}
	auto fc = std::make_unique<BranchedLogForestCache>(build_branched_log_forest_cache(cache));
	std::unique_lock wlock(mu);
	auto [ins, _] = registry.try_emplace(key, std::move(fc));
	return *(ins->second);
}

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
	const BranchedLogForestCache& forest_cache
) {
	const uint64_t total_len = cache.total_length;
	const uint64_t num_forests = forest_cache.forest_offsets.size() - 1;

	std::vector<BranchedLogPolyBuild_> h_poly(num_forests);
	for (uint64_t forest_idx = 1; forest_idx < num_forests; ++forest_idx) {
		const uint64_t start = forest_cache.forest_offsets[forest_idx];
		const uint64_t end = forest_cache.forest_offsets[forest_idx + 1];
		std::vector<uint64_t> factors(
			forest_cache.forest_trees.begin() + start,
			forest_cache.forest_trees.begin() + end);
		std::sort(factors.begin(), factors.end());
		h_poly[forest_idx].push_back({ 1.0, std::move(factors) });
	}

	std::vector<std::vector<BranchedLogPolyBuild_>> powers(
		cache.max_nodes + 1,
		std::vector<BranchedLogPolyBuild_>(num_forests));
	if (cache.max_nodes >= 1)
		powers[1] = h_poly;

	for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
		for (uint64_t forest_idx = 0; forest_idx < num_forests; ++forest_idx) {
			BranchedLogPolyMap_ terms;
			const uint64_t start = forest_cache.forest_coprod_offsets[forest_idx];
			const uint64_t end = forest_cache.forest_coprod_offsets[forest_idx + 1];
			for (uint64_t pos = start; pos < end; pos += 2) {
				const uint64_t left = forest_cache.forest_coprod_data[pos];
				const uint64_t right = forest_cache.forest_coprod_data[pos + 1];
				multiply_branched_log_poly_into_(powers[k - 1][left], h_poly[right], terms);
			}
			powers[k][forest_idx] = flatten_branched_log_poly_(std::move(terms));
		}
	}

	BranchedLogPolyCache_ out;
	for (uint64_t flat_idx = 1; flat_idx < total_len; ++flat_idx) {
		BranchedLogPolyMap_ terms;
		const uint64_t forest_idx = forest_cache.single_tree_forest[flat_idx];
		for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
			const double coeff = (k % 2 == 0) ? -1.0 / static_cast<double>(k) : 1.0 / static_cast<double>(k);
			for (const auto& term : powers[k][forest_idx])
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
	const BranchedSigCache& cache,
	const BranchedLogForestCache& forest_cache
) {
	static std::unordered_map<
		std::pair<uint64_t, uint64_t>,
		std::unique_ptr<BranchedLogPolyCache_>,
		PairHash
	> registry;
	static std::shared_mutex mu;

	const std::pair<uint64_t, uint64_t> key{
		cache.dimension,
		cache.max_nodes | (static_cast<uint64_t>(cache.planar) << 63)
	};
	{
		std::shared_lock rlock(mu);
		auto it = registry.find(key);
		if (it != registry.end())
			return *(it->second);
	}
	auto pc = std::make_unique<BranchedLogPolyCache_>(
		build_branched_log_poly_cache_(cache, forest_cache));
	std::unique_lock wlock(mu);
	auto [ins, _] = registry.try_emplace(key, std::move(pc));
	return *(ins->second);
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

	const auto& forest_cache = get_cached_branched_log_forest_cache(cache);
	const auto& poly_cache = get_cached_branched_log_poly_cache_(cache, forest_cache);
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

	const auto& forest_cache = get_cached_branched_log_forest_cache(cache);
	const auto& poly_cache = get_cached_branched_log_poly_cache_(cache, forest_cache);
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
}  // namespace


template<std::floating_point T>
void branched_sig_to_log_sig_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs,
	bool planar,
	bool scalar_term
) {
	if (scalar_term) {
		branched_sig_to_log_sig_<T, true>(bsig, out, batch_size, dimension, max_nodes, n_jobs, planar);
	} else {
		branched_sig_to_log_sig_<T, false>(bsig, out, batch_size, dimension, max_nodes, n_jobs, planar);
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
	int n_jobs,
	bool planar,
	bool scalar_term
) {
	if (scalar_term) {
		branched_sig_to_log_sig_backprop_<T, true>(bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar);
	} else {
		branched_sig_to_log_sig_backprop_<T, false>(bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar);
	}
}



extern "C" {

	CPSIG_API int branched_sig_to_log_sig_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_<float>(bsig, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_<double>(bsig, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_backprop_<float>(bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_backprop_<double>(bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

}
