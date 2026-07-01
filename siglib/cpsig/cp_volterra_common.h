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

// Shared helpers for the diagonal (cp_volterra_signature.cpp) and general
// (cp_volterra_signature_general.cpp) finite state-space Volterra paths: the
// phi_1 scalar function, the multi-index layout/enumeration, and the real
// shuffle-tensor builder (lane-tiled, width L; L == 1 is the plain layout used
// by the general path).

#include <algorithm>
#include <atomic>
#include <complex>
#include <concepts>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "macros.h"

namespace volterra_detail {

// Registry of prepared kernel caches keyed by an opaque handle, one map per
// dtype. get() returns a reference that outlives the lock; entries are only
// removed by free()/clear(), which callers must not race with running
// computations (the Python layer guarantees this).
template<template<class> class CacheT>
struct HandleStore {
	std::mutex mu;
	std::atomic<uint64_t> next_handle{ 1 };
	std::unordered_map<uint64_t, CacheT<float>> map_f;
	std::unordered_map<uint64_t, CacheT<double>> map_d;

	template<std::floating_point T>
	std::unordered_map<uint64_t, CacheT<T>>& map() {
		if constexpr (std::is_same_v<T, float>)
			return map_f;
		else
			return map_d;
	}

	template<std::floating_point T>
	uint64_t store(CacheT<T>&& prepared) {
		const uint64_t handle = next_handle.fetch_add(1);
		if (handle == 0)
			throw std::overflow_error("prepare_volterra_sig handle overflow");
		std::lock_guard lock(mu);
		map<T>().emplace(handle, std::move(prepared));
		return handle;
	}

	template<std::floating_point T>
	const CacheT<T>& get(uint64_t handle) {
		std::lock_guard lock(mu);
		auto& m = map<T>();
		const auto it = m.find(handle);
		if (it == m.end())
			throw std::invalid_argument("volterra_sig received an invalid prepared handle");
		return it->second;
	}

	template<std::floating_point T>
	void free(uint64_t handle) {
		if (handle == 0)
			return;
		std::lock_guard lock(mu);
		map<T>().erase(handle);
	}

	void clear() {
		std::lock_guard lock(mu);
		map_f.clear();
		map_d.clear();
	}
};

template<std::floating_point T>
T phi1_neg(T x) {
	const T ax = std::abs(x);
	if (ax < static_cast<T>(1e-5)) {
		const T x2 = x * x;
		const T x3 = x2 * x;
		const T x4 = x3 * x;
		return static_cast<T>(1) - x / static_cast<T>(2) + x2 / static_cast<T>(6)
			- x3 / static_cast<T>(24) + x4 / static_cast<T>(120);
	}
	return (static_cast<T>(1) - std::exp(-x)) / x;
}

inline uint64_t checked_product(uint64_t a, uint64_t b, const char* name) {
	if (a != 0 && b > UINT64_MAX / a)
		throw std::overflow_error(std::string(name) + " overflow");
	return a * b;
}

inline uint64_t checked_sum(uint64_t a, uint64_t b, const char* name) {
	if (a > UINT64_MAX - b)
		throw std::overflow_error(std::string(name) + " overflow");
	return a + b;
}

inline uint64_t populate_multiindex_tensor_layout(
	uint64_t* mi_off,
	uint64_t* mi_ts,
	const uint64_t* mi_level_index,
	uint64_t target_dimension,
	uint64_t n_levels
) {
	uint64_t off = 0;
	uint64_t ts = 1;
	for (uint64_t L = 0; L < n_levels; ++L) {
		const uint64_t start = mi_level_index[L];
		const uint64_t end = mi_level_index[L + 1];
		if (end < start)
			throw std::invalid_argument("volterra_sig invalid multi-index layout");
		const uint64_t mi_count = end - start;
		mi_off[L] = off;
		mi_ts[L] = ts;
		const uint64_t level_size = checked_product(mi_count, ts, "volterra_sig shuffle workspace");
		off = checked_sum(off, level_size, "volterra_sig shuffle workspace");
		if (L + 1 < n_levels)
			ts = checked_product(ts, target_dimension, "volterra_sig shuffle workspace");
	}
	return off;
}

inline void append_multiindices_for_level(
	uint64_t q,
	uint64_t pos,
	uint64_t remaining,
	std::vector<uint64_t>& current,
	std::vector<uint64_t>& multiindices
) {
	if (pos + 1 == q) {
		current[pos] = remaining;
		multiindices.insert(multiindices.end(), current.begin(), current.end());
		return;
	}

	for (uint64_t value = remaining;; --value) {
		current[pos] = value;
		append_multiindices_for_level(q, pos + 1, remaining - value, current, multiindices);
		if (value == 0)
			break;
	}
}

inline bool multiindex_matches_minus(
	const std::vector<uint64_t>& multiindices,
	uint64_t q,
	uint64_t candidate,
	uint64_t idx,
	uint64_t p
) {
	const uint64_t* lhs = multiindices.data() + candidate * q;
	const uint64_t* rhs = multiindices.data() + idx * q;
	for (uint64_t i = 0; i < q; ++i) {
		const uint64_t target = rhs[i] - (i == p ? 1 : 0);
		if (lhs[i] != target)
			return false;
	}
	return true;
}

inline void populate_multiindex_layout(
	uint64_t q,
	uint64_t max_degree,
	std::vector<uint64_t>& level_index,
	std::vector<uint64_t>& multiindices,
	std::vector<uint64_t>& minus_index
) {
	level_index.resize(max_degree + 2);
	std::vector<uint64_t> current(q, 0);

	for (uint64_t level = 0; level <= max_degree; ++level) {
		level_index[level] = multiindices.size() / q;
		append_multiindices_for_level(q, 0, level, current, multiindices);
	}
	level_index[max_degree + 1] = multiindices.size() / q;

	const uint64_t total = level_index[max_degree + 1];
	minus_index.assign(total * q, UINT64_MAX);
	for (uint64_t level = 1; level <= max_degree; ++level) {
		const uint64_t start = level_index[level];
		const uint64_t end = level_index[level + 1];
		const uint64_t prev_start = level_index[level - 1];
		const uint64_t prev_end = level_index[level];
		for (uint64_t idx = start; idx < end; ++idx) {
			const uint64_t* mi = multiindices.data() + idx * q;
			for (uint64_t p = 0; p < q; ++p) {
				if (mi[p] == 0)
					continue;
				for (uint64_t candidate = prev_start; candidate < prev_end; ++candidate) {
					if (multiindex_matches_minus(multiindices, q, candidate, idx, p)) {
						minus_index[idx * q + p] = candidate;
						break;
					}
				}
			}
		}
	}
}

template<std::floating_point T>
std::complex<T> multiindex_gamma(
	const uint64_t* multiindex,
	uint64_t num_components,
	const std::vector<std::complex<T>>& beta
) {
	std::complex<T> acc(static_cast<T>(1), static_cast<T>(0));
	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t i = 0; i < multiindex[p]; ++i)
			acc *= beta[p];
	}
	return acc;
}

// Builds normalized shuffle tensors by multi-index from the real increments y.
// The trailing lane dimension has width L (lane index innermost and
// contiguous); L == 1 gives the plain per-path layout.
template<std::floating_point T, uint64_t L>
void build_shuffle_tensors(
	const T* y,
	T* shuffle_tensors,
	uint64_t num_components,
	uint64_t target_dimension,
	uint64_t degree,
	const uint64_t* mi_level_index,
	const uint64_t* minus_index,
	const uint64_t* mi_off,
	const uint64_t* mi_ts
) {
	for (uint64_t b = 0; b < L; ++b)
		shuffle_tensors[b] = static_cast<T>(1);
	for (uint64_t word_len = 1; word_len <= degree - 1; ++word_len) {
		const uint64_t ts = mi_ts[word_len];
		const uint64_t ts_prev = mi_ts[word_len - 1];
		const uint64_t start = mi_level_index[word_len];
		const uint64_t end = mi_level_index[word_len + 1];
		const uint64_t prev_start = mi_level_index[word_len - 1];
		for (uint64_t idx = start; idx < end; ++idx) {
			T* dst = shuffle_tensors + (mi_off[word_len] + (idx - start) * ts) * L;
			std::fill(dst, dst + ts * L, static_cast<T>(0));
			for (uint64_t p = 0; p < num_components; ++p) {
				const uint64_t prev = minus_index[idx * num_components + p];
				if (prev == UINT64_MAX)
					continue;
				const T* RESTRICT src = shuffle_tensors + (mi_off[word_len - 1] + (prev - prev_start) * ts_prev) * L;
				const T* yv = y + (p * target_dimension) * L;
				for (uint64_t a = 0; a < target_dimension; ++a) {
					const T* RESTRICT ya = yv + a * L;
					T* dst_a = dst + a * ts_prev * L;
					for (uint64_t c = 0; c < ts_prev; ++c) {
						const T* RESTRICT src_c = src + c * L;
						T* RESTRICT dst_c = dst_a + c * L;
						for (uint64_t b = 0; b < L; ++b)
							dst_c[b] += ya[b] * src_c[b];
					}
				}
			}
		}
	}
}

}  // namespace volterra_detail
