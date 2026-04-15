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

// Host-side expansion matrix computation for logsig_to_sig methods 1/2.
// Pure C++ — no CUDA headers, no cpsig dependency.

#include "cu_exp_host.h"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <stdexcept>

// =========================================================================
// Minimal Lyndon word infrastructure (self-contained, no external deps)
// =========================================================================

typedef std::vector<uint64_t> Word;

// Golden-ratio (Fibonacci) hash spreader — same constant used by boost::hash_combine.
static constexpr std::size_t kFibHashConst = 0x9e3779b97f4a7c15ULL;

struct WordHash {
	std::size_t operator()(const Word& w) const noexcept {
		std::size_t h = 0;
		for (uint64_t x : w)
			h ^= std::hash<uint64_t>{}(x) + kFibHashConst + (h << 6) + (h >> 2);
		return h;
	}
};

static bool is_lyndon(const Word& w) {
	uint64_t n = w.size();
	if (n <= 1) return n == 1;
	for (uint64_t i = 1; i < n; ++i)
		if (!std::lexicographical_compare(w.begin(), w.end(), w.begin() + i, w.end()))
			return false;
	return true;
}

static void lyndon_words_of_length(std::vector<Word>& res, uint64_t n, uint64_t d) {
	if (n == 1) { for (uint64_t i = 0; i < d; ++i) res.push_back({i}); return; }
	std::vector<uint64_t> w(n, 0);
	while (true) {
		if (is_lyndon(Word(w.begin(), w.end())))
			res.push_back(Word(w.begin(), w.end()));
		uint64_t i = n - 1;
		while (i < n && w[i] == d - 1) --i;
		if (i >= n) break;
		++w[i];
		for (uint64_t j = i + 1; j < n; ++j) w[j] = 0;
	}
}

static std::vector<Word> all_lyndon_words(uint64_t d, uint64_t degree) {
	std::vector<Word> res;
	for (uint64_t n = 1; n <= degree; ++n)
		lyndon_words_of_length(res, n, d);
	return res;
}

static uint64_t word_to_idx(const Word& w, uint64_t d) {
	uint64_t idx = 0;
	for (auto c : w) { idx *= d; idx += c + 1; }
	return idx;
}

static Word longest_lyndon_suffix(const Word& w, const std::unordered_set<Word, WordHash>& lyndon_set) {
	for (uint64_t i = 1; i < w.size(); ++i) {
		Word suffix(w.begin() + i, w.end());
		if (lyndon_set.count(suffix)) return suffix;
	}
	throw std::runtime_error("No Lyndon suffix found");
}

static void host_populate_level_index(uint64_t* li, uint64_t d, uint64_t count) {
	li[0] = 0;
	for (uint64_t i = 1; i < count; ++i) {
		if (d != 0 && li[i - 1] > UINT64_MAX / d)
			throw std::overflow_error("host_populate_level_index: level_index overflow");
		const uint64_t mul = li[i - 1] * d;
		if (mul > UINT64_MAX - 1)
			throw std::overflow_error("host_populate_level_index: level_index overflow");
		li[i] = mul + 1;
	}
}

// =========================================================================
// Build expansion matrix
// =========================================================================

template<typename T>
static void build_expansion_matrix_impl(
	T* h_expand,
	uint64_t sig_len, uint64_t m,
	uint64_t dimension, uint64_t degree, int method
) {
	auto level_index = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index.get(), dimension, degree + 2);

	auto lyndon_words = all_lyndon_words(dimension, degree);
	std::unordered_set<Word, WordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());

	std::unordered_map<Word, uint64_t, WordHash> word_idx_map;
	for (uint64_t i = 0; i < m; ++i)
		word_idx_map[lyndon_words[i]] = i;

	// Compute bracket expansions
	auto expansions = std::make_unique<T[]>(m * sig_len);

	for (uint64_t i = 0; i < m; ++i) {
		T* exp_i = expansions.get() + i * sig_len;
		std::fill(exp_i, exp_i + sig_len, static_cast<T>(0));

		if (lyndon_words[i].size() == 1) {
			exp_i[word_to_idx(lyndon_words[i], dimension)] = static_cast<T>(1);
		}
		else {
			Word v = longest_lyndon_suffix(lyndon_words[i], lyndon_set);
			Word u(lyndon_words[i].begin(), lyndon_words[i].end() - v.size());
			uint64_t u_idx = word_idx_map.at(u);
			uint64_t v_idx = word_idx_map.at(v);
			const T* exp_u = expansions.get() + u_idx * sig_len;
			const T* exp_v = expansions.get() + v_idx * sig_len;

			for (uint64_t tl = 2; tl <= degree; ++tl) {
				for (uint64_t l1 = 1; l1 < tl; ++l1) {
					uint64_t l2 = tl - l1;
					T* r = exp_i + level_index[tl];
					for (const T* lu = exp_u + level_index[l1]; lu < exp_u + level_index[l1 + 1]; ++lu)
						for (const T* rv = exp_v + level_index[l2]; rv < exp_v + level_index[l2 + 1]; ++rv)
							*(r++) += *lu * *rv;
					r = exp_i + level_index[tl];
					for (const T* lv = exp_v + level_index[l1]; lv < exp_v + level_index[l1 + 1]; ++lv)
						for (const T* ru = exp_u + level_index[l2]; ru < exp_u + level_index[l2 + 1]; ++ru)
							*(r++) -= *lv * *ru;
				}
			}
		}
	}

	// Build the projection matrix P and its inverse (for method=1)
	// P maps Lyndon bracket basis → Lyndon word coordinates
	// P^{-1} maps Lyndon word coordinates → bracket coefficients
	//
	// For method=2: E[j,i] = expansion[i][j] (direct)
	// For method=1: E[:,k] = sum_i (P^{-1})[i,k] * expansion[i]

	std::fill(h_expand, h_expand + sig_len * m, static_cast<T>(0));

	if (method == 2) {
		for (uint64_t i = 0; i < m; ++i)
			for (uint64_t j = 0; j < sig_len; ++j)
				h_expand[j * m + i] = expansions[i * sig_len + j];
	}
	else {
		// method=1: need P^{-1} to convert Lyndon word positions → bracket coefficients
		// Build P (lower triangular, ones on diagonal) then solve P * coefs = unit_k for each k

		// Build P: P[i,j] = expansion[j] evaluated at lyndon_idx[i]
		// i.e. P[i,j] = expansion_of_word_j at tensor position of word_i
		auto lyndon_idx = std::make_unique<uint64_t[]>(m);
		for (uint64_t i = 0; i < m; ++i)
			lyndon_idx[i] = word_to_idx(lyndon_words[i], dimension);

		// P is lower triangular (words ordered by length then lex)
		// P[i,j] = expansion[j][lyndon_idx[i]]
		auto P = std::make_unique<T[]>(m * m);
		for (uint64_t i = 0; i < m; ++i)
			for (uint64_t j = 0; j <= i; ++j)
				P[i * m + j] = expansions[j * sig_len + lyndon_idx[i]];

		// For each column k: solve P * coefs = e_k via forward substitution
		for (uint64_t k = 0; k < m; ++k) {
			auto coefs = std::make_unique<T[]>(m);
			std::fill(coefs.get(), coefs.get() + m, static_cast<T>(0));
			coefs[k] = static_cast<T>(1);

			// Forward substitution: P is lower triangular with P[i,i] = 1
			for (uint64_t i = 0; i < m; ++i) {
				for (uint64_t j = 0; j < i; ++j)
					coefs[i] -= P[i * m + j] * coefs[j];
				// coefs[i] /= P[i,i] but P[i,i] = 1
			}

			// E[:,k] = sum_i coefs[i] * expansion[i]
			for (uint64_t i = 0; i < m; ++i) {
				if (coefs[i] == static_cast<T>(0)) continue;
				const T* exp_i = expansions.get() + i * sig_len;
				for (uint64_t j = 0; j < sig_len; ++j)
					h_expand[j * m + k] += coefs[i] * exp_i[j];
			}
		}
	}
}

void build_expansion_matrix_f(
	float* h_expand, uint64_t sig_len, uint64_t m,
	uint64_t dimension, uint64_t degree, int method
) {
	build_expansion_matrix_impl<float>(h_expand, sig_len, m, dimension, degree, method);
}

void build_expansion_matrix_d(
	double* h_expand, uint64_t sig_len, uint64_t m,
	uint64_t dimension, uint64_t degree, int method
) {
	build_expansion_matrix_impl<double>(h_expand, sig_len, m, dimension, degree, method);
}

uint64_t get_lyndon_count(uint64_t dimension, uint64_t degree) {
	return all_lyndon_words(dimension, degree).size();
}
