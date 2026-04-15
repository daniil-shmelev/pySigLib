/* Copyright 2025 Daniil Shmelev
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
#include "words.h"

bool is_lyndon(const word& w) {
	const uint64_t n = w.size();
	if (n == 0)
		return false;
	if (n == 1)
		return true;
	for (uint64_t i = 1; i < n; ++i) {
		if (!std::lexicographical_compare(
			w.begin(), w.end(),
			w.begin() + i, w.end()
		))
			return false;
	}
	return true;
}

void all_lyndon_words_of_length_n(std::vector<word>& res, uint64_t n, uint64_t dimension) {
	word w;
	w.push_back(0);

	while (!w.empty())
	{
		uint64_t m = w.size();
		if (m == n)
			res.push_back(w);

		while (w.size() < n)
			w.push_back(w[w.size() - m]);

		while (!w.empty() && w.back() == dimension - 1)
			w.pop_back();

		if (!w.empty())
			++w.back();
	}
}

std::vector<word> all_lyndon_words(uint64_t dimension, uint64_t degree) {
	std::vector<word> res;
	for (uint64_t n = 1; n <= degree; ++n)
		all_lyndon_words_of_length_n(res, n, dimension);
	return res;
}

uint64_t word_to_idx(const word& w, uint64_t dimension) {
	if (!w.size())
		return 0;

	uint64_t idx = 0;
	for (uint64_t i : w) {
		if (dimension != 0 && idx > UINT64_MAX / dimension)
			throw std::overflow_error("word_to_idx: index overflow");
		const uint64_t mul = idx * dimension;
		const uint64_t add = i + 1;
		if (mul > UINT64_MAX - add)
			throw std::overflow_error("word_to_idx: index overflow");
		idx = mul + add;
	}
	return idx;
}

std::vector<uint64_t> all_lyndon_idx(uint64_t dimension, uint64_t degree) {
	std::vector<word> words = all_lyndon_words(dimension, degree);
	std::vector<uint64_t> res;
	res.reserve(words.size());
	for (const auto& w : words) {
		res.push_back(word_to_idx(w, dimension));
	}
	return res;
}

word longest_lyndon_suffix_(const word& w, const std::unordered_set<word, WordHash>& lyndon_set) {
	uint64_t n = w.size();
	for (uint64_t i = 1; i < n; ++i) {
		word suffix(w.begin() + i, w.end());
		if (lyndon_set.find(suffix) != lyndon_set.end()) {
			return suffix;
		}
	}
	throw std::runtime_error("Error looking for lyndon suffix");
}

std::pair<word, word> standard_factorization(
	const word& w,
	const std::unordered_set<word, WordHash>& lyndon_set
) {
	word m = longest_lyndon_suffix_(w, lyndon_set);
	word l(w.begin(), w.end() - m.size());
	return { l, m };
}

word concatenate_words(const word& a, const word& b) {
	word c(a);
	c.insert(c.end(), b.begin(), b.end());
	return c;
}

uint64_t concatenate_idx(uint64_t i, uint64_t j, uint64_t len_j, uint64_t dimension) {
	// If i and j correspond to word_to_idx(a) and word_to_idx(b),
	// then this function outputs word_to_idx(c) where c is the
	// concatenation of a and b.
	const uint64_t p = ::power(dimension, len_j);
	if (!p || (i != 0 && i > UINT64_MAX / p))
		throw std::overflow_error("concatenate_idx: index overflow");
	const uint64_t mul = i * p;
	if (mul > UINT64_MAX - j)
		throw std::overflow_error("concatenate_idx: index overflow");
	return mul + j;
}

void lyndon_proj_matrix(
	SparseIntMatrix& out,
	const std::vector<word>& lyndon_words,
	std::vector<uint64_t> lyndon_idx, // copy here is intentional
	uint64_t dimension,
	uint64_t degree
) {
	// Note the final output here will drop the diagonal of 1s

	std::unordered_set<word, WordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());
	uint64_t n = sig_length(dimension, degree);
	if (n == 0)
		throw std::overflow_error("lyndon_proj_matrix: sig_length overflow");
	uint64_t m = lyndon_words.size();

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	SparseIntMatrix full_mat_transpose(m, n);

	std::unordered_map<word, uint64_t, WordHash> col_idx;

	for (uint64_t i = 0; i < m; ++i) {
		col_idx[lyndon_words[i]] = i;
	}

	for (uint64_t i = 0; i < m; ++i) {
		const word& w = lyndon_words[i];

		if (w.size() == 1) {
			full_mat_transpose.insert_entry(i, w[0] + 1, 1);
		}
		else {
			word v = longest_lyndon_suffix_(w, lyndon_set);
			word u(w.begin(), w.end() - v.size());

			uint64_t jw = col_idx[w];
			uint64_t jv = col_idx[v];
			uint64_t ju = col_idx[u];

			for (const auto& eu : full_mat_transpose.rows[ju]) {
				if (eu.val) {
					for (const auto& ev : full_mat_transpose.rows[jv]) {
						if (ev.val) {
							uint64_t ic = concatenate_idx(eu.col, ev.col, v.size(), dimension);
							int val = eu.val * ev.val;
							full_mat_transpose.add_to_entry(jw, ic, val);
							ic = concatenate_idx(ev.col, eu.col, u.size(), dimension);
							full_mat_transpose.add_to_entry(jw, ic, -val);
						}
					}
				}
			}
		}
	}

	SparseIntMatrix full_mat;
	full_mat_transpose.transpose(full_mat);
	out.resize(full_mat.m, full_mat.m);

	for (uint64_t i = 0; i < m; ++i) {
		out.rows[i] = full_mat.rows[lyndon_idx[i]];
	}

	out.drop_diagonal();
}
