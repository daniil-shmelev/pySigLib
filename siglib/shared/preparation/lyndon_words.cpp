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

#include "lyndon_words.h"
#include "tensor_basis.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

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
	const uint64_t p = tensor_power(dimension, len_j);
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
	const uint64_t n = tensor_sig_length(dimension, degree);
	if (n == 0)
		throw std::overflow_error("lyndon_proj_matrix: sig_length overflow");
	if (lyndon_idx.size() != lyndon_words.size())
		throw std::invalid_argument("lyndon_proj_matrix: index count mismatch");

	std::unordered_map<word, uint64_t, WordHash> flat_idx;
	flat_idx.reserve(lyndon_words.size());
	for (uint64_t i = 0; i < lyndon_words.size(); ++i)
		flat_idx[lyndon_words[i]] = lyndon_idx[i];

	lyndon_proj_matrix_from_words(
		out,
		lyndon_words,
		n,
		[&flat_idx](const word& w) {
			return flat_idx.at(w);
		},
		[dimension](uint64_t i, uint64_t j, uint64_t len_j) {
			return concatenate_idx(i, j, len_j, dimension);
		});
}

void lyndon_proj_matrix_from_words(
	SparseIntMatrix& out,
	const std::vector<word>& lyndon_words,
	uint64_t flat_word_count,
	const std::function<uint64_t(const word&)>& word_to_flat_idx,
	const std::function<uint64_t(uint64_t, uint64_t, uint64_t)>& concatenate_flat_idx
) {
	std::unordered_set<word, WordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());
	const uint64_t m = lyndon_words.size();
	SparseIntMatrix full_mat_transpose(m, flat_word_count);
	std::unordered_map<word, uint64_t, WordHash> col_idx;
	col_idx.reserve(m);

	for (uint64_t i = 0; i < m; ++i)
		col_idx[lyndon_words[i]] = i;

	for (uint64_t i = 0; i < m; ++i) {
		const word& w = lyndon_words[i];

		if (w.size() == 1) {
			const uint64_t flat_idx = word_to_flat_idx(w);
			if (flat_idx >= flat_word_count)
				throw std::out_of_range("lyndon projection word index out of range");
			full_mat_transpose.insert_entry(i, flat_idx, 1);
		}
		else {
			word v = longest_lyndon_suffix_(w, lyndon_set);
			word u(w.begin(), w.end() - v.size());

			const uint64_t jw = col_idx.at(w);
			const uint64_t jv = col_idx.at(v);
			const uint64_t ju = col_idx.at(u);

			for (const auto& eu : full_mat_transpose.rows[ju]) {
				if (eu.val == 0)
					continue;
				for (const auto& ev : full_mat_transpose.rows[jv]) {
					if (ev.val == 0)
					continue;
					uint64_t flat_idx = concatenate_flat_idx(eu.col, ev.col, v.size());
					if (flat_idx >= flat_word_count)
						throw std::out_of_range("lyndon projection concatenation index out of range");
					const int coeff = eu.val * ev.val;
					full_mat_transpose.add_to_entry(jw, flat_idx, coeff);
					flat_idx = concatenate_flat_idx(ev.col, eu.col, u.size());
					if (flat_idx >= flat_word_count)
						throw std::out_of_range("lyndon projection concatenation index out of range");
					full_mat_transpose.add_to_entry(jw, flat_idx, -coeff);
				}
			}
		}
	}

	SparseIntMatrix full_mat;
	full_mat_transpose.transpose(full_mat);
	out.resize(m, m);
	for (uint64_t i = 0; i < m; ++i) {
		const uint64_t flat_idx = word_to_flat_idx(lyndon_words[i]);
		if (flat_idx >= flat_word_count)
			throw std::out_of_range("lyndon projection word index out of range");
		out.rows[i] = full_mat.rows[flat_idx];
	}

	out.drop_diagonal();
}
