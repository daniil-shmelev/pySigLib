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

#pragma once
#include "cppch.h"
#include "cp_utils.h"
#include "sparse.h"

typedef std::vector<uint64_t> word;

// Golden-ratio (Fibonacci) hash spreader - same constant used by boost::hash_combine.
static constexpr std::size_t kFibHashConst = 0x9e3779b97f4a7c15ULL;

struct WordHash {
	std::size_t operator()(const word& w) const noexcept {
		std::size_t h = 0;
		for (uint64_t x : w) {
			h ^= std::hash<uint64_t>{}(x)
				+ kFibHashConst
				+ (h << 6)
				+ (h >> 2);
		}
		return h;
	}
};

struct PairHash {
	std::size_t operator()(const std::pair<uint64_t, uint64_t>& p) const noexcept {
		std::size_t h1 = std::hash<uint64_t>{}(p.first);
		std::size_t h2 = std::hash<uint64_t>{}(p.second);
		return h1 ^ (h2 + kFibHashConst + (h1 << 6) + (h1 >> 2));
	}
};

bool is_lyndon(const word& w);
std::vector<word> all_lyndon_words(uint64_t dimension, uint64_t degree);
std::vector<uint64_t> all_lyndon_idx(uint64_t dimension, uint64_t degree);
uint64_t word_to_idx(const word& w, uint64_t dimension);
word longest_lyndon_suffix_(const word& w, const std::unordered_set<word, WordHash>& lyndon_words);
word concatenate_words(const word& a, const word& b);
uint64_t concatenate_idx(uint64_t i, uint64_t j, uint64_t len_j, uint64_t dimension);

std::pair<word, word> standard_factorization(
	const word& w,
	const std::unordered_set<word, WordHash>& lyndon_set
);

void lyndon_proj_matrix(
	SparseIntMatrix& out,
	const std::vector<word>& lyndon_words,
	std::vector<uint64_t> lyndon_idx, // copy here is intentional
	uint64_t dimension,
	uint64_t degree
);

void lyndon_proj_matrix_from_words(
	SparseIntMatrix& out,
	const std::vector<word>& lyndon_words,
	uint64_t flat_word_count,
	const std::function<uint64_t(const word&)>& word_to_flat_idx,
	const std::function<uint64_t(uint64_t, uint64_t, uint64_t)>& concatenate_flat_idx
);

