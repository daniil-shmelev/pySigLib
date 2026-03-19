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
#include <cstdint>

constexpr uint64_t BCH_MAX_HARDCODED_DEGREE = 12;

struct BchHardcodedData {
	const double* coefficients;
	const uint64_t* left_factor;
	const uint64_t* right_factor;
	uint64_t size;  // = log_sig_length(2, degree)
};

const BchHardcodedData* get_hardcoded_bch_data(uint64_t degree);
