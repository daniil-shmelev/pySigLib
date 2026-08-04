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
#include "log_sig_method.h"

// Host-side functions for logsig_to_sig methods 1/2.
// Compiled by MSVC C++20, callable from CUDA host code.

void build_expansion_matrix_f(
	float* h_expand,
	uint64_t sig_len, uint64_t m,
	uint64_t dimension, uint64_t degree, LogSigMethod method
);

void build_expansion_matrix_d(
	double* h_expand,
	uint64_t sig_len, uint64_t m,
	uint64_t dimension, uint64_t degree, LogSigMethod method
);

uint64_t get_lyndon_count(uint64_t dimension, uint64_t degree);
