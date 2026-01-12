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

#include "multithreading.h"
#include "macros.h"

template<std::floating_point T>
void sig_coef_(
	const T* path,
	double* out,
	const uint64_t* multi_idx,
	uint64_t num_multi_idx, // len(multi_idx)
	const uint64_t* degrees, // [ len(multi_index[i]) for i in 0:num_multi_index ]
	uint64_t dimension,
	uint64_t length,
	bool time_aug = false,
	bool lead_lag = false,
	double end_time = 1.
);

template<std::floating_point T>
void batch_sig_coef_(
	const T* path,
	double* out,
	const uint64_t* multi_idx,
	uint64_t num_multi_idx, // len(multi_idx)
	const uint64_t* degrees, // [ len(multi_index[i]) for i in 0:num_multi_index ]
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug = false,
	bool lead_lag = false,
	double end_time = 1.,
	int n_jobs = 1
);