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

template<std::floating_point T>
void branched_sig_coef_(
	const T* path,
	T* out,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs = 1,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.),
	bool planar = false,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
);

template<std::floating_point T>
void branched_sig_coef_backprop_(
	const T* path,
	T* out,
	const T* coefs,
	const T* derivs,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs = 1,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.),
	bool planar = false,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
);
