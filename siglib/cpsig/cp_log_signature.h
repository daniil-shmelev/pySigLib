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

#include "multithreading.h"
#include "cp_tensor_log.h"

#include "cp_path.h"

template<std::floating_point T>
void get_log_sig_(
	const T* sig,
	T* out,
	uint64_t dimension,
	uint64_t degree,
	int method = 0
)
{
	switch (method) {
	case 0:
		log_sig_expanded<T>(sig, out, dimension, degree);
		break;
	case 1:
		log_sig_lyndon_words<T>(sig, out, dimension, degree);
		break;
	case 2:
		log_sig_lyndon_basis<T>(sig, out, dimension, degree);
		break;
	default:
		throw std::runtime_error("method must be one of 0, 1 or 2");
	}
}

template<std::floating_point T>
void sig_to_log_sig_(
	const T* sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	int n_jobs = 1
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("signature received dimension 0"); }
	if (degree == 0) { throw std::invalid_argument("log signature received degree 0"); }

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);

	const uint64_t result_length = method ? ::log_sig_length(aug_dimension, degree) : ::sig_length(aug_dimension, degree);

	//General case
	const uint64_t sig_len = sig_length(aug_dimension, degree);

	auto log_sig_func = [&](const T* sig_ptr, T* out_ptr) {
		get_log_sig_<T>(sig_ptr, out_ptr, aug_dimension, degree, method);
	};

	multi_threaded_batch(log_sig_func, sig, out, batch_size, sig_len, result_length, n_jobs);
	return;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//// backpropagation
////////////////////////////////////////////////////////////////////////////////////////////////

template<std::floating_point T>
void get_sig_to_log_sig_backprop_(
	const T* sig,
	T* out,
	T* log_sig_derivs,
	uint64_t dimension,
	uint64_t degree,
	int method = 0
) {
	switch (method) {
	case 0:
		tensor_log_backprop_<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	case 1:
		tensor_log_backprop_lyndon_words<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	case 2:
		tensor_log_backprop_lyndon_basis<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	default:
		throw std::runtime_error("method must be one of 0, 1 or 2");
	}
}

template<std::floating_point T>
void sig_to_log_sig_backprop_(
	const T* sig,
	T* out,
	const T* log_sig_derivs,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	int n_jobs = 1
) {
	if (dimension == 0) { throw std::invalid_argument("sig_backprop received path of dimension 0"); }

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	
	const uint64_t sig_len_ = ::sig_length(aug_dimension, degree);
	const uint64_t log_sig_len_ = method ? ::log_sig_length(aug_dimension, degree) : ::sig_length(aug_dimension, degree);

	//General case
	auto log_sig_derivs_copy_uptr = std::make_unique<T[]>(log_sig_len_ * batch_size);
	T* log_sig_derivs_copy = log_sig_derivs_copy_uptr.get();
	std::memcpy(log_sig_derivs_copy, log_sig_derivs, log_sig_len_ * batch_size * sizeof(T));

	auto log_sig_backprop_func = [&](const T* sig_ptr, T* log_sig_derivs_ptr, T* out_ptr) {
		get_sig_to_log_sig_backprop_<T>(sig_ptr, out_ptr, log_sig_derivs_ptr, aug_dimension, degree, method);
	};

	multi_threaded_batch_2(
		log_sig_backprop_func,
		sig,
		log_sig_derivs_copy,
		out,
		batch_size,
		sig_len_,
		log_sig_len_,
		sig_len_,
		n_jobs
	);
	return;
}
