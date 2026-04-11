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
#include "cp_tensor_poly.h"
#include "cp_signature.h"
#include "words.h"
#include "log_sig_cache.h"

#include "cp_path.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

// partial_logs will store intermediate steps in the calculation for backprop.
// We make the decision here to recompute these on the backward pass, rather
// than keeping them from the forward pass. The log calculation is minor compared
// to that of the signature, so this is a small overhead, but avoids us allocating
// massive chunks of memory on the forward.
template<std::floating_point T>
void tensor_log_(
	T* sig,
	uint64_t dimension,
	uint64_t degree,
	T* partial_logs = nullptr
) {
	sig[0] = static_cast<T>(0.);
	if (degree == 1)
		return;

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	uint64_t buff1_size = ::sig_length(dimension, degree - 1);
	std::unique_ptr<T[]> buff1_uptr = std::make_unique<T[]>(buff1_size);
	T* buff1 = buff1_uptr.get();
	std::fill(buff1, buff1 + buff1_size, static_cast<T>(0.));

	uint64_t buff2_size = ::sig_length(dimension, degree);
	auto buff2_uptr = std::make_unique<T[]>(buff2_size);
	T* buff2 = buff2_uptr.get();
	std::fill(buff2, buff2 + buff2_size, static_cast<T>(0.));

	for (int64_t k = degree; k > 0; --k) {
		T constant = static_cast<T>(1.) / k;

		for (uint64_t target_level = 2; target_level <= 1 + degree - k; ++target_level) {

			std::fill(buff2 + level_index[target_level], buff2 + level_index[target_level + 1], static_cast<T>(0.));

			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				uint64_t right_level = target_level - left_level;

				T* res_ptr = buff2 + level_index[target_level];
				T* const left_ptr_end = sig + level_index[left_level + 1];
#ifdef VEC
				const uint64_t right_level_size = level_index[right_level + 1] - level_index[right_level];
				const T* right_start = buff1 + level_index[right_level];
				for (T* left_ptr = sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr) {
					vec_mult_add(res_ptr, right_start, *left_ptr, right_level_size);
					res_ptr += right_level_size;
				}
#else
				T* const right_ptr_end = buff1 + level_index[right_level + 1];
				for (T* left_ptr = sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr)
					for (T* right_ptr = buff1 + level_index[right_level]; right_ptr < right_ptr_end; ++right_ptr)
						*(res_ptr++) += *left_ptr * *right_ptr;
#endif
			}
		}

		if (k == 1) continue;

		for (uint64_t target_level = 1; target_level <= 1 + degree - k; ++target_level) {

			uint64_t target_level_size = level_index[target_level + 1] - level_index[target_level];
			T* const res_ptr = buff1 + level_index[target_level];
			T* const ptr_1 = sig + level_index[target_level];
			T* const ptr_2 = buff2 + level_index[target_level];

			for (uint64_t i = 0; i < target_level_size; ++i) {
				res_ptr[i] = constant * ptr_1[i] - ptr_2[i];
			}
		}
		if (partial_logs && k > 2 && k != static_cast<int64_t>(degree)) {
			std::memcpy(partial_logs, buff1, sizeof(T) * buff1_size);
			partial_logs += buff1_size;
		}
	}
	for (uint64_t target_level = 2; target_level <= degree; ++target_level) {

		uint64_t target_level_size = level_index[target_level + 1] - level_index[target_level];
		T* const res_ptr = sig + level_index[target_level];
		T* const ptr = buff2 + level_index[target_level];

		for (uint64_t i = 0; i < target_level_size; ++i) {
			res_ptr[i] -= ptr[i];
		}
	}
	if (partial_logs)
		std::memcpy(partial_logs, buff1, sizeof(T) * buff1_size);
}

template<std::floating_point T>
void log_sig_expanded(
	const T* sig,
	T* out,
	uint64_t dimension,
	uint64_t degree
) {
	std::memcpy(out, sig, ::sig_length(dimension, degree) * sizeof(T));
	tensor_log_<T>(out, dimension, degree);
}

template<std::floating_point T>
void log_sig_lyndon_words(
	const T* sig,
	T* out,
	uint64_t dimension,
	uint64_t degree
) {
	const BasisCache& cache_ = get_basis_cache(dimension, degree, 1);

	auto log_sig_uptr = std::make_unique<T[]>(::sig_length(dimension, degree));
	T* log_sig = log_sig_uptr.get();
	std::memcpy(log_sig, sig, ::sig_length(dimension, degree) * sizeof(T));

	tensor_log_<T>(log_sig, dimension, degree);

	uint64_t m = cache_.lyndon_idx.size();
	for (uint64_t i = 0; i < m; ++i) {
		out[i] = log_sig[cache_.lyndon_idx[i]];
	}
}

template<std::floating_point T>
void log_sig_lyndon_basis(
	const T* sig,
	T* out,
	uint64_t dimension,
	uint64_t degree
) {
	log_sig_lyndon_words(sig, out, dimension, degree);
	const BasisCache& cache_ = get_basis_cache(dimension, degree, 2);
	cache_.inv_proj_mat.mul_vec_inplace_lower(out);
}

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
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0
)
{
	if (dimension == 0) { throw std::invalid_argument("log signature received path of dimension 0"); }
	if (degree == 0) { throw std::invalid_argument("log signature received degree 0"); }

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);

	get_log_sig_<T>(sig, out, aug_dimension, degree, method);
}

template<std::floating_point T>
void batch_sig_to_log_sig_(
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
	const T* const data_end = sig + sig_len * batch_size;

	auto log_sig_func = [&](const T* sig_ptr, T* out_ptr) {
		get_log_sig_<T>(sig_ptr, out_ptr, aug_dimension, degree, method);
	};

	if (n_jobs != 1) {
		multi_threaded_batch(log_sig_func, sig, out, batch_size, sig_len, result_length, n_jobs);
	}
	else {
		const T* sig_ptr = sig;
		T* out_ptr = out;
		for (; sig_ptr < data_end;
			sig_ptr += sig_len, out_ptr += result_length) {

			log_sig_func(sig_ptr, out_ptr);
		}
	}
	return;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//// backpropagation
////////////////////////////////////////////////////////////////////////////////////////////////

template<std::floating_point T>
void tensor_log_backprop_(
	T* out,
	T* derivs,
	const T* sig,
	uint64_t dimension,
	uint64_t degree
) {
	uint64_t sig_len_ = ::sig_length(dimension, degree);
	uint64_t sig_len_2_ = ::sig_length(dimension, degree - 1);

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	std::memcpy(out, derivs, sig_len_ * sizeof(T));
	// Forward forces log_sig[0] = 0 (constant), so d/d(sig[0]) = 0.
	out[0] = static_cast<T>(0.);

	auto sig_copy_uptr = std::make_unique<T[]>(sig_len_);
	T* sig_copy = sig_copy_uptr.get();
	std::memcpy(sig_copy, sig, sig_len_ * sizeof(T));

	auto partial_logs_uptr = std::make_unique<T[]>(sig_len_2_ * (degree - 1));
	T* partial_logs = partial_logs_uptr.get();

	auto other_derivs_uptr = std::make_unique<T[]>(sig_len_);
	T* other_derivs = other_derivs_uptr.get();
	std::fill(other_derivs, other_derivs + sig_len_, static_cast<T>(0.));

	if (degree <= 1)
		return;

	tensor_log_<T>(sig_copy, dimension, degree, partial_logs);

	T factor = static_cast<T>(-1.);
	for (uint64_t depth = 1; depth + 1 < degree; ++depth) {
		T scalar = static_cast<T>(1.) / (1 + depth);
		T* partial = partial_logs + (degree - 2 - depth) * sig_len_2_;
		uncombine_sig_deriv_zero(sig, partial, derivs, other_derivs, dimension, degree + 1 - depth, level_index);
		for (uint64_t lev = 1; lev <= degree - depth; ++lev) {
			T* it = out + level_index[lev];
			for (uint64_t i = level_index[lev]; i < level_index[lev + 1]; ++i) {
				*(it++) += factor * (derivs[i] + scalar * other_derivs[i]);
			}
		}
		std::swap(other_derivs, derivs);
		factor = -factor;
	}
	// backprop level 2
	T scalar = factor / degree;
	T* out_ptr = out + level_index[1];
	T* derivs_ptr = derivs + level_index[2];
	const T* sig_ptr = sig + level_index[1];
	for (uint64_t i = 0; i < dimension; ++i) {
		for (uint64_t j = 0; j < dimension; ++j) {
			out_ptr[i] += derivs_ptr[i + dimension * j] * sig_ptr[j] * scalar;
			out_ptr[j] += derivs_ptr[i + dimension * j] * sig_ptr[i] * scalar;
		}
	}
}

template<std::floating_point T>
void tensor_log_backprop_lyndon_words(
	T* out,
	T* log_sig_derivs,
	const T* sig,
	uint64_t dimension,
	uint64_t degree
) {
	const BasisCache& cache_ = get_basis_cache(dimension, degree, 1);

	uint64_t sig_len_ = ::sig_length(dimension, degree);
	auto log_sig_derivs_copy_uptr = std::make_unique<T[]>(sig_len_);
	T* log_sig_derivs_copy = log_sig_derivs_copy_uptr.get();
	std::fill(log_sig_derivs_copy, log_sig_derivs_copy + sig_len_, static_cast<T>(0.));

	uint64_t m = cache_.lyndon_idx.size();
	for (uint64_t i = 0; i < m; ++i) {
		log_sig_derivs_copy[cache_.lyndon_idx[i]] = log_sig_derivs[i];
	}

	tensor_log_backprop_<T>(out, log_sig_derivs_copy, sig, dimension, degree);
}

template<std::floating_point T>
void tensor_log_backprop_lyndon_basis(
	T* out,
	T* log_sig_derivs,
	const T* sig,
	uint64_t dimension,
	uint64_t degree
) {
	const BasisCache& cache_ = get_basis_cache(dimension, degree, 2);

	cache_.inv_proj_mat_transpose.mul_vec_inplace_upper(log_sig_derivs);

	uint64_t sig_len_ = ::sig_length(dimension, degree);
	auto log_sig_derivs_copy_uptr = std::make_unique<T[]>(sig_len_);
	T* log_sig_derivs_copy = log_sig_derivs_copy_uptr.get();
	std::fill(log_sig_derivs_copy, log_sig_derivs_copy + sig_len_, static_cast<T>(0.));

	uint64_t m = cache_.lyndon_idx.size();
	for (uint64_t i = 0; i < m; ++i) {
		log_sig_derivs_copy[cache_.lyndon_idx[i]] = log_sig_derivs[i];
	}

	tensor_log_backprop_<T>(out, log_sig_derivs_copy, sig, dimension, degree);
}

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
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0
) {
	if (dimension == 0) { throw std::invalid_argument("sig_backprop received path of dimension 0"); }

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);

	const uint64_t log_sig_len_ = method ? ::log_sig_length(aug_dimension, degree) : ::sig_length(aug_dimension, degree);

	auto log_sig_derivs_copy_uptr = std::make_unique<T[]>(log_sig_len_);
	T* log_sig_derivs_copy = log_sig_derivs_copy_uptr.get();
	std::memcpy(log_sig_derivs_copy, log_sig_derivs, log_sig_len_ * sizeof(T));

	get_sig_to_log_sig_backprop_<T>(sig, out, log_sig_derivs_copy, aug_dimension, degree, method);
}

template<std::floating_point T>
void batch_sig_to_log_sig_backprop_(
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
	const T* const data_end = sig + sig_len_ * batch_size;

	auto log_sig_derivs_copy_uptr = std::make_unique<T[]>(log_sig_len_ * batch_size);
	T* log_sig_derivs_copy = log_sig_derivs_copy_uptr.get();
	std::memcpy(log_sig_derivs_copy, log_sig_derivs, log_sig_len_ * batch_size * sizeof(T));

	auto log_sig_backprop_func = [&](const T* sig_ptr, T* log_sig_derivs_ptr, T* out_ptr) {
		get_sig_to_log_sig_backprop_<T>(sig_ptr, out_ptr, log_sig_derivs_ptr, aug_dimension, degree, method);
	};

	if (n_jobs != 1) {
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
	}
	else {
		T* log_sig_derivs_ptr = log_sig_derivs_copy;
		const T* sig_ptr = sig;
		T* out_ptr = out;
		for (; sig_ptr < data_end;
			log_sig_derivs_ptr += log_sig_len_, sig_ptr += sig_len_, out_ptr += sig_len_) {

			log_sig_backprop_func(sig_ptr, log_sig_derivs_ptr, out_ptr);
		}
	}
	return;
}
