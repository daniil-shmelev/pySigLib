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
#include "cpsig.h"
#include "macros.h"
#include "multithreading.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

// Calculate power
// Return 0 on error (integer overflow)
uint64_t power(uint64_t base, uint64_t exp) noexcept;

void populate_level_index(uint64_t* level_index, uint64_t dimension, uint64_t degree);

template<std::floating_point T>
FORCE_INLINE void sig_combine_inplace_(
	T* sig1, 
	const T* sig2, 
	uint64_t degree, 
	const uint64_t* level_index
) {

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {
		for (int64_t left_level = target_level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			T* result_ptr = sig1 + level_index[target_level];
			const T* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];
#ifdef VEC
			const uint64_t right_level_size = level_index[right_level + 1] - level_index[right_level];
			const T* right_start = sig2 + level_index[right_level];
			for (T* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				vec_mult_add(result_ptr, right_start, *left_ptr, right_level_size);
				result_ptr += right_level_size;
			}
#else
			for (T* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				const T* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];
				for (const T* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*(result_ptr++) += (*left_ptr) * (*right_ptr);
				}
			}
#endif
		}

		//left_level = 0
#ifdef VEC
		{
			const uint64_t level_size = level_index[target_level + 1] - level_index[target_level];
			vec_mult_add(sig1 + level_index[target_level], sig2 + level_index[target_level], static_cast<T>(1.), level_size);
		}
#else
		T* result_ptr = sig1 + level_index[target_level];
		const T* const right_ptr_upper_bound = sig2 + level_index[target_level + 1];
		for (const T* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
			*(result_ptr++) += *right_ptr;
		}
#endif
	}

}

template<std::floating_point T>
FORCE_INLINE void sig_uncombine_linear_inplace_(
	T* sig1, 
	const T* sig2, 
	uint64_t degree, 
	const uint64_t* level_index
) {
	//SIG2 MUST BE THE SIGNATURE OF A LINEAR SEGMENT

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {
		for (int64_t left_level = target_level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			if (right_level % 2) {

				T* result_ptr = sig1 + level_index[target_level];
				const T* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];
				for (T* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
					const T* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];
					for (const T* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
						*(result_ptr++) -= (*left_ptr) * (*right_ptr);
					}
				}
			}
			else {

				T* result_ptr = sig1 + level_index[target_level];
				const T* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];
				for (T* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
					const T* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];
					for (const T* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
						*(result_ptr++) += (*left_ptr) * (*right_ptr);
					}
				}
			}
		}

		//left_level = 0
		if (target_level % 2) {
			T* result_ptr = sig1 + level_index[target_level];
			const T* const right_ptr_upper_bound = sig2 + level_index[target_level + 1];
			for (const T* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
				*(result_ptr++) -= *right_ptr;
			}
		}
		else {
			T* result_ptr = sig1 + level_index[target_level];
			const T* const right_ptr_upper_bound = sig2 + level_index[target_level + 1];
			for (const T* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
				*(result_ptr++) += *right_ptr;
			}
		}
	}

}

template<std::floating_point T>
FORCE_INLINE void uncombine_sig_deriv(
	const T* sig1,
	const T* sig2,
	T* sig_concat_deriv, 
	T* sig2_deriv,
	uint64_t dimension,
	uint64_t degree, 
	const uint64_t* level_index
) {
	//sig1, sig2 are two signatures, and sig_concat is
	//the signature of the concatenated paths, sig1 * sig2.
	//sig_concat_deriv is dF/d(sig_concat)
	//This function computes dF/d(sig1) and dF/d(sig2) and writes these
	//into sig_concat_deriv and sig2_deriv respectively

	const uint64_t sig_len_ = sig_length(dimension, degree);
	std::memcpy(sig2_deriv, sig_concat_deriv, sig_len_ * sizeof(T));

	for (uint64_t level = degree; level > 0; --level) {
		for (uint64_t left_level = level - 1, right_level = 1; left_level > 0; --left_level, ++right_level) {
			T* result_ptr = sig_concat_deriv + level_index[level];
			T* right_ptr_ = sig2_deriv + level_index[right_level];
			const T* const right_ptr_upper_bound = sig2_deriv + level_index[right_level + 1];
			const T* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];

			for (const T* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				for (T* right_ptr = right_ptr_; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*right_ptr += *(result_ptr++) * *left_ptr;
				}
			}
		}
	}


	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		for (uint64_t level = left_level + 1, right_level = 1; level <= degree; ++level, ++right_level) {
			T* result_ptr = sig_concat_deriv + level_index[level];
			const T* const left_ptr_upper_bound = sig_concat_deriv + level_index[left_level + 1];
			const T* right_ptr_ = sig2 + level_index[right_level];
			const T* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];

			for (T* left_ptr = sig_concat_deriv + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				for (const T* right_ptr = right_ptr_; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*left_ptr += *(result_ptr++) * (*right_ptr);
				}
			}
		}
	}

}

template<std::floating_point T>
FORCE_INLINE void uncombine_sig_deriv_zero(
	const T* sig1,
	const T* sig2,
	T* sig_concat_deriv,
	T* sig2_deriv,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	const uint64_t sig_len_ = sig_length(dimension, degree - 1);
	std::fill(sig2_deriv, sig2_deriv + sig_len_, static_cast<T>(0.));

	for (int64_t level = degree; level > 0; --level) {
		for (int64_t left_level = level - 1, right_level = 1; left_level > 0; --left_level, ++right_level) {
			T* result_ptr = sig_concat_deriv + level_index[level];
			T* right_ptr_ = sig2_deriv + level_index[right_level];
			const T* const right_ptr_upper_bound = sig2_deriv + level_index[right_level + 1];
			const T* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];

			for (const T* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				for (T* right_ptr = right_ptr_; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*right_ptr += *(result_ptr++) * *left_ptr;
				}
			}
		}
	}


	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		std::fill(sig_concat_deriv + level_index[left_level], sig_concat_deriv + level_index[left_level + 1], static_cast<T>(0.));
		for (uint64_t level = left_level + 1, right_level = 1; level <= degree; ++level, ++right_level) {
			T* result_ptr = sig_concat_deriv + level_index[level];
			const T* const left_ptr_upper_bound = sig_concat_deriv + level_index[left_level + 1];
			const T* right_ptr_ = sig2 + level_index[right_level];
			const T* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];

			for (T* left_ptr = sig_concat_deriv + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				for (const T* right_ptr = right_ptr_; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*left_ptr += *(result_ptr++) * (*right_ptr);
				}
			}
		}
	}

}

template<std::floating_point T>
FORCE_INLINE void linear_sig_deriv_to_increment_deriv(
	const T* sig,
	T* sig_deriv,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	//Given sig is the signature of a line segment [a,b] and sig_deriv
	//is the derivative dF/d(sig), then this function computes dF/d(b-a)
	// and writes it into sig_deriv[1:1+dimension].

	for (uint64_t level = degree; level > 1; --level) {
		const T one_over_level = static_cast<T>(1. / level);
		const uint64_t level_size = level_index[level] - level_index[level - 1];
		for (uint64_t j = 0; j < level_size; ++j) {
			const uint64_t offs1 = level_index[level] + dimension * j - 1;
			const uint64_t offs2 = level_index[level - 1] + j;
			for (uint64_t dd = 1; dd <= dimension; ++dd) {
				const T ii = sig_deriv[offs1 + dd] * one_over_level;
				sig_deriv[offs2] += sig[dd] * ii;
				sig_deriv[dd] += sig[offs2] * ii;
			}
		}
	}
}

template<std::floating_point T>
void sig_combine_(
	const T* sig1,
	const T* sig2,
	T* out,
	uint64_t dimension,
	uint64_t degree
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	std::memcpy(out, sig1, sizeof(T) * level_index[degree + 1]);

	sig_combine_inplace_(out, sig2, degree, level_index);
}

template<std::floating_point T>
void batch_sig_combine_(
	const T* sig1,
	const T* sig2,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int n_jobs = 1
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine received dimension 0"); }

	const uint64_t siglength = ::sig_length(dimension, degree);
	const T* const sig1_end = sig1 + siglength * batch_size;

	auto sig_combine_func = [&](const T* sig1_ptr, const T* sig2_ptr, T* out_ptr) {
		sig_combine_(sig1_ptr, sig2_ptr, out_ptr, dimension, degree);
	};

	if (n_jobs != 1) {
		multi_threaded_batch_2<const T, const T, T>(sig_combine_func, sig1, sig2, out, batch_size, siglength, siglength, siglength, n_jobs);
	}
	else {
		const T* sig1_ptr = sig1;
		const T* sig2_ptr = sig2;
		T* out_ptr = out;
		for (;
			sig1_ptr < sig1_end;
			sig1_ptr += siglength,
			sig2_ptr += siglength,
			out_ptr += siglength) {

			sig_combine_func(sig1_ptr, sig2_ptr, out_ptr);
		}
	}
	return;
}

template<std::floating_point T>
void sig_combine_backprop_(
	const T* sig_combined_deriv,
	T* sig1_deriv,
	T* sig2_deriv,
	const T* sig1,
	const T* sig2,
	uint64_t dimension,
	uint64_t degree
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine_backprop received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	std::memcpy(sig1_deriv, sig_combined_deriv, sizeof(T) * level_index[degree + 1]);

	uncombine_sig_deriv(sig1, sig2, sig1_deriv, sig2_deriv, dimension, degree, level_index);
	sig1_deriv[0] = static_cast<T>(1);
	sig2_deriv[0] = static_cast<T>(1);
	return;
}

template<std::floating_point T>
void batch_sig_combine_backprop_(
	const T* sig_combined_deriv,
	T* sig1_deriv,
	T* sig2_deriv,
	const T* sig1,
	const T* sig2,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int n_jobs = 1
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine_backprop received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	const uint64_t siglength = level_index[degree + 1];

	std::memcpy(sig1_deriv, sig_combined_deriv, sizeof(T) * siglength * batch_size);

	auto sig_combine_backprop_func = [&](const T* sig_combined_deriv_ptr, T* sig1_deriv_ptr, T* sig2_deriv_ptr, const T* sig1_ptr, const T* sig2_ptr) {
		sig_combine_backprop_(sig_combined_deriv_ptr, sig1_deriv_ptr, sig2_deriv_ptr, sig1_ptr, sig2_ptr, dimension, degree);
	};

	if (n_jobs != 1) {
		multi_threaded_batch_4<T>(sig_combine_backprop_func, sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, siglength, siglength, siglength, siglength, siglength, n_jobs);
	}
	else {
		const T* sig_combined_derivs_ptr = sig_combined_deriv;
		T* sig1_deriv_ptr = sig1_deriv;
		T* sig2_deriv_ptr = sig2_deriv;
		const T* sig1_ptr = sig1;
		const T* sig2_ptr = sig2;
		const T* sig1_end = sig1 + batch_size * siglength;
		for (;
			sig1_ptr < sig1_end;
			sig_combined_derivs_ptr += siglength,
			sig1_deriv_ptr += siglength,
			sig2_deriv_ptr += siglength,
			sig1_ptr += siglength,
			sig2_ptr += siglength
			) {

			sig_combine_backprop_func(sig_combined_derivs_ptr, sig1_deriv_ptr, sig2_deriv_ptr, sig1_ptr, sig2_ptr);
		}
	}
	return;
}
