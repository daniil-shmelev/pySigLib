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
#include "multithreading.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

template<std::floating_point T>
FORCE_INLINE void tensor_product_add_(
	const T* RESTRICT a,
	uint64_t a_degree,
	const T* RESTRICT b,
	uint64_t b_degree,
	T* RESTRICT out,
	uint64_t out_degree,
	const uint64_t* level_index,
	uint64_t min_out_level = 2,
	uint64_t min_b_level = 1,
	T scale = static_cast<T>(1)
) {
	for (uint64_t level = min_out_level; level <= out_degree; ++level) {
		const uint64_t first = level > b_degree ? level - b_degree : 1;
		const uint64_t last = level > min_b_level
			? std::min(a_degree, level - min_b_level) : 0;
		for (uint64_t left = first; left <= last; ++left) {
			const uint64_t right = level - left;
			const uint64_t left_size = level_index[left + 1] - level_index[left];
			const uint64_t right_size = level_index[right + 1] - level_index[right];
			T* dst = out + level_index[level];
			const T* ap = a + level_index[left];
			const T* bp = b + level_index[right];
			for (uint64_t i = 0; i < left_size; ++i) {
#ifdef VEC
				vec_mult_add(dst, bp, ap[i] * scale, right_size);
				dst += right_size;
#else
				const T av = ap[i] * scale;
				for (uint64_t j = 0; j < right_size; ++j)
					*(dst++) += av * bp[j];
#endif
			}
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void tensor_product_backprop_(
	const T* a,
	uint64_t a_degree,
	const T* b,
	uint64_t b_degree,
	const T* d_out,
	uint64_t out_degree,
	T* d_a,
	T* d_b,
	const uint64_t* level_index,
	uint64_t min_out_level = 2,
	uint64_t min_b_level = 1,
	T scale = static_cast<T>(1)
) {
	for (uint64_t level = min_out_level; level <= out_degree; ++level) {
		const uint64_t first = level > b_degree ? level - b_degree : 1;
		const uint64_t last = level > min_b_level
			? std::min(a_degree, level - min_b_level) : 0;
		for (uint64_t left = first; left <= last; ++left) {
			const uint64_t right = level - left;
			const uint64_t left_size = level_index[left + 1] - level_index[left];
			const uint64_t right_size = level_index[right + 1] - level_index[right];
			const T* grad = d_out + level_index[level];
			const T* ap = a + level_index[left];
			const T* bp = b + level_index[right];
			T* dap = d_a + level_index[left];
			T* dbp = d_b + level_index[right];
			for (uint64_t i = 0; i < left_size; ++i) {
#ifdef VEC
				dap[i] += dot_product(grad + i * right_size, bp, right_size) * scale;
				vec_mult_add(dbp, grad + i * right_size, ap[i] * scale, right_size);
#else
				T da = static_cast<T>(0);
				for (uint64_t j = 0; j < right_size; ++j) {
					const T dg = grad[i * right_size + j] * scale;
					da += dg * bp[j];
					dbp[j] += dg * ap[i];
				}
				dap[i] += da;
#endif
			}
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void sig_combine_with_level_index_(
	const T* RESTRICT sig1,
	const T* RESTRICT sig2,
	T* RESTRICT out,
	uint64_t degree,
	const uint64_t* level_index,
	bool scalar_term = true
) {
	if (scalar_term) out[0] = static_cast<T>(1);
	for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i)
		out[i] = sig1[i] + sig2[i];
	tensor_product_add_(sig1, degree, sig2, degree, out, degree, level_index);
}

template<std::floating_point T>
FORCE_INLINE void sig_combine_backprop_with_level_index_(
	const T* sig1,
	const T* sig2,
	const T* d_out,
	T* d_sig1,
	T* d_sig2,
	uint64_t degree,
	const uint64_t* level_index
) {
	const uint64_t length = level_index[degree + 1];
	if (d_sig1 != d_out) std::copy(d_out, d_out + length, d_sig1);
	std::copy(d_out, d_out + length, d_sig2);
	tensor_product_backprop_(sig1, degree, sig2, degree, d_out, degree,
		d_sig1, d_sig2, level_index);
}

template<std::floating_point T>
FORCE_INLINE void sig_combine_inplace_(
	T* sig1,
	const T* sig2,
	uint64_t degree,
	const uint64_t* level_index,
	bool scalar_term = true
) {
	if (scalar_term) sig1[0] = static_cast<T>(1.);

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
	const T* __restrict sig1,
	const T* __restrict sig2,
	T* __restrict sig_concat_deriv,
	T* __restrict sig2_deriv,
	uint64_t,
	uint64_t degree,
	const uint64_t* level_index
) {
	//sig1, sig2 are two signatures, and sig_concat is
	//the signature of the concatenated paths, sig1 * sig2.
	//sig_concat_deriv is dF/d(sig_concat)
	//This function computes dF/d(sig1) and dF/d(sig2) and writes these
	//into sig_concat_deriv and sig2_deriv respectively

	sig_combine_backprop_with_level_index_(sig1, sig2, sig_concat_deriv,
		sig_concat_deriv, sig2_deriv, degree, level_index);
}

template<std::floating_point T>
FORCE_INLINE void uncombine_sig_deriv_zero(
	const T* __restrict sig1,
	const T* __restrict sig2,
	T* __restrict sig_concat_deriv,
	T* __restrict sig2_deriv,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	const uint64_t sig_len_ = sig_length(dimension, degree - 1);
	std::fill(sig2_deriv, sig2_deriv + sig_len_, static_cast<T>(0.));

	for (int64_t level = degree; level > 0; --level) {
		for (int64_t left_level = level - 1, right_level = 1; left_level > 0; --left_level, ++right_level) {
			T* result_ptr = sig_concat_deriv + level_index[level];
			T* const right_base = sig2_deriv + level_index[right_level];
			const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
			const T* const left_base = sig1 + level_index[left_level];

			for (uint64_t i = 0; i < left_size; ++i) {
				const T scalar = left_base[i];
				for (uint64_t k = 0; k < right_size; ++k) {
					right_base[k] += result_ptr[k] * scalar;
				}
				result_ptr += right_size;
			}
		}
	}


	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		std::fill(sig_concat_deriv + level_index[left_level], sig_concat_deriv + level_index[left_level + 1], static_cast<T>(0.));
		for (uint64_t level = left_level + 1, right_level = 1; level <= degree; ++level, ++right_level) {
			T* result_ptr = sig_concat_deriv + level_index[level];
			T* const left_base = sig_concat_deriv + level_index[left_level];
			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
			const T* const right_base = sig2 + level_index[right_level];
			const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];

			for (uint64_t i = 0; i < left_size; ++i) {
				T accum = 0;
				for (uint64_t k = 0; k < right_size; ++k) {
					accum += result_ptr[k] * right_base[k];
				}
				left_base[i] += accum;
				result_ptr += right_size;
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
	uint64_t degree,
	bool scalar_term = true
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	const uint64_t full_len = level_index[degree + 1];
	if (scalar_term) {
		std::memcpy(out, sig1, sizeof(T) * full_len);
		sig_combine_inplace_(out, sig2, degree, level_index);
	} else {
		// Caller's buffers omit the scalar at index 0.
		// Decrement level_index so offsets map to the no-scalar layout.
		auto ns_li_uptr = std::make_unique<uint64_t[]>(degree + 2);
		uint64_t* ns_li = ns_li_uptr.get();
		ns_li[0] = 0; // unused
		for (uint64_t k = 1; k < degree + 2; ++k) ns_li[k] = level_index[k] - 1;

		std::memcpy(out, sig1, sizeof(T) * (full_len - 1));
		sig_combine_inplace_(out, sig2, degree, ns_li, /*scalar_term=*/false);
	}
}

template<std::floating_point T>
void batch_sig_combine_(
	const T* sig1,
	const T* sig2,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool scalar_term = true,
	int n_jobs = 1
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine received dimension 0"); }

	const uint64_t full_len = ::sig_length(dimension, degree);
	const uint64_t stride = scalar_term ? full_len : full_len - 1;
	auto sig_combine_func = [&](const T* sig1_ptr, const T* sig2_ptr, T* out_ptr) {
		sig_combine_(sig1_ptr, sig2_ptr, out_ptr, dimension, degree, scalar_term);
	};

	multi_threaded_batch(sig_combine_func, batch_size, n_jobs,
		make_batch(sig1, stride), make_batch(sig2, stride), make_batch(out, stride));
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
	sig1_deriv[0] = static_cast<T>(0);
	sig2_deriv[0] = static_cast<T>(0);
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
	bool scalar_term = true,
	int n_jobs = 1
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine_backprop received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	const uint64_t full_len = level_index[degree + 1];
	const uint64_t stride = scalar_term ? full_len : full_len - 1;

	if (scalar_term) {
		std::memcpy(sig1_deriv, sig_combined_deriv, sizeof(T) * full_len * batch_size);
	}

	auto sig_combine_backprop_func = [&](const T* d_ptr, T* d1_ptr, T* d2_ptr, const T* s1_ptr, const T* s2_ptr) {
		if (scalar_term) {
			sig_combine_backprop_(d_ptr, d1_ptr, d2_ptr, s1_ptr, s2_ptr, dimension, degree);
		} else {
			std::vector<T> df(full_len), s1(full_len), s2(full_len);
			std::vector<T> d1(full_len), d2(full_len);
			df[0] = T(0); std::memcpy(df.data()+1, d_ptr, (full_len-1)*sizeof(T));
			s1[0] = T(1); std::memcpy(s1.data()+1, s1_ptr, (full_len-1)*sizeof(T));
			s2[0] = T(1); std::memcpy(s2.data()+1, s2_ptr, (full_len-1)*sizeof(T));
			std::memcpy(d1.data(), df.data(), full_len*sizeof(T));
			uncombine_sig_deriv(s1.data(), s2.data(), d1.data(), d2.data(), dimension, degree, level_index);
			std::memcpy(d1_ptr, d1.data()+1, (full_len-1)*sizeof(T));
			std::memcpy(d2_ptr, d2.data()+1, (full_len-1)*sizeof(T));
		}
	};

	multi_threaded_batch(sig_combine_backprop_func, batch_size, n_jobs,
		make_batch(sig_combined_deriv, stride),
		make_batch(sig1_deriv, stride),
		make_batch(sig2_deriv, stride),
		make_batch(sig1, stride),
		make_batch(sig2, stride));
	return;
}
