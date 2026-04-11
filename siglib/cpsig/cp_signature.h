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

#include "cp_path.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif


// Thin wrapper: computes displacement from two Points, delegates to shared core
template<std::floating_point T>
FORCE_INLINE void linear_signature_(
	const Point<T>& start_pt,
	const Point<T>& end_pt,
	T* out,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	out[0] = static_cast<T>(1);
	if (degree == 0) return;
	for (uint64_t i = 0; i < dimension; ++i)
		out[i + 1] = end_pt[i] - start_pt[i];

	for (uint64_t level = 2; level <= degree; ++level) {
		T one_over_level = static_cast<T>(1.) / level;
		T* result_ptr = out + level_index[level];
		const T* const left_end = out + level_index[level];
#ifdef VEC
		for (const T* left_ptr = out + level_index[level - 1]; left_ptr != left_end; ++left_ptr, result_ptr += dimension) {
			vec_mult_assign(result_ptr, out + 1, (*left_ptr) * one_over_level, dimension);
		}
#else
		for (const T* left_ptr = out + level_index[level - 1]; left_ptr != left_end; ++left_ptr) {
			T val = (*left_ptr) * one_over_level;
			for (uint64_t d = 0; d < dimension; ++d)
				*(result_ptr++) = val * out[1 + d];
		}
#endif
	}
}

template<std::floating_point T>
void signature_naive_(
	const Path<T>& path,
	T* out,
	uint64_t degree
)
{
	const uint64_t dimension = path.dimension();

	Point<T> prev_pt(path.begin());
	Point<T> next_pt(path.begin());
	++next_pt;

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	linear_signature_(prev_pt, next_pt, out, dimension, degree, level_index);

	if (path.length() == 2) { return; }

	++prev_pt;
	++next_pt;

	auto linear_signature_uptr = std::make_unique<T[]>(::sig_length(dimension, degree));
	T* linear_signature = linear_signature_uptr.get();

	Point<T> last_pt(path.end());

	for (; next_pt != last_pt; ++prev_pt, ++next_pt) {
		linear_signature_(prev_pt, next_pt, linear_signature, dimension, degree, level_index);
		sig_combine_inplace_(out, linear_signature, degree, level_index);
	}
}

template<std::floating_point T>
FORCE_INLINE void signature_horner_(
	const Path<T>& path,
	T* out,
	uint64_t degree,
	uint64_t dimension, // path.dimension()
	T* increments
)
{
	Point<T> prev_pt = path.begin();
	Point<T> next_pt = path.begin();
	++next_pt;

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* const level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	linear_signature_(prev_pt, next_pt, out, dimension, degree, level_index);

	if (path.length() == 2) { return; }

	++prev_pt;
	++next_pt;

	auto horner_step_uptr = std::make_unique<T[]>(level_index[degree + 1] - level_index[degree]);
	T* const horner_step = horner_step_uptr.get();

	Point<T> last_pt(path.end());

	for (; next_pt != last_pt; ++prev_pt, ++next_pt) {
		for (uint64_t i = 0; i < dimension; ++i)
			increments[i] = next_pt[i] - prev_pt[i];

		for (int64_t target_level = static_cast<int64_t>(degree); target_level > 1LL; --target_level) {

			T one_over_level = static_cast<T>(1.) / target_level;

			//left_level = 0
			//assign z / target_level to horner_step
			for (uint64_t i = 0; i < dimension; ++i)
				horner_step[i] = increments[i] * one_over_level;

			for (int64_t left_level = 1LL, right_level = target_level - 1LL;
				left_level < target_level - 1LL;
				++left_level, --right_level) { //for each, add current left_level and times by z / right_level

				const uint64_t left_level_size = level_index[left_level + 1] - level_index[left_level];
				one_over_level = static_cast<T>(1.) / right_level;

				//Horner stuff
#ifdef VEC
				//Add and multiply
				T left_over_level;
				T* out_ptr = out + level_index[left_level + 1];
				T* result_ptr = horner_step + level_index[left_level + 2] - level_index[left_level + 1] - dimension;
				for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr, result_ptr -= dimension) {
					left_over_level = (*left_ptr + *(--out_ptr)) * one_over_level;
					vec_mult_assign(result_ptr, increments, left_over_level, dimension);
				}
#else
				//Horner stuff
				//Add
				T* left_ptr_1 = out + level_index[left_level];
				for (uint64_t i = 0; i < left_level_size; ++i) {
					horner_step[i] += *(left_ptr_1++);
				}

				//Multiply
				T left_over_level;
				T* result_ptr = horner_step + level_index[left_level + 2] - level_index[left_level + 1];
				for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr) {
					left_over_level = (*left_ptr) * one_over_level;
					for (T* right_ptr = increments + dimension - 1; right_ptr != increments - 1; --right_ptr) {
						*(--result_ptr) = left_over_level * (*right_ptr);
					}
				}
#endif
			}

			//======================= Do last iteration (left_level = target_level - 1) separately for speed, and add result straight into out

			const uint64_t left_level_size = level_index[target_level] - level_index[target_level - 1];

			//Horner stuff
#ifdef VEC
			//Add, Multiply and add, writing straight into out
			T* out_ptr = out + level_index[target_level];
			T* result_ptr = out + level_index[target_level + 1] - dimension;
			for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr, result_ptr -= dimension) {
				const T scalar = *left_ptr + *(--out_ptr);
				vec_mult_add(result_ptr, increments, scalar, dimension);
			}
#else
			//Add
			T* left_ptr_1 = out + level_index[target_level - 1];
			for (uint64_t i = 0; i < left_level_size; ++i) {
				horner_step[i] += *(left_ptr_1++);
			}

			//Multiply and add, writing straight into out
			T* result_ptr = out + level_index[target_level + 1];
			for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr) {
				for (T* right_ptr = increments + dimension - 1; right_ptr != increments - 1; --right_ptr) {
					*(--result_ptr) += (*left_ptr) * (*right_ptr); //no one_over_level here, as right_level = 1
				}
			}
#endif
		}
		//Update target_level == 1
#ifdef VEC
		vec_mult_add(out + 1, increments, static_cast<T>(1.), dimension);
#else
		for (uint64_t i = 0; i < dimension; ++i)
			out[i + 1] += increments[i];
#endif
	}
}

template<std::floating_point T, uint64_t dimension>
void signature_horner_template_(
	const Path<T>& path,
	T* out,
	uint64_t degree
) {
	T increments[dimension];
	signature_horner_(path, out, degree, dimension, increments);
}

template<std::floating_point T>
void call_signature_horner_(
	const Path<T>& path,
	T* out,
	uint64_t degree
) {
	const uint64_t dimension = path.dimension();
	switch (dimension) {
	case 1:  return signature_horner_template_<T, 1>(path, out, degree);
	case 2:  return signature_horner_template_<T, 2>(path, out, degree);
	case 3:  return signature_horner_template_<T, 3>(path, out, degree);
	case 4:  return signature_horner_template_<T, 4>(path, out, degree);
	case 5:  return signature_horner_template_<T, 5>(path, out, degree);
	case 6:  return signature_horner_template_<T, 6>(path, out, degree);
	case 7:  return signature_horner_template_<T, 7>(path, out, degree);
	case 8:  return signature_horner_template_<T, 8>(path, out, degree);
	case 9:  return signature_horner_template_<T, 9>(path, out, degree);
	case 10: return signature_horner_template_<T, 10>(path, out, degree);
	case 11: return signature_horner_template_<T, 11>(path, out, degree);
	case 12: return signature_horner_template_<T, 12>(path, out, degree);
	case 13: return signature_horner_template_<T, 13>(path, out, degree);
	case 14: return signature_horner_template_<T, 14>(path, out, degree);
	case 15: return signature_horner_template_<T, 15>(path, out, degree);
	case 16: return signature_horner_template_<T, 16>(path, out, degree);
	case 17: return signature_horner_template_<T, 17>(path, out, degree);
	case 18: return signature_horner_template_<T, 18>(path, out, degree);
	case 19: return signature_horner_template_<T, 19>(path, out, degree);
	case 20: return signature_horner_template_<T, 20>(path, out, degree);
	default:
		auto increments_uptr = std::make_unique<T[]>(dimension);
		T* const increments = increments_uptr.get();
		return signature_horner_<T>(path, out, degree, dimension, increments);
	}
}

template<std::floating_point T>
void signature_horner_step_(
	T* sig,
	const T* increments,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index,
	T* horner_step
)
{
	//Combines sig with the signature of a linear path given by increments using horner's algorithm

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 1; --target_level) {

		T one_over_level = static_cast<T>(1.) / target_level;

		//left_level = 0
		//assign z / target_level to horner_step
		for (uint64_t i = 0; i < dimension; ++i)
			horner_step[i] = increments[i] * one_over_level;

		for (int64_t left_level = 1, right_level = target_level - 1;
			left_level < target_level - 1;
			++left_level, --right_level) { //for each, add current left_level and times by z / right_level

			const uint64_t left_level_size = level_index[left_level + 1] - level_index[left_level];
			one_over_level = static_cast<T>(1. / right_level);

			//Horner stuff
#ifdef VEC
			//Add and multiply (fused)
			T left_over_level;
			T* sig_ptr = sig + level_index[left_level + 1];
			T* result_ptr = horner_step + level_index[left_level + 2] - level_index[left_level + 1] - dimension;
			for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr, result_ptr -= dimension) {
				left_over_level = (*left_ptr + *(--sig_ptr)) * one_over_level;
				vec_mult_assign(result_ptr, increments, left_over_level, dimension);
			}
#else
			//Add
			T* left_ptr_1 = sig + level_index[left_level];
			for (uint64_t i = 0; i < left_level_size; ++i) {
				horner_step[i] += *(left_ptr_1++);
			}

			//Multiply
			T left_over_level;
			T* result_ptr = horner_step + level_index[left_level + 2] - level_index[left_level + 1];
			for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr) {
				left_over_level = (*left_ptr) * one_over_level;
				for (const T* right_ptr = increments + dimension - 1; right_ptr != increments - 1; --right_ptr) {
					*(--result_ptr) = left_over_level * (*right_ptr);
				}
			}
#endif
		}

		//======================= Do last iteration (left_level = target_level - 1) separately for speed, and add result straight into out

		const uint64_t left_level_size = level_index[target_level] - level_index[target_level - 1];

		//Horner stuff
#ifdef VEC
		//Add, Multiply and add (fused), writing straight into out
		T* sig_ptr = sig + level_index[target_level];
		T* result_ptr = sig + level_index[target_level + 1] - dimension;
		for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr, result_ptr -= dimension) {
			const T scalar = *left_ptr + *(--sig_ptr);
			vec_mult_add(result_ptr, increments, scalar, dimension);
		}
#else
		//Add
		T* left_ptr_1 = sig + level_index[target_level - 1];
		for (uint64_t i = 0; i < left_level_size; ++i) {
			horner_step[i] += *(left_ptr_1++);
		}

		//Multiply and add, writing straight into out
		T* result_ptr = sig + level_index[target_level + 1];
		for (T* left_ptr = horner_step + left_level_size - 1; left_ptr != horner_step - 1; --left_ptr) {
			for (const T* right_ptr = increments + dimension - 1; right_ptr != increments - 1; --right_ptr) {
				*(--result_ptr) += (*left_ptr) * (*right_ptr); //no one_over_level here, as right_level = 1
			}
		}
#endif
	}
	//Update target_level == 1
#ifdef VEC
	vec_mult_add(sig + 1, increments, static_cast<T>(1.), dimension);
#else
	for (uint64_t i = 0; i < dimension; ++i)
		sig[i + 1] += increments[i];
#endif
}

template<std::floating_point T>
void signature_(
	const T* path, 
	T* out, 
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length, 
	uint64_t degree, 
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = 1.,
	bool horner = true,
	int n_jobs = 1
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("signature received path of dimension 0"); }

	Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t result_length = ::sig_length(dummy_path_obj.dimension(), degree);

	if (dummy_path_obj.length() <= 1) {
		T* const out_end = out + result_length * batch_size;
		std::fill(out, out_end, static_cast<T>(0.));
		for (T* out_ptr = out;
			out_ptr < out_end;
			out_ptr += result_length) {
			out_ptr[0] = 1.;
		}
		return;
	}
	if (degree == 0) { 
		std::fill(out, out + batch_size, static_cast<T>(1.));
		return; }

	//General case and degree = 1 case
	const uint64_t flat_path_length = dimension * length;

	auto sig_func = [&](const T* path_ptr, T* out_ptr) {
		Path<T> path_obj(path_ptr, dimension, length, time_aug, lead_lag, end_time);
		if (degree == 1) {
			Point<T> first_pt = path_obj.begin();
			Point<T> last_pt = --path_obj.end();
			out_ptr[0] = 1.;
			for (uint64_t i = 0; i < path_obj.dimension(); ++i)
				out_ptr[i + 1] = last_pt[i] - first_pt[i];
		}
		else if (horner) {
			call_signature_horner_<T>(path_obj, out_ptr, degree);
		}
		else {
			signature_naive_<T>(path_obj, out_ptr, degree);
		}
	};

	multi_threaded_batch(sig_func, path, out, batch_size, flat_path_length, result_length, n_jobs);
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// backpropagation
//////////////////////////////////////////////////////////////////////////////////////////////

template<std::floating_point T>
void sig_backprop_(
	const T* path, 
	T* out,
	const T* sig_derivs, 
	const T* sig, 
	uint64_t batch_size,
	uint64_t dimension, 
	uint64_t length, 
	uint64_t degree, 
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = 1.,
	int n_jobs = 1
)
{
	std::fill(out, out + length * dimension * batch_size, static_cast<T>(0.));
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("sig_backprop received path of dimension 0"); }

	Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t flat_path_length = dimension * length;
	const uint64_t sig_len_ = ::sig_length(dummy_path_obj.dimension(), degree);

	if (dummy_path_obj.length() <= 1 || degree == 0) {
		T* const out_end = out + flat_path_length * batch_size;
		std::fill(out, out_end, static_cast<T>(0.));
		return;
	}

	//General case
	auto sig_derivs_copy_uptr = std::make_unique<T[]>(sig_len_ * batch_size);
	T* sig_derivs_copy = sig_derivs_copy_uptr.get();
	std::memcpy(sig_derivs_copy, sig_derivs, sig_len_ * batch_size * sizeof(T));

	auto sig_copy_uptr = std::make_unique<T[]>(sig_len_ * batch_size);
	T* sig_copy = sig_copy_uptr.get();
	std::memcpy(sig_copy, sig, sig_len_ * batch_size * sizeof(T));

	auto sig_backprop_func = [&](const T* path_ptr, T* sig_derivs_ptr, T* sig_ptr, T* out_ptr) {
		Path<T> path_obj(path_ptr, dimension, length, time_aug, lead_lag, end_time);
		sig_backprop_inplace_<T>(path_obj, out_ptr, sig_derivs_ptr, sig_ptr, degree, sig_len_);
	};

	multi_threaded_batch_3(
		sig_backprop_func,
		path,
		sig_derivs_copy,
		sig_copy,
		out,
		batch_size,
		flat_path_length,
		sig_len_,
		sig_len_,
		flat_path_length,
		n_jobs
	);
	return;
}

template<std::floating_point T>
void sig_backprop_inplace_(
	const Path<T>& path, 
	T* out, 
	T* sig_derivs,
	T* sig,
	uint64_t degree, 
	uint64_t sig_len
) {

	const uint64_t data_dimension = path.data_dimension();
	const uint64_t dimension = path.dimension();

	const uint64_t data_length = path.data_length();

	auto local_derivs_uptr = std::make_unique<T[]>(sig_len);
	T* local_derivs = local_derivs_uptr.get();

	auto linear_signature_uptr = std::make_unique<T[]>(sig_len);
	T* linear_signature = linear_signature_uptr.get();

	auto increments_uptr = std::make_unique<T[]>(dimension);
	T* increments = increments_uptr.get();

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	auto horner_step_uptr = std::make_unique<T[]>(level_index[degree + 1] - level_index[degree]);
	T* horner_step = horner_step_uptr.get();

	Point<T> prev_pt(path.end());
	Point<T> next_pt(path.end());
	--prev_pt;
	--prev_pt;
	--next_pt;

	Point<T> first_pt(path.begin());

	if (path.lead_lag()) {
		T* pos = out + (data_length - 1) * data_dimension;
		T* neg = pos - data_dimension;
		bool parity = false;

		for (; next_pt != first_pt; --prev_pt, --next_pt, parity = !parity) {

			for (uint64_t i = 0; i < dimension; ++i)
				increments[i] = prev_pt[i] - next_pt[i];

			linear_signature_(prev_pt, next_pt, linear_signature, dimension, degree, level_index);
			signature_horner_step_(sig, increments, dimension, degree, level_index, horner_step);
			uncombine_sig_deriv(sig, linear_signature, sig_derivs, local_derivs, dimension, degree, level_index);
			linear_sig_deriv_to_increment_deriv(linear_signature, local_derivs, dimension, degree, level_index);


			//TODO: can we exploit the structure and avoid computing derivatives which are a priori zero?

			T* s = parity ? local_derivs + 1 + data_dimension : local_derivs + 1;
			for (uint64_t d = 0; d < data_dimension; ++d) {
				pos[d] += s[d];
				neg[d] -= s[d];
			}
			if (parity) {
				pos -= data_dimension;
				neg -= data_dimension;
			}
		}
	}
	else {
		for (T* pos = out + (path.length() - 1) * data_dimension; next_pt != first_pt; --prev_pt, --next_pt, pos -= data_dimension) {

			for (uint64_t i = 0; i < dimension; ++i)
				increments[i] = prev_pt[i] - next_pt[i];

			linear_signature_(prev_pt, next_pt, linear_signature, dimension, degree, level_index);
			signature_horner_step_(sig, increments, dimension, degree, level_index, horner_step);
			uncombine_sig_deriv(sig, linear_signature, sig_derivs, local_derivs, dimension, degree, level_index);
			linear_sig_deriv_to_increment_deriv(linear_signature, local_derivs, dimension, degree, level_index);

			//T* pos = out + i * dimension;
			T* neg = pos - data_dimension;
			T* s = local_derivs + 1;
			for (uint64_t d = 0; d < data_dimension; ++d) {
				pos[d] += s[d];
				neg[d] -= s[d];
			}

		}
	}
}
