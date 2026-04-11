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

#include "cppch.h"
#include "cpsig.h"
#include "cp_sig_coef.h"
#include "macros.h"

#include "cp_path.h"

template<std::floating_point T>
FORCE_INLINE void single_sig_coef_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t degree,
	T* coefs,
	T* buff,
	const T* one_over_fact,
	bool prefixes
) {
	Point<T> prev_pt(path.begin());
	Point<T> next_pt(++path.begin());
	const Point<T> end_pt(path.end());

	coefs[0] = static_cast<T>(1.);

	// buff[i] = prod_{k=1}^{i} incr[k]
	buff[0] = static_cast<T>(1.);
	for (uint64_t i = 1; i < degree + 1; ++i) {
		buff[i] = buff[i - 1] * (next_pt[multi_idx[i - 1]] - prev_pt[multi_idx[i - 1]]);
	}

	for (uint64_t i = 1; i < degree + 1; ++i) {
		coefs[i] = buff[i] * one_over_fact[i];
	}

	++prev_pt;
	++next_pt;

	for (; next_pt != end_pt; ++prev_pt, ++next_pt) {

		T last_coef = coefs[0]; // = 1
		for (uint64_t i = 1; i < degree + 1; ++i) {

			// At every step:
			// buff[j] = coefs[j] * prod_{k=j+1}^{i} incr[k],
			
			// So that Chen's relation is:
			// coefs[i] = coefs[i] + sum_{j=0}^{i-1} buff[j] * one_over_fact[i - j]

			const T new_incr = next_pt[multi_idx[i - 1]] - prev_pt[multi_idx[i - 1]];

			for (uint64_t j = 0; j < i - 1; ++j) {
				buff[j] *= new_incr;
			}

			buff[i - 1] = last_coef * new_incr;

			T acc = static_cast<T>(0.);
			for (uint64_t j = 0; j < i; ++j) {
				acc += buff[j] * one_over_fact[i - j];
			}
			last_coef = coefs[i];
			coefs[i] += acc;
		}
	}

	if (!prefixes) {
		*out = coefs[degree];
	}
	else {
		for (uint64_t i = 0; i < degree; ++i) {
			out[i] = coefs[i + 1];
		}
	}
}

template<std::floating_point T, uint64_t degree>
void single_sig_coef_template_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	T* coefs,
	T* buff,
	const T* one_over_fact,
	bool prefixes
) {
	single_sig_coef_(path, out, multi_idx, degree, coefs, buff, one_over_fact, prefixes);
}

template<std::floating_point T>
void call_single_sig_coef_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t degree,
	T* coefs,
	T* buff,
	const T* one_over_fact,
	bool prefixes
) {
	switch (degree) {
	case 1:  return single_sig_coef_template_<T, 1>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 2:  return single_sig_coef_template_<T, 2>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 3:  return single_sig_coef_template_<T, 3>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 4:  return single_sig_coef_template_<T, 4>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 5:  return single_sig_coef_template_<T, 5>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 6:  return single_sig_coef_template_<T, 6>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 7:  return single_sig_coef_template_<T, 7>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 8:  return single_sig_coef_template_<T, 8>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 9:  return single_sig_coef_template_<T, 9>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 10: return single_sig_coef_template_<T, 10>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 11: return single_sig_coef_template_<T, 11>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 12: return single_sig_coef_template_<T, 12>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 13: return single_sig_coef_template_<T, 13>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 14: return single_sig_coef_template_<T, 14>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 15: return single_sig_coef_template_<T, 15>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 16: return single_sig_coef_template_<T, 16>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 17: return single_sig_coef_template_<T, 17>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 18: return single_sig_coef_template_<T, 18>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 19: return single_sig_coef_template_<T, 19>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	case 20: return single_sig_coef_template_<T, 20>(path, out, multi_idx, coefs, buff, one_over_fact, prefixes);
	default:
		return single_sig_coef_<T>(path, out, multi_idx, degree, coefs, buff, one_over_fact, prefixes);
	}

}


template<std::floating_point T>
void sig_coef_(
	const T* path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t num_multi_idx, // len(multi_idx)
	const uint64_t* degrees, // [ len(multi_index[i]) for i in 0:num_multi_index ]
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time,
	bool prefixes
) {

	if (dimension == 0) { throw std::invalid_argument("sig_coef received path of dimension 0"); }

	uint64_t max_degree = 0;
	uint64_t result_length = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		max_degree = std::max(max_degree, degrees[i]);
		result_length += (prefixes && degrees[i]) ? degrees[i] : 1;
	}

	if (length <= 1) {
		std::fill(out, out + result_length, static_cast<T>(0.));
		return;
	}

	//TODO: check indices < dim

	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);

	T* out_ptr = out;

	auto buff_uptr = std::make_unique<T[]>(max_degree + 1);
	T* buff = buff_uptr.get();

	auto one_over_fact_uptr = std::make_unique<T[]>(max_degree + 1);
	T* one_over_fact = one_over_fact_uptr.get();

	one_over_fact[0] = static_cast<T>(1.);
	for (uint64_t i = 1; i < max_degree + 1; ++i) {
		one_over_fact[i] = one_over_fact[i - 1] / i;
	}

	auto coefs_uptr = std::make_unique<T[]>(max_degree + 1);
	T* coefs = coefs_uptr.get();

	const uint64_t* multi_idx_ptr = multi_idx;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		uint64_t degree = degrees[i];

		if (!degree) {
			*out_ptr = static_cast<T>(1.);
			++out_ptr;
			continue;
		}

		call_single_sig_coef_<T>(path_obj, out_ptr, multi_idx_ptr, degree, coefs, buff, one_over_fact, prefixes);
		out_ptr += prefixes ? degree : 1;
		multi_idx_ptr += degree;
	}

}

template<std::floating_point T>
void sig_coef_(
	const T* path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t num_multi_idx, // len(multi_idx)
	const uint64_t* degrees, // [ len(multi_index[i]) for i in 0:num_multi_index ]
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time,
	bool prefixes,
	int n_jobs
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("sig_coef received path of dimension 0"); }

	Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	//General case and degree = 1 case
	const uint64_t flat_path_length = dimension * length;
	uint64_t result_length = 0;

	if (prefixes) {
		for (uint64_t i = 0; i < num_multi_idx; ++i)
			result_length += degrees[i] ? degrees[i] : 1;
	}
	else {
		result_length = num_multi_idx;
	}

	auto sig_func = [&](const T* path_ptr, T* out_ptr) {
		sig_coef_<T>(path_ptr, out_ptr, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time, prefixes);
	};

	multi_threaded_batch(sig_func, path, out, batch_size, flat_path_length, result_length, n_jobs);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// backpropagation
//////////////////////////////////////////////////////////////////////////////////////////////

FORCE_INLINE bool sig_coef_backprop_skip(uint64_t data_dimension, uint64_t pre_time_aug_dim, uint64_t idx, bool lead_lag, bool parity) {
	// Determine whether the derivative with respect to incr[i] needs to be computed, or can be skipped

	return (idx >= pre_time_aug_dim) || (lead_lag && (parity == (idx < data_dimension)));
}

template<std::floating_point T>
FORCE_INLINE void uncombine_coefs_(
	T* coefs,
	const T* incr,
	T* buff,
	uint64_t degree,
	const T* signed_one_over_fact
) {
	// Reconstruct coefs for previous time step using Chen's inverse relation:
	// S(x_{1:n-1}) = S(x_{1:n}) * S(-x_{n-1:n})
	//
	// coefs[i] = coefs[i]
	//    + sum_{j=0}^{i-1} coefs[j] * prod_{k=j}^{i-1} incr[k] * (-1)^{i-j} / (i-j)!
	//
	// Maintain buff[j] = coefs[j] * prod_{k=j}^{i-1} incr[k] incrementally:
	// at each step i, multiply existing entries by incr[i-1]
	// and add the new entry buff[i-1] = coefs[i-1] * incr[i-1].
	// Overwrite coefs in place, using last_coef to hold the original coefs[i-1].

	T last_coef = coefs[0]; // = 1
	for (uint64_t i = 1; i < degree + 1; ++i) {
		const T inc = incr[i - 1];

		for (uint64_t j = 0; j < i - 1; ++j) {
			buff[j] *= inc;
		}

		buff[i - 1] = last_coef * inc;

		T acc = static_cast<T>(0.);
		for (uint64_t j = 0; j < i; ++j) {
			acc += buff[j] * signed_one_over_fact[i - j];
		}
		last_coef = coefs[i];
		coefs[i] += acc;
	}
}

template<std::floating_point T>
FORCE_INLINE void update_path_derivs_(
	const uint64_t* multi_idx,
	uint64_t degree,
	const T* coefs,
	const T* incr,
	T* buff,
	const T* derivs,
	const T* one_over_fact,
	uint64_t data_dimension,
	uint64_t pre_time_aug_dim,
	bool lead_lag,
	bool parity,
	T* pos,
	T* neg
) {
	// Update path derivs
	// dL / d incr[i] = sum_k (dL / d next_coefs[k]) * (d next_coefs[k] / d incr[i])
	// where next_coefs are S(x_{1:n})
	// Then backprop incr derivs -> path derivs.

	// i = 0 case: compute right-growing products on the fly
	T update;
	uint64_t idx = multi_idx[0];
	if (!sig_coef_backprop_skip(data_dimension, pre_time_aug_dim, idx, lead_lag, parity)) {
		update = derivs[0];
		T right_prod = static_cast<T>(1.);
		for (uint64_t m = 1; m < degree; ++m) {
			right_prod *= incr[m];
			update += derivs[m] * right_prod * one_over_fact[m + 1];
		}
		update *= coefs[0];

		// incr derivs -> path derivs
		if (lead_lag && parity) {
			idx -= data_dimension;
		}
		pos[idx] += update;
		neg[idx] -= update;
	}

	// i >= 1 case: maintain buff[k] = coefs[k] * prod_{l=k+1}^{i} incr[l]
	// incrementally across iterations of i
	for (uint64_t i = 1; i < degree; ++i) {

		// Maintain buff incrementally (even when skipping)
		const T inc = incr[i - 1];
		for (uint64_t k = 0; k < i - 1; ++k) {
			buff[k] *= inc;
		}
		buff[i - 1] = coefs[i - 1] * inc;

		idx = multi_idx[i];
		if (!sig_coef_backprop_skip(data_dimension, pre_time_aug_dim, idx, lead_lag, parity)) {

			// Left part: compute s using the incrementally maintained buff
			T s = coefs[i];
			for (uint64_t k = 0; k < i; ++k) {
				s += buff[k] * one_over_fact[i - k + 1];
			}
			update = derivs[i] * s;

			// Right part: compute right-growing products on the fly
			T right_prod = static_cast<T>(1.);
			for (uint64_t m = i + 1; m < degree; ++m) {
				right_prod *= incr[m];
				s = coefs[i] * one_over_fact[m - i + 1];
				for (uint64_t k = 0; k < i; ++k) {
					s += buff[k] * one_over_fact[m - k + 1];
				}
				s *= right_prod;
				update += derivs[m] * s;
			}

			// incr derivs -> path derivs
			if (lead_lag && parity) {
				idx -= data_dimension;
			}
			pos[idx] += update;
			neg[idx] -= update;
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void update_coef_derivs_(
	T* derivs,
	const T* incr,
	uint64_t degree,
	const T* one_over_fact
) {
	// Update sig coef derivs (derivs[i] = dL / d coefs[i])
	// dL / d coefs[i] += sum_{k=i+1}^{degree-1} derivs[k] * prod_{l=i+1}^{k} incr[l] / (k-i)!
	//
	// Compute right-growing products prod_{l=i+1}^{k} incr[l] on the fly.

	for (uint64_t i = 0; i < degree - 1; ++i) {
		T acc = static_cast<T>(0.);
		T right_prod = static_cast<T>(1.);
		for (uint64_t k = i + 1; k < degree; ++k) {
			right_prod *= incr[k];
			acc += derivs[k] * right_prod * one_over_fact[k - i];
		}
		derivs[i] += acc;
	}
}

template<std::floating_point T>
FORCE_INLINE void single_sig_coef_backprop_(
	const Path<T>& path,
	T* out, // Should be zeroed
	const uint64_t* multi_idx,
	uint64_t degree,
	T* coefs,
	T* incr,
	T* buff,
	T* derivs,
	const T* one_over_fact,
	const T* signed_one_over_fact
) {
	const bool lead_lag = path.lead_lag();
	const uint64_t pre_time_aug_dim = path.dimension() - (path.time_aug() ? 1 : 0);
	const uint64_t data_dimension = path.data_dimension();
	const uint64_t data_length = path.data_length();

	Point<T> next_pt(--path.end());
	Point<T> prev_pt(----path.end());
	const Point<T> first_pt(path.begin());

	T* pos = out + data_dimension * (data_length - 1);
	T* neg = pos - data_dimension;
	bool parity = false;

	for (; next_pt != first_pt; --next_pt, --prev_pt, parity = !parity) {

		// Populate incr
		for (uint64_t i = 0; i < degree; ++i) {
			const uint64_t idx = multi_idx[i];
			incr[i] = next_pt[idx] - prev_pt[idx];
		}

		uncombine_coefs_(coefs, incr, buff, degree, signed_one_over_fact);

		update_path_derivs_(multi_idx, degree, coefs, incr, buff, derivs, one_over_fact,
			data_dimension, pre_time_aug_dim, lead_lag, parity, pos, neg);

		update_coef_derivs_(derivs, incr, degree, one_over_fact);

		if (!lead_lag || parity) {
			pos -= data_dimension;
			neg -= data_dimension;
		}
	}
}

template<std::floating_point T, uint64_t degree>
void single_sig_coef_backprop_template_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	T* coefs,
	T* incr,
	T* buff,
	T* derivs,
	const T* one_over_fact,
	const T* signed_one_over_fact
) {
	single_sig_coef_backprop_(path, out, multi_idx, degree, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
}

template<std::floating_point T>
void call_single_sig_coef_backprop_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t degree,
	T* coefs,
	T* incr,
	T* buff,
	T* derivs,
	const T* one_over_fact,
	const T* signed_one_over_fact
) {
	switch (degree) {
	case 1:  return single_sig_coef_backprop_template_<T, 1>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 2:  return single_sig_coef_backprop_template_<T, 2>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 3:  return single_sig_coef_backprop_template_<T, 3>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 4:  return single_sig_coef_backprop_template_<T, 4>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 5:  return single_sig_coef_backprop_template_<T, 5>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 6:  return single_sig_coef_backprop_template_<T, 6>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 7:  return single_sig_coef_backprop_template_<T, 7>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 8:  return single_sig_coef_backprop_template_<T, 8>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 9:  return single_sig_coef_backprop_template_<T, 9>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 10: return single_sig_coef_backprop_template_<T, 10>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 11: return single_sig_coef_backprop_template_<T, 11>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 12: return single_sig_coef_backprop_template_<T, 12>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 13: return single_sig_coef_backprop_template_<T, 13>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 14: return single_sig_coef_backprop_template_<T, 14>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 15: return single_sig_coef_backprop_template_<T, 15>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 16: return single_sig_coef_backprop_template_<T, 16>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 17: return single_sig_coef_backprop_template_<T, 17>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 18: return single_sig_coef_backprop_template_<T, 18>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 19: return single_sig_coef_backprop_template_<T, 19>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	case 20: return single_sig_coef_backprop_template_<T, 20>(path, out, multi_idx, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	default:
		return single_sig_coef_backprop_(path, out, multi_idx, degree, coefs, incr, buff, derivs, one_over_fact, signed_one_over_fact);
	}


}


template<std::floating_point T>
void sig_coef_backprop_(
	const T* path,
	T* out,
	const T* coefs,
	T* derivs,
	const uint64_t* multi_idx,
	uint64_t num_multi_idx, // len(multi_idx)
	const uint64_t* degrees, // [ len(multi_index[i]) for i in 0:num_multi_index ]
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time
) {

	if (dimension == 0) { throw std::invalid_argument("sig_coef_backprop received path of dimension 0"); }

	const uint64_t path_length = dimension * length;
	std::fill(out, out + path_length, static_cast<T>(0.));

	if (length <= 1) {
		return;
	}

	//TODO: check indices < dim

	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);

	uint64_t max_degree = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		max_degree = std::max(max_degree, degrees[i]);
	}

	auto one_over_fact_uptr = std::make_unique<T[]>(max_degree + 1);
	T* one_over_fact = one_over_fact_uptr.get();

	auto signed_one_over_fact_uptr = std::make_unique<T[]>(max_degree + 1);
	T* signed_one_over_fact = signed_one_over_fact_uptr.get();

	one_over_fact[0] = static_cast<T>(1.);
	signed_one_over_fact[0] = static_cast<T>(1.);
	T sgn = static_cast<T>(-1.);
	for (uint64_t i = 1; i < max_degree + 1; ++i, sgn = -sgn) {
		one_over_fact[i] = one_over_fact[i - 1] / i;
		signed_one_over_fact[i] = sgn * one_over_fact[i];
	}

	auto coefs_copy_uptr = std::make_unique<T[]>(max_degree + 1);
	T* coefs_copy = coefs_copy_uptr.get();

	auto incr_uptr = std::make_unique<T[]>(max_degree);
	T* incr = incr_uptr.get();

	auto buff_uptr = std::make_unique<T[]>(max_degree + 1);
	T* buff = buff_uptr.get();

	const uint64_t* multi_idx_ptr = multi_idx;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		uint64_t degree = degrees[i];

		if (!degree) {
			coefs += 1;
			derivs += 1;
			continue;
		}

		coefs_copy[0] = static_cast<T>(1.);
		std::copy(coefs, coefs + degree, coefs_copy + 1);

		call_single_sig_coef_backprop_<T>(path_obj, out, multi_idx_ptr, degree, coefs_copy, incr, buff, derivs, one_over_fact, signed_one_over_fact);
		coefs += degree;
		derivs += degree;
		multi_idx_ptr += degree;
	}

}

template<std::floating_point T>
void sig_coef_backprop_(
	const T* path,
	T* out,
	const T* coefs,
	T* derivs,
	const uint64_t* multi_idx,
	uint64_t num_multi_idx, // len(multi_idx)
	const uint64_t* degrees, // [ len(multi_index[i]) for i in 0:num_multi_index ]
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time,
	int n_jobs
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("sig_coef_backprop received path of dimension 0"); }

	Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	//General case and degree = 1 case
	const uint64_t flat_path_length = dimension * length;
	uint64_t coefs_len = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		coefs_len += degrees[i] ? degrees[i] : 1;
	}

	auto sig_func = [&](const T* path_ptr, const T* coefs_ptr, T* derivs_ptr, T* out_ptr) {
		sig_coef_backprop_<T>(path_ptr, out_ptr, coefs_ptr, derivs_ptr, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time);
	};

	multi_threaded_batch_3(sig_func, path, coefs, derivs, out, batch_size, flat_path_length, coefs_len, coefs_len, flat_path_length, n_jobs);
}

extern "C" {


	CPSIG_API int sig_coef_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, bool prefixes, int n_jobs) noexcept {
		SAFE_CALL(sig_coef_<float>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs));
	}

	CPSIG_API int sig_coef_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, bool prefixes, int n_jobs) noexcept {
		SAFE_CALL(sig_coef_<double>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs));
	}


	CPSIG_API int sig_coef_backprop_f(const float* path, float* out, const float* coefs, float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, int n_jobs) noexcept {
		SAFE_CALL(sig_coef_backprop_<float>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int sig_coef_backprop_d(const double* path, double* out, const double* coefs, double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(sig_coef_backprop_<double>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

}
