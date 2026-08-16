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
	T* incr,
	const T* one_over_fact,
	bool prefixes
) {
	Point<T> prev_pt(path.begin());
	Point<T> next_pt(++path.begin());
	const Point<T> end_pt(path.end());

	coefs[0] = static_cast<T>(1.);

	T product = static_cast<T>(1.);
	for (uint64_t i = 0; i < degree; ++i) {
		incr[i] = next_pt[multi_idx[i]] - prev_pt[multi_idx[i]];
		product *= incr[i];
		coefs[i + 1] = product * one_over_fact[i + 1];
	}

	++prev_pt;
	++next_pt;

	for (; next_pt != end_pt; ++prev_pt, ++next_pt) {
		for (uint64_t i = 0; i < degree; ++i) {
			incr[i] = next_pt[multi_idx[i]] - prev_pt[multi_idx[i]];
		}

		for (uint64_t i = degree; i > 0; --i) {
			T acc = one_over_fact[i];
			for (uint64_t j = 0; j + 1 < i; ++j) {
				acc = acc * incr[j] + coefs[j + 1] * one_over_fact[i - j - 1];
			}
			coefs[i] += acc * incr[i - 1];
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
	T* incr,
	const T* one_over_fact,
	bool prefixes
) {
	single_sig_coef_(path, out, multi_idx, degree, coefs, incr, one_over_fact, prefixes);
}

template<std::floating_point T>
void call_single_sig_coef_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t degree,
	T* coefs,
	T* incr,
	const T* one_over_fact,
	bool prefixes
) {
	switch (degree) {
	case 1:  return single_sig_coef_template_<T, 1>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 2:  return single_sig_coef_template_<T, 2>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 3:  return single_sig_coef_template_<T, 3>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 4:  return single_sig_coef_template_<T, 4>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 5:  return single_sig_coef_template_<T, 5>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 6:  return single_sig_coef_template_<T, 6>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 7:  return single_sig_coef_template_<T, 7>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 8:  return single_sig_coef_template_<T, 8>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 9:  return single_sig_coef_template_<T, 9>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 10: return single_sig_coef_template_<T, 10>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 11: return single_sig_coef_template_<T, 11>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 12: return single_sig_coef_template_<T, 12>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 13: return single_sig_coef_template_<T, 13>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 14: return single_sig_coef_template_<T, 14>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 15: return single_sig_coef_template_<T, 15>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 16: return single_sig_coef_template_<T, 16>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 17: return single_sig_coef_template_<T, 17>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 18: return single_sig_coef_template_<T, 18>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 19: return single_sig_coef_template_<T, 19>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	case 20: return single_sig_coef_template_<T, 20>(path, out, multi_idx, coefs, incr, one_over_fact, prefixes);
	default:
		return single_sig_coef_<T>(path, out, multi_idx, degree, coefs, incr, one_over_fact, prefixes);
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
	uint64_t idx_total = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		max_degree = std::max(max_degree, degrees[i]);
		result_length += (prefixes && degrees[i]) ? degrees[i] : 1;
		idx_total += degrees[i];
	}

	const uint64_t aug_dim = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	for (uint64_t k = 0; k < idx_total; ++k)
		if (multi_idx[k] >= aug_dim)
			throw std::invalid_argument("sig_coef: multi_idx element out of range");

	if (length <= 1) {
		std::fill(out, out + result_length, static_cast<T>(0.));
		return;
	}

	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);

	T* out_ptr = out;

	auto workspace = std::make_unique<T[]>(3 * max_degree + 2);
	T* incr = workspace.get();
	T* one_over_fact = incr + max_degree;
	T* coefs = one_over_fact + max_degree + 1;

	one_over_fact[0] = static_cast<T>(1.);
	for (uint64_t i = 1; i < max_degree + 1; ++i) {
		one_over_fact[i] = one_over_fact[i - 1] / i;
	}

	const uint64_t* multi_idx_ptr = multi_idx;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		uint64_t degree = degrees[i];

		if (!degree) {
			*out_ptr = static_cast<T>(1.);
			++out_ptr;
			continue;
		}

		call_single_sig_coef_<T>(path_obj, out_ptr, multi_idx_ptr, degree, coefs, incr, one_over_fact, prefixes);
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

	multi_threaded_batch(sig_func, batch_size, n_jobs,
		make_batch(path, flat_path_length), make_batch(out, result_length));
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
	uint64_t degree,
	const T* signed_one_over_fact
) {
	// Reconstruct coefs for previous time step using Chen's inverse relation:
	// S(x_{1:n-1}) = S(x_{1:n}) * S(-x_{n-1:n})
	//
	// coefs[i] = coefs[i]
	//    + sum_{j=0}^{i-1} coefs[j] * prod_{k=j}^{i-1} incr[k] * (-1)^{i-j} / (i-j)!
	//
	for (uint64_t i = degree; i > 0; --i) {
		T acc = signed_one_over_fact[i];
		for (uint64_t j = 0; j + 1 < i; ++j) {
			acc = acc * incr[j] + coefs[j + 1] * signed_one_over_fact[i - j - 1];
		}
		coefs[i] += acc * incr[i - 1];
	}
}

template<std::floating_point T>
FORCE_INLINE void backprop_horner_(
	const uint64_t* multi_idx,
	uint64_t degree,
	const T* coefs,
	const T* incr,
	T* states,
	T* incr_derivs,
	T* derivs,
	const T* one_over_fact,
	uint64_t data_dimension,
	uint64_t pre_time_aug_dim,
	bool lead_lag,
	bool parity,
	T* pos,
	T* neg
) {
	std::fill(incr_derivs, incr_derivs + degree, static_cast<T>(0.));

	for (uint64_t i = 1; i < degree + 1; ++i) {
		states[0] = one_over_fact[i];
		for (uint64_t j = 0; j + 1 < i; ++j) {
			states[j + 1] = states[j] * incr[j]
				+ coefs[j + 1] * one_over_fact[i - j - 1];
		}

		const T output_deriv = derivs[i - 1];
		incr_derivs[i - 1] += output_deriv * states[i - 1];

		T state_deriv = output_deriv * incr[i - 1];
		for (uint64_t j = i - 1; j-- > 0;) {
			incr_derivs[j] += state_deriv * states[j];
			derivs[j] += state_deriv * one_over_fact[i - j - 1];
			state_deriv *= incr[j];
		}
	}

	for (uint64_t i = 0; i < degree; ++i) {
		uint64_t idx = multi_idx[i];
		if (sig_coef_backprop_skip(data_dimension, pre_time_aug_dim, idx, lead_lag, parity)) {
			continue;
		}
		if (lead_lag && parity) {
			idx -= data_dimension;
		}
		pos[idx] += incr_derivs[i];
		neg[idx] -= incr_derivs[i];
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
	T* states,
	T* incr_derivs,
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

	while (next_pt != first_pt) {

		// Populate incr
		for (uint64_t i = 0; i < degree; ++i) {
			const uint64_t idx = multi_idx[i];
			incr[i] = next_pt[idx] - prev_pt[idx];
		}

		uncombine_coefs_(coefs, incr, degree, signed_one_over_fact);

		backprop_horner_(multi_idx, degree, coefs, incr, states, incr_derivs, derivs, one_over_fact,
			data_dimension, pre_time_aug_dim, lead_lag, parity, pos, neg);

		--next_pt;
		if (next_pt != first_pt) {
			--prev_pt;
			if (!lead_lag || parity) {
				pos -= data_dimension;
				neg -= data_dimension;
			}
			parity = !parity;
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
	T* states,
	T* incr_derivs,
	T* derivs,
	const T* one_over_fact,
	const T* signed_one_over_fact
) {
	single_sig_coef_backprop_(path, out, multi_idx, degree, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
}

template<std::floating_point T>
void call_single_sig_coef_backprop_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t degree,
	T* coefs,
	T* incr,
	T* states,
	T* incr_derivs,
	T* derivs,
	const T* one_over_fact,
	const T* signed_one_over_fact
) {
	switch (degree) {
	case 1:  return single_sig_coef_backprop_template_<T, 1>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 2:  return single_sig_coef_backprop_template_<T, 2>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 3:  return single_sig_coef_backprop_template_<T, 3>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 4:  return single_sig_coef_backprop_template_<T, 4>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 5:  return single_sig_coef_backprop_template_<T, 5>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 6:  return single_sig_coef_backprop_template_<T, 6>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 7:  return single_sig_coef_backprop_template_<T, 7>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 8:  return single_sig_coef_backprop_template_<T, 8>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 9:  return single_sig_coef_backprop_template_<T, 9>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 10: return single_sig_coef_backprop_template_<T, 10>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 11: return single_sig_coef_backprop_template_<T, 11>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 12: return single_sig_coef_backprop_template_<T, 12>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 13: return single_sig_coef_backprop_template_<T, 13>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 14: return single_sig_coef_backprop_template_<T, 14>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 15: return single_sig_coef_backprop_template_<T, 15>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 16: return single_sig_coef_backprop_template_<T, 16>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 17: return single_sig_coef_backprop_template_<T, 17>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 18: return single_sig_coef_backprop_template_<T, 18>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 19: return single_sig_coef_backprop_template_<T, 19>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	case 20: return single_sig_coef_backprop_template_<T, 20>(path, out, multi_idx, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
	default:
		return single_sig_coef_backprop_(path, out, multi_idx, degree, coefs, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
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

	uint64_t max_degree = 0;
	uint64_t idx_total = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		max_degree = std::max(max_degree, degrees[i]);
		idx_total += degrees[i];
	}

	const uint64_t aug_dim = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	for (uint64_t k = 0; k < idx_total; ++k)
		if (multi_idx[k] >= aug_dim)
			throw std::invalid_argument("sig_coef_backprop: multi_idx element out of range");

	const uint64_t path_length = dimension * length;
	std::fill(out, out + path_length, static_cast<T>(0.));

	if (length <= 1) {
		return;
	}

	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);

	auto workspace = std::make_unique<T[]>(6 * max_degree + 3);
	T* one_over_fact = workspace.get();
	T* signed_one_over_fact = one_over_fact + max_degree + 1;
	T* coefs_copy = signed_one_over_fact + max_degree + 1;
	T* incr = coefs_copy + max_degree + 1;
	T* states = incr + max_degree;
	T* incr_derivs = states + max_degree;

	one_over_fact[0] = static_cast<T>(1.);
	signed_one_over_fact[0] = static_cast<T>(1.);
	T sgn = static_cast<T>(-1.);
	for (uint64_t i = 1; i < max_degree + 1; ++i, sgn = -sgn) {
		one_over_fact[i] = one_over_fact[i - 1] / i;
		signed_one_over_fact[i] = sgn * one_over_fact[i];
	}

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

		call_single_sig_coef_backprop_<T>(path_obj, out, multi_idx_ptr, degree, coefs_copy, incr, states, incr_derivs, derivs, one_over_fact, signed_one_over_fact);
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

	multi_threaded_batch(sig_func, batch_size, n_jobs,
		make_batch(path, flat_path_length), make_batch(coefs, coefs_len), make_batch(derivs, coefs_len), make_batch(out, flat_path_length));
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
