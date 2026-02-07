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
void single_sig_coef_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	uint64_t degree,
	T* prev_coefs,
	T* next_coefs,
	T* incr_prod,
	const T* one_over_fact,
	bool prefixes
) {
	Point<T> prev_pt(path.begin());
	Point<T> next_pt(++path.begin());
	const Point<T> end_pt(path.end());

	prev_coefs[0] = static_cast<T>(1.);
	next_coefs[0] = static_cast<T>(1.);

	incr_prod[0] = static_cast<T>(1.);

	for (uint64_t i = 1; i < degree + 1; ++i) {
		incr_prod[i] = incr_prod[i - 1] * (next_pt[multi_idx[i - 1]] - prev_pt[multi_idx[i - 1]]);
	}

	for (uint64_t i = 1; i < degree + 1; ++i) {
		prev_coefs[i] = incr_prod[i] * one_over_fact[i];
	}

	++prev_pt;
	++next_pt;

	for (; next_pt != end_pt; ++prev_pt, ++next_pt) {

		// incr_prod here takes a different role than before
		// At step i, incr_prod[i - k] = prod_{j=k}^i incr[j]

		for (uint64_t i = 1; i < degree + 1; ++i) {
			next_coefs[i] = prev_coefs[i];

			const T new_incr = next_pt[multi_idx[i - 1]] - prev_pt[multi_idx[i - 1]];
			incr_prod[i] = new_incr;

			for (uint64_t k = 1; k < i; ++k) {
				incr_prod[k] *= new_incr;
			}
			
			// Can we vectorise this?
			for (uint64_t k = 1; k <= i; ++k) {
				next_coefs[i] += prev_coefs[i - k] * incr_prod[i - k + 1] * one_over_fact[k];
			}
		}

		std::swap(next_coefs, prev_coefs);
	}

	if (!prefixes) {
		*out = prev_coefs[degree];
	}
	else {
		for (uint64_t i = 0; i < degree; ++i) {
			out[i] = prev_coefs[i + 1];
		}
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

	if (length <= 1) {
		std::fill(out, out + num_multi_idx, static_cast<T>(0.));
		return;
	}

	//TODO: check indices < dim

	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);

	T* out_ptr = out;

	// Each buffer is of length (len(multi_indices[i]) + 1)
	// So we need a total size of sum{ len(multi_indices[i]) + 1 } = sum{ len(multi_indices[i]) } + len(multi_indices)
	uint64_t coef_buffer_len = num_multi_idx;
	uint64_t max_degree = 0;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		coef_buffer_len += degrees[i];
		max_degree = std::max(max_degree, degrees[i]);
	}

	auto incr_prod_uptr = std::make_unique<T[]>(max_degree + 1);
	T* incr_prod = incr_prod_uptr.get();

	auto one_over_fact_uptr = std::make_unique<T[]>(max_degree + 1);
	T* one_over_fact = one_over_fact_uptr.get();

	one_over_fact[0] = 1.;
	for (uint64_t i = 1; i < max_degree + 1; ++i) {
		one_over_fact[i] = one_over_fact[i - 1] / i;
	}

	auto prev_coefs_uptr = std::make_unique<T[]>(coef_buffer_len);
	T* prev_coefs = prev_coefs_uptr.get();

	auto next_coefs_uptr = std::make_unique<T[]>(coef_buffer_len);
	T* next_coefs = next_coefs_uptr.get();

	const uint64_t* multi_idx_ptr = multi_idx;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		single_sig_coef_<T>(path_obj, out_ptr, multi_idx_ptr, degrees[i], prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
		out_ptr += prefixes ? degrees[i] : 1;
		prev_coefs += degrees[i] + 1;
		next_coefs += degrees[i] + 1;
		multi_idx_ptr += degrees[i];
	}

}

template<std::floating_point T>
void batch_sig_coef_(
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
	const T* const data_end = path + flat_path_length * batch_size;

	std::function<void(const T*, T*)> sig_func;

	sig_func = [&](const T* path_ptr, T* out_ptr) {
		sig_coef_<T>(path_ptr, out_ptr, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time, prefixes);
		};

	const T* path_ptr;
	T* out_ptr;

	if (n_jobs != 1) {
		multi_threaded_batch(sig_func, path, out, batch_size, flat_path_length, num_multi_idx, n_jobs);
	}
	else {
		for (path_ptr = path, out_ptr = out;
			path_ptr < data_end;
			path_ptr += flat_path_length, out_ptr += num_multi_idx) {

			sig_func(path_ptr, out_ptr);
		}
	}
	return;
}

extern "C" {

	CPSIG_API int sig_coef_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, bool prefixes) noexcept {
		SAFE_CALL(sig_coef_<float>(path, out, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time, prefixes));
	}

	CPSIG_API int sig_coef_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, bool prefixes) noexcept {
		SAFE_CALL(sig_coef_<double>(path, out, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time, prefixes));
	}

	CPSIG_API int batch_sig_coef_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, bool prefixes, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_coef_<float>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs));
	}

	CPSIG_API int batch_sig_coef_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, bool prefixes, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_coef_<double>(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs));
	}

}
