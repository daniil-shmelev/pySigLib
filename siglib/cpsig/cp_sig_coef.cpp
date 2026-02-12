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
		uint64_t degree = degrees[i];

		if (!degree) {
			*out_ptr = static_cast<T>(1.);
			++out_ptr;
			continue;
		}

		single_sig_coef_<T>(path_obj, out_ptr, multi_idx_ptr, degree, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
		out_ptr += prefixes ? degree : 1;
		prev_coefs += degree + 1;
		next_coefs += degree + 1;
		multi_idx_ptr += degree;
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
	uint64_t result_length = 0;

	if (prefixes) {
		for (uint64_t i = 0; i < num_multi_idx; ++i)
			result_length += degrees[i] ? degrees[i] : 1;
	}
	else {
		result_length = num_multi_idx;
	}

	const T* const data_end = path + flat_path_length * batch_size;

	std::function<void(const T*, T*)> sig_func;

	sig_func = [&](const T* path_ptr, T* out_ptr) {
		sig_coef_<T>(path_ptr, out_ptr, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time, prefixes);
		};

	const T* path_ptr;
	T* out_ptr;

	if (n_jobs != 1) {
		multi_threaded_batch(sig_func, path, out, batch_size, flat_path_length, result_length, n_jobs);
	}
	else {
		for (path_ptr = path, out_ptr = out;
			path_ptr < data_end;
			path_ptr += flat_path_length, out_ptr += result_length) {

			sig_func(path_ptr, out_ptr);
		}
	}
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// backpropagation
//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class UpperTriangularMatrix {
public:
	explicit UpperTriangularMatrix(size_t n)
		: n_(n), data_(n* (n + 1) / 2) {
	}

	size_t size() const { return n_; }

	// Write access
	T& operator()(size_t i, size_t j) {
		if (i > j)
			throw std::out_of_range("Accessing non-strict upper element");
		/*uint64_t idx = index(i, j);
		T res = data_[index(i, j)];
		return res;*/
		return data_[index(i, j)];
	}

	// Read access
	T operator()(size_t i, size_t j) const {
		if (i > j)
			return T(1.);
		return data_[index(i, j)];
	}

	T get_neg(size_t i, size_t j) const {
		if (i > j)
			return T(1.);
		if ((i + j) % 2)
			return data_[index(i, j)];
		else
			return -data_[index(i, j)];
	}

private:
	size_t n_;
	std::vector<T> data_;

	size_t index(size_t i, size_t j) const {
		return i * n_ - (i * (i - 1)) / 2 + (j - i);
	}
};

template<std::floating_point T>
void single_sig_coef_backprop_(
	const Path<T>& path,
	T* out, // Should be zeroed
	const T* coefs, // This should be an array of size degree, of the prefixes of the coeff.
	const uint64_t* multi_idx,
	uint64_t degree,
	T* prev_coefs,
	T* next_coefs,
	T* incr,
	UpperTriangularMatrix<T>& incr_prod, // Array of size degree * degree, to be filled with incr_prod[i,j] = prod_{k = i}^j incr[k]
	T* prev_derivs, // Derivs wrt coeffs. Should be the same length as the above array.
	T* next_derivs,
	const T* one_over_fact
) {
	const bool lead_lag = path.lead_lag();
	const uint64_t pre_time_aug_dim = path.dimension() - (path.time_aug() ? 1 : 0);
	const uint64_t data_dimension = path.data_dimension();
	const uint64_t data_length = path.data_length();

	Point<T> next_pt(--path.end());
	Point<T> prev_pt(----path.end());
	const Point<T> first_pt(path.begin());

	prev_coefs[0] = static_cast<T>(1.);
	for (uint64_t i = 0; i < degree; ++i) {
		prev_coefs[i + 1] = coefs[i];
	}
	next_coefs[0] = static_cast<T>(1.);

	T* pos = out + data_dimension * (data_length - 1);
	T* neg = pos - data_dimension;

	for (; next_pt != first_pt; --next_pt, --prev_pt, pos -= data_dimension, neg -= data_dimension) {

		// Populate incr
		for (uint64_t i = 0; i < degree; ++i) {
			incr[i] = prev_pt[multi_idx[i]] - next_pt[multi_idx[i]];
		}

		// Populate incr_prod
		for (int64_t i = degree - 1; i >= 0; --i) {
			incr_prod(i, i) = incr[i];
			for (uint64_t j = i + 1; j < degree; ++j) {
				incr_prod(i, j) = incr[j] * incr_prod(i, j - 1);
			}
		}

		// Reconstruct coefs
		for (uint64_t i = 1; i < degree + 1; ++i) {
			next_coefs[i] = prev_coefs[i];

			for (uint64_t k = 1; k <= i; ++k) {
				next_coefs[i] += prev_coefs[i - k] * incr_prod(i - k, i - 1) * one_over_fact[k];
			}
		}

		// Update path derivs
		T update = 0.;
		for (uint64_t m = 0; m < degree; ++m) {
			update += prev_derivs[m] * next_coefs[0] * incr_prod.get_neg(1, m) * one_over_fact[m + 1];;
		}
		pos[multi_idx[0]] += update;
		neg[multi_idx[0]] -= update;

		for (uint64_t i = 1; i < degree; ++i) {
			T update = 0.;
			for (uint64_t m = i; m < degree; ++m) {
				T s = 0;
				for (uint64_t k = 0; k <= i; ++k) {
					s += next_coefs[k] * incr_prod.get_neg(k, i-1) * one_over_fact[m - k + 1];
				}
				s *= incr_prod.get_neg(i + 1, m);
				update += prev_derivs[m] * s;
			}
			
			pos[multi_idx[i]] += update;
			neg[multi_idx[i]] -= update;
		}


		// Update sig coef derivs
		for (uint64_t i = 0; i < degree; ++i) {
			next_derivs[i] = prev_derivs[i];

			for (uint64_t k = i + 1; k < degree; ++k) {
				next_derivs[i] += prev_derivs[k] * incr_prod.get_neg(i + 1, k) * one_over_fact[k - i];
			}
		}

		std::swap(next_coefs, prev_coefs);
		std::swap(next_derivs, prev_derivs);
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

	if (dimension == 0) { throw std::invalid_argument("sig_coef received path of dimension 0"); }

	const uint64_t path_length = dimension * length;
	std::fill(out, out + path_length, static_cast<T>(0.));

	if (length <= 1) {
		return;
	}

	//TODO: check indices < dim

	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);

	// Each buffer is of length (len(multi_indices[i]) + 1)
	// So we need a total size of sum{ len(multi_indices[i]) + 1 } = sum{ len(multi_indices[i]) } + len(multi_indices)
	uint64_t coef_buffer_len = num_multi_idx;
	uint64_t max_degree = 0;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		coef_buffer_len += degrees[i];
		max_degree = std::max(max_degree, degrees[i]);
	}

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

	auto next_derivs_uptr = std::make_unique<T[]>(coef_buffer_len);
	T* next_derivs = next_derivs_uptr.get();

	auto incr_uptr = std::make_unique<T[]>(max_degree);
	T* incr = incr_uptr.get();

	const uint64_t* multi_idx_ptr = multi_idx;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		uint64_t degree = degrees[i];

		if (!degree) {
			coefs += 1;
			derivs += 1;
			continue;
		}

		UpperTriangularMatrix<T> incr_prod(degree);
		single_sig_coef_backprop_<T>(path_obj, out, coefs, multi_idx_ptr, degree, prev_coefs, next_coefs, incr, incr_prod, derivs, next_derivs, one_over_fact);
		prev_coefs += degree + 1;
		next_coefs += degree + 1;
		coefs += degree;
		derivs += degree;
		next_derivs += degree + 1;
		multi_idx_ptr += degree;
	}

}

template<std::floating_point T>
void batch_sig_coef_backprop_(
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
	if (dimension == 0) { throw std::invalid_argument("sig_coef received path of dimension 0"); }

	Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	//General case and degree = 1 case
	const uint64_t flat_path_length = dimension * length;
	uint64_t coefs_len = 0;
	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		coefs_len += degrees[i] ? degrees[i] : 1;
	}
	const T* const data_end = path + flat_path_length * batch_size;

	std::function<void(const T*, const T*, T*, T*)> sig_func;

	sig_func = [&](const T* path_ptr, const T* coefs_ptr, T* derivs_ptr, T* out_ptr) {
		sig_coef_backprop_<T>(path_ptr, out_ptr, coefs_ptr, derivs_ptr, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time);
		};

	const T* path_ptr;
	T* out_ptr;
	const T* coefs_ptr;
	T* derivs_ptr;

	if (n_jobs != 1) {
		multi_threaded_batch_3(sig_func, path, coefs, derivs, out, batch_size, flat_path_length, flat_path_length, coefs_len, coefs_len, n_jobs);
	}
	else {
		for (path_ptr = path, out_ptr = out, coefs_ptr = coefs, derivs_ptr = derivs;
			path_ptr < data_end;
			path_ptr += flat_path_length, out_ptr += flat_path_length, coefs_ptr += coefs_len, derivs_ptr += coefs_len) {

			sig_func(path_ptr, coefs_ptr, derivs_ptr, out_ptr);
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

	CPSIG_API int sig_coef_backprop_f(const float* path, float* out, float* coefs, float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time) noexcept {
		SAFE_CALL(sig_coef_backprop_<float>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time));
	}

	CPSIG_API int sig_coef_backprop_d(const double* path, double* out, double* coefs, double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(sig_coef_backprop_<double>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time));
	}

	CPSIG_API int batch_sig_coef_backprop_f(const float* path, float* out, float* coefs, float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_coef_backprop_<float>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

	CPSIG_API int batch_sig_coef_backprop_d(const double* path, double* out, double* coefs, double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_coef_backprop_<double>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
	}

}
