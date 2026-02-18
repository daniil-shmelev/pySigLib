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

template<std::floating_point T, uint64_t degree>
void single_sig_coef_template_(
	const Path<T>& path,
	T* out,
	const uint64_t* multi_idx,
	T* prev_coefs,
	T* next_coefs,
	T* incr_prod,
	const T* one_over_fact,
	bool prefixes
) {
	single_sig_coef_(path, out, multi_idx, degree, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
}

template<std::floating_point T>
void call_single_sig_coef_(
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
	switch (degree) {
	case 1:  return single_sig_coef_template_<T, 1>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 2:  return single_sig_coef_template_<T, 2>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 3:  return single_sig_coef_template_<T, 3>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 4:  return single_sig_coef_template_<T, 4>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 5:  return single_sig_coef_template_<T, 5>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 6:  return single_sig_coef_template_<T, 6>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 7:  return single_sig_coef_template_<T, 7>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 8:  return single_sig_coef_template_<T, 8>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 9:  return single_sig_coef_template_<T, 9>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 10: return single_sig_coef_template_<T, 10>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 11: return single_sig_coef_template_<T, 11>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 12: return single_sig_coef_template_<T, 12>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 13: return single_sig_coef_template_<T, 13>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 14: return single_sig_coef_template_<T, 14>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 15: return single_sig_coef_template_<T, 15>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 16: return single_sig_coef_template_<T, 16>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 17: return single_sig_coef_template_<T, 17>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 18: return single_sig_coef_template_<T, 18>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 19: return single_sig_coef_template_<T, 19>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	case 20: return single_sig_coef_template_<T, 20>(path, out, multi_idx, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
	default:
		return single_sig_coef_<T>(path, out, multi_idx, degree, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
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

	auto incr_prod_uptr = std::make_unique<T[]>(max_degree + 1);
	T* incr_prod = incr_prod_uptr.get();

	auto one_over_fact_uptr = std::make_unique<T[]>(max_degree + 1);
	T* one_over_fact = one_over_fact_uptr.get();

	one_over_fact[0] = static_cast<T>(1.);
	for (uint64_t i = 1; i < max_degree + 1; ++i) {
		one_over_fact[i] = one_over_fact[i - 1] / i;
	}

	auto prev_coefs_uptr = std::make_unique<T[]>(max_degree + 1);
	T* prev_coefs = prev_coefs_uptr.get();

	auto next_coefs_uptr = std::make_unique<T[]>(max_degree + 1);
	T* next_coefs = next_coefs_uptr.get();

	const uint64_t* multi_idx_ptr = multi_idx;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		uint64_t degree = degrees[i];

		if (!degree) {
			*out_ptr = static_cast<T>(1.);
			++out_ptr;
			continue;
		}

		call_single_sig_coef_<T>(path_obj, out_ptr, multi_idx_ptr, degree, prev_coefs, next_coefs, incr_prod, one_over_fact, prefixes);
		out_ptr += prefixes ? degree : 1;
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

	auto sig_func = [&](const T* path_ptr, T* out_ptr) {
		sig_coef_<T>(path_ptr, out_ptr, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time, prefixes);
	};

	if (n_jobs != 1) {
		multi_threaded_batch(sig_func, path, out, batch_size, flat_path_length, result_length, n_jobs);
	}
	else {
		const T* path_ptr = path;
		T* out_ptr = out;
		for (; path_ptr < data_end;
			path_ptr += flat_path_length, out_ptr += result_length) {

			sig_func(path_ptr, out_ptr);
		}
	}
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// backpropagation
//////////////////////////////////////////////////////////////////////////////////////////////

template <std::floating_point T>
class UpperTriangularMatrix {
public:
	explicit UpperTriangularMatrix(size_t n)
		: n_(n), data_(n * (n + 1) / 2), row_offsets_(n)
	{
		compute_offsets();
	}

	size_t size() const { return n_; }

	void resize(size_t n) {
		n_ = n;
		const size_t needed = n * (n + 1) / 2;
		if (needed > data_.size()) {
			data_.resize(needed);
		}
		if (n > row_offsets_.size()) {
			row_offsets_.resize(n);
		}
		compute_offsets();
	}

	inline T& operator()(size_t i, size_t j) {
#ifdef _DEBUG
		if (i > j || j >= n_) {
			throw std::out_of_range("Accessing out-of-range element");
		}
#endif
		return data_[row_offsets_[i] + (j - i)];
	}

	inline const T& operator()(size_t i, size_t j) const {
#ifdef _DEBUG
		if (i > j || j >= n_) {
			throw std::out_of_range("Accessing out-of-range element");
		}
#endif
		return data_[row_offsets_[i] + (j - i)];
	}

	inline const T* row_ptr(size_t r) const { return data_.data() + row_offsets_[r]; }

	void populate(const T* incr) {
		// Populate the matrix so that mat(i,j) = prod_{k=i}^j incr[k]

		// Set last diagonal entry: mat(n_ - 1, n_ - 1) = incr[n_ - 1]
		data_[row_offsets_[n_ - 1]] = incr[n_ - 1];

		// Loop backwards over rows using the relation:
		// mat(i-1, j) = mat(i, j) * incr[i-1];
		size_t prev_row_length = 1;
		for (int64_t i = n_ - 2; i >= 0; --i, ++prev_row_length) {

			T xi = incr[i];
			T* curr_row = data_.data() + row_offsets_[i];
			curr_row[0] = xi; // Diagonal entry for current row: mat(i,i) = incr[i]

			const T* prev_row = data_.data() + row_offsets_[i + 1];

			// Multiply previous row by incr[i-1] and assign to current row
			for (size_t k = 0; k < prev_row_length; ++k) {
				curr_row[k + 1] = xi * prev_row[k];
			}
		}
	}

private:
	void compute_offsets() {
		// Row i has (n_ - i) elements
		size_t offset = 0;
		for (size_t i = 0; i < n_; ++i) {
			row_offsets_[i] = offset;
			offset += n_ - i;
		}
	}

	size_t n_;
	std::vector<T> data_;
	std::vector<size_t> row_offsets_;
};

FORCE_INLINE bool sig_coef_backprop_skip(uint64_t data_dimension, uint64_t pre_time_aug_dim, uint64_t idx, bool lead_lag, bool parity) {
	// Determine whether the derivative with respect to incr[i] needs to be computed, or can be skipped

	return (idx >= pre_time_aug_dim) || (lead_lag && (parity == (idx < data_dimension)));
}

template<std::floating_point T>
FORCE_INLINE void single_sig_coef_backprop_(
	const Path<T>& path,
	T* out, // Should be zeroed
	const T* coefs, // This should be an array of size degree, of the prefixes of the coeff.
	const uint64_t* multi_idx,
	uint64_t degree,
	T* next_coefs,
	T* prev_coefs,
	T* incr,
	UpperTriangularMatrix<T>& incr_prod, // Upper-triangular matrix, to be filled with incr_prod[i,j] = prod_{k = i}^j incr[k]
	T* next_derivs, // Derivs wrt coeffs
	T* prev_derivs,
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

	next_coefs[0] = static_cast<T>(1.);
	for (uint64_t i = 0; i < degree; ++i) {
		next_coefs[i + 1] = coefs[i];
	}
	prev_coefs[0] = static_cast<T>(1.);

	T* pos = out + data_dimension * (data_length - 1);
	T* neg = pos - data_dimension;
	bool parity = false;

	for (; next_pt != first_pt; --next_pt, --prev_pt, parity = !parity) {

		/////////////////////////////////////////////////////////////////////////
		// Populate incr
		/////////////////////////////////////////////////////////////////////////
		for (uint64_t i = 0; i < degree; ++i) {
			const uint64_t idx = multi_idx[i];
			incr[i] = next_pt[idx] - prev_pt[idx];
		}

		/////////////////////////////////////////////////////////////////////////
		// Populate incr_prod
		// incr_prod(i,j) = prod_{k=i}^j incr[k]
		/////////////////////////////////////////////////////////////////////////
		incr_prod.populate(incr);

		/////////////////////////////////////////////////////////////////////////
		// Reconstruct coefs by using chens relation:
		// S(x_{1:i-1}) = S(x_{1:i}) * S(-x_{i-1:i})
		// Noting that S(-x_{i-1:i}) is given by signed entries of incr_prod
		/////////////////////////////////////////////////////////////////////////
		for (uint64_t i = 1; i < degree + 1; ++i) {
			T acc = static_cast<T>(0.);
			for (uint64_t k = 1; k <= i; ++k) {
				acc += next_coefs[i - k] * incr_prod(i - k, i - 1) * signed_one_over_fact[k];
			}
			prev_coefs[i] = next_coefs[i] + acc;
		}

		/////////////////////////////////////////////////////////////////////////
		// Update path derivs:
		// Compute dL / d incr[i] = sum_k (dL / d next_coefs[k]) * (d next_coefs[k] / d incr[i])
		// backprop incr derivs -> path derivs
		/////////////////////////////////////////////////////////////////////////

		// Separate out i = 0 from main loop - slightly simpler logic
		T update;
		uint64_t idx = multi_idx[0];
		if (!sig_coef_backprop_skip(data_dimension, pre_time_aug_dim, idx, lead_lag, parity)) {
			update = next_derivs[0];
			const T* incr_prod_row_1 = incr_prod.row_ptr(1);
			for (uint64_t m = 1; m < degree; ++m) {
				update += next_derivs[m] * incr_prod_row_1[m - 1] * one_over_fact[m + 1];
			}
			update *= prev_coefs[0];

			// incr derivs -> path derivs
			if (lead_lag && parity) {
				idx -= data_dimension;
			}
			pos[idx] += update;
			neg[idx] -= update;
		}

		// prev_derivs is currently unused - use as a buffer to store intermediate values
		T* buff = prev_derivs;

		for (uint64_t i = 1; i < degree; ++i) {
			idx = multi_idx[i];
			if (!sig_coef_backprop_skip(data_dimension, pre_time_aug_dim, idx, lead_lag, parity)) { // Skip if idx is time-aug dimension
				T s = prev_coefs[i];
				for (uint64_t k = 0; k < i; ++k) {
					buff[k] = prev_coefs[k] * incr_prod(k, i - 1);
					s += buff[k] * one_over_fact[i - k + 1];
				}
				update = next_derivs[i] * s;

				for (uint64_t m = i + 1; m < degree; ++m) {
					s = prev_coefs[i] * one_over_fact[m - i + 1];
					for (uint64_t k = 0; k < i; ++k) {
						s += buff[k] * one_over_fact[m - k + 1];
					}
					s *= incr_prod(i + 1, m);
					update += next_derivs[m] * s;
				}

				// incr derivs -> path derivs
				if (lead_lag && parity) {
					idx -= data_dimension;
				}
				pos[idx] += update;
				neg[idx] -= update;
			}
		}

		/////////////////////////////////////////////////////////////////////////
		// Update sig coef derivs using:
		// dL / d prev_coefs[i] = sum_k (dL / d next_coefs[k]) * (d next_coefs[k] / d prev_coefs[i])
		/////////////////////////////////////////////////////////////////////////
		for (uint64_t i = 0; i < degree - 1; ++i) {
			T acc = static_cast<T>(0.);
			const T* incr_prod_row_i1 = incr_prod.row_ptr(i+1);
			for (uint64_t k = i + 1; k < degree; ++k) {
				acc += next_derivs[k] * incr_prod_row_i1[k - i - 1] * one_over_fact[k - i];
			}
			prev_derivs[i] = next_derivs[i] + acc;
		}
		prev_derivs[degree - 1] = next_derivs[degree - 1];

		std::swap(prev_coefs, next_coefs);
		std::swap(prev_derivs, next_derivs);

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
	const T* coefs,
	const uint64_t* multi_idx,
	T* next_coefs,
	T* prev_coefs,
	T* incr,
	UpperTriangularMatrix<T>& incr_prod,
	T* next_derivs,
	T* prev_derivs,
	const T* one_over_fact,
	const T* signed_one_over_fact
) {
	single_sig_coef_backprop_(path, out, coefs, multi_idx, degree, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
}

template<std::floating_point T>
void call_single_sig_coef_backprop_(
	const Path<T>& path,
	T* out,
	const T* coefs,
	const uint64_t* multi_idx,
	uint64_t degree,
	T* next_coefs,
	T* prev_coefs,
	T* incr,
	UpperTriangularMatrix<T>& incr_prod,
	T* next_derivs,
	T* prev_derivs,
	const T* one_over_fact,
	const T* signed_one_over_fact
) {
	switch (degree) {
	case 1:  return single_sig_coef_backprop_template_<T, 1>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 2:  return single_sig_coef_backprop_template_<T, 2>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 3:  return single_sig_coef_backprop_template_<T, 3>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 4:  return single_sig_coef_backprop_template_<T, 4>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 5:  return single_sig_coef_backprop_template_<T, 5>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 6:  return single_sig_coef_backprop_template_<T, 6>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 7:  return single_sig_coef_backprop_template_<T, 7>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 8:  return single_sig_coef_backprop_template_<T, 8>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 9:  return single_sig_coef_backprop_template_<T, 9>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 10: return single_sig_coef_backprop_template_<T, 10>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 11: return single_sig_coef_backprop_template_<T, 11>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 12: return single_sig_coef_backprop_template_<T, 12>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 13: return single_sig_coef_backprop_template_<T, 13>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 14: return single_sig_coef_backprop_template_<T, 14>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 15: return single_sig_coef_backprop_template_<T, 15>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 16: return single_sig_coef_backprop_template_<T, 16>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 17: return single_sig_coef_backprop_template_<T, 17>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 18: return single_sig_coef_backprop_template_<T, 18>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 19: return single_sig_coef_backprop_template_<T, 19>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	case 20: return single_sig_coef_backprop_template_<T, 20>(path, out, coefs, multi_idx, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
	default:
		return single_sig_coef_backprop_(path, out, coefs, multi_idx, degree, next_coefs, prev_coefs, incr, incr_prod, next_derivs, prev_derivs, one_over_fact, signed_one_over_fact);
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

	auto prev_coefs_uptr = std::make_unique<T[]>(max_degree + 1);
	T* prev_coefs = prev_coefs_uptr.get();

	auto next_coefs_uptr = std::make_unique<T[]>(max_degree + 1);
	T* next_coefs = next_coefs_uptr.get();

	auto next_derivs_uptr = std::make_unique<T[]>(max_degree);
	T* next_derivs = next_derivs_uptr.get();

	auto incr_uptr = std::make_unique<T[]>(max_degree);
	T* incr = incr_uptr.get();

	UpperTriangularMatrix<T> incr_prod(max_degree);

	const uint64_t* multi_idx_ptr = multi_idx;

	for (uint64_t i = 0; i < num_multi_idx; ++i) {
		uint64_t degree = degrees[i];

		if (!degree) {
			coefs += 1;
			derivs += 1;
			continue;
		}

		incr_prod.resize(degree);
		call_single_sig_coef_backprop_<T>(path_obj, out, coefs, multi_idx_ptr, degree, prev_coefs, next_coefs, incr, incr_prod, derivs, next_derivs, one_over_fact, signed_one_over_fact);
		coefs += degree;
		derivs += degree;
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

	auto sig_func = [&](const T* path_ptr, const T* coefs_ptr, T* derivs_ptr, T* out_ptr) {
		sig_coef_backprop_<T>(path_ptr, out_ptr, coefs_ptr, derivs_ptr, multi_idx, num_multi_idx, degrees, dimension, length, time_aug, lead_lag, end_time);
	};

	if (n_jobs != 1) {
		multi_threaded_batch_3(sig_func, path, coefs, derivs, out, batch_size, flat_path_length, coefs_len, coefs_len, flat_path_length, n_jobs);
	}
	else {
		const T* path_ptr = path;
		T* out_ptr = out;
		const T* coefs_ptr = coefs;
		T* derivs_ptr = derivs;
		for (; path_ptr < data_end;
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
