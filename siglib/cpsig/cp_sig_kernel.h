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

#include "cp_path.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

/* Signature kernel PDE solver (CPU).
 *
 * Forward recurrence (Day/Wazwaz second-order scheme):
 *   K[i+1, j+1] = (K[i, j+1] + K[i+1, j]) * A(d) - K[i, j] * B(d)
 * where d = <dx_i, dy_j> * dyadic_frac,  A(d) = 1 + d/2 + d^2/12,  B(d) = 1 - d^2/12.
 *
 * Backward recurrence (arXiv:2509.10613):
 *   d[row, col] += d[row, col+1] * A(D_{row-1, col})
 *               +  d[row+1, col] * A(D_{row, col-1})
 *               -  d[row+1, col+1] * B(D_{row, col})
 */

template<std::floating_point T>
FORCE_INLINE void get_a_b(T& a, T& b, const T* gram, uint64_t idx, T dyadic_frac) {
	static const T twelth = static_cast<T>(1.) / 12;
	const T gram_val = gram[idx] * dyadic_frac;
	const T gram_val_2 = gram_val * gram_val * twelth;
	a = static_cast<T>(1.) + static_cast<T>(0.5) * gram_val + gram_val_2;
	b = static_cast<T>(1.) - gram_val_2;
}

template<std::floating_point T>
FORCE_INLINE void get_a(T& a, const T* gram, uint64_t idx, T dyadic_frac) {
	static const T twelth = static_cast<T>(1.) / 12;
	T gram_val = gram[idx] * dyadic_frac;
	a = static_cast<T>(1.) + gram_val * (static_cast < T>(0.5) + gram_val * twelth);
}

template<std::floating_point T>
FORCE_INLINE void get_b(T& b, const T* gram, uint64_t idx, T dyadic_frac) {
	static const T twelth = static_cast<T>(1.) / 12;
	const T gram_val = gram[idx] * dyadic_frac;
	b = static_cast<T>(1.) - gram_val * gram_val * twelth;
}

template<std::floating_point T>
FORCE_INLINE void get_a_b_deriv(T& a_deriv, T& b_deriv, const T* gram, uint64_t idx, T dyadic_frac) {
	static const T twelth = static_cast<T>(1.) / 12;
	static const T sixth = static_cast<T>(1.) / 6;
	const T gram_val = gram[idx] * dyadic_frac;
	b_deriv = -gram_val * sixth * dyadic_frac;
	a_deriv = static_cast<T>(0.5) * dyadic_frac - b_deriv;
}

template<std::floating_point T>
void get_sig_kernel_grid_(
	const T* gram,
	uint64_t length1,
	uint64_t length2,
	T* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const T twelth = static_cast<T>(1.) / 12;

	const uint64_t grid_size_1 = 1ULL << dyadic_order_1;
	const uint64_t grid_size_2 = 1ULL << dyadic_order_2;
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	T* pde_grid = out;

	for (uint64_t i = 0; i < dyadic_length_1; ++i)
		pde_grid[i * dyadic_length_2] = static_cast<T>(1.);
	std::fill(pde_grid, pde_grid + dyadic_length_2, static_cast<T>(1.));
	auto a_terms_uptr = std::make_unique<T[]>(length2 - 1);
	auto b_terms_uptr = std::make_unique<T[]>(length2 - 1);
	T* const a_terms = a_terms_uptr.get();
	T* const b_terms = b_terms_uptr.get();

	T* k11 = pde_grid;
	T* k12 = k11 + 1;
	T* k21 = k11 + dyadic_length_2;
	T* k22 = k21 + 1;

	const T* gram_ptr = gram;

	for (uint64_t ii = 0; ii < length1 - 1; ++ii, gram_ptr += length2 - 1) {
		for (uint64_t m = 0; m < length2 - 1; ++m) {
			const T deriv = gram_ptr[m] * dyadic_frac;
			const T deriv2 = deriv * deriv * twelth;
			a_terms[m] = static_cast<T>(1.) + static_cast<T>(0.5) * deriv + deriv2;
			b_terms[m] = static_cast<T>(1.) - deriv2;
		}

		for (uint64_t i = 0; i < grid_size_1; ++i, ++k11, ++k12, ++k21, ++k22) {
			for (uint64_t jj = 0; jj < length2 - 1; ++jj) {
				const T a = a_terms[jj];
				const T b = b_terms[jj];
				for (uint64_t j = 0; j < grid_size_2; ++j) {
					*(k22++) = (*(k21++) + *(k12++)) * a - *(k11++) * b;
				}
			}
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void pde_stencil(T* next, const T* prev, const T* prev_prev, uint64_t j, T deriv) {
	static const T twelth = static_cast<T>(1.) / 12;
	const T deriv2 = deriv * deriv * twelth;
	*(next + j) = (*(prev + j) + *(prev + j - 1)) * (
		static_cast<T>(1.) + static_cast<T>(0.5) * deriv + deriv2)
		- *(prev_prev + j - 1) * (static_cast<T>(1.) - deriv2);
}

// order=true: j along dim 2 (shorter), i along dim 1.
// order=false: swapped. Resolved at compile time.
template<bool order>
FORCE_INLINE uint64_t diag_gram_idx(uint64_t i, uint64_t j, uint64_t do1, uint64_t do2, uint64_t gram_stride) {
	if constexpr (order) {
		return ((i - 1) >> do1) * gram_stride + ((j - 1) >> do2);
	} else {
		return ((j - 1) >> do1) * gram_stride + ((i - 1) >> do2);
	}
}

// Anti-diagonal sweep for scalar output. Stores 3 diagonal buffers
// of length min(L1, L2) instead of the full grid.
template<std::floating_point T, bool order>
void get_sig_kernel_diag_internal_(
	const T* gram,
	uint64_t length2,
	T* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	uint64_t dyadic_length_1,
	uint64_t dyadic_length_2
) {
	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;
	const uint64_t long_len = order ? dyadic_length_1 : dyadic_length_2;
	const uint64_t short_len = order ? dyadic_length_2 : dyadic_length_1;

	const uint64_t diag_len = short_len;
	auto diagonals_uptr = std::make_unique<T[]>(diag_len * 3);
	T* const diagonals = diagonals_uptr.get();

	T* prev_prev_diag = diagonals;
	T* prev_diag = diagonals + diag_len;
	T* next_diag = diagonals + 2 * diag_len;

	std::fill(diagonals, diagonals + 3 * diag_len, static_cast<T>(1.));

	for (uint64_t p = 2; p < num_anti_diag; ++p) {
		const uint64_t startj = long_len > p ? 1 : p - long_len + 1;
		const uint64_t endj = short_len > p ? p : short_len;

		for (uint64_t j = startj; j < endj; ++j) {
			const uint64_t i = p - j;
			const T deriv = gram[diag_gram_idx<order>(i, j, dyadic_order_1, dyadic_order_2, length2 - 1)] * dyadic_frac;
			pde_stencil(next_diag, prev_diag, prev_prev_diag, j, deriv);
		}

		T* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;
	}

	*out = prev_diag[diag_len - 1];
}

template<std::floating_point T>
void get_sig_kernel_diag_(
	const T* gram,
	uint64_t length1,
	uint64_t length2,
	T* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	if (dyadic_length_2 <= dyadic_length_1)
		get_sig_kernel_diag_internal_<T, true>(gram, length2, out, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
	else
		get_sig_kernel_diag_internal_<T, false>(gram, length2, out, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
}

template<std::floating_point T>
void sig_kernel_(
	const T* gram,
	T* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	if (return_grid)
		get_sig_kernel_grid_(gram, length1, length2, out, dyadic_order_1, dyadic_order_2);
	else
		get_sig_kernel_diag_(gram, length1, length2, out, dyadic_order_1, dyadic_order_2);
}

template<std::floating_point T>
void batch_sig_kernel_(
	const T* gram,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	int n_jobs,
	bool return_grid
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	if (!gram) {
		std::fill(out, out + batch_size, static_cast<T>(1.));
		return;
	}

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const T* const data_end_1 = gram + gram_length * batch_size;
	const uint64_t result_length = return_grid ? (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1) : 1;

	auto sig_kernel_func = [&](const T* const gram_ptr, T* const out_ptr) {
		if (return_grid) {
			get_sig_kernel_grid_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2);
		}
		else {
			get_sig_kernel_diag_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2);
		}
	};

	if (n_jobs != 1) {
		multi_threaded_batch(sig_kernel_func, gram, out, batch_size, gram_length, result_length, n_jobs);
	}
	else {
		const T* gram_ptr = gram;
		T* out_ptr = out;
		for (;
			gram_ptr < data_end_1;
			gram_ptr += gram_length, out_ptr += result_length) {

			sig_kernel_func(gram_ptr, out_ptr);
		}
	}
	return;
}

template<std::floating_point T>
void get_sig_kernel_backprop_(
	const T* gram,
	T* out,
	const T* derivs,
	const T* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid
) {
	const uint64_t dl1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dl2 = ((length2 - 1) << dyadic_order_2) + 1;
	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const uint64_t grid_length = dl1 * dl2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const uint64_t gram_stride = length2 - 1;

	auto at = [dl2](int64_t row, int64_t col) -> uint64_t { return row * dl2 + col; };
	auto gram_at = [&](int64_t row, int64_t col) -> uint64_t {
		return (row >> dyadic_order_1) * gram_stride + (col >> dyadic_order_2);
	};

	auto d_grid_uptr = std::make_unique<T[]>(grid_length);
	T* const d_grid = d_grid_uptr.get();

	if (return_grid) {
		std::copy(derivs, derivs + grid_length, d_grid);
	}
	else {
		std::fill(d_grid, d_grid + grid_length, static_cast<T>(0.));
		d_grid[grid_length - 1] = *derivs;
	}

	std::fill(out, out + gram_length, static_cast<T>(0.));

	T a, da, db;

	auto accum_gram_grad = [&](int64_t row, int64_t col) {
		const uint64_t gi = gram_at(row - 1, col - 1);
		get_a_b_deriv(da, db, gram, gi, dyadic_frac);
		out[gi] += d_grid[at(row, col)] * (
			(k_grid[at(row - 1, col)] + k_grid[at(row, col - 1)]) * da
			- k_grid[at(row - 1, col - 1)] * db
		);
	};

	// 1. Last element (seed)
	accum_gram_grad(dl1 - 1, dl2 - 1);

	// 2. Last row: horizontal propagation only
	for (int64_t col = dl2 - 2; col >= 1; --col) {
		get_a(a, gram, gram_at(dl1 - 2, col), dyadic_frac);
		d_grid[at(dl1 - 1, col)] += d_grid[at(dl1 - 1, col + 1)] * a;
		accum_gram_grad(dl1 - 1, col);
	}

	// 3. Last column: vertical propagation only
	for (int64_t row = dl1 - 2; row >= 1; --row) {
		get_a(a, gram, gram_at(row, dl2 - 2), dyadic_frac);
		d_grid[at(row, dl2 - 1)] += d_grid[at(row + 1, dl2 - 1)] * a;
		accum_gram_grad(row, dl2 - 1);
	}

	// 4. Interior
	for (int64_t row = dl1 - 2; row >= 1; --row) {
		for (int64_t col = dl2 - 2; col >= 1; --col) {
			T a_right, a_below, b_diag;
			get_a(a_right, gram, gram_at(row - 1, col), dyadic_frac);
			get_a(a_below, gram, gram_at(row, col - 1), dyadic_frac);
			get_b(b_diag, gram, gram_at(row, col), dyadic_frac);

			d_grid[at(row, col)] += d_grid[at(row, col + 1)] * a_right
				+ d_grid[at(row + 1, col)] * a_below
				- d_grid[at(row + 1, col + 1)] * b_diag;

			accum_gram_grad(row, col);
		}
	}
}

template<std::floating_point T>
void sig_kernel_backprop_(
	const T* gram,
	T* out,
	const T* derivs,
	const T* k_grid,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	get_sig_kernel_backprop_<T>(gram, out, derivs, k_grid, length1, length2, dyadic_order_1, dyadic_order_2, return_grid);
}

template<std::floating_point T>
void batch_sig_kernel_backprop_(
	const T* gram,
	T* out,
	const T* derivs,
	const T* k_grid,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid,
	int n_jobs
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);

	if (!gram) {
		std::fill(out, out + batch_size * gram_length, static_cast<T>(0.));
		return;
	}

	const T* const data_end_1 = gram + gram_length * batch_size;

	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;
	const uint64_t grid_length = dyadic_length_1 * dyadic_length_2;

	const uint64_t derivs_stride = return_grid ? grid_length : 1;

	auto sig_kernel_backprop_func = [&](const T* gram_ptr, const T* deriv_ptr, const T* k_grid_ptr, T* out_ptr) {
		sig_kernel_backprop_(gram_ptr, out_ptr, deriv_ptr, k_grid_ptr, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid);
	};

	if (n_jobs != 1) {
		multi_threaded_batch_3(sig_kernel_backprop_func, gram, derivs, k_grid, out, batch_size, gram_length, derivs_stride, grid_length, gram_length, n_jobs);
	}
	else {
		const T* gram_ptr = gram;
		T* out_ptr = out;
		const T* deriv_ptr = derivs;
		const T* k_grid_ptr = k_grid;
		for (;
			gram_ptr < data_end_1;
			gram_ptr += gram_length, out_ptr += gram_length, deriv_ptr += derivs_stride, k_grid_ptr += grid_length) {

			sig_kernel_backprop_func(gram_ptr, deriv_ptr, k_grid_ptr, out_ptr);
		}
	}
	return;
}