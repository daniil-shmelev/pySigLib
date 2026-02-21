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
void get_sig_kernel_(
	const T* gram,
	uint64_t length1,
	uint64_t length2,
	T* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid
) {
	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const T twelth = static_cast<T>(1.) / 12;

	// Dyadically refined grid dimensions
	const uint64_t grid_size_1 = 1ULL << dyadic_order_1;
	const uint64_t grid_size_2 = 1ULL << dyadic_order_2;
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	// Allocate(flattened) PDE grid
	T* pde_grid;
	if (return_grid)
		pde_grid = out;
	else {
		auto pde_grid_uptr = std::make_unique<T[]>(dyadic_length_1 * dyadic_length_2);
		pde_grid = pde_grid_uptr.get();
	}

	// Initialization of K array
	for (uint64_t i = 0; i < dyadic_length_1; ++i) {
		pde_grid[i * dyadic_length_2] = static_cast<T>(1.0); // Set K[i, 0] = 1.0
	}

	std::fill(pde_grid, pde_grid + dyadic_length_2, static_cast<T>(1.0)); // Set K[0, j] = 1.0

	auto deriv_term_1_uptr = std::make_unique<T[]>(length2 - 1);
	T* const deriv_term_1 = deriv_term_1_uptr.get();

	auto deriv_term_2_uptr = std::make_unique<T[]>(length2 - 1);
	T* const deriv_term_2 = deriv_term_2_uptr.get();

	T* k11 = pde_grid;
	T* k12 = k11 + 1;
	T* k21 = k11 + dyadic_length_2;
	T* k22 = k21 + 1;

	const T* gram_ptr = gram;

	for (uint64_t ii = 0; ii < length1 - 1; ++ii, gram_ptr += length2 - 1) {
		for (uint64_t m = 0; m < length2 - 1; ++m) {
			const T deriv = gram_ptr[m] * dyadic_frac;//dot_product(diff1Ptr, diff2Ptr, dimension);
			const T deriv2 = deriv * deriv * twelth;
			deriv_term_1[m] = static_cast<T>(1.0) + static_cast<T>(0.5) * deriv + deriv2;
			deriv_term_2[m] = static_cast<T>(1.0) - deriv2;
		}

		for (uint64_t i = 0;
			i < grid_size_1;
			++i, ++k11, ++k12, ++k21, ++k22) {

			for (uint64_t jj = 0; jj < length2 - 1; ++jj) {
				const T t1 = deriv_term_1[jj];
				const T t2 = deriv_term_2[jj];
				for (uint64_t j = 0; j < grid_size_2; ++j) {
					*(k22++) = (*(k21++) + *(k12++)) * t1 - *(k11++) * t2;
				}
			}
		}
	}

	if (!return_grid)
		*out = pde_grid[dyadic_length_1 * dyadic_length_2 - 1];
}

template<std::floating_point T, bool order> //order is True if dyadic_length_2 <= dyadic_length_1
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
	const T twelth = static_cast<T>(1.) / 12;
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;

	// Allocate three diagonals
	const uint64_t diag_len = std::min(dyadic_length_1, dyadic_length_2);
	auto diagonals_uptr = std::make_unique<T[]>(diag_len * 3);
	T* const diagonals = diagonals_uptr.get();

	T* prev_prev_diag = diagonals;
	T* prev_diag = diagonals + diag_len;
	T* next_diag = diagonals + 2 * diag_len;

	// Initialization
	std::fill(diagonals, diagonals + 3 * diag_len, static_cast<T>(1.));

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		if (order) {
			uint64_t startj, endj;
			if (dyadic_length_1 > p) startj = 1;
			else startj = p - dyadic_length_1 + 1;
			if (dyadic_length_2 > p) endj = p;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t ii = ((i - 1) >> dyadic_order_1);
				const uint64_t jj = ((j - 1) >> dyadic_order_2);

				const T deriv = gram[ii * (length2 - 1) + jj] * dyadic_frac;
				const T deriv2 = deriv * deriv * twelth;

				*(next_diag + j) = (*(prev_diag + j) + *(prev_diag + j - 1)) * (
					static_cast<T>(1.) + static_cast<T>(0.5) * deriv + deriv2) - *(prev_prev_diag + j - 1) * (static_cast<T>(1.) - deriv2);

			}
		}
		else {
			uint64_t startj, endj;
			if (dyadic_length_2 > p) startj = 1;
			else startj = p - dyadic_length_2 + 1;
			if (dyadic_length_1 > p) endj = p;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t ii = ((i - 1) >> dyadic_order_2);
				const uint64_t jj = ((j - 1) >> dyadic_order_1);

				const T deriv = gram[jj * (length2 - 1) + ii] * dyadic_frac;
				const T deriv2 = deriv * deriv * twelth;

				*(next_diag + j) = (*(prev_diag + j) + *(prev_diag + j - 1)) * (
					static_cast<T>(1.) + static_cast<T>(0.5) * deriv + deriv2) - *(prev_prev_diag + j - 1) * (static_cast<T>(1.) - deriv2);

			}
		}

		// Rotate the diagonals (swap pointers, no data copying)
		T* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;
	}

	*out = prev_diag[diag_len - 1];
}

template<std::floating_point T, bool order>//order is True if dyadic_length_2 <= dyadic_length_1
void get_sig_kernel_backprop_diag_internal_(
	const T* gram,
	T* out,
	T deriv,
	const T* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	uint64_t dyadic_length_1,
	uint64_t dyadic_length_2
) {
	// General structure of the grids:
	// 
	// dF / dk = 0 for the first row and column of k_grid, so disregard these.
	// Flip the remaining grid, so that the last element is now in the top left.
	// Now, add a row and column of zeros as initial conditions to the grid, such that it now
	// has the same dimensions as k_grid.
	// The resulting grid is what is traversed by 'diagonals' below.
	// 
	// The grids for A, B, dA and dB are flipped and padded similarly, such that
	// the value at index [1,1] is the value at [-1,-1] in the original grids.
	// We will only need one diagonal for A and one for B, containing the values
	// needed to update the leading diagonal of dF / dk. For dA and dB, we don't
	// need to use diagonals, we can just get the values once when updating dF / dk.
	// Note that for A, these values are lagged, i.e. we need values A(i-1,j) and 
	// A(i,j-1) to update dF / dk(i,j).

	// As with the diagonal method for sig_kernel, it matters which of
	// dyadic_length_1 and dyadic_length_2 is longer.
	const uint64_t ord_dyadic_order_1 = order ? dyadic_order_1 : dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? dyadic_order_2 : dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? dyadic_length_1 : dyadic_length_2;
	const uint64_t ord_dyadic_length_2 = order ? dyadic_length_2 : dyadic_length_1;

	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;
	const uint64_t grid_length = dyadic_length_1 * dyadic_length_2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);

	// Allocate three diagonals
	const uint64_t diag_len = std::min(dyadic_length_1, dyadic_length_2);
	auto diagonals_uptr = std::make_unique<T[]>(diag_len * 3);
	T* const diagonals = diagonals_uptr.get();

	// Allocate diagonals to store A, B, A_deriv, B_deriv
	auto a_uptr = std::make_unique<T[]>(diag_len);
	T* const a = a_uptr.get();

	auto b_uptr = std::make_unique<T[]>(diag_len);
	T* const b = b_uptr.get();

	// Ptrs for diagonals
	T* prev_prev_diag = diagonals;
	T* prev_diag = prev_prev_diag + diag_len;
	T* next_diag = prev_diag + diag_len;

	// k_grid ptrs
	const T* k11, * k12, * k21;

	// Initialization
	std::fill(out, out + (length1 - 1) * (length2 - 1), static_cast<T>(0.));
	std::fill(diagonals, diagonals + 3 * diag_len, static_cast<T>(0.));
	std::fill(a, a + diag_len, static_cast<T>(0.));
	std::fill(b, b + diag_len, static_cast<T>(0.));
	
	*(prev_diag + 1) = deriv;
	T da, db;
	get_a_b_deriv(da, db, gram, gram_length - 1, dyadic_frac);

	//Update dF / dx for first value
	k21 = k_grid + grid_length - 2;
	k12 = k_grid + grid_length - dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
	k11 = k12 - 1;
	out[gram_length - 1] += deriv * ( ((*k21) + (*k12)) * da - *(k11) * db );

	for (uint64_t p = 3; p < num_anti_diag; ++p) { // First three antidiagonals are initialised

		//Update b
		uint64_t startj, endj;
		uint64_t p_ = p - 2;
		startj = ord_dyadic_length_1 > p_ ? 1 : p_ - ord_dyadic_length_1 + 1;
		endj = ord_dyadic_length_2 > p_ ? p_ : ord_dyadic_length_2;

		uint64_t i = p_ - startj; // Calculate corresponding i (since i + j = p)
		uint64_t i_rev = ord_dyadic_length_1 - i - 1;
		uint64_t j_rev = ord_dyadic_length_2 - startj - 1;

		for (uint64_t j = startj;
			j < endj;
			++j, --i, ++i_rev, --j_rev) {
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;

			get_b(b[j], gram, gram_idx, dyadic_frac);
		}

		//Update a
		p_ = p - 1;
		startj = ord_dyadic_length_1 > p_ ? 1 : p_ - ord_dyadic_length_1 + 1;
		endj = ord_dyadic_length_2 > p_ ? p_ : ord_dyadic_length_2;

		i = p_ - startj; // Calculate corresponding i (since i + j = p)
		i_rev = ord_dyadic_length_1 - i - 1;
		j_rev = ord_dyadic_length_2 - startj - 1;

		for (uint64_t j = startj;
			j < endj;
			++j, --i, ++i_rev, --j_rev) {
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;

			get_a(a[j], gram, gram_idx, dyadic_frac);
		}

		//Update diagonals
		startj = ord_dyadic_length_1 > p ? 1 : p - ord_dyadic_length_1 + 1;
		endj = ord_dyadic_length_2 > p ? p : ord_dyadic_length_2;

		i = p - startj; // Calculate corresponding i (since i + j = p)
		i_rev = ord_dyadic_length_1 - i - 1;
		j_rev = ord_dyadic_length_2 - startj - 1;
		uint64_t idx = order ? (i_rev + 1) * dyadic_length_2 + (j_rev + 1) : (j_rev + 1) * dyadic_length_2 + (i_rev + 1); //NOT ord_dyadic_length_2 here, as we are indexing k_grid
		k12 = k_grid + idx - 1;
		k21 = k_grid + idx - dyadic_length_2; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
		k11 = k21 - 1;

		for (uint64_t j = startj;
			j < endj;
			++j, --i, ++i_rev, --j_rev) {
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);

			//Get da, db
			const uint64_t gram_idx = order ? ii * (length2 - 1) + jj : jj * (length2 - 1) + ii;
			get_a_b_deriv(da, db, gram, gram_idx, dyadic_frac);

			// Update dF / dk
			*(next_diag + j) = *(prev_diag + j - 1) * a[j-1] + *(prev_diag + j) * a[j] - *(prev_prev_diag + j - 1) * b[j-1];

			// Update dF / dx
			out[gram_idx] += *(next_diag + j) * ( (*(k12) + *(k21)) * da - *(k11) * db );

			if (order) {
				k12 += dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
				k21 += dyadic_length_2 - 1;
				k11 += dyadic_length_2 - 1;
			}
			else {
				k12 -= dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
				k21 -= dyadic_length_2 - 1;
				k11 -= dyadic_length_2 - 1;
			}
		}

		// Rotate the diagonals (swap pointers, no data copying)
		T* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;
	}
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
	// Dyadically refined grid dimensions
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
		get_sig_kernel_(gram, length1, length2, out, dyadic_order_1, dyadic_order_2, true);
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
			get_sig_kernel_(gram_ptr, length1, length2, out_ptr, dyadic_order_1, dyadic_order_2, true);
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
	T deriv,
	const T* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1 + dyadic_order_2));
	static const T sixth = static_cast<T>(1.) / 6;
	static const T twelth = static_cast<T>(1.) / 12;
	const uint64_t grid_length = dyadic_length_1 * dyadic_length_2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);

	// Allocate grid for dF / dk
	auto d_grid_uptr = std::make_unique<T[]>(grid_length);
	T* const d_grid = d_grid_uptr.get();

	std::fill(out, out + (length1 - 1) * (length2 - 1), static_cast<T>(0.));

	// a, b, da, db
	T a, a_deriv, b_deriv;
	T a10, a01, b11;

	// indices
	uint64_t grid_idx, gram_idx;

	// k_grid and d_grid ptrs
	const T* k11, * k12, * k21;
	const T* d11, * d12, * d21;

	//Start with the last dF / dk, which is known ============================================
	d_grid[grid_length - 1] = deriv;

	//Compute dA(i-1, j-1) and dB(i-1, j-1)
	gram_idx = gram_length - 1;
	get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

	//Update dF / dx
	k21 = k_grid + grid_length - 2;
	k12 = k_grid + grid_length - dyadic_length_2 - 1;
	k11 = k12 - 1;
	out[gram_length - 1] += d_grid[grid_length - 1] * ((*k12 + *k21) * a_deriv - *k11 * b_deriv);

	//Loop over last row ============================================
	grid_idx = grid_length - 2;
	k21 = k_grid + grid_idx - 1;
	k12 = k_grid + grid_idx - dyadic_length_2;
	k11 = k12 - 1;

	for (int64_t i = dyadic_length_2 - 2;
		i >= 1;
		--i, --grid_idx, --k12, --k21, --k11) {

		const int64_t j = dyadic_length_1 - 1;

		//Precompute indices
		const uint64_t cur_ii = (i >> dyadic_order_2);
		const uint64_t prev_ii = ((i - 1) >> dyadic_order_2);
		const uint64_t prev_jj = ((j - 1) >> dyadic_order_1) * (length2 - 1);

		//Compute A(i, j-1)
		get_a(a, gram, prev_jj + cur_ii, dyadic_frac);

		//Update dF / dk
		d_grid[grid_idx] = d_grid[grid_idx + 1] * a;

		//Compute dA(i-1, j-1) and dB(i-1, j-1)
		gram_idx = prev_jj + prev_ii;
		get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

		//Update dF / dx
		out[gram_idx] += d_grid[grid_idx] * ((*k12 + *k21) * a_deriv - *k11 * b_deriv);
	}

	grid_idx = grid_length - 1 - dyadic_length_2;
	k21 = k_grid + grid_idx - 1;
	k12 = k_grid + grid_idx - dyadic_length_2;
	k11 = k12 - 1;
	//Loop over last column ============================================
	for (int64_t j = dyadic_length_1 - 2;
		j >= 1;
		--j,
		grid_idx -= dyadic_length_2,
		k21 -= dyadic_length_2,
		k12 -= dyadic_length_2,
		k11 -= dyadic_length_2) {

		const int64_t i = dyadic_length_2 - 1;

		//Precompute indices
		const uint64_t prev_ii = ((i - 1) >> dyadic_order_2);
		const uint64_t cur_jj = (j >> dyadic_order_1) * (length2 - 1);
		const uint64_t prev_jj = ((j - 1) >> dyadic_order_1) * (length2 - 1);

		//Compute A(i-1, j)
		get_a(a, gram, cur_jj + prev_ii, dyadic_frac);

		//Update dF / dk
		d_grid[grid_idx] = d_grid[grid_idx + dyadic_length_2] * a;

		//Compute dA(i-1, j-1) and dB(i-1, j-1)
		gram_idx = prev_jj + prev_ii;
		get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

		//Update dF / dx
		out[gram_idx] += d_grid[grid_idx] * ((*k12 + *k21) * a_deriv - *k11 * b_deriv);
	}

	// Loop over remaining grid ============================================
	grid_idx = grid_length - 2 - dyadic_length_2;
	k21 = k_grid + grid_idx - 1;
	k12 = k_grid + grid_idx - dyadic_length_2;
	k11 = k12 - 1;
	d21 = d_grid + grid_idx + 1;
	d12 = d_grid + grid_idx + dyadic_length_2;
	d11 = d12 + 1;
	for (int64_t j = dyadic_length_1 - 2;
		j >= 1;
		--j,
		grid_idx -= 2,
		k12 -= 2,
		k21 -= 2,
		k11 -= 2,
		d12 -= 2,
		d21 -= 2,
		d11 -= 2) {

		for (int64_t i = dyadic_length_2 - 2;
			i >= 1;
			--i,
			--grid_idx,
			--k12,
			--k21,
			--k11,
			--d12,
			--d21,
			--d11) {

			//Precompute indices
			const uint64_t cur_ii = (i >> dyadic_order_2);
			const uint64_t prev_ii = ((i - 1) >> dyadic_order_2);
			const uint64_t cur_jj = (j >> dyadic_order_1) * (length2 - 1);
			const uint64_t prev_jj = ((j - 1) >> dyadic_order_1) * (length2 - 1);

			// Compute A(i, j-1)
			get_a(a10, gram, prev_jj + cur_ii, dyadic_frac);

			// Compute A(i-1, j)
			get_a(a01, gram, cur_jj + prev_ii, dyadic_frac);

			// Compute B(i, j)
			get_b(b11, gram, cur_jj + cur_ii, dyadic_frac);

			//Update dF / dk
			d_grid[grid_idx] = (*d21) * a10 + (*d12) * a01 - (*d11) * b11;

			//Compute dA(i-1, j-1) and dB(i-1, j-1)
			gram_idx = prev_jj + prev_ii;
			get_a_b_deriv(a_deriv, b_deriv, gram, gram_idx, dyadic_frac);

			//Update dF / dx
			out[gram_idx] += d_grid[grid_idx] * ((*k12 + *k21) * a_deriv - *k11 * b_deriv);
		}
	}

	return;
}

template<std::floating_point T>
void sig_kernel_backprop_(
	const T* gram,
	T* out,
	T deriv,
	const T* k_grid,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	get_sig_kernel_backprop_<T>(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2);
	//get_sig_kernel_backprop_diag_(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2);
}

template<std::floating_point T>
void get_sig_kernel_backprop_diag_(
	const T* gram,
	T* out,
	T deriv,
	const T* k_grid,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	// Dyadically refined grid dimensions
	const uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;

	if (dyadic_length_2 <= dyadic_length_1)
		get_sig_kernel_backprop_diag_internal_<true>(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
	else
		get_sig_kernel_backprop_diag_internal_<false>(gram, out, deriv, k_grid, length1, length2, dyadic_order_1, dyadic_order_2, dyadic_length_1, dyadic_length_2);
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

	auto sig_kernel_backprop_func = [&](const T* gram_ptr, const T* deriv_ptr, const T* k_grid_ptr, T* out_ptr) {
		sig_kernel_backprop_(gram_ptr, out_ptr, *deriv_ptr, k_grid_ptr, dimension, length1, length2, dyadic_order_1, dyadic_order_2);
	};

	if (n_jobs != 1) {
		multi_threaded_batch_3(sig_kernel_backprop_func, gram, derivs, k_grid, out, batch_size, gram_length, 1, grid_length, gram_length, n_jobs);
	}
	else {
		const T* gram_ptr = gram;
		T* out_ptr = out;
		const T* deriv_ptr = derivs;
		const T* k_grid_ptr = k_grid;
		for (;
			gram_ptr < data_end_1;
			gram_ptr += gram_length, out_ptr += gram_length, deriv_ptr += 1, k_grid_ptr += grid_length) {

			sig_kernel_backprop_func(gram_ptr, deriv_ptr, k_grid_ptr, out_ptr);
		}
	}
	return;
}