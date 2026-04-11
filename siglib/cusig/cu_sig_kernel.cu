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

#include "cupch.h"
#include "cusig.h"
#include "cu_sig_kernel.h"

/* Signature kernel PDE solver (CUDA).
 * Same scheme as cp_sig_kernel.h. Each block processes one batch item
 * with 32 threads sweeping anti-diagonals in tiles along the shorter axis.
 * The `order` template (true when L2 <= L1) controls axis orientation. */

struct SigKernelParams {
	uint64_t length2;
	uint64_t dyadic_order_1, dyadic_order_2;
	uint64_t dyadic_length_1, dyadic_length_2;
	uint64_t main_dyadic_length, num_anti_diag;
	uint64_t gram_length, grid_length;
};

static void validate_sig_kernel_dims_(uint64_t length1, uint64_t length2,
	uint64_t dyadic_order_1, uint64_t dyadic_order_2) {
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("sig_kernel_cuda: paths must have length >= 1");
	if (dyadic_order_1 >= 63 || dyadic_order_2 >= 63 || dyadic_order_1 + dyadic_order_2 >= 63)
		throw std::invalid_argument("sig_kernel_cuda: dyadic_order too large");
	if ((length1 - 1) > (UINT64_MAX >> dyadic_order_1) ||
	    (length2 - 1) > (UINT64_MAX >> dyadic_order_2))
		throw std::invalid_argument("sig_kernel_cuda: path length * dyadic refinement would overflow");
}

static SigKernelParams make_params(uint64_t length1_, uint64_t length2_,
	uint64_t dyadic_order_1_, uint64_t dyadic_order_2_) {
	SigKernelParams p;
	p.length2 = length2_;
	p.dyadic_order_1 = dyadic_order_1_;
	p.dyadic_order_2 = dyadic_order_2_;
	p.dyadic_length_1 = ((length1_ - 1) << dyadic_order_1_) + 1;
	p.dyadic_length_2 = ((length2_ - 1) << dyadic_order_2_) + 1;
	p.main_dyadic_length = p.dyadic_length_2 <= p.dyadic_length_1 ? p.dyadic_length_1 : p.dyadic_length_2;
	p.num_anti_diag = 33 + p.main_dyadic_length - 1;
	p.gram_length = (length1_ - 1) * (length2_ - 1);
	p.grid_length = p.dyadic_length_1 * p.dyadic_length_2;
	return p;
}

// Device utility kernels
template<typename T>
__global__ void fill_kernel(T* ptr, T val, uint64_t n) {
	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) ptr[idx] = val;
}

template<typename T>
__global__ void gather_last(const T* src, T* dst, uint64_t stride, uint64_t n) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) dst[i] = src[(i + 1) * stride - 1];
}


template<typename T>
inline __device__ void get_a_b(T& a, T& b, const T* gram, uint64_t idx, T dyadic_frac) {
	const T twelth = static_cast<T>(1.) / 12;
	const T gram_val = gram[idx] * dyadic_frac;
	const T gram_val_2 = gram_val * gram_val * twelth;
	a = static_cast<T>(1.) + static_cast<T>(0.5) * gram_val + gram_val_2;
	b = static_cast<T>(1.) - gram_val_2;
}

template<typename T>
inline __device__ void get_a(T& a, const T* gram, uint64_t idx, T dyadic_frac) {
	const T twelth = static_cast<T>(1.) / 12;
	T gram_val = gram[idx] * dyadic_frac;
	a = static_cast<T>(1.) + gram_val * (0.5 + gram_val * twelth);
}

template<typename T>
inline __device__ void get_b(T& b, const T* gram, uint64_t idx, T dyadic_frac) {
	const T twelth = static_cast<T>(1.) / 12;
	const T gram_val = gram[idx] * dyadic_frac;
	b = static_cast<T>(1.) - gram_val * gram_val * twelth;
}

template<typename T>
inline __device__ void get_a_b_deriv(T& a_deriv, T& b_deriv, const T* gram, uint64_t idx, T dyadic_frac) {
	const T sixth = static_cast<T>(1.) / 6;
	const T gram_val = gram[idx] * dyadic_frac;
	b_deriv = -gram_val * sixth * dyadic_frac;
	a_deriv = static_cast<T>(0.5) * dyadic_frac - b_deriv;
}

template<bool order>
inline __device__ uint64_t diag_gram_idx(uint64_t ii, uint64_t jj, uint64_t gram_stride) {
	if constexpr (order)
		return ii * gram_stride + jj;
	else
		return jj * gram_stride + ii;
}

template<typename T, bool order>
inline __device__ T gram_val(const T* gram, uint64_t ii, uint64_t jj, T dyadic_frac, uint64_t gram_stride) {
	return gram[diag_gram_idx<order>(ii, jj, gram_stride)] * dyadic_frac;
}

template<typename T>
inline __device__ T pde_stencil(T prev_j, T prev_jm1, T pprev_jm1, T deriv) {
	const T twelth = static_cast<T>(1.) / 12;
	const T deriv2 = deriv * deriv * twelth;
	return (prev_j + prev_jm1) * (static_cast<T>(1.) + static_cast<T>(0.5) * deriv + deriv2)
		- pprev_jm1 * (static_cast<T>(1.) - deriv2);
}

template<typename T, bool order> //order is True if dyadic_length_2 <= dyadic_length_1
__device__ void goursat_pde_32(
	T* const initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	T* const diagonals,
	const T* const gram,
	const uint64_t iteration,
	const int num_threads,
	T dyadic_frac,
	const SigKernelParams& p
) {
	const int thread_id = threadIdx.x;

	const uint64_t ord_dyadic_order_1 = order ? p.dyadic_order_1 : p.dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? p.dyadic_order_2 : p.dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? p.dyadic_length_1 : p.dyadic_length_2;

	// Initialise to 1
	for (int i = 0; i < 3; ++i)
		diagonals[i * 33 + thread_id + 1] = static_cast<T>(1.);

	// Indices determine the start points of the antidiagonals in memory
	// Instead of swaping memory, we swap indices to avoid memory copy
	int prev_prev_diag_idx = 0;
	int prev_diag_idx = 33;
	int next_diag_idx = 66;

	if (thread_id == 0) {
		diagonals[prev_prev_diag_idx] = initial_condition[0];
		diagonals[prev_diag_idx] = initial_condition[1];
	}

	__syncwarp(0xFFFFFFFF);

	for (uint64_t q = 2; q < p.num_anti_diag; ++q) { // First two antidiagonals are initialised to 1

		uint64_t startj, endj;
		if (ord_dyadic_length_1 > q) startj = 1;
		else startj = q - ord_dyadic_length_1 + 1;
		if (num_threads + 1 > q) endj = q;
		else endj = num_threads + 1;

		const uint64_t j = startj + thread_id;

		if (j < endj) {

			// Make sure correct initial condition is filled in for first thread
			if (thread_id == 0 && q < ord_dyadic_length_1) {
				diagonals[next_diag_idx] = initial_condition[q];
			}

			const uint64_t i = q - j;  // Calculate corresponding i (since i + j = q)
			const uint64_t ii = ((i - 1) >> ord_dyadic_order_1);
			const uint64_t jj = ((j + iteration * 32 - 1) >> ord_dyadic_order_2);

			const T deriv = gram_val<T, order>(gram, ii, jj, dyadic_frac, p.length2 - 1);
			diagonals[next_diag_idx + j] = pde_stencil(
				diagonals[prev_diag_idx + j], diagonals[prev_diag_idx + j - 1],
				diagonals[prev_prev_diag_idx + j - 1], deriv);

		}

		// Wait for all threads to finish
		__syncwarp(0xFFFFFFFF);

		// Overwrite initial condition with result
		// Safe to do since we won't be using initial_condition[q-num_threads] any more
		if (thread_id == 0 && q >= num_threads && q - num_threads < ord_dyadic_length_1)
			initial_condition[q - num_threads] = diagonals[next_diag_idx + num_threads];

		int temp = prev_prev_diag_idx;
		prev_prev_diag_idx = prev_diag_idx;
		prev_diag_idx = next_diag_idx;
		next_diag_idx = temp;
	}
}

template<typename T>
__global__ void goursat_pde(
	T* const initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	const T* const gram,
	T dyadic_frac,
	const SigKernelParams p
) {
	const int blockId = blockIdx.x;
	const T* const gram_ = gram + blockId * p.gram_length;

	__shared__ T diagonals[99]; // Three diagonals of length 33 (32 + initial condition) are rotated and reused

	if (p.dyadic_length_2 <= p.dyadic_length_1) {
		T* const initial_condition_ = initial_condition + blockId * p.dyadic_length_1;

		const uint64_t num_full_runs = (p.dyadic_length_2 - 1) / 32;
		const uint64_t remainder = (p.dyadic_length_2 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32<T, true>(initial_condition_, diagonals, gram_, i, 32, dyadic_frac, p);

		if (remainder)
			goursat_pde_32<T, true>(initial_condition_, diagonals, gram_, num_full_runs, remainder, dyadic_frac, p);
	}
	else {
		T* const initial_condition_ = initial_condition + blockId * p.dyadic_length_2;

		const uint64_t num_full_runs = (p.dyadic_length_1 - 1) / 32;
		const uint64_t remainder = (p.dyadic_length_1 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32<T, false>(initial_condition_, diagonals, gram_, i, 32, dyadic_frac, p);

		if (remainder)
			goursat_pde_32<T, false>(initial_condition_, diagonals, gram_, num_full_runs, remainder, dyadic_frac, p);
	}
}

template<typename T, bool order>
__device__ void goursat_pde_32_full(
	T* const pde_grid, //32 x L2
	const T* const gram,
	const uint64_t iteration,
	const int num_threads,
	T dyadic_frac,
	const SigKernelParams& p
) {
	const int thread_id = threadIdx.x;
	T* const pde_grid_ = order ? pde_grid + iteration * 32 : pde_grid + iteration * 32 * p.dyadic_length_2;

	const uint64_t ord_dyadic_order_1 = order ? p.dyadic_order_1 : p.dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? p.dyadic_order_2 : p.dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? p.dyadic_length_1 : p.dyadic_length_2;

	__syncwarp(0xFFFFFFFF);

	for (uint64_t q = 2; q < p.num_anti_diag; ++q) { // First two antidiagonals are initialised to 1

		uint64_t startj, endj;
		if (ord_dyadic_length_1 > q) startj = 1;
		else startj = q - ord_dyadic_length_1 + 1;
		if (num_threads + 1 > q) endj = q;
		else endj = num_threads + 1;

		const uint64_t j = startj + thread_id;

		if (j < endj) {

			const uint64_t i = q - j;  // Calculate corresponding i (since i + j = q)
			const uint64_t ii = ((i - 1) >> ord_dyadic_order_1);
			const uint64_t jj = ((j + iteration * 32 - 1) >> ord_dyadic_order_2);

			const T deriv = gram_val<T, order>(gram, ii, jj, dyadic_frac, p.length2 - 1);

			if constexpr (order) {
				pde_grid_[i * p.dyadic_length_2 + j] = pde_stencil(
					pde_grid_[(i - 1) * p.dyadic_length_2 + j], pde_grid_[i * p.dyadic_length_2 + (j - 1)],
					pde_grid_[(i - 1) * p.dyadic_length_2 + j - 1], deriv);
			}
			else {
				pde_grid_[j * p.dyadic_length_2 + i] = pde_stencil(
					pde_grid_[(j - 1) * p.dyadic_length_2 + i], pde_grid_[j * p.dyadic_length_2 + (i - 1)],
					pde_grid_[(j - 1) * p.dyadic_length_2 + i - 1], deriv);
			}

		}

		// Wait for all threads to finish
		__syncwarp(0xFFFFFFFF);
	}
}

template<typename T>
__global__ void goursat_pde_full(
	T* const pde_grid,
	const T* const gram,
	T dyadic_frac,
	const SigKernelParams p
) {
	const int blockId = blockIdx.x;

	const T* const gram_ = gram + blockId * p.gram_length;
	T* const pde_grid_ = pde_grid + blockId * p.grid_length;

	if (p.dyadic_length_2 <= p.dyadic_length_1) {
		const uint64_t num_full_runs = (p.dyadic_length_2 - 1) / 32;
		const uint64_t remainder = (p.dyadic_length_2 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32_full<T, true>(pde_grid_, gram_, i, 32, dyadic_frac, p);

		if (remainder)
			goursat_pde_32_full<T, true>(pde_grid_, gram_, num_full_runs, remainder, dyadic_frac, p);
	}
	else {
		const uint64_t num_full_runs = (p.dyadic_length_1 - 1) / 32;
		const uint64_t remainder = (p.dyadic_length_1 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32_full<T, false>(pde_grid_, gram_, i, 32, dyadic_frac, p);

		if (remainder)
			goursat_pde_32_full<T, false>(pde_grid_, gram_, num_full_runs, remainder, dyadic_frac, p);
	}
}

template<typename T>
void sig_kernel_cuda_(
	const T* const gram,
	T* const out,
	const uint64_t batch_size_,
	const uint64_t dimension_,
	const uint64_t length1_,
	const uint64_t length2_,
	const uint64_t dyadic_order_1_,
	const uint64_t dyadic_order_2_,
	const bool return_grid
) {
	if (dimension_ == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	validate_sig_kernel_dims_(length1_, length2_, dyadic_order_1_, dyadic_order_2_);

	const SigKernelParams p = make_params(length1_, length2_, dyadic_order_1_, dyadic_order_2_);
	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1_ + dyadic_order_2_));

	if (!return_grid) {
		CudaBuf<T> initial_condition(p.main_dyadic_length * batch_size_ * sizeof(T));
		const uint64_t fill_n = p.main_dyadic_length * batch_size_;
		fill_kernel<<<static_cast<unsigned int>((fill_n + 255) / 256), 256U>>>(initial_condition.get(), static_cast<T>(1.), fill_n);

		goursat_pde<<<static_cast<unsigned int>(batch_size_), 32U>>>(initial_condition.get(), gram, dyadic_frac, p);

		gather_last<<<static_cast<unsigned int>((batch_size_ + 255) / 256), 256U>>>(initial_condition.get(), out, p.main_dyadic_length, batch_size_);
	}
	else {
		const uint64_t fill_n = batch_size_ * p.grid_length;
		fill_kernel<<<static_cast<unsigned int>((fill_n + 255) / 256), 256U>>>(out, static_cast<T>(1.), fill_n);

		goursat_pde_full<<<static_cast<unsigned int>(batch_size_), 32U>>>(out, gram, dyadic_frac, p);
	}

	check_cuda_kernel_launch();
}

template<typename T, bool order> //order is True if dyadic_length_2 <= dyadic_length_1
__device__ void goursat_pde_32_deriv(
	const T* derivs,
	const T* const k_grid,
	T* const out,
	T* const initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	T* const a_initial_condition,
	T* const b_initial_condition,
	T* const diagonals,
	T* const a,
	T* const b,
	const T* const gram,
	const uint64_t iteration,
	const int num_threads,
	T dyadic_frac,
	bool return_grid,
	uint64_t derivs_length,
	const SigKernelParams& p
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

	const int thread_id = threadIdx.x;

	// As with the diagonal method for sig_kernel, it matters which of
	// dyadic_length_1 and dyadic_length_2 is longer.
	const uint64_t ord_dyadic_order_1 = order ? p.dyadic_order_1 : p.dyadic_order_2;
	const uint64_t ord_dyadic_order_2 = order ? p.dyadic_order_2 : p.dyadic_order_1;
	const uint64_t ord_dyadic_length_1 = order ? p.dyadic_length_1 : p.dyadic_length_2;
	const uint64_t ord_dyadic_length_2 = order ? p.dyadic_length_2 : p.dyadic_length_1;

	// Ptrs for diagonals
	T* prev_prev_diag = diagonals;
	T* prev_diag = prev_prev_diag + 33;
	T* next_diag = prev_diag + 33;

	// k_grid ptrs
	const T* k11, * k12, * k21;

	// Initialization
	for (int i = 0; i < 3; ++i)
		diagonals[i * 33 + thread_id + 1] = static_cast<T>(0.);

	a[thread_id + 1] = static_cast<T>(1.);
	b[thread_id + 1] = static_cast<T>(1.);

	if (thread_id == 0) {
		a[0] = static_cast<T>(1.);
		b[0] = static_cast<T>(1.);

		*prev_prev_diag = initial_condition[0];
		*prev_diag = initial_condition[1];

		T last_deriv = derivs[derivs_length - 1];

		if (iteration == 0) {
			*(prev_diag + 1) = last_deriv;
			T da, db;
			get_a_b_deriv(da, db, gram, p.gram_length - 1, dyadic_frac);

			//Update dF / dx for first value
			k21 = k_grid + p.grid_length - 2;
			k12 = k_grid + p.grid_length - p.dyadic_length_2 - 1; //NOT ord_dyadic_length_2 here, as we are indexing k_grid
			k11 = k12 - 1;
			out[p.gram_length - 1] += last_deriv * (((*k21) + (*k12)) * da - *(k11)*db);
		}
	}

	__syncwarp(0xFFFFFFFF);

	for (uint64_t q = (iteration == 0) ? 3 : 2; q < p.num_anti_diag + 2; ++q) {

		uint64_t startj, endj;
		int64_t q_ = q - 2;
		startj = ord_dyadic_length_1 > q_ ? 1 : q_ - ord_dyadic_length_1 + 1;
		endj = num_threads + 1 > q_ ? q_ : num_threads + 1;

		uint64_t j = startj + thread_id;

		// Make sure initial condition is filled in for first thread
		if (thread_id == 0 && q_ < ord_dyadic_length_1) {
			b[0] = b_initial_condition[q_];
		}

		if (j < endj) {
			const uint64_t i = q_ - j;
			const uint64_t i_rev = ord_dyadic_length_1 - i - 1;
			const uint64_t j_rev = ord_dyadic_length_2 - j - 1 - iteration * 32;
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = diag_gram_idx<order>(ii, jj, p.length2 - 1);

			get_b(b[j], gram, gram_idx, dyadic_frac);
		}

		__syncwarp(0xFFFFFFFF);

		if (thread_id == 0 && q_ >= num_threads && q_ - num_threads < ord_dyadic_length_1) {
			b_initial_condition[q_ - num_threads] = b[num_threads];
		}

		q_ = q - 1;
		startj = ord_dyadic_length_1 > q_ ? 1 : q_ - ord_dyadic_length_1 + 1;
		endj = num_threads + 1 > q_ ? q_ : num_threads + 1;

		j = startj + thread_id;

		// Make sure initial condition is filled in for first thread
		if (thread_id == 0 && q_ < ord_dyadic_length_1) {
			a[0] = a_initial_condition[q_];
		}

		if (j < endj) {
			const uint64_t i = q_ - j;
			const uint64_t i_rev = ord_dyadic_length_1 - i - 1;
			const uint64_t j_rev = ord_dyadic_length_2 - j - 1 - iteration * 32;
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = diag_gram_idx<order>(ii, jj, p.length2 - 1);

			get_a(a[j], gram, gram_idx, dyadic_frac);
		}

		__syncwarp(0xFFFFFFFF);

		if (thread_id == 0 && q_ >= num_threads && q_ - num_threads < ord_dyadic_length_1) {
			a_initial_condition[q_ - num_threads] = a[num_threads];
		}

		startj = ord_dyadic_length_1 > q ? 1 : q - ord_dyadic_length_1 + 1;
		endj = num_threads + 1 > q ? q : num_threads + 1;

		j = startj + thread_id;

		// Make sure initial condition is filled in for first thread
		if (thread_id == 0 && q < ord_dyadic_length_1) {
			*(next_diag) = initial_condition[q];
		}

		if (j < endj) {
			const uint64_t i = q - j;
			const uint64_t i_rev = ord_dyadic_length_1 - i - 1;
			const uint64_t j_rev = ord_dyadic_length_2 - j - 1 - iteration * 32;
			const uint64_t idx = order ? (i_rev + 1) * p.dyadic_length_2 + (j_rev + 1) : (j_rev + 1) * p.dyadic_length_2 + (i_rev + 1); //NOT ord_dyadic_length_2 here as we are indexing k_grid
			const uint64_t ii = (i_rev >> ord_dyadic_order_1);
			const uint64_t jj = (j_rev >> ord_dyadic_order_2);
			const uint64_t gram_idx = diag_gram_idx<order>(ii, jj, p.length2 - 1);

			T da, db;
			get_a_b_deriv(da, db, gram, gram_idx, dyadic_frac);

			*(next_diag + j) = *(prev_diag + j - 1) * a[j - 1] + *(prev_diag + j) * a[j] - *(prev_prev_diag + j - 1) * b[j - 1];

			if (return_grid)
				*(next_diag + j) += *(derivs + idx);

			k12 = k_grid + idx - 1;
			k21 = k_grid + idx - p.dyadic_length_2; //NOT ord_dyadic_length_2 here as we are indexing k_grid
			k11 = k_grid + idx - p.dyadic_length_2 - 1;
			T result = *(next_diag + j) * ((*(k12)+*(k21)) * da - *(k11)*db);

			// Avoid race conditions for non-zero dyadic orders
			myAtomicAdd(&out[gram_idx], result);
		}

		__syncwarp(0xFFFFFFFF);

		if (thread_id == 0 && q >= num_threads && q - num_threads < ord_dyadic_length_1) {
			initial_condition[q - num_threads] = *(next_diag + num_threads);
		}

		T* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;
	}
}

template<typename T>
__global__ void goursat_pde_deriv(
	T* initial_condition, //This is the top row of the grid, which will be overwritten
	T* a_initial_condition,
	T* b_initial_condition,
	const T* gram,
	const T* derivs,
	const T* k_grid,
	T* out,
	T dyadic_frac,
	bool return_grid,
	uint64_t derivs_length,
	const SigKernelParams p
) {
	const int blockId = blockIdx.x;
	const T* const gram_ = gram + blockId * p.gram_length;
	const T* derivs_ = derivs + blockId * derivs_length;
	const T* const k_grid_ = k_grid + blockId * p.grid_length;
	T* const out_ = out + blockId * p.gram_length;

	__shared__ T diagonals[99]; // Three diagonals of length 33 (32 + initial condition) are rotated and reused
	__shared__ T a[33];
	__shared__ T b[33];

	if (p.dyadic_length_2 <= p.dyadic_length_1) {
		T* const initial_condition_ = initial_condition + blockId * p.dyadic_length_1;
		T* const a_initial_condition_ = a_initial_condition + blockId * p.dyadic_length_1;
		T* const b_initial_condition_ = b_initial_condition + blockId * p.dyadic_length_1;

		const uint64_t num_full_runs = (p.dyadic_length_2 - 1) / 32;
		const uint64_t remainder = (p.dyadic_length_2 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32_deriv<T, true>(derivs_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, i, 32, dyadic_frac, return_grid, derivs_length, p);

		if (remainder)
			goursat_pde_32_deriv<T, true>(derivs_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, num_full_runs, remainder, dyadic_frac, return_grid, derivs_length, p);
	}
	else {
		T* const initial_condition_ = initial_condition + blockId * p.dyadic_length_2;
		T* const a_initial_condition_ = a_initial_condition + blockId * p.dyadic_length_2;
		T* const b_initial_condition_ = b_initial_condition + blockId * p.dyadic_length_2;

		const uint64_t num_full_runs = (p.dyadic_length_1 - 1) / 32;
		const uint64_t remainder = (p.dyadic_length_1 - 1) % 32;

		for (int i = 0; i < num_full_runs; ++i)
			goursat_pde_32_deriv<T, false>(derivs_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, i, 32, dyadic_frac, return_grid, derivs_length, p);

		if (remainder)
			goursat_pde_32_deriv<T, false>(derivs_, k_grid_, out_, initial_condition_, a_initial_condition_, b_initial_condition_, diagonals, a, b, gram_, num_full_runs, remainder, dyadic_frac, return_grid, derivs_length, p);
	}
}

template<typename T>
void sig_kernel_backprop_cuda_(
	const T* gram,
	T* const out,
	const T* derivs,
	const T* k_grid,
	const uint64_t batch_size_,
	const uint64_t dimension_,
	const uint64_t length1_,
	const uint64_t length2_,
	const uint64_t dyadic_order_1_,
	const uint64_t dyadic_order_2_,
	const bool return_grid
) {
	if (dimension_ == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	validate_sig_kernel_dims_(length1_, length2_, dyadic_order_1_, dyadic_order_2_);

	const SigKernelParams p = make_params(length1_, length2_, dyadic_order_1_, dyadic_order_2_);
	const T dyadic_frac = static_cast<T>(1.) / (1ULL << (dyadic_order_1_ + dyadic_order_2_));
	const uint64_t derivs_length_ = return_grid ? p.grid_length : 1;

	CUDA_CHECK(cudaMemset(out, 0, batch_size_ * p.gram_length * sizeof(T)));

	// Single allocation for all 3 initial condition buffers
	const uint64_t ic_size = p.main_dyadic_length * batch_size_;
	CudaBuf<T> d_ic_buf(3 * ic_size * sizeof(T));
	CUDA_CHECK(cudaMemset(d_ic_buf.get(), 0, 3 * ic_size * sizeof(T)));
	T* d_initial_condition = d_ic_buf.get();
	T* d_a_initial_condition = d_ic_buf.get() + ic_size;
	T* d_b_initial_condition = d_ic_buf.get() + 2 * ic_size;

	goursat_pde_deriv<<<static_cast<unsigned int>(batch_size_), 32U>>>(d_initial_condition, d_a_initial_condition, d_b_initial_condition, gram, derivs, k_grid, out, dyadic_frac, return_grid, derivs_length_, p);

	check_cuda_kernel_launch();
}

#include "cu_macros.h"


extern "C" {


	CUSIG_API int sig_kernel_cuda_f(const float* const gram, float* const out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_cuda_<float>(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int sig_kernel_cuda_d(const double* const gram, double* const out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_cuda_<double>(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}


	CUSIG_API int sig_kernel_backprop_cuda_f(const float* const gram, float* const out, const float* const derivs, const float* const k_grid, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_backprop_cuda_<float>(gram, out, derivs, k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int sig_kernel_backprop_cuda_d(const double* const gram, double* const out, const double* const derivs, const double* const k_grid, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_backprop_cuda_<double>(gram, out, derivs, k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid));
	}
}
