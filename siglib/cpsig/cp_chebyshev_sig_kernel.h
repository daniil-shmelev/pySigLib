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

#pragma once
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "multithreading.h"
#include "../shared/chebyshev_sig_kernel.h"

namespace cpsig {

template<std::floating_point T>
inline thread_local std::vector<T> chebyshev_forward_workspace;

template<std::floating_point T>
inline thread_local std::vector<T> chebyshev_backward_workspace;

inline uint64_t chebyshev_checked_product(uint64_t a, uint64_t b) {
	if (b && a > std::numeric_limits<size_t>::max() / b)
		throw std::invalid_argument("Chebyshev signature kernel dimensions overflow");
	return a * b;
}

template<std::floating_point T, int P = -1, int K = -1>
void chebyshev_forward_pair(
	const T* gram, T* out, T* state, uint64_t length1, uint64_t length2,
	uint64_t degree, uint64_t terms, bool return_grid
) {
	using chebyshev_sig_kernel::Corner;
	const uint64_t p = P < 0 ? degree : P;
	const uint64_t rows = length1 - 1, cols = length2 - 1;
	const uint64_t width = p + 5, size = p + 1, state_width = 2 * p + 8;
	auto& frontier = chebyshev_forward_workspace<T>;
	frontier.assign(cols * width, T(0));
	for (uint64_t j = 0; j < cols; ++j) {
		frontier[j * width + size] = T(1);
		frontier[j * width + size + 2] = T(1);
	}
	if (return_grid) {
		std::fill_n(out, length2, T(1));
		for (uint64_t i = 1; i < length1; ++i) out[i * length2] = T(1);
	}
	std::array<T, 36> bottom_storage{}, top_storage{}, right{};
	Corner<T> corner{ T(1), T(0) };
	for (uint64_t i = 0; i < rows; ++i) {
		T* bottom = bottom_storage.data();
		T* top = top_storage.data();
		std::fill_n(bottom, width, T(0));
		bottom[size] = bottom[size + 2] = T(1);
		for (uint64_t j = 0; j < cols; ++j) {
			T* left = frontier.data() + j * width;
			const Corner<T> start{ bottom[size], bottom[size + 1] };
			const Corner<T> bottom_end{ bottom[size + 2], bottom[size + 3] };
			const Corner<T> left_end{ left[size + 2], left[size + 3] };
			if (state) {
				T* cell = state + (i * cols + j) * state_width;
				std::copy_n(bottom, size, cell);
				std::copy_n(left, size, cell + size);
				cell[2 * size] = start.hi; cell[2 * size + 1] = start.lo;
				cell[2 * size + 2] = bottom_end.hi; cell[2 * size + 3] = bottom_end.lo;
				cell[2 * size + 4] = left_end.hi; cell[2 * size + 5] = left_end.lo;
			}
			corner = chebyshev_sig_kernel::cell_forward<T, P, K, true>(
				gram[i * cols + j], bottom, left, start, bottom_end, left_end,
				top, right.data(), degree, terms);
			top[size] = left_end.hi; top[size + 1] = left_end.lo;
			top[size + 2] = corner.hi; top[size + 3] = corner.lo;
			right[size] = bottom_end.hi; right[size + 1] = bottom_end.lo;
			right[size + 2] = corner.hi; right[size + 3] = corner.lo;
			std::copy_n(right.data(), width, left);
			std::swap(bottom, top);
			if (return_grid) out[(i + 1) * length2 + j + 1] = chebyshev_sig_kernel::corner_value(corner);
		}
	}
	if (!return_grid) *out = chebyshev_sig_kernel::corner_value(corner);
}

template<std::floating_point T, int P = -1, int K = -1>
void chebyshev_backward_pair(
	const T* gram, T* grad, const T* derivs, const T* state,
	uint64_t length1, uint64_t length2, uint64_t degree, uint64_t terms, bool return_grid
) {
	const uint64_t p = P < 0 ? degree : P;
	const uint64_t rows = length1 - 1, cols = length2 - 1;
	const uint64_t size = p + 1, width = p + 3, state_width = 2 * p + 8;
	const uint64_t state_length = state ? 0 : rows * cols * state_width;
	auto& workspace = chebyshev_backward_workspace<T>;
	workspace.assign(state_length + cols * width, T(0));
	if (!state) {
		T ignored;
		chebyshev_forward_pair<T, P, K>(gram, &ignored, workspace.data(), length1, length2, degree, terms, false);
		state = workspace.data();
	}
	T* frontier = workspace.data() + state_length;
	std::array<T, 34> horizontal{}, bottom_adj{}, left_adj{};
	for (uint64_t i = rows; i-- > 0;) {
		std::fill_n(horizontal.data(), width, T(0));
		for (uint64_t j = cols; j-- > 0;) {
			T* vertical = frontier + j * width;
			const T* cell = state + (i * cols + j) * state_width;
			T corner_adj = horizontal[size + 1] + vertical[size + 1];
			if (return_grid) corner_adj += derivs[(i + 1) * length2 + j + 1];
			else if (i + 1 == rows && j + 1 == cols) corner_adj += derivs[0];
			const auto local = chebyshev_sig_kernel::cell_backward<T, P, K>(
				gram[i * cols + j], cell, cell + size, { cell[2 * size], cell[2 * size + 1] },
				horizontal.data(), vertical, corner_adj, bottom_adj.data(), left_adj.data(), degree, terms);
			grad[i * cols + j] = local.coefficient;
			bottom_adj[size] = local.start;
			bottom_adj[size + 1] = local.bottom_end + vertical[size];
			left_adj[size] = T(0);
			left_adj[size + 1] = local.left_end + horizontal[size];
			std::copy_n(bottom_adj.data(), width, horizontal.data());
			std::copy_n(left_adj.data(), width, vertical);
		}
	}
}

template<std::floating_point T, int P, int K, bool Backward>
void chebyshev_pair(
	const T* gram, T* out, const T* derivs, const T* saved_state, T* state,
	uint64_t length1, uint64_t length2, uint64_t p, uint64_t k, bool grid
) {
	if constexpr (Backward) chebyshev_backward_pair<T, P, K>(gram, out, derivs, saved_state, length1, length2, p, k, grid);
	else chebyshev_forward_pair<T, P, K>(gram, out, state, length1, length2, p, k, grid);
}

template<std::floating_point T, bool Backward>
void chebyshev_pair_dispatch(
	const T* gram, T* out, const T* derivs, const T* saved_state, T* state,
	uint64_t length1, uint64_t length2, uint64_t p, uint64_t k, bool grid
) {
#define SIG_CHEB_PAIR(P, K) chebyshev_pair<T, P, K, Backward>(gram, out, derivs, saved_state, state, length1, length2, p, k, grid)
	if (p == k) {
		switch (p) {
		case 1: SIG_CHEB_PAIR(1, 1); return;
		case 2: SIG_CHEB_PAIR(2, 2); return;
		case 3: SIG_CHEB_PAIR(3, 3); return;
		case 4: SIG_CHEB_PAIR(4, 4); return;
		case 5: SIG_CHEB_PAIR(5, 5); return;
		case 6: SIG_CHEB_PAIR(6, 6); return;
		case 7: SIG_CHEB_PAIR(7, 7); return;
		case 8: SIG_CHEB_PAIR(8, 8); return;
		case 9: SIG_CHEB_PAIR(9, 9); return;
		case 10: SIG_CHEB_PAIR(10, 10); return;
		case 11: SIG_CHEB_PAIR(11, 11); return;
		case 12: SIG_CHEB_PAIR(12, 12); return;
		default: break;
		}
	}
	if (p == 1 && k == 2) { SIG_CHEB_PAIR(1, 2); return; }
	if (p == 1 && k == 3) { SIG_CHEB_PAIR(1, 3); return; }
	if (p == 2 && k == 3) { SIG_CHEB_PAIR(2, 3); return; }
	if (p == 4 && k == 6) { SIG_CHEB_PAIR(4, 6); return; }
	if (p == 6 && k == 4) { SIG_CHEB_PAIR(6, 4); return; }
	SIG_CHEB_PAIR(-1, -1);
#undef SIG_CHEB_PAIR
}

template<std::floating_point T, bool Backward>
void chebyshev_sig_kernel(
	const T* gram, T* out, const T* derivs, const T* saved_state, T* state,
	uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2,
	uint64_t p, uint64_t k, bool grid, int n_jobs
) {
	if (dimension == 0 || length1 == 0 || length2 == 0)
		throw std::invalid_argument("Chebyshev signature kernel paths require positive length and dimension");
	if (p == 0 || p > 31 || k > 31 || p > 31 - k)
		throw std::invalid_argument("Chebyshev signature kernel requires p >= 1, K >= 0 and p + K <= 31");
	const uint64_t cells = chebyshev_checked_product(length1 - 1, length2 - 1);
	const uint64_t grid_size = grid ? chebyshev_checked_product(length1, length2) : 1;
	const uint64_t state_size = chebyshev_checked_product(cells, 2 * p + 8);
	const uint64_t out_size = Backward ? cells : grid_size;
	chebyshev_checked_product(chebyshev_checked_product(batch_size, cells), sizeof(T));
	chebyshev_checked_product(chebyshev_checked_product(batch_size, out_size), sizeof(T));
	chebyshev_checked_product(chebyshev_checked_product(batch_size, grid_size), sizeof(T));
	chebyshev_checked_product(chebyshev_checked_product(batch_size, state_size), sizeof(T));
	const uint64_t forward_size = chebyshev_checked_product(length2 - 1, p + 5);
	const uint64_t reverse_size = chebyshev_checked_product(length2 - 1, p + 3);
	chebyshev_checked_product(forward_size, sizeof(T));
	if (state_size > std::numeric_limits<size_t>::max() - reverse_size)
		throw std::invalid_argument("Chebyshev signature kernel workspace overflows");
	chebyshev_checked_product(state_size + reverse_size, sizeof(T));
	if (batch_size == 0) return;
	if constexpr (Backward) { if (cells == 0) return; }
	if (!out) throw std::invalid_argument("Chebyshev signature kernel output pointer must not be null");
	if (cells == 0) { std::fill_n(out, batch_size * out_size, T(1)); return; }
	if (!gram) throw std::invalid_argument("Chebyshev signature kernel Gram pointer must not be null");
	if constexpr (Backward) {
		if (!derivs) throw std::invalid_argument("Chebyshev signature kernel derivative pointer must not be null");
	}
	auto work = [&](uint64_t first, uint64_t last) {
		for (uint64_t i = first; i < last; ++i) {
			chebyshev_pair_dispatch<T, Backward>(
				gram + i * cells, out + i * out_size, derivs ? derivs + i * grid_size : nullptr,
				saved_state ? saved_state + i * state_size : nullptr, state ? state + i * state_size : nullptr,
				length1, length2, p, k, grid);
		}
	};
	if (batch_size == 1 || n_jobs == 1) work(0, batch_size);
	else spawn_batch_threads(batch_size, n_jobs, work);
}

}  // namespace cpsig
