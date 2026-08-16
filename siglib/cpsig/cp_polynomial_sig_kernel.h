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
#include "cppch.h"

#include "macros.h"
#include "multithreading.h"
#include "cp_vector_funcs.h"
#include "../shared/preparation/polynomial_sig_kernel/polynomial_sig_kernel_tables.h"


template<std::floating_point T>
using sig_poly_tables = PolynomialSigKernelTables<T>;

template<std::floating_point T>
struct sig_poly_table_cache {
	std::mutex mutex;
	std::vector<std::shared_ptr<const sig_poly_tables<T>>> tables;
};

template<std::floating_point T>
sig_poly_table_cache<T>& sig_poly_table_cache_() {
	static sig_poly_table_cache<T> cache;
	return cache;
}

template<std::floating_point T>
std::shared_ptr<const sig_poly_tables<T>> get_sig_poly_tables_(uint64_t order) {
	auto& cache = sig_poly_table_cache_<T>();
	std::lock_guard<std::mutex> lock(cache.mutex);
	if (cache.tables.empty())
		cache.tables.resize(65);
	auto& tables = cache.tables[order];
	if (!tables)
		tables = std::make_shared<const sig_poly_tables<T>>(order);
	return tables;
}

template<std::floating_point T>
void clear_sig_poly_table_cache_() {
	auto& cache = sig_poly_table_cache_<T>();
	std::lock_guard<std::mutex> lock(cache.mutex);
	if (cache.tables.empty())
		return;
	decltype(cache.tables)().swap(cache.tables);
}

template<std::floating_point T>
inline thread_local std::vector<T> polynomial_sig_kernel_workspace_;

template<std::floating_point T>
inline thread_local std::vector<T> polynomial_sig_kernel_backprop_workspace_;

template<std::floating_point T, uint64_t FixedSize = 0>
FORCE_INLINE void sig_poly_tile_update_(
	T rho,
	const T* RESTRICT bottom,
	const T* RESTRICT left,
	T bottom_endpoint,
	T left_endpoint,
	T* RESTRICT top,
	T* RESTRICT right,
	T* RESTRICT rho_powers,
	T& top_endpoint,
	T& right_endpoint,
	const sig_poly_tables<T>& tables
) {
	const uint64_t size = FixedSize == 0 ? tables.size : FixedSize;
	rho_powers[0] = static_cast<T>(1);
	for (uint64_t n = 1; n < size; ++n)
		rho_powers[n] = rho_powers[n - 1] * rho;

	top[0] = left_endpoint;
	right[0] = bottom_endpoint;
	top_endpoint = left_endpoint;
	right_endpoint = bottom_endpoint;

	for (uint64_t n = 1; n < size; ++n) {
		T top_opposite = static_cast<T>(0);
		T right_opposite = static_cast<T>(0);
		const T* const mat2_row = tables.mat2.data() + n * size;
		if (size >= 8) {
			dot_product_pair(
				mat2_row, left, bottom, size, top_opposite, right_opposite);
		}
		else {
			for (uint64_t k = 0; k < size; ++k) {
				top_opposite += mat2_row[k] * left[k];
				right_opposite += mat2_row[k] * bottom[k];
			}
		}

		T top_same = bottom[n];
		T right_same = left[n];
		const T* const mat1_row = tables.mat1.data() + n * size;
		for (uint64_t k = 1; k < n; ++k) {
			const T factor = mat1_row[k] * rho_powers[n - k];
			top_same += factor * bottom[k];
			right_same += factor * left[k];
		}

		const T top_value = top_same + rho_powers[n] * top_opposite;
		const T right_value = right_same + rho_powers[n] * right_opposite;
		top[n] = top_value;
		right[n] = right_value;
		top_endpoint += top_value;
		right_endpoint += right_value;
	}
}

template<std::floating_point T, typename Tables, uint64_t FixedSize, auto TileUpdate>
void polynomial_sig_kernel_pair_(
	const T* RESTRICT gram,
	T* RESTRICT out,
	T* RESTRICT state,
	uint64_t length1,
	uint64_t length2,
	bool return_grid,
	const Tables& tables
) {
	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	if (rows == 0 || cols == 0) {
		std::fill(out, out + (return_grid ? length1 * length2 : 1), static_cast<T>(1));
		return;
	}

	const uint64_t size = FixedSize == 0 ? tables.size : FixedSize;
	auto& workspace = polynomial_sig_kernel_workspace_<T>;
	workspace.resize((cols + 4) * size + cols);
	T* const frontier = workspace.data();
	T* const bottom_storage = frontier + cols * size;
	T* const top_storage = bottom_storage + size;
	T* const right = top_storage + size;
	T* const rho_powers = right + size;
	T* const frontier_endpoints = rho_powers + size;
	if (return_grid) {
		std::fill(out, out + length2, static_cast<T>(1));
		for (uint64_t i = 1; i < length1; ++i)
			out[i * length2] = static_cast<T>(1);
	}

	std::fill(frontier, frontier + cols * size, static_cast<T>(0));
	for (uint64_t j = 0; j < cols; ++j) {
		frontier[j * size] = static_cast<T>(1);
		frontier_endpoints[j] = static_cast<T>(1);
	}

	T top_endpoint = static_cast<T>(1);
	T right_endpoint = static_cast<T>(1);
	for (uint64_t i = 0; i < rows; ++i) {
		T* bottom = bottom_storage;
		T* top = top_storage;
		T bottom_endpoint = static_cast<T>(1);
		std::fill(bottom, bottom + size, static_cast<T>(0));
		bottom[0] = static_cast<T>(1);

		const T* const gram_row = gram + i * cols;
		for (uint64_t j = 0; j < cols; ++j) {
			T* const left = frontier + j * size;
			if (state) {
				T* const tile_state = state + 2 * (i * cols + j) * size;
				std::copy_n(bottom, size, tile_state);
				std::copy_n(left, size, tile_state + size);
			}
			TileUpdate(
				gram_row[j], bottom, left, bottom_endpoint, frontier_endpoints[j],
				top, right, rho_powers, top_endpoint, right_endpoint, tables
			);
			for (uint64_t k = 0; k < size; ++k)
				left[k] = right[k];
			frontier_endpoints[j] = right_endpoint;
			std::swap(bottom, top);
			bottom_endpoint = top_endpoint;
			if (return_grid)
				out[(i + 1) * length2 + j + 1] = static_cast<T>(0.5) * (top_endpoint + right_endpoint);
		}
	}

	if (!return_grid)
		*out = static_cast<T>(0.5) * (top_endpoint + right_endpoint);
}

template<std::floating_point T>
void sig_kernel_poly_pair_dispatch_(
	const T* RESTRICT gram,
	T* RESTRICT out,
	T* RESTRICT state,
	uint64_t length1,
	uint64_t length2,
	bool return_grid,
	const sig_poly_tables<T>& tables
) {
	switch (tables.size) {
	case 3: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 3, sig_poly_tile_update_<T, 3>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 4: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 4, sig_poly_tile_update_<T, 4>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 5: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 5, sig_poly_tile_update_<T, 5>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 6: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 6, sig_poly_tile_update_<T, 6>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 7: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 7, sig_poly_tile_update_<T, 7>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 8: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 8, sig_poly_tile_update_<T, 8>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 9: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 9, sig_poly_tile_update_<T, 9>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 10: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 10, sig_poly_tile_update_<T, 10>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 11: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 11, sig_poly_tile_update_<T, 11>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 12: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 12, sig_poly_tile_update_<T, 12>>(gram, out, state, length1, length2, return_grid, tables); return;
	case 13: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 13, sig_poly_tile_update_<T, 13>>(gram, out, state, length1, length2, return_grid, tables); return;
	default: polynomial_sig_kernel_pair_<T, sig_poly_tables<T>, 0, sig_poly_tile_update_<T>>(gram, out, state, length1, length2, return_grid, tables);
	}
}

template<std::floating_point T, uint64_t FixedSize = 0>
FORCE_INLINE T sig_poly_tile_backprop_(
	T rho,
	const T* RESTRICT bottom,
	const T* RESTRICT left,
	const T* RESTRICT top_derivs,
	const T* RESTRICT right_derivs,
	T* RESTRICT bottom_derivs,
	T* RESTRICT left_derivs,
	T* RESTRICT rho_powers,
	const sig_poly_tables<T>& tables
) {
	const uint64_t size = FixedSize == 0 ? tables.size : FixedSize;
	rho_powers[0] = static_cast<T>(1);
	for (uint64_t n = 1; n < size; ++n)
		rho_powers[n] = rho_powers[n - 1] * rho;

	for (uint64_t k = 0; k < size; ++k) {
		bottom_derivs[k] = right_derivs[0];
		left_derivs[k] = top_derivs[0];
	}

	T rho_deriv = static_cast<T>(0);
	for (uint64_t n = 1; n < size; ++n) {
		const T top_deriv = top_derivs[n];
		const T right_deriv = right_derivs[n];
		bottom_derivs[n] += top_deriv;
		left_derivs[n] += right_deriv;

		const T* const mat1_row = tables.mat1.data() + n * size;
		const T* const mat1_deriv_row = tables.mat1_deriv.data() + n * size;
		for (uint64_t k = 1; k < n; ++k) {
			const uint64_t power = n - k;
			const T factor = mat1_row[k] * rho_powers[power];
			bottom_derivs[k] += factor * top_deriv;
			left_derivs[k] += factor * right_deriv;
			rho_deriv += mat1_deriv_row[k] * rho_powers[power - 1]
				* (bottom[k] * top_deriv + left[k] * right_deriv);
		}

		T top_opposite = static_cast<T>(0);
		T right_opposite = static_cast<T>(0);
		const T* const mat2_row = tables.mat2.data() + n * size;
		for (uint64_t k = 0; k < size; ++k) {
			const T factor = rho_powers[n] * mat2_row[k];
			top_opposite += mat2_row[k] * left[k];
			right_opposite += mat2_row[k] * bottom[k];
			left_derivs[k] += factor * top_deriv;
			bottom_derivs[k] += factor * right_deriv;
		}
		rho_deriv += static_cast<T>(n) * rho_powers[n - 1]
			* (top_opposite * top_deriv + right_opposite * right_deriv);
	}
	return rho_deriv;
}

template<std::floating_point T, uint64_t FixedSize, auto TileBackprop>
void polynomial_sig_kernel_backprop_pair_(
	const T* RESTRICT gram,
	T* RESTRICT gram_derivs,
	const T* RESTRICT output_derivs,
	const T* RESTRICT state,
	uint64_t length1,
	uint64_t length2,
	bool return_grid,
	const sig_poly_tables<T>& tables
) {
	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	if (rows == 0 || cols == 0)
		return;
	const uint64_t size = FixedSize == 0 ? tables.size : FixedSize;
	const uint64_t state_size = 2 * rows * cols * size;
	const bool has_state = state != nullptr;
	auto& workspace = polynomial_sig_kernel_backprop_workspace_<T>;
	workspace.resize((has_state ? 0 : state_size) + (cols + 4) * size);
	T* const regenerated_state = has_state ? nullptr : workspace.data();
	if (!has_state) {
		T ignored;
		sig_kernel_poly_pair_dispatch_<T>(
			gram, &ignored, regenerated_state, length1, length2, false, tables);
		state = regenerated_state;
	}

	T* const adjoints = workspace.data() + (has_state ? 0 : state_size);
	T* const vertical = adjoints;
	T* const horizontal = vertical + cols * size;
	T* const bottom_derivs = horizontal + size;
	T* const left_derivs = bottom_derivs + size;
	T* const rho_powers = left_derivs + size;
	std::fill(vertical, vertical + cols * size, static_cast<T>(0));

	for (uint64_t i = rows; i-- > 0;) {
		std::fill(horizontal, horizontal + size, static_cast<T>(0));
		for (uint64_t j = cols; j-- > 0;) {
			T* const right_derivs = vertical + j * size;
			T local_deriv = static_cast<T>(0);
			if (return_grid)
				local_deriv = static_cast<T>(0.5) * output_derivs[(i + 1) * length2 + j + 1];
			else if (i + 1 == rows && j + 1 == cols)
				local_deriv = static_cast<T>(0.5) * output_derivs[0];
			if (local_deriv != static_cast<T>(0)) {
				for (uint64_t k = 0; k < size; ++k) {
					horizontal[k] += local_deriv;
					right_derivs[k] += local_deriv;
				}
			}

			const T* const tile_state = state + 2 * (i * cols + j) * size;
			gram_derivs[i * cols + j] = TileBackprop(
				gram[i * cols + j], tile_state, tile_state + size,
				horizontal, right_derivs, bottom_derivs, left_derivs,
				rho_powers, tables);
			std::copy_n(bottom_derivs, size, horizontal);
			std::copy_n(left_derivs, size, right_derivs);
		}
	}
}

template<std::floating_point T>
void sig_kernel_poly_backprop_pair_dispatch_(
	const T* RESTRICT gram,
	T* RESTRICT gram_derivs,
	const T* RESTRICT output_derivs,
	const T* RESTRICT state,
	uint64_t length1,
	uint64_t length2,
	bool return_grid,
	const sig_poly_tables<T>& tables
) {
	switch (tables.size) {
	case 3: polynomial_sig_kernel_backprop_pair_<T, 3, sig_poly_tile_backprop_<T, 3>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 4: polynomial_sig_kernel_backprop_pair_<T, 4, sig_poly_tile_backprop_<T, 4>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 5: polynomial_sig_kernel_backprop_pair_<T, 5, sig_poly_tile_backprop_<T, 5>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 6: polynomial_sig_kernel_backprop_pair_<T, 6, sig_poly_tile_backprop_<T, 6>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 7: polynomial_sig_kernel_backprop_pair_<T, 7, sig_poly_tile_backprop_<T, 7>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 8: polynomial_sig_kernel_backprop_pair_<T, 8, sig_poly_tile_backprop_<T, 8>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 9: polynomial_sig_kernel_backprop_pair_<T, 9, sig_poly_tile_backprop_<T, 9>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 10: polynomial_sig_kernel_backprop_pair_<T, 10, sig_poly_tile_backprop_<T, 10>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 11: polynomial_sig_kernel_backprop_pair_<T, 11, sig_poly_tile_backprop_<T, 11>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 12: polynomial_sig_kernel_backprop_pair_<T, 12, sig_poly_tile_backprop_<T, 12>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	case 13: polynomial_sig_kernel_backprop_pair_<T, 13, sig_poly_tile_backprop_<T, 13>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables); return;
	default: polynomial_sig_kernel_backprop_pair_<T, 0, sig_poly_tile_backprop_<T>>(gram, gram_derivs, output_derivs, state, length1, length2, return_grid, tables);
	}
}

template<std::floating_point T, typename Tables, auto PairUpdate>
void polynomial_sig_kernel_(
	const T* gram,
	T* out,
	T* state,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t order,
	bool return_grid,
	int n_jobs,
	const Tables& tables
) {
	if (batch_size == 0)
		return;
	if (!out && batch_size != 0)
		throw std::invalid_argument("signature kernel output pointer must not be null");
	if (length1 == 1 || length2 == 1) {
		const uint64_t out_size = return_grid ? length1 * length2 : 1;
		std::fill(out, out + batch_size * out_size, static_cast<T>(1));
		return;
	}
	if (!gram)
		throw std::invalid_argument("signature kernel gram pointer must not be null");

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const uint64_t out_length = return_grid ? length1 * length2 : 1;
	if (state) {
		const uint64_t state_length = 2 * gram_length * tables.size;
		auto kernel_func = [&](const T* const gram_ptr, T* const out_ptr, T* const state_ptr) {
			PairUpdate(gram_ptr, out_ptr, state_ptr, length1, length2, return_grid, tables);
		};
		multi_threaded_batch(kernel_func, batch_size, n_jobs,
			make_batch(gram, gram_length), make_batch(out, out_length), make_batch(state, state_length));
	}
	else {
		auto kernel_func = [&](const T* const gram_ptr, T* const out_ptr) {
			PairUpdate(gram_ptr, out_ptr, nullptr, length1, length2, return_grid, tables);
		};
		multi_threaded_batch(kernel_func, batch_size, n_jobs,
			make_batch(gram, gram_length), make_batch(out, out_length));
	}
}

template<std::floating_point T>
void sig_kernel_poly_(
	const T* gram,
	T* out,
	T* state,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t order,
	bool return_grid,
	int n_jobs
) {
	if (dimension == 0)
		throw std::invalid_argument("signature kernel received path of dimension 0");
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("signature kernel paths must have length >= 1");
	if (order < 2 || order > 64)
		throw std::invalid_argument("signature kernel polynomial order must be between 2 and 64");
	const auto tables = get_sig_poly_tables_<T>(order);
	polynomial_sig_kernel_<T, sig_poly_tables<T>, sig_kernel_poly_pair_dispatch_<T>>(
		gram, out, state, batch_size, dimension, length1, length2, order,
		return_grid, n_jobs, *tables);
}

template<std::floating_point T>
void sig_kernel_poly_backprop_(
	const T* gram,
	T* gram_derivs,
	const T* output_derivs,
	const T* state,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t order,
	bool return_grid,
	int n_jobs
) {
	if (dimension == 0)
		throw std::invalid_argument("signature kernel received path of dimension 0");
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("signature kernel paths must have length >= 1");
	if (order < 2 || order > 64)
		throw std::invalid_argument("signature kernel polynomial order must be between 2 and 64");
	if (batch_size == 0 || length1 == 1 || length2 == 1)
		return;
	if (!gram)
		throw std::invalid_argument("signature kernel gram pointer must not be null");
	if (!gram_derivs)
		throw std::invalid_argument("signature kernel gram derivative pointer must not be null");
	if (!output_derivs)
		throw std::invalid_argument("signature kernel output derivative pointer must not be null");

	const auto tables = get_sig_poly_tables_<T>(order);
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const uint64_t output_length = return_grid ? length1 * length2 : 1;
	if (state) {
		const uint64_t state_length = 2 * gram_length * tables->size;
		auto kernel_func = [&](const T* const gram_ptr, T* const gram_derivs_ptr,
			const T* const output_derivs_ptr, const T* const state_ptr) {
			sig_kernel_poly_backprop_pair_dispatch_<T>(
				gram_ptr, gram_derivs_ptr, output_derivs_ptr, state_ptr,
				length1, length2, return_grid, *tables);
		};
		multi_threaded_batch(kernel_func, batch_size, n_jobs,
			make_batch(gram, gram_length), make_batch(gram_derivs, gram_length),
			make_batch(output_derivs, output_length), make_batch(state, state_length));
	}
	else {
		auto kernel_func = [&](const T* const gram_ptr, T* const gram_derivs_ptr,
			const T* const output_derivs_ptr) {
			sig_kernel_poly_backprop_pair_dispatch_<T>(
				gram_ptr, gram_derivs_ptr, output_derivs_ptr, nullptr,
				length1, length2, return_grid, *tables);
		};
		multi_threaded_batch(kernel_func, batch_size, n_jobs,
			make_batch(gram, gram_length), make_batch(gram_derivs, gram_length),
			make_batch(output_derivs, output_length));
	}
}
