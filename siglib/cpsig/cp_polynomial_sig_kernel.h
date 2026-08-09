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


template<std::floating_point T>
struct polysig_tables {
	uint64_t size;
	std::vector<T> mat1;
	std::vector<T> mat2;

	explicit polysig_tables(uint64_t order) :
		size(order + 1), mat1(size * size, static_cast<T>(0)), mat2(size * size, static_cast<T>(0)) {
		long double inverse_factorial = 1.0L;
		for (uint64_t n = 1; n < size; ++n) {
			inverse_factorial /= static_cast<long double>(n);
			long double mat2_value = inverse_factorial * inverse_factorial;
			mat2[n * size] = static_cast<T>(mat2_value);
			for (uint64_t k = 1; k < size; ++k) {
				mat2_value *= static_cast<long double>(k) / static_cast<long double>(n + k);
				mat2[n * size + k] = static_cast<T>(mat2_value);
			}
		}

		long double inverse_nm1_factorial = 1.0L;
		for (uint64_t n = 2; n < size; ++n) {
			const long double inverse_n_factorial = inverse_nm1_factorial / static_cast<long double>(n);
			long double mat1_value = inverse_n_factorial * inverse_nm1_factorial;
			mat1[n * size + 1] = static_cast<T>(mat1_value);
			for (uint64_t k = 2; k < n; ++k) {
				mat1_value *= static_cast<long double>(k * (n - k + 1));
				mat1[n * size + k] = static_cast<T>(mat1_value);
			}
			inverse_nm1_factorial = inverse_n_factorial;
		}
	}
};

template<std::floating_point T>
inline thread_local std::vector<T> polynomial_sig_kernel_workspace_;

template<std::floating_point T, uint64_t FixedSize = 0>
FORCE_INLINE void polysig_tile_update_(
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
	const polysig_tables<T>& tables
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
	uint64_t length1,
	uint64_t length2,
	const Tables& tables
) {
	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	if (rows == 0 || cols == 0) {
		*out = static_cast<T>(1);
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
			TileUpdate(
				gram_row[j], bottom, left, bottom_endpoint, frontier_endpoints[j],
				top, right, rho_powers, top_endpoint, right_endpoint, tables
			);
			for (uint64_t k = 0; k < size; ++k)
				left[k] = right[k];
			frontier_endpoints[j] = right_endpoint;
			std::swap(bottom, top);
			bottom_endpoint = top_endpoint;
		}
	}

	*out = static_cast<T>(0.5) * (top_endpoint + right_endpoint);
}

template<std::floating_point T>
void polysig_kernel_pair_dispatch_(
	const T* RESTRICT gram,
	T* RESTRICT out,
	uint64_t length1,
	uint64_t length2,
	const polysig_tables<T>& tables
) {
	switch (tables.size) {
	case 3: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 3, polysig_tile_update_<T, 3>>(gram, out, length1, length2, tables); return;
	case 4: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 4, polysig_tile_update_<T, 4>>(gram, out, length1, length2, tables); return;
	case 5: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 5, polysig_tile_update_<T, 5>>(gram, out, length1, length2, tables); return;
	case 6: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 6, polysig_tile_update_<T, 6>>(gram, out, length1, length2, tables); return;
	case 7: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 7, polysig_tile_update_<T, 7>>(gram, out, length1, length2, tables); return;
	case 8: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 8, polysig_tile_update_<T, 8>>(gram, out, length1, length2, tables); return;
	case 9: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 9, polysig_tile_update_<T, 9>>(gram, out, length1, length2, tables); return;
	case 10: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 10, polysig_tile_update_<T, 10>>(gram, out, length1, length2, tables); return;
	case 11: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 11, polysig_tile_update_<T, 11>>(gram, out, length1, length2, tables); return;
	case 12: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 12, polysig_tile_update_<T, 12>>(gram, out, length1, length2, tables); return;
	case 13: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 13, polysig_tile_update_<T, 13>>(gram, out, length1, length2, tables); return;
	default: polynomial_sig_kernel_pair_<T, polysig_tables<T>, 0, polysig_tile_update_<T>>(gram, out, length1, length2, tables);
	}
}

template<std::floating_point T, typename Tables, auto PairUpdate>
void polynomial_sig_kernel_(
	const T* gram,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t order,
	int n_jobs,
	const Tables& tables
) {
	if (batch_size == 0)
		return;
	if (!out && batch_size != 0)
		throw std::invalid_argument("signature kernel output pointer must not be null");
	if (length1 == 1 || length2 == 1) {
		std::fill(out, out + batch_size, static_cast<T>(1));
		return;
	}
	if (!gram)
		throw std::invalid_argument("signature kernel gram pointer must not be null");

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	auto kernel_func = [&](const T* const gram_ptr, T* const out_ptr) {
		PairUpdate(gram_ptr, out_ptr, length1, length2, tables);
	};

	multi_threaded_batch(kernel_func, batch_size, n_jobs,
		make_batch(gram, gram_length), make_batch(out, static_cast<uint64_t>(1)));
}

template<std::floating_point T>
void polysig_kernel_(
	const T* gram,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t order,
	int n_jobs
) {
	if (dimension == 0)
		throw std::invalid_argument("signature kernel received path of dimension 0");
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("signature kernel paths must have length >= 1");
	if (order < 2 || order > 64)
		throw std::invalid_argument("signature kernel polynomial order must be between 2 and 64");
	const polysig_tables<T> tables(order);
	polynomial_sig_kernel_<T, polysig_tables<T>, polysig_kernel_pair_dispatch_<T>>(
		gram, out, batch_size, dimension, length1, length2, order, n_jobs, tables);
}
