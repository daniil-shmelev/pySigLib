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

#include <cstdint>

namespace log_pde_cuda_detail {

constexpr unsigned int PATH_THREADS = 32;

struct Params {
	const uint64_t* offset;
	uint64_t batch_size;
	uint64_t dimension;
	uint64_t degree_x;
	uint64_t degree_y;
	uint64_t degree_f;
	uint64_t degree_g;
	uint64_t steps_x;
	uint64_t steps_y;
	uint64_t factor_x;
	uint64_t factor_y;
	uint64_t shift_x;
	uint64_t shift_y;
	double scale_x;
	double scale_y;
	uint64_t nodes_x;
	uint64_t nodes_y;
	uint64_t x_len;
	uint64_t y_len;
	uint64_t f_len;
	uint64_t g_len;
	uint64_t state_len;
	uint64_t cell_cache_len;
	uint64_t grid_len;
	uint64_t path_threads;
	uint64_t paths_per_block;
};

template<typename T>
struct ConstView {
	const T* data;
	uint64_t stride;
	T scale;

	__device__ T operator[](uint64_t index) const {
		return data[index * stride] * scale;
	}

	__device__ ConstView sub(uint64_t index) const {
		return { data + index * stride, stride, scale };
	}
};

template<typename T>
struct View {
	T* data;
	uint64_t stride;

	__device__ T& operator[](uint64_t index) const {
		return data[index * stride];
	}

	__device__ View sub(uint64_t index) const {
		return { data + index * stride, stride };
	}

	__device__ ConstView<T> read(T scale = static_cast<T>(1)) const {
		return { data, stride, scale };
	}
};

template<typename T>
__device__ void tensor_product_add(
	const Params& p, ConstView<T> a, uint64_t a_degree,
	ConstView<T> b, uint64_t b_degree, View<T> out, uint64_t out_degree,
	uint64_t min_out_level = 2, uint64_t min_b_level = 1,
	T scale = static_cast<T>(1)
) {
	for (uint64_t level = min_out_level; level <= out_degree; ++level) {
		const uint64_t first = level > b_degree ? level - b_degree : 1;
		const uint64_t last_bound = level > min_b_level ? level - min_b_level : 0;
		const uint64_t last = a_degree < last_bound ? a_degree : last_bound;
		for (uint64_t left = first; left <= last; ++left) {
			const uint64_t right = level - left;
			const uint64_t left_size = p.offset[left + 1] - p.offset[left];
			const uint64_t right_size = p.offset[right + 1] - p.offset[right];
			const uint64_t out_start = p.offset[level];
			const uint64_t a_start = p.offset[left];
			const uint64_t b_start = p.offset[right];
			for (uint64_t i = 0; i < left_size; ++i) {
				const T value = a[a_start + i] * scale;
				for (uint64_t j = 0; j < right_size; ++j)
					out[out_start + i * right_size + j] += value * b[b_start + j];
			}
		}
	}
}

template<typename T>
__device__ void tensor_product_backward(
	const Params& p, ConstView<T> a, uint64_t a_degree,
	ConstView<T> b, uint64_t b_degree, ConstView<T> d_out, uint64_t out_degree,
	View<T> d_a, View<T> d_b, uint64_t min_out_level = 2,
	uint64_t min_b_level = 1, T scale = static_cast<T>(1)
) {
	for (uint64_t level = min_out_level; level <= out_degree; ++level) {
		const uint64_t first = level > b_degree ? level - b_degree : 1;
		const uint64_t last_bound = level > min_b_level ? level - min_b_level : 0;
		const uint64_t last = a_degree < last_bound ? a_degree : last_bound;
		for (uint64_t left = first; left <= last; ++left) {
			const uint64_t right = level - left;
			const uint64_t left_size = p.offset[left + 1] - p.offset[left];
			const uint64_t right_size = p.offset[right + 1] - p.offset[right];
			const uint64_t out_start = p.offset[level];
			const uint64_t a_start = p.offset[left];
			const uint64_t b_start = p.offset[right];
			for (uint64_t i = 0; i < left_size; ++i) {
				T da = static_cast<T>(0);
				for (uint64_t j = 0; j < right_size; ++j) {
					const T grad = d_out[out_start + i * right_size + j] * scale;
					da += grad * b[b_start + j];
					d_b[b_start + j] += grad * a[a_start + i];
				}
				d_a[a_start + i] += da;
			}
		}
	}
}

template<typename T>
__device__ void sig_combine(
	const Params& p, ConstView<T> a, ConstView<T> b, View<T> out, uint64_t degree
) {
	const uint64_t length = degree == 0 ? 0 : p.offset[degree + 1];
	for (uint64_t i = 0; i < length; ++i) out[i] = a[i] + b[i];
	tensor_product_add(p, a, degree, b, degree, out, degree);
}

template<typename T>
__device__ void sig_combine_backward(
	const Params& p, ConstView<T> a, ConstView<T> b, ConstView<T> d_out,
	View<T> d_a, View<T> d_b, uint64_t degree
) {
	const uint64_t length = degree == 0 ? 0 : p.offset[degree + 1];
	for (uint64_t i = 0; i < length; ++i) {
		d_a[i] = d_out[i];
		d_b[i] = d_out[i];
	}
	tensor_product_backward(p, a, degree, b, degree, d_out, degree, d_a, d_b);
}

template<typename T>
__device__ void tensor_exp(
	const Params& p, ConstView<T> log_sig, View<T> out,
	uint64_t log_degree, uint64_t out_degree, View<T> powers, View<T> next
) {
	if (out_degree == 0) return;
	const uint64_t length = p.offset[out_degree + 1];
	const uint64_t copy_degree = log_degree < out_degree ? log_degree : out_degree;
	const uint64_t copy_end = copy_degree == 0 ? 0 : p.offset[copy_degree + 1];
	for (uint64_t i = 0; i < length; ++i) {
		powers[i] = static_cast<T>(0);
		next[i] = static_cast<T>(0);
		out[i] = static_cast<T>(0);
	}
	for (uint64_t i = 0; i < copy_end; ++i) {
		powers[i] = log_sig[i];
		out[i] = powers[i];
	}

	for (uint64_t power = 2; power <= out_degree; ++power) {
		for (uint64_t i = p.offset[power]; i < length; ++i) next[i] = static_cast<T>(0);
		const T inv_power = static_cast<T>(1) / static_cast<T>(power);
		tensor_product_add(p, log_sig, log_degree, powers.read(), out_degree,
			next, out_degree, power, power - 1, inv_power);
		for (uint64_t i = p.offset[power]; i < length; ++i) out[i] += next[i];
		const View<T> temp = powers;
		powers = next;
		next = temp;
	}
}

template<typename T>
__device__ void tensor_exp_backward(
	const Params& p, View<T> d_log, ConstView<T> d_sig, ConstView<T> log_sig,
	uint64_t log_degree, uint64_t out_degree, View<T> powers, View<T> d_powers
) {
	if (out_degree == 0) return;
	const uint64_t length = p.offset[out_degree + 1];
	const uint64_t copy_degree = log_degree < out_degree ? log_degree : out_degree;
	const uint64_t copy_end = copy_degree == 0 ? 0 : p.offset[copy_degree + 1];
	for (uint64_t i = 0; i < out_degree * length; ++i) {
		powers[i] = static_cast<T>(0);
		d_powers[i] = static_cast<T>(0);
	}
	for (uint64_t i = 0; i < copy_end; ++i) powers[i] = log_sig[i];
	for (uint64_t power = 0; power < out_degree; ++power) {
		const View<T> destination = d_powers.sub(power * length);
		for (uint64_t i = 0; i < length; ++i) destination[i] = d_sig[i];
	}

	for (uint64_t power = 2; power <= out_degree; ++power) {
		const T inv_power = static_cast<T>(1) / static_cast<T>(power);
		tensor_product_add(p, log_sig, log_degree,
			powers.sub((power - 2) * length).read(), out_degree,
			powers.sub((power - 1) * length), out_degree,
			power, power - 1, inv_power);
	}

	for (uint64_t power = out_degree; power >= 2; --power) {
		const T inv_power = static_cast<T>(1) / static_cast<T>(power);
		tensor_product_backward(p, log_sig, log_degree,
			powers.sub((power - 2) * length).read(), out_degree,
			d_powers.sub((power - 1) * length).read(), out_degree,
			d_log, d_powers.sub((power - 2) * length),
			power, power - 1, inv_power);
	}
	for (uint64_t i = 0; i < copy_end; ++i) d_log[i] += d_powers[i];
}

template<typename T>
__device__ void tensor_adjoint_add(
	const Params& p, ConstView<T> w, uint64_t w_degree,
	ConstView<T> y, uint64_t y_degree, View<T> out, uint64_t out_degree, bool left
) {
	for (uint64_t level = 1; level <= out_degree; ++level) {
		const uint64_t out_size = p.offset[level + 1] - p.offset[level];
		for (uint64_t wi = 1; wi <= w_degree && level + wi <= y_degree; ++wi) {
			const uint64_t w_size = p.offset[wi + 1] - p.offset[wi];
			for (uint64_t i = 0; i < out_size; ++i) {
				T sum = static_cast<T>(0);
				for (uint64_t j = 0; j < w_size; ++j) {
					const uint64_t index = left ? j * out_size + i : i * w_size + j;
					sum += y[p.offset[level + wi] + index] * w[p.offset[wi] + j];
				}
				out[p.offset[level] + i] += sum;
			}
		}
	}
}

template<typename T>
__device__ void tensor_adjoint_backward(
	const Params& p, ConstView<T> w, uint64_t w_degree,
	ConstView<T> y, uint64_t y_degree, ConstView<T> d_out, uint64_t out_degree,
	View<T> d_w, View<T> d_y, bool left
) {
	for (uint64_t level = 1; level <= out_degree; ++level) {
		const uint64_t out_size = p.offset[level + 1] - p.offset[level];
		for (uint64_t wi = 1; wi <= w_degree && level + wi <= y_degree; ++wi) {
			const uint64_t w_size = p.offset[wi + 1] - p.offset[wi];
			for (uint64_t i = 0; i < out_size; ++i) {
				const T grad = d_out[p.offset[level] + i];
				for (uint64_t j = 0; j < w_size; ++j) {
					const uint64_t index = left ? j * out_size + i : i * w_size + j;
					const uint64_t y_index = p.offset[level + wi] + index;
					d_w[p.offset[wi] + j] += y[y_index] * grad;
					d_y[y_index] += grad * w[p.offset[wi] + j];
				}
			}
		}
	}
}

}  // namespace log_pde_cuda_detail
