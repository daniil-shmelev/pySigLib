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
#include "multithreading.h"
#include "macros.h"

static void validate_sig_kernel_dims_(
	uint64_t length1, uint64_t length2,
	uint64_t dyadic_order_1, uint64_t dyadic_order_2
) {
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("sig_kernel: paths must have length >= 1");
	if (dyadic_order_1 >= 63 || dyadic_order_2 >= 63 || dyadic_order_1 + dyadic_order_2 >= 63)
		throw std::invalid_argument("sig_kernel: dyadic_order too large");
	if ((length1 - 1) > (UINT64_MAX >> dyadic_order_1) ||
	    (length2 - 1) > (UINT64_MAX >> dyadic_order_2))
		throw std::invalid_argument("sig_kernel: path length * dyadic refinement would overflow");
}

template<std::floating_point T>
struct BranchedSigKernelWorkspace_ {
	std::vector<T> prev;
	std::vector<T> curr;
	std::vector<T> prefix;
	std::vector<T> d_curr;
	std::vector<T> d_prev;
	std::vector<T> d_inc;
	std::vector<T> stack;

	void ensure_grid(uint64_t grid_length, uint64_t dl2) {
		if (prev.size() < grid_length) prev.resize(grid_length);
		if (curr.size() < grid_length) curr.resize(grid_length);
		if (prefix.size() < dl2) prefix.resize(dl2);
	}

	void ensure_reverse(uint64_t grid_length) {
		if (d_curr.size() < grid_length) d_curr.resize(grid_length);
		if (d_prev.size() < grid_length) d_prev.resize(grid_length);
		if (d_inc.size() < grid_length) d_inc.resize(grid_length);
	}

	void ensure_stack(uint64_t levels, uint64_t grid_length) {
		const uint64_t size = levels * grid_length;
		if (stack.size() < size) stack.resize(size);
	}
};

template<std::floating_point T>
T branched_sig_kernel_depth_one_scalar_(const T* gram, uint64_t gram_length) {
	T total = static_cast<T>(0.);
	for (uint64_t i = 0; i < gram_length; ++i) 
		total += gram[i];
	return std::exp(total);
}

template<std::floating_point T>
void branched_sig_kernel_depth_step_(
	const T* RESTRICT gram,
	const T* RESTRICT prev,
	T* RESTRICT curr,
	T* RESTRICT prefix,
	uint64_t length1,
	uint64_t length2,
	uint64_t dl1,
	uint64_t dl2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const uint64_t gram_stride = length2 - 1;
	const T quarter_scale = static_cast<T>(0.25) / static_cast<T>(1ULL << (dyadic_order_1 + dyadic_order_2));

	std::fill(prefix, prefix + dl2, static_cast<T>(0.));
	std::fill(curr, curr + dl2, static_cast<T>(1.));
	for (uint64_t row = 1; row < dl1; ++row) {
		curr[row * dl2] = static_cast<T>(1.);
	}

	if (dyadic_order_1 == 0 && dyadic_order_2 == 0) {
		for (uint64_t row = 1; row < dl1; ++row) {
			T row_sum = static_cast<T>(0.);
			const uint64_t row_base = row * dl2;
			const uint64_t prev_row_base = row_base - dl2;
			const T* gram_row = gram + (row - 1) * gram_stride;
			for (uint64_t col = 1; col < dl2; ++col) {
				const uint64_t idx = row_base + col;
				const T cell = gram_row[col - 1] * quarter_scale * (
					prev[prev_row_base + col - 1] + prev[row_base + col - 1] +
					prev[prev_row_base + col] + prev[idx]);
				row_sum += cell;
				const T val = prefix[col] + row_sum;
				prefix[col] = val;
				curr[idx] = std::exp(val);
			}
		}
		return;
	}

	for (uint64_t row = 1; row < dl1; ++row) {
		T row_sum = static_cast<T>(0.);
		const uint64_t row_base = row * dl2;
		const uint64_t prev_row_base = row_base - dl2;
		const T* gram_row = gram + ((row - 1) >> dyadic_order_1) * gram_stride;
		for (uint64_t col = 1; col < dl2; ++col) {
			const uint64_t idx = row_base + col;
			const T cell = gram_row[(col - 1) >> dyadic_order_2] * quarter_scale * (
				prev[prev_row_base + col - 1] + prev[row_base + col - 1] +
				prev[prev_row_base + col] + prev[idx]);
			row_sum += cell;
			const T val = prefix[col] + row_sum;
			prefix[col] = val;
			curr[idx] = std::exp(val);
		}
	}
}

template<std::floating_point T>
T branched_sig_kernel_final_integral_(
	const T* RESTRICT gram,
	const T* RESTRICT prev,
	uint64_t length1,
	uint64_t length2,
	uint64_t dl2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const uint64_t gram_stride = length2 - 1;
	const T quarter_scale = static_cast<T>(0.25) / static_cast<T>(1ULL << (dyadic_order_1 + dyadic_order_2));
	T total = static_cast<T>(0.);

	if (dyadic_order_1 == 0 && dyadic_order_2 == 0) {
		for (uint64_t row = 1; row < length1; ++row) {
			const uint64_t row_base = row * dl2;
			const uint64_t prev_row_base = row_base - dl2;
			const T* gram_row = gram + (row - 1) * gram_stride;
			for (uint64_t col = 1; col < length2; ++col) {
				total += gram_row[col - 1] * quarter_scale * (
					prev[prev_row_base + col - 1] + prev[row_base + col - 1] +
					prev[prev_row_base + col] + prev[row_base + col]);
			}
		}
		return total;
	}

	for (uint64_t row = 1; row < ((length1 - 1) << dyadic_order_1) + 1; ++row) {
		const uint64_t row_base = row * dl2;
		const uint64_t prev_row_base = row_base - dl2;
		const T* gram_row = gram + ((row - 1) >> dyadic_order_1) * gram_stride;
		for (uint64_t col = 1; col < ((length2 - 1) << dyadic_order_2) + 1; ++col) {
			total += gram_row[(col - 1) >> dyadic_order_2] * quarter_scale * (
				prev[prev_row_base + col - 1] + prev[row_base + col - 1] +
				prev[prev_row_base + col] + prev[row_base + col]);
		}
	}

	return total;
}

template<std::floating_point T>
void get_branched_sig_kernel_grid_(
	const T* gram,
	uint64_t length1,
	uint64_t length2,
	uint64_t depth,
	T* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	BranchedSigKernelWorkspace_<T>& ws
) {
	const uint64_t dl1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dl2 = ((length2 - 1) << dyadic_order_2) + 1;
	const uint64_t grid_length = dl1 * dl2;

	if (depth == 0 || dl1 == 1 || dl2 == 1) {
		std::fill(out, out + grid_length, static_cast<T>(1.));
		return;
	}

	ws.ensure_grid(grid_length, dl2);
	T* buf_a = ws.prev.data();
	T* buf_b = ws.curr.data();
	T* prev = buf_a;
	std::fill(prev, prev + grid_length, static_cast<T>(1.));

	for (uint64_t m = 1; m <= depth; ++m) {
		T* curr = m == depth ? out : (prev == buf_a ? buf_b : buf_a);
		branched_sig_kernel_depth_step_(
			gram, prev, curr, ws.prefix.data(), length1, length2,
			dl1, dl2, dyadic_order_1, dyadic_order_2);
		prev = curr;
	}
}

template<std::floating_point T>
void get_branched_sig_kernel_scalar_(
	const T* gram,
	uint64_t length1,
	uint64_t length2,
	uint64_t depth,
	T* out,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	BranchedSigKernelWorkspace_<T>& ws
) {
	const uint64_t dl1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dl2 = ((length2 - 1) << dyadic_order_2) + 1;
	const uint64_t grid_length = dl1 * dl2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);

	if (depth == 0 || dl1 == 1 || dl2 == 1) {
		*out = static_cast<T>(1.);
		return;
	}
	if (depth == 1) {
		*out = branched_sig_kernel_depth_one_scalar_(gram, gram_length);
		return;
	}

	ws.ensure_grid(grid_length, dl2);
	T* buf_a = ws.prev.data();
	T* buf_b = ws.curr.data();
	T* prev = buf_a;
	std::fill(prev, prev + grid_length, static_cast<T>(1.));

	for (uint64_t m = 1; m < depth; ++m) {
		T* curr = prev == buf_a ? buf_b : buf_a;
		branched_sig_kernel_depth_step_(
			gram, prev, curr, ws.prefix.data(), length1, length2,
			dl1, dl2, dyadic_order_1, dyadic_order_2);
		prev = curr;
	}

	const T total = branched_sig_kernel_final_integral_(
		gram, prev, length1, length2, dl2,
		dyadic_order_1, dyadic_order_2);
	*out = std::exp(total);
}

template<std::floating_point T>
void branched_sig_kernel_(
	const T* gram,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t depth,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid,
	int n_jobs
) {
	if (dimension == 0) { throw std::invalid_argument("branched signature kernel received path of dimension 0"); }
	validate_sig_kernel_dims_(length1, length2, dyadic_order_1, dyadic_order_2);

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const uint64_t grid_length = (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1);
	const uint64_t result_length = return_grid ? grid_length : 1;

	if (!gram || depth == 0 || gram_length == 0) {
		std::fill(out, out + batch_size * result_length, static_cast<T>(1.));
		return;
	}

	auto kernel_func = [&](const T* const gram_ptr, T* const out_ptr, BranchedSigKernelWorkspace_<T>& ws) {
		if (return_grid) {
			get_branched_sig_kernel_grid_(gram_ptr, length1, length2, depth, out_ptr, dyadic_order_1, dyadic_order_2, ws);
		}
		else {
			get_branched_sig_kernel_scalar_(gram_ptr, length1, length2, depth, out_ptr, dyadic_order_1, dyadic_order_2, ws);
		}
	};

	if (n_jobs == 1 || batch_size == 1) {
		BranchedSigKernelWorkspace_<T> ws;
		const T* gram_ptr = gram;
		T* out_ptr = out;
		for (uint64_t i = 0; i < batch_size; ++i, gram_ptr += gram_length, out_ptr += result_length) {
			kernel_func(gram_ptr, out_ptr, ws);
		}
		return;
	}

	spawn_batch_threads(batch_size, n_jobs, [&](uint64_t start, uint64_t end) {
		BranchedSigKernelWorkspace_<T> ws;
		const T* gram_ptr = gram + start * gram_length;
		T* out_ptr = out + start * result_length;
		for (uint64_t i = start; i < end; ++i, gram_ptr += gram_length, out_ptr += result_length) {
			kernel_func(gram_ptr, out_ptr, ws);
		}
	});
}

template<std::floating_point T>
void build_branched_sig_kernel_stack_(
	const T* gram,
	T* stack,
	uint64_t length1,
	uint64_t length2,
	uint64_t depth,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const uint64_t dl1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dl2 = ((length2 - 1) << dyadic_order_2) + 1;
	const uint64_t grid_length = dl1 * dl2;
	BranchedSigKernelWorkspace_<T> local_ws;
	local_ws.ensure_grid(grid_length, dl2);

	std::fill(stack, stack + grid_length, static_cast<T>(1.));
	if (depth == 0 || dl1 == 1 || dl2 == 1) {
		for (uint64_t m = 1; m <= depth; ++m) {
			std::fill(stack + m * grid_length, stack + (m + 1) * grid_length, static_cast<T>(1.));
		}
		return;
	}

	for (uint64_t m = 1; m <= depth; ++m) {
		const T* prev = stack + (m - 1) * grid_length;
		T* curr = stack + m * grid_length;
		branched_sig_kernel_depth_step_(
			gram, prev, curr, local_ws.prefix.data(), length1, length2,
			dl1, dl2, dyadic_order_1, dyadic_order_2);
	}
}

template<std::floating_point T>
void branched_sig_kernel_seed_scalar_reverse_(
	const T* RESTRICT gram,
	const T* RESTRICT prev,
	T* RESTRICT out,
	T* RESTRICT d_prev,
	const T deriv,
	uint64_t length1,
	uint64_t length2,
	uint64_t dl2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
) {
	const uint64_t gram_stride = length2 - 1;
	const T quarter_scale = static_cast<T>(0.25) / static_cast<T>(1ULL << (dyadic_order_1 + dyadic_order_2));

	if (dyadic_order_1 == 0 && dyadic_order_2 == 0) {
		for (uint64_t row = 1; row < length1; ++row) {
			const uint64_t row_base = row * dl2;
			const uint64_t prev_row_base = row_base - dl2;
			const T* gram_row = gram + (row - 1) * gram_stride;
			for (uint64_t col = 1; col < length2; ++col) {
				const uint64_t idx = row_base + col;
				const uint64_t gi = (row - 1) * gram_stride + (col - 1);
				const T weight = deriv * quarter_scale;
				const T gram_weight = weight * gram_row[col - 1];
				const uint64_t i00 = prev_row_base + col - 1;
				const uint64_t i10 = row_base + col - 1;
				const uint64_t i01 = prev_row_base + col;
				out[gi] += weight * (prev[i00] + prev[i10] + prev[i01] + prev[idx]);
				d_prev[i00] += gram_weight;
				d_prev[i10] += gram_weight;
				d_prev[i01] += gram_weight;
				d_prev[idx] += gram_weight;
			}
		}
		return;
	}

	for (uint64_t row = 1; row < ((length1 - 1) << dyadic_order_1) + 1; ++row) {
		const uint64_t row_base = row * dl2;
		const uint64_t prev_row_base = row_base - dl2;
		const uint64_t gi_row = ((row - 1) >> dyadic_order_1) * gram_stride;
		const T* gram_row = gram + gi_row;
		for (uint64_t col = 1; col < ((length2 - 1) << dyadic_order_2) + 1; ++col) {
			const uint64_t gi = gi_row + ((col - 1) >> dyadic_order_2);
			const uint64_t idx = row_base + col;
			const uint64_t i00 = prev_row_base + col - 1;
			const uint64_t i10 = row_base + col - 1;
			const uint64_t i01 = prev_row_base + col;
			const T weight = deriv * quarter_scale;
			const T gram_weight = weight * gram_row[(col - 1) >> dyadic_order_2];
			out[gi] += weight * (prev[i00] + prev[i10] + prev[i01] + prev[idx]);
			d_prev[i00] += gram_weight;
			d_prev[i10] += gram_weight;
			d_prev[i01] += gram_weight;
			d_prev[idx] += gram_weight;
		}
	}
}

template<std::floating_point T>
void get_branched_sig_kernel_backprop_(
	const T* gram,
	T* RESTRICT out,
	const T* derivs,
	const T* RESTRICT k_stack,
	uint64_t length1,
	uint64_t length2,
	uint64_t depth,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid,
	BranchedSigKernelWorkspace_<T>& ws
) {
	const uint64_t dl1 = ((length1 - 1) << dyadic_order_1) + 1;
	const uint64_t dl2 = ((length2 - 1) << dyadic_order_2) + 1;
	const uint64_t grid_length = dl1 * dl2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const uint64_t gram_stride = length2 - 1;
	const T quarter_scale = static_cast<T>(0.25) / static_cast<T>(1ULL << (dyadic_order_1 + dyadic_order_2));

	auto at = [dl2](uint64_t row, uint64_t col) -> uint64_t { return row * dl2 + col; };
	auto gram_at = [&](uint64_t row, uint64_t col) -> uint64_t {
		return ((row - 1) >> dyadic_order_1) * gram_stride + ((col - 1) >> dyadic_order_2);
	};

	std::fill(out, out + gram_length, static_cast<T>(0.));
	if (depth == 0 || gram_length == 0 || dl1 == 1 || dl2 == 1) {
		return;
	}

	if (!return_grid && depth == 1) {
		const T k = branched_sig_kernel_depth_one_scalar_(gram, gram_length);
		std::fill(out, out + gram_length, (*derivs) * k);
		return;
	}

	const T* stack = k_stack;
	if (!stack) {
		const uint64_t build_depth = return_grid ? depth : depth - 1;
		ws.ensure_stack(build_depth + 1, grid_length);
		build_branched_sig_kernel_stack_(gram, ws.stack.data(), length1, length2, build_depth, dyadic_order_1, dyadic_order_2);
		stack = ws.stack.data();
	}

	ws.ensure_reverse(grid_length);
	T* d_curr = ws.d_curr.data();
	T* d_prev = ws.d_prev.data();
	T* d_inc = ws.d_inc.data();
	uint64_t start_depth = depth;

	if (return_grid) {
		std::copy(derivs, derivs + grid_length, d_curr);
	}
	else {
		const T* prev = stack + (depth - 1) * grid_length;
		const T total = branched_sig_kernel_final_integral_(
			gram, prev, length1, length2, dl2,
			dyadic_order_1, dyadic_order_2);
		const T scalar_deriv = (*derivs) * std::exp(total);
		std::fill(d_curr, d_curr + grid_length, static_cast<T>(0.));
		branched_sig_kernel_seed_scalar_reverse_(
			gram, prev, out, d_curr, scalar_deriv, length1, length2,
			dl2, dyadic_order_1, dyadic_order_2);
		start_depth = depth - 1;
	}

	for (uint64_t m = start_depth; m > 0; --m) {
		const T* prev = stack + (m - 1) * grid_length;
		const T* curr = stack + m * grid_length;

		std::fill(d_prev, d_prev + grid_length, static_cast<T>(0.));
		std::fill(d_inc, d_inc + grid_length, static_cast<T>(0.));

		for (uint64_t row = 1; row < dl1; ++row) {
			for (uint64_t col = 1; col < dl2; ++col) {
				const uint64_t idx = at(row, col);
				d_inc[idx] = d_curr[idx] * curr[idx];
			}
		}

		for (uint64_t row = dl1 - 1; row >= 1; --row) {
			for (uint64_t col = dl2 - 1; col >= 1; --col) {
				const uint64_t idx = at(row, col);
				T val = d_inc[idx];
				if (row + 1 < dl1) val += d_inc[at(row + 1, col)];
				if (col + 1 < dl2) val += d_inc[at(row, col + 1)];
				if (row + 1 < dl1 && col + 1 < dl2) val -= d_inc[at(row + 1, col + 1)];
				d_inc[idx] = val;
			}
		}

		for (uint64_t row = 1; row < dl1; ++row) {
			for (uint64_t col = 1; col < dl2; ++col) {
				const uint64_t idx = at(row, col);
				const uint64_t gi = gram_at(row, col);
				const T weight = d_inc[idx] * quarter_scale;
				const T gram_weight = weight * gram[gi];
				const uint64_t i00 = at(row - 1, col - 1);
				const uint64_t i10 = at(row, col - 1);
				const uint64_t i01 = at(row - 1, col);
				out[gi] += weight * (prev[i00] + prev[i10] + prev[i01] + prev[idx]);
				d_prev[i00] += gram_weight;
				d_prev[i10] += gram_weight;
				d_prev[i01] += gram_weight;
				d_prev[idx] += gram_weight;
			}
		}

		std::swap(d_curr, d_prev);
	}
}

template<std::floating_point T>
void batch_branched_sig_kernel_backprop_(
	const T* gram,
	T* out,
	const T* derivs,
	const T* k_stack,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t depth,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid,
	int n_jobs
) {
	if (dimension == 0) { throw std::invalid_argument("branched signature kernel received path of dimension 0"); }
	validate_sig_kernel_dims_(length1, length2, dyadic_order_1, dyadic_order_2);

	const uint64_t gram_length = (length1 - 1) * (length2 - 1);
	const uint64_t grid_length = (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1);
	const uint64_t derivs_stride = return_grid ? grid_length : 1;
	const uint64_t stack_stride = (depth + 1) * grid_length;

	if (!gram || depth == 0 || gram_length == 0) {
		std::fill(out, out + batch_size * gram_length, static_cast<T>(0.));
		return;
	}

	if (n_jobs == 1 || batch_size == 1) {
		BranchedSigKernelWorkspace_<T> ws;
		for (uint64_t i = 0; i < batch_size; ++i) {
			get_branched_sig_kernel_backprop_(
				gram + i * gram_length,
				out + i * gram_length,
				derivs + i * derivs_stride,
				k_stack ? k_stack + i * stack_stride : nullptr,
				length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid, ws);
		}
		return;
	}

	spawn_batch_threads(batch_size, n_jobs, [&](uint64_t start, uint64_t end) {
		BranchedSigKernelWorkspace_<T> ws;
		for (uint64_t i = start; i < end; ++i) {
			get_branched_sig_kernel_backprop_(
				gram + i * gram_length,
				out + i * gram_length,
				derivs + i * derivs_stride,
				k_stack ? k_stack + i * stack_stride : nullptr,
				length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid, ws);
		}
	});
}


extern "C" {

	CPSIG_API int branched_sig_kernel_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(branched_sig_kernel_<float>(gram, out, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}

	CPSIG_API int branched_sig_kernel_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(branched_sig_kernel_<double>(gram, out, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}

	CPSIG_API int branched_sig_kernel_backprop_f(const float* gram, float* out, const float* derivs, const float* k_stack, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(batch_branched_sig_kernel_backprop_<float>(gram, out, derivs, k_stack, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}

	CPSIG_API int branched_sig_kernel_backprop_d(const double* gram, double* out, const double* derivs, const double* k_stack, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(batch_branched_sig_kernel_backprop_<double>(gram, out, derivs, k_stack, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}
}
