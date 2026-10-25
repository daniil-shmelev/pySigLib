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

#include "cusig.h"
#include "cu_utils.h"
#include "../shared/chebyshev_sig_kernel.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr uint64_t chebyshev_tile_width = 32;
constexpr uint64_t chebyshev_forward_warps = 2;
constexpr bool chebyshev_stage_gram = true;

template<typename T>
__global__ void chebyshev_fill_kernel(T* data, uint64_t count, T value) {
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < count; index += static_cast<uint64_t>(blockDim.x) * gridDim.x)
		data[index] = value;
}

template<typename T>
__global__ void chebyshev_init_kernel(T* work, uint64_t count, uint64_t degree) {
	const uint64_t slots = degree + 5;
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < count; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t slot = (index / chebyshev_tile_width) % slots;
		work[index] = slot == degree + 1 || slot == degree + 3
			? static_cast<T>(1) : static_cast<T>(0);
	}
}

template<typename T, int FixedDegree = -1, int FixedOrder = -1>
__global__ void chebyshev_forward_kernel(
	const T* __restrict__ gram,
	T* __restrict__ out,
	T* __restrict__ state,
	T* __restrict__ work,
	uint64_t rows,
	uint64_t cols,
	uint64_t degree_,
	uint64_t neumann_order_,
	uint64_t tile_rows,
	uint64_t tile_cols,
	uint64_t tile_diagonal,
	uint64_t batch_size,
	bool return_grid
) {
	const uint64_t lane = threadIdx.x % chebyshev_tile_width;
	const uint64_t warp = threadIdx.x / chebyshev_tile_width;
	const uint64_t batch = cuda_batch_index() * (blockDim.x / chebyshev_tile_width) + warp;
	if (batch >= batch_size)
		return;
	const uint64_t degree = FixedDegree < 0 ? degree_ : static_cast<uint64_t>(FixedDegree);
	const uint64_t neumann_order = FixedOrder < 0 ? neumann_order_ : static_cast<uint64_t>(FixedOrder);
	constexpr uint64_t capacity = FixedDegree < 0 ? 32 : FixedDegree + 1;
	const uint64_t first_tile_row = tile_diagonal >= tile_cols
		? tile_diagonal - tile_cols + 1 : 0;
	const uint64_t tile_row = first_tile_row + blockIdx.x;
	const uint64_t tile_col = tile_diagonal - tile_row;
	const uint64_t slots = degree + 5;
	const uint64_t tile_stride = slots * chebyshev_tile_width;
	T* const batch_work = work + batch * (tile_rows + tile_cols) * tile_stride;
	T* const horizontal_global = batch_work + tile_col * tile_stride;
	T* const vertical_global = batch_work + (tile_cols + tile_row) * tile_stride;
	extern __shared__ char shared_bytes[];
	T* const bottom = reinterpret_cast<T*>(shared_bytes) + 2 * warp * tile_stride;
	T* const left = bottom + tile_stride;
	for (uint64_t k = 0; k < slots; ++k) {
		bottom[k * chebyshev_tile_width + lane] = horizontal_global[k * chebyshev_tile_width + lane];
		left[k * chebyshev_tile_width + lane] = vertical_global[k * chebyshev_tile_width + lane];
	}
	__syncwarp();

	const uint64_t row_start = tile_row * chebyshev_tile_width;
	const uint64_t col_start = tile_col * chebyshev_tile_width;
	const uint64_t active_rows = min(chebyshev_tile_width, rows - row_start);
	const uint64_t active_cols = min(chebyshev_tile_width, cols - col_start);
	constexpr bool stage_gram = chebyshev_stage_gram && std::is_same_v<T, float>
		&& FixedDegree >= 1 && FixedDegree <= 3 && FixedOrder >= FixedDegree && FixedOrder <= 3;
	T* gram_tile = nullptr;
	if constexpr (stage_gram) {
		gram_tile = reinterpret_cast<T*>(shared_bytes)
			+ 2 * (blockDim.x / chebyshev_tile_width) * tile_stride
			+ warp * chebyshev_tile_width * chebyshev_tile_width;
		for (uint64_t row = 0; row < active_rows; ++row) {
			if (lane < active_cols)
				gram_tile[row * chebyshev_tile_width + lane]
					= gram[(batch * rows + row_start + row) * cols + col_start + lane];
		}
		__syncwarp();
	}
	const uint64_t state_width = 2 * degree + 8;
	for (uint64_t diagonal = 0; diagonal < active_rows + active_cols - 1; ++diagonal) {
		const uint64_t first_row = diagonal >= active_cols ? diagonal - active_cols + 1 : 0;
		const uint64_t last_row = min(diagonal, active_rows - 1);
		if (lane <= last_row - first_row) {
			const uint64_t local_row = first_row + lane;
			const uint64_t local_col = diagonal - local_row;
			const uint64_t global_row = row_start + local_row;
			const uint64_t global_col = col_start + local_col;
			const uint64_t cell = (batch * rows + global_row) * cols + global_col;
			T f[capacity], g[capacity], top[capacity], right[capacity];
			f[0] = g[0] = T(0);
			for (uint64_t k = 1; k <= degree; ++k) {
				f[k] = bottom[k * chebyshev_tile_width + local_col];
				g[k] = left[k * chebyshev_tile_width + local_row];
			}
			const chebyshev_sig_kernel::Corner<T> start{
				bottom[(degree + 1) * chebyshev_tile_width + local_col],
				bottom[(degree + 2) * chebyshev_tile_width + local_col]};
			const chebyshev_sig_kernel::Corner<T> bottom_end{
				bottom[(degree + 3) * chebyshev_tile_width + local_col],
				bottom[(degree + 4) * chebyshev_tile_width + local_col]};
			const chebyshev_sig_kernel::Corner<T> left_end{
				left[(degree + 3) * chebyshev_tile_width + local_row],
				left[(degree + 4) * chebyshev_tile_width + local_row]};
			if (state) {
				T* const saved = state + cell * state_width;
				for (uint64_t k = 0; k <= degree; ++k) {
					saved[k] = f[k];
					saved[degree + 1 + k] = g[k];
				}
				const uint64_t offset = 2 * (degree + 1);
				saved[offset] = start.hi;
				saved[offset + 1] = start.lo;
				saved[offset + 2] = bottom_end.hi;
				saved[offset + 3] = bottom_end.lo;
				saved[offset + 4] = left_end.hi;
				saved[offset + 5] = left_end.lo;
			}
			const T coefficient = stage_gram
				? gram_tile[local_row * chebyshev_tile_width + local_col] : gram[cell];
			const auto corner = chebyshev_sig_kernel::cell_forward<T, FixedDegree, FixedOrder, true>(
				coefficient, f, g, start, bottom_end, left_end,
				top, right, degree, neumann_order);
			for (uint64_t k = 1; k <= degree; ++k) {
				bottom[k * chebyshev_tile_width + local_col] = top[k];
				left[k * chebyshev_tile_width + local_row] = right[k];
			}
			bottom[(degree + 1) * chebyshev_tile_width + local_col] = left_end.hi;
			bottom[(degree + 2) * chebyshev_tile_width + local_col] = left_end.lo;
			bottom[(degree + 3) * chebyshev_tile_width + local_col] = corner.hi;
			bottom[(degree + 4) * chebyshev_tile_width + local_col] = corner.lo;
			left[(degree + 3) * chebyshev_tile_width + local_row] = corner.hi;
			left[(degree + 4) * chebyshev_tile_width + local_row] = corner.lo;
			if (return_grid)
				out[(batch * (rows + 1) + global_row + 1) * (cols + 1) + global_col + 1] = corner.hi + corner.lo;
			else if (global_row + 1 == rows && global_col + 1 == cols)
				out[batch] = corner.hi + corner.lo;
		}
		__syncwarp();
	}
	for (uint64_t k = 0; k < slots; ++k) {
		horizontal_global[k * chebyshev_tile_width + lane] = bottom[k * chebyshev_tile_width + lane];
		vertical_global[k * chebyshev_tile_width + lane] = left[k * chebyshev_tile_width + lane];
	}
}

template<typename T, int FixedDegree = -1, int FixedOrder = -1>
__global__ void chebyshev_backprop_kernel(
	const T* __restrict__ gram,
	T* __restrict__ gram_derivs,
	const T* __restrict__ output_derivs,
	const T* __restrict__ state,
	T* __restrict__ work,
	uint64_t rows,
	uint64_t cols,
	uint64_t degree_,
	uint64_t neumann_order_,
	uint64_t tile_rows,
	uint64_t tile_cols,
	uint64_t tile_diagonal,
	uint64_t batch_size,
	bool return_grid
) {
	const uint64_t batch = cuda_batch_index();
	if (batch >= batch_size)
		return;
	const uint64_t degree = FixedDegree < 0 ? degree_ : static_cast<uint64_t>(FixedDegree);
	const uint64_t neumann_order = FixedOrder < 0 ? neumann_order_ : static_cast<uint64_t>(FixedOrder);
	constexpr uint64_t capacity = FixedDegree < 0 ? 32 : FixedDegree + 1;
	const uint64_t first_tile_row = tile_diagonal >= tile_cols
		? tile_diagonal - tile_cols + 1 : 0;
	const uint64_t tile_row = first_tile_row + blockIdx.x;
	const uint64_t tile_col = tile_diagonal - tile_row;
	const uint64_t slots = degree + 3;
	const uint64_t tile_stride = slots * chebyshev_tile_width;
	T* const batch_work = work + batch * (tile_rows + tile_cols) * tile_stride;
	T* const horizontal_global = batch_work + tile_col * tile_stride;
	T* const vertical_global = batch_work + (tile_cols + tile_row) * tile_stride;
	extern __shared__ char shared_bytes[];
	T* const top = reinterpret_cast<T*>(shared_bytes);
	T* const right = top + tile_stride;
	const uint64_t lane = threadIdx.x;
	for (uint64_t k = 0; k < slots; ++k) {
		top[k * chebyshev_tile_width + lane] = horizontal_global[k * chebyshev_tile_width + lane];
		right[k * chebyshev_tile_width + lane] = vertical_global[k * chebyshev_tile_width + lane];
	}
	__syncwarp();

	const uint64_t row_start = tile_row * chebyshev_tile_width;
	const uint64_t col_start = tile_col * chebyshev_tile_width;
	const uint64_t active_rows = min(chebyshev_tile_width, rows - row_start);
	const uint64_t active_cols = min(chebyshev_tile_width, cols - col_start);
	const uint64_t state_width = 2 * degree + 8;
	for (uint64_t diagonal = active_rows + active_cols - 1; diagonal-- > 0;) {
		const uint64_t first_row = diagonal >= active_cols ? diagonal - active_cols + 1 : 0;
		const uint64_t last_row = min(diagonal, active_rows - 1);
		if (lane <= last_row - first_row) {
			const uint64_t local_row = first_row + lane;
			const uint64_t local_col = diagonal - local_row;
			const uint64_t global_row = row_start + local_row;
			const uint64_t global_col = col_start + local_col;
			const uint64_t cell = (batch * rows + global_row) * cols + global_col;
			const T* const saved = state + cell * state_width;
			T top_deriv[capacity], right_deriv[capacity], bottom_deriv[capacity], left_deriv[capacity];
			for (uint64_t k = 0; k <= degree; ++k) {
				top_deriv[k] = top[k * chebyshev_tile_width + local_col];
				right_deriv[k] = right[k * chebyshev_tile_width + local_row];
			}
			T corner_deriv = top[(degree + 2) * chebyshev_tile_width + local_col]
				+ right[(degree + 2) * chebyshev_tile_width + local_row];
			if (return_grid)
				corner_deriv += output_derivs[(batch * (rows + 1) + global_row + 1) * (cols + 1) + global_col + 1];
			else if (global_row + 1 == rows && global_col + 1 == cols)
				corner_deriv += output_derivs[batch];
			const T top_start_deriv = top[(degree + 1) * chebyshev_tile_width + local_col];
			const T right_start_deriv = right[(degree + 1) * chebyshev_tile_width + local_row];
			const chebyshev_sig_kernel::Corner<T> start{
				saved[2 * (degree + 1)], saved[2 * (degree + 1) + 1]};
			const auto endpoints = chebyshev_sig_kernel::cell_backward<T, FixedDegree, FixedOrder>(
				gram[cell], saved, saved + degree + 1, start, top_deriv, right_deriv,
				corner_deriv, bottom_deriv, left_deriv, degree, neumann_order);
			gram_derivs[cell] = endpoints.coefficient;
			for (uint64_t k = 0; k <= degree; ++k) {
				top[k * chebyshev_tile_width + local_col] = bottom_deriv[k];
				right[k * chebyshev_tile_width + local_row] = left_deriv[k];
			}
			top[(degree + 1) * chebyshev_tile_width + local_col] = endpoints.start;
			right[(degree + 1) * chebyshev_tile_width + local_row] = static_cast<T>(0);
			top[(degree + 2) * chebyshev_tile_width + local_col] = endpoints.bottom_end + right_start_deriv;
			right[(degree + 2) * chebyshev_tile_width + local_row] = endpoints.left_end + top_start_deriv;
		}
		__syncwarp();
	}
	for (uint64_t k = 0; k < slots; ++k) {
		horizontal_global[k * chebyshev_tile_width + lane] = top[k * chebyshev_tile_width + lane];
		vertical_global[k * chebyshev_tile_width + lane] = right[k * chebyshev_tile_width + lane];
	}
}

template<typename T, bool Backprop, int FixedDegree = -1, int FixedOrder = -1>
void launch_chebyshev_tiles(
	const T* gram, T* out, T* forward_state, const T* output_derivs,
	const T* backward_state, T* work, uint64_t batch_size,
	uint64_t rows, uint64_t cols, uint64_t degree, uint64_t neumann_order,
	uint64_t tile_rows, uint64_t tile_cols, bool return_grid
) {
	const uint64_t warps = !Backprop && std::is_same_v<T, float>
		&& degree <= 12 && batch_size >= chebyshev_forward_warps
		? chebyshev_forward_warps : 1;
	const uint64_t batch_blocks = batch_size / warps + (batch_size % warps != 0);
	constexpr bool stage_gram = !Backprop && chebyshev_stage_gram && std::is_same_v<T, float>
		&& FixedDegree >= 1 && FixedDegree <= 3 && FixedOrder >= FixedDegree && FixedOrder <= 3;
	const size_t gram_elements = stage_gram ? chebyshev_tile_width * chebyshev_tile_width : 0;
	const size_t smem = (2 * (degree + (Backprop ? 3 : 5)) * chebyshev_tile_width
		+ gram_elements) * sizeof(T) * warps;
	const auto smem_limits = cuda_shared_memory_limits();
	if constexpr (Backprop)
		configure_dynamic_smem(chebyshev_backprop_kernel<T, FixedDegree, FixedOrder>, smem,
			"CUDA Chebyshev kernel backprop", smem_limits);
	else
		configure_dynamic_smem(chebyshev_forward_kernel<T, FixedDegree, FixedOrder>, smem,
			"CUDA Chebyshev kernel", smem_limits);
	const uint64_t diagonals = tile_rows + tile_cols - 1;
	for (uint64_t step = 0; step < diagonals; ++step) {
		const uint64_t diagonal = Backprop ? diagonals - 1 - step : step;
		const uint64_t first_row = diagonal >= tile_cols ? diagonal - tile_cols + 1 : 0;
		const uint64_t last_row = std::min(diagonal, tile_rows - 1);
		const auto chunk = make_cuda_batch_grid_chunk(last_row - first_row + 1, batch_blocks, 0);
		if constexpr (Backprop)
			chebyshev_backprop_kernel<T, FixedDegree, FixedOrder><<<chunk.grid, 32U, smem>>>(
				gram, out, output_derivs, backward_state, work, rows, cols, degree,
				neumann_order, tile_rows, tile_cols, diagonal, batch_size, return_grid);
		else
			chebyshev_forward_kernel<T, FixedDegree, FixedOrder><<<chunk.grid, static_cast<unsigned>(warps * 32), smem>>>(
				gram, out, forward_state, work, rows, cols, degree, neumann_order,
				tile_rows, tile_cols, diagonal, batch_size, return_grid);
	}
}

template<typename T, bool Backprop>
void dispatch_chebyshev_tiles(
	const T* gram, T* out, T* forward_state, const T* output_derivs,
	const T* backward_state, T* work, uint64_t batch_size,
	uint64_t rows, uint64_t cols, uint64_t degree, uint64_t neumann_order,
	uint64_t tile_rows, uint64_t tile_cols, bool return_grid
) {
#define LAUNCH_CHEBYSHEV(p, k) \
	launch_chebyshev_tiles<T, Backprop, p, k>(gram, out, forward_state, output_derivs, \
		backward_state, work, batch_size, rows, cols, degree, neumann_order, \
		tile_rows, tile_cols, return_grid)
	if (degree == 1 && neumann_order == 1) { LAUNCH_CHEBYSHEV(1, 1); }
	else if (degree == 1 && neumann_order == 2) { LAUNCH_CHEBYSHEV(1, 2); }
	else if (degree == 1 && neumann_order == 3) { LAUNCH_CHEBYSHEV(1, 3); }
	else if (degree == 2 && neumann_order == 2) { LAUNCH_CHEBYSHEV(2, 2); }
	else if (degree == 2 && neumann_order == 3) { LAUNCH_CHEBYSHEV(2, 3); }
	else if (degree == 3 && neumann_order == 3) { LAUNCH_CHEBYSHEV(3, 3); }
	else if (degree == 4 && neumann_order == 4) { LAUNCH_CHEBYSHEV(4, 4); }
	else if (degree == 5 && neumann_order == 5) { LAUNCH_CHEBYSHEV(5, 5); }
	else if (degree == 4 && neumann_order == 6) { LAUNCH_CHEBYSHEV(4, 6); }
	else if (degree == 6 && neumann_order == 4) { LAUNCH_CHEBYSHEV(6, 4); }
	else if (degree == 6 && neumann_order == 6) { LAUNCH_CHEBYSHEV(6, 6); }
	else if (degree == 7 && neumann_order == 7) { LAUNCH_CHEBYSHEV(7, 7); }
	else if (degree == 8 && neumann_order == 8) { LAUNCH_CHEBYSHEV(8, 8); }
	else if (degree == 9 && neumann_order == 9) { LAUNCH_CHEBYSHEV(9, 9); }
	else if (degree == 10 && neumann_order == 10) { LAUNCH_CHEBYSHEV(10, 10); }
	else if (degree == 11 && neumann_order == 11) { LAUNCH_CHEBYSHEV(11, 11); }
	else if (degree == 12 && neumann_order == 12) { LAUNCH_CHEBYSHEV(12, 12); }
	else { LAUNCH_CHEBYSHEV(-1, -1); }
#undef LAUNCH_CHEBYSHEV
}

void validate_chebyshev_cuda(uint64_t dimension, uint64_t length1,
	uint64_t length2, uint64_t degree, uint64_t neumann_order) {
	if (dimension == 0)
		throw std::invalid_argument("sig_kernel_chebyshev_cuda: path dimension must be positive");
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("sig_kernel_chebyshev_cuda: paths must have length >= 1");
	if (degree < 1 || degree > 31 || neumann_order > 31 - degree)
		throw std::invalid_argument("sig_kernel_chebyshev_cuda: require degree >= 1 and degree + neumann_order <= 31");
}

template<typename T>
void sig_kernel_chebyshev_cuda_impl(
	const T* gram, T* out, T* state, uint64_t batch_size, uint64_t dimension,
	uint64_t length1, uint64_t length2, uint64_t degree,
	uint64_t neumann_order, bool return_grid
) {
	constexpr const char* op_name = "sig_kernel_chebyshev_cuda";
	validate_chebyshev_cuda(dimension, length1, length2, degree, neumann_order);
	if (batch_size == 0)
		return;
	if (!out)
		throw std::invalid_argument("sig_kernel_chebyshev_cuda: out must not be null");
	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	const size_t result_size = return_grid ? checked_cuda_size_mul(length1, length2, op_name) : 1;
	const size_t result_count = checked_cuda_size_mul(batch_size, result_size, op_name);
	(void)checked_cuda_size_mul(result_count, sizeof(T), op_name);
	if (rows == 0 || cols == 0) {
		chebyshev_fill_kernel<<<make_cuda_1d_grid(result_count, 256), 256U>>>(out, result_count, static_cast<T>(1));
		check_cuda_kernel_launch();
		return;
	}
	if (!gram)
		throw std::invalid_argument("sig_kernel_chebyshev_cuda: gram must not be null");
	const size_t cells = checked_cuda_size_mul(rows, cols, op_name);
	const size_t batch_cells = checked_cuda_size_mul(batch_size, cells, op_name);
	(void)checked_cuda_size_mul(batch_cells, sizeof(T), op_name);
	const size_t state_width = 2 * degree + 8;
	const size_t state_size = checked_cuda_size_mul(cells, state_width, op_name);
	if (state)
		(void)checked_cuda_size_mul(checked_cuda_size_mul(batch_size, state_size, op_name), sizeof(T), op_name);
	const uint64_t tile_rows = rows / chebyshev_tile_width + (rows % chebyshev_tile_width != 0);
	const uint64_t tile_cols = cols / chebyshev_tile_width + (cols % chebyshev_tile_width != 0);
	const size_t boundary_tiles = checked_cuda_size_add(tile_rows, tile_cols, op_name);
	const size_t row_elements = checked_cuda_size_mul(boundary_tiles, (degree + 5) * chebyshev_tile_width, op_name);
	CudaBatchWorkspace<T> workspace(batch_size, row_elements, op_name);
	for (uint64_t offset = 0; offset < batch_size;) {
		const uint64_t count = std::min(workspace.capacity(), batch_size - offset);
		const size_t work_count = count * row_elements;
		chebyshev_init_kernel<<<make_cuda_1d_grid(work_count, 256), 256U>>>(workspace.get(), work_count, degree);
		T* const batch_out = out + offset * result_size;
		if (return_grid)
			chebyshev_fill_kernel<<<make_cuda_1d_grid(count * result_size, 256), 256U>>>(batch_out, count * result_size, static_cast<T>(1));
		dispatch_chebyshev_tiles<T, false>(gram + offset * cells, batch_out,
			state ? state + offset * state_size : nullptr, nullptr, nullptr, workspace.get(),
			count, rows, cols, degree, neumann_order, tile_rows, tile_cols, return_grid);
		offset += count;
	}
	check_cuda_kernel_launch();
}

template<typename T>
void sig_kernel_chebyshev_backprop_cuda_impl(
	const T* gram, T* gram_derivs, const T* output_derivs, const T* state,
	uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2,
	uint64_t degree, uint64_t neumann_order, bool return_grid
) {
	constexpr const char* op_name = "sig_kernel_chebyshev_backprop_cuda";
	validate_chebyshev_cuda(dimension, length1, length2, degree, neumann_order);
	if (batch_size == 0 || length1 == 1 || length2 == 1)
		return;
	if (!gram || !gram_derivs || !output_derivs)
		throw std::invalid_argument("sig_kernel_chebyshev_backprop_cuda: gram, gram_derivs, and output_derivs must not be null");
	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	const size_t cells = checked_cuda_size_mul(rows, cols, op_name);
	(void)checked_cuda_size_mul(checked_cuda_size_mul(batch_size, cells, op_name), sizeof(T), op_name);
	const size_t result_size = return_grid ? checked_cuda_size_mul(length1, length2, op_name) : 1;
	(void)checked_cuda_size_mul(checked_cuda_size_mul(batch_size, result_size, op_name), sizeof(T), op_name);
	const size_t state_size = checked_cuda_size_mul(cells, 2 * degree + 8, op_name);
	if (state)
		(void)checked_cuda_size_mul(checked_cuda_size_mul(batch_size, state_size, op_name), sizeof(T), op_name);
	const uint64_t tile_rows = rows / chebyshev_tile_width + (rows % chebyshev_tile_width != 0);
	const uint64_t tile_cols = cols / chebyshev_tile_width + (cols % chebyshev_tile_width != 0);
	const size_t boundary_tiles = checked_cuda_size_add(tile_rows, tile_cols, op_name);
	const size_t frontier_size = checked_cuda_size_mul(boundary_tiles, (degree + 3) * chebyshev_tile_width, op_name);
	const size_t row_elements = state ? frontier_size
		: checked_cuda_size_add(frontier_size, checked_cuda_size_add(state_size, 1, op_name), op_name);
	CudaBatchWorkspace<T> workspace(batch_size, row_elements, op_name);
	for (uint64_t offset = 0; offset < batch_size;) {
		const uint64_t count = std::min(workspace.capacity(), batch_size - offset);
		const T* batch_state = state ? state + offset * state_size : nullptr;
		if (!batch_state) {
			T* const regenerated = workspace.get() + count * frontier_size;
			T* const ignored_output = regenerated + count * state_size;
			sig_kernel_chebyshev_cuda_impl(gram + offset * cells, ignored_output,
				regenerated, count, dimension, length1, length2, degree, neumann_order, false);
			batch_state = regenerated;
		}
		CUDA_CHECK(cudaMemset(workspace.get(), 0, count * frontier_size * sizeof(T)));
		dispatch_chebyshev_tiles<T, true>(gram + offset * cells, gram_derivs + offset * cells,
			nullptr, output_derivs + offset * result_size, batch_state, workspace.get(),
			count, rows, cols, degree, neumann_order, tile_rows, tile_cols, return_grid);
		offset += count;
	}
	check_cuda_kernel_launch();
}

} // namespace

#include "cu_macros.h"

extern "C" {

	CUSIG_API int sig_kernel_chebyshev_cuda_f(
		const float* gram, float* out, float* state, uint64_t batch_size,
		uint64_t dimension, uint64_t length1, uint64_t length2,
		uint64_t degree, uint64_t neumann_order, bool return_grid) noexcept {
		CUDA_SAFE_CALL(sig_kernel_chebyshev_cuda_impl(gram, out, state, batch_size,
			dimension, length1, length2, degree, neumann_order, return_grid));
	}

	CUSIG_API int sig_kernel_chebyshev_cuda_d(
		const double* gram, double* out, double* state, uint64_t batch_size,
		uint64_t dimension, uint64_t length1, uint64_t length2,
		uint64_t degree, uint64_t neumann_order, bool return_grid) noexcept {
		CUDA_SAFE_CALL(sig_kernel_chebyshev_cuda_impl(gram, out, state, batch_size,
			dimension, length1, length2, degree, neumann_order, return_grid));
	}

	CUSIG_API int sig_kernel_chebyshev_backprop_cuda_f(
		const float* gram, float* gram_derivs, const float* output_derivs,
		const float* state, uint64_t batch_size, uint64_t dimension,
		uint64_t length1, uint64_t length2, uint64_t degree,
		uint64_t neumann_order, bool return_grid) noexcept {
		CUDA_SAFE_CALL(sig_kernel_chebyshev_backprop_cuda_impl(gram, gram_derivs,
			output_derivs, state, batch_size, dimension, length1, length2,
			degree, neumann_order, return_grid));
	}

	CUSIG_API int sig_kernel_chebyshev_backprop_cuda_d(
		const double* gram, double* gram_derivs, const double* output_derivs,
		const double* state, uint64_t batch_size, uint64_t dimension,
		uint64_t length1, uint64_t length2, uint64_t degree,
		uint64_t neumann_order, bool return_grid) noexcept {
		CUDA_SAFE_CALL(sig_kernel_chebyshev_backprop_cuda_impl(gram, gram_derivs,
			output_derivs, state, batch_size, dimension, length1, length2,
			degree, neumann_order, return_grid));
	}

}
