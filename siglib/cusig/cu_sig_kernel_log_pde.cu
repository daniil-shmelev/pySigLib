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

#include "cupch.h"
#include "cusig.h"
#include "cu_macros.h"
#include "cu_sig_kernel_log_pde_cell.cuh"

#include <type_traits>

namespace log_pde_cuda_detail {

struct HostLayout {
	Params p{};
	std::vector<uint64_t> offset;

	HostLayout(
		uint64_t batch_size,
		uint64_t dimension,
		uint64_t length_x,
		uint64_t length_y,
		uint64_t log_step_x,
		uint64_t log_step_y,
		uint64_t degree_x,
		uint64_t degree_y,
		uint64_t dyadic_order_x,
		uint64_t dyadic_order_y
	) {
		if (dimension == 0) throw std::invalid_argument("log-PDE CUDA dimension must be positive");
		if (length_x <= 1 || length_y <= 1)
			throw std::invalid_argument("log-PDE CUDA requires paths with at least two points");
		if (log_step_x == 0 || log_step_y == 0)
			throw std::invalid_argument("log-PDE CUDA block sizes must be positive");
		if (degree_x == 0 || degree_y == 0)
			throw std::invalid_argument("log-PDE CUDA degrees must be positive");
		if (dyadic_order_x >= 63 || dyadic_order_y >= 63)
			throw std::invalid_argument("log-PDE CUDA dyadic order is too large");

		const uint64_t intervals_x = length_x - 1;
		const uint64_t intervals_y = length_y - 1;
		if (intervals_x % log_step_x != 0 || intervals_y % log_step_y != 0)
			throw std::invalid_argument("log-PDE CUDA block size must divide the path intervals");

		p.batch_size = batch_size;
		p.dimension = dimension;
		p.degree_x = degree_x;
		p.degree_y = degree_y;
		p.degree_f = degree_y - 1;
		p.degree_g = degree_x - 1;
		p.steps_x = intervals_x / log_step_x;
		p.steps_y = intervals_y / log_step_y;
		p.factor_x = 1ULL << dyadic_order_x;
		p.factor_y = 1ULL << dyadic_order_y;
		p.shift_x = dyadic_order_x;
		p.shift_y = dyadic_order_y;
		p.scale_x = 1.0 / static_cast<double>(p.factor_x);
		p.scale_y = 1.0 / static_cast<double>(p.factor_y);
		p.nodes_x = p.steps_x * p.factor_x + 1;
		p.nodes_y = p.steps_y * p.factor_y + 1;

		const uint64_t max_degree = degree_x > degree_y ? degree_x : degree_y;
		offset.assign(max_degree + 2, 0);
		uint64_t level_size = 1;
		for (uint64_t level = 1; level <= max_degree; ++level) {
			level_size *= dimension;
			offset[level + 1] = offset[level] + level_size;
		}

		auto length = [&](uint64_t degree) {
			return degree == 0 ? uint64_t(0) : offset[degree + 1];
		};
		p.x_len = length(p.degree_x);
		p.y_len = length(p.degree_y);
		p.f_len = length(p.degree_f);
		p.g_len = length(p.degree_g);
		p.state_len = 1 + p.f_len + p.g_len;
		p.cell_cache_len = 2 * (p.f_len + p.g_len) + 3;
		p.grid_len = p.nodes_x * p.nodes_y;
		const uint64_t max_cells_x = p.nodes_x - 1;
		const uint64_t max_cells_y = p.nodes_y - 1;
		const uint64_t max_diagonal_cells = max_cells_x < max_cells_y ?
			max_cells_x : max_cells_y;
		p.path_threads = 1;
		while (p.path_threads < max_diagonal_cells && p.path_threads < PATH_THREADS)
			p.path_threads *= 2;
		p.paths_per_block = PATH_THREADS / p.path_threads;
	}
};

struct ForwardWorkspaceLayout {
	uint64_t boundary_x;
	uint64_t boundary_y;
	uint64_t diagonal;
	uint64_t scratch;
	uint64_t total;
};

ForwardWorkspaceLayout forward_workspace_layout(const Params& p) {
	ForwardWorkspaceLayout result{};
	result.boundary_x = (p.steps_x + 1) * p.f_len;
	result.boundary_y = (p.steps_y + 1) * p.g_len;
	const uint64_t diagonal_nodes = p.nodes_x < p.nodes_y ? p.nodes_x : p.nodes_y;
	result.diagonal = diagonal_nodes * p.state_len;
	result.scratch = 4 * (p.f_len + p.g_len);
	result.total = result.boundary_x + result.boundary_y + 3 * result.diagonal +
		p.path_threads * result.scratch;
	return result;
}

struct BackwardWorkspaceLayout {
	uint64_t boundary_x;
	uint64_t boundary_y;
	uint64_t tape;
	uint64_t cache;
	uint64_t adjoint;
	uint64_t coarse;
	uint64_t cell_scratch;
	uint64_t scratch;
	uint64_t total;
};

BackwardWorkspaceLayout backward_workspace_layout(const Params& p) {
	BackwardWorkspaceLayout result{};
	result.boundary_x = (p.steps_x + 1) * p.f_len;
	result.boundary_y = (p.steps_y + 1) * p.g_len;
	result.tape = p.grid_len * p.state_len;
	result.cache = (p.nodes_x - 1) * (p.nodes_y - 1) * p.cell_cache_len;
	result.adjoint = result.tape;
	const uint64_t coarse_x = (p.steps_x + 1) * p.f_len;
	const uint64_t coarse_y = (p.steps_y + 1) * p.g_len;
	result.coarse = coarse_x > coarse_y ? coarse_x : coarse_y;
	const uint64_t max_diagonal_cells = p.nodes_x < p.nodes_y ?
		p.nodes_x - 1 : p.nodes_y - 1;
	const uint64_t neighbor_states = max_diagonal_cells <= p.path_threads ? 2 : 3;
	result.cell_scratch = 4 * (p.f_len + p.g_len) + p.x_len + p.y_len;
	result.scratch = result.cell_scratch + neighbor_states * p.state_len;
	const uint64_t boundary_x_scratch = (4 + 2 * p.degree_f) * p.f_len;
	const uint64_t boundary_y_scratch = (4 + 2 * p.degree_g) * p.g_len;
	if (boundary_x_scratch > result.scratch) result.scratch = boundary_x_scratch;
	if (boundary_y_scratch > result.scratch) result.scratch = boundary_y_scratch;
	result.total = result.boundary_x + result.boundary_y + result.tape + result.cache +
		result.adjoint + result.coarse + p.path_threads * result.scratch;
	return result;
}

template<typename T>
__global__ void log_pde_forward_kernel(
	Params p,
	const T* __restrict__ logs_x,
	const T* __restrict__ logs_y,
	T* __restrict__ out,
	T* __restrict__ workspace,
	ForwardWorkspaceLayout layout,
	bool use_shared_workspace,
	bool return_grid
) {
	extern __shared__ __align__(16) unsigned char shared_storage[];
	T* shared_diagonals = reinterpret_cast<T*>(shared_storage);
	const uint64_t group = threadIdx.x / p.path_threads;
	const uint64_t thread = threadIdx.x - group * p.path_threads;
	const uint64_t candidate_batch = blockIdx.x * p.paths_per_block + group;
	const bool active = candidate_batch < p.batch_size;
	const uint64_t batch = active ? candidate_batch : 0;
	const ConstView<T> x = {
		logs_x + batch * p.steps_x * p.x_len, 1, static_cast<T>(1)
	};
	const ConstView<T> y = {
		logs_y + batch * p.steps_y * p.y_len, 1, static_cast<T>(1)
	};
	View<T> storage = { workspace + batch * layout.total, 1 };
	const View<T> boundary_x = storage;
	const View<T> boundary_y = boundary_x.sub(layout.boundary_x);
	const View<T> global_diagonals = boundary_y.sub(layout.boundary_y);
	const View<T> global_scratch = global_diagonals.sub(3 * layout.diagonal);
	const uint64_t shared_path_len = 3 * layout.diagonal + p.path_threads * layout.scratch;
	const View<T> shared_path = {
		shared_diagonals + group * shared_path_len, 1
	};
	const View<T> diagonals = use_shared_workspace ? shared_path : global_diagonals;
	const View<T> scratch_storage = use_shared_workspace ?
		shared_path.sub(3 * layout.diagonal) : global_scratch;
	const View<T> thread_storage = scratch_storage.sub(thread * layout.scratch);

	if (active && thread == 0) {
		build_boundary(p, x, p.steps_x, p.degree_x, p.degree_f,
			boundary_x, thread_storage);
		build_boundary(p, y, p.steps_y, p.degree_y, p.degree_g,
			boundary_y, thread_storage);
	}
	CellScratch<T> cell = make_forward_cell_scratch(thread_storage, p);
	const T scale_x = static_cast<T>(p.scale_x);
	const T scale_y = static_cast<T>(p.scale_y);
	const uint64_t last_diagonal = p.nodes_x + p.nodes_y - 2;
	for (uint64_t diagonal = 0; diagonal <= last_diagonal; ++diagonal) {
		const uint64_t i_min = diagonal >= p.nodes_y ? diagonal - p.nodes_y + 1 : 0;
		const uint64_t i_max = diagonal < p.nodes_x ? diagonal : p.nodes_x - 1;
		const uint64_t node_count = i_max - i_min + 1;
		const View<T> current = diagonals.sub((diagonal % 3) * layout.diagonal);
		const uint64_t previous_i_min = diagonal > 0 && diagonal - 1 >= p.nodes_y ?
			diagonal - p.nodes_y : 0;
		const uint64_t northwest_i_min = diagonal > 1 && diagonal - 2 >= p.nodes_y ?
			diagonal - p.nodes_y - 1 : 0;
		for (uint64_t offset = thread; active && offset < node_count;
			offset += p.path_threads) {
			const uint64_t i = i_min + offset;
			const uint64_t j = diagonal - i;
			const View<T> state = current.sub(offset * p.state_len);
			if (i == 0 || j == 0) {
				for (uint64_t k = 0; k < p.state_len; ++k)
					state[k] = static_cast<T>(0);
				state[0] = static_cast<T>(1);
				if (i != 0) {
					const uint64_t shifted = i >> p.shift_x;
					const uint64_t coarse = shifted < p.steps_x ? shifted : p.steps_x;
					const ConstView<T> source = boundary_x.sub(coarse * p.f_len).read();
					for (uint64_t k = 0; k < p.f_len; ++k) state[1 + k] = source[k];
				}
				if (j != 0) {
					const uint64_t shifted = j >> p.shift_y;
					const uint64_t coarse = shifted < p.steps_y ? shifted : p.steps_y;
					const ConstView<T> source = boundary_y.sub(coarse * p.g_len).read();
					for (uint64_t k = 0; k < p.g_len; ++k)
						state[1 + p.f_len + k] = source[k];
				}
			}
			else {
				const ConstView<T> dx = x.sub(((i - 1) >> p.shift_x) * p.x_len);
				const ConstView<T> scaled_dx = { dx.data, dx.stride, scale_x };
				const ConstView<T> dy = y.sub(((j - 1) >> p.shift_y) * p.y_len);
				const ConstView<T> scaled_dy = { dy.data, dy.stride, scale_y };
				const ConstView<T> previous = diagonals.sub(
					((diagonal - 1) % 3) * layout.diagonal).read();
				const ConstView<T> northwest = diagonals.sub(
					((diagonal - 2) % 3) * layout.diagonal).read();
				cell_forward(p, scaled_dx, scaled_dy,
					northwest.sub((i - 1 - northwest_i_min) * p.state_len),
					previous.sub((i - 1 - previous_i_min) * p.state_len),
					previous.sub((i - previous_i_min) * p.state_len),
					state, false, static_cast<T>(0), cell);
			}
			if (return_grid)
				out[batch * p.grid_len + i * p.nodes_y + j] = state[0];
		}
		__syncwarp();
	}
	if (active && !return_grid && thread == 0)
		out[batch] = diagonals[(last_diagonal % 3) * layout.diagonal];
}

template<typename T>
__launch_bounds__(PATH_THREADS, 16)
__global__ void log_pde_backward_kernel(
	Params p,
	const T* __restrict__ logs_x,
	const T* __restrict__ logs_y,
	T* __restrict__ d_logs_x,
	T* __restrict__ d_logs_y,
	const T* __restrict__ derivs,
	const T* __restrict__ k_grid,
	T* __restrict__ workspace,
	BackwardWorkspaceLayout layout,
	bool return_grid
) {
	const uint64_t group = threadIdx.x / p.path_threads;
	const uint64_t thread = threadIdx.x - group * p.path_threads;
	const uint64_t candidate_batch = blockIdx.x * p.paths_per_block + group;
	const bool active = candidate_batch < p.batch_size;
	const uint64_t batch = active ? candidate_batch : 0;
	const ConstView<T> x = {
		logs_x + batch * p.steps_x * p.x_len, 1, static_cast<T>(1)
	};
	const ConstView<T> y = {
		logs_y + batch * p.steps_y * p.y_len, 1, static_cast<T>(1)
	};
	const View<T> dx_logs = { d_logs_x + batch * p.steps_x * p.x_len, 1 };
	const View<T> dy_logs = { d_logs_y + batch * p.steps_y * p.y_len, 1 };
	View<T> storage = { workspace + batch * layout.total, 1 };
	const View<T> boundary_x = storage;
	const View<T> boundary_y = boundary_x.sub(layout.boundary_x);
	const View<T> tape = boundary_y.sub(layout.boundary_y);
	const View<T> cell_cache = tape.sub(layout.tape);
	const View<T> adjoint = cell_cache.sub(layout.cache);
	const View<T> coarse = adjoint.sub(layout.adjoint);
	const View<T> scratch_storage = coarse.sub(layout.coarse);
	const View<T> thread_storage = scratch_storage.sub(thread * layout.scratch);

	if (active && thread == 0) {
		build_boundary(p, x, p.steps_x, p.degree_x, p.degree_f,
			boundary_x, thread_storage);
		build_boundary(p, y, p.steps_y, p.degree_y, p.degree_g,
			boundary_y, thread_storage);
	}
	for (uint64_t index = thread; active && index < layout.tape; index += p.path_threads)
		tape[index] = static_cast<T>(0);
	for (uint64_t index = thread; active && index < layout.adjoint; index += p.path_threads)
		adjoint[index] = static_cast<T>(0);
	for (uint64_t index = thread; active && index < p.steps_x * p.x_len;
		index += p.path_threads)
		dx_logs[index] = static_cast<T>(0);
	for (uint64_t index = thread; active && index < p.steps_y * p.y_len;
		index += p.path_threads)
		dy_logs[index] = static_cast<T>(0);
	__syncwarp();

	for (uint64_t j = thread; active && j < p.nodes_y; j += p.path_threads) {
		const View<T> state = tape.sub(j * p.state_len);
		state[0] = static_cast<T>(1);
		const uint64_t shifted = j >> p.shift_y;
		const uint64_t coarse_y = shifted < p.steps_y ? shifted : p.steps_y;
		const ConstView<T> source = boundary_y.sub(coarse_y * p.g_len).read();
		for (uint64_t k = 0; k < p.g_len; ++k)
			state[1 + p.f_len + k] = source[k];
	}
	for (uint64_t i = thread + 1; active && i < p.nodes_x; i += p.path_threads) {
		const View<T> state = tape.sub(i * p.nodes_y * p.state_len);
		state[0] = static_cast<T>(1);
		const uint64_t shifted = i >> p.shift_x;
		const uint64_t coarse_x = shifted < p.steps_x ? shifted : p.steps_x;
		const ConstView<T> source = boundary_x.sub(coarse_x * p.f_len).read();
		for (uint64_t k = 0; k < p.f_len; ++k) state[1 + k] = source[k];
	}
	__syncwarp();

	CellScratch<T> forward_cell = make_forward_cell_scratch(thread_storage, p);
	const T scale_x = static_cast<T>(p.scale_x);
	const T scale_y = static_cast<T>(p.scale_y);
	const uint64_t last_diagonal = p.nodes_x + p.nodes_y - 2;
	for (uint64_t diagonal = 2; diagonal <= last_diagonal; ++diagonal) {
		const uint64_t i_min = diagonal >= p.nodes_y ? diagonal - p.nodes_y + 1 : 1;
		const uint64_t i_max = diagonal - 1 < p.nodes_x - 1 ? diagonal - 1 : p.nodes_x - 1;
		const uint64_t cell_count = i_max >= i_min ? i_max - i_min + 1 : 0;
		for (uint64_t offset = thread; active && offset < cell_count;
			offset += p.path_threads) {
			const uint64_t i = i_min + offset;
			const uint64_t j = diagonal - i;
			const ConstView<T> dx = x.sub(((i - 1) >> p.shift_x) * p.x_len);
			const ConstView<T> scaled_dx = { dx.data, dx.stride, scale_x };
			const ConstView<T> dy = y.sub(((j - 1) >> p.shift_y) * p.y_len);
			const ConstView<T> scaled_dy = { dy.data, dy.stride, scale_y };
			const uint64_t node = i * p.nodes_y + j;
			const bool has_grid = k_grid != nullptr;
			const T known_u = has_grid ? k_grid[batch * p.grid_len + node] : static_cast<T>(0);
			const View<T> se = tape.sub(node * p.state_len);
			cell_forward(p, scaled_dx, scaled_dy,
				tape.sub(((i - 1) * p.nodes_y + j - 1) * p.state_len).read(),
				tape.sub(((i - 1) * p.nodes_y + j) * p.state_len).read(),
				tape.sub((i * p.nodes_y + j - 1) * p.state_len).read(),
				se, has_grid, known_u, forward_cell);
			View<T> cache = cell_cache.sub(
				((i - 1) * (p.nodes_y - 1) + j - 1) * p.cell_cache_len);
			for (uint64_t k = 0; k < p.f_len; ++k) cache[k] = forward_cell.dx_adj_dy[k];
			cache = cache.sub(p.f_len);
			for (uint64_t k = 0; k < p.g_len; ++k) cache[k] = forward_cell.dy_adj_dx[k];
			cache = cache.sub(p.g_len);
			for (uint64_t k = 0; k < p.f_len; ++k) cache[k] = forward_cell.fp[k];
			cache = cache.sub(p.f_len);
			for (uint64_t k = 0; k < p.g_len; ++k) cache[k] = forward_cell.gp[k];
			cache = cache.sub(p.g_len);
			cache[0] = forward_cell.gamma;
			cache[1] = forward_cell.u_base;
			cache[2] = forward_cell.u_provisional;
		}
		__syncwarp();
	}

	if (active && return_grid) {
		for (uint64_t node = thread; node < p.grid_len; node += p.path_threads)
			adjoint[node * p.state_len] = derivs[batch * p.grid_len + node];
	}
	else if (active && thread == 0) {
		adjoint[(p.grid_len - 1) * p.state_len] = derivs[batch];
	}
	__syncwarp();

	const uint64_t max_diagonal_cells = p.nodes_x < p.nodes_y ?
		p.nodes_x - 1 : p.nodes_y - 1;
	const bool needs_local_nw = max_diagonal_cells > p.path_threads;
	const View<T> local_nw = thread_storage.sub(layout.cell_scratch);
	const View<T> local_north = needs_local_nw ? local_nw.sub(p.state_len) : local_nw;
	const View<T> local_west = local_north.sub(p.state_len);
	CellScratch<T> cell = make_backward_cell_scratch(thread_storage, p);
	for (uint64_t diagonal = last_diagonal; diagonal >= 2; --diagonal) {
		const uint64_t i_min = diagonal >= p.nodes_y ? diagonal - p.nodes_y + 1 : 1;
		const uint64_t i_max = diagonal - 1 < p.nodes_x - 1 ? diagonal - 1 : p.nodes_x - 1;
		const uint64_t cell_count = i_max >= i_min ? i_max - i_min + 1 : 0;
		const bool direct_neighbors = cell_count <= p.path_threads;
		for (uint64_t offset = thread; active && offset < cell_count;
			offset += p.path_threads) {
			const uint64_t i = i_min + offset;
			const uint64_t j = diagonal - i;
			const uint64_t coarse_x = (i - 1) >> p.shift_x;
			const uint64_t coarse_y = (j - 1) >> p.shift_y;
			const ConstView<T> dx = x.sub(coarse_x * p.x_len);
			const ConstView<T> scaled_dx = { dx.data, dx.stride, scale_x };
			const ConstView<T> dy = y.sub(coarse_y * p.y_len);
			const ConstView<T> scaled_dy = { dy.data, dy.stride, scale_y };
			const uint64_t node = i * p.nodes_y + j;
			const View<T> d_nw = adjoint.sub(
				((i - 1) * p.nodes_y + j - 1) * p.state_len);
			const View<T> backward_nw = direct_neighbors ? d_nw : local_nw;
			if (!direct_neighbors)
				for (uint64_t k = 0; k < p.state_len; ++k) local_nw[k] = static_cast<T>(0);
			for (uint64_t k = 0; k < p.state_len; ++k) {
				local_north[k] = static_cast<T>(0);
				local_west[k] = static_cast<T>(0);
			}
			cell_backward(p, scaled_dx, scaled_dy,
				tape.sub(((i - 1) * p.nodes_y + j - 1) * p.state_len).read(),
				tape.sub(((i - 1) * p.nodes_y + j) * p.state_len).read(),
				tape.sub((i * p.nodes_y + j - 1) * p.state_len).read(),
				tape.sub(node * p.state_len).read(),
				cell_cache.sub(((i - 1) * (p.nodes_y - 1) + j - 1) * p.cell_cache_len).read(),
				adjoint.sub(node * p.state_len).read(),
				backward_nw, local_north, local_west, cell);
			if (!direct_neighbors) {
				const View<T> d_north = adjoint.sub(
					((i - 1) * p.nodes_y + j) * p.state_len);
				const View<T> d_west = adjoint.sub(
					(i * p.nodes_y + j - 1) * p.state_len);
				for (uint64_t k = 0; k < p.state_len; ++k) {
					atomicAdd(d_nw.data + k * d_nw.stride, local_nw[k]);
					atomicAdd(d_north.data + k * d_north.stride, local_north[k]);
					atomicAdd(d_west.data + k * d_west.stride, local_west[k]);
				}
			}
			for (uint64_t k = 0; k < p.x_len; ++k) {
				const T value = cell.d_dx[k] * scale_x;
				if (p.factor_x == 1) dx_logs[coarse_x * p.x_len + k] += value;
				else atomicAdd(dx_logs.data + (coarse_x * p.x_len + k) * dx_logs.stride, value);
			}
			for (uint64_t k = 0; k < p.y_len; ++k) {
				const T value = cell.d_dy[k] * scale_y;
				if (p.factor_y == 1) dy_logs[coarse_y * p.y_len + k] += value;
				else atomicAdd(dy_logs.data + (coarse_y * p.y_len + k) * dy_logs.stride, value);
			}
		}
		if (direct_neighbors) {
			const bool cell_active = active && thread < cell_count;
			View<T> north_target = { nullptr, 1 };
			View<T> west_target = { nullptr, 1 };
			if (cell_active) {
				const uint64_t i = i_min + thread;
				const uint64_t j = diagonal - i;
				north_target = adjoint.sub(
					((i - 1) * p.nodes_y + j) * p.state_len);
			}
			if (active && thread == 0) {
				const uint64_t i = i_min + cell_count - 1;
				const uint64_t j = diagonal - i;
				west_target = adjoint.sub(
					(i * p.nodes_y + j - 1) * p.state_len);
			}
			for (uint64_t k = 0; k < p.state_len; ++k) {
				const T north = cell_active ? local_north[k] : static_cast<T>(0);
				const T west = cell_active ? local_west[k] : static_cast<T>(0);
				const T previous_west = __shfl_up_sync(
					0xffffffffu, west, 1, static_cast<int>(p.path_threads));
				const T last_west = __shfl_sync(
					0xffffffffu, west, static_cast<int>(cell_count - 1),
					static_cast<int>(p.path_threads));
				if (cell_active)
					north_target[k] += north +
						(thread == 0 ? static_cast<T>(0) : previous_west);
				if (active && thread == 0) west_target[k] += last_west;
			}
		}
		__syncwarp();
		if (diagonal == 2) break;
	}

	if (active && thread == 0 && p.f_len != 0) {
		boundary_backward(p, x, p.steps_x, p.nodes_x, p.shift_x,
			p.degree_x, p.degree_f, boundary_x.read(), adjoint.read(),
			p.nodes_y * p.state_len, 1, coarse, dx_logs, thread_storage);
	}
	if (active && thread == 0 && p.g_len != 0) {
		boundary_backward(p, y, p.steps_y, p.nodes_y, p.shift_y,
			p.degree_y, p.degree_g, boundary_y.read(), adjoint.read(),
			p.state_len, 1 + p.f_len, coarse, dy_logs, thread_storage);
	}
}
template<typename T>
__global__ void gather_path_blocks_kernel(
	const T* path,
	T* blocks,
	uint64_t total,
	uint64_t dimension,
	uint64_t length,
	uint64_t steps,
	uint64_t log_step
) {
	for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
		index < total; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		uint64_t value = index;
		const uint64_t coordinate = value % dimension;
		value /= dimension;
		const uint64_t local_point = value % (log_step + 1);
		value /= log_step + 1;
		const uint64_t step = value % steps;
		const uint64_t batch = value / steps;
		blocks[index] = path[(batch * length + step * log_step + local_point) *
			dimension + coordinate];
	}
}

template<typename T>
__global__ void pack_logs_kernel(
	const T* full_logs,
	T* logs,
	uint64_t total,
	uint64_t steps,
	uint64_t log_len
) {
	const uint64_t full_len = log_len + 1;
	for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
		index < total; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		uint64_t value = index;
		const uint64_t coefficient = value % log_len;
		value /= log_len;
		const uint64_t step = value % steps;
		const uint64_t batch = value / steps;
		logs[(batch * steps + step) * log_len + coefficient] =
			full_logs[(batch * steps + step) * full_len + coefficient + 1];
	}
}

template<typename T>
__global__ void unpack_log_derivs_kernel(
	const T* d_logs,
	T* full_derivs,
	uint64_t total,
	uint64_t steps,
	uint64_t log_len
) {
	const uint64_t full_len = log_len + 1;
	for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
		index < total; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		uint64_t value = index;
		const uint64_t coefficient = value % full_len;
		value /= full_len;
		const uint64_t step = value % steps;
		const uint64_t batch = value / steps;
		full_derivs[index] = coefficient == 0 ? static_cast<T>(0) :
			d_logs[(batch * steps + step) * log_len + coefficient - 1];
	}
}

template<typename T>
__global__ void scatter_block_derivs_kernel(
	const T* block_derivs,
	T* d_path,
	uint64_t total,
	uint64_t dimension,
	uint64_t length,
	uint64_t steps,
	uint64_t log_step
) {
	for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
		index < total; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		uint64_t value = index;
		const uint64_t coordinate = value % dimension;
		value /= dimension;
		const uint64_t point = value % length;
		const uint64_t batch = value / length;
		T result = static_cast<T>(0);
		const uint64_t step = point / log_step;
		if (step < steps) {
			const uint64_t local_point = point - step * log_step;
			result += block_derivs[((batch * steps + step) * (log_step + 1) +
				local_point) * dimension + coordinate];
		}
		if (point != 0 && point % log_step == 0) {
			result += block_derivs[((batch * steps + step - 1) * (log_step + 1) +
				log_step) * dimension + coordinate];
		}
		d_path[index] = result;
	}
}

template<typename T>
struct PreparedPath {
	CudaBuf<T> blocks;
	CudaBuf<T> signatures;
	CudaBuf<T> logs;
	uint64_t steps{};
	uint64_t log_step{};
	uint64_t degree{};
	uint64_t log_len{};
	uint64_t full_len{};
};

template<typename T>
PreparedPath<T> prepare_path_logs(
	const T* path,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t log_step,
	uint64_t degree,
	uint64_t log_len,
	bool retain_primal
) {
	PreparedPath<T> result;
	result.steps = (length - 1) / log_step;
	result.log_step = log_step;
	result.degree = degree;
	result.log_len = log_len;
	result.full_len = log_len + 1;
	const uint64_t block_batch = batch_size * result.steps;
	const uint64_t block_path_len = (log_step + 1) * dimension;
	const uint64_t block_elements = block_batch * block_path_len;
	const uint64_t signature_elements = block_batch * result.full_len;
	const uint64_t log_elements = batch_size * result.steps * log_len;

	result.blocks = CudaBuf<T>(static_cast<size_t>(block_elements) * sizeof(T));
	result.signatures = CudaBuf<T>(static_cast<size_t>(signature_elements) * sizeof(T));
	result.logs = CudaBuf<T>(static_cast<size_t>(log_elements) * sizeof(T));
	CudaBuf<T> full_logs(static_cast<size_t>(signature_elements) * sizeof(T));
	const unsigned int gather_blocks = static_cast<unsigned int>(
		(block_elements + 255) / 256 > 65535 ? 65535 : (block_elements + 255) / 256);
	gather_path_blocks_kernel<<<gather_blocks, 256>>>(
		path, result.blocks.get(), block_elements, dimension, length, result.steps, log_step);
	check_cuda_error();
	int status;
	if constexpr (std::is_same_v<T, float>)
		status = signature_cuda_f(result.blocks.get(), result.signatures.get(), block_batch,
			dimension, log_step + 1, degree, false, false, 1.f, true, true,
			nullptr, 0, 0, 0);
	else
		status = signature_cuda_d(result.blocks.get(), result.signatures.get(), block_batch,
			dimension, log_step + 1, degree, false, false, 1., true, true,
			nullptr, 0, 0, 0);
	if (status != 0)
		throw coded_runtime_error(status, "signature_cuda failed in log-PDE preprocessing");
	if constexpr (std::is_same_v<T, float>)
		status = sig_to_log_sig_cuda_f(result.signatures.get(), full_logs.get(), block_batch,
			dimension, degree, 0, true);
	else
		status = sig_to_log_sig_cuda_d(result.signatures.get(), full_logs.get(), block_batch,
			dimension, degree, 0, true);
	if (status != 0)
		throw coded_runtime_error(status,
			"sig_to_log_sig_cuda failed in log-PDE preprocessing");
	const unsigned int pack_blocks = static_cast<unsigned int>(
		(log_elements + 255) / 256 > 65535 ? 65535 : (log_elements + 255) / 256);
	pack_logs_kernel<<<pack_blocks, 256>>>(full_logs.get(),
		result.logs.get(), log_elements, result.steps, log_len);
	check_cuda_kernel_launch();
	if (!retain_primal) {
		result.blocks.reset();
		result.signatures.reset();
	}
	return result;
}

template<typename T>
void path_logs_backward(
	const PreparedPath<T>& prepared,
	const T* d_logs,
	T* d_path,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length
) {
	const uint64_t block_batch = batch_size * prepared.steps;
	const uint64_t signature_elements = block_batch * prepared.full_len;
	const uint64_t block_elements = block_batch * (prepared.log_step + 1) * dimension;
	CudaBuf<T> full_log_derivs(static_cast<size_t>(signature_elements) * sizeof(T));
	CudaBuf<T> sig_derivs(static_cast<size_t>(signature_elements) * sizeof(T));
	CudaBuf<T> block_derivs(static_cast<size_t>(block_elements) * sizeof(T));
	const unsigned int signature_blocks = static_cast<unsigned int>(
		(signature_elements + 255) / 256 > 65535 ? 65535 :
		(signature_elements + 255) / 256);
	unpack_log_derivs_kernel<<<signature_blocks, 256>>>(d_logs,
		full_log_derivs.get(), signature_elements, prepared.steps, prepared.log_len);
	check_cuda_error();
	int status;
	if constexpr (std::is_same_v<T, float>)
		status = sig_to_log_sig_backprop_cuda_f(prepared.signatures.get(), sig_derivs.get(),
			full_log_derivs.get(), block_batch, dimension, prepared.degree, 0, true);
	else
		status = sig_to_log_sig_backprop_cuda_d(prepared.signatures.get(), sig_derivs.get(),
			full_log_derivs.get(), block_batch, dimension, prepared.degree, 0, true);
	if (status != 0)
		throw coded_runtime_error(status,
			"sig_to_log_sig_backprop_cuda failed in log-PDE backpropagation");
	if constexpr (std::is_same_v<T, float>)
		status = sig_backprop_cuda_f(prepared.blocks.get(), block_derivs.get(),
			sig_derivs.get(), prepared.signatures.get(), block_batch, dimension,
			prepared.log_step + 1, prepared.degree, false, false, 1.f, true,
			nullptr, 0, 0, 0);
	else
		status = sig_backprop_cuda_d(prepared.blocks.get(), block_derivs.get(),
			sig_derivs.get(), prepared.signatures.get(), block_batch, dimension,
			prepared.log_step + 1, prepared.degree, false, false, 1., true,
			nullptr, 0, 0, 0);
	if (status != 0)
		throw coded_runtime_error(status,
			"sig_backprop_cuda failed in log-PDE backpropagation");
	const uint64_t path_elements = batch_size * length * dimension;
	const unsigned int path_blocks = static_cast<unsigned int>(
		(path_elements + 255) / 256 > 65535 ? 65535 : (path_elements + 255) / 256);
	scatter_block_derivs_kernel<<<path_blocks, 256>>>(block_derivs.get(),
		d_path, path_elements, dimension, length, prepared.steps, prepared.log_step);
	check_cuda_kernel_launch();
}

template<typename T>
void sig_kernel_log_pde_cuda_(
	const T* path_x,
	const T* path_y,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length_x,
	uint64_t length_y,
	uint64_t log_step_x,
	uint64_t log_step_y,
	uint64_t degree_x,
	uint64_t degree_y,
	uint64_t dyadic_order_x,
	uint64_t dyadic_order_y,
	bool return_grid
) {
	HostLayout layout(1, dimension, length_x, length_y, log_step_x, log_step_y,
		degree_x, degree_y, dyadic_order_x, dyadic_order_y);
	if (batch_size == 0) return;
	CudaBuf<uint64_t> device_offset(layout.offset.size() * sizeof(uint64_t));
	CUDA_CHECK(cudaMemcpy(device_offset.get(), layout.offset.data(),
		layout.offset.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
	Params base = layout.p;
	base.offset = device_offset.get();
	const ForwardWorkspaceLayout workspace_layout = forward_workspace_layout(base);
	const uint64_t elements_per_batch = workspace_layout.total +
		base.steps_x * base.x_len + base.steps_y * base.y_len +
		base.steps_x * (log_step_x + 1) * base.dimension +
		2 * base.steps_x * (base.x_len + 1) +
		base.steps_y * (log_step_y + 1) * base.dimension +
		2 * base.steps_y * (base.y_len + 1);
	size_t free_bytes = 0;
	size_t total_bytes = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
	(void)total_bytes;
	uint64_t chunk_size = static_cast<uint64_t>(free_bytes) * 7 /
		(10 * elements_per_batch * sizeof(T));
	if (chunk_size == 0) chunk_size = 1;
	if (chunk_size > batch_size) chunk_size = batch_size;

	for (uint64_t start = 0; start < batch_size; start += chunk_size) {
		const uint64_t count = batch_size - start < chunk_size ? batch_size - start : chunk_size;
		Params p = base;
		p.batch_size = count;
		const T* chunk_x = path_x + start * length_x * dimension;
		const T* chunk_y = path_y + start * length_y * dimension;
		PreparedPath<T> prepared_x = prepare_path_logs(chunk_x, count, dimension,
			length_x, log_step_x, degree_x, p.x_len, false);
		PreparedPath<T> prepared_y = prepare_path_logs(chunk_y, count, dimension,
			length_y, log_step_y, degree_y, p.y_len, false);
		const uint64_t workspace_elements = count * workspace_layout.total;
		CudaBuf<T> workspace(static_cast<size_t>(workspace_elements) * sizeof(T));
		T* chunk_out = out + start * (return_grid ? p.grid_len : 1);
		const uint64_t launch_blocks = (count + p.paths_per_block - 1) /
			p.paths_per_block;
		if (launch_blocks > UINT_MAX)
			throw std::overflow_error("log-PDE CUDA launch size overflow");
		const uint64_t shared_path_elements = 3 * workspace_layout.diagonal +
			p.path_threads * workspace_layout.scratch;
		const uint64_t shared_elements = p.paths_per_block * shared_path_elements;
		const size_t requested_shared_bytes = static_cast<size_t>(shared_elements) * sizeof(T);
		const bool use_shared_workspace = requested_shared_bytes <= 48 * 1024;
		const size_t shared_bytes = use_shared_workspace ? requested_shared_bytes : 0;
		log_pde_forward_kernel<T><<<static_cast<unsigned int>(launch_blocks),
			PATH_THREADS, shared_bytes>>>(p,
			prepared_x.logs.get(), prepared_y.logs.get(), chunk_out, workspace.get(),
			workspace_layout, use_shared_workspace, return_grid);
		check_cuda_kernel_launch();
	}
}

template<typename T>
void sig_kernel_log_pde_backprop_cuda_(
	const T* path_x,
	const T* path_y,
	T* d_path_x,
	T* d_path_y,
	const T* derivs,
	const T* k_grid,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length_x,
	uint64_t length_y,
	uint64_t log_step_x,
	uint64_t log_step_y,
	uint64_t degree_x,
	uint64_t degree_y,
	uint64_t dyadic_order_x,
	uint64_t dyadic_order_y,
	bool return_grid
) {
	HostLayout layout(1, dimension, length_x, length_y, log_step_x, log_step_y,
		degree_x, degree_y, dyadic_order_x, dyadic_order_y);
	if (batch_size == 0) return;
	CudaBuf<uint64_t> device_offset(layout.offset.size() * sizeof(uint64_t));
	CUDA_CHECK(cudaMemcpy(device_offset.get(), layout.offset.data(),
		layout.offset.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
	Params base = layout.p;
	base.offset = device_offset.get();
	const BackwardWorkspaceLayout workspace_layout = backward_workspace_layout(base);
	const uint64_t blocks_x = base.steps_x * (log_step_x + 1) * base.dimension;
	const uint64_t blocks_y = base.steps_y * (log_step_y + 1) * base.dimension;
	const uint64_t signatures_x = base.steps_x * (base.x_len + 1);
	const uint64_t signatures_y = base.steps_y * (base.y_len + 1);
	const uint64_t logs_x = base.steps_x * base.x_len;
	const uint64_t logs_y = base.steps_y * base.y_len;
	const uint64_t transient_x = blocks_x + 2 * signatures_x;
	const uint64_t transient_y = blocks_y + 2 * signatures_y;
	const uint64_t elements_per_batch = workspace_layout.total +
		2 * (logs_x + logs_y) + blocks_x + blocks_y + signatures_x + signatures_y +
		(transient_x > transient_y ? transient_x : transient_y);
	size_t free_bytes = 0;
	size_t total_bytes = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
	(void)total_bytes;
	uint64_t chunk_size = static_cast<uint64_t>(free_bytes) * 7 /
		(10 * elements_per_batch * sizeof(T));
	if (chunk_size == 0) chunk_size = 1;
	if (chunk_size > batch_size) chunk_size = batch_size;

	for (uint64_t start = 0; start < batch_size; start += chunk_size) {
		const uint64_t count = batch_size - start < chunk_size ? batch_size - start : chunk_size;
		Params p = base;
		p.batch_size = count;
		const T* chunk_x = path_x + start * length_x * dimension;
		const T* chunk_y = path_y + start * length_y * dimension;
		PreparedPath<T> prepared_x = prepare_path_logs(chunk_x, count, dimension,
			length_x, log_step_x, degree_x, p.x_len, true);
		PreparedPath<T> prepared_y = prepare_path_logs(chunk_y, count, dimension,
			length_y, log_step_y, degree_y, p.y_len, true);
		const uint64_t d_logs_x_elements = count * p.steps_x * p.x_len;
		const uint64_t d_logs_y_elements = count * p.steps_y * p.y_len;
		CudaBuf<T> d_logs_x(static_cast<size_t>(d_logs_x_elements) * sizeof(T));
		CudaBuf<T> d_logs_y(static_cast<size_t>(d_logs_y_elements) * sizeof(T));
		const uint64_t workspace_elements = count * workspace_layout.total;
		CudaBuf<T> workspace(static_cast<size_t>(workspace_elements) * sizeof(T));
		const T* chunk_derivs = derivs + start * (return_grid ? p.grid_len : 1);
		const T* chunk_grid = k_grid == nullptr ? nullptr : k_grid + start * p.grid_len;
		const uint64_t launch_blocks = (count + p.paths_per_block - 1) /
			p.paths_per_block;
		if (launch_blocks > UINT_MAX)
			throw std::overflow_error("log-PDE CUDA launch size overflow");
		log_pde_backward_kernel<T><<<static_cast<unsigned int>(launch_blocks),
			PATH_THREADS>>>(p, prepared_x.logs.get(), prepared_y.logs.get(),
			d_logs_x.get(), d_logs_y.get(), chunk_derivs, chunk_grid,
			workspace.get(), workspace_layout, return_grid);
		check_cuda_kernel_launch();
		path_logs_backward(prepared_x, d_logs_x.get(),
			d_path_x + start * length_x * dimension, count, dimension, length_x);
		path_logs_backward(prepared_y, d_logs_y.get(),
			d_path_y + start * length_y * dimension, count, dimension, length_y);
	}
}

}  // namespace log_pde_cuda_detail

extern "C" {

	CUSIG_API int sig_kernel_log_pde_cuda_f(
		const float* path_x, const float* path_y, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y, uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y, bool return_grid
	) noexcept {
		CUSIG_SAFE_CALL(log_pde_cuda_detail::sig_kernel_log_pde_cuda_<float>(
			path_x, path_y, out, batch_size, dimension, length_x, length_y,
			log_step_x, log_step_y, degree_x, degree_y,
			dyadic_order_x, dyadic_order_y, return_grid));
	}

	CUSIG_API int sig_kernel_log_pde_cuda_d(
		const double* path_x, const double* path_y, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y, uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y, bool return_grid
	) noexcept {
		CUSIG_SAFE_CALL(log_pde_cuda_detail::sig_kernel_log_pde_cuda_<double>(
			path_x, path_y, out, batch_size, dimension, length_x, length_y,
			log_step_x, log_step_y, degree_x, degree_y,
			dyadic_order_x, dyadic_order_y, return_grid));
	}

	CUSIG_API int sig_kernel_log_pde_backprop_cuda_f(
		const float* path_x, const float* path_y, float* d_path_x, float* d_path_y,
		const float* derivs, const float* k_grid, uint64_t batch_size,
		uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y, uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y, bool return_grid
	) noexcept {
		CUSIG_SAFE_CALL(log_pde_cuda_detail::sig_kernel_log_pde_backprop_cuda_<float>(
			path_x, path_y, d_path_x, d_path_y, derivs, k_grid, batch_size, dimension,
			length_x, length_y, log_step_x, log_step_y, degree_x, degree_y,
			dyadic_order_x, dyadic_order_y, return_grid));
	}

	CUSIG_API int sig_kernel_log_pde_backprop_cuda_d(
		const double* path_x, const double* path_y, double* d_path_x, double* d_path_y,
		const double* derivs, const double* k_grid, uint64_t batch_size,
		uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y, uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y, bool return_grid
	) noexcept {
		CUSIG_SAFE_CALL(log_pde_cuda_detail::sig_kernel_log_pde_backprop_cuda_<double>(
			path_x, path_y, d_path_x, d_path_y, derivs, k_grid, batch_size, dimension,
			length_x, length_y, log_step_x, log_step_y, degree_x, degree_y,
			dyadic_order_x, dyadic_order_y, return_grid));
	}
}
