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
#include "cu_utils.h"

template<typename T>
struct SigPolyTableCacheEntry {
	int device;
	uint64_t order;
	T* data;
};

static std::mutex sig_poly_table_cache_mutex_;

template<typename T>
static std::vector<SigPolyTableCacheEntry<T>>& sig_poly_table_cache_() {
	static std::vector<SigPolyTableCacheEntry<T>> cache;
	return cache;
}

template<typename T>
static const T* get_sig_poly_tables_(uint64_t order) {
	int device = 0;
	CUDA_CHECK(cudaGetDevice(&device));
	std::lock_guard<std::mutex> lock(sig_poly_table_cache_mutex_);
	auto& cache = sig_poly_table_cache_<T>();
	for (const auto& entry : cache) {
		if (entry.device == device && entry.order == order)
			return entry.data;
	}

	const uint64_t size = order + 1;
	const uint64_t matrix_size = size * size;
	std::vector<T> host_tables(2 * matrix_size, static_cast<T>(0));
	T* const mat1 = host_tables.data();
	T* const mat2 = mat1 + matrix_size;

	long double inverse_factorial = 1.0L;
	for (uint64_t n = 1; n < size; ++n) {
		inverse_factorial /= static_cast<long double>(n);
		long double value = inverse_factorial * inverse_factorial;
		mat2[n * size] = static_cast<T>(value);
		for (uint64_t k = 1; k < size; ++k) {
			value *= static_cast<long double>(k) / static_cast<long double>(n + k);
			mat2[n * size + k] = static_cast<T>(value);
		}
	}

	long double inverse_nm1_factorial = 1.0L;
	for (uint64_t n = 2; n < size; ++n) {
		const long double inverse_n_factorial =
			inverse_nm1_factorial / static_cast<long double>(n);
		long double value = inverse_n_factorial * inverse_nm1_factorial;
		mat1[n * size + 1] = static_cast<T>(value);
		for (uint64_t k = 2; k < n; ++k) {
			value *= static_cast<long double>(k * (n - k + 1));
			mat1[n * size + k] = static_cast<T>(value);
		}
		inverse_nm1_factorial = inverse_n_factorial;
	}

	CudaBuf<T> device_tables(host_tables.size() * sizeof(T));
	CUDA_CHECK(cudaMemcpy(device_tables.get(), host_tables.data(),
		host_tables.size() * sizeof(T), cudaMemcpyHostToDevice));
	T* const result = device_tables.release();
	cache.push_back({device, order, result});
	return result;
}

template<typename T>
static void release_sig_poly_tables_() {
	auto& cache = sig_poly_table_cache_<T>();
	for (const auto& entry : cache) {
		(void)cudaSetDevice(entry.device);
		(void)cudaFree(entry.data);
	}
	cache.clear();
}

void release_sig_kernel_poly_state() {
	std::lock_guard<std::mutex> lock(sig_poly_table_cache_mutex_);
	int original_device = 0;
	const bool restore_device = cudaGetDevice(&original_device) == cudaSuccess;
	release_sig_poly_tables_<float>();
	release_sig_poly_tables_<double>();
	if (restore_device)
		(void)cudaSetDevice(original_device);
}

template<typename T>
__global__ void sig_poly_fill_kernel_(T* data, T value, uint64_t count) {
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < count; index += static_cast<uint64_t>(blockDim.x) * gridDim.x)
		data[index] = value;
}

template<typename T>
__global__ void sig_poly_init_frontiers_kernel_(
	T* work,
	uint64_t boundary_tiles,
	uint64_t slots,
	uint64_t tile_width
) {
	const uint64_t count = boundary_tiles * slots * tile_width;
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < count; index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
		const uint64_t coefficient = (index / tile_width) % slots;
		work[index] = coefficient == 0 || coefficient + 1 == slots
			? static_cast<T>(1) : static_cast<T>(0);
	}
}

template<typename T, uint64_t FixedSize = 0>
__global__ void sig_poly_tile_forward_kernel_(
	const T* __restrict__ gram,
	T* __restrict__ out,
	T* __restrict__ state,
	T* __restrict__ work,
	const T* __restrict__ tables,
	uint64_t rows,
	uint64_t cols,
	uint64_t length1,
	uint64_t length2,
	uint64_t size_,
	uint64_t tile_width,
	uint64_t tile_rows,
	uint64_t tile_cols,
	uint64_t tile_diagonal,
	bool return_grid,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch = cuda_batch_index();
	if (local_batch >= batch_chunk_size)
		return;
	const uint64_t batch = batch_offset + local_batch;
	const uint64_t first_tile_row = tile_diagonal >= tile_cols
		? tile_diagonal - tile_cols + 1 : 0;
	const uint64_t tile_row = first_tile_row + blockIdx.x;
	const uint64_t tile_col = tile_diagonal - tile_row;
	if (tile_row >= tile_rows || tile_col >= tile_cols)
		return;

	const uint64_t size = FixedSize == 0 ? size_ : FixedSize;
	const uint64_t slots = size + 1;
	const uint64_t boundary_tile_stride = slots * tile_width;
	const uint64_t work_batch_stride =
		(tile_rows + tile_cols) * boundary_tile_stride;
	T* const batch_work = work + batch * work_batch_stride;
	T* const horizontal_global = batch_work + tile_col * boundary_tile_stride;
	T* const vertical_global = batch_work
		+ tile_cols * boundary_tile_stride + tile_row * boundary_tile_stride;

	extern __shared__ char shared_bytes[];
	T* const bottom = reinterpret_cast<T*>(shared_bytes);
	T* const left = bottom + boundary_tile_stride;
	T* const top = left + boundary_tile_stride;
	T* const right = top + boundary_tile_stride;
	if constexpr (FixedSize != 0)
		(void)right;
	const uint64_t lane = threadIdx.x;

	if (lane < tile_width) {
		for (uint64_t k = 0; k < slots; ++k) {
			bottom[k * tile_width + lane] = horizontal_global[k * tile_width + lane];
			left[k * tile_width + lane] = vertical_global[k * tile_width + lane];
		}
	}
	__syncwarp();

	const uint64_t row_start = tile_row * tile_width;
	const uint64_t col_start = tile_col * tile_width;
	const uint64_t active_rows = min(tile_width, rows - row_start);
	const uint64_t active_cols = min(tile_width, cols - col_start);
	const T* const mat1 = tables;
	const T* const mat2 = tables + size * size;
	const T* const batch_gram = gram + batch * rows * cols;
	T* const batch_out = out + batch * (return_grid ? length1 * length2 : 1);

	for (uint64_t diagonal = 0;
		diagonal < active_rows + active_cols - 1; ++diagonal) {
		const uint64_t first_row = diagonal >= active_cols
			? diagonal - active_cols + 1 : 0;
		const uint64_t last_row = min(diagonal, active_rows - 1);
		const uint64_t cells = last_row - first_row + 1;

		if (lane < cells) {
			const uint64_t local_row = first_row + lane;
			const uint64_t local_col = diagonal - local_row;
			const uint64_t global_row = row_start + local_row;
			const uint64_t global_col = col_start + local_col;
			const T rho = batch_gram[global_row * cols + global_col];
			T bottom_local[FixedSize == 0 ? 1 : FixedSize];
			T left_local[FixedSize == 0 ? 1 : FixedSize];
			T top_local[FixedSize == 0 ? 1 : FixedSize];
			T right_local[FixedSize == 0 ? 1 : FixedSize];
			if constexpr (FixedSize == 0) {
				(void)bottom_local;
				(void)left_local;
				(void)top_local;
				(void)right_local;
			}
			if constexpr (FixedSize != 0) {
				for (uint64_t k = 0; k < size; ++k) {
					bottom_local[k] = bottom[k * tile_width + local_col];
					left_local[k] = left[k * tile_width + local_row];
				}
			}
			if (state) {
				T* const cell_state = state
					+ 2 * (batch * rows * cols + global_row * cols + global_col) * size;
				for (uint64_t k = 0; k < size; ++k) {
					if constexpr (FixedSize == 0) {
						cell_state[k] = bottom[k * tile_width + local_col];
						cell_state[size + k] = left[k * tile_width + local_row];
					}
					else {
						cell_state[k] = bottom_local[k];
						cell_state[size + k] = left_local[k];
					}
				}
			}

			const T bottom_endpoint = bottom[size * tile_width + local_col];
			const T left_endpoint = left[size * tile_width + local_row];
			if constexpr (FixedSize == 0) {
				top[lane] = left_endpoint;
				right[lane] = bottom_endpoint;
			}
			else {
				top_local[0] = left_endpoint;
				right_local[0] = bottom_endpoint;
			}
			T top_endpoint = left_endpoint;
			T right_endpoint = bottom_endpoint;
			T rho_n = static_cast<T>(1);

			for (uint64_t n = 1; n < size; ++n) {
				rho_n *= rho;
				T top_opposite = static_cast<T>(0);
				T right_opposite = static_cast<T>(0);
				const T* const mat2_row = mat2 + n * size;
				for (uint64_t k = 0; k < size; ++k) {
					const T coefficient = mat2_row[k];
					if constexpr (FixedSize == 0) {
						top_opposite += coefficient * left[k * tile_width + local_row];
						right_opposite += coefficient * bottom[k * tile_width + local_col];
					}
					else {
						top_opposite += coefficient * left_local[k];
						right_opposite += coefficient * bottom_local[k];
					}
				}

				T top_same;
				T right_same;
				if constexpr (FixedSize == 0) {
					top_same = bottom[n * tile_width + local_col];
					right_same = left[n * tile_width + local_row];
				}
				else {
					top_same = bottom_local[n];
					right_same = left_local[n];
				}
				T rho_power = rho;
				const T* const mat1_row = mat1 + n * size;
				for (uint64_t k = n - 1; k > 0; --k) {
					const T factor = mat1_row[k] * rho_power;
					if constexpr (FixedSize == 0) {
						top_same += factor * bottom[k * tile_width + local_col];
						right_same += factor * left[k * tile_width + local_row];
					}
					else {
						top_same += factor * bottom_local[k];
						right_same += factor * left_local[k];
					}
					rho_power *= rho;
				}

				const T top_value = top_same + rho_n * top_opposite;
				const T right_value = right_same + rho_n * right_opposite;
				if constexpr (FixedSize == 0) {
					top[n * tile_width + lane] = top_value;
					right[n * tile_width + lane] = right_value;
				}
				else {
					top_local[n] = top_value;
					right_local[n] = right_value;
				}
				top_endpoint += top_value;
				right_endpoint += right_value;
			}

			if constexpr (FixedSize == 0) {
				top[size * tile_width + lane] = top_endpoint;
				right[size * tile_width + lane] = right_endpoint;
			}
			else {
				for (uint64_t k = 0; k < size; ++k) {
					bottom[k * tile_width + local_col] = top_local[k];
					left[k * tile_width + local_row] = right_local[k];
				}
				bottom[size * tile_width + local_col] = top_endpoint;
				left[size * tile_width + local_row] = right_endpoint;
			}
			const T value = static_cast<T>(0.5) * (top_endpoint + right_endpoint);
			if (return_grid)
				batch_out[(global_row + 1) * length2 + global_col + 1] = value;
			else if (global_row + 1 == rows && global_col + 1 == cols)
				batch_out[0] = value;
		}

		if constexpr (FixedSize == 0) {
			__syncwarp();
			if (lane < cells) {
				const uint64_t local_row = first_row + lane;
				const uint64_t local_col = diagonal - local_row;
				for (uint64_t k = 0; k < slots; ++k) {
					bottom[k * tile_width + local_col] = top[k * tile_width + lane];
					left[k * tile_width + local_row] = right[k * tile_width + lane];
				}
			}
		}
		__syncwarp();
	}

	if (lane < tile_width) {
		for (uint64_t k = 0; k < slots; ++k) {
			horizontal_global[k * tile_width + lane] = bottom[k * tile_width + lane];
			vertical_global[k * tile_width + lane] = left[k * tile_width + lane];
		}
	}
}

template<typename T, uint64_t FixedSize = 0>
static void launch_sig_poly_tiles_(
	const T* gram,
	T* out,
	T* state,
	T* work,
	const T* tables,
	uint64_t batch_size,
	uint64_t length1,
	uint64_t length2,
	uint64_t size,
	uint64_t tile_width,
	uint64_t tile_rows,
	uint64_t tile_cols,
	bool return_grid,
	const CudaSharedMemoryLimits& smem_limits
) {
	const uint64_t frontier_count = FixedSize == 0 ? 4 : 2;
	const size_t smem = static_cast<size_t>(
		frontier_count * (size + 1) * tile_width) * sizeof(T);
	configure_dynamic_smem(
		sig_poly_tile_forward_kernel_<T, FixedSize>, smem,
		"CUDA polynomial sig kernel", smem_limits);
	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	for (uint64_t diagonal = 0; diagonal < tile_rows + tile_cols - 1; ++diagonal) {
		const uint64_t first_row = diagonal >= tile_cols
			? diagonal - tile_cols + 1 : 0;
		const uint64_t last_row = min(diagonal, tile_rows - 1);
		const uint64_t tiles = last_row - first_row + 1;
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				tiles, batch_size, batch_offset);
			sig_poly_tile_forward_kernel_<T, FixedSize><<<
				batch_chunk.grid, 32U, smem>>>(
					gram, out, state, work, tables, rows, cols, length1, length2,
					size, tile_width, tile_rows, tile_cols, diagonal,
					return_grid, batch_chunk.offset, batch_chunk.size);
			batch_offset += batch_chunk.size;
		}
	}
}

template<typename T, uint64_t FixedSize = 0>
__global__ void sig_poly_tile_backprop_kernel_(
	const T* __restrict__ gram,
	T* __restrict__ gram_derivs,
	const T* __restrict__ output_derivs,
	const T* __restrict__ state,
	T* __restrict__ work,
	const T* __restrict__ tables,
	uint64_t rows,
	uint64_t cols,
	uint64_t length1,
	uint64_t length2,
	uint64_t size_,
	uint64_t tile_width,
	uint64_t tile_rows,
	uint64_t tile_cols,
	uint64_t tile_diagonal,
	bool return_grid,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch = cuda_batch_index();
	if (local_batch >= batch_chunk_size)
		return;
	const uint64_t batch = batch_offset + local_batch;
	const uint64_t first_tile_row = tile_diagonal >= tile_cols
		? tile_diagonal - tile_cols + 1 : 0;
	const uint64_t tile_row = first_tile_row + blockIdx.x;
	const uint64_t tile_col = tile_diagonal - tile_row;
	if (tile_row >= tile_rows || tile_col >= tile_cols)
		return;

	const uint64_t size = FixedSize == 0 ? size_ : FixedSize;
	const uint64_t boundary_tile_stride = size * tile_width;
	const uint64_t work_batch_stride =
		(tile_rows + tile_cols) * boundary_tile_stride;
	T* const batch_work = work + batch * work_batch_stride;
	T* const horizontal_global = batch_work + tile_col * boundary_tile_stride;
	T* const vertical_global = batch_work
		+ tile_cols * boundary_tile_stride + tile_row * boundary_tile_stride;

	extern __shared__ char shared_bytes[];
	T* const top_adjoints = reinterpret_cast<T*>(shared_bytes);
	T* const right_adjoints = top_adjoints + boundary_tile_stride;
	T* const bottom_outputs = right_adjoints + boundary_tile_stride;
	T* const left_outputs = bottom_outputs + boundary_tile_stride;
	if constexpr (FixedSize != 0) {
		(void)bottom_outputs;
		(void)left_outputs;
	}
	const uint64_t lane = threadIdx.x;
	if (lane < tile_width) {
		for (uint64_t k = 0; k < size; ++k) {
			top_adjoints[k * tile_width + lane] =
				horizontal_global[k * tile_width + lane];
			right_adjoints[k * tile_width + lane] =
				vertical_global[k * tile_width + lane];
		}
	}
	__syncwarp();

	const uint64_t row_start = tile_row * tile_width;
	const uint64_t col_start = tile_col * tile_width;
	const uint64_t active_rows = min(tile_width, rows - row_start);
	const uint64_t active_cols = min(tile_width, cols - col_start);
	const T* const mat1 = tables;
	const T* const mat2 = tables + size * size;
	const T* const batch_gram = gram + batch * rows * cols;
	T* const batch_gram_derivs = gram_derivs + batch * rows * cols;
	const T* const batch_output_derivs = output_derivs
		+ batch * (return_grid ? length1 * length2 : 1);
	const T* const batch_state = state + 2 * batch * rows * cols * size;

	for (uint64_t diagonal = active_rows + active_cols - 1; diagonal-- > 0;) {
		const uint64_t first_row = diagonal >= active_cols
			? diagonal - active_cols + 1 : 0;
		const uint64_t last_row = min(diagonal, active_rows - 1);
		const uint64_t cells = last_row - first_row + 1;

		if (lane < cells) {
			const uint64_t local_row = first_row + lane;
			const uint64_t local_col = diagonal - local_row;
			const uint64_t global_row = row_start + local_row;
			const uint64_t global_col = col_start + local_col;
			const uint64_t cell = global_row * cols + global_col;
			const T rho = batch_gram[cell];
			const T* const cell_state = batch_state + 2 * cell * size;
			const T* const bottom_state = cell_state;
			const T* const left_state = cell_state + size;
			T bottom_local[FixedSize == 0 ? 1 : FixedSize];
			T left_local[FixedSize == 0 ? 1 : FixedSize];
			T top_local[FixedSize == 0 ? 1 : FixedSize];
			T right_local[FixedSize == 0 ? 1 : FixedSize];
			T bottom_deriv_local[FixedSize == 0 ? 1 : FixedSize];
			T left_deriv_local[FixedSize == 0 ? 1 : FixedSize];
			if constexpr (FixedSize == 0) {
				(void)bottom_local;
				(void)left_local;
				(void)top_local;
				(void)right_local;
				(void)bottom_deriv_local;
				(void)left_deriv_local;
			}

			T local_deriv = static_cast<T>(0);
			if (return_grid) {
				local_deriv = static_cast<T>(0.5)
					* batch_output_derivs[(global_row + 1) * length2 + global_col + 1];
			}
			else if (global_row + 1 == rows && global_col + 1 == cols) {
				local_deriv = static_cast<T>(0.5) * batch_output_derivs[0];
			}

			if constexpr (FixedSize != 0) {
				for (uint64_t k = 0; k < size; ++k) {
					bottom_local[k] = bottom_state[k];
					left_local[k] = left_state[k];
					top_local[k] = top_adjoints[k * tile_width + local_col] + local_deriv;
					right_local[k] = right_adjoints[k * tile_width + local_row] + local_deriv;
					bottom_deriv_local[k] = right_local[0];
					left_deriv_local[k] = top_local[0];
				}
			}
			else {
				const T top_zero = top_adjoints[local_col] + local_deriv;
				const T right_zero = right_adjoints[local_row] + local_deriv;
				for (uint64_t k = 0; k < size; ++k) {
					bottom_outputs[k * tile_width + lane] = right_zero;
					left_outputs[k * tile_width + lane] = top_zero;
				}
			}

			T rho_deriv = static_cast<T>(0);
			T rho_n = static_cast<T>(1);
			for (uint64_t n = 1; n < size; ++n) {
				const T rho_nm1 = rho_n;
				rho_n *= rho;
				const T top_deriv = FixedSize == 0
					? top_adjoints[n * tile_width + local_col] + local_deriv
					: top_local[n];
				const T right_deriv = FixedSize == 0
					? right_adjoints[n * tile_width + local_row] + local_deriv
					: right_local[n];
				if constexpr (FixedSize == 0) {
					bottom_outputs[n * tile_width + lane] += top_deriv;
					left_outputs[n * tile_width + lane] += right_deriv;
				}
				else {
					bottom_deriv_local[n] += top_deriv;
					left_deriv_local[n] += right_deriv;
				}

				T rho_power = rho;
				T rho_power_deriv = static_cast<T>(1);
				const T* const mat1_row = mat1 + n * size;
				for (uint64_t k = n - 1; k > 0; --k) {
					const T coefficient = mat1_row[k];
					const T factor = coefficient * rho_power;
					const T bottom_value = FixedSize == 0 ? bottom_state[k] : bottom_local[k];
					const T left_value = FixedSize == 0 ? left_state[k] : left_local[k];
					if constexpr (FixedSize == 0) {
						bottom_outputs[k * tile_width + lane] += factor * top_deriv;
						left_outputs[k * tile_width + lane] += factor * right_deriv;
					}
					else {
						bottom_deriv_local[k] += factor * top_deriv;
						left_deriv_local[k] += factor * right_deriv;
					}
					rho_deriv += static_cast<T>(n - k) * coefficient * rho_power_deriv
						* (bottom_value * top_deriv + left_value * right_deriv);
					rho_power *= rho;
					rho_power_deriv *= rho;
				}

				T top_opposite = static_cast<T>(0);
				T right_opposite = static_cast<T>(0);
				const T* const mat2_row = mat2 + n * size;
				for (uint64_t k = 0; k < size; ++k) {
					const T coefficient = mat2_row[k];
					const T factor = rho_n * coefficient;
					const T bottom_value = FixedSize == 0 ? bottom_state[k] : bottom_local[k];
					const T left_value = FixedSize == 0 ? left_state[k] : left_local[k];
					top_opposite += coefficient * left_value;
					right_opposite += coefficient * bottom_value;
					if constexpr (FixedSize == 0) {
						left_outputs[k * tile_width + lane] += factor * top_deriv;
						bottom_outputs[k * tile_width + lane] += factor * right_deriv;
					}
					else {
						left_deriv_local[k] += factor * top_deriv;
						bottom_deriv_local[k] += factor * right_deriv;
					}
				}
				rho_deriv += static_cast<T>(n) * rho_nm1
					* (top_opposite * top_deriv + right_opposite * right_deriv);
			}
			batch_gram_derivs[cell] = rho_deriv;

			if constexpr (FixedSize != 0) {
				for (uint64_t k = 0; k < size; ++k) {
					top_adjoints[k * tile_width + local_col] = bottom_deriv_local[k];
					right_adjoints[k * tile_width + local_row] = left_deriv_local[k];
				}
			}
		}

		if constexpr (FixedSize == 0) {
			__syncwarp();
			if (lane < cells) {
				const uint64_t local_row = first_row + lane;
				const uint64_t local_col = diagonal - local_row;
				for (uint64_t k = 0; k < size; ++k) {
					top_adjoints[k * tile_width + local_col] =
						bottom_outputs[k * tile_width + lane];
					right_adjoints[k * tile_width + local_row] =
						left_outputs[k * tile_width + lane];
				}
			}
		}
		__syncwarp();
	}

	if (lane < tile_width) {
		for (uint64_t k = 0; k < size; ++k) {
			horizontal_global[k * tile_width + lane] =
				top_adjoints[k * tile_width + lane];
			vertical_global[k * tile_width + lane] =
				right_adjoints[k * tile_width + lane];
		}
	}
}

template<typename T, uint64_t FixedSize = 0>
static void launch_sig_poly_backprop_tiles_(
	const T* gram,
	T* gram_derivs,
	const T* output_derivs,
	const T* state,
	T* work,
	const T* tables,
	uint64_t batch_size,
	uint64_t length1,
	uint64_t length2,
	uint64_t size,
	uint64_t tile_width,
	uint64_t tile_rows,
	uint64_t tile_cols,
	bool return_grid,
	const CudaSharedMemoryLimits& smem_limits
) {
	const uint64_t frontier_count = FixedSize == 0 ? 4 : 2;
	const size_t smem = static_cast<size_t>(
		frontier_count * size * tile_width) * sizeof(T);
	configure_dynamic_smem(
		sig_poly_tile_backprop_kernel_<T, FixedSize>, smem,
		"CUDA polynomial sig kernel backprop", smem_limits);
	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	for (uint64_t diagonal = tile_rows + tile_cols - 1; diagonal-- > 0;) {
		const uint64_t first_row = diagonal >= tile_cols
			? diagonal - tile_cols + 1 : 0;
		const uint64_t last_row = min(diagonal, tile_rows - 1);
		const uint64_t tiles = last_row - first_row + 1;
		for (uint64_t batch_offset = 0; batch_offset < batch_size;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				tiles, batch_size, batch_offset);
			sig_poly_tile_backprop_kernel_<T, FixedSize><<<
				batch_chunk.grid, 32U, smem>>>(
					gram, gram_derivs, output_derivs, state, work, tables,
					rows, cols, length1, length2, size, tile_width,
					tile_rows, tile_cols, diagonal, return_grid,
					batch_chunk.offset, batch_chunk.size);
			batch_offset += batch_chunk.size;
		}
	}
}

template<typename T>
static void sig_kernel_poly_cuda_(
	const T* gram,
	T* out,
	T* state,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t order,
	bool return_grid
) {
	if (dimension == 0)
		throw std::invalid_argument("sig_kernel_poly_cuda: path dimension must be positive");
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("sig_kernel_poly_cuda: paths must have length >= 1");
	if (order < 2 || order > 64)
		throw std::invalid_argument("sig_kernel_poly_cuda: order must be between 2 and 64");
	if (batch_size == 0)
		return;
	if (!out)
		throw std::invalid_argument("sig_kernel_poly_cuda: out must not be null");

	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	const uint64_t result_size = return_grid ? length1 * length2 : 1;
	if (rows == 0 || cols == 0) {
		const uint64_t count = batch_size * result_size;
		sig_poly_fill_kernel_<<<make_cuda_1d_grid(count, 256), 256U>>>(
			out, static_cast<T>(1), count);
		check_cuda_kernel_launch();
		return;
	}
	if (!gram)
		throw std::invalid_argument("sig_kernel_poly_cuda: gram must not be null");

	const uint64_t size = order + 1;
	const uint64_t slots = size + 1;
	const auto smem_limits = cuda_shared_memory_limits();
	uint64_t tile_width = 32;
	while (tile_width > 1
		&& static_cast<size_t>(4 * slots * tile_width) * sizeof(T) > smem_limits.optin_bytes)
		tile_width /= 2;
	if (static_cast<size_t>(4 * slots * tile_width) * sizeof(T) > smem_limits.optin_bytes)
		throw std::invalid_argument(
			"sig_kernel_poly_cuda: insufficient shared memory for this order");

	const uint64_t tile_rows = (rows + tile_width - 1) / tile_width;
	const uint64_t tile_cols = (cols + tile_width - 1) / tile_width;
	const uint64_t boundary_tiles = batch_size * (tile_rows + tile_cols);
	const uint64_t work_size = boundary_tiles * slots * tile_width;
	CudaBuf<T> work(work_size * sizeof(T));
	sig_poly_init_frontiers_kernel_<<<
		make_cuda_1d_grid(work_size, 256), 256U>>>(
			work.get(), boundary_tiles, slots, tile_width);
	if (return_grid) {
		const uint64_t count = batch_size * result_size;
		sig_poly_fill_kernel_<<<make_cuda_1d_grid(count, 256), 256U>>>(
			out, static_cast<T>(1), count);
	}

	const T* const tables = get_sig_poly_tables_<T>(order);
#define LAUNCH_SIG_POLY_FIXED(size_value) \
	launch_sig_poly_tiles_<T, size_value>(gram, out, state, work.get(), tables, batch_size, \
		length1, length2, size, tile_width, tile_rows, tile_cols, return_grid, smem_limits)
	switch (size) {
	case 3: LAUNCH_SIG_POLY_FIXED(3); break;
	case 4: LAUNCH_SIG_POLY_FIXED(4); break;
	case 5: LAUNCH_SIG_POLY_FIXED(5); break;
	case 6: LAUNCH_SIG_POLY_FIXED(6); break;
	case 7: LAUNCH_SIG_POLY_FIXED(7); break;
	case 8: LAUNCH_SIG_POLY_FIXED(8); break;
	case 9: LAUNCH_SIG_POLY_FIXED(9); break;
	case 10: LAUNCH_SIG_POLY_FIXED(10); break;
	case 11: LAUNCH_SIG_POLY_FIXED(11); break;
	case 12: LAUNCH_SIG_POLY_FIXED(12); break;
	case 13: LAUNCH_SIG_POLY_FIXED(13); break;
	default:
		launch_sig_poly_tiles_<T>(gram, out, state, work.get(), tables, batch_size,
			length1, length2, size, tile_width, tile_rows, tile_cols,
			return_grid, smem_limits);
	}
#undef LAUNCH_SIG_POLY_FIXED
	check_cuda_kernel_launch();
}

template<typename T>
static void sig_kernel_poly_backprop_cuda_(
	const T* gram,
	T* gram_derivs,
	const T* output_derivs,
	const T* state,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t order,
	bool return_grid
) {
	if (dimension == 0)
		throw std::invalid_argument(
			"sig_kernel_poly_backprop_cuda: path dimension must be positive");
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument(
			"sig_kernel_poly_backprop_cuda: paths must have length >= 1");
	if (order < 2 || order > 64)
		throw std::invalid_argument(
			"sig_kernel_poly_backprop_cuda: order must be between 2 and 64");
	if (batch_size == 0 || length1 == 1 || length2 == 1)
		return;
	if (!gram)
		throw std::invalid_argument(
			"sig_kernel_poly_backprop_cuda: gram must not be null");
	if (!gram_derivs)
		throw std::invalid_argument(
			"sig_kernel_poly_backprop_cuda: gram_derivs must not be null");
	if (!output_derivs)
		throw std::invalid_argument(
			"sig_kernel_poly_backprop_cuda: output_derivs must not be null");

	const uint64_t rows = length1 - 1;
	const uint64_t cols = length2 - 1;
	const uint64_t size = order + 1;
	const size_t max_elements = std::numeric_limits<size_t>::max() / sizeof(T);
	if (cols > max_elements / rows)
		throw std::overflow_error(
			"sig_kernel_poly_backprop_cuda: grid size overflow");
	const uint64_t cells = rows * cols;
	if (cells > max_elements / (2 * size))
		throw std::overflow_error(
			"sig_kernel_poly_backprop_cuda: state size overflow");
	const size_t state_batch_size = static_cast<size_t>(2 * cells * size);
	if (batch_size > max_elements / state_batch_size)
		throw std::overflow_error(
			"sig_kernel_poly_backprop_cuda: batch state size overflow");

	CudaBuf<T> regenerated_state;
	CudaBuf<T> ignored_output;
	if (!state) {
		regenerated_state = CudaBuf<T>(
			static_cast<size_t>(batch_size) * state_batch_size * sizeof(T));
		ignored_output = CudaBuf<T>(static_cast<size_t>(batch_size) * sizeof(T));
		sig_kernel_poly_cuda_(
			gram, ignored_output.get(), regenerated_state.get(), batch_size,
			dimension, length1, length2, order, false);
		state = regenerated_state.get();
	}

	const auto smem_limits = cuda_shared_memory_limits();
	uint64_t tile_width = 32;
	while (tile_width > 1
		&& static_cast<size_t>(4 * size * tile_width) * sizeof(T)
			> smem_limits.optin_bytes)
		tile_width /= 2;
	if (static_cast<size_t>(4 * size * tile_width) * sizeof(T)
		> smem_limits.optin_bytes)
		throw std::invalid_argument(
			"sig_kernel_poly_backprop_cuda: insufficient shared memory for this order");

	const uint64_t tile_rows = (rows + tile_width - 1) / tile_width;
	const uint64_t tile_cols = (cols + tile_width - 1) / tile_width;
	const uint64_t work_size = batch_size * (tile_rows + tile_cols)
		* size * tile_width;
	CudaBuf<T> work(static_cast<size_t>(work_size) * sizeof(T));
	CUDA_CHECK(cudaMemset(work.get(), 0,
		static_cast<size_t>(work_size) * sizeof(T)));

	const T* const tables = get_sig_poly_tables_<T>(order);
#define LAUNCH_SIG_POLY_BACKPROP_FIXED(size_value) \
	launch_sig_poly_backprop_tiles_<T, size_value>(gram, gram_derivs, output_derivs, \
		state, work.get(), tables, batch_size, length1, length2, size, tile_width, \
		tile_rows, tile_cols, return_grid, smem_limits)
	switch (size) {
	case 3: LAUNCH_SIG_POLY_BACKPROP_FIXED(3); break;
	case 4: LAUNCH_SIG_POLY_BACKPROP_FIXED(4); break;
	case 5: LAUNCH_SIG_POLY_BACKPROP_FIXED(5); break;
	case 6: LAUNCH_SIG_POLY_BACKPROP_FIXED(6); break;
	case 7: LAUNCH_SIG_POLY_BACKPROP_FIXED(7); break;
	case 8: LAUNCH_SIG_POLY_BACKPROP_FIXED(8); break;
	case 9: LAUNCH_SIG_POLY_BACKPROP_FIXED(9); break;
	case 10: LAUNCH_SIG_POLY_BACKPROP_FIXED(10); break;
	case 11: LAUNCH_SIG_POLY_BACKPROP_FIXED(11); break;
	case 12: LAUNCH_SIG_POLY_BACKPROP_FIXED(12); break;
	case 13: LAUNCH_SIG_POLY_BACKPROP_FIXED(13); break;
	default:
		launch_sig_poly_backprop_tiles_<T>(
			gram, gram_derivs, output_derivs, state, work.get(), tables,
			batch_size, length1, length2, size, tile_width, tile_rows,
			tile_cols, return_grid, smem_limits);
	}
#undef LAUNCH_SIG_POLY_BACKPROP_FIXED
	check_cuda_kernel_launch();
}

#include "cu_macros.h"

extern "C" {

	CUSIG_API int sig_kernel_poly_cuda_f(
		const float* gram, float* out, float* state,
		uint64_t batch_size, uint64_t dimension,
		uint64_t length1, uint64_t length2, uint64_t order, bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_poly_cuda_<float>(
			gram, out, state, batch_size, dimension, length1, length2, order, return_grid));
	}

	CUSIG_API int sig_kernel_poly_cuda_d(
		const double* gram, double* out, double* state,
		uint64_t batch_size, uint64_t dimension,
		uint64_t length1, uint64_t length2, uint64_t order, bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_poly_cuda_<double>(
			gram, out, state, batch_size, dimension, length1, length2, order, return_grid));
	}

	CUSIG_API int sig_kernel_poly_backprop_cuda_f(
		const float* gram, float* gram_derivs, const float* output_derivs,
		const float* state, uint64_t batch_size, uint64_t dimension,
		uint64_t length1, uint64_t length2, uint64_t order,
		bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_poly_backprop_cuda_<float>(
			gram, gram_derivs, output_derivs, state, batch_size, dimension,
			length1, length2, order, return_grid));
	}

	CUSIG_API int sig_kernel_poly_backprop_cuda_d(
		const double* gram, double* gram_derivs, const double* output_derivs,
		const double* state, uint64_t batch_size, uint64_t dimension,
		uint64_t length1, uint64_t length2, uint64_t order,
		bool return_grid) noexcept {
		CUSIG_SAFE_CALL(sig_kernel_poly_backprop_cuda_<double>(
			gram, gram_derivs, output_derivs, state, batch_size, dimension,
			length1, length2, order, return_grid));
	}
}
