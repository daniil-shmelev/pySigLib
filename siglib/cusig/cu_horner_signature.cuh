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

#include "cu_runtime_utils.h"
#include "cu_path_reduction.h"
#include "cu_utils.h"
#include <utility>

namespace {
#define HORNER_INLINE __device__ __forceinline__
#define HORNER_LARGE HORNER_INLINE
#include "horner_signature.h"
#include "horner_path.h"
#undef HORNER_INLINE
#undef HORNER_LARGE

namespace horner_path {

inline constexpr uint64_t cuda_chunk_steps = 128;

template<typename T>
struct StridedWorkspace {
	T* data;
	uint64_t stride;
	__device__ __forceinline__ T& operator[](uint64_t index) const { return data[index * stride]; }
	__device__ __forceinline__ StridedWorkspace operator+(uint64_t index) const {
		return {data + index * stride, stride};
	}
};

template<typename T>
struct IncrementGradient {
	T* data;
	uint64_t dimension;
	__device__ __forceinline__ void operator()(uint64_t step, int letter, T value) const {
		data[step * dimension + letter] = value;
	}
};

template<int Dimension, typename T>
__device__ __forceinline__ T signature_tree_product(const T* tree, uint64_t nodes,
	uint64_t left, uint64_t right, uint64_t coefficient) {
	uint64_t word = coefficient;
	int level = 1;
	uint64_t width = Dimension;
	while (word >= width) { word -= width; width *= Dimension; ++level; }
	T value = tree[coefficient * nodes + left] + tree[coefficient * nodes + right];
	uint64_t left_width = Dimension;
	uint64_t right_width = width / Dimension;
	for (int split = 1; split < level; ++split) {
		const uint64_t left_offset = (left_width - Dimension) / (Dimension - 1);
		const uint64_t right_offset = (right_width - Dimension) / (Dimension - 1);
		value += tree[(left_offset + word / right_width) * nodes + left]
			* tree[(right_offset + word % right_width) * nodes + right];
		left_width *= Dimension;
		right_width /= Dimension;
	}
	return value;
}

template<int Dimension, int Degree, typename T>
__device__ __forceinline__ void signature_tree_pullback(const T* tree, T* gradient,
	uint64_t nodes, uint64_t base, uint64_t parent, uint64_t coefficient) {
	const uint64_t left = base + 2 * parent;
	const uint64_t right = left + 1;
	uint64_t word = coefficient;
	int level = 1;
	uint64_t width = Dimension;
	while (word >= width) { word -= width; width *= Dimension; ++level; }
	T left_gradient = gradient[coefficient * nodes + base + parent];
	T right_gradient = left_gradient;
	uint64_t extra_width = Dimension;
	for (int extra = 1; extra <= Degree - level; ++extra) {
		const uint64_t extra_offset = (extra_width - Dimension) / (Dimension - 1);
		const uint64_t offset = (width * extra_width - Dimension) / (Dimension - 1);
		for (uint64_t j = 0; j < extra_width; ++j) {
			left_gradient += gradient[(offset + word * extra_width + j) * nodes + base + parent]
				* tree[(extra_offset + j) * nodes + right];
			right_gradient += tree[(extra_offset + j) * nodes + left]
				* gradient[(offset + j * width + word) * nodes + base + parent];
		}
		extra_width *= Dimension;
	}
	gradient[coefficient * nodes + left] = left_gradient;
	gradient[coefficient * nodes + right] = right_gradient;
}

template<int AugDimension, typename T>
__device__ __forceinline__ T signature_path_gradient(const T* data, uint64_t point,
	uint64_t letter, uint64_t dimension, uint64_t length, bool lead_lag) {
	T value = T(0);
	if (lead_lag) {
		if (point > 0) value += data[(2 * point - 2) * AugDimension + dimension + letter]
			+ data[(2 * point - 1) * AugDimension + letter];
		if (point + 1 < length) value -= data[2 * point * AugDimension + dimension + letter]
			+ data[(2 * point + 1) * AugDimension + letter];
	}
	else {
		if (point > 0) value += data[(point - 1) * AugDimension + letter];
		if (point + 1 < length) value -= data[point * AugDimension + letter];
	}
	return value;
}

template<typename Kernel, typename T, bool Backward>
__global__ void signature_block_ker(const T* path, T* out, const T* deriv,
	T* workspace, T* increment_gradient, uint64_t dimension,
	uint64_t length, bool time_aug, bool lead_lag, T time_scale, bool scalar_term,
	uint64_t batch_offset, uint64_t batch_count, uint64_t chunks) {
	const uint64_t local_batch = cuda_batch_index();
	if (local_batch >= batch_count) return;
	const uint64_t batch = batch_offset + local_batch;
	const uint64_t nodes = 2 * chunks;
	const Path<T> input{path + batch * dimension * length, dimension, length, time_aug, lead_lag, time_scale};
	const uint64_t steps = input.steps();
	extern __shared__ __align__(16) unsigned char shared[];
	T* tree = reinterpret_cast<T*>(shared);
	if (threadIdx.x < chunks) {
		const uint64_t begin = threadIdx.x * (steps / chunks) + min(uint64_t(threadIdx.x), steps % chunks);
		const uint64_t end = begin + steps / chunks + (threadIdx.x < steps % chunks);
		T values[Kernel::signature_size];
		forward_range<Kernel>(input, values, begin, end);
		for (int i = 0; i < Kernel::signature_size; ++i) tree[i * nodes + chunks + threadIdx.x] = values[i];
	}
	__syncthreads();
	for (uint64_t first = chunks / 2; first > 0; first /= 2) {
		for (uint64_t index = threadIdx.x; index < first * Kernel::signature_size; index += blockDim.x) {
			const uint64_t coefficient = index / first;
			const uint64_t parent = first + index % first;
			tree[coefficient * nodes + parent] = signature_tree_product<Kernel::dimension>(
				tree, nodes, 2 * parent, 2 * parent + 1, coefficient);
		}
		__syncthreads();
	}
	if constexpr (Backward) {
		T* gradient = tree + nodes * Kernel::signature_size;
		const T* derivative = deriv + batch * (Kernel::signature_size + scalar_term) + scalar_term;
		for (int i = threadIdx.x; i < Kernel::signature_size; i += blockDim.x) gradient[i * nodes + 1] = derivative[i];
		__syncthreads();
		for (uint64_t first = 1; first < chunks; first *= 2) {
			for (uint64_t index = threadIdx.x; index < first * Kernel::signature_size; index += blockDim.x)
				signature_tree_pullback<Kernel::dimension, Kernel::degree>(tree, gradient, nodes,
					0, first + index % first, index / first);
			__syncthreads();
		}
		T* increments = increment_gradient + local_batch * steps * Kernel::dimension;
		if (threadIdx.x < chunks) {
			const uint64_t begin = threadIdx.x * (steps / chunks) + min(uint64_t(threadIdx.x), steps % chunks);
			const uint64_t end = begin + steps / chunks + (threadIdx.x < steps % chunks);
			T values[Kernel::signature_size];
			for (int i = 0; i < Kernel::signature_size; ++i) values[i] = gradient[i * nodes + chunks + threadIdx.x];
			backward_range<Kernel>(input, values,
				StridedWorkspace<T>{workspace + local_batch * chunks + threadIdx.x, batch_count * chunks},
				begin, end, IncrementGradient<T>{increments, Kernel::dimension});
		}
		__syncthreads();
		for (uint64_t index = threadIdx.x; index < dimension * length; index += blockDim.x)
			out[batch * dimension * length + index] = signature_path_gradient<Kernel::dimension>(
				increments, index / dimension, index % dimension, dimension, length, lead_lag);
	}
	else {
		T* output = out + batch * (Kernel::signature_size + scalar_term);
		if (scalar_term && threadIdx.x == 0) output[0] = T(1);
		for (int i = threadIdx.x; i < Kernel::signature_size; i += blockDim.x)
			output[i + scalar_term] = tree[i * nodes + 1];
	}
}

template<typename Kernel, typename T, bool Backward>
bool signature_block_cuda(const T* path, T* out, const T* deriv, uint64_t batch_size, uint64_t dimension,
	uint64_t length, bool time_aug, bool lead_lag, T time_scale, bool scalar_term) {
	if constexpr (Backward && Kernel::signature_size > 40) return false;
	const uint64_t steps = (length - 1) * (lead_lag ? 2 : 1);
	const uint64_t requested = steps / cuda_chunk_steps + (steps % cuda_chunk_steps != 0);
	const uint64_t limit = std::min<uint64_t>(128,
		CUDA_BASE_DYNAMIC_SMEM / ((Backward ? 4 : 2) * Kernel::signature_size * sizeof(T)));
	uint64_t chunks = 1;
	while (2 * chunks <= limit && chunks < requested) chunks *= 2;
	if (steps > 512 * chunks) return false;
	const size_t shared_bytes = (Backward ? 4 : 2) * chunks * Kernel::signature_size * sizeof(T);
	if constexpr (Backward) {
		constexpr const char* operation = "Horner signature block backprop";
		const uint64_t chunk_steps = steps / chunks + (steps % chunks != 0);
		const size_t workspace_size = checked_cuda_size_mul(chunks,
			(chunk_steps / checkpoint_steps + 1 + checkpoint_steps) * Kernel::state_size, operation);
		const size_t row_size = checked_cuda_size_add(workspace_size,
			checked_cuda_size_mul(steps, Kernel::dimension, operation), operation);
		cuda_path_workspace_launch<T>(batch_size, row_size,
			[&](T* workspace, uint64_t offset, uint64_t count) {
				const auto batch = make_cuda_batch_grid_chunk(1, count, 0);
				signature_block_ker<Kernel, T, true><<<batch.grid, 128, shared_bytes>>>(path, out, deriv,
					workspace, workspace + count * workspace_size, dimension, length, time_aug, lead_lag,
					time_scale, scalar_term, offset, count, chunks);
			}, operation);
		return true;
	}
	else {
		for (uint64_t offset = 0; offset < batch_size;) {
			const auto batch = make_cuda_batch_grid_chunk(1, batch_size, offset);
			signature_block_ker<Kernel, T, false><<<batch.grid, 128, shared_bytes>>>(path, out, nullptr, nullptr, nullptr, dimension,
				length, time_aug, lead_lag, time_scale, scalar_term, offset, batch.size, chunks);
			offset += batch.size;
		}
		check_cuda_kernel_launch();
		return true;
	}
}

template<typename Kernel, typename T, bool Backward>
__global__ void signature_chunks_ker(const T* path, T* tree, const T* tree_gradient,
	T* increment_gradient, T* workspace, uint64_t dimension, uint64_t length,
	bool time_aug, bool lead_lag, T time_scale, uint64_t batch_offset,
	uint64_t batch_count, uint64_t chunks) {
	constexpr int size = Kernel::signature_size;
	const uint64_t work = batch_count * chunks;
	const uint64_t nodes = 2 * work;
	const uint64_t steps = (length > 0 ? length - 1 : 0) * (lead_lag ? 2 : 1);
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < work; index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
		const uint64_t local_batch = index / chunks;
		const uint64_t chunk = index % chunks;
		const uint64_t node = local_batch * 2 * chunks + chunks + chunk;
		const uint64_t begin = chunk * cuda_chunk_steps;
		const uint64_t end = min(begin + cuda_chunk_steps, steps);
		const Path<T> input{path + (batch_offset + local_batch) * dimension * length,
			dimension, length, time_aug, lead_lag, time_scale};
		T values[size];
		if constexpr (Backward) {
			if (begin >= end) continue;
			for (int i = 0; i < size; ++i) values[i] = tree_gradient[i * nodes + node];
			backward_range<Kernel>(input, values, StridedWorkspace<T>{workspace + index, work},
				begin, end, IncrementGradient<T>{increment_gradient + local_batch * steps * Kernel::dimension,
					Kernel::dimension});
		}
		else {
			forward_range<Kernel>(input, values, begin, end);
			for (int i = 0; i < size; ++i) tree[i * nodes + node] = values[i];
		}
	}
}

template<int Dimension, int Degree, typename T, bool Backward>
__global__ void signature_tree_ker(T* tree, T* gradient, uint64_t chunks,
	uint64_t batch_count, uint64_t first, uint64_t size) {
	const uint64_t nodes = 2 * chunks * batch_count;
	const uint64_t parents = first * batch_count;
	const uint64_t work = parents * size;
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < work; index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
		const uint64_t coefficient = index / parents;
		const uint64_t parent_index = index % parents;
		const uint64_t batch = parent_index / first;
		const uint64_t parent = first + parent_index % first;
		const uint64_t base = batch * 2 * chunks;
		const uint64_t left = base + 2 * parent;
		const uint64_t right = left + 1;
		if constexpr (Backward) {
			signature_tree_pullback<Dimension, Degree>(tree, gradient, nodes, base, parent, coefficient);
		}
		else {
			tree[coefficient * nodes + base + parent] = signature_tree_product<Dimension>(tree, nodes, left, right, coefficient);
		}
	}
}

template<typename T, bool Backward>
__global__ void signature_tree_root_ker(const T* tree, T* gradient, T* out, const T* deriv,
	uint64_t batch_offset, uint64_t batch_count, uint64_t chunks, uint64_t size, bool scalar_term) {
	const uint64_t stride = size + scalar_term;
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < batch_count * stride; index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
		const uint64_t batch = index / stride;
		const uint64_t coefficient = index % stride;
		if (scalar_term && coefficient == 0) {
			if constexpr (!Backward) out[(batch_offset + batch) * stride] = T(1);
			continue;
		}
		const uint64_t node = (coefficient - scalar_term) * 2 * chunks * batch_count + batch * 2 * chunks + 1;
		if constexpr (Backward) gradient[node] = deriv[(batch_offset + batch) * stride + coefficient];
		else out[(batch_offset + batch) * stride + coefficient] = tree[node];
	}
}

template<int AugDimension, typename T>
__global__ void signature_increment_gradient_ker(const T* increment_gradient, T* out,
	uint64_t batch_offset, uint64_t batch_count, uint64_t dimension, uint64_t length, bool lead_lag) {
	const uint64_t steps = (length > 0 ? length - 1 : 0) * (lead_lag ? 2 : 1);
	const uint64_t path_size = dimension * length;
	for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		index < batch_count * path_size; index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
		const uint64_t batch = index / path_size;
		const uint64_t point = index % path_size / dimension;
		const uint64_t letter = index % dimension;
		const T* data = increment_gradient + batch * steps * AugDimension;
		out[batch_offset * path_size + index] = signature_path_gradient<AugDimension>(
			data, point, letter, dimension, length, lead_lag);
	}
}

template<typename Kernel, typename T, bool Backward>
void signature_chunked_cuda(const T* path, T* out, const T* deriv, uint64_t batch_size,
	uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T time_scale, bool scalar_term) {
	constexpr const char* operation = "chunked Horner signature";
	constexpr uint64_t size = Kernel::signature_size;
	const uint64_t steps = (length - 1) * (lead_lag ? 2 : 1);
	uint64_t chunks = 1;
	const uint64_t required_chunks = steps / cuda_chunk_steps + (steps % cuda_chunk_steps != 0);
	while (chunks < required_chunks) chunks = checked_cuda_size_mul(chunks, 2, operation);
	const uint64_t tree_size = checked_cuda_size_mul(checked_cuda_size_mul(chunks, 2, operation), size, operation);
	const uint64_t workspace_size = checked_cuda_size_mul(chunks,
		(cuda_chunk_steps / checkpoint_steps + 1 + checkpoint_steps) * Kernel::state_size, operation);
	const uint64_t gradient_size = checked_cuda_size_mul(steps, Kernel::dimension, operation);
	const uint64_t row_size = Backward ? checked_cuda_size_add(checked_cuda_size_mul(tree_size, 2, operation),
		checked_cuda_size_add(workspace_size, gradient_size, operation), operation) : tree_size;
	cuda_path_workspace_launch<T>(batch_size, row_size,
		[&](T* tree, uint64_t offset, uint64_t count) {
		T* gradient = Backward ? tree + count * tree_size : nullptr;
		T* workspace = Backward ? gradient + count * tree_size : nullptr;
		T* increments = Backward ? workspace + count * workspace_size : nullptr;
		signature_chunks_ker<Kernel, T, false><<<make_cuda_1d_grid(count * chunks, 64), 64>>>(
			path, tree, gradient, increments, workspace, dimension, length, time_aug, lead_lag,
			time_scale, offset, count, chunks);
		for (uint64_t first = chunks / 2; first > 0; first /= 2)
			signature_tree_ker<Kernel::dimension, Kernel::degree, T, false><<<make_cuda_1d_grid(count * first * size, 128), 128>>>(
				tree, gradient, chunks, count, first, size);
		if constexpr (Backward) {
			signature_tree_root_ker<T, true><<<make_cuda_1d_grid(count * (size + scalar_term), 128), 128>>>(
				tree, gradient, out, deriv, offset, count, chunks, size, scalar_term);
			for (uint64_t first = 1; first < chunks; first *= 2)
				signature_tree_ker<Kernel::dimension, Kernel::degree, T, true><<<make_cuda_1d_grid(count * first * size, 128), 128>>>(
					tree, gradient, chunks, count, first, size);
			signature_chunks_ker<Kernel, T, true><<<make_cuda_1d_grid(count * chunks, 64), 64>>>(
				path, tree, gradient, increments, workspace, dimension, length, time_aug, lead_lag,
				time_scale, offset, count, chunks);
			signature_increment_gradient_ker<Kernel::dimension><<<make_cuda_1d_grid(count * dimension * length, 128), 128>>>(
				increments, out, offset, count, dimension, length, lead_lag);
		}
		else {
			signature_tree_root_ker<T, false><<<make_cuda_1d_grid(count * (size + scalar_term), 128), 128>>>(
				tree, gradient, out, deriv, offset, count, chunks, size, scalar_term);
		}
	}, operation);
}

template<typename Kernel, typename T>
__global__ void signature_backward_ker(const T* path, T* out, const T* deriv,
	uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T time_scale,
	bool scalar_term, uint64_t batch_offset, uint64_t batch_count, T* workspace, uint64_t workspace_stride) {
	const uint64_t stride = Kernel::size + scalar_term;
	for (uint64_t local = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		local < batch_count; local += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
		const uint64_t batch = batch_offset + local;
		const Path<T> input{path + batch * dimension * length, dimension, length, time_aug, lead_lag, time_scale};
		backward<Kernel>(input, deriv + batch * stride + scalar_term,
			out + batch * dimension * length, workspace + local * workspace_stride);
	}
}

template<typename T>
bool signature_horner_cuda(const T* path, T* out, const T* deriv, uint64_t batch_size,
	uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, T end_time, bool scalar_term) {
	const uint64_t steps = (length - 1) * (lead_lag ? 2 : 1);
	const T time_scale = end_time / T(steps);
	return dispatch_horner(dimension * (lead_lag ? 2 : 1) + time_aug, degree, [&]<typename Kernel>() {
		if (!deriv) {
			if (steps < 1023) return false;
			if constexpr (sizeof(T) == 8 && ((Kernel::dimension == 3 && Kernel::degree == 5)
				|| (Kernel::dimension == 4 && Kernel::degree == 4)
				|| ((Kernel::dimension == 6 || Kernel::dimension == 8) && Kernel::degree == 3))) {
				if (steps < 8191) return false;
				if (!signature_block_cuda<Kernel, T, false>(path, out, deriv, batch_size, dimension, length, time_aug, lead_lag, time_scale, scalar_term))
					signature_chunked_cuda<Kernel, T, false>(path, out, deriv, batch_size, dimension, length, time_aug, lead_lag, time_scale, scalar_term);
				return true;
			}
			else if constexpr (Kernel::size > 258 || (sizeof(T) == 8 && Kernel::size > 155)) return false;
			else {
				if (Kernel::dimension == 2 && Kernel::degree == 4 && time_aug && !lead_lag && steps < 2047) return false;
				return signature_block_cuda<Kernel, T, false>(path, out, deriv, batch_size, dimension, length, time_aug, lead_lag, time_scale, scalar_term);
			}
		}
		if constexpr ((Kernel::dimension == 4 && Kernel::degree != 3) || Kernel::dimension == 8
			|| (Kernel::dimension == 2 && Kernel::degree == 2) || (Kernel::dimension == 7 && Kernel::degree == 3)) return false;
		else {
			// The general word kernels remain faster outside these measured ranges.
			bool selected = false;
			if constexpr (Kernel::degree == 2)
				selected = Kernel::dimension <= 5 || batch_size <= 4 || steps >= 1023
					|| (steps > 128 && batch_size <= 16);
			else if constexpr (Kernel::dimension == 2)
				selected = (steps <= 32 && batch_size <= 64 && (sizeof(T) == 4 || (Kernel::degree == 4 && batch_size <= 4)))
					|| (Kernel::degree >= 4 && steps >= 1023 && batch_size >= 32)
					|| (Kernel::degree == 3 && steps >= 8191 && batch_size >= 128)
					|| (Kernel::degree >= 4 && batch_size >= 16 && (steps >= 8191
						|| (sizeof(T) == 4 && steps >= (Kernel::degree == 4 ? 4095 : 2047))
						|| (lead_lag && steps >= 2047)))
					|| (Kernel::degree == 6 && sizeof(T) == 4 && steps > 128 && steps <= 512 && batch_size >= 32);
			else if constexpr (Kernel::dimension == 3 && Kernel::degree == 3)
				selected = steps <= 512 || (steps >= 1023 && batch_size >= (sizeof(T) == 4 ? 32 : 64));
			else if constexpr (Kernel::dimension == 3 && Kernel::degree == 4)
				selected = (sizeof(T) == 4 && ((steps <= 32 && batch_size <= 4)
					|| (steps > 128 && steps <= 512))) || (steps >= 1023 && batch_size >= 32)
					|| (sizeof(T) == 4 && steps >= (lead_lag ? 2047 : 4095) && batch_size >= 16)
					|| (steps >= 8191 && batch_size >= 16)
					|| (sizeof(T) == 8 && steps > 128 && steps <= 512 && batch_size >= 32);
			else if constexpr (Kernel::dimension == 3 && Kernel::degree == 5)
				selected = steps >= 1023 && (sizeof(T) == 8 || batch_size >= 32 || (steps >= 8191 && batch_size >= 16));
			else if constexpr (Kernel::degree == 3)
				selected = (sizeof(T) == 4 && ((steps <= 32 && batch_size <= 4)
					|| (Kernel::dimension != 4 && steps > 128 && steps <= 512) || (steps >= 1023 && batch_size >= 32)))
					|| (Kernel::dimension == 6 && sizeof(T) == 8 && steps <= 32 && batch_size >= 16 && batch_size <= 64);
			if constexpr (sizeof(T) == 4 && Kernel::degree == 3 && (Kernel::dimension == 4 || Kernel::dimension == 5))
				selected = selected || (batch_size >= 16 && steps >= (lead_lag ? 2047 : 4095));
			if (!selected) return false;
			if (steps > cuda_chunk_steps) {
				if (!signature_block_cuda<Kernel, T, true>(path, out, deriv, batch_size, dimension, length, time_aug, lead_lag, time_scale, scalar_term))
					signature_chunked_cuda<Kernel, T, true>(path, out, deriv, batch_size, dimension, length, time_aug, lead_lag, time_scale, scalar_term);
				return true;
			}
			constexpr const char* operation = "Horner signature backprop";
			const size_t rows = checked_cuda_size_add(steps / checkpoint_steps, 1 + checkpoint_steps, operation);
			const size_t row_elements = checked_cuda_size_mul(rows, Kernel::size, operation);
			cuda_path_workspace_launch<T>(batch_size, row_elements, [&](T* workspace, uint64_t offset, uint64_t count) {
				signature_backward_ker<Kernel><<<make_cuda_1d_grid(count, 64), 64>>>(path, out, deriv, dimension, length,
					time_aug, lead_lag, time_scale, scalar_term, offset, count, workspace, row_elements);
			}, operation);
			return true;
		}
	});
}

} // namespace horner_path
} // namespace
