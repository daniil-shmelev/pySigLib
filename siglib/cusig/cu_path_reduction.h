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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

// Shared by ordinary and branched paths, and by float and double calls.
// The mutex covers kernel completion as well as allocation and shutdown.
struct CudaPathWorkspaces {
	struct Entry {
		std::unique_ptr<CudaBatchWorkspace<std::byte>> buffer;
		size_t requested_bytes = 0;
	};
	std::mutex mutex;
	std::unordered_map<int, Entry> devices;
};

inline CudaPathWorkspaces& cuda_path_workspaces() {
	static CudaPathWorkspaces workspaces;
	return workspaces;
}

inline void release_cuda_path_workspaces() {
	auto& workspaces = cuda_path_workspaces();
	std::lock_guard lock(workspaces.mutex);
	workspaces.devices.clear();
}

#ifdef __CUDACC__

template<typename T, typename Segment, typename Combine>
__global__ void cuda_path_reduce_kernel(
	const T* __restrict__ path,
	T* __restrict__ out,
	T* __restrict__ workspace,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t element_size,
	uint64_t memo_size,
	Segment segment,
	Combine combine
) {
	const uint64_t batch = cuda_batch_index();
	if (batch >= batch_size)
		return;
	const T* path_row = path + batch * length * dimension;
	T* out_row = out + batch * element_size;
	T* memo = workspace + batch * memo_size;
	extern __shared__ char shared[];

	segment(path_row, path_row + dimension, out_row);
	for (uint64_t index = 1; index + 1 < length; ++index) {
		const T* left = path_row + index * dimension;
		segment(left, left + dimension, memo + element_size);
		__syncthreads();
		combine(out_row, memo + element_size, out_row, memo, shared);
	}
}

template<typename T, typename Segment, typename Combine>
void cuda_path_reduce(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t element_size,
	uint64_t memo_nodes,
	unsigned int threads,
	size_t shared_bytes,
	Segment segment,
	Combine combine,
	const char* op_name
) {
	if (batch_size == 0)
		return;
	const size_t memo_size = checked_cuda_size_mul(
		static_cast<size_t>(memo_nodes), element_size, op_name);
	const size_t path_stride = checked_cuda_size_mul(
		static_cast<size_t>(length), dimension, op_name);
	const size_t row_bytes = checked_cuda_size_mul(memo_size, sizeof(T), op_name);
	const size_t requested_bytes = checked_cuda_size_mul(
		std::min(batch_size, CUDA_BATCH_GRID_CAPACITY), row_bytes, op_name);
	auto& workspaces = cuda_path_workspaces();
	std::lock_guard lock(workspaces.mutex);
	int device;
	CUDA_CHECK(cudaGetDevice(&device));
	auto& entry = workspaces.devices[device];
	const auto allocated_bytes = [&] {
		return entry.buffer ? entry.buffer->capacity() * entry.buffer->row_elements() : 0;
	};
	if (!entry.buffer || requested_bytes > entry.requested_bytes
		|| row_bytes > allocated_bytes()) {
		entry.buffer.reset();
		entry.buffer = std::make_unique<CudaBatchWorkspace<std::byte>>(
			batch_size, row_bytes, op_name);
		// Remember the request even when memory pressure forced a smaller chunk.
		entry.requested_bytes = requested_bytes;
	}
	const uint64_t capacity = std::min<uint64_t>(
		CUDA_BATCH_GRID_CAPACITY, allocated_bytes() / row_bytes);

	for (uint64_t offset = 0; offset < batch_size;
		offset += capacity) {
		const uint64_t count = std::min(
			capacity, batch_size - offset);
		const dim3 grid = make_cuda_batch_grid_chunk(1, count, 0).grid;
		cuda_path_reduce_kernel<T><<<grid, threads, shared_bytes>>>(
			path + offset * path_stride, out + offset * element_size,
			reinterpret_cast<T*>(entry.buffer->get()), count, length, dimension, element_size,
			memo_size, segment, combine);
	}
	check_cuda_kernel_launch();
}

#endif
