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

#include "cuda_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

inline size_t checked_cuda_size_add(
	size_t left, size_t right, const char* op_name
) {
	if (right > std::numeric_limits<size_t>::max() - left)
		throw std::overflow_error(std::string(op_name) + ": size addition overflow");
	return left + right;
}

inline size_t checked_cuda_size_mul(
	size_t left, size_t right, const char* op_name
) {
	if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
		throw std::overflow_error(std::string(op_name) + ": size multiplication overflow");
	return left * right;
}

inline void check_cuda_error() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const int error_code = static_cast<int>(err);
		throw std::runtime_error(
			"CUDA Error (" + std::to_string(error_code) + "): "
			+ cudaGetErrorString(err));
	}
}

inline void check_cuda_kernel_launch() {
	cudaDeviceSynchronize();
	check_cuda_error();
}

constexpr unsigned int CUDA_BATCH_GRID_LIMIT = 65535;
constexpr uint64_t CUDA_GRID_X_LIMIT = 2147483647;
constexpr uint64_t CUDA_BATCH_GRID_CAPACITY =
	static_cast<uint64_t>(CUDA_BATCH_GRID_LIMIT) * CUDA_BATCH_GRID_LIMIT;

inline unsigned int make_cuda_1d_grid(uint64_t work_size, unsigned int block_size) {
	if (work_size == 0)
		throw std::invalid_argument("CUDA work size must be positive");
	return static_cast<unsigned int>(std::min<uint64_t>(
		work_size / block_size + (work_size % block_size != 0),
		CUDA_BATCH_GRID_LIMIT));
}

struct CudaBatchGridChunk {
	dim3 grid;
	uint64_t offset;
	uint64_t size;
};

inline uint64_t cuda_global_batch_index(
	uint64_t batch_offset, uint64_t local_batch_index
) {
	if (local_batch_index > std::numeric_limits<uint64_t>::max() - batch_offset)
		throw std::overflow_error("CUDA global batch index overflow");
	return batch_offset + local_batch_index;
}

inline uint64_t cuda_workspace_initial_capacity(
	uint64_t batch_size, size_t row_bytes, size_t free_bytes
) {
	if (batch_size == 0)
		return 0;
	if (row_bytes == 0)
		throw std::invalid_argument("CUDA workspace row size must be positive");
	uint64_t capacity = static_cast<uint64_t>((free_bytes / 2) / row_bytes);
	if (capacity == 0)
		capacity = 1;
	return std::min<uint64_t>(
		capacity, std::min<uint64_t>(batch_size, CUDA_BATCH_GRID_CAPACITY));
}

inline uint64_t cuda_workspace_retry_capacity(uint64_t capacity) {
	return capacity <= 1 ? 0 : std::max<uint64_t>(1, capacity / 2);
}

inline CudaBatchGridChunk make_cuda_batch_grid_chunk(
	uint64_t grid_x,
	uint64_t batch_size,
	uint64_t batch_offset,
	uint64_t max_chunk_size = CUDA_BATCH_GRID_CAPACITY
) {
	if (grid_x == 0 || grid_x > CUDA_GRID_X_LIMIT)
		throw std::invalid_argument("CUDA grid x dimension is out of range");
	if (batch_size == 0)
		throw std::invalid_argument("CUDA batch size must be positive");
	if (batch_offset >= batch_size)
		throw std::invalid_argument("CUDA batch offset is out of range");
	const uint64_t chunk_size = std::min<uint64_t>(
		batch_size - batch_offset,
		std::min<uint64_t>(CUDA_BATCH_GRID_CAPACITY, max_chunk_size));
	if (chunk_size == 0)
		throw std::invalid_argument("CUDA maximum chunk size must be positive");
	const uint64_t grid_z = chunk_size / CUDA_BATCH_GRID_LIMIT
		+ (chunk_size % CUDA_BATCH_GRID_LIMIT != 0);
	const uint64_t grid_y = chunk_size / grid_z + (chunk_size % grid_z != 0);
	return {
		dim3(
			static_cast<unsigned int>(grid_x),
			static_cast<unsigned int>(grid_y),
			static_cast<unsigned int>(grid_z)),
		batch_offset,
		chunk_size
	};
}

#ifdef __CUDACC__
__device__ __forceinline__ uint64_t cuda_batch_index() {
	return static_cast<uint64_t>(blockIdx.y)
		+ static_cast<uint64_t>(gridDim.y) * blockIdx.z;
}
#endif

#define CUDA_CHECK(call) do { \
	cudaError_t _cuda_err = (call); \
	if (_cuda_err != cudaSuccess) \
		throw std::runtime_error("CUDA Error (" + std::to_string(static_cast<int>(_cuda_err)) + "): " + cudaGetErrorString(_cuda_err)); \
} while (0)

template<typename T>
class CudaBuf {
public:
	CudaBuf() noexcept : ptr_(nullptr) {}
	explicit CudaBuf(size_t bytes) {
		void* p = nullptr;
		CUDA_CHECK(cudaMalloc(&p, bytes));
		ptr_ = static_cast<T*>(p);
	}
	CudaBuf(const CudaBuf&) = delete;
	CudaBuf& operator=(const CudaBuf&) = delete;
	CudaBuf(CudaBuf&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
	CudaBuf& operator=(CudaBuf&& other) noexcept {
		if (this != &other) {
			reset();
			ptr_ = other.ptr_;
			other.ptr_ = nullptr;
		}
		return *this;
	}
	~CudaBuf() { reset(); }
	T* get() const noexcept { return ptr_; }
	operator T*() const noexcept { return ptr_; }
	T* release() noexcept {
		T* ptr = ptr_;
		ptr_ = nullptr;
		return ptr;
	}
	void reset() noexcept {
		if (ptr_) {
			cudaFree(ptr_);
			ptr_ = nullptr;
		}
	}

private:
	T* ptr_;
};

template<typename T>
class CudaBatchWorkspace {
public:
	CudaBatchWorkspace(
		uint64_t batch_size,
		size_t row_elements,
		const char* op_name
	) : ptr_(nullptr), capacity_(0), row_elements_(row_elements) {
		if (batch_size == 0)
			return;
		if (row_elements == 0) {
			capacity_ = std::min<uint64_t>(batch_size, CUDA_BATCH_GRID_CAPACITY);
			return;
		}
		const size_t row_bytes = checked_cuda_size_mul(
			row_elements, sizeof(T), op_name);
		size_t free_bytes = 0;
		size_t total_bytes = 0;
		CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
		uint64_t candidate = cuda_workspace_initial_capacity(
			batch_size, row_bytes, free_bytes);

		while (true) {
			const size_t allocation_bytes = checked_cuda_size_mul(
				static_cast<size_t>(candidate), row_bytes, op_name);
			void* allocation = nullptr;
			const cudaError_t error = cudaMalloc(&allocation, allocation_bytes);
			if (error == cudaSuccess) {
				ptr_ = static_cast<T*>(allocation);
				capacity_ = candidate;
				break;
			}
			if (error != cudaErrorMemoryAllocation)
				CUDA_CHECK(error);
			(void)cudaGetLastError();
			if (candidate == 1) {
				throw std::runtime_error(
					"CUDA Error (" + std::to_string(
						static_cast<int>(cudaErrorMemoryAllocation)) + "): "
					+ std::string(op_name) + " requires "
					+ std::to_string(row_bytes)
					+ " workspace bytes for one sample, but only "
					+ std::to_string(free_bytes) + " bytes were free");
			}
			candidate = cuda_workspace_retry_capacity(candidate);
		}
	}

	CudaBatchWorkspace(const CudaBatchWorkspace&) = delete;
	CudaBatchWorkspace& operator=(const CudaBatchWorkspace&) = delete;
	CudaBatchWorkspace(CudaBatchWorkspace&&) = delete;
	CudaBatchWorkspace& operator=(CudaBatchWorkspace&&) = delete;
	~CudaBatchWorkspace() {
		if (ptr_)
			cudaFree(ptr_);
	}

	T* get() const noexcept { return ptr_; }
	uint64_t capacity() const noexcept { return capacity_; }
	size_t row_elements() const noexcept { return row_elements_; }

private:
	T* ptr_;
	uint64_t capacity_;
	size_t row_elements_;
};
