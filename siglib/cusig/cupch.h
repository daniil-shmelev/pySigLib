/* Copyright 2025 Daniil Shmelev
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

// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.

#pragma once

// add headers that you want to pre-compile here

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include <windows.h>
#endif

#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include "../shared/errors.h"

// =========================================================================
// Shared CUDA error-checking helpers
// =========================================================================

// Checks cudaGetLastError() and throws std::runtime_error if a CUDA error occurred.
inline void check_cuda_error() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const int error_code = static_cast<int>(err);
		throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
	}
}

// Synchronizes the device and then checks for CUDA errors.
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

inline CudaBatchGridChunk make_cuda_batch_grid_chunk(
	uint64_t grid_x,
	uint64_t batch_size,
	uint64_t batch_offset
) {
	if (grid_x == 0 || grid_x > CUDA_GRID_X_LIMIT)
		throw std::invalid_argument("CUDA grid x dimension is out of range");
	if (batch_size == 0)
		throw std::invalid_argument("CUDA batch size must be positive");
	if (batch_offset >= batch_size)
		throw std::invalid_argument("CUDA batch offset is out of range");
	const uint64_t chunk_size = std::min<uint64_t>(
		batch_size - batch_offset, CUDA_BATCH_GRID_CAPACITY);
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

// Wraps a CUDA runtime call; throws std::runtime_error if it returns non-success.
#define CUDA_CHECK(call) do { \
	cudaError_t _cuda_err = (call); \
	if (_cuda_err != cudaSuccess) \
		throw std::runtime_error("CUDA Error (" + std::to_string(static_cast<int>(_cuda_err)) + "): " + cudaGetErrorString(_cuda_err)); \
} while (0)

// RAII wrapper for a cudaMalloc'd device buffer. Frees automatically on scope exit
// (including exception unwind). `release()` transfers ownership out.
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
	CudaBuf(CudaBuf&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
	CudaBuf& operator=(CudaBuf&& o) noexcept {
		if (this != &o) { reset(); ptr_ = o.ptr_; o.ptr_ = nullptr; }
		return *this;
	}
	~CudaBuf() { reset(); }
	T* get() const noexcept { return ptr_; }
	operator T*() const noexcept { return ptr_; }
	T* release() noexcept { T* p = ptr_; ptr_ = nullptr; return p; }
	void reset() noexcept { if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; } }
private:
	T* ptr_;
};
