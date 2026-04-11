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

#ifndef CUPCH_H
#define CUPCH_H

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
#include <string>
#include <filesystem>
#include <mutex>
#include <shared_mutex>

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

#endif //PCH_H
