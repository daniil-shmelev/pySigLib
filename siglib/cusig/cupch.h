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

#endif //PCH_H
