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

#include <cstdint>
#include <stdexcept>
#include <string>

constexpr size_t CUDA_BASE_DYNAMIC_SMEM = 48 * 1024;

struct CudaSharedMemoryLimits {
	size_t default_bytes;
	size_t optin_bytes;
};

inline CudaSharedMemoryLimits cuda_shared_memory_limits() {
	int device = 0;
	int default_bytes = 0;
	int optin_bytes = 0;
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaDeviceGetAttribute(
		&default_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
	CUDA_CHECK(cudaDeviceGetAttribute(
		&optin_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
	if (optin_bytes == 0)
		optin_bytes = default_bytes;
	return {
		static_cast<size_t>(default_bytes),
		static_cast<size_t>(optin_bytes)
	};
}

template<typename Kernel>
inline void configure_dynamic_smem(
	Kernel kernel,
	size_t smem,
	const char* op_name,
	const CudaSharedMemoryLimits& limits
) {
	if (smem > limits.optin_bytes) {
		throw std::invalid_argument(
			std::string(op_name) +
			" requires more dynamic shared memory than this CUDA device supports");
	}
	if (smem > limits.default_bytes) {
		CUDA_CHECK(cudaFuncSetAttribute(
			kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
			static_cast<int>(smem)));
	}
}

template<typename Kernel>
inline void configure_dynamic_smem(Kernel kernel, size_t smem, const char* op_name) {
	if (smem <= CUDA_BASE_DYNAMIC_SMEM)
		return;
	configure_dynamic_smem(kernel, smem, op_name, cuda_shared_memory_limits());
}

// =========================================================================
// Shared host-side helpers for signature length and level index computation
// =========================================================================

inline uint64_t host_power(uint64_t base, uint64_t exp) {
	uint64_t result = 1;
	while (exp > 0) {
		if (exp & 1) {
			// result * base overflows iff base > UINT64_MAX / result
			if (result != 0 && base > UINT64_MAX / result)
				return 0;
			result *= base;
		}
		exp >>= 1;
		if (exp > 0) {
			// base * base overflows iff base > UINT64_MAX / base
			if (base != 0 && base > UINT64_MAX / base)
				return 0;
			base *= base;
		}
	}
	return result;
}

inline uint64_t host_sig_length(uint64_t dimension, uint64_t degree) {
	if (dimension == 0) return 1;
	if (dimension == 1) {
		if (degree == UINT64_MAX)
			throw std::overflow_error("host_sig_length: degree overflow");
		return degree + 1;
	}
	const auto pwr = host_power(dimension, degree + 1);
	if (!pwr)
		throw std::overflow_error("host_sig_length: sig length overflow");
	return (pwr - 1) / (dimension - 1);
}

inline void host_populate_level_index(uint64_t* level_index, uint64_t dimension, uint64_t count) {
	level_index[0] = 0;
	for (uint64_t i = 1; i < count; ++i) {
		if (dimension != 0 && level_index[i - 1] > UINT64_MAX / dimension)
			throw std::overflow_error("host_populate_level_index: level_index overflow");
		const uint64_t mul = level_index[i - 1] * dimension;
		if (mul > UINT64_MAX - 1)
			throw std::overflow_error("host_populate_level_index: level_index overflow");
		level_index[i] = mul + 1;
	}
}

// =========================================================================
// Shared helper: choose threads per block from largest level size
// =========================================================================

inline unsigned int host_choose_threads_per_block(uint64_t max_level_size) {
	unsigned int tpb = 32;
	if (max_level_size > 32)   tpb = 64;
	if (max_level_size > 64)   tpb = 128;
	if (max_level_size > 128)  tpb = 256;
	if (max_level_size > 512)  tpb = 512;
	if (max_level_size > 1024) tpb = 1024;
	return tpb;
}
