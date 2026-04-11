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

// =========================================================================
// Custom atomicAdd for double: native on sm_60+, CAS fallback otherwise.
// For float, forward to the built-in atomicAdd.
// =========================================================================

template<typename T>
__device__ __forceinline__ void myAtomicAdd(T* address, T val);

template<>
__device__ __forceinline__ void myAtomicAdd<float>(float* address, float val) {
	atomicAdd(address, val);
}

template<>
__device__ __forceinline__ void myAtomicAdd<double>(double* address, double val) {
#if __CUDA_ARCH__ >= 600
	atomicAdd(address, val);
#else
	// CAS fallback for pre-Pascal. Compares raw bit patterns so that a
	// NaN at *address does not spin forever (NaN != NaN under FP compare).
	unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
	unsigned long long int assumed;
	unsigned long long int old = atomicAdd(address_as_ull, 0ULL);  // atomic load
	do {
		assumed = old;
		const double d_assumed = __longlong_as_double(assumed);
		const unsigned long long int desired = __double_as_longlong(d_assumed + val);
		old = atomicCAS(address_as_ull, assumed, desired);
	} while (assumed != old);
#endif
}
