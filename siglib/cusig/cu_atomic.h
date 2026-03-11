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
// Custom atomicAdd for double: CAS-based implementation for sm_50/52
// (Native atomicAdd(double*,double) requires sm_60+)
// For float, we just forward to the built-in atomicAdd.
// =========================================================================

template<typename T>
__device__ __forceinline__ void myAtomicAdd(T* address, T val);

template<>
__device__ __forceinline__ void myAtomicAdd<float>(float* address, float val) {
	atomicAdd(address, val);
}

template<>
__device__ __forceinline__ void myAtomicAdd<double>(double* address, double val) {
	unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
}
