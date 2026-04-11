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

#include "cppch.h"
#include "cpsig.h"
#include "cp_sig_kernel.h"
#include "macros.h"


extern "C" {


	CPSIG_API int sig_kernel_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(sig_kernel_<float>(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}

	CPSIG_API int sig_kernel_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(sig_kernel_<double>(gram, out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}


	CPSIG_API int sig_kernel_backprop_f(const float* gram, float* out, const float* derivs, const float* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_kernel_backprop_<float>(gram, out, derivs, k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}

	CPSIG_API int sig_kernel_backprop_d(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_kernel_backprop_<double>(gram, out, derivs, k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, return_grid, n_jobs));
	}
}
