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

#include "cppch.h"
#include "cpsig.h"
#include "cp_sig_kernel_log_pde.h"
#include "macros.h"

extern "C" {

	CPSIG_API int sig_kernel_log_pde_f(
		const float* path_x, const float* path_y, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y,
		uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y,
		bool return_grid, int n_jobs
	) noexcept {
		SAFE_CALL(sig_kernel_log_pde_<float>(
			path_x, path_y, out, batch_size, dimension, length_x, length_y,
			log_step_x, log_step_y, degree_x, degree_y,
			dyadic_order_x, dyadic_order_y, return_grid, n_jobs));
	}

	CPSIG_API int sig_kernel_log_pde_d(
		const double* path_x, const double* path_y, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y,
		uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y,
		bool return_grid, int n_jobs
	) noexcept {
		SAFE_CALL(sig_kernel_log_pde_<double>(
			path_x, path_y, out, batch_size, dimension, length_x, length_y,
			log_step_x, log_step_y, degree_x, degree_y,
			dyadic_order_x, dyadic_order_y, return_grid, n_jobs));
	}

	CPSIG_API int sig_kernel_log_pde_backprop_f(
		const float* path_x, const float* path_y,
		float* d_path_x, float* d_path_y, const float* derivs, const float* k_grid,
		uint64_t batch_size, uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y,
		uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y,
		bool return_grid, int n_jobs
	) noexcept {
		SAFE_CALL(sig_kernel_log_pde_backprop_<float>(
			path_x, path_y, d_path_x, d_path_y, derivs, k_grid,
			batch_size, dimension, length_x, length_y, log_step_x, log_step_y,
			degree_x, degree_y, dyadic_order_x, dyadic_order_y, return_grid, n_jobs));
	}

	CPSIG_API int sig_kernel_log_pde_backprop_d(
		const double* path_x, const double* path_y,
		double* d_path_x, double* d_path_y, const double* derivs, const double* k_grid,
		uint64_t batch_size, uint64_t dimension, uint64_t length_x, uint64_t length_y,
		uint64_t log_step_x, uint64_t log_step_y,
		uint64_t degree_x, uint64_t degree_y,
		uint64_t dyadic_order_x, uint64_t dyadic_order_y,
		bool return_grid, int n_jobs
	) noexcept {
		SAFE_CALL(sig_kernel_log_pde_backprop_<double>(
			path_x, path_y, d_path_x, d_path_y, derivs, k_grid,
			batch_size, dimension, length_x, length_y, log_step_x, log_step_y,
			degree_x, degree_y, dyadic_order_x, dyadic_order_y, return_grid, n_jobs));
	}
}
