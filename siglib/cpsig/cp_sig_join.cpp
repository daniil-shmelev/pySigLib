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
#include "cp_sig_join.h"
#include "macros.h"

extern "C" {

	CPSIG_API int linear_sig_f(const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_linear_sig_<float>(displacement, out, batch_size, dimension, degree, n_jobs));
	}
	CPSIG_API int linear_sig_d(const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_linear_sig_<double>(displacement, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int sig_join_f(const float* sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(sig_join_<float>(sig, displacement, out, batch_size, dimension, degree, prepend, n_jobs));
	}
	CPSIG_API int sig_join_d(const double* sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(sig_join_<double>(sig, displacement, out, batch_size, dimension, degree, prepend, n_jobs));
	}

	CPSIG_API int sig_join_backprop_f(const float* d_out, float* d_sig, float* d_displacement, const float* sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_join_backprop_<float>(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend, n_jobs));
	}
	CPSIG_API int sig_join_backprop_d(const double* d_out, double* d_sig, double* d_displacement, const double* sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_join_backprop_<double>(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend, n_jobs));
	}

}
