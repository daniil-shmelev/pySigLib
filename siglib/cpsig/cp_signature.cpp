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
#include "cp_signature.h"
#include "macros.h"

extern "C" {


	CPSIG_API int signature_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, bool horner, bool scalar_term, int n_jobs, const float* primitives, uint64_t primitives_len) noexcept {
		SAFE_CALL(signature_<float>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, scalar_term, n_jobs, primitives, primitives_len));
	}

	CPSIG_API int signature_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner, bool scalar_term, int n_jobs, const double* primitives, uint64_t primitives_len) noexcept {
		SAFE_CALL(signature_<double>(path, out, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, horner, scalar_term, n_jobs, primitives, primitives_len));
	}


	CPSIG_API int sig_backprop_f(const float* path, float* out, const float* sig_derivs, const float* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, bool scalar_term, int n_jobs, const float* primitives, uint64_t primitives_len) noexcept {
		SAFE_CALL(sig_backprop_<float>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs, primitives, primitives_len));
	}

	CPSIG_API int sig_backprop_d(const double* path, double* out, const double* sig_derivs, const double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool scalar_term, int n_jobs, const double* primitives, uint64_t primitives_len) noexcept {
		SAFE_CALL(sig_backprop_<double>(path, out, sig_derivs, sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs, primitives, primitives_len));
	}

}
