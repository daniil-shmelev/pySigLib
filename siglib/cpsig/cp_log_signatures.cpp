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
#include "cp_log_signature.h"
#include "cp_exp_signature.h"
#include "macros.h"

extern "C" {

	CPSIG_API int sig_to_log_sig_f(const float* sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(sig_to_log_sig_<float>(sig, out, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

	CPSIG_API int sig_to_log_sig_d(const double* sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(sig_to_log_sig_<double>(sig, out, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

	CPSIG_API int sig_to_log_sig_backprop_f(const float* sig, float* out, const float* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(sig_to_log_sig_backprop_<float>(sig, out, log_sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

	CPSIG_API int sig_to_log_sig_backprop_d(const double* sig, double* out, const double* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(sig_to_log_sig_backprop_<double>(sig, out, log_sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_f(const float* log_sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(logsig_to_sig_<float>(log_sig, out, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_d(const double* log_sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(logsig_to_sig_<double>(log_sig, out, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_backprop_f(const float* log_sig, float* out, const float* sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(logsig_to_sig_backprop_<float>(log_sig, out, sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_backprop_d(const double* log_sig, double* out, const double* sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(logsig_to_sig_backprop_<double>(log_sig, out, sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, method, scalar_term, n_jobs));
	}

}
