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
#include "macros.h"
#include "cp_dispatch.h"
#include "cache_lifecycle/cp_bch_cache.h"
#include "cache_lifecycle/cp_poly_cache.h"
#include "cache_lifecycle/cp_branched_cache.h"
#include "cache_lifecycle/cp_branched_log_cache.h"
#include "cache_lifecycle/cp_branched_sig_coef_cache.h"


namespace cpsig {
HWY_EXPORT_T(sig_to_log_sig_f, sig_to_log_sig_<float>);
HWY_EXPORT_T(sig_to_log_sig_d, sig_to_log_sig_<double>);
HWY_EXPORT_T(sig_to_log_sig_backprop_f, sig_to_log_sig_backprop_<float>);
HWY_EXPORT_T(sig_to_log_sig_backprop_d, sig_to_log_sig_backprop_<double>);
HWY_EXPORT_T(logsig_to_sig_f, logsig_to_sig_<float>);
HWY_EXPORT_T(logsig_to_sig_d, logsig_to_sig_<double>);
HWY_EXPORT_T(logsig_to_sig_backprop_f, logsig_to_sig_backprop_<float>);
HWY_EXPORT_T(logsig_to_sig_backprop_d, logsig_to_sig_backprop_<double>);

extern "C" {

	CPSIG_API int sig_to_log_sig_f(const float* sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_to_log_sig_f)(sig, out, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

	CPSIG_API int sig_to_log_sig_d(const double* sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_to_log_sig_d)(sig, out, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

	CPSIG_API int sig_to_log_sig_backprop_f(const float* sig, float* out, const float* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_to_log_sig_backprop_f)(sig, out, log_sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

	CPSIG_API int sig_to_log_sig_backprop_d(const double* sig, double* out, const double* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_to_log_sig_backprop_d)(sig, out, log_sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_f(const float* log_sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(logsig_to_sig_f)(log_sig, out, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_d(const double* log_sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(logsig_to_sig_d)(log_sig, out, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_backprop_f(const float* log_sig, float* out, const float* sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(logsig_to_sig_backprop_f)(log_sig, out, sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

	CPSIG_API int logsig_to_sig_backprop_d(const double* log_sig, double* out, const double* sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(logsig_to_sig_backprop_d)(log_sig, out, sig_derivs, batch_size, dimension, degree, time_aug, lead_lag, parse_log_sig_conversion_method(method), scalar_term, n_jobs));
	}

}

}  // namespace cpsig
