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
HWY_EXPORT_T(linear_sig_f, batch_linear_sig_<float>);
HWY_EXPORT_T(linear_sig_d, batch_linear_sig_<double>);
HWY_EXPORT_T(sig_join_f, sig_join_<float>);
HWY_EXPORT_T(sig_join_d, sig_join_<double>);
HWY_EXPORT_T(sig_join_backprop_f, batch_sig_join_backprop_<float>);
HWY_EXPORT_T(sig_join_backprop_d, batch_sig_join_backprop_<double>);

extern "C" {

	CPSIG_API int linear_sig_f(const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(linear_sig_f)(displacement, out, batch_size, dimension, degree, scalar_term, n_jobs));
	}
	CPSIG_API int linear_sig_d(const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(linear_sig_d)(displacement, out, batch_size, dimension, degree, scalar_term, n_jobs));
	}

	CPSIG_API int sig_join_f(const float* sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_join_f)(sig, displacement, out, batch_size, dimension, degree, prepend, scalar_term, n_jobs));
	}
	CPSIG_API int sig_join_d(const double* sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_join_d)(sig, displacement, out, batch_size, dimension, degree, prepend, scalar_term, n_jobs));
	}

	CPSIG_API int sig_join_backprop_f(const float* d_out, float* d_sig, float* d_displacement, const float* sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_join_backprop_f)(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend, scalar_term, n_jobs));
	}
	CPSIG_API int sig_join_backprop_d(const double* d_out, double* d_sig, double* d_displacement, const double* sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_join_backprop_d)(d_out, d_sig, d_displacement, sig, displacement, batch_size, dimension, degree, prepend, scalar_term, n_jobs));
	}

}

}  // namespace cpsig
