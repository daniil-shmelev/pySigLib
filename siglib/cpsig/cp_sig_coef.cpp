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
#include "macros.h"
#include "cp_dispatch.h"
#include "cp_coefficient_workspace.h"
#include "cache_lifecycle/cp_bch_cache.h"
#include "cache_lifecycle/cp_poly_cache.h"
#include "cache_lifecycle/cp_branched_cache.h"
#include "cache_lifecycle/cp_branched_log_cache.h"
#include "cache_lifecycle/cp_branched_sig_coef_cache.h"


namespace cpsig {
HWY_EXPORT_T(sig_coef_f, sig_coef_<float>);
HWY_EXPORT_T(sig_coef_d, sig_coef_<double>);
HWY_EXPORT_T(sig_coef_backprop_f, sig_coef_backprop_<float>);
HWY_EXPORT_T(sig_coef_backprop_d, sig_coef_backprop_<double>);
HWY_EXPORT_T(sig_coef_dispatch_f, sig_coef_dispatch_<float>);
HWY_EXPORT_T(sig_coef_dispatch_d, sig_coef_dispatch_<double>);
HWY_EXPORT_T(sig_coef_backprop_dispatch_f, sig_coef_backprop_dispatch_<float>);
HWY_EXPORT_T(sig_coef_backprop_dispatch_d, sig_coef_backprop_dispatch_<double>);

extern "C" {

	CPSIG_API void cpsig_coefficient_workspace_start() noexcept {
		coefficient_workspace::live = 0;
		coefficient_workspace::peak = 0;
		coefficient_workspace::enabled = true;
	}

	CPSIG_API uint64_t cpsig_coefficient_workspace_stop() noexcept {
		coefficient_workspace::enabled = false;
		return coefficient_workspace::peak.load();
	}

	CPSIG_API int sig_coef_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, bool prefixes, int n_jobs, int method) noexcept {
		if (method == 0 && length > 1) {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_f)(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs));
		} else {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_dispatch_f)(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs, method));
		}
	}

	CPSIG_API int sig_coef_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, bool prefixes, int n_jobs, int method) noexcept {
		if (method == 0 && length > 1) {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_d)(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs));
		} else {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_dispatch_d)(path, out, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs, method));
		}
	}


	CPSIG_API int sig_coef_backprop_f(const float* path, float* out, const float* coefs, float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, int n_jobs, int method) noexcept {
		if (method == 0 && length > 1) {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_backprop_f)(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
		} else {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_backprop_dispatch_f)(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs, method));
		}
	}

	CPSIG_API int sig_coef_backprop_d(const double* path, double* out, const double* coefs, double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, int n_jobs, int method) noexcept {
		if (method == 0 && length > 1) {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_backprop_d)(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs));
		} else {
			SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_coef_backprop_dispatch_d)(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs, method));
		}
	}

}

}  // namespace cpsig
