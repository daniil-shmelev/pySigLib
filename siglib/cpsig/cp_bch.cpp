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
#include "cache_lifecycle/cp_bch_cache.h"
#include "cache_lifecycle/cp_poly_cache.h"
#include "cache_lifecycle/cp_branched_cache.h"
#include "cache_lifecycle/cp_branched_log_cache.h"
#include "cache_lifecycle/cp_branched_sig_coef_cache.h"


namespace cpsig {
HWY_EXPORT_T(log_sig_reduce_f, log_sig_reduce_<float>);
HWY_EXPORT_T(log_sig_reduce_d, log_sig_reduce_<double>);
HWY_EXPORT_T(log_sig_combine_f, log_sig_combine_<float>);
HWY_EXPORT_T(log_sig_combine_d, log_sig_combine_<double>);
HWY_EXPORT_T(log_sig_from_path_f, batch_log_sig_from_path_<float>);
HWY_EXPORT_T(log_sig_from_path_d, batch_log_sig_from_path_<double>);
HWY_EXPORT_T(log_sig_from_path_backprop_f, batch_log_sig_from_path_backprop_<float>);
HWY_EXPORT_T(log_sig_from_path_backprop_d, batch_log_sig_from_path_backprop_<double>);
HWY_EXPORT_T(log_sig_combine_backprop_f, log_sig_combine_backprop_<float>);
HWY_EXPORT_T(log_sig_combine_backprop_d, log_sig_combine_backprop_<double>);
HWY_EXPORT_T(log_sig_join_f, log_sig_join_<float>);
HWY_EXPORT_T(log_sig_join_d, log_sig_join_<double>);
HWY_EXPORT_T(log_sig_join_backprop_f, log_sig_join_backprop_<float>);
HWY_EXPORT_T(log_sig_join_backprop_d, log_sig_join_backprop_<double>);

extern "C" {
	CPSIG_API int log_sig_reduce_f(const float* inputs, float* out, uint64_t batch, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_reduce_f)(inputs, out, batch, length, dimension, degree, n_jobs));
	}
	CPSIG_API int log_sig_reduce_d(const double* inputs, double* out, uint64_t batch, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_reduce_d)(inputs, out, batch, length, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_combine_f(const float* log_sig1, const float* log_sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_combine_f)(log_sig1, log_sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_combine_d(const double* log_sig1, const double* log_sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_combine_d)(log_sig1, log_sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_from_path_f(const float* path, float* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_from_path_f)(path, out, batch_size, length, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_from_path_d(const double* path, double* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_from_path_d)(path, out, batch_size, length, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_from_path_backprop_f(const float* d_out, float* d_path, const float* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_from_path_backprop_f)(d_out, d_path, path, batch_size, length, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_from_path_backprop_d(const double* d_out, double* d_path, const double* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_from_path_backprop_d)(d_out, d_path, path, batch_size, length, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_combine_backprop_f(const float* d_out, float* d_ls1, float* d_ls2,
		const float* ls1, const float* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_combine_backprop_f)(d_out, d_ls1, d_ls2, ls1, ls2, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_combine_backprop_d(const double* d_out, double* d_ls1, double* d_ls2,
		const double* ls1, const double* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_combine_backprop_d)(d_out, d_ls1, d_ls2, ls1, ls2, batch_size, dimension, degree, n_jobs));
	}

	// log_sig_join
	CPSIG_API int log_sig_join_f(const float* log_sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_join_f)(log_sig, displacement, out, batch_size, dimension, degree, n_jobs));
	}
	CPSIG_API int log_sig_join_d(const double* log_sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_join_d)(log_sig, displacement, out, batch_size, dimension, degree, n_jobs));
	}

	// log_sig_join_backprop
	CPSIG_API int log_sig_join_backprop_f(const float* d_out, float* d_logsig, float* d_displacement, const float* log_sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_join_backprop_f)(d_out, d_logsig, d_displacement, log_sig, displacement, batch_size, dimension, degree, n_jobs));
	}
	CPSIG_API int log_sig_join_backprop_d(const double* d_out, double* d_logsig, double* d_displacement, const double* log_sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(log_sig_join_backprop_d)(d_out, d_logsig, d_displacement, log_sig, displacement, batch_size, dimension, degree, n_jobs));
	}

}

}  // namespace cpsig
