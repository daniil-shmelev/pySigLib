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
HWY_EXPORT_T(branched_sig_coef_f, branched_sig_coef_<float>);
HWY_EXPORT_T(branched_sig_coef_d, branched_sig_coef_<double>);
HWY_EXPORT_T(branched_sig_coef_backprop_f, branched_sig_coef_backprop_<float>);
HWY_EXPORT_T(branched_sig_coef_backprop_d, branched_sig_coef_backprop_<double>);

extern "C" {

	CPSIG_API int prepare_branched_sig_coef(const uint64_t* tree_data, uint64_t tree_data_len, uint64_t data_dimension, uint64_t dimension, uint64_t max_nodes, bool planar, bool use_disk) noexcept {
		SAFE_CALL(prepare_branched_sig_coef_cache(tree_data, tree_data_len, data_dimension, dimension, max_nodes, planar, use_disk));
	}

	CPSIG_API int branched_sig_coef_f(const float* path, float* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_coef_f)(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CPSIG_API int branched_sig_coef_d(const double* path, double* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_coef_d)(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CPSIG_API int branched_sig_coef_backprop_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_coef_backprop_f)(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CPSIG_API int branched_sig_coef_backprop_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_coef_backprop_d)(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

}

}  // namespace cpsig
