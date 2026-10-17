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

static void prepare_branched_log_sig_(
	uint64_t dimension,
	uint64_t max_nodes,
	BranchedLogSigMethod method,
	bool use_disk,
	bool planar
) {
	if (method != BranchedLogSigMethod::Expanded && !planar)
		throw std::invalid_argument("compressed branched log signatures are not available for planar=False");
	prepare_branched_sig_cache(dimension, max_nodes, use_disk, planar);
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	prepare_branched_log_sig_cache(cache, method, use_disk);
}




namespace cpsig {
HWY_EXPORT_T(branched_sig_to_log_sig_f, branched_sig_to_log_sig_<float>);
HWY_EXPORT_T(branched_sig_to_log_sig_d, branched_sig_to_log_sig_<double>);
HWY_EXPORT_T(branched_sig_to_log_sig_backprop_f, branched_sig_to_log_sig_backprop_<float>);
HWY_EXPORT_T(branched_sig_to_log_sig_backprop_d, branched_sig_to_log_sig_backprop_<double>);
HWY_EXPORT_T(branched_log_sig_from_path_f, branched_log_sig_from_path_<float>);
HWY_EXPORT_T(branched_log_sig_join_f, batch_branched_log_sig_join_<float>);
HWY_EXPORT_T(branched_log_sig_join_d, batch_branched_log_sig_join_<double>);
HWY_EXPORT_T(branched_log_sig_join_backprop_f, batch_branched_log_sig_join_backprop_<float>);
HWY_EXPORT_T(branched_log_sig_join_backprop_d, batch_branched_log_sig_join_backprop_<double>);
HWY_EXPORT_T(branched_log_sig_from_path_d, branched_log_sig_from_path_<double>);
HWY_EXPORT_T(branched_log_sig_from_path_backprop_f, branched_log_sig_from_path_backprop_<float>);
HWY_EXPORT_T(branched_log_sig_from_path_backprop_d, branched_log_sig_from_path_backprop_<double>);

extern "C" {

	CPSIG_API int prepare_branched_log_sig(uint64_t dimension, uint64_t max_nodes, int method, bool use_disk, bool planar) noexcept {
		SAFE_CALL(prepare_branched_log_sig_(dimension, max_nodes, parse_branched_log_sig_method(method), use_disk, planar));
	}

	CPSIG_API int branched_sig_to_log_sig_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_to_log_sig_f)(bsig, out, batch_size, dimension, max_nodes, parse_branched_log_sig_method(method), n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_to_log_sig_d)(bsig, out, batch_size, dimension, max_nodes, parse_branched_log_sig_method(method), n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_to_log_sig_backprop_f)(bsig, derivs, out, batch_size, dimension, max_nodes, parse_branched_log_sig_method(method), n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_to_log_sig_backprop_d)(bsig, derivs, out, batch_size, dimension, max_nodes, parse_branched_log_sig_method(method), n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_log_sig_from_path_f(const float* path, float* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_from_path_f)(path, out, batch_size, length, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_log_sig_join_f(const float* logsig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_join_f)(logsig, displacement, out, batch_size, dimension, max_nodes, prepend, n_jobs));
	}
	CPSIG_API int branched_log_sig_join_d(const double* logsig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_join_d)(logsig, displacement, out, batch_size, dimension, max_nodes, prepend, n_jobs));
	}
	CPSIG_API int branched_log_sig_join_backprop_f(const float* derivs, float* d_logsig, float* d_displacement, const float* logsig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_join_backprop_f)(derivs, d_logsig, d_displacement, logsig, displacement, batch_size, dimension, max_nodes, prepend, n_jobs));
	}
	CPSIG_API int branched_log_sig_join_backprop_d(const double* derivs, double* d_logsig, double* d_displacement, const double* logsig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool prepend, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_join_backprop_d)(derivs, d_logsig, d_displacement, logsig, displacement, batch_size, dimension, max_nodes, prepend, n_jobs));
	}

	CPSIG_API int branched_log_sig_from_path_d(const double* path, double* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_from_path_d)(path, out, batch_size, length, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_log_sig_from_path_backprop_f(const float* derivs, float* path_derivs, const float* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_from_path_backprop_f)(derivs, path_derivs, path, batch_size, length, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_log_sig_from_path_backprop_d(const double* derivs, double* path_derivs, const double* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_log_sig_from_path_backprop_d)(derivs, path_derivs, path, batch_size, length, dimension, max_nodes, n_jobs));
	}

}

}  // namespace cpsig
