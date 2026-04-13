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
#include "cp_branched_signature.h"
#include "macros.h"

extern "C" {

	CPSIG_API int prepare_branched_sig(uint64_t dimension, uint64_t max_nodes, bool use_disk) noexcept {
		SAFE_CALL(prepare_branched_sig_cache(dimension, max_nodes, use_disk));
	}

	CPSIG_API int branched_sig_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time) noexcept {
		SAFE_CALL(branched_signature_<float>(path, out, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time));
	}

	CPSIG_API int branched_sig_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(branched_signature_<double>(path, out, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time));
	}

	CPSIG_API int branched_sig_combine_f(const float* bsig1, const float* bsig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_sig_combine_<float>(bsig1, bsig2, out, batch_size, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_sig_combine_d(const double* bsig1, const double* bsig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_sig_combine_<double>(bsig1, bsig2, out, batch_size, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_sig_combine_backprop_f(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_sig_combine_backprop_<float>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_sig_combine_backprop_d(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs) noexcept {
		SAFE_CALL(branched_sig_combine_backprop_<double>(bsig1, bsig2, derivs, out1, out2, batch_size, dimension, max_nodes, n_jobs));
	}

	CPSIG_API int branched_sig_backprop_f(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time) noexcept {
		SAFE_CALL(branched_sig_backprop_<float>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time));
	}

	CPSIG_API int branched_sig_backprop_d(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time) noexcept {
		SAFE_CALL(branched_sig_backprop_<double>(path, out, bsig_derivs, bsig, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time));
	}

}
