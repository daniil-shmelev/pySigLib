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
#include "cp_bch.h"

std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<BchCache>, PairHash> bch_cache;

extern "C" {

	CPSIG_API int log_sig_combine_f(const float* log_sig1, const float* log_sig2, float* out,
		uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(log_sig_combine_<float>(log_sig1, log_sig2, out, dimension, degree));
	}

	CPSIG_API int log_sig_combine_d(const double* log_sig1, const double* log_sig2, double* out,
		uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(log_sig_combine_<double>(log_sig1, log_sig2, out, dimension, degree));
	}

	CPSIG_API int batch_log_sig_combine_f(const float* log_sig1, const float* log_sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_log_sig_combine_<float>(log_sig1, log_sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int batch_log_sig_combine_d(const double* log_sig1, const double* log_sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_log_sig_combine_<double>(log_sig1, log_sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int batch_log_sig_from_path_f(const float* path, float* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_log_sig_from_path_<float>(path, out, batch_size, length, dimension, degree, n_jobs));
	}

	CPSIG_API int batch_log_sig_from_path_d(const double* path, double* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_log_sig_from_path_<double>(path, out, batch_size, length, dimension, degree, n_jobs));
	}

	CPSIG_API int log_sig_combine_backprop_f(const float* d_out, float* d_ls1, float* d_ls2,
		const float* ls1, const float* ls2, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(log_sig_combine_backprop_<float>(d_out, d_ls1, d_ls2, ls1, ls2, dimension, degree));
	}

	CPSIG_API int log_sig_combine_backprop_d(const double* d_out, double* d_ls1, double* d_ls2,
		const double* ls1, const double* ls2, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(log_sig_combine_backprop_<double>(d_out, d_ls1, d_ls2, ls1, ls2, dimension, degree));
	}

	CPSIG_API int batch_log_sig_combine_backprop_f(const float* d_out, float* d_ls1, float* d_ls2,
		const float* ls1, const float* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_log_sig_combine_backprop_<float>(d_out, d_ls1, d_ls2, ls1, ls2, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int batch_log_sig_combine_backprop_d(const double* d_out, double* d_ls1, double* d_ls2,
		const double* ls1, const double* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_log_sig_combine_backprop_<double>(d_out, d_ls1, d_ls2, ls1, ls2, batch_size, dimension, degree, n_jobs));
	}

}
