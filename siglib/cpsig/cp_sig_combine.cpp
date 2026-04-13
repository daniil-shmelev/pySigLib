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
#include "cp_sig_combine.h"
#include "macros.h"

extern "C" {

    CPSIG_API int sig_combine_f(const float* sig1, const float* sig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
        SAFE_CALL(batch_sig_combine_<float>(sig1, sig2, out, batch_size, dimension, degree, n_jobs));
    }

	CPSIG_API int sig_combine_d(const double* sig1, const double* sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_combine_<double>(sig1, sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int sig_combine_backprop_f(const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv, const float* sig1, const float* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_combine_backprop_<float>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, n_jobs));
	}

    CPSIG_API int sig_combine_backprop_d(const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
        SAFE_CALL(batch_sig_combine_backprop_<double>(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, n_jobs));
    }

}
