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
#include "cp_chebyshev_sig_kernel.h"

namespace cpsig {
extern "C" {

CPSIG_API int sig_kernel_chebyshev_f(const float* gram, float* out, float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t p, uint64_t k, bool return_grid, int n_jobs) noexcept {
	SAFE_CALL((chebyshev_sig_kernel<float, false>(gram, out, nullptr, nullptr, state, batch_size, dimension, length1, length2, p, k, return_grid, n_jobs)));
}

CPSIG_API int sig_kernel_chebyshev_d(const double* gram, double* out, double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t p, uint64_t k, bool return_grid, int n_jobs) noexcept {
	SAFE_CALL((chebyshev_sig_kernel<double, false>(gram, out, nullptr, nullptr, state, batch_size, dimension, length1, length2, p, k, return_grid, n_jobs)));
}

CPSIG_API int sig_kernel_chebyshev_backprop_f(const float* gram, float* grad, const float* derivs, const float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t p, uint64_t k, bool return_grid, int n_jobs) noexcept {
	SAFE_CALL((chebyshev_sig_kernel<float, true>(gram, grad, derivs, state, nullptr, batch_size, dimension, length1, length2, p, k, return_grid, n_jobs)));
}

CPSIG_API int sig_kernel_chebyshev_backprop_d(const double* gram, double* grad, const double* derivs, const double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t p, uint64_t k, bool return_grid, int n_jobs) noexcept {
	SAFE_CALL((chebyshev_sig_kernel<double, true>(gram, grad, derivs, state, nullptr, batch_size, dimension, length1, length2, p, k, return_grid, n_jobs)));
}

}  // extern "C"
}  // namespace cpsig
