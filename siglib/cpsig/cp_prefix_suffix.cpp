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

namespace cpsig {
HWY_EXPORT_T(signature_prefix_suffix_f, signature_prefix_suffix_<float>);
HWY_EXPORT_T(signature_prefix_suffix_d, signature_prefix_suffix_<double>);
HWY_EXPORT_T(signature_prefix_suffix_backprop_f, signature_prefix_suffix_backprop_<float>);
HWY_EXPORT_T(signature_prefix_suffix_backprop_d, signature_prefix_suffix_backprop_<double>);

extern "C" {
CPSIG_API int signature_prefix_suffix_backprop_f(const float* path, float* out, const float* deriv, uint64_t batch, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, bool scalar_term, int n_jobs) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(signature_prefix_suffix_backprop_f)(path, out, deriv, batch, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs));
}
CPSIG_API int signature_prefix_suffix_backprop_d(const double* path, double* out, const double* deriv, uint64_t batch, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool scalar_term, int n_jobs) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(signature_prefix_suffix_backprop_d)(path, out, deriv, batch, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs));
}
CPSIG_API int signature_prefix_suffix_f(const float* path, float* out, uint64_t batch, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, bool scalar_term, int n_jobs) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(signature_prefix_suffix_f)(path, out, batch, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs));
}
CPSIG_API int signature_prefix_suffix_d(const double* path, double* out, uint64_t batch, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool scalar_term, int n_jobs) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(signature_prefix_suffix_d)(path, out, batch, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs));
}
}

} // namespace cpsig
