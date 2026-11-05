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
#include "cp_dispatch.h"
#include "macros.h"
#include "sig_lasso_core.h"

namespace cpsig {
HWY_EXPORT_T(sig_working_set_f, sig_working_set_<float>);
HWY_EXPORT_T(sig_working_set_d, sig_working_set_<double>);
HWY_EXPORT_T(sig_working_set_predict_f, sig_working_set_predict_<float>);
HWY_EXPORT_T(sig_working_set_predict_d, sig_working_set_predict_<double>);

extern "C" {
CPSIG_API int sig_working_set_version() noexcept {
    return 2;
}
CPSIG_API int sig_working_set_f(const float* path, const float* y, const double* alphas, const uint64_t* letters, const uint64_t* offsets,
    uint64_t words, uint32_t feature_type, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_working_set_f)(path, y, alphas, letters, offsets, words, feature_type, options, allocate, context));
}
CPSIG_API int sig_working_set_d(const double* path, const double* y, const double* alphas, const uint64_t* letters, const uint64_t* offsets,
    uint64_t words, uint32_t feature_type, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_working_set_d)(path, y, alphas, letters, offsets, words, feature_type, options, allocate, context));
}
CPSIG_API int sig_working_set_predict_f(const float* path, const float* coefficients, const float* intercept, float* output, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, uint32_t feature_type, const SigLassoOptions* options) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_working_set_predict_f)(path, coefficients, intercept, output, letters, offsets, words, feature_type, options));
}
CPSIG_API int sig_working_set_predict_d(const double* path, const double* coefficients, const double* intercept, double* output, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, uint32_t feature_type, const SigLassoOptions* options) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_working_set_predict_d)(path, coefficients, intercept, output, letters, offsets, words, feature_type, options));
}
}
} // namespace cpsig
