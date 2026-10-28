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
HWY_EXPORT_T(sig_lasso_f, sig_lasso_<float>);
HWY_EXPORT_T(sig_lasso_d, sig_lasso_<double>);

extern "C" {
CPSIG_API int sig_lasso_f(const float* path, const float* y, const double* alphas,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_lasso_f)(path, y, alphas, letters, offsets, words, options, allocate, context));
}
CPSIG_API int sig_lasso_d(const double* path, const double* y, const double* alphas,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_lasso_d)(path, y, alphas, letters, offsets, words, options, allocate, context));
}
}
} // namespace cpsig
