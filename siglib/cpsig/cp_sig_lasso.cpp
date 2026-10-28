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

template<typename T>
void copy_lasso_result(const siglasso::Result& result, SigLassoAllocator allocate, void* context) {
    const auto output = siglasso::allocate_output(result, allocate, context);
    const auto convert = [](double value) { return static_cast<T>(siglasso::finite(static_cast<T>(value))); };
    std::transform(result.coefficients.begin(), result.coefficients.end(), static_cast<T*>(output.coefficients), convert);
    std::transform(result.intercepts.begin(), result.intercepts.end(), static_cast<T*>(output.intercepts), convert);
    std::transform(result.objectives.begin(), result.objectives.end(), static_cast<T*>(output.objectives), convert);
    std::transform(result.gaps.begin(), result.gaps.end(), static_cast<T*>(output.gaps), convert);
}

extern "C" {
CPSIG_API int sig_lasso_f(const float* path, const float* y, const double* alphas,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) noexcept {
    SAFE_CALL(siglasso::Result result; HWY_DYNAMIC_DISPATCH_T(sig_lasso_f)(path, y, alphas, letters, offsets, words, options, &result); copy_lasso_result<float>(result, allocate, context));
}
CPSIG_API int sig_lasso_d(const double* path, const double* y, const double* alphas,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) noexcept {
    SAFE_CALL(siglasso::Result result; HWY_DYNAMIC_DISPATCH_T(sig_lasso_d)(path, y, alphas, letters, offsets, words, options, &result); copy_lasso_result<double>(result, allocate, context));
}
}
} // namespace cpsig
