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
HWY_EXPORT_T(sig_lasso_v2_f, sig_lasso_<float>);
HWY_EXPORT_T(sig_lasso_v2_d, sig_lasso_<double>);
static siglasso::Results lasso_results;

template<typename T>
void copy_lasso_result(uint64_t handle, uint64_t* letters, uint64_t* offsets, T* coefficients,
    T* intercepts, T* objectives, T* gaps, uint64_t* converged, uint64_t* iterations) {
    auto result = lasso_results.get(handle);
    size_t count = 0; offsets[0] = 0;
    for (size_t j = 0; j < result->words.size(); ++j) {
        for (auto a : result->words[j]) letters[count++] = a;
        offsets[j + 1] = count;
    }
    const auto convert = [](double value) { return static_cast<T>(siglasso::finite(static_cast<T>(value))); };
    std::transform(result->coefficients.begin(), result->coefficients.end(), coefficients, convert);
    std::transform(result->intercepts.begin(), result->intercepts.end(), intercepts, convert);
    std::transform(result->objectives.begin(), result->objectives.end(), objectives, convert);
    std::transform(result->gaps.begin(), result->gaps.end(), gaps, convert);
    std::copy(result->converged.begin(), result->converged.end(), converged);
    std::copy(result->iterations.begin(), result->iterations.end(), iterations);
}

extern "C" {
CPSIG_API uint32_t sig_lasso_abi_version() noexcept { return 2; }
CPSIG_API int sig_lasso_v2_f(const float* path, const float* y, const double* weights, const double* alphas,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, uint64_t* handle) noexcept {
    SAFE_CALL(siglasso::Result result; HWY_DYNAMIC_DISPATCH_T(sig_lasso_v2_f)(path, y, weights, alphas, letters, offsets, words, options, &result); *handle = lasso_results.insert(std::move(result)));
}
CPSIG_API int sig_lasso_v2_d(const double* path, const double* y, const double* weights, const double* alphas,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, uint64_t* handle) noexcept {
    SAFE_CALL(siglasso::Result result; HWY_DYNAMIC_DISPATCH_T(sig_lasso_v2_d)(path, y, weights, alphas, letters, offsets, words, options, &result); *handle = lasso_results.insert(std::move(result)));
}
CPSIG_API int sig_lasso_shape_v2(uint64_t handle, SigLassoShape* shape) noexcept {
    SAFE_CALL(auto result = lasso_results.get(handle); shape->words = result->words.size(); shape->targets = result->targets;
        shape->letters = 0; for (const auto& word : result->words) shape->letters += word.size(); shape->stats = result->stats);
}
CPSIG_API int sig_lasso_result_v2_f(uint64_t handle, uint64_t* letters, uint64_t* offsets, float* coefficients,
    float* intercepts, float* objectives, float* gaps, uint64_t* converged, uint64_t* iterations) noexcept {
    SAFE_CALL(copy_lasso_result(handle, letters, offsets, coefficients, intercepts, objectives, gaps, converged, iterations));
}
CPSIG_API int sig_lasso_result_v2_d(uint64_t handle, uint64_t* letters, uint64_t* offsets, double* coefficients,
    double* intercepts, double* objectives, double* gaps, uint64_t* converged, uint64_t* iterations) noexcept {
    SAFE_CALL(copy_lasso_result(handle, letters, offsets, coefficients, intercepts, objectives, gaps, converged, iterations));
}
CPSIG_API int sig_lasso_release_v2(uint64_t handle) noexcept { SAFE_CALL(lasso_results.erase(handle)); }
}
} // namespace cpsig
