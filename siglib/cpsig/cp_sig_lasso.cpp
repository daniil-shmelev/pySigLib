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
HWY_EXPORT_T(sig_working_set_tt_envelopes_d, sig_working_set_tt_envelopes_<double>);
HWY_EXPORT_T(sig_working_set_query_benchmark_d, sig_working_set_query_benchmark_<double>);

extern "C" {
CPSIG_API int sig_working_set_experimental_bound(uint32_t bound) noexcept {
    if (bound > 4) return -1;
    siglasso::experimental_signature_bound.store(bound);
    return 0;
}
CPSIG_API int sig_working_set_experimental_pool(uint32_t capacity) noexcept {
    if (capacity && (capacity < 16 || capacity > 1024)) return -1;
    siglasso::experimental_candidate_pool.store(capacity);
    return 0;
}
CPSIG_API int sig_working_set_experimental_strategy(uint32_t strategy) noexcept {
    if (strategy > 7) return -1;
    siglasso::experimental_search_strategy.store(strategy);
    return 0;
}
CPSIG_API int sig_working_set_experimental_stats(double* output) noexcept {
    if (!output) return -1;
    std::copy(siglasso::experimental_last_stats.begin(), siglasso::experimental_last_stats.end(), output);
    return 0;
}
CPSIG_API int sig_working_set_experimental_search(uint32_t batch, uint32_t rank, uint32_t order, uint32_t partitions,
    uint32_t search, uint64_t cache) noexcept {
    if (!batch || batch > 64 || rank > 1024 || order > 7 || partitions > 64 || search > 4 || cache > (4ull << 30)) return -1;
    siglasso::experimental_search_batch.store(batch); siglasso::experimental_pool_rank.store(rank);
    siglasso::experimental_traversal_order.store(order); siglasso::experimental_search_partitions.store(partitions);
    siglasso::experimental_search_method.store(search); siglasso::experimental_cache_bytes.store(static_cast<size_t>(cache));
    return 0;
}
CPSIG_API int sig_working_set_experimental_refresh(double fraction) noexcept {
    if (!std::isfinite(fraction) || fraction < 0 || fraction > 1) return -1;
    siglasso::experimental_pool_refresh.store(fraction);
    return 0;
}
CPSIG_API int sig_working_set_experimental_kernels(uint32_t terminal, uint32_t solver) noexcept {
    if (terminal > 15 || solver > 31) return -1;
    siglasso::experimental_terminal_kernel.store(terminal);
    siglasso::experimental_restricted_solver.store(solver);
    return 0;
}
CPSIG_API int sig_working_set_experimental_features(uint32_t flags) noexcept {
    if (flags > 2047) return -1;
    siglasso::experimental_feature_kernel.store(flags);
    return 0;
}
CPSIG_API int sig_working_set_experimental_summaries(uint32_t flags, uint64_t bytes) noexcept {
    if (flags > 4 || bytes > (4ull << 30)) return -1;
    siglasso::experimental_summary_flags.store(flags);
    siglasso::experimental_summary_bytes.store(static_cast<size_t>(bytes));
    return 0;
}
CPSIG_API int sig_working_set_optimization_stats(double* output, uint64_t count) noexcept {
    if (!output || count != siglasso::OptimizationCounterCount) return -1;
    std::copy(siglasso::experimental_optimization_stats.begin(), siglasso::experimental_optimization_stats.end(), output);
    return 0;
}
CPSIG_API int sig_working_set_tt_envelopes_d(const double* path, const SigLassoOptions* options, double* output) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_working_set_tt_envelopes_d)(path, options, output));
}
CPSIG_API int sig_working_set_version() noexcept {
    return 2;
}
CPSIG_API int sig_working_set_query_benchmark_d(const double* paths, const double* residuals, const double* thresholds,
    uint32_t response_batch, const SigLassoOptions* options, double* scores, double* diagnostics) noexcept {
    SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(sig_working_set_query_benchmark_d)(paths, residuals, thresholds, response_batch, options, scores, diagnostics));
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
