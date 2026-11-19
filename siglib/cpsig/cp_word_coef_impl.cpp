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
#include <atomic>
#include "cp_simd_config.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cp_word_coef_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "cp_simd-inl.h"
#include "cp_dispatch.h"
#include "multithreading.h"
#include "word_coefficient.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {
namespace pc = word_coefficient;

#include "cp_prefix_suffix_coef.h"

template<class T, bool Backward>
void word_cpu_entry(const T* path, T* out, const T* coefs, const T* deriv, const uint64_t* indices,
    uint64_t num_words, const uint64_t* degrees, uint64_t batch, uint64_t dimension, uint64_t length,
    bool time_aug, bool lead_lag, T end_time, bool prefixes, int jobs, int method) {
    const uint64_t d = pc::add(pc::mul(dimension, lead_lag ? 2 : 1), time_aug ? 1 : 0);
    pc::Words words(indices, degrees, num_words, d, Backward || prefixes);
    pc::mul(pc::mul(batch, pc::mul(length, dimension)), sizeof(T));
    pc::mul(pc::mul(batch, words.width), sizeof(T));
    if (!jobs) throw std::invalid_argument("n_jobs cannot be zero");
    jobs = jobs > 0 ? jobs : int(get_max_threads()) + 1 + jobs;
    if (jobs < 1) throw std::invalid_argument("n_jobs is below the supported range");
    pc::check_method(method);
    if (!batch) return;
    if (method == pc::horner) {
        if constexpr (Backward) {
            sig_coef_backprop_<T>(path, out, coefs, deriv, indices, num_words, degrees,
                batch, dimension, length, time_aug, lead_lag, end_time, jobs);
        } else sig_coef_<T>(path, out, indices, num_words, degrees, batch, dimension, length,
            time_aug, lead_lag, end_time, prefixes, jobs);
        return;
    }
    const uint64_t budget = pc::workspace_limit();
    const uint64_t scratch_size = pc::add(pc::mul(6, words.max_degree), 1);
    const uint64_t workers = std::min<uint64_t>({batch, uint64_t(jobs), budget / pc::mul(scratch_size, sizeof(T))});
    if (!workers) throw std::runtime_error("Coefficient workspace limit is too small");
    if (length > 1 && prefix_suffix_cpu<T, Backward>(path, out, coefs, deriv, words, batch, dimension,
        length, prefixes, time_aug, lead_lag, end_time, jobs)) return;
    const uint64_t stride = length * dimension;
    auto work = [&](uint64_t begin, uint64_t end) HWY_ATTR {
        std::vector<T> scratch(scratch_size);
        for (uint64_t b = begin; b < end; ++b) {
            if constexpr (Backward) std::fill_n(out + b * stride, stride, T(0));
            for (uint64_t w = 0; w < num_words; ++w) {
                const uint64_t pos = b * words.width + words.packed[2 * num_words + w];
                pc::horner_word<Backward>(path + b * stride, out + (Backward ? b * stride : pos),
                    coefs ? coefs + pos : nullptr, Backward ? deriv + (coefs ? pos : b * num_words + w) : nullptr,
                    words.word(w), degrees[w], dimension, length, time_aug, lead_lag, end_time, prefixes, !coefs,
                    scratch.data(), [](T* dst, T v) { *dst += v; });
            }
        }
    };
    if (workers == 1) work(0, batch);
    else spawn_batch_threads(batch, int(workers), work);
}

template<std::floating_point T>
void sig_coef_dispatch_(const T* path, T* out, const uint64_t* indices, uint64_t words, const uint64_t* degrees,
    uint64_t batch, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time, bool prefixes, int n_jobs, int method) {
    word_cpu_entry<T, false>(path, out, nullptr, nullptr, indices, words, degrees, batch, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs, method);
}

template<std::floating_point T>
void sig_coef_backprop_dispatch_(const T* path, T* out, const T* coefs, const T* deriv, const uint64_t* indices,
    uint64_t words, const uint64_t* degrees, uint64_t batch, uint64_t dimension, uint64_t length,
    bool time_aug, bool lead_lag, T end_time, int n_jobs, int method) {
    word_cpu_entry<T, true>(path, out, coefs, deriv, indices, words, degrees, batch, dimension, length, time_aug, lead_lag, end_time, true, n_jobs, method);
}

#define INSTANTIATE_WORD_COEF(T) \
template void sig_coef_dispatch_<T>(const T*, T*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, int, int); \
template void sig_coef_backprop_dispatch_<T>(const T*, T*, const T*, const T*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, T, int, int)
INSTANTIATE_WORD_COEF(float);
INSTANTIATE_WORD_COEF(double);
#undef INSTANTIATE_WORD_COEF

} // namespace HWY_NAMESPACE
} // namespace cpsig
HWY_AFTER_NAMESPACE();
