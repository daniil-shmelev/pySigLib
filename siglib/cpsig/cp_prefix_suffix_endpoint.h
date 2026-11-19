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

#include "cp_simd_config.h"
#include "hwy/highway.h"
#if defined(CPSIG_ENDPOINT_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_ENDPOINT_H_
#undef CPSIG_ENDPOINT_H_
#else
#define CPSIG_ENDPOINT_H_
#endif
#include "prefix_suffix_endpoint.h"
#include "cp_prefix_suffix_endpoint_plan.h"
#include "cp_signature.h"
#include "cp_prefix_suffix_gemm.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {
namespace cpu_endpoint {
using namespace prefix_suffix;
using prefix_suffix_endpoint::Shape;

template<typename T>
void compute(const T* path, T* out, const Shape& s, int jobs, uint64_t dimension, uint64_t length, bool ta, bool ll, T end_time, bool scalar) {
    const uint64_t tasks = mul(s.batch, s.chunks);
    if (!tasks) return;
    const uint64_t budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", prefix_suffix_endpoint::cpu_budget) / sizeof(T);
    const prefix_suffix_endpoint::CpuWorkspace workspace(tasks, s.total, s.lower, s.width, s.d, s.block, jobs, budget);
    if (!workspace.workers) throw std::bad_alloc();
    const int threads = static_cast<int>(workspace.workers);
    std::vector<T> first(tasks * s.total), second(tasks * s.total);
    spawn_batch_threads(tasks, threads, [&](uint64_t begin, uint64_t end) {
        std::vector<T> scratch(workspace.per_worker);
        T* state = scratch.data(); T* work = state + s.lower;
        T* w = work + s.width; T* v = w + s.block * s.width;
        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t sample = task / s.chunks, start = task % s.chunks * s.block;
            const uint64_t count = start < s.n ? std::min(s.block, s.n - start) : 0;
            std::fill(state, state + s.lower, T(0)); state[0] = 1;
            Path<T> input(path + sample * length * dimension, dimension, length, ta, ll, end_time);
            for (uint64_t i = 0; i < count; ++i) {
                const auto point = input[start + i], next = input[start + i + 1];
                T* inc = v + i * s.d;
                for (uint64_t c = 0; c < s.d; ++c) inc[c] = next[c] - point[c];
                for (uint64_t c = 0; c < s.d; ++c) work[c] = inc[c] / T(s.m);
                for (uint64_t k = 1; k + 1 < s.m; ++k) {
                    const T reciprocal = T(1) / T(s.m - k);
                    for (uint64_t p = s.power[k]; p-- > 0;)
                        vec_mult_assign(work + p * s.d, inc, (work[p] + state[s.offset[k] + p]) * reciprocal, s.d);
                }
                for (uint64_t p = 0; p < s.width; ++p) w[i * s.width + p] = work[p] + state[s.offset[s.m - 1] + p];
                signature_horner_step_(state, inc, s.d, s.m - 1, s.offset.data(), work);
            }
            T* result = first.data() + task * s.total;
            std::copy_n(state, s.lower, result);
            if (count)
                factor_gemm(prefix_suffix::Matrix<T>{w, s.width, count, 1, s.width},
                    prefix_suffix::Matrix<T>{v, count, s.d, s.d, 1},
                    prefix_suffix::Matrix<T>{result + s.lower, s.width, s.d, s.d, 1}, true);
        }
    });
    uint64_t count = s.chunks;
    while (count > 1) {
        const uint64_t next = (count + 1) / 2;
        spawn_batch_threads(s.batch * next, threads, [&](uint64_t begin, uint64_t end) {
            for (uint64_t task = begin; task < end; ++task) {
                const uint64_t sample = task / next, pair = task % next;
                const T* left = first.data() + (sample * count + 2 * pair) * s.total;
                T* result = second.data() + task * s.total;
                std::copy_n(left, s.total, result);
                if (2 * pair + 1 == count) continue;
                const T* right = left + s.total;
                sig_combine_inplace_(result, right, s.m, s.offset.data());
            }
        });
        first.swap(second); count = next;
    }
    const uint64_t skip = scalar ? 0 : 1;
    for (uint64_t sample = 0; sample < s.batch; ++sample)
        std::copy_n(first.data() + sample * s.total + skip, s.total - skip, out + sample * (s.total - skip));
}

} // namespace cpu_endpoint
} // namespace HWY_NAMESPACE
} // namespace cpsig
HWY_AFTER_NAMESPACE();
#endif
