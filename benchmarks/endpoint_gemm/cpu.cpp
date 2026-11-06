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

#include "common.h"
#include "cp_signature.h"
#include <cblas.h>

namespace endpoint {
using namespace cpsig::HWY_NAMESPACE;

template<typename T>
void compute(const T* path, T* out, const Shape& s, int jobs) {
    const uint64_t tasks = mul(s.batch, s.chunks);
    if (!tasks) return;
    const uint64_t per_worker = add(add(s.lower, s.width), mul(s.block, add(s.width, s.d)));
    const uint64_t stored = mul(2, mul(tasks, s.total));
    const uint64_t budget = (512ULL << 20) / sizeof(T);
    if (stored > budget || per_worker > budget - stored) throw std::bad_alloc();
    const int requested = jobs < 1 ? int(get_max_threads()) : jobs;
    const int threads = int(std::min<uint64_t>(std::max(1, requested), (budget - stored) / per_worker));
    std::vector<T> first(tasks * s.total), second(tasks * s.total);
    spawn_batch_threads(tasks, threads, [&](uint64_t begin, uint64_t end) {
        std::vector<T> scratch(per_worker);
        T* state = scratch.data(); T* work = state + s.lower;
        T* w = work + s.width; T* v = w + s.block * s.width;
        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t sample = task / s.chunks, start = task % s.chunks * s.block;
            const uint64_t count = start < s.n ? std::min(s.block, s.n - start) : 0;
            std::fill(state, state + s.lower, T(0)); state[0] = 1;
            for (uint64_t i = 0; i < count; ++i) {
                const T* point = path + (sample * (s.n + 1) + start + i) * s.d;
                T* inc = v + i * s.d;
                for (uint64_t c = 0; c < s.d; ++c) inc[c] = point[s.d + c] - point[c];
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
            if (count) {
                if constexpr (std::is_same_v<T, float>)
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, int(s.width), int(s.d), int(count),
                        1, w, int(s.width), v, int(s.d), 0, result + s.lower, int(s.d));
                else
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, int(s.width), int(s.d), int(count),
                        1, w, int(s.width), v, int(s.d), 0, result + s.lower, int(s.d));
            }
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
                for (uint64_t k = 1; k <= s.m; ++k) {
                    for (uint64_t p = 0; p < s.power[k]; ++p) result[s.offset[k] + p] += right[s.offset[k] + p];
                    for (uint64_t j = 1; j < k; ++j)
                        for (uint64_t p = 0; p < s.power[j]; ++p)
                            vec_mult_add(result + s.offset[k] + p * s.power[k - j], right + s.offset[k - j],
                                left[s.offset[j] + p], s.power[k - j]);
                }
            }
        });
        first.swap(second); count = next;
    }
    std::copy_n(first.data(), s.batch * s.total, out);
}

template<typename T>
int call(const T* path, T* out, uint64_t batch, uint64_t d, uint64_t points, uint64_t m, uint64_t block, int jobs) noexcept {
    try { compute(path, out, Shape(batch, d, points, m, block), jobs); return 0; }
    catch (const std::exception& e) { error = e.what(); return 1; }
}
}

ENDPOINT_API const char* endpoint_error() noexcept { return endpoint::error.c_str(); }
ENDPOINT_API int endpoint_f(const float* p, float* o, uint64_t b, uint64_t d, uint64_t n, uint64_t m, uint64_t block, int jobs) noexcept {
    return endpoint::call(p, o, b, d, n, m, block, jobs);
}
ENDPOINT_API int endpoint_d(const double* p, double* o, uint64_t b, uint64_t d, uint64_t n, uint64_t m, uint64_t block, int jobs) noexcept {
    return endpoint::call(p, o, b, d, n, m, block, jobs);
}
