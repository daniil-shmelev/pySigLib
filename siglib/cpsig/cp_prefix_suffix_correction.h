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

namespace corrected {
using namespace prefix_suffix;
#include "prefix_suffix_correction-inl.h"

template<bool Backward, typename T>
void compute(const T* input, T* output, const T* deriv, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t degree, bool time_aug, T end_time, bool scalar, int n_jobs,
    uint64_t split, const T* correction, uint64_t correction_len, uint64_t batch_stride, uint64_t segment_stride) {
    const int jobs = n_jobs > 0 ? n_jobs : static_cast<int>(get_max_threads()) + 1 + n_jobs;
    if (!n_jobs || jobs < 1) throw std::invalid_argument("invalid n_jobs");
    if (!batch) return;
    if (length > 1 && openblas_get_parallel() != 0)
        throw std::runtime_error("prefix-suffix requires sequential OpenBLAS built with USE_THREAD=0 and USE_LOCKING=1");
    const Plan plan(add(dimension, time_aug ? 1 : 0), dimension, degree, length, split, correction_len, jobs, sizeof(T), Backward);
    const auto& layout = plan.layout;
    const Algebra l = plan.algebra();
    const uint64_t path_stride = mul(length, dimension), sig_stride = layout.offset[degree + 1] - (scalar ? 0 : 1);
    mul(mul(batch, path_stride), sizeof(T)); mul(mul(batch, sig_stride), sizeof(T));
    const uint64_t budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", 2ULL << 30), bytes = mul(plan.count, sizeof(T));
    if (bytes > budget) throw std::runtime_error("CPU corrected prefix-suffix requires " + std::to_string(bytes)
        + " workspace bytes for one path; configured limit is " + std::to_string(budget));
    const uint64_t owners = std::min({batch, uint64_t(jobs), budget / bytes});
    const int path_jobs = std::max(1, jobs / int(owners));
    get_tbb_arena(jobs).execute([&] { parallel_range(owners, jobs, [&](uint64_t first, uint64_t end) {
        std::vector<T> storage(plan.count);
        T* memory = storage.data();
        T* increments = memory + plan.increments; T* prefixes = memory + plan.prefixes;
        T* suffixes = memory + plan.suffixes; T* dprefix = memory + plan.dprefixes;
        T* dsuffix = memory + plan.dsuffixes; T* dv = memory + plan.gradient;
        T* left = memory + plan.left; T* right = memory + plan.right;
        T* dleft = memory + plan.dleft; T* dright = memory + plan.dright;
        T* work = memory + plan.work;
        CpuMatrices<T> matrices{path_jobs};
        for (uint64_t owner = first; owner < end; ++owner)
            for (uint64_t sample = batch * owner / owners; sample < batch * (owner + 1) / owners; ++sample) {
                const T* c = correction + sample * batch_stride;
                Path<T> path(input + sample * path_stride, dimension, length, time_aug, false, end_time);
                for (uint64_t i = 0; i < plan.n; ++i) {
                    auto before = path[i], after = path[i + 1];
                    for (uint64_t channel = 0; channel < l.d; ++channel)
                        increments[i * l.d + channel] = after[channel] - before[channel];
                }
                scans(l, plan.n, increments, c, segment_stride, prefixes, suffixes, work);
                T* out = output + sample * (Backward ? path_stride : sig_stride);
                const T* g = Backward ? deriv + sample * sig_stride : nullptr;
                if constexpr (Backward) {
                    zero(dprefix, plan.scan_count); zero(dsuffix, plan.scan_count); zero(dv, plan.n * l.d);
                    for (uint64_t j = scalar ? 0 : 1; j < l.size; ++j)
                        dprefix[plan.n * l.size + j] = g[j - (scalar ? 0 : 1)];
                } else {
                    zero(out, sig_stride);
                    copy(prefixes + plan.n * l.size + (scalar ? 0 : 1), out, l.size - (scalar ? 0 : 1));
                }
                auto fill = [&]<bool Reverse>(uint64_t a, uint64_t b, uint64_t base, uint64_t count, uint64_t rows) {
                    const uint64_t p = l.power[a], q = l.power[b];
                    const uint64_t workers = std::min<uint64_t>(count, path_jobs);
                    parallel_range(workers, path_jobs, [&](uint64_t begin, uint64_t finish) {
                        for (uint64_t worker = begin; worker < finish; ++worker)
                            for (uint64_t i = worker; i < count; i += workers) {
                                const uint64_t segment = base + i;
                                factors<Reverse>(l, a, b, increments + segment * l.d, c + segment * segment_stride,
                                    prefixes + segment * l.size, suffixes + (segment + 1) * l.size,
                                    plan.nodes_count, plan.nodes.data(), plan.weights.data(),
                                    left + i * rows * p, right + i * rows * q,
                                    Backward ? dleft + i * rows * p : work, Backward ? dright + i * rows * q : work,
                                    Backward ? dprefix + segment * l.size : work, Backward ? dsuffix + (segment + 1) * l.size : work,
                                    Backward ? dv + segment * l.d : work, work + worker * plan.scratch_size);
                            }
                    });
                };
                contract<Backward>(plan, matrices, memory, 0, out, g, scalar, fill);
                if constexpr (Backward) {
                    reverse_scans(l, plan.n, increments, c, segment_stride, prefixes, suffixes, dprefix, dsuffix, dv, work);
                    for (uint64_t i = 0; i < length; ++i)
                        for (uint64_t channel = 0; channel < dimension; ++channel)
                            out[i * dimension + channel] = (i ? dv[(i - 1) * l.d + channel] : T(0))
                                - (i < plan.n ? dv[i * l.d + channel] : T(0));
                }
            }
    }); });
}
} // namespace corrected
