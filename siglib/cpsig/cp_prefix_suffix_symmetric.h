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

// Included in the active Highway namespace.
enum class FactorProduct { forward, left_adjoint, right_adjoint };

template<FactorProduct product, typename T>
void symmetric_gemm(const SymmetricPlan& plan, const CpuMatrices<T>& matrices, T* scratch,
    T* left, T* right, T* level, uint64_t a, uint64_t p, uint64_t q, uint64_t pitch,
    uint64_t base, uint64_t rows, bool first) {
    constexpr bool forward = product == FactorProduct::forward;
    constexpr bool left_adjoint = product == FactorProduct::left_adjoint;
    for (uint64_t j = 0; j < a; ++j) {
        const uint64_t start = (j + a - base % a) % a;
        if (start >= rows) continue;
        const uint64_t n = (rows - 1 - start) / a + 1;
        T* lrow = left + start * p;
        T* rrow = right + start * pitch;
        if (!j) {
            if constexpr (forward)
                matrices.gemm({lrow, p, n, 1, a * p}, {rrow, n, q, a * pitch, 1}, {level, p, q, q, 1}, first);
            else if constexpr (left_adjoint)
                matrices.gemm({rrow, n, q, a * pitch, 1}, {level, q, p, 1, q}, {lrow, n, p, a * p, 1}, true);
            else
                matrices.gemm({lrow, n, p, a * p, 1}, {level, p, q, q, 1}, {rrow, n, q, a * pitch, 1}, true);
            continue;
        }
        const auto& map = plan.powers[j - 1];
        const uint64_t unique = map.representative.size(), heads = p / map.size;
        const uint64_t tile = std::max<uint64_t>(1, plan.scratch / ((n + q) * unique));
        for (uint64_t head = 0; head < heads; head += tile) {
            const uint64_t count = std::min(heads - head, tile), width = count * unique;
            T* packed = scratch;
            T* reduced = packed + n * width;
            if constexpr (!left_adjoint) {
                parallel_range(n, matrices.jobs, [&](uint64_t begin, uint64_t end) {
                    for (uint64_t i = begin; i < end; ++i) {
                        const T* in = lrow + i * a * p + head * map.size;
                        T* out = packed + i * width;
                        for (uint64_t h = 0; h < count; ++h)
                            for (unsigned repr : map.representative) *out++ = in[h * map.size + repr];
                    }
                });
            }
            if constexpr (!forward) {
                parallel_range(width, matrices.jobs, [&](uint64_t begin, uint64_t end) {
                    for (uint64_t x = begin; x < end; ++x) {
                        const uint64_t g = x % unique, h = head + x / unique;
                        const T* in = level + h * map.size * q;
                        T* out = reduced + x * q;
                        std::copy_n(in + map.representative[g] * q, q, out);
                        for (unsigned t = map.offsets[g] + 1; t < map.offsets[g + 1]; ++t)
                            vec_mult_add(out, in + map.members[t] * q, T(1), q);
                    }
                });
            }
            if constexpr (forward) {
                matrices.gemm({packed, width, n, 1, width}, {rrow, n, q, a * pitch, 1},
                    {reduced, width, q, q, 1}, true);
                parallel_range(count * map.size, matrices.jobs, [&](uint64_t begin, uint64_t end) {
                    for (uint64_t x = begin; x < end; ++x) {
                        const uint64_t g = (x / map.size) * unique + map.group[x % map.size];
                        vec_mult_add(level + (head * map.size + x) * q, reduced + g * q, T(1), q);
                    }
                });
            } else if constexpr (left_adjoint) {
                matrices.gemm({rrow, n, q, a * pitch, 1}, {reduced, q, width, 1, q}, {packed, n, width, width, 1}, true);
                parallel_range(n, matrices.jobs, [&](uint64_t begin, uint64_t end) {
                    for (uint64_t i = begin; i < end; ++i) {
                        T* out = lrow + i * a * p + head * map.size;
                        std::fill_n(out, count * map.size, T(0));
                        const T* in = packed + i * width;
                        for (uint64_t h = 0; h < count; ++h)
                            for (unsigned repr : map.representative) out[h * map.size + repr] = *in++;
                    }
                });
            } else {
                matrices.gemm({packed, n, width, width, 1}, {reduced, width, q, q, 1},
                    {rrow, n, q, a * pitch, 1}, head == 0);
            }
        }
    }
}
