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

struct CpuBackpropPlan {
    uint64_t n, qlen, scan_count, maxp, maxq, rows, temporary, count;
    CpuBackpropPlan(const Layout& l, uint64_t segments, int jobs, uint64_t itemsize) : n(segments) {
        qlen = l.offset[l.scan_degree + 1];
        scan_count = mul(add(n, 1), qlen);
        maxp = l.power[l.left_degree(l.m)]; maxq = l.power[l.right_degree(l.m)];
        const uint64_t factor_bytes = option("PYSIGLIB_PS_FACTOR_BYTES", 64ULL << 20);
        rows = l.scan_degree < l.m ? std::max<uint64_t>(1, std::min(mul(n, l.left_degree(l.m)),
            factor_bytes / mul(mul(2, add(maxp, maxq)), itemsize))) : 0;

        temporary = l.offset[std::max(l.scan_degree, l.left_degree(l.m)) + 1];
        count = add(mul(4, scan_count), mul(2, mul(n, l.d)));
        count = add(count, mul(2, mul(rows, add(maxp, maxq))));
        count = add(count, add(mul(2, qlen), mul(mul(4, temporary), jobs)));
    }
};

template<typename T>
void backprop_path(const T* input, T* output, const T* deriv, uint64_t dimension, uint64_t length,
    bool time_aug, bool lead_lag, T end_time, bool scalar, int jobs, const Layout& l,
    const CpuBackpropPlan& plan, T* storage) {
    const uint64_t n = plan.n, qlen = plan.qlen;
    T* prefixes = storage;
    T* suffixes = prefixes + plan.scan_count;
    T* dprefix = suffixes + plan.scan_count;
    T* dsuffix = dprefix + plan.scan_count;
    T* increments = dsuffix + plan.scan_count;
    T* dincrements = increments + n * l.d;
    T* left = dincrements + n * l.d;
    T* right = left + plan.rows * plan.maxp;
    T* dleft = right + plan.rows * plan.maxq;
    T* dright = dleft + plan.rows * plan.maxp;
    T* state = dright + plan.rows * plan.maxq;
    T* adjoint = state + qlen;
    T* temporary = adjoint + qlen;
    std::fill_n(dprefix, 2 * plan.scan_count, T(0));
    std::fill_n(dincrements, n * l.d, T(0));
    Path<T> path(input, dimension, length, time_aug, lead_lag, end_time);
    prefix_scan(path, n, l.d, l, qlen, increments, prefixes, state, temporary);
    suffix_scan(n, l, qlen, increments, suffixes, state, temporary);
    CpuMatrices<T> matrices{jobs};
    for (uint64_t k = l.scan_degree + 1; k <= l.m; ++k) {
        const uint64_t a = l.left_degree(k), b = l.right_degree(k), p = l.power[a], q = l.power[b], total = n * a;
        for (uint64_t base = 0; base < total; base += plan.rows) {
            const uint64_t count = std::min(plan.rows, total - base);
            const uint64_t first_segment = base / a, segments = (base + count - 1) / a - first_segment + 1;
            const uint64_t owners = std::min<uint64_t>(segments, jobs);
            auto factor_ranges = [&](auto process) {
                parallel_range(owners, jobs, [&](uint64_t first_owner, uint64_t end_owner) {
                    for (uint64_t owner = first_owner; owner < end_owner; ++owner) {
                        T* powers = temporary + owner * 4 * plan.temporary;
                        for (uint64_t i = first_segment + segments * owner / owners;
                            i < first_segment + segments * (owner + 1) / owners; ++i) {
                            cpu_detail::tensor_powers(increments + i * l.d, powers, a, l);
                            process(i, std::max(base, i * a), std::min(base + count, (i + 1) * a), powers);
                        }
                    }
                });
            };
            factor_ranges([&](uint64_t i, uint64_t begin, uint64_t end, T* powers) {
                for (uint64_t t = begin; t < end; ++t) {
                    const uint64_t j = t % a, row = t - base, tail = l.power[j + 1];
                    const T* prefix = prefixes + i * qlen + l.offset[a - 1 - j];
                    for (uint64_t x = 0; x < l.power[a - 1 - j]; ++x)
                        for (uint64_t y = 0; y < tail; ++y)
                            left[row * p + x * tail + y] = prefix[x] * powers[l.offset[j + 1] + y];
                    right_factor(right + row * q, suffixes + (i + 1) * qlen, increments + i * l.d, j, b, l);
                }
            });
            T* g = const_cast<T*>(deriv + l.offset[k] - (scalar ? 0 : 1));
            matrices.gemm(Matrix<T>{right, count, q, q, 1}, Matrix<T>{g, q, p, 1, q},
                Matrix<T>{dleft, count, p, p, 1}, true);
            matrices.gemm(Matrix<T>{left, count, p, p, 1}, Matrix<T>{g, p, q, q, 1},
                Matrix<T>{dright, count, q, q, 1}, true);
            factor_ranges([&](uint64_t i, uint64_t begin, uint64_t end, T* powers) {
                T* dpowers = powers + plan.temporary;
                T* history = dpowers + plan.temporary;
                T* work = history + plan.temporary;
                std::fill_n(dpowers, l.offset[a + 1], T(0));
                for (uint64_t t = begin; t < end; ++t) {
                    const uint64_t j = t % a, row = t - base, tail = l.power[j + 1];
                    const uint64_t level = l.offset[a - 1 - j];
                    const T* prefix = prefixes + i * qlen + level;
                    T* dp = dprefix + i * qlen + level;
                    for (uint64_t x = 0; x < l.power[a - 1 - j]; ++x) {
                        if (tail >= 16) {
                            dp[x] += dot_product_mult_add(powers + l.offset[j + 1], dleft + row * p + x * tail,
                                dpowers + l.offset[j + 1], prefix[x], tail);
                        } else {
                            T value = 0;
                            for (uint64_t y = 0; y < tail; ++y) {
                                const T g = dleft[row * p + x * tail + y];
                                value += g * powers[l.offset[j + 1] + y];
                                dpowers[l.offset[j + 1] + y] += g * prefix[x];
                            }
                            dp[x] += value;
                        }
                    }
                    cpu_detail::right_factor_backprop(suffixes + (i + 1) * qlen, increments + i * l.d,
                        dright + row * q, dsuffix + (i + 1) * qlen, dincrements + i * l.d,
                        j, b, l, history, work);
                }
                cpu_detail::tensor_powers_backprop(increments + i * l.d, powers, dpowers,
                    dincrements + i * l.d, a, l);
            });
        }
    }
    adjoint[0] = 0;
    for (uint64_t x = 1; x < qlen; ++x) adjoint[x] = deriv[x - (scalar ? 0 : 1)];
    for (uint64_t i = n; i-- > 0;) {
        cpu_detail::horner_backprop<false>(prefixes + i * qlen, increments + i * l.d,
            adjoint, dincrements + i * l.d, l.scan_degree, l, temporary, temporary + plan.temporary);
        for (uint64_t x = 1; x < qlen; ++x) adjoint[x] += dprefix[i * qlen + x];
    }
    std::fill_n(adjoint, qlen, T(0));
    for (uint64_t i = 0; i < n; ++i) {
        cpu_detail::horner_backprop<true>(suffixes + (i + 1) * qlen, increments + i * l.d,
            adjoint, dincrements + i * l.d, l.scan_degree, l, temporary, temporary + plan.temporary);
        for (uint64_t x = 1; x < qlen; ++x) adjoint[x] += dsuffix[(i + 1) * qlen + x];
    }
    std::fill_n(output, length * dimension, T(0));
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t c = 0; c < dimension * (lead_lag ? 2 : 1); ++c) {
            const uint64_t first = lead_lag ? i / 2 + (i % 2 && c >= dimension) : i;
            const uint64_t next = lead_lag ? (i + 1) / 2 + ((i + 1) % 2 && c >= dimension) : i + 1;
            const T value = dincrements[i * l.d + c];
            output[first * dimension + c % dimension] -= value;
            output[next * dimension + c % dimension] += value;
        }
}
