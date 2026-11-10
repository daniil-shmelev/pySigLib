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

// A separate benchmark library. No production dispatch changes.
#include "../../siglib/cpsig/cp_prefix_suffix_impl.cpp"
#include <atomic>

namespace cpsig::HWY_NAMESPACE {
#include "cpu_baseline.h"
struct ReversePlan {
    uint64_t block, qlen, temporary, rows, maxp, maxq, count;
    ReversePlan(const Layout& l, uint64_t n, uint64_t requested, int jobs, uint64_t itemsize) {
        block = std::min(n, requested);
        qlen = l.offset[l.scan_degree + 1];
        temporary = l.offset[std::max(l.scan_degree, l.left_degree(l.m)) + 1];
        maxp = l.power[l.left_degree(l.m)]; maxq = l.power[l.right_degree(l.m)];
        rows = mul(block, l.left_degree(l.m));
        if (mul(mul(rows, add(maxp, maxq)), itemsize) > option("PYSIGLIB_PS_FACTOR_BYTES", 64ULL << 20))
            throw std::runtime_error("requested block exceeds factor workspace limit");
        count = add(mul(add(mul(3, block), 3), qlen), mul(add(mul(2, block), 1), l.d));
        count = add(count, add(mul(rows, add(maxp, maxq)), mul(mul(4, temporary), jobs)));
    }
};

template<typename T>
void reverse_path(const T* path, const T* final, const T* deriv, T* out, uint64_t n,
    bool scalar, int jobs, const Layout& l, const ReversePlan& plan, T* memory) {
    const uint64_t h = l.scan_degree, qlen = plan.qlen, block = plan.block;
    T* prefixes = memory; T* suffixes = prefixes + block * qlen;
    T* sources = suffixes + block * qlen; T* inc = sources + block * qlen;
    T* dinc = inc + block * l.d; T* negative = dinc + block * l.d;
    T* prefix = negative + l.d; T* suffix = prefix + qlen; T* adjoint = suffix + qlen;
    T* left = adjoint + qlen; T* right = left + plan.rows * plan.maxp;
    T* temporary = right + plan.rows * plan.maxq;
    std::fill_n(out, (n + 1) * l.d, T(0));
    CpuMatrices<T> matrices{jobs};
    for (bool reverse : {true, false}) {
        std::fill_n(prefix, 3 * qlen, T(0)); prefix[0] = suffix[0] = 1;
        std::copy_n(final + scalar, qlen - 1, (reverse ? prefix : suffix) + 1);
        if (reverse) std::copy_n(deriv + scalar, qlen - 1, adjoint + 1);
        for (uint64_t position = 0; position < n; position += block) {
            const uint64_t size = std::min(block, n - position);
            const uint64_t first = reverse ? n - position - size : position;
            std::fill_n(sources, size * qlen, T(0));
            std::fill_n(dinc, size * l.d, T(0));
            for (uint64_t step = 0; step < size; ++step) {
                const uint64_t slot = reverse ? size - 1 - step : step, i = first + slot;
                T* v = inc + slot * l.d;
                for (uint64_t c = 0; c < l.d; ++c) negative[c] = -(v[c] = path[(i + 1) * l.d + c] - path[i * l.d + c]);
                if (reverse) signature_horner_step_(prefix, negative, l.d, h, l.offset.data(), temporary);
                else cpu_detail::prepend_generic(suffix, negative, l, temporary);
                std::copy_n(prefix, qlen, prefixes + slot * qlen);
                std::copy_n(suffix, qlen, suffixes + slot * qlen);
                if (reverse) cpu_detail::prepend_generic(suffix, v, l, temporary);
                else signature_horner_step_(prefix, v, l.d, h, l.offset.data(), temporary);
            }
            for (uint64_t k = h + 1; k <= l.m; ++k) {
                const uint64_t a = l.left_degree(k), b = l.right_degree(k), p = l.power[a], q = l.power[b], rows = size * a;
                const uint64_t owners = std::min<uint64_t>(size, jobs);
                auto ranges = [&](auto process) {
                    parallel_range(owners, jobs, [&](uint64_t begin, uint64_t end) {
                        for (uint64_t owner = begin; owner < end; ++owner)
                            for (uint64_t i = size * owner / owners; i < size * (owner + 1) / owners; ++i)
                                process(i, temporary + owner * 4 * plan.temporary);
                    });
                };
                ranges([&](uint64_t i, T* powers) {
                    const T* v = inc + i * l.d;
                    if (!reverse) cpu_detail::tensor_powers(v, powers, a, l);
                    for (uint64_t j = 0; j < a; ++j) {
                        const uint64_t row = i * a + j, tail = l.power[j + 1];
                        if (reverse) right_factor(right + row * q, suffixes + i * qlen, v, j, b, l);
                        else for (uint64_t x = 0; x < l.power[a - 1 - j]; ++x)
                            for (uint64_t y = 0; y < tail; ++y)
                                left[row * p + x * tail + y] = prefixes[i * qlen + l.offset[a - 1 - j] + x] * powers[l.offset[j + 1] + y];
                    }
                });
                T* g = const_cast<T*>(deriv + l.offset[k] - !scalar);
                if (reverse) matrices.gemm(Matrix<T>{right, rows, q, q, 1}, Matrix<T>{g, q, p, 1, q}, Matrix<T>{left, rows, p, p, 1}, true);
                else matrices.gemm(Matrix<T>{left, rows, p, p, 1}, Matrix<T>{g, p, q, q, 1}, Matrix<T>{right, rows, q, q, 1}, true);
                ranges([&](uint64_t i, T* powers) {
                    const T* v = inc + i * l.d; T* dv = dinc + i * l.d;
                    T* ds = sources + i * qlen; T* dpowers = powers + plan.temporary;
                    if (reverse) {
                        cpu_detail::tensor_powers(v, powers, a, l);
                        std::fill_n(dpowers, l.offset[a + 1], T(0));
                    }
                    for (uint64_t j = 0; j < a; ++j) {
                        const uint64_t row = i * a + j, tail = l.power[j + 1], level = l.offset[a - 1 - j];
                        if (reverse) {
                            for (uint64_t x = 0; x < l.power[a - 1 - j]; ++x) {
                                if (tail >= 16) ds[level + x] += dot_product_mult_add(powers + l.offset[j + 1], left + row * p + x * tail,
                                    dpowers + l.offset[j + 1], prefixes[i * qlen + level + x], tail);
                                else for (uint64_t y = 0; y < tail; ++y) {
                                    const T value = left[row * p + x * tail + y];
                                    ds[level + x] += value * powers[l.offset[j + 1] + y];
                                    dpowers[l.offset[j + 1] + y] += value * prefixes[i * qlen + level + x];
                                }
                            }
                        } else cpu_detail::right_factor_backprop(suffixes + i * qlen, v, right + row * q, ds, dv, j, b, l, powers, dpowers);
                    }
                    if (reverse) cpu_detail::tensor_powers_backprop(v, powers, dpowers, dv, a, l);
                });
            }
            for (uint64_t step = 0; step < size; ++step) {
                const uint64_t slot = reverse ? size - 1 - step : step, i = first + slot;
                T* dv = dinc + slot * l.d;
                if (reverse) cpu_detail::horner_backprop<false>(prefixes + slot * qlen, inc + slot * l.d, adjoint, dv, h, l, temporary, temporary + plan.temporary);
                else cpu_detail::horner_backprop<true>(suffixes + slot * qlen, inc + slot * l.d, adjoint, dv, h, l, temporary, temporary + plan.temporary);
                for (uint64_t x = 1; x < qlen; ++x) adjoint[x] += sources[slot * qlen + x];
                for (uint64_t c = 0; c < l.d; ++c) {
                    out[i * l.d + c] -= dv[c]; out[(i + 1) * l.d + c] += dv[c];
                }
            }
        }
    }
}

template<typename T>
void measure(const T* path, const T* final, const T* deriv, T* out, uint64_t batch, uint64_t d,
    uint64_t n, uint64_t m, uint64_t split, uint64_t block, bool scalar, int jobs, uint64_t* peak) {
    if (!batch || !n || m < 2 || !jobs) throw std::invalid_argument("invalid experiment shape");
    const Layout l(d, m, split);
    const uint64_t stride = l.offset[m + 1] - !scalar, path_stride = (n + 1) * d;
    const int path_jobs = n * l.power[m] >= parallel_threshold ? std::max(1, jobs / int(std::min<uint64_t>(batch, jobs))) : 1;
    const CpuBackpropPlan baseline(l, n, path_jobs, sizeof(T));
    const ReversePlan reversal(l, n, block ? block : 1, path_jobs, sizeof(T));
    const uint64_t count = block ? reversal.count : baseline.count, bytes = mul(count, sizeof(T));
    const uint64_t budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", 2ULL << 30);
    if (bytes > budget) throw std::runtime_error("experiment exceeds workspace limit");
    const uint64_t owners = std::min({batch, uint64_t(jobs), budget / bytes});
    std::atomic<uint64_t> active{0}, maximum{0};
    get_tbb_arena(jobs).execute([&] { parallel_range(owners, jobs, [&](uint64_t first, uint64_t end) {
        std::vector<T> storage(count);
        const uint64_t current = active.fetch_add(bytes) + bytes;
        uint64_t old = maximum.load();
        while (old < current && !maximum.compare_exchange_weak(old, current)) {}
        for (uint64_t i = batch * first / owners; i < batch * end / owners; ++i) {
            if (block) reverse_path(path + i * path_stride, final + i * stride, deriv + i * stride,
                out + i * path_stride, n, scalar, path_jobs, l, reversal, storage.data());
            else backprop_path(path + i * path_stride, out + i * path_stride, deriv + i * stride,
                d, n + 1, false, false, T(1), scalar, path_jobs, l, baseline, storage.data());
        }
        active.fetch_sub(bytes);
    }); });
    *peak = maximum.load();
}
}

static thread_local std::string last_error;
extern "C" __declspec(dllexport) const char* experiment_error() noexcept { return last_error.c_str(); }
#define EXPORT(T, S) \
extern "C" __declspec(dllexport) int experiment_##S(const T* path, const T* final, const T* deriv, T* out, \
    uint64_t batch, uint64_t d, uint64_t n, uint64_t m, uint64_t split, uint64_t block, bool scalar, int jobs, uint64_t* peak) noexcept { \
    try { cpsig::HWY_NAMESPACE::measure(path, final, deriv, out, batch, d, n, m, split, block, scalar, jobs, peak); return 0; } \
    catch (const std::exception& e) { last_error = e.what(); return 1; } \
}
EXPORT(float, f)
EXPORT(double, d)
#undef EXPORT
