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
#include "cp_simd_config.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cp_branched_polynomial_sig_kernel_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cpsig.h"
#include "log_sig_method.h"
#include "macros.h"
#include "multithreading.h"
#include "../shared/branched_polynomial.h"
#include "../shared/preparation/polynomial_sig_kernel/branched_polynomial_tables.h"
#include "cache_lifecycle/cp_poly_cache.h"
#include "cp_simd_math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {
#include "cp_kernel_api.h"

namespace {
template<typename T> struct CpuPolyOps {
    T* prefix_buffer;
    static uint64_t index() { return 0; }
    static uint64_t stride() { return 1; }
    static void sync() {}
    static T sum(T x) { return x; }
    static void exp(T* values, uint64_t count) { vec_exp(values, count); }
    static void exp_grid(const T* cells, T* out, uint64_t nx, uint64_t ny) {
        std::fill_n(out, ny + 1, T(1));
        for (uint64_t i = 0; i < nx; ++i) {
            T* row = out + (i + 1) * (ny + 1);
            row[0] = T(1);
            if constexpr (std::is_same_v<T, double>) {
                for (uint64_t j = 0; j < ny; ++j) row[j + 1] = std::exp(cells[i * ny + j]);
            } else {
                std::copy_n(cells + i * ny, ny, row + 1);
                vec_exp(row + 1, ny);
            }
        }
    }
    static void exp_grid_backprop(T* cells, const T* derivs, uint64_t nx, uint64_t ny) {
        if constexpr (std::is_same_v<T, double>) {
            for (uint64_t i = 0; i < nx; ++i)
                for (uint64_t j = 0; j < ny; ++j)
                    cells[i * ny + j] = std::exp(cells[i * ny + j]) * derivs[(i + 1) * (ny + 1) + j + 1];
            return;
        }
        vec_exp(cells, nx * ny);
        const Simd<T> d;
        constexpr uint64_t lanes = vec_batch_lanes<T>;
        for (uint64_t i = 0; i < nx; ++i) {
            T* row = cells + i * ny;
            const T* grad = derivs + (i + 1) * (ny + 1) + 1;
            uint64_t j = 0;
            for (; j + lanes <= ny; j += lanes)
                hn::StoreU(hn::Mul(hn::LoadU(d, row + j), hn::LoadU(d, grad + j)), d, row + j);
            for (; j < ny; ++j) row[j] *= grad[j];
        }
    }
    void prefix(T* values, uint64_t nx, uint64_t ny, bool reverse) {
        for (uint64_t j = 0; j < ny; ++j) {
            T val = 0;
            for (uint64_t k = 0; k < nx; ++k) {
                const auto i = reverse ? nx-1-k : k;
                val += values[i*ny+j]; values[i*ny+j] = val;
            }
        }
        for (uint64_t i = 0; i < nx; ++i) {
            T val = 0;
            for (uint64_t k = 0; k < ny; ++k) {
                const auto j = reverse ? ny-1-k : k;
                val += values[i*ny+j]; values[i*ny+j] = val;
            }
        }
    }
    void integrate_x(const T* src, T* dst, size_t nx, size_t width,
                     size_t q, const T* A, const T* w, bool transpose) {
        std::fill_n(prefix_buffer, width, T(0));
        for (size_t step = 0; step < nx; ++step) {
            const size_t i = transpose ? nx - 1 - step : step;
            for (size_t a = 0; a < q; ++a) {
                T* out = dst + (i * q + a) * width;
                const T factor = transpose ? w[a] : 1.0;
                for (size_t j = 0; j < width; ++j) out[j] = factor * prefix_buffer[j];
                for (size_t b = 0; b < q; ++b) {
                    const T coeff = transpose ? A[b * q + a] : A[a * q + b];
                    const T* in = src + (i * q + b) * width;
                    for (size_t j = 0; j < width; ++j) out[j] += coeff * in[j];
                }
            }
            for (size_t b = 0; b < q; ++b) {
                const T* in = src + (i * q + b) * width;
                const T factor = transpose ? 1.0 : w[b];
                for (size_t j = 0; j < width; ++j) prefix_buffer[j] += factor * in[j];
            }
        }
    }

    void integrate_y(const T* src, T* dst, size_t height, size_t ny,
                     size_t q, const T* A, const T* w, bool transpose) {
        const size_t width = ny * q;
        for (size_t i = 0; i < height; ++i) {
            T prefix = 0.0;
            for (size_t step = 0; step < ny; ++step) {
                const size_t j = transpose ? ny - 1 - step : step;
                const T* in = src + i * width + j * q;
                T* out = dst + i * width + j * q;
                for (size_t a = 0; a < q; ++a) {
                    T val = (transpose ? w[a] : 1.0) * prefix;
                    for (size_t b = 0; b < q; ++b)
                        val += (transpose ? A[b * q + a] : A[a * q + b]) * in[b];
                    out[a] = val;
                }
                for (size_t b = 0; b < q; ++b) prefix += (transpose ? 1.0 : w[b]) * in[b];
            }
        }
    }

};
}

template<std::floating_point T>
void branched_sig_kernel_poly_(const T* gram, T* out, T* state,
         uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2,
         uint64_t depth, uint64_t order, bool grid, int jobs) {
    const BranchedPolyDims d(l1, l2, depth, order, grid, batch, dim);
    if (jobs == 0) throw std::invalid_argument("n_jobs cannot be 0");
    if (batch == 0) return;
    if (!out) throw std::invalid_argument("branched polynomial kernel output pointer must not be null");
    if (d.cells && !gram) throw std::invalid_argument("branched polynomial kernel gram pointer must not be null");
    if (depth <= 1 || !d.cells) {
        branched_sig_kernel_<T>(gram, out, batch, dim, l1, l2, depth, 0, 0, grid, jobs);
        return;
    }
    const auto tables = get_branched_poly_tables<T>(order);
    spawn_batch_threads(batch, jobs, [&](uint64_t begin, uint64_t end) {
        thread_local std::vector<T> storage;
        storage.resize(d.scratch);
        BranchedPolyWorkspace<T> work(storage.data(), d);
        CpuPolyOps<T> ops{work.column_prefix};
        for (uint64_t b = begin; b < end; ++b)
            branched_poly_forward_pair(gram+b*d.cells, out+b*d.output,
                state ? state+b*d.state : nullptr, work, d, depth, tables->data.data(), grid, ops);
    });
}

template<std::floating_point T>
void branched_sig_kernel_poly_backprop_(const T* gram, T* grad, const T* derivs, const T* state,
              uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2,
              uint64_t depth, uint64_t order, bool grid, int jobs) {
    const BranchedPolyDims d(l1, l2, depth, order, grid, batch, dim);
    if (jobs == 0) throw std::invalid_argument("n_jobs cannot be 0");
    if (batch == 0 || !d.cells) return;
    if (!gram) throw std::invalid_argument("branched polynomial kernel gram pointer must not be null");
    if (!grad) throw std::invalid_argument("branched polynomial kernel gram derivative pointer must not be null");
    if (!derivs) throw std::invalid_argument("branched polynomial kernel output derivative pointer must not be null");
    if (depth <= 1) {
        batch_branched_sig_kernel_backprop_<T>(gram, grad, derivs, nullptr, batch, dim, l1, l2, depth, 0, 0, grid, jobs);
        return;
    }
    const auto tables = get_branched_poly_tables<T>(order);
    const auto elements = branched_poly_add(d.scratch, state ? 0 : d.state);
    branched_poly_mul(elements, sizeof(T));
    spawn_batch_threads(batch, jobs, [&](uint64_t begin, uint64_t end) {
        thread_local std::vector<T> storage;
        storage.resize(elements);
        BranchedPolyWorkspace<T> work(storage.data(), d);
        CpuPolyOps<T> ops{work.column_prefix};
        for (uint64_t b = begin; b < end; ++b)
            branched_poly_backprop_pair(gram+b*d.cells, grad+b*d.cells, derivs+b*d.output,
                state ? state+b*d.state : nullptr, work, d, depth, tables->data.data(), grid, ops);
    });
}

template void branched_sig_kernel_poly_<float>(const float* gram, float* out, float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs);

template void branched_sig_kernel_poly_backprop_<float>(const float* gram, float* grad, const float* derivs, const float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs);

template void branched_sig_kernel_poly_<double>(const double* gram, double* out, double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs);

template void branched_sig_kernel_poly_backprop_<double>(const double* gram, double* grad, const double* derivs, const double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs);

}
}
HWY_AFTER_NAMESPACE();
