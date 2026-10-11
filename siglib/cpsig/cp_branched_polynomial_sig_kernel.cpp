// Copyright 2026. Licensed under the Apache License, Version 2.0.
#include "cppch.h"
#include "cpsig.h"
#include "macros.h"
#include "multithreading.h"
#include "../shared/branched_polynomial.h"
#include "../shared/preparation/polynomial_sig_kernel/branched_polynomial_tables.h"

namespace {
template<typename T> struct CpuPolyOps {
    T* prefix_buffer;
    static uint64_t index() { return 0; }
    static uint64_t stride() { return 1; }
    static void sync() {}
    static T sum(T x) { return x; }
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
template<typename T> struct TableCache {
    std::mutex mutex;
    std::vector<std::shared_ptr<const BranchedPolynomialTables<T>>> entries;
};
template<typename T> TableCache<T>& table_cache() { static TableCache<T> c; return c; }
template<typename T> auto get_tables(uint64_t order) {
    auto& c = table_cache<T>();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (c.entries.empty()) c.entries.resize(65);
    auto& entry = c.entries[order];
    if (!entry) entry = std::make_shared<const BranchedPolynomialTables<T>>(order);
    return entry;
}
template<typename T>
void forward(const T* gram, T* out, T* state,
         uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2,
         uint64_t depth, uint64_t order, bool grid, int jobs) {
    const BranchedPolyDims d(l1, l2, depth, order, grid, batch, dim);
    if (jobs == 0) throw std::invalid_argument("n_jobs cannot be 0");
    if (batch == 0) return;
    if (!out) throw std::invalid_argument("branched polynomial kernel output pointer must not be null");
    if (d.cells && !gram) throw std::invalid_argument("branched polynomial kernel gram pointer must not be null");
    if (depth <= 1 || !d.cells) {
        int error;
        if constexpr (std::is_same_v<T, float>)
            error = branched_sig_kernel_f(gram, out, batch, dim, l1, l2, depth, 0, 0, grid, jobs);
        else
            error = branched_sig_kernel_d(gram, out, batch, dim, l1, l2, depth, 0, 0, grid, jobs);
        if (error) throw std::runtime_error(get_pysiglib_last_error());
        return;
    }
    const auto tables = get_tables<T>(order);
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

template<typename T>
void backprop(const T* gram, T* grad, const T* derivs, const T* state,
              uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2,
              uint64_t depth, uint64_t order, bool grid, int jobs) {
    const BranchedPolyDims d(l1, l2, depth, order, grid, batch, dim);
    if (jobs == 0) throw std::invalid_argument("n_jobs cannot be 0");
    if (batch == 0 || !d.cells) return;
    if (!gram) throw std::invalid_argument("branched polynomial kernel gram pointer must not be null");
    if (!grad) throw std::invalid_argument("branched polynomial kernel gram derivative pointer must not be null");
    if (!derivs) throw std::invalid_argument("branched polynomial kernel output derivative pointer must not be null");
    if (depth <= 1) {
        int error;
        if constexpr (std::is_same_v<T, float>)
            error = branched_sig_kernel_backprop_f(gram, grad, derivs, nullptr, batch, dim, l1, l2, depth, 0, 0, grid, jobs);
        else
            error = branched_sig_kernel_backprop_d(gram, grad, derivs, nullptr, batch, dim, l1, l2, depth, 0, 0, grid, jobs);
        if (error) throw std::runtime_error(get_pysiglib_last_error());
        return;
    }
    const auto tables = get_tables<T>(order);
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
}
void clear_branched_poly_table_cache() {
    { auto& c = table_cache<float>(); std::lock_guard<std::mutex> lock(c.mutex); decltype(c.entries)().swap(c.entries); }
    { auto& c = table_cache<double>(); std::lock_guard<std::mutex> lock(c.mutex); decltype(c.entries)().swap(c.entries); }
}
extern "C" CPSIG_API int branched_sig_kernel_poly_f(const float* gram, float* out, float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
    SAFE_CALL(forward<float>(gram, out, state, batch, dim, l1, l2, depth, order, grid, jobs));
}
extern "C" CPSIG_API int branched_sig_kernel_poly_backprop_f(const float* gram, float* grad, const float* derivs, const float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
    SAFE_CALL(backprop<float>(gram, grad, derivs, state, batch, dim, l1, l2, depth, order, grid, jobs));
}
extern "C" CPSIG_API int branched_sig_kernel_poly_d(const double* gram, double* out, double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
    SAFE_CALL(forward<double>(gram, out, state, batch, dim, l1, l2, depth, order, grid, jobs));
}
extern "C" CPSIG_API int branched_sig_kernel_poly_backprop_d(const double* gram, double* grad, const double* derivs, const double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
    SAFE_CALL(backprop<double>(gram, grad, derivs, state, batch, dim, l1, l2, depth, order, grid, jobs));
}
