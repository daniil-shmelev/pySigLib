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

#pragma once
#include "sig_lasso_core.h"
#include <gtest/gtest.h>
#include <random>
#ifdef LASSO_TEST_CUDA
#include "cu_sig_lasso_active.cuh"
#define LASSO_SUITE LassoActiveCUDA
#else
#define LASSO_SUITE LassoActiveCPU
#endif

namespace {
class MatrixBackend : public siglasso::Backend {
public:
#ifdef LASSO_TEST_CUDA
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
#endif
    explicit MatrixBackend(size_t n) : Backend([&] { SigLassoOptions o{}; o.samples = n; o.degree = 1; o.n_jobs = 1; return o; }()) {
#ifdef LASSO_TEST_CUDA
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess || cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUDA test setup failed");
        cublasSetStream(blas, stream);
#endif
    }
    ~MatrixBackend() {
#ifdef LASSO_TEST_CUDA
        cublasDestroy(blas); cudaStreamDestroy(stream);
#endif
    }
    siglasso::Array alloc(size_t n) override {
        const size_t bytes = n * sizeof(double); budget->acquire(bytes);
        double* values = nullptr;
#ifdef LASSO_TEST_CUDA
        if (cudaMalloc(reinterpret_cast<void**>(&values), bytes) != cudaSuccess) { budget->release(bytes); throw std::bad_alloc(); }
#else
        values = new double[n];
#endif
        return {values, n, std::shared_ptr<void>(values, [account = budget, bytes](void* ptr) {
#ifdef LASSO_TEST_CUDA
            cudaFree(ptr);
#else
            delete[] static_cast<double*>(ptr);
#endif
            account->release(bytes);
        })};
    }
    void upload(const double* a, double* b, size_t n) override {
#ifdef LASSO_TEST_CUDA
        ASSERT_EQ(cudaMemcpyAsync(b, a, n * sizeof(double), cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
#else
        std::copy_n(a, n, b);
#endif
    }
    void download(const double* a, double* b, size_t n) override {
#ifdef LASSO_TEST_CUDA
        ASSERT_EQ(cudaMemcpyAsync(b, a, n * sizeof(double), cudaMemcpyDeviceToHost, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
#else
        std::copy_n(a, n, b);
#endif
    }
    void copy(const double* a, double* b, size_t n) override {
#ifdef LASSO_TEST_CUDA
        ASSERT_EQ(cudaMemcpyAsync(b, a, n * sizeof(double), cudaMemcpyDeviceToDevice, stream), cudaSuccess);
#else
        std::copy_n(a, n, b);
#endif
    }
    void fill(const siglasso::Array& a, double value) override { const std::vector<double> data(a.size, value); upload(data.data(), a.data, a.size); }
    double dot(const double* a, const double* b, size_t n) override {
#ifdef LASSO_TEST_CUDA
        double value = 0; if (cublasDdot(blas, static_cast<int>(n), a, 1, b, 1, &value) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("test dot failed"); return value;
#else
        return std::inner_product(a, a + n, b, 0.0);
#endif
    }
    void scale(double value, double* a, size_t n) override {
#ifdef LASSO_TEST_CUDA
        ASSERT_EQ(cublasDscal(blas, static_cast<int>(n), &value, a, 1), CUBLAS_STATUS_SUCCESS);
#else
        for (size_t j = 0; j < n; ++j) a[j] *= value;
#endif
    }
    void axpy(double value, const double* a, double* b, size_t n) override {
#ifdef LASSO_TEST_CUDA
        ASSERT_EQ(cublasDaxpy(blas, static_cast<int>(n), &value, a, 1, b, 1), CUBLAS_STATUS_SUCCESS);
#else
        for (size_t j = 0; j < n; ++j) b[j] += value * a[j];
#endif
    }
    double absolute_sum(const siglasso::Array& a) override {
#ifdef LASSO_TEST_CUDA
        double value = 0; if (cublasDasum(blas, static_cast<int>(a.size), a.data, 1, &value) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("test sum failed"); return value;
#else
        double value = 0; for (size_t j = 0; j < a.size; ++j) value += std::abs(a.data[j]); return value;
#endif
    }
    double center(double*, size_t, bool) override { throw std::logic_error("unused test operation"); }
    void correlations(const std::vector<siglasso::Column>& columns, const siglasso::Array& y, std::vector<double>& out, const std::vector<size_t>* selected) override {
        out.resize(selected ? selected->size() : columns.size());
        for (size_t j = 0; j < out.size(); ++j) out[j] = dot(columns[selected ? (*selected)[j] : j].values.data, y.data, y.size) / y.size;
    }
    std::unique_ptr<siglasso::ActiveSet> active_set(const siglasso::Array& y, double alpha) override {
#ifdef LASSO_TEST_CUDA
        return siglasso::make_cuda_active(*this, y, alpha, stream, blas, 0);
#else
        return siglasso::make_cpu_active(*this, y, alpha);
#endif
    }
    void prepare() override { throw std::logic_error("unused test operation"); }
    siglasso::Array factors(size_t) override { throw std::logic_error("unused test operation"); }
    siglasso::Array energy_gram(size_t, size_t) override { throw std::logic_error("unused test operation"); }
    void moments(siglasso::Node&, const siglasso::Array&) override { throw std::logic_error("unused test operation"); }
    void bounds(const std::vector<siglasso::Probe>&, const siglasso::Array&, double, std::vector<double>&) override { throw std::logic_error("unused test operation"); }
    void children(const std::vector<siglasso::Probe>&, std::vector<siglasso::NodePtr>&) override { throw std::logic_error("unused test operation"); }
    void full_columns(std::vector<siglasso::Column>&) override { throw std::logic_error("unused test operation"); }
    siglasso::Array response(size_t) override { throw std::logic_error("unused test operation"); }
};

TEST(LASSO_SUITE, WideCompatibleAndUncappedNullspace) {
#ifdef LASSO_TEST_CUDA
    int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) GTEST_SKIP();
#endif
    for (double duplicate_scale : {1., 2., -2.}) {
        MatrixBackend b(2);
        auto y = b.from_host({10., 10.}); auto active = b.active_set(y, .01);
        std::vector<siglasso::Column> columns;
        for (const std::vector<double>& value : {std::vector<double>{1, 0}, {0, 1}, {duplicate_scale, 0}}) {
            auto x = b.from_host(value); columns.push_back({{columns.size()}, x, 0, b.dot(x.data, x.data, 2)});
        }
        active->activate(columns[0], 0, 1); active->activate(columns[1], 1, 1);
        ASSERT_EQ(active->correct(), siglasso::Correction::stationary);
        active->activate(columns[2], 2, duplicate_scale);
        auto status = active->correct();
        ASSERT_NE(status, siglasso::Correction::failed);
        EXPECT_GT(b.stats.singular_steps, 0);
        if (duplicate_scale != 1) {
            EXPECT_EQ(status, siglasso::Correction::boundary); EXPECT_EQ(active->size(), 2);
            std::vector<size_t> slots; std::vector<double> beta; active->output(slots, beta);
            EXPECT_EQ(std::count(slots.begin(), slots.end(), 0), 0);
        }
        for (size_t step = 0; status == siglasso::Correction::boundary && step < 10; ++step) status = active->correct();
        ASSERT_EQ(status, siglasso::Correction::stationary);
        const auto residual = b.host(active->residual());
        EXPECT_NEAR(residual[0], .02 / std::abs(duplicate_scale), 1e-11); EXPECT_NEAR(residual[1], .02, 1e-11);
    }
}

TEST(LASSO_SUITE, QRUpdatesMatchFreshFactorisation) {
#ifdef LASSO_TEST_CUDA
    int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) GTEST_SKIP();
#endif
    MatrixBackend b(32); std::mt19937 generator(871); std::normal_distribution<double> normal;
    std::vector<double> target(32); for (auto& v : target) v = normal(generator);
    auto y = b.from_host(target); auto active = b.active_set(y, .008);
    std::vector<siglasso::Column> columns; std::vector<size_t> pool;
    for (size_t j = 0; j < 55; ++j) {
        std::vector<double> value(32); for (auto& v : value) v = normal(generator);
        auto x = b.from_host(value); columns.push_back({{j}, x, 0, b.dot(x.data, x.data, 32)}); pool.push_back(j);
    }
    size_t step = 0; std::unordered_set<size_t> visited; bool reentered = false;
    for (; step < 500; ++step) {
        const auto choice = active->select(columns, pool);
        if (choice.slot == SIZE_MAX) break;
        reentered = !visited.insert(choice.slot).second || reentered;
        active->activate(columns[choice.slot], choice.slot, choice.correlation);
        siglasso::Correction status;
        do { status = active->correct(); ASSERT_NE(status, siglasso::Correction::failed); } while (status == siglasso::Correction::boundary);
        auto fresh = b.active_set(y, .008); std::vector<size_t> slots; std::vector<double> beta; active->output(slots, beta);
        for (size_t j = 0; j < slots.size(); ++j) fresh->activate(columns[slots[j]], slots[j], beta[j]);
        ASSERT_EQ(fresh->correct(true), siglasso::Correction::stationary);
        auto actual = b.host(active->residual()), expected = b.host(fresh->residual());
        for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-9);
    }
    EXPECT_LT(step, 500); EXPECT_GT(b.stats.qr_deletions, 0); EXPECT_TRUE(reentered);
}

TEST(LASSO_SUITE, SimultaneousBoundaryDrops) {
#ifdef LASSO_TEST_CUDA
    int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) GTEST_SKIP();
#endif
    MatrixBackend b(2); auto y = b.from_host({10., 10.}); auto active = b.active_set(y, .01);
    std::vector<siglasso::Column> columns;
    for (const std::vector<double>& values : {std::vector<double>{1, 0}, {0, 1}, {2, 2}}) {
        auto x = b.from_host(values); columns.push_back({{columns.size()}, x, 0, b.dot(x.data, x.data, 2)});
    }
    active->activate(columns[0], 0, 1); active->activate(columns[1], 1, 1);
    ASSERT_EQ(active->correct(), siglasso::Correction::stationary);
    active->activate(columns[2], 2, 1);
    ASSERT_EQ(active->correct(), siglasso::Correction::boundary);
    EXPECT_EQ(active->size(), 1); EXPECT_EQ(b.stats.drops, 2); EXPECT_EQ(b.stats.qr_deletions, 2);
    ASSERT_EQ(active->correct(), siglasso::Correction::stationary);
    const auto residual = b.host(active->residual());
    EXPECT_NEAR(residual[0], .005, 1e-11); EXPECT_NEAR(residual[1], .005, 1e-11);
}
} // namespace
