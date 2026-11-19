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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cusig.h"
#include "cu_runtime_utils.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

TEST(PrefixSuffixCoefficientCuda, Identity) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    const uint64_t degrees[] = {0, 1}, word = 0;
    CudaBuf<float> path(3 * sizeof(float)), out(6 * sizeof(float));
    CudaBuf<uint64_t> indices(sizeof(word)), metadata(sizeof(degrees));
    ASSERT_EQ(cudaMemcpy(indices.get(), &word, sizeof(word), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(metadata.get(), degrees, sizeof(degrees), cudaMemcpyHostToDevice), cudaSuccess);
    for (int method : {0, 1}) for (uint64_t length : {0, 1}) {
        ASSERT_EQ(sig_coef_cuda_f(path.get(), out.get(), method ? &word : indices.get(), 2,
            method ? degrees : metadata.get(), 3, 1, length, true, false, false, 1.f, method), 0);
        float result[6];
        ASSERT_EQ(cudaMemcpy(result, out.get(), sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
        for (int b = 0; b < 3; ++b) {
            EXPECT_EQ(result[2 * b], 1);
            EXPECT_EQ(result[2 * b + 1], 0);
        }
    }
}

TEST(PrefixSuffixCoefficientCuda, AllPrefixAdjoints) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    const uint64_t word[] = {0, 0, 0, 0}, degree = 4, length = 130;
    std::vector<double> path(length), gradient(length);
    for (uint64_t i = 0; i < length; ++i) path[i] = double(i) / double(length - 1);
    const double seeds[] = {1, 2, 3, 4};
    CudaBuf<double> x(length * sizeof(double)), out(4 * sizeof(double));
    CudaBuf<double> g(length * sizeof(double)), d(4 * sizeof(double));
    ASSERT_EQ(cudaMemcpyAsync(x.get(), path.data(), length * sizeof(double), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d.get(), seeds, sizeof(seeds), cudaMemcpyHostToDevice, stream), cudaSuccess);
    CudaBuf<uint64_t> device_word(sizeof(word)), device_degree(sizeof(degree));
    ASSERT_EQ(cudaMemcpyAsync(device_word.get(), word, sizeof(word), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_degree.get(), &degree, sizeof(degree), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int method : {0, 1}) {
        const uint64_t* word_data = method ? word : device_word.get();
        const uint64_t* degree_data = method ? &degree : device_degree.get();
        ASSERT_EQ(sig_coef_cuda_d(x.get(), out.get(), word_data, 1, degree_data, 1, 1, length,
            true, false, false, 1., method), 0);
        ASSERT_EQ(sig_coef_backprop_cuda_d(x.get(), g.get(), out.get(), d.get(), word_data, 1, degree_data,
            1, 1, length, false, false, 1., method), 0);
        double coefficients[4];
        ASSERT_EQ(cudaMemcpy(coefficients, out.get(), sizeof(coefficients), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_NEAR(coefficients[3], 1. / 24., 1e-12);
        ASSERT_EQ(cudaMemcpy(gradient.data(), g.get(), length * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_NEAR(gradient.front(), -(1. + 2. + 1.5 + 4. / 6.), 1e-10);
        EXPECT_NEAR(gradient.back(), -gradient.front(), 1e-10);
        for (uint64_t i = 1; i + 1 < length; ++i) EXPECT_NEAR(gradient[i], 0., 1e-10);
    }
    ASSERT_EQ(sig_coef_cuda_d(x.get(), out.get(), word, 1, &degree, 1, 1, length, true), 0);
    ASSERT_EQ(sig_coef_backprop_cuda_d(x.get(), g.get(), out.get(), d.get(), word, 1, &degree, 1, 1, length), 0);
    std::vector<double> default_gradient(length);
    ASSERT_EQ(cudaMemcpy(default_gradient.data(), g.get(), length * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(default_gradient, gradient);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PrefixSuffixCoefficientCuda, InvalidMethod) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    const uint64_t word = 0, degree = 1;
    CudaBuf<float> path(2 * sizeof(float)), out(sizeof(float)), gradient(2 * sizeof(float));
    for (int method : {-1, 2, 3, 4, 5, 6}) {
        EXPECT_NE(sig_coef_cuda_f(path.get(), out.get(), &word, 1, &degree, 1, 1, 2,
            true, false, false, 1.f, method), 0);
        EXPECT_NE(sig_coef_backprop_cuda_f(path.get(), gradient.get(), out.get(), out.get(), &word, 1, &degree,
            1, 1, 2, false, false, 1.f, method), 0);
    }
}

TEST(PrefixSuffixCoefficientCuda, TerminalWordGradients) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    struct Shape { unsigned n, batch, words, degree; };
    std::mt19937 rng(103);
    std::uniform_real_distribution<double> random(-1., 1.);
    for (Shape s : {Shape{127, 9, 4, 5}, Shape{255, 65, 8, 8}, Shape{511, 2, 2, 10}, Shape{1023, 3, 8, 8}, Shape{8193, 1, 1, 16}, Shape{8193, 1, 8, 8}}) {
        const unsigned dim = 3, length = s.n + 1, total = dim * length * s.batch;
        std::vector<double> path(total), adjoints(s.batch * s.words), expanded(adjoints.size() * s.degree, 0.);
        std::vector<uint64_t> indices(s.words * s.degree), degrees(s.words, s.degree);
        for (unsigned i = 0; i < total; ++i)
            path[i] = random(rng) / std::sqrt(double(s.n)) + (i / dim % length ? path[i - dim] : 0.);
        for (auto& letter : indices) letter = rng() % dim;
        for (size_t i = 0; i < adjoints.size(); ++i)
            adjoints[i] = expanded[(i + 1) * s.degree - 1] = random(rng);
        CudaBuf<double> x(total * sizeof(double)), g(total * sizeof(double)), y(expanded.size() * sizeof(double));
        CudaBuf<double> a(adjoints.size() * sizeof(double)), full(expanded.size() * sizeof(double));
        CudaBuf<uint64_t> letters(indices.size() * sizeof(uint64_t)), lens(s.words * sizeof(uint64_t));
        ASSERT_EQ(cudaMemcpy(x.get(), path.data(), total * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(a.get(), adjoints.data(), adjoints.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(full.get(), expanded.data(), expanded.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(letters.get(), indices.data(), indices.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(lens.get(), degrees.data(), s.words * sizeof(uint64_t), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(sig_coef_cuda_d(x.get(), y.get(), indices.data(), s.words, degrees.data(), s.batch, dim, length, false, false, false, 1., 1), 0);
        ASSERT_EQ(sig_coef_backprop_cuda_d(x.get(), g.get(), nullptr, a.get(), indices.data(), s.words, degrees.data(), s.batch, dim, length, false, false, 1., 1), 0);
        std::vector<double> actual(total), expected(total);
        ASSERT_EQ(cudaMemcpy(actual.data(), g.get(), total * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(sig_coef_cuda_d(x.get(), y.get(), letters.get(), s.words, lens.get(), s.batch, dim, length, true, false, false, 1., 0), 0);
        ASSERT_EQ(sig_coef_backprop_cuda_d(x.get(), g.get(), y.get(), full.get(), letters.get(), s.words, lens.get(), s.batch, dim, length, false, false, 1., 0), 0);
        ASSERT_EQ(cudaMemcpy(expected.data(), g.get(), total * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
        double scale = 0., error = 0.;
        for (unsigned i = 0; i < total; ++i) {
            ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite(expected[i]));
            scale = std::max(scale, std::abs(expected[i]));
            error = std::max(error, std::abs(actual[i] - expected[i]));
        }
        EXPECT_LE(error, 1e-9 * scale);
    }
}
