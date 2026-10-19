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
#include <vector>

TEST(PascalCoefficientCuda, HornerIdentity) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    const uint64_t degrees[] = {0, 1}, word = 0;
    CudaBuf<float> path(3 * sizeof(float)), out(6 * sizeof(float));
    CudaBuf<uint64_t> indices(sizeof(word)), metadata(sizeof(degrees));
    ASSERT_EQ(cudaMemcpy(indices.get(), &word, sizeof(word), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(metadata.get(), degrees, sizeof(degrees), cudaMemcpyHostToDevice), cudaSuccess);
    for (uint64_t length : {0, 1}) {
        ASSERT_EQ(sig_coef_cuda_f(path.get(), out.get(), indices.get(), 2, metadata.get(), 3, 1, length, true), 0);
        float result[6];
        ASSERT_EQ(cudaMemcpy(result, out.get(), sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
        for (int b = 0; b < 3; ++b) {
            EXPECT_EQ(result[2 * b], 1);
            EXPECT_EQ(result[2 * b + 1], 0);
        }
    }
}

TEST(PascalCoefficientCuda, AllPrefixAdjointsAndStream) {
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
    for (int method : {0, 1, 2}) {
        const uint64_t* word_data = method == 0 ? device_word.get() : word;
        const uint64_t* degree_data = method == 0 ? device_degree.get() : &degree;
        ASSERT_EQ(sig_coef_cuda_d(x.get(), out.get(), word_data, 1, degree_data, 1, 1, length,
            true, false, false, 1., method, stream), 0);
        ASSERT_EQ(sig_coef_backprop_cuda_d(x.get(), g.get(), out.get(), d.get(), word_data, 1, degree_data,
            1, 1, length, false, false, 1., method, stream), 0);
        double coefficients[4];
        ASSERT_EQ(cudaMemcpy(coefficients, out.get(), sizeof(coefficients), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_NEAR(coefficients[3], 1. / 24., 1e-12);
        ASSERT_EQ(cudaMemcpy(gradient.data(), g.get(), length * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_NEAR(gradient.front(), -(1. + 2. + 1.5 + 4. / 6.), 1e-10);
        EXPECT_NEAR(gradient.back(), -gradient.front(), 1e-10);
        for (uint64_t i = 1; i + 1 < length; ++i) EXPECT_NEAR(gradient[i], 0., 1e-10);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PascalCoefficientCuda, InvalidMethod) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    const uint64_t word = 0, degree = 1;
    CudaBuf<float> path(2 * sizeof(float)), out(sizeof(float)), gradient(2 * sizeof(float));
    for (int method : {-1, 3, 4, 5, 6}) {
        EXPECT_NE(sig_coef_cuda_f(path.get(), out.get(), &word, 1, &degree, 1, 1, 2,
            true, false, false, 1.f, method, nullptr), 0);
        EXPECT_NE(sig_coef_backprop_cuda_f(path.get(), gradient.get(), out.get(), out.get(), &word, 1, &degree,
            1, 1, 2, false, false, 1.f, method, nullptr), 0);
    }
}
