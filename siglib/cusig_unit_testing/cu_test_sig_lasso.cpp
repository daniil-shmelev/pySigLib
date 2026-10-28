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

#include "cusig.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <limits>

TEST(sigLassoCudaTest, FitOnCallerStream) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    const std::array<double, 8> paths{0, -2, 0, -1, 0, 1, 0, 2};
    const std::array<double, 4> response{-2, -1, 1, 2};
    const double alpha = .1;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    double *device_paths = nullptr, *device_y = nullptr, *device_result = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_paths), sizeof(paths)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_y), sizeof(response)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_result), 8 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_paths, paths.data(), sizeof(paths), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_y, response.data(), sizeof(response), cudaMemcpyHostToDevice, stream), cudaSuccess);
    SigLassoOptions options{};
    options.samples = 4; options.length = 2; options.dimension = options.degree = options.targets = 1;
    options.max_iter = 100; options.batch_size = 4; options.tolerance = 1e-10; options.n_jobs = 1; options.intercept = 1;
    for (uint32_t full : {0U, 1U}) {
        options.full = full;
        uint64_t letters[1], offsets[2], converged[1], iterations[1];
        SigLassoStats stats{};
        struct Output { SigLassoShape shape; SigLassoOutput buffers; size_t calls = 0; } output{
            {}, {letters, offsets, converged, iterations, device_result, device_result + 1, device_result + 2, device_result + 3, device_result + 4, &stats}};
        const auto allocate = [](void* context, const SigLassoShape* shape, SigLassoOutput* buffers) {
            auto& output = *static_cast<Output*>(context);
            output.shape = *shape; ++output.calls;
            if (shape->words != 1 || shape->letters != 1 || shape->targets != 1) return 1;
            *buffers = output.buffers;
            return 0;
        };
        const auto fail = [](void*, const SigLassoShape*, SigLassoOutput*) { return 1; };
        EXPECT_NE(sig_lasso_cuda_d(device_paths, device_y, &alpha, nullptr, nullptr, 0, &options, stream, 0, fail, nullptr), 0);
        ASSERT_EQ(sig_lasso_cuda_d(device_paths, device_y, &alpha, nullptr, nullptr, 0, &options, stream, 0, allocate, &output), 0);
        std::array<double, 8> values{};
        ASSERT_EQ(cudaMemcpyAsync(values.data(), device_result, sizeof(values), cudaMemcpyDeviceToHost, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_NEAR(values[0], 1.4, 1e-10); EXPECT_NEAR(values[1], 0, 1e-12); EXPECT_LE(values[3], 1e-10);
        EXPECT_TRUE(converged[0]); EXPECT_EQ(iterations[0], 1); EXPECT_EQ(output.shape.stats.activations, 1); EXPECT_EQ(output.calls, 1);
        EXPECT_EQ(letters[0], 0); EXPECT_EQ(offsets[0], 0); EXPECT_EQ(offsets[1], 1);
        EXPECT_EQ(values[4], -1); EXPECT_EQ(values[7], 1); EXPECT_GT(stats.output_bytes, 0);
    }
    cudaFree(device_result); cudaFree(device_y); cudaFree(device_paths); cudaStreamDestroy(stream);
}

TEST(sigLassoCudaTest, RejectsNonfiniteVariation) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    double *device_paths = nullptr, *device_y = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_paths), 2 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_y), sizeof(double)), cudaSuccess);
    const double response = 0, alpha = .1;
    ASSERT_EQ(cudaMemcpy(device_y, &response, sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    SigLassoOptions options{};
    options.samples = options.dimension = options.degree = options.targets = 1;
    options.max_iter = 10; options.batch_size = 2; options.tolerance = 1e-8; options.n_jobs = 1;
    std::array<double, 2> paths{-1e308, 1e308};
    for (size_t length : {2U, 1U}) {
        if (length == 1) paths[0] = std::numeric_limits<double>::quiet_NaN();
        ASSERT_EQ(cudaMemcpy(device_paths, paths.data(), length * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
        options.length = length;
        bool called = false;
        const auto allocate = [](void* context, const SigLassoShape*, SigLassoOutput*) {
            *static_cast<bool*>(context) = true;
            return 1;
        };
        EXPECT_NE(sig_lasso_cuda_d(device_paths, device_y, &alpha, nullptr, nullptr, 0, &options, nullptr, 0, allocate, &called), 0);
        EXPECT_FALSE(called);
    }
    cudaFree(device_y); cudaFree(device_paths);
}
