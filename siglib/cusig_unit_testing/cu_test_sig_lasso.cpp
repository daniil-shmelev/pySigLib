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

TEST(sigLassoCudaTest, VersionedFitOnCallerStream) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
    EXPECT_EQ(sig_lasso_abi_version_cuda(), 2);
    const std::array<double, 8> paths{0, -2, 0, -1, 0, 1, 0, 2};
    const std::array<double, 4> response{-2, -1, 1, 2};
    const double weights[]{1, 1}, alpha = .1;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    double *device_paths = nullptr, *device_y = nullptr, *device_result = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_paths), sizeof(paths)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_y), sizeof(response)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_result), 4 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_paths, paths.data(), sizeof(paths), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_y, response.data(), sizeof(response), cudaMemcpyHostToDevice, stream), cudaSuccess);
    SigLassoOptions options{};
    options.samples = 4; options.length = 2; options.dimension = options.degree = options.targets = 1;
    options.max_iter = 100; options.batch_size = 4; options.tolerance = 1e-10; options.n_jobs = 1; options.intercept = 1;
    for (uint32_t full : {0U, 1U}) {
        options.full = full; uint64_t handle = 0;
        ASSERT_EQ(sig_lasso_v2_cuda_d(device_paths, device_y, weights, &alpha, nullptr, nullptr, 0, &options, stream, 0, &handle), 0);
        SigLassoShape shape{}; ASSERT_EQ(sig_lasso_shape_v2_cuda(handle, &shape), 0); ASSERT_EQ(shape.words, 1);
        uint64_t letters[1], offsets[2], converged[1], iterations[1];
        ASSERT_EQ(sig_lasso_result_v2_cuda_d(handle, letters, offsets, device_result, device_result + 1, device_result + 2,
            device_result + 3, converged, iterations), 0);
        std::array<double, 4> values{};
        ASSERT_EQ(cudaMemcpyAsync(values.data(), device_result, sizeof(values), cudaMemcpyDeviceToHost, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_NEAR(values[0], .96, 1e-10); EXPECT_NEAR(values[1], 0, 1e-12); EXPECT_LE(values[3], 1e-10);
        EXPECT_TRUE(converged[0]); EXPECT_EQ(iterations[0], 1); EXPECT_EQ(shape.stats.activations, 1);
        EXPECT_EQ(sig_lasso_release_v2_cuda(handle), 0);
        EXPECT_NE(sig_lasso_shape_v2_cuda(handle, &shape), 0);
    }
    cudaFree(device_result); cudaFree(device_y); cudaFree(device_paths); cudaStreamDestroy(stream);
}
