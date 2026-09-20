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
#include <vector>
#include "cusig.h"

TEST(PrefixSuffixCuda, NonDefaultStream) {
    const double path[] = {0, 0, 0.2, -0.1, 0.1, 0.3, -0.2, 0.1};
    double *input = nullptr, *output = nullptr, *baseline = nullptr;
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&input, sizeof(path)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output, 126 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&baseline, 126 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(input, path, sizeof(path), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(signature_prefix_suffix_cuda_d(input, output, 1, 2, 4, 6, false, false, 1., false, stream), 0);
    ASSERT_EQ(signature_cuda_d(input, baseline, 1, 2, 4, 6, false, false, 1., true, false), 0);
    std::vector<double> actual(126), expected(126);
    ASSERT_EQ(cudaMemcpy(actual.data(), output, actual.size() * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(expected.data(), baseline, expected.size() * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-12);
    cudaFree(input); cudaFree(output); cudaFree(baseline); cudaStreamDestroy(stream);
}

TEST(PrefixSuffixCuda, BackpropNonDefaultStream) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    for (uint64_t degree : {2, 3, 5, 6}) for (bool time_aug : {false, true}) for (bool lead_lag : {false, true}) {
        const uint64_t batch = 3, dimension = 2, length = 10;
        const uint64_t d = dimension * (lead_lag ? 2 : 1) + time_aug;
        uint64_t size = 1, power = 1;
        for (uint64_t k = 1; k <= degree; ++k) { power *= d; size += power; }
        std::vector<double> path(batch * length * dimension), deriv(batch * size), actual(path.size()), expected(path.size());
        for (uint64_t x = 0; x < path.size(); ++x) path[x] = double(int((x * 7) % 19) - 9) / 32;
        for (uint64_t x = 0; x < deriv.size(); ++x) deriv[x] = double(int(x % 13) - 6) / 16;
        double *input, *sig, *cotangent, *output, *baseline;
        ASSERT_EQ(cudaMalloc(&input, path.size() * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&sig, deriv.size() * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&cotangent, deriv.size() * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&output, path.size() * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&baseline, path.size() * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(input, path.data(), path.size() * sizeof(double), cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(cotangent, deriv.data(), deriv.size() * sizeof(double), cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(signature_prefix_suffix_backprop_cuda_d(input, output, cotangent, batch, dimension, length,
            degree, time_aug, lead_lag, 0.7, true, stream), 0) << cusig_last_error_message();
        ASSERT_EQ(signature_cuda_d(input, sig, batch, dimension, length, degree, time_aug, lead_lag, 0.7, true, true), 0);
        ASSERT_EQ(sig_backprop_cuda_d(input, baseline, cotangent, sig, batch, dimension, length, degree,
            time_aug, lead_lag, 0.7, true), 0);
        ASSERT_EQ(cudaMemcpy(actual.data(), output, actual.size() * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(expected.data(), baseline, expected.size() * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
        for (uint64_t x = 0; x < actual.size(); ++x)
            EXPECT_NEAR(actual[x], expected[x], 1e-10) << degree << "," << time_aug << "," << lead_lag;
        cudaFree(input); cudaFree(sig); cudaFree(cotangent); cudaFree(output); cudaFree(baseline);
    }
    cudaStreamDestroy(stream);
}
