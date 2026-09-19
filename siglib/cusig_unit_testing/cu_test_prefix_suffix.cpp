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
